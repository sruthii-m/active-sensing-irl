"""Feedforward AIRL using external imitation, launched from an installed wheel."""
import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

import imitation

# Fail before any training if repository-local imitation shadows the library.
if not (Path(imitation.__file__).parent / "algorithms" / "adversarial" / "airl.py").exists():
    raise ImportError("External imitation required. Install this package non-editably in .venv-airl and run python -I -m airl_runner.train")

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import RewardNet, ShapedRewardNet
from imitation.util.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from active_sensing_bench.data import digest, load_split, transition_arrays, write_json
from active_sensing_bench.env import DecisionEnv
from active_sensing_bench.evaluate import PPOPolicy, likelihood
from active_sensing_bench.networks import mlp
from active_sensing_bench.retrain_policy import make_ppo
from active_sensing_bench.reward_exports import FrozenReward, code_version


class BaseReward(RewardNet):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space, normalize_images=False)
        self.network = mlp(78, 1)

    def forward(self, state, action, next_state, done):
        return self.network(torch.cat((state, action), dim=-1)).squeeze(-1)


class Potential(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = mlp(71, 1)

    def forward(self, state):
        return self.network(state).squeeze(-1)


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if min(args.rounds, args.envs, args.steps_per_env, args.demo_batch_size, args.disc_updates) < 1:
        raise ValueError("Positive AIRL budgets required")
    training, manifest = load_split(args.data, "train")
    validation, _ = load_split(args.data, "validation")
    condition = manifest["condition"]
    if condition["collection_limit"] is not None:
        raise ValueError("AIRL requires complete intrinsic-task episodes in this runner")
    arrays = transition_arrays(training)
    # A prebatched iterable samples transitions with replacement. This also
    # supports small smoke datasets without dropping a partial library batch.
    rng = np.random.default_rng(args.seed)
    def demonstrations():
        while True:
            indices = rng.integers(len(arrays["x"]), size=args.demo_batch_size)
            yield dict(obs=arrays["x"][indices], acts=arrays["action"][indices],
                       next_obs=arrays["next_x"][indices], dones=arrays["terminal"][indices])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    print("External imitation:", imitation.__file__, flush=True)
    config = {**vars(args), "condition": condition, "method": "airl", "weighting": "transition",
              "manifest_sha256": digest(Path(args.data) / "manifest.json"), "code_version": code_version(),
              "allow_variable_horizon": True,
              "variable_horizon_reason": "goal-entry feedback and the intrinsic 400-step deadline are part of the task",
              "length_independent_reward_claim": False, "fixed_horizon_diagnostic": "required before a length-independent claim",
              "checkpoint_rule": "minimum validation episode-mean NLL; paired reward/policy snapshot",
              "reward_family": "action-dependent neural base reward", "model_access": "online environment",
              "demonstration_count": len(training), "training_transition_count": len(arrays["x"]),
              "expert_id": manifest["expert_id"], "imitation_file": str(imitation.__file__),
              "versions": {p: importlib.metadata.version(p) for p in ("imitation", "stable-baselines3", "gymnasium", "torch")}}
    write_json(out / "config.json", config)
    write_json(out / "split_manifest.json", manifest)
    vec = DummyVecEnv([lambda: Monitor(DecisionEnv(condition["representation"], condition["view_size"])) for _ in range(args.envs)])
    vec.seed(args.seed)
    generator = make_ppo(vec, args.seed, args.steps_per_env, args.ppo_batch_size, args.ppo_epochs, args.gamma, args.lr)
    base = BaseReward(vec.observation_space, vec.action_space)
    reward_net = ShapedRewardNet(base, Potential(), discount_factor=args.gamma)
    trainer = AIRL(demonstrations=demonstrations(), demo_batch_size=args.demo_batch_size,
                   venv=vec, gen_algo=generator, reward_net=reward_net,
                   n_disc_updates_per_round=args.disc_updates, disc_opt_kwargs={"lr": args.lr},
                   allow_variable_horizon=True, gen_replay_buffer_capacity=args.envs * args.steps_per_env,
                   custom_logger=configure(str(out / "logs"), ["csv"]))
    best, history, selected_round = float("inf"), [], 0
    try:
        for round_index in range(args.rounds + 1):
            if round_index:
                trainer.train(total_timesteps=args.envs * args.steps_per_env)
            metrics = likelihood(PPOPolicy(generator), validation)
            score = metrics["episode_mean_nll"]
            history.append(dict(round=round_index, **{k: v for k, v in metrics.items() if k != "episodes"}))
            if score < best:
                best, selected_round = score, round_index
                generator.save(out / "policy.zip")
                if trainer.reward_test is not base:
                    raise RuntimeError("AIRL reward_test is not the unshaped base reward")
                frozen = FrozenReward(trainer.reward_test.network, method="airl", representation=condition["representation"],
                                      view_size=condition["view_size"], gamma=args.gamma)
                check = slice(0, min(32, len(arrays["x"])))
                expected = trainer.reward_test.predict(arrays["x"][check], arrays["action"][check],
                                                       arrays["next_x"][check], arrays["terminal"][check])
                np.testing.assert_allclose(frozen(arrays["x"][check], arrays["action"][check]), expected, atol=1e-6)
                frozen.save(out / "reward.pt")
                torch.save(reward_net.state_dict(), out / "shaped_reward_training_only.pt")
            print(f"AIRL round={round_index} validation_episode_nll={score:.6f}", flush=True)
        report = dict(selected_round=selected_round, best_validation_nll=best, history=history,
                      online_interactions=generator.num_timesteps,
                      train_episode_ids=[e["episode_id"] for e in training],
                      expert_episode_lengths=[e["length"] for e in training])
        write_json(out / "selection.json", report)
        return report
    finally:
        vec.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=49)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps-per-env", type=int, default=512)
    parser.add_argument("--demo-batch-size", type=int, default=512)
    parser.add_argument("--disc-updates", type=int, default=4)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
