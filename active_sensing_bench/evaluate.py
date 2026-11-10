"""Shared held-out action likelihood and paired independent rollouts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import digest, load_split, write_json
from .env import DecisionEnv
from .networks import mlp


class MLPPolicy:
    def __init__(self, network, temperature=1):
        self.network, self.temperature = network.eval(), temperature

    @torch.no_grad()
    def log_probs(self, x):
        return (self.network(torch.as_tensor(x, dtype=torch.float32)) / self.temperature).log_softmax(-1).numpy()

    def act(self, x, rng, deterministic=True):
        lp = self.log_probs(x[None])[0]
        return int(lp.argmax() if deterministic else rng.choice(7, p=np.exp(lp)))


class PPOPolicy:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def log_probs(self, x):
        tensor = self.model.policy.obs_to_tensor(x)[0]
        return self.model.policy.get_distribution(tensor).distribution.logits.cpu().numpy()

    def act(self, x, rng, deterministic=True):
        lp = self.log_probs(x[None])[0]
        return int(lp.argmax() if deterministic else rng.choice(7, p=np.exp(lp)))


def load_policy(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if (directory / "policy.pt").exists():
        payload = torch.load(directory / "policy.pt", map_location="cpu", weights_only=True)
        model = mlp()
        model.load_state_dict(payload["network"])
        return MLPPolicy(model, payload["temperature"]), config
    from stable_baselines3 import PPO
    return PPOPolicy(PPO.load(directory / "policy.zip", device="cpu")), config


def likelihood(policy, episodes):
    rows = []
    for episode in episodes:
        lp = policy.log_probs(episode["observations"][:-1])
        actions = episode["actions"]
        nll = -lp[np.arange(len(actions)), actions]
        rows.append(dict(episode_id=episode["episode_id"], length=len(actions),
                         nll_sum=float(nll.sum()), nll_mean=float(nll.mean()),
                         correct=int((lp.argmax(1) == actions).sum())))
    steps = sum(r["length"] for r in rows)
    return dict(episode_mean_nll=float(np.mean([r["nll_mean"] for r in rows])),
                step_mean_nll=sum(r["nll_sum"] for r in rows) / steps,
                trajectory_sum_nll=sum(r["nll_sum"] for r in rows) / len(rows),
                accuracy=sum(r["correct"] for r in rows) / steps, episodes=rows)


def rollouts(policy, representation, view_size=3, episodes=200, seed=2000000, deterministic=True):
    if episodes < 1:
        raise ValueError("Positive rollout count required")
    env, rng, rows = DecisionEnv(representation, view_size), np.random.default_rng(seed), []
    try:
        for i in range(episodes):
            x, _ = env.reset(seed=seed + i)
            total, length, unresolved, navigation = 0., 0, 0, 0
            while True:
                action = policy.act(x, rng, deterministic)
                if x[7:].max() < 1:
                    unresolved += 1
                    navigation += int(action < 3)
                x, reward, terminated, truncated, info = env.step(action)
                total += reward
                length += 1
                if terminated or truncated:
                    break
            rows.append(dict(reset_seed=seed+i, true_return=total, success=info["terminated"],
                             length=length, unresolved_steps=unresolved, unresolved_navigation=navigation))
    finally:
        env.close()
    return dict(mean_true_return=float(np.mean([r["true_return"] for r in rows])),
                success_rate=float(np.mean([r["success"] for r in rows])),
                mean_length=float(np.mean([r["length"] for r in rows])),
                deterministic=deterministic, pairing="same reset seeds across agents", episodes=rows)


def reward_coverage(reward, episodes, max_states=256):
    """Declared deterministic coverage, with no reward-dependent sample selection."""
    from .belief_model import decode_state
    from .env import move
    from .state_views import goal_index
    all_states = np.concatenate([e["observations"][:-1] for e in episodes])
    indices = np.linspace(0, len(all_states) - 1, min(max_states, len(all_states)), dtype=int)
    states = np.repeat(all_states[indices], 7, axis=0)
    actions = np.tile(np.arange(7), len(indices))
    before = reward.fingerprint()
    values = reward(states, actions)
    reference = np.zeros(len(states))
    for i, (x, action) in enumerate(zip(states, actions)):
        pose, t = decode_state(x)
        nxt = move(pose, action)
        if action == 2 and nxt[:2] != pose[:2]:
            reference[i] = x[7 + goal_index(nxt[:2])] * (1 - .9 * (t + 1) / 400)
    correlation = None
    if np.std(values) > 1e-12 and np.std(reference) > 1e-12:
        correlation = float(np.corrcoef(values, reference)[0, 1])
    return dict(coverage="equally spaced held-out expert states, all seven actions; fixed before reward queries",
                state_count=len(indices), state_action_count=len(states), finite=bool(np.isfinite(values).all()),
                minimum=float(values.min()), maximum=float(values.max()),
                immediate_true_reward_pearson_diagnostic=correlation,
                interpretation="Correlation is descriptive, not shaping-invariant reward recovery or a pass/fail test",
                reward_freeze_check=before == reward.fingerprint()), dict(states=states, actions=actions,
                recovered_reward=values, expected_true_immediate_reward=reference)


def evaluate_run(run, data, output, episodes=200, seed=2000000):
    policy, config = load_policy(run)
    test, manifest = load_split(data, "test")
    if config["condition"] != manifest["condition"]:
        raise ValueError("Evaluation condition mismatch")
    if Path(output).exists():
        raise FileExistsError(output)
    report = dict(test_data_sha256=manifest["splits"]["test"]["sha256"],
                  imitation=likelihood(policy, test), rollouts=rollouts(policy,
                  config["condition"]["representation"], config["condition"]["view_size"], episodes, seed))
    reward_path = Path(run) / "reward.pt"
    if reward_path.exists():
        from .reward_exports import FrozenReward
        report["reward"], coverage = reward_coverage(FrozenReward.load(reward_path), test)
        report["reward"]["bundle_sha256"] = digest(reward_path)
        coverage_path = Path(output).with_suffix(".coverage.npz")
        if coverage_path.exists():
            raise FileExistsError(coverage_path)
        np.savez_compressed(coverage_path, **coverage)
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2000000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    evaluate_run(args.run, args.data, args.output, args.episodes, args.seed)


if __name__ == "__main__":
    main()
