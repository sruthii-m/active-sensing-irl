"""Fresh PPO on a frozen reward, or an identical true-reward optimizer control."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .data import digest, write_json
from .env import DecisionEnv
from .evaluate import PPOPolicy, rollouts
from .reward_exports import FrozenReward


class RewardEnv(gym.Wrapper):
    def __init__(self, env, reward):
        super().__init__(env)
        self.reward = reward
        self.x = None

    def reset(self, **kwargs):
        self.x, info = self.env.reset(**kwargs)
        return self.x, info

    def step(self, action):
        nxt, true_reward, terminated, truncated, info = self.env.step(action)
        learned = self.reward(self.x, int(action))
        self.x = nxt
        return nxt, learned, terminated, truncated, info


def make_ppo(env, seed=0, steps=512, batch_size=256, epochs=10, gamma=.99, lr=3e-4):
    return PPO("MlpPolicy", env, seed=seed, learning_rate=lr, n_steps=steps,
               batch_size=batch_size, n_epochs=epochs, gamma=gamma, gae_lambda=.95,
               clip_range=.2, ent_coef=0, device="cpu",
               policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=torch.nn.ReLU))


def train(args):
    if min(args.timesteps, args.envs, args.steps_per_env, args.batch_size, args.epochs, args.eval_episodes) < 1:
        raise ValueError("Positive training and evaluation budgets required")
    torch.set_num_threads(1)
    reward = None if args.reward == "true" else FrozenReward.load(args.reward)
    if reward and reward.metadata["domain"] != args.representation:
        raise ValueError("Frozen reward and policy representation differ")
    if reward and args.gamma != reward.metadata["gamma"]:
        raise ValueError("Use the reward's training discount for the matched recovery control")
    before = None if reward is None else reward.fingerprint()
    source_digest = None if reward is None else digest(args.reward)
    def factory():
        env = DecisionEnv(args.representation, args.view_size)
        return Monitor(env if reward is None else RewardEnv(env, reward))
    vec = DummyVecEnv([factory for _ in range(args.envs)])
    vec.seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    condition = dict(representation=args.representation, view_size=args.view_size, horizon=400, collection_limit=None)
    config = {**vars(args), "condition": condition, "reward_fingerprint": before,
              "optimizer": "PPO 128,128 pi/vf; ReLU; GAE .95; clip .2; entropy 0",
              "training_sensor": None if reward is None else reward.metadata["view_size"],
              "sensor_transfer": bool(reward and reward.metadata["view_size"] != args.view_size),
              "extrapolation_flag": bool(reward and reward.metadata["view_size"] != args.view_size),
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "stable-baselines3", "gymnasium", "minigrid")}}
    write_json(out / "config.json", config)
    try:
        model = make_ppo(vec, args.seed, args.steps_per_env, args.batch_size, args.epochs, args.gamma, args.lr)
        model.learn(total_timesteps=args.timesteps)
        model.save(out / "policy.zip")
        report = rollouts(PPOPolicy(model), args.representation, args.view_size, args.eval_episodes, args.eval_seed)
        if reward and (before != reward.fingerprint() or source_digest != digest(args.reward)):
            raise RuntimeError("Reward changed during fresh policy training")
        report.update(actual_interactions=model.num_timesteps, reward_freeze_check=reward is None or before == reward.fingerprint(),
                      scope="optimizer smoke/learnability control; budget adequacy must be assessed from true reward")
        write_json(out / "evaluation.json", report)
        return report
    finally:
        vec.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", required=True, help="true or a frozen reward.pt bundle")
    parser.add_argument("--representation", choices=("full", "belief"), required=True)
    parser.add_argument("--view-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=200704)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps-per-env", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=2000000)
    parser.add_argument("--output", required=True)
    report = train(parser.parse_args())
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
