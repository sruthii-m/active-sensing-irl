"""Partially-observed expert: RecurrentPPO (sb3-contrib, LSTM/GRU) on the
egocentric foraging env.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

from environment import make_partial_obs_learning_env
from environment.forage_env import fraction_goal_initially_visible


def evaluate_success_rate(model: RecurrentPPO, *, n_episodes: int = 50, env_kwargs: dict | None = None) -> float:
    """Fraction of episodes that reach the goal, under the deterministic
    policy, carrying the LSTM hidden state across steps within an episode."""
    env = make_partial_obs_learning_env(**(env_kwargs or {}))
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        terminated = truncated = False
        episode_reward = 0.0
        while not (terminated or truncated):
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_start = np.zeros((1,), dtype=bool)
        if episode_reward > 0:
            successes += 1
    env.close()
    return successes / n_episodes


def train(
    *,
    total_timesteps: int,
    seed: int = 0,
    save_path: str,
    env_kwargs: dict | None = None,
    learning_rate: float = 4e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    net_arch: list[int] | None = None,
    lstm_hidden_size: int = 128,
    n_envs: int = 1,
) -> RecurrentPPO:
    """Train the egocentric recurrent expert used for later demonstrations."""
    actual_env_kwargs = env_kwargs or {}
    initially_visible = fraction_goal_initially_visible(
        num_episodes=200, **actual_env_kwargs
    )
    print(f"Initial goal visibility over 200 resets: {initially_visible:.1%}")
    env = make_vec_env(lambda: make_partial_obs_learning_env(**actual_env_kwargs), n_envs=n_envs)
    model = RecurrentPPO(
        "MlpLstmPolicy", env, seed=seed, verbose=1,
        learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
        n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
        clip_range=clip_range, ent_coef=ent_coef,
        policy_kwargs={"net_arch": net_arch or [64, 64], "lstm_hidden_size": lstm_hidden_size},
    )
    model.learn(total_timesteps=total_timesteps)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the partial-observation foraging expert.")
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="artifacts/experts/forage_partial_obs")
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--n-envs", type=int, default=1, help="Parallel envs collecting rollouts (helps discover sparse reward faster).")
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()
    trained = train(
        total_timesteps=args.timesteps, seed=args.seed, save_path=args.save_path,
        learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip_range, ent_coef=args.ent_coef, net_arch=args.net_arch,
        lstm_hidden_size=args.lstm_hidden_size, n_envs=args.n_envs,
    )
    success_rate = evaluate_success_rate(trained, n_episodes=args.eval_episodes)
    print(f"Success rate over {args.eval_episodes} episodes: {success_rate:.2%}")
