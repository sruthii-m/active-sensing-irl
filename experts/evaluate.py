"""Success rate / reward curve for a trained expert."""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from environment import (
    make_full_obs_learning_env,
    make_partial_obs_learning_env,
)


def evaluate(
    model_path: str,
    env_kwargs: dict | None = None,
    num_episodes: int = 100,
    observation_mode: str = "full",
) -> dict:
    """Evaluate a saved expert deterministically and report the baseline gate."""
    if observation_mode == "full":
        env = make_full_obs_learning_env(**(env_kwargs or {}))
        model, recurrent = PPO.load(model_path), False
    elif observation_mode == "partial":
        from sb3_contrib import RecurrentPPO

        env = make_partial_obs_learning_env(**(env_kwargs or {}))
        model, recurrent = RecurrentPPO.load(model_path), True
    else:
        raise ValueError("observation_mode must be 'full' or 'partial'.")

    returns, lengths, successes = [], [], []
    for episode in range(num_episodes):
        observation, _ = env.reset(seed=episode)
        terminated = truncated = False
        episode_return = 0.0
        length = 0
        state = None
        episode_start = np.array([True])
        while not (terminated or truncated):
            if recurrent:
                action, state = model.predict(
                    observation, state=state, episode_start=episode_start, deterministic=True
                )
                episode_start = np.array([False])
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            length += 1
        returns.append(episode_return)
        lengths.append(length)
        successes.append(bool(terminated and episode_return > 0))
    env.close()
    return {
        "episodes": num_episodes,
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a foraging expert.")
    parser.add_argument("model_path")
    parser.add_argument("--mode", choices=("full", "partial"), default="full")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    print(evaluate(args.model_path, num_episodes=args.episodes, observation_mode=args.mode))
