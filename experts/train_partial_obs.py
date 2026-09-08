"""Partially-observed expert: RecurrentPPO (sb3-contrib, LSTM/GRU) on the
egocentric foraging env. Match train_full_state_expert.py's hyperparameters
and network capacity as closely as possible so later behavioral differences
are attributable to observability, not algorithm choice. Confirm with
forage_env.fraction_goal_initially_visible that episodes actually contain
enough initial uncertainty before trusting results here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO

from perception_irl.environment import make_partial_obs_learning_env
from perception_irl.environment.forage_env import fraction_goal_initially_visible


def train(*, total_timesteps: int, seed: int = 0, save_path: str, env_kwargs: dict | None = None) -> RecurrentPPO:
    """Train the egocentric recurrent expert used for later demonstrations."""
    actual_env_kwargs = env_kwargs or {}
    initially_visible = fraction_goal_initially_visible(
        num_episodes=200, **actual_env_kwargs
    )
    print(f"Initial goal visibility over 200 resets: {initially_visible:.1%}")
    env = make_partial_obs_learning_env(**actual_env_kwargs)
    model = RecurrentPPO(
        "MlpLstmPolicy", env, seed=seed, verbose=1, learning_rate=3e-4,
        n_steps=256, batch_size=64, gamma=0.99,
        policy_kwargs={"net_arch": [64, 64], "lstm_hidden_size": 128},
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
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, seed=args.seed, save_path=args.save_path)
