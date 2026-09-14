"""Roll out the partial-obs expert and save trajectories in the same format
as generate_full_state_demos.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO

from environment import make_partial_obs_learning_env


def generate(model_path: str, env_kwargs: dict, num_episodes: int, out_path: str, seed: int = 0, max_length: int | None = 100):
    model = RecurrentPPO.load(model_path)
    env = make_partial_obs_learning_env(**(env_kwargs or {}))

    states, next_states, actions, rewards, dones, lengths = [], [], [], [], [], []
    attempts = 0
    while len(lengths) < num_episodes:
        obs, _ = env.reset(seed=seed + attempts)
        attempts += 1
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        ep_states, ep_next_states, ep_actions, ep_rewards, ep_dones = [], [], [], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_states.append(np.asarray(obs, dtype=np.float32))
            ep_next_states.append(np.asarray(next_obs, dtype=np.float32))
            ep_actions.append(int(action))
            ep_rewards.append(float(reward))
            ep_dones.append(bool(terminated))
            obs = next_obs
            episode_start = np.zeros((1,), dtype=bool)

        # drop failures and lucky near-timeout wins
        if sum(ep_rewards) <= 0 or (max_length is not None and len(ep_actions) > max_length):
            continue

        states.append(np.stack(ep_states))
        next_states.append(np.stack(ep_next_states))
        actions.append(np.array(ep_actions, dtype=np.int64))
        rewards.append(np.array(ep_rewards, dtype=np.float32))
        dones.append(np.array(ep_dones, dtype=bool))
        lengths.append(len(ep_actions))
    env.close()
    print(f"Kept {num_episodes} successful trajectories out of {attempts} rollouts.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "states": states,
                "next_states": next_states,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "lengths": np.array(lengths, dtype=np.int64),
            },
            f,
        )
    print(
        f"Saved {num_episodes} trajectories to {out_path} "
        f"(mean length {np.mean(lengths):.1f}, success rate 100.0%)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate IQ-Learn demos from the partial-obs expert.")
    parser.add_argument("--model-path", default="artifacts/experts/forage_partial_obs")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--out-path", default="artifacts/demos/forage_partial_obs_demos.pkl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=100, help="Reject successful episodes longer than this (near-timeout, likely lucky rather than confident).")
    args = parser.parse_args()
    generate(args.model_path, env_kwargs=None, num_episodes=args.num_episodes, out_path=args.out_path, seed=args.seed, max_length=args.max_length)
