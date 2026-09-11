"""Roll out the trained full-state expert and save trajectories in the format
iq_learn/dataset/expert_dataset.py expects, for train_iq.py's `demo:` config.

Gate before moving to step 3: a policy trained against the recovered reward
reproduces expert performance (reward correlation alone is a weak check given
this task's sparse terminal reward).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from environment import make_full_obs_learning_env


def generate(model_path: str, env_kwargs: dict, num_episodes: int, out_path: str, seed: int = 0):
    model = PPO.load(model_path)
    env = make_full_obs_learning_env(**(env_kwargs or {}))

    states, next_states, actions, rewards, dones, lengths = [], [], [], [], [], []
    attempts = 0
    while len(lengths) < num_episodes:
        obs, _ = env.reset(seed=seed + attempts)
        attempts += 1
        ep_states, ep_next_states, ep_actions, ep_rewards, ep_dones = [], [], [], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_states.append(np.asarray(obs, dtype=np.float32))
            ep_next_states.append(np.asarray(next_obs, dtype=np.float32))
            ep_actions.append(int(action))
            ep_rewards.append(float(reward))
            ep_dones.append(bool(terminated))
            obs = next_obs

        # Only keep successful rollouts as demonstrations — a timed-out episode
        # is not expert behavior and would teach IQ-Learn that wandering for
        # max_steps is optimal.
        if sum(ep_rewards) <= 0:
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
        f"(mean length {np.mean(lengths):.1f}, success rate {np.mean([r.sum() > 0 for r in rewards]):.1%})"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate IQ-Learn demos from the full-state expert.")
    parser.add_argument("--model-path", default="artifacts/experts/forage_full_state")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--out-path", default="artifacts/demos/forage_full_state_demos.pkl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate(args.model_path, env_kwargs=None, num_episodes=args.num_episodes, out_path=args.out_path, seed=args.seed)
