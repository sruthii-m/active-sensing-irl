"""Roll out the trained full-state expert and save trajectories for train_iq.py's `demo:` config."""

from __future__ import annotations

import pickle
from pathlib import Path

from environment.experiment_config import add_environment_args, environment_kwargs

import numpy as np
from stable_baselines3 import PPO
from imitation.episode_contract import task_terminal

from environment import make_full_obs_learning_env


def generate(model_path: str, env_kwargs: dict, num_episodes: int, out_path: str, seed: int = 0, keep_all: bool = True):
    model = PPO.load(model_path)
    env = make_full_obs_learning_env(**(env_kwargs or {}))

    states, next_states, actions, rewards, dones, lengths = [], [], [], [], [], []
    terminated_flags, truncated_flags, episode_seeds = [], [], []
    horizon = env.unwrapped.max_steps
    attempts = 0
    while len(lengths) < num_episodes:
        obs, _ = env.reset(seed=seed + attempts)
        attempts += 1
        ep_states, ep_next_states, ep_actions, ep_rewards, ep_dones = [], [], [], [], []
        ep_terminated, ep_truncated = [], []
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_states.append(np.asarray(obs, dtype=np.float32))
            ep_next_states.append(np.asarray(next_obs, dtype=np.float32))
            ep_actions.append(int(action))
            ep_rewards.append(float(reward))
            ep_terminated.append(bool(terminated))
            ep_truncated.append(bool(truncated))
            ep_dones.append(task_terminal(terminated, truncated, len(ep_actions), horizon))
            obs = next_obs

        # Optional legacy filtering; failures are also part of expert behavior.
        if not keep_all and sum(ep_rewards) <= 0:
            continue

        states.append(np.stack(ep_states))
        next_states.append(np.stack(ep_next_states))
        actions.append(np.array(ep_actions, dtype=np.int64))
        rewards.append(np.array(ep_rewards, dtype=np.float32))
        dones.append(np.array(ep_dones, dtype=bool))
        lengths.append(len(ep_actions))
        terminated_flags.append(np.asarray(ep_terminated, dtype=bool))
        truncated_flags.append(np.asarray(ep_truncated, dtype=bool))
        episode_seeds.append(seed + attempts - 1)
    env.close()
    print(f"Kept {num_episodes} trajectories out of {attempts} rollouts.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "selection": "all" if keep_all else "successful_filtered",
                "expert_model": model_path,
                "deterministic": True,
                "env_kwargs": env_kwargs or {},
                "seed_start": seed,
                "attempts": attempts,
                "states": states,
                "next_states": next_states,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "task_terminal": dones,
                "terminated": terminated_flags,
                "truncated": truncated_flags,
                "task_horizon": horizon,
                "terminal_semantics": "intrinsic_horizon",
                "starts_at_reset": True,
                "episode_seeds": episode_seeds,
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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--keep-all", dest="keep_all", action="store_true", help="Retain every expert rollout, including failures and long episodes (default).")
    selection.add_argument("--successful-only", dest="keep_all", action="store_false", help="Legacy filtered-data experiment only: exclude failures.")
    parser.set_defaults(keep_all=True)
    add_environment_args(parser)
    args = parser.parse_args()
    generate(args.model_path, env_kwargs=environment_kwargs(args), num_episodes=args.num_episodes, out_path=args.out_path, seed=args.seed, keep_all=args.keep_all)
