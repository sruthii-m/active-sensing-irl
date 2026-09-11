"""Step 2 of the active-sensing-irl plan: offline IQ-Learn on full-state
demonstrations. Two gates must both pass before moving to step 3 (partial-obs):

  1. Recovered reward correlates with ground-truth reward.
  2. A policy read off the learned Q-function (softmax(Q/alpha), argmax at
     eval time) reproduces the expert's success rate in the real env --
     reward correlation alone is a weak check given this task's sparse
     terminal reward (a constant-zero reward "correlates" with a
     mostly-zero target almost as well as a correct one would).
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import torch

from environment import make_full_obs_learning_env
from imitation.iq_learn_agent import OfflineSoftQAgent, make_args


def load_transitions(demo_path: str):
    with open(demo_path, "rb") as f:
        data = pickle.load(f)
    obs = np.concatenate(data["states"], axis=0)
    next_obs = np.concatenate(data["next_states"], axis=0)
    actions = np.concatenate(data["actions"], axis=0)
    rewards = np.concatenate(data["rewards"], axis=0)
    dones = np.concatenate(data["dones"], axis=0)
    return obs, next_obs, actions, rewards, dones


def evaluate_policy_success_rate(agent: OfflineSoftQAgent, n_episodes: int, env_kwargs: dict | None = None) -> float:
    env = make_full_obs_learning_env(**(env_kwargs or {}))
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0.0
        while not (terminated or truncated):
            action = agent.choose_action(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        if episode_reward > 0:
            successes += 1
    env.close()
    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser(description="Offline IQ-Learn on full-state demonstrations.")
    parser.add_argument("--demo-path", default="artifacts/demos/forage_full_state_demos.pkl")
    parser.add_argument("--save-path", default="artifacts/iq_learn/forage_full_state_qnet.pt")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init-temp", type=float, default=0.001)
    parser.add_argument("--target-update-freq", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs, next_obs, actions, rewards, dones = load_transitions(args.demo_path)
    print(f"Loaded {len(actions)} transitions from {args.demo_path}")

    agent = OfflineSoftQAgent(
        obs_dim=obs.shape[1],
        action_dim=7,
        args=make_args(gamma=args.gamma, init_temp=args.init_temp, critic_target_update_frequency=args.target_update_freq),
        lr=args.lr,
    )

    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1)
    n = obs_t.shape[0]

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,))
        loss_dict = agent.update(
            obs_t[idx], next_obs_t[idx], actions_t[idx], dones_t[idx],
            step=step, target_update_freq=args.target_update_freq,
        )
        if step % 2000 == 0:
            print(f"step {step}: {loss_dict}")

    with torch.no_grad():
        recovered_reward = agent.infer_reward(obs_t, actions_t, next_obs_t).squeeze(1).numpy()
    ground_truth_reward = rewards
    correlation = np.corrcoef(recovered_reward, ground_truth_reward)[0, 1]
    print(f"\nGate 1 -- reward correlation (recovered vs. ground truth): {correlation:.3f}")

    success_rate = evaluate_policy_success_rate(agent, n_episodes=args.eval_episodes)
    print(f"Gate 2 -- policy success rate from learned Q ({args.eval_episodes} episodes): {success_rate:.2%}")

    import pathlib
    pathlib.Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.save_path)
    print(f"Saved learned Q-network to {args.save_path}")


if __name__ == "__main__":
    main()
