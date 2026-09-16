"""Offline IQ-Learn with a recurrent encoder.
    Uses same architecture on validated full-state demos to ensure training loop works, 
    then applies same gates with partial observation.
    FULL-STATE CHECK:
    Reward correlation (recovered vs. ground truth): 0.606
    Policy success rate from learned Q (100 episodes): 88.00%

"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import torch

from environment import make_full_obs_learning_env, make_partial_obs_learning_env
from imitation.recurrent_iq_learn_agent import RecurrentOfflineSoftQAgent, make_args

ENVS = {
    "full_state": make_full_obs_learning_env, 
    "partial_obs": make_partial_obs_learning_env,
}


def load_episodes(demo_path: str, device: str):
    with open(demo_path, "rb") as f:
        data = pickle.load(f)
    episodes = []
    for states, next_states, actions, rewards, dones in zip(
        data["states"], data["next_states"], data["actions"], data["rewards"], data["dones"]
    ):
        episodes.append(
            (
                torch.as_tensor(states, dtype=torch.float32, device=device),
                torch.as_tensor(next_states, dtype=torch.float32, device=device),
                torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1),
                torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
                torch.as_tensor(rewards, dtype=torch.float32, device=device),
            )
        )
    return episodes


def evaluate_policy_success_rate(agent: RecurrentOfflineSoftQAgent, env_name: str, n_episodes: int, device: str) -> float:
    env = ENVS[env_name]()
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        agent.reset_hidden()
        terminated = truncated = False
        episode_reward = 0.0
        while not (terminated or truncated):
            action = agent.act(torch.as_tensor(obs, dtype=torch.float32, device=device), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        if episode_reward > 0:
            successes += 1
    env.close()
    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser(description="Offline recurrent IQ-Learn.")
    parser.add_argument("--env", choices=list(ENVS), required=True)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--episodes-per-batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init-temp", type=float, default=0.001)
    parser.add_argument("--chi2-alpha", type=float, default=0.5, help="method.alpha: chi2_loss = reward^2/(4*alpha), so HIGHER alpha means WEAKER regularization (less pull toward zero Q-margins).")
    parser.add_argument("--target-update-freq", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--rnn-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--grid-shape", type=int, nargs=3, default=None,
        help="H W C to reshape the leading part of obs into for a CNN front-end (e.g. 7 7 3 for partial_obs). Omit for a plain MLP encoder.",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Defaults to cuda if available, else cpu.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    episodes = load_episodes(args.demo_path, device)
    obs_dim = episodes[0][0].shape[1]
    print(f"Loaded {len(episodes)} episodes from {args.demo_path} (obs_dim={obs_dim}, env={args.env})")

    agent = RecurrentOfflineSoftQAgent(
        obs_dim=obs_dim,
        action_dim=7,
        args=make_args(gamma=args.gamma, init_temp=args.init_temp, critic_target_update_frequency=args.target_update_freq, alpha=args.chi2_alpha),
        latent_dim=args.latent_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        device=device,
        lr=args.lr,
        grid_shape=tuple(args.grid_shape) if args.grid_shape else None,
    )

    rng = np.random.default_rng(args.seed)
    start = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.choice(len(episodes), size=min(args.episodes_per_batch, len(episodes)), replace=False)
        batch = [(episodes[i][0], episodes[i][1], episodes[i][2], episodes[i][3]) for i in idx]
        loss_dict = agent.update(batch, step=step, target_update_freq=args.target_update_freq)
        if step % 100 == 0:
            elapsed = time.time() - start
            steps_per_sec = step / elapsed
            eta_min = (args.steps - step) / steps_per_sec / 60
            print(
                f"step {step}/{args.steps} ({steps_per_sec:.1f} steps/s, ETA {eta_min:.1f} min): {loss_dict}",
                flush=True,
            )

    # Gate 1: reward correlation, computed over every stored transition.
    all_recovered, all_ground_truth = [], []
    with torch.no_grad():
        for states, next_states, actions, dones, rewards in episodes:
            hiddens = agent.encode_episode(states, next_states)
            h_t, h_tp1 = hiddens[:-1], hiddens[1:]
            recovered = agent.critic(h_t, actions) - agent.gamma * agent.getV(h_tp1)
            all_recovered.append(recovered.squeeze(1).cpu().numpy())
            all_ground_truth.append(rewards.cpu().numpy())
    recovered = np.concatenate(all_recovered)
    ground_truth = np.concatenate(all_ground_truth)
    correlation = np.corrcoef(recovered, ground_truth)[0, 1]
    print(f"\nReward correlation (recovered vs. ground truth): {correlation:.3f}")

    success_rate = evaluate_policy_success_rate(agent, args.env, args.eval_episodes, device)
    print(f"Policy success rate from learned Q ({args.eval_episodes} episodes): {success_rate:.2%}")

    if args.save_path:
        import pathlib
        pathlib.Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        agent.save(args.save_path)
        print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
