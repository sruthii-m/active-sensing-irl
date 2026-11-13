"""Offline full-state IQ-Learn selected by held-out expert-action likelihood.

Reward correlation and task success are diagnostics evaluated after restoring
the minimum validation episode-mean NLL checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle

from environment.experiment_config import add_environment_args, environment_kwargs

import numpy as np
import torch

from environment import make_full_obs_learning_env
from imitation.iq_learn_agent import OfflineSoftQAgent, make_args
from imitation.selection import LikelihoodStopping, evaluate_imitation, load_training_validation


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
    parser.add_argument("--validation-demo-path", help="Separate selection demos; never use the final test set.")
    parser.add_argument("--validation-fraction", type=float, default=.2, help="Grouped episode holdout when no validation file is provided.")
    parser.add_argument("--steps", "--max-steps", dest="steps", type=int, default=20_000, help="Safety cap; training may stop on validation NLL plateau.")
    parser.add_argument("--eval-freq", type=int, default=2000, help="Steps between held-out likelihood evaluations.")
    parser.add_argument("--patience", type=int, default=5, help="Stop after this many evals with no improvement > --min-delta; zero disables early stopping.")
    parser.add_argument("--min-delta", type=float, default=.001, help="Minimum decrease in validation episode-mean NLL (nats) to reset patience.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init-temp", type=float, default=0.001)
    parser.add_argument('--terminal-semantics', choices=['intrinsic_horizon', 'recorded'], default='intrinsic_horizon',
                        help='Zero bootstrap at the task deadline; recorded reproduces legacy demo masks.')
    parser.add_argument("--target-update-freq", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    add_environment_args(parser)
    args = parser.parse_args()
    if min(args.steps, args.eval_freq, args.batch_size, args.target_update_freq, args.eval_episodes) < 1 or args.patience < 0 or not np.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("Positive step/evaluation counts and nonnegative patience/min-delta required.")
    if not np.isfinite(args.init_temp) or args.init_temp <= 0:
        parser.error("--init-temp must be positive and finite.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split_audit = {}
    episodes, validation, train_indices, validation_indices = load_training_validation(
        args.demo_path, args.validation_demo_path, args.validation_fraction, args.seed, "cpu",
        split_audit=split_audit,
        terminal_semantics=args.terminal_semantics, horizon=4 * args.size ** 2,
    )
    print(f'Training episodes: {len(episodes)}; validation episodes: {len(validation)} (never used for gradient updates).')
    obs_t, next_obs_t, actions_t, dones_t, rewards_t = (
        torch.cat([ep[index] for ep in episodes], dim=0) for index in range(5)
    )
    print(f"Loaded {len(obs_t)} training transitions from {args.demo_path}")

    agent = OfflineSoftQAgent(
        obs_dim=obs_t.shape[1],
        action_dim=7,
        args=make_args(gamma=args.gamma, init_temp=args.init_temp, critic_target_update_frequency=args.target_update_freq),
        lr=args.lr,
        terminal_semantics=split_audit['terminals']['training']['effective_semantics'],
    )

    n = obs_t.shape[0]
    stopping = LikelihoodStopping(args.min_delta, args.patience)
    best_state, best_step = None, 0
    selection = dict(config=vars(args), metric='episode_mean_nll', units='nats',
                     demo_sha256=hashlib.sha256(Path(args.demo_path).read_bytes()).hexdigest(),
                     validation_sha256=hashlib.sha256(Path(args.validation_demo_path or args.demo_path).read_bytes()).hexdigest(),
                     train_indices=train_indices, validation_indices=validation_indices,
                     split_audit=split_audit, evaluations=[])
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    def save_selection():
        Path(str(args.save_path) + '.selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))

    def validate(step):
        nonlocal best_state, best_step
        metrics = evaluate_imitation(agent, validation)
        improved, stop = stopping.observe(metrics['episode_mean_nll'])
        selection['evaluations'].append(dict(step=step, **metrics))
        if improved:
            best_step = step
            best_state = copy.deepcopy(agent.q_net.state_dict())
            agent.save(args.save_path)
        selection.update(best_step=best_step, best_validation_nll=stopping.best)
        save_selection()
        print(f"  [validation @ step {step}] episode NLL: {metrics['episode_mean_nll']:.6f}; best: {stopping.best:.6f}", flush=True)
        return stop

    validate(0)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,))
        loss_dict = agent.update(
            obs_t[idx], next_obs_t[idx], actions_t[idx], dones_t[idx],
            step=step, target_update_freq=args.target_update_freq,
        )
        if step % 2000 == 0:
            print(f"step {step}: {loss_dict}")
        if step % args.eval_freq == 0 or step == args.steps:
            if validate(step):
                print(f'Validation NLL plateaued for {args.patience} evaluations. Stopping at step {step}.')
                break

    agent.q_net.load_state_dict(best_state)
    agent.target_net.load_state_dict(best_state)
    print(f'Restored step {best_step}: best validation episode NLL {stopping.best:.6f}.')

    # Diagnostic only: reward correlation on the training transitions.
    with torch.no_grad():
        recovered_reward = (agent.critic(obs_t, actions_t)
                            - (1 - dones_t) * agent.gamma * agent.getV(next_obs_t)).squeeze(1).numpy()
    ground_truth_reward = rewards_t.numpy()
    correlation = (np.corrcoef(recovered_reward, ground_truth_reward)[0, 1]
                   if np.ptp(recovered_reward) > 0 and np.ptp(ground_truth_reward) > 0
                   else float('nan'))
    print(f"\nReward correlation on training transitions (diagnostic): {correlation:.3f}")

    success_rate = evaluate_policy_success_rate(agent, n_episodes=args.eval_episodes, env_kwargs=environment_kwargs(args))
    print(f"Policy success rate from learned Q (diagnostic, {args.eval_episodes} episodes): {success_rate:.2%}")

    selection.update(final_training_step=step, final_success_rate=success_rate,
                     stop_reason='validation_plateau' if stopping.patience > 0 and stopping.stale >= stopping.patience else 'max_steps')
    save_selection()
    agent.save(args.save_path)
    print(f"Saved learned Q-network to {args.save_path}")


if __name__ == "__main__":
    main()
