"""Offline recurrent IQ-Learn selected by held-out expert-action likelihood."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
import pickle
import time

from environment.experiment_config import add_environment_args, environment_kwargs

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
        data["states"], data["next_states"], data["actions"], data["rewards"], data["dones"], strict=True
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
    if not episodes or any(len(ep[0]) == 0 or any(len(v) != len(ep[0]) for v in ep) for ep in episodes):
        raise ValueError("Need nonempty episodes with matching transition arrays.")
    return episodes


def episode_fingerprint(episode):
    digest = hashlib.sha256()
    for tensor in episode[:4]:
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validation_split(episodes, fraction, seed):
    """Keep identical observation/action trajectories in the same split."""
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must lie strictly between zero and one.')
    fingerprints = [episode_fingerprint(ep) for ep in episodes]
    groups = sorted(set(fingerprints))
    if len(groups) < 2:
        raise ValueError('Need two distinct trajectories or a separate validation file.')
    order = np.random.default_rng(seed).permutation(len(groups))
    count = min(len(groups)-1, max(1, round(fraction * len(groups))))
    held = {groups[i] for i in order[:count]}
    train = [i for i, key in enumerate(fingerprints) if key not in held]
    valid = [i for i, key in enumerate(fingerprints) if key in held]
    return train, valid


@torch.no_grad()
def evaluate_imitation(agent, episodes):
    """Equal episode mean NLL; every episode starts with fresh recurrent memory."""
    modes = (agent.encoder.training, agent.q_net.training)
    agent.encoder.eval()
    agent.q_net.eval()
    episode_nll, total_nll, steps, correct = [], 0., 0, 0
    try:
        for states, next_states, actions, *_ in episodes:
            hidden = agent.encode_episode(states, next_states)[:-1]
            q = agent.q_net(hidden)
            logp = (q / agent.alpha).log_softmax(-1)
            nll = -logp.gather(1, actions.long()).squeeze(1)
            episode_nll.append(float(nll.mean()))
            total_nll += float(nll.sum())
            steps += len(states)
            correct += int((q.argmax(-1) == actions.reshape(-1)).sum())
    finally:
        agent.encoder.train(modes[0])
        agent.q_net.train(modes[1])
    score = float(np.mean(episode_nll))
    if not np.isfinite(score):
        raise ValueError('Nonfinite validation NLL; refusing checkpoint selection.')
    return dict(episode_mean_nll=score, step_mean_nll=total_nll/steps,
                action_accuracy=correct/steps, episodes=len(episodes), steps=steps)


@dataclass
class LikelihoodStopping:
    min_delta: float
    patience: int
    best: float = float('inf')
    anchor: float = float('inf')
    stale: int = 0

    def observe(self, nll):
        if not np.isfinite(nll):
            raise ValueError('Validation NLL must be finite.')
        improved = nll < self.best
        if improved:
            self.best = nll
        # Save every best model, but reset patience only for meaningful progress.
        if nll < self.anchor - self.min_delta:
            self.anchor, self.stale = nll, 0
        else:
            self.stale += 1
        return improved, self.patience > 0 and self.stale >= self.patience


def evaluate_policy_success_rate(agent: RecurrentOfflineSoftQAgent, env_name: str, n_episodes: int, device: str, env_kwargs: dict | None = None) -> float:
    env = ENVS[env_name](**(env_kwargs or {}))
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
    parser.add_argument("--checkpoint-dir", help="Save each validation-step model for analysis across learning; requires a new or empty directory.")
    parser.add_argument("--validation-demo-path", help="Separate selection demos; never use the final test set.")
    parser.add_argument("--validation-fraction", type=float, default=.2, help="Grouped episode holdout when no validation file is provided.")
    parser.add_argument("--max-steps", type=int, default=200_000, help="Safety cap; training may stop on validation NLL plateau.")
    parser.add_argument("--eval-freq", type=int, default=2000, help="Steps between held-out likelihood evaluations.")
    parser.add_argument("--patience", type=int, default=5, help="Stop after this many evals with no improvement > --min-delta.")
    parser.add_argument("--min-delta", type=float, default=0.001, help="Minimum decrease in validation episode-mean NLL (nats) to reset patience.")
    parser.add_argument("--plateau-eval-episodes", type=int, default=50, help="Deprecated compatibility option; success is evaluated only after checkpoint selection.")
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
    add_environment_args(parser)
    args = parser.parse_args()
    if args.checkpoint_dir:
        snapshot_dir = Path(args.checkpoint_dir)
        if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
            parser.error('--checkpoint-dir must be empty to avoid mixing training runs.')
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    if min(args.max_steps, args.eval_freq, args.episodes_per_batch, args.target_update_freq, args.eval_episodes) < 1 or args.patience < 0 or not np.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("Positive step/evaluation counts and nonnegative patience/min-delta required.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_episodes = load_episodes(args.demo_path, device)
    if args.validation_demo_path:
        validation = load_episodes(args.validation_demo_path, device)
        if {episode_fingerprint(ep) for ep in all_episodes} & {episode_fingerprint(ep) for ep in validation}:
            raise ValueError('Training and validation contain identical trajectories.')
        episodes = all_episodes
        train_indices, validation_indices = list(range(len(episodes))), list(range(len(validation)))
    else:
        train_indices, validation_indices = validation_split(all_episodes, args.validation_fraction, args.seed)
        episodes = [all_episodes[i] for i in train_indices]
        validation = [all_episodes[i] for i in validation_indices]
    if any(ep[0].shape[1:] != episodes[0][0].shape[1:] for ep in validation):
        raise ValueError('Validation observation shape does not match training data.')
    print(f'Training episodes: {len(episodes)}; validation episodes: {len(validation)} (never used for gradient updates).')
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
    stopping = LikelihoodStopping(args.min_delta, args.patience)
    best_state, best_step = None, 0
    selection = dict(config=vars(args), metric='episode_mean_nll', units='nats',
                     demo_sha256=hashlib.sha256(Path(args.demo_path).read_bytes()).hexdigest(),
                     validation_sha256=hashlib.sha256(Path(args.validation_demo_path or args.demo_path).read_bytes()).hexdigest(),
                     train_indices=train_indices, validation_indices=validation_indices, evaluations=[])

    def validate(step):
        nonlocal best_state, best_step
        metrics = evaluate_imitation(agent, validation)
        improved, stop = stopping.observe(metrics['episode_mean_nll'])
        record = dict(step=step, **metrics)
        if args.checkpoint_dir:
            snapshot = snapshot_dir / f'step_{step:09d}.pt'
            agent.save(str(snapshot))
            record['checkpoint'] = str(snapshot.resolve())
        selection['evaluations'].append(record)
        if improved:
            best_step = step
            best_state = copy.deepcopy((agent.encoder.state_dict(), agent.q_net.state_dict()))
            if args.save_path:
                Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
                agent.save(args.save_path)
        selection.update(best_step=best_step, best_validation_nll=stopping.best)
        if args.save_path:
            Path(str(args.save_path) + '.selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
        if args.checkpoint_dir:
            (snapshot_dir / 'manifest.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
        print(f"  [validation @ step {step}] episode NLL: {metrics['episode_mean_nll']:.6f}; best: {stopping.best:.6f}", flush=True)
        return stop

    validate(0)
    for step in range(1, args.max_steps + 1):
        idx = rng.choice(len(episodes), size=min(args.episodes_per_batch, len(episodes)), replace=False)
        batch = [(episodes[i][0], episodes[i][1], episodes[i][2], episodes[i][3]) for i in idx]
        loss_dict = agent.update(batch, step=step, target_update_freq=args.target_update_freq)
        if step % 100 == 0:
            elapsed = time.time() - start
            steps_per_sec = step / elapsed
            print(f"step {step}/{args.max_steps} ({steps_per_sec:.1f} steps/s): {loss_dict}", flush=True)

        if step % args.eval_freq == 0 or step == args.max_steps:
            if validate(step):
                print(f'Validation NLL plateaued for {args.patience} evaluations. Stopping at step {step}.')
                break

    agent.encoder.load_state_dict(best_state[0])
    agent.q_net.load_state_dict(best_state[1])
    agent.target_net.load_state_dict(best_state[1])
    print(f'Restored step {best_step}: best validation episode NLL {stopping.best:.6f}.')

    # Diagnostic only: reward correlation on the training transitions.
    all_recovered, all_ground_truth = [], []
    with torch.no_grad():
        for states, next_states, actions, dones, rewards in episodes:
            hiddens = agent.encode_episode(states, next_states)
            h_t, h_tp1 = hiddens[:-1], hiddens[1:]
            recovered = agent.critic(h_t, actions) - (1 - dones) * agent.gamma * agent.getV(h_tp1)
            all_recovered.append(recovered.squeeze(1).cpu().numpy())
            all_ground_truth.append(rewards.cpu().numpy())
    recovered = np.concatenate(all_recovered)
    ground_truth = np.concatenate(all_ground_truth)
    correlation = np.corrcoef(recovered, ground_truth)[0, 1]
    print(f"\nReward correlation (recovered vs. ground truth): {correlation:.3f}")

    success_rate = evaluate_policy_success_rate(agent, args.env, args.eval_episodes, device, environment_kwargs(args))
    print(f"Policy success rate from learned Q ({args.eval_episodes} episodes): {success_rate:.2%}")

    if args.save_path:
        selection.update(final_training_step=step, final_success_rate=success_rate,
                         stop_reason='validation_plateau' if stopping.patience > 0 and stopping.stale >= stopping.patience else 'max_steps')
        Path(str(args.save_path) + '.selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
        agent.save(args.save_path)
        print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
