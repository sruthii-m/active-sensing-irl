"""Shared expert-action likelihood and validation selection for offline IQ-Learn."""

from dataclasses import dataclass
import hashlib
import pickle

import numpy as np
import torch


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


def load_training_validation(demo_path, validation_demo_path, validation_fraction, seed, device):
    all_episodes = load_episodes(demo_path, device)
    if validation_demo_path:
        validation = load_episodes(validation_demo_path, device)
        if {episode_fingerprint(ep) for ep in all_episodes} & {episode_fingerprint(ep) for ep in validation}:
            raise ValueError('Training and validation contain identical trajectories.')
        episodes = all_episodes
        train_indices = list(range(len(episodes)))
        validation_indices = list(range(len(validation)))
    else:
        train_indices, validation_indices = validation_split(all_episodes, validation_fraction, seed)
        episodes = [all_episodes[i] for i in train_indices]
        validation = [all_episodes[i] for i in validation_indices]
    if any(ep[0].shape[1:] != episodes[0][0].shape[1:] for ep in validation):
        raise ValueError('Validation observation shape does not match training data.')
    return episodes, validation, train_indices, validation_indices


@torch.no_grad()
def evaluate_imitation(agent, episodes):
    """Equal episode mean NLL; recurrent memory starts fresh for each episode."""
    modules = [agent.q_net]
    if hasattr(agent, 'encoder'):
        modules.append(agent.encoder)
    modes = [module.training for module in modules]
    for module in modules:
        module.eval()
    episode_nll, total_nll, steps, correct = [], 0., 0, 0
    try:
        for states, next_states, actions, *_ in episodes:
            features = agent.encode_episode(states, next_states)[:-1] if hasattr(agent, 'encoder') else states
            q = agent.q_net(features)
            logp = (q / agent.alpha).log_softmax(-1)
            nll = -logp.gather(1, actions.long()).squeeze(1)
            episode_nll.append(float(nll.mean()))
            total_nll += float(nll.sum())
            steps += len(states)
            correct += int((q.argmax(-1) == actions.reshape(-1)).sum())
    finally:
        for module, mode in zip(modules, modes):
            module.train(mode)
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
