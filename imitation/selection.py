"""Shared expert-action likelihood and validation selection for offline IQ-Learn."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
import warnings

import numpy as np
import torch
from imitation.episode_contract import terminal_masks


def load_episodes(demo_path: str, device: str, *, terminal_semantics='recorded', horizon=None, terminal_audit=None):
    with open(demo_path, "rb") as f:
        data = pickle.load(f)
    return _episodes_from_data(data, device, terminal_semantics, horizon, terminal_audit)


def _episodes_from_data(data, device, terminal_semantics='recorded', horizon=None, terminal_audit=None):
    episodes = []
    masks = terminal_masks(data, terminal_semantics, horizon, terminal_audit)
    for states, next_states, actions, rewards, dones in zip(
        data["states"], data["next_states"], data["actions"], data["rewards"], masks, strict=True
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


def _generation_range(data):
    """Inclusive reset-seed interval, including discarded generation attempts."""
    for key in ('seed_start', 'attempts'):
        if key in data:
            value = data[key]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'Invalid demonstration provenance: {key} must be an integer.')
            if value < (1 if key == 'attempts' else 0):
                raise ValueError(f'Invalid demonstration provenance: {key} is out of range.')
    if 'attempts' in data:
        if data['attempts'] < len(data['states']):
            raise ValueError('Invalid demonstration provenance: fewer attempts than episodes.')
        if data.get('selection') == 'all' and data['attempts'] != len(data['states']):
            raise ValueError('Invalid demonstration provenance: unfiltered attempts must equal episodes.')
    if 'seed_start' not in data or 'attempts' not in data:
        return None
    start = int(data['seed_start'])
    return [start, start + int(data['attempts']) - 1]


def _validation_provenance(training_data, validation_data):
    train_range, valid_range = map(_generation_range, (training_data, validation_data))
    disjoint = None
    if train_range is not None and valid_range is not None:
        disjoint = train_range[1] < valid_range[0] or valid_range[1] < train_range[0]
        if not disjoint:
            raise ValueError('Training and validation generation seed ranges overlap.')
    fields = ('env_kwargs', 'expert_model', 'deterministic', 'selection')
    for key in fields:
        if key in training_data and key in validation_data and training_data[key] != validation_data[key]:
            raise ValueError(f'Incompatible training/validation generation provenance: {key} differs.')

    def missing(data):
        absent = [key for key in ('seed_start', 'attempts', *fields) if key not in data]
        for key, valid in (
            ('env_kwargs', isinstance(data.get('env_kwargs'), dict)),
            ('expert_model', isinstance(data.get('expert_model'), str) and bool(data.get('expert_model'))),
            ('deterministic', isinstance(data.get('deterministic'), bool)),
            ('selection', data.get('selection') in ('all', 'successful_filtered')),
        ):
            if not valid and key not in absent:
                absent.append(key)
        return sorted(absent)

    missing_train, missing_valid = missing(training_data), missing(validation_data)
    compatible = disjoint is True and not missing_train and not missing_valid
    return dict(status='compatible_disjoint_seed_ranges' if compatible else 'unverified',
                train_seed_range=train_range, validation_seed_range=valid_range,
                seed_ranges_disjoint=disjoint, missing_or_invalid_train_fields=missing_train,
                missing_or_invalid_validation_fields=missing_valid)


def load_training_validation(demo_path, validation_demo_path, validation_fraction, seed, device,
                             *, split_audit=None, terminal_semantics='recorded', horizon=None):
    """Load training/validation episodes, auditing files without deduplicating.

    Repeated observed trajectories are allowed only with complete, compatible,
    disjoint recorded generation provenance. Metadata is not independent proof
    of generation or layout independence. Legacy files retain strict checking.
    ``split_audit``, when supplied, receives JSON-serializable diagnostics.
    """
    terminal_audit = dict(training={}, validation={})
    if validation_demo_path:
        if Path(demo_path).samefile(validation_demo_path):
            raise ValueError('Training and validation must not use the same file.')
        with open(demo_path, 'rb') as stream:
            training_data = pickle.load(stream)
        with open(validation_demo_path, 'rb') as stream:
            validation_data = pickle.load(stream)
        provenance = _validation_provenance(training_data, validation_data)
        all_episodes = _episodes_from_data(training_data, device, terminal_semantics, horizon, terminal_audit['training'])
        validation = _episodes_from_data(validation_data, device, terminal_semantics, horizon, terminal_audit['validation'])
        if terminal_audit['training']['effective_semantics'] != terminal_audit['validation']['effective_semantics']:
            raise ValueError('Training and validation have different recorded terminal semantics; normalize both explicitly.')
        episodes = all_episodes
        train_indices = list(range(len(episodes)))
        validation_indices = list(range(len(validation)))
    else:
        all_episodes = load_episodes(demo_path, device, terminal_semantics=terminal_semantics,
                                     horizon=horizon, terminal_audit=terminal_audit['training'])
        train_indices, validation_indices = validation_split(all_episodes, validation_fraction, seed)
        episodes = [all_episodes[i] for i in train_indices]
        validation = [all_episodes[i] for i in validation_indices]
        provenance = dict(status='grouped_trajectory_holdout')
        terminal_audit['validation'] = dict(source='shared_file_before_split')
    if any(ep[0].shape[1:] != episodes[0][0].shape[1:] for ep in validation):
        raise ValueError('Validation observation shape does not match training data.')
    train_keys = [episode_fingerprint(ep) for ep in episodes]
    valid_keys = [episode_fingerprint(ep) for ep in validation]
    shared = set(train_keys) & set(valid_keys)
    report = dict(strategy='explicit_validation' if validation_demo_path else 'grouped_trajectory_holdout',
                  provenance=provenance, terminals=terminal_audit, train_episodes=len(episodes), validation_episodes=len(validation),
                  unique_train_trajectories=len(set(train_keys)), unique_validation_trajectories=len(set(valid_keys)),
                  shared_trajectory_groups=len(shared),
                  train_episodes_matching_validation=sum(key in shared for key in train_keys),
                  validation_episodes_matching_training=sum(key in shared for key in valid_keys),
                  repeated_trajectories_allowed=bool(shared), layout_disjointness='not_established')
    if shared:
        if provenance['status'] != 'compatible_disjoint_seed_ranges':
            raise ValueError('Training and validation contain identical trajectories. '
                             'Complete matching generation provenance with disjoint reset-seed ranges '
                             'is required to accept independently sampled repetitions.')
        warnings.warn(
            f'Training and validation share {len(shared)} observed trajectory groups '
            f'({report["validation_episodes_matching_training"]}/{len(validation)} validation episodes). '
            'Retaining all episodes because recorded generation provenance is compatible and seed ranges '
            'are disjoint. This does not establish disjoint layouts or independently verify provenance.',
            UserWarning, stacklevel=2)
    if split_audit is not None:
        split_audit.update(report)
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
            features = agent.encode_episode(states, next_states, actions)[:-1] if hasattr(agent, 'encoder') else states
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
