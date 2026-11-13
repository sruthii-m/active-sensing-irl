"""Bootstrap masks for reset-started trajectories of a finite-horizon task.

The task deadline is terminal. An earlier data-collection cutoff is not.
Legacy files are normalized in memory only, with the correction count reported.
"""
import numpy as np


def task_terminal(terminated, truncated, timestep, horizon):
    return bool(terminated or (truncated and timestep >= horizon))


def terminal_masks(data, semantics='recorded', horizon=None, audit=None):
    if semantics not in ('recorded', 'intrinsic_horizon'):
        raise ValueError('Unknown terminal semantics.')
    effective = 'intrinsic_horizon' if semantics == 'intrinsic_horizon' else data.get('terminal_semantics', 'recorded')
    if effective not in ('recorded', 'intrinsic_horizon'):
        raise ValueError('Unknown demonstration terminal semantics.')
    recorded = [np.asarray(d, dtype=bool).copy() for d in data['dones']]
    masks = [d.copy() for d in recorded]
    if semantics == 'intrinsic_horizon':
        env = data.get('env_kwargs', {})
        declared = data.get('task_horizon') or env.get('max_steps')
        if declared is None and 'size' in env:
            declared = 4 * env['size'] ** 2
        if horizon is not None and declared is not None and horizon != declared:
            raise ValueError('Requested task horizon disagrees with demonstration metadata.')
        horizon = declared if declared is not None else horizon
        if horizon is None or horizon < 1 or int(horizon) != horizon:
            raise ValueError('Intrinsic terminal semantics require a known positive task horizon.')
        if data.get('starts_at_reset', True) is not True:
            raise ValueError('Cannot infer deadline terminals for trajectories that do not start at reset.')
        for i, mask in enumerate(masks):
            if mask.ndim != 1 or len(mask) == 0 or len(mask) > horizon:
                raise ValueError('Invalid episode length or terminal-mask shape for the task horizon.')
            if 'terminated' in data:
                mask[:] = np.asarray(data['terminated'][i], dtype=bool)
            if len(mask) == horizon:
                mask[-1] = True
    if audit is not None:
        audit.update(semantics=semantics, effective_semantics=effective, task_horizon=horizon,
                     changed_transitions=sum(int(np.count_nonzero(a != b)) for a, b in zip(masks, recorded)),
                     reset_start='recorded' if data.get('starts_at_reset') else 'legacy_complete_episode_contract')
    return masks
