"""Frozen-policy belief probes, with layout-disjoint evaluation and memory controls.

Targets are external goal-location posteriors, not privileged actual-goal labels.
All probes receive known pose to make world-coordinate targets comparable.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from analysis.discrete_goal_belief import DiscreteGoalBelief
from analysis.policy_io import add_policy_args, load_policy, representation
from environment import make_partial_obs_learning_env


def layout_split(layouts, seed):
    """Assign entire repeated layouts, never individual timesteps, to one split."""
    unique, inverse = np.unique(layouts, axis=0, return_inverse=True)
    if len(unique) < 10:
        raise ValueError('Need at least 10 distinct layouts; increase --episodes.')
    order = np.random.default_rng(seed).permutation(len(unique))
    labels = np.full(len(unique), 2, dtype=np.int64)
    labels[order[:int(.6 * len(unique))]] = 0
    labels[order[int(.6 * len(unique)):int(.8 * len(unique))]] = 1
    return labels[inverse], inverse


@torch.no_grad()
def collect(args):
    env = make_partial_obs_learning_env(size=args.size, num_distractors=0,
                                       agent_view_size=args.view_size, max_steps=args.max_steps)
    try:
        agent = load_policy(args, env.observation_space.shape[0])
        if not hasattr(agent, 'encoder'):
            raise ValueError('Belief decoding requires a recurrent checkpoint.')
        for module in (agent.encoder, agent.q_net):
            module.requires_grad_(False)
            module.eval()
        cells = [(x, y) for x in range(1, args.size - 1)
                 for y in range(1, args.size - 1)]
        rows = {k: [] for k in ('hidden', 'observation', 'reset_hidden', 'pose',
                               'belief', 'episode', 'timestep', 'action', 'layout')}
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.rollout_seed + episode)
            base = env.unwrapped
            goal = next(p for p in cells if (obj := base.grid.get(*p)) is not None
                        and obj.type == 'goal')
            # Empty room geometry is fixed: goal, spawn and initial direction
            # fully identify a layout/initial condition. No goal input to probes.
            layout = (*goal, *base.agent_pos, base.agent_dir)
            belief = DiscreteGoalBelief.from_env(env)
            hidden, done, step = None, False, 0
            while not done:
                belief.observe(env)
                hidden = representation(agent, obs, hidden)
                reset = representation(agent, obs, None)
                action = int(agent.q_net(hidden).argmax(-1).item())
                pose = np.r_[np.asarray(base.agent_pos) / (args.size - 1),
                             np.eye(4)[base.agent_dir]]
                distribution = belief.distribution
                values = dict(hidden=hidden[0].numpy().copy(),
                              reset_hidden=reset[0].numpy().copy(),
                              observation=np.asarray(obs).copy(), pose=pose,
                              belief=[distribution.get(p, 0.) for p in cells],
                              episode=episode, timestep=step, action=action, layout=layout)
                for key, value in values.items():
                    rows[key].append(value)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step += 1
            if (episode + 1) % 50 == 0:
                print(f'Collected {episode + 1}/{args.episodes} episodes', flush=True)
        data = {k: np.asarray(v) for k, v in rows.items()}
        data['cells'] = np.asarray(cells)
        data['split'], data['layout_id'] = layout_split(data['layout'], args.split_seed)
        return data
    finally:
        env.close()


def distribution_scores(logp, target):
    """CE and KL for soft posterior labels, retaining zero-probability cells."""
    entropy = -(target * target.clamp_min(1e-30).log()).sum(-1)
    ce = -(target * logp).sum(-1)
    return ce, ce - entropy


def episode_weights(episodes):
    _, inverse, counts = np.unique(episodes, return_inverse=True, return_counts=True)
    return 1. / counts[inverse]


def fit_probe(features, data, args, validation_mask=None):
    train, valid = data['split'] == 0, data['split'] == 1
    mean = features[train].mean(0)
    scale = features[train].std(0).clip(min=1e-5)
    x = torch.tensor((features - mean) / scale, dtype=torch.float32)
    target = torch.tensor(data['belief'], dtype=torch.float32)
    torch.manual_seed(args.probe_seed)
    model = nn.Sequential(nn.Linear(x.shape[1], args.width), nn.ReLU(),
                          nn.Linear(args.width, target.shape[1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # Equal episode weight prevents long failures dominating probe training.
    weights = torch.tensor(episode_weights(data['episode'][train]), dtype=torch.float32)
    weights /= weights.mean()
    ti = torch.from_numpy(np.flatnonzero(train))
    eligible = data['belief'].max(-1) < 1 if validation_mask is None else validation_mask
    vi = torch.from_numpy(np.flatnonzero(valid & eligible))
    if not len(vi):
        raise ValueError('Validation split has no eligible steps; collect more episodes.')
    vw = torch.tensor(episode_weights(data['episode'][vi.numpy()]), dtype=torch.float32)
    best, state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in torch.randperm(len(ti)).split(args.batch_size):
            indices = ti[batch]
            ce, _ = distribution_scores(model(x[indices]).log_softmax(-1), target[indices])
            loss = (ce * weights[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            _, kl = distribution_scores(model(x[vi]).log_softmax(-1), target[vi])
            score = float((kl * vw).sum() / vw.sum())
        metric = 'validation_unresolved_kl' if validation_mask is None else 'validation_kl'
        history.append(dict(epoch=epoch, **{metric: score}))
        if score < best:
            best, state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
        if args.patience and epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logp = model(x).log_softmax(-1)
        ce, kl = distribution_scores(logp, target)
    return dict(state_dict=state, mean=torch.tensor(mean), scale=torch.tensor(scale),
                input_dim=x.shape[1], output_dim=target.shape[1], width=args.width,
                selected_epoch=best_epoch, history=history), ce.numpy(), kl.numpy(), logp.exp().numpy()


def load_cached_rollouts(args):
    """Reuse immutable features and splits only with matching policy/provenance."""
    path = Path(args.cached_rollouts)
    source = json.loads(path.with_name('summary.json').read_text())
    digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    if digest != source['checkpoint_sha256']:
        raise ValueError('Cached rollouts were collected from a different checkpoint.')
    for key in ('size', 'view_size', 'max_steps', 'episodes', 'rollout_seed', 'grid_shape',
                'init_temp', 'gamma'):
        if getattr(args, key) != source['config'][key]:
            raise ValueError(f'Cached rollout configuration mismatch: {key}')
    source_split_seed = source['config'].get('split_seed', source['config']['probe_seed'])
    if args.split_seed != source_split_seed:
        raise ValueError('Cached splits are fixed; --split-seed must match the source.')
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    for key in ('episode', 'layout_id'):
        groups = [set(data[key][data['split'] == i]) for i in range(3)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f'Cached data leak {key} between splits.')
    return data


def summarize_scores(values, episodes, mask):
    if not mask.any():
        return dict(steps=0, episodes=0, episode_mean=None, step_mean=None)
    weights = episode_weights(episodes[mask])
    return dict(steps=int(mask.sum()), episodes=int(len(np.unique(episodes[mask]))),
                episode_mean=float(np.average(values[mask], weights=weights)),
                step_mean=float(values[mask].mean()))


def paired_interval(delta, data, mask, seed, draws=2000):
    """Resample layouts; average episodes equally within each resample."""
    ids = np.unique(data['layout_id'][mask])
    if not len(ids):
        return dict(mean=None, ci95=None, layouts=0)
    sums, counts = [], []
    for layout in ids:
        m = mask & (data['layout_id'] == layout)
        means = [delta[m & (data['episode'] == e)].mean()
                 for e in np.unique(data['episode'][m])]
        sums.append(sum(means))
        counts.append(len(means))
    sums, counts = np.asarray(sums), np.asarray(counts)
    samples = np.random.default_rng(seed).integers(len(ids), size=(draws, len(ids)))
    boot = sums[samples].sum(1) / counts[samples].sum(1)
    return dict(mean=float(sums.sum() / counts.sum()),
                ci95=np.quantile(boot, [.025, .975]).tolist(), layouts=len(ids))


def run(args):
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f'Choose a new --output-dir; already exists: {out}')
    torch.set_num_threads(args.threads)
    data = load_cached_rollouts(args) if args.cached_rollouts else collect(args)
    out.mkdir(parents=True)
    np.savez_compressed(out / 'rollouts.npz', **data)
    unresolved = data['belief'].max(-1) < 1
    test = data['split'] == 2
    if not (test & unresolved).any():
        raise ValueError('Test split has no unresolved steps; collect more episodes.')
    results, predictions, scores = {}, {}, {}
    for name in ('observation', 'reset_hidden', 'hidden'):
        print(f'Fitting {name} + pose probe', flush=True)
        features = np.concatenate([data[name], data['pose']], axis=1).astype(np.float32)
        fitted, ce, kl, prediction = fit_probe(features, data, args)
        torch.save({k: v for k, v in fitted.items() if k != 'history'}, out / f'{name}_probe.pt')
        (out / f'{name}_training.json').write_text(json.dumps(fitted['history'], indent=2))
        predictions[name] = prediction[test]
        scores[name] = kl
        results[name] = dict(selected_epoch=fitted['selected_epoch'],
                            epochs_run=len(fitted['history']),
                            selected_at_budget=fitted['selected_epoch'] == args.epochs,
                            validation_kl=fitted['history'][fitted['selected_epoch'] - 1]['validation_unresolved_kl'],
                            parameter_count=sum(v.numel() for v in fitted['state_dict'].values()))
        for phase, phase_mask in [('all', test), ('unresolved', test & unresolved),
                                  ('resolved', test & ~unresolved)]:
            results[name][phase] = dict(
                kl=summarize_scores(kl, data['episode'], phase_mask),
                cross_entropy=summarize_scores(ce, data['episode'], phase_mask))
    comparisons = {name: paired_interval(scores[name] - scores['hidden'], data,
                                         test & unresolved, args.probe_seed)
                   for name in ('observation', 'reset_hidden')}
    summary = dict(config=vars(args),
                   rollout_sha256=hashlib.sha256((out / 'rollouts.npz').read_bytes()).hexdigest(),
                   checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                   target='Uniform goal-location posterior over interior world cells',
                   primary_metric='test unresolved KL(b || decoded_b), equal episode weighting',
                   split_unit='goal cell + initial position + initial direction; 60/20/20 by layout',
                   pose_control='All probes receive normalized x,y and one-hot direction',
                   interpretation='Decodability, not causal use. Layouts are held out from probe fitting, not necessarily IQ training. One probe initialization; intervals exclude training variability.',
                   splits={name: dict(episodes=len(np.unique(data['episode'][data['split'] == i])),
                                      layouts=len(np.unique(data['layout_id'][data['split'] == i])),
                                      steps=int((data['split'] == i).sum()))
                           for i, name in enumerate(('train', 'validation', 'test'))},
                   probes=results, baseline_minus_hidden_kl=comparisons)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    np.savez_compressed(out / 'test_predictions.npz', target=data['belief'][test],
                        episode=data['episode'][test], timestep=data['timestep'][test],
                        cells=data['cells'], **predictions)
    with (out / 'test_episode_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'reset_seed', 'layout_id', 'phase',
                                              'steps', 'observation', 'reset_hidden', 'hidden'])
        writer.writeheader()
        for episode in np.unique(data['episode'][test]):
            for phase, phase_mask in [('unresolved', unresolved), ('resolved', ~unresolved)]:
                mask = test & (data['episode'] == episode) & phase_mask
                if mask.any():
                    writer.writerow(dict(episode=int(episode), reset_seed=args.rollout_seed + int(episode),
                                         layout_id=int(data['layout_id'][mask][0]), phase=phase,
                                         steps=int(mask.sum()),
                                         **{k: float(v[mask].mean()) for k, v in scores.items()}))
    plot_results(summary, out)
    print(json.dumps(dict(output=str(out), baseline_minus_hidden_kl=comparisons), indent=2))


def plot_results(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['observation', 'reset_hidden', 'hidden']
    labels = ['Observation', 'Reset GRU', 'History GRU']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for phase, offset in [('unresolved', -.18), ('resolved', .18)]:
        values = [summary['probes'][n][phase]['kl']['episode_mean'] for n in names]
        axes[0].bar(np.arange(3) + offset, [np.nan if v is None else v for v in values],
                    width=.36, label=phase)
    axes[0].set(xticks=range(3), xticklabels=labels, ylabel='Held-out KL (nats); lower is better',
                title='Full posterior decoding; all probes receive pose')
    axes[0].legend()
    for i, name in enumerate(names[:2]):
        r = summary['baseline_minus_hidden_kl'][name]
        axes[1].plot([r['ci95'][0], r['ci95'][1]], [i, i], color='C0')
        axes[1].plot(r['mean'], i, 'o', color='C0')
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].set(yticks=[0, 1], yticklabels=labels[:2],
                xlabel='Baseline KL − history GRU KL (nats)',
                title='Unresolved steps: positive favors memory')
    fig.suptitle('Frozen-policy belief probes: episode means, layout-bootstrap 95% intervals')
    fig.savefig(out / 'belief_decoding.png', dpi=180)
    fig.savefig(out / 'belief_decoding.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--view-size', type=int, default=3)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--rollout-seed', type=int, default=200000)
    parser.add_argument('--probe-seed', type=int, default=0)
    parser.add_argument('--split-seed', type=int, default=0,
                        help='Independent of decoder initialization; cached splits stay fixed.')
    parser.add_argument('--cached-rollouts', help='Existing rollouts.npz with adjacent summary.json.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=0,
                        help='Stop after this many epochs without validation improvement; 0 disables.')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    for key in ('episodes', 'epochs', 'batch_size', 'width', 'threads'):
        if getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    if args.patience < 0:
        parser.error('--patience must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
