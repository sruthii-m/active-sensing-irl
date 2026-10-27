"""Exploratory donor-memory transfers in a posterior-predictive subspace.

Fixed-rank ridge directions are learned only on probe training layouts, with
linear pose/time effects removed. No action outputs are used to find directions.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from analysis.evaluate_belief_decoding import episode_weights, load_cached_rollouts
from analysis.evaluate_matched_histories import policy_probabilities
from analysis.policy_io import load_policy
from analysis.validate_matched_histories import reconstructed_pair
from environment import make_partial_obs_learning_env


def fit_subspace(data, rank=8, ridge=.01):
    train = data['split'] == 0
    h = data['hidden'][train].astype(float)
    target = data['belief'][train].astype(float)
    weights = episode_weights(data['episode'][train])
    weights /= weights.sum()
    mean = np.average(h, axis=0, weights=weights)
    scale = np.sqrt(np.average((h - mean)**2, axis=0, weights=weights)).clip(1e-5)
    x = (h - mean) / scale
    nuisance = np.c_[np.ones(len(h)), data['pose'][train], data['timestep'][train] / 400.]
    root = np.sqrt(weights[:, None])
    x_coef = np.linalg.lstsq(nuisance * root, x * root, rcond=None)[0]
    y_coef = np.linalg.lstsq(nuisance * root, target * root, rcond=None)[0]
    # Explicit contractions avoid platform BLAS floating-point flag artifacts.
    x = (x - np.einsum('ij,jk->ik', nuisance, x_coef)) * root
    y = (target - np.einsum('ij,jk->ik', nuisance, y_coef)) * root
    gram = np.einsum('ni,nj->ij', x, x)
    cross = np.einsum('ni,nj->ij', x, y)
    coefficients = np.linalg.solve(gram + ridge * np.eye(x.shape[1]), cross)
    if not np.isfinite(coefficients).all():
        raise ValueError('Nonfinite posterior regression coefficients.')
    u, singular, _ = np.linalg.svd(coefficients, full_matrices=False)
    if rank < 1 or rank > min(coefficients.shape) or singular[rank-1] < 1e-10:
        raise ValueError('Requested rank exceeds available posterior-predictive directions.')
    return dict(mean=mean, scale=scale, basis=u[:, :rank], singular=singular,
                coefficients=coefficients, nuisance_x=x_coef, nuisance_y=y_coef)


def donor_delta(source, donor, basis):
    coordinates = np.einsum('i,ij->j', donor - source, basis)
    return np.einsum('j,ij->i', coordinates, basis)


def cluster_summary(rows, key):
    seeds = sorted({r['reset_seed'] for r in rows})
    if not seeds:
        return dict(mean=None, ci95=None, reset_seeds=0, pairs=0)
    values = np.array([np.mean([r[key] for r in rows if r['reset_seed'] == s]) for s in seeds])
    draws = np.random.default_rng(0).integers(len(seeds), size=(2000, len(seeds)))
    return dict(mean=float(values.mean()), ci95=np.quantile(values[draws].mean(1), [.025,.975]).tolist(),
                reset_seeds=len(seeds), pairs=len({r['pair_id'] for r in rows}))


def excluded_seeds(config, seeds, data):
    """Exclude entire reset layouts used anywhere in the original probe dataset."""
    old = {tuple(row) for row in data['layout']}
    env = make_partial_obs_learning_env(size=config.size, num_distractors=0,
                                       agent_view_size=config.view_size, max_steps=config.max_steps)
    excluded = set()
    try:
        for seed in seeds:
            env.reset(seed=seed)
            base = env.unwrapped
            goal = next((x, y) for x in range(1, config.size - 1) for y in range(1, config.size - 1)
                        if (obj := base.grid.get(x, y)) is not None and obj.type == 'goal')
            if (*goal, *base.agent_pos, base.agent_dir) in old:
                excluded.add(seed)
    finally:
        env.close()
    return excluded


@torch.no_grad()
def run(args):
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    torch.set_num_threads(1)
    source = Path(args.comparison)
    summary = json.loads((source / 'summary.json').read_text())
    rows = json.loads((source / 'pairs.json').read_text())
    config = SimpleNamespace(**summary['source_config'])
    histories = Path(summary['config']['histories'])
    for name, digest in summary['source_hashes'].items():
        if hashlib.sha256((histories / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Matched source changed: {name}')
    probe_source = Path(args.rollouts)
    provenance = json.loads(probe_source.with_name('summary.json').read_text())
    if provenance['checkpoint_sha256'] != summary['checkpoint_sha256']:
        raise ValueError('Probe and matched histories use different policies.')
    cache_args = SimpleNamespace(**provenance['config'])
    cache_args.cached_rollouts = str(probe_source)
    cache_args.split_seed = provenance['config'].get('split_seed', provenance['config']['probe_seed'])
    data = load_cached_rollouts(cache_args)
    selection = None
    if getattr(args, 'selection', None):
        selection = json.loads(Path(args.selection).read_text())
        if (selection['rollout_sha256'] != hashlib.sha256(probe_source.read_bytes()).hexdigest()
                or selection['checkpoint_sha256'] != summary['checkpoint_sha256']
                or selection['config']['ridge'] != args.ridge):
            raise ValueError('Rank selection provenance or ridge does not match this run.')
        if selection['selected_rank'] == 0:
            raise ValueError('Validation selected no memory contribution; no transfer is warranted.')
        args.rank = selection['selected_rank']
    fitted = fit_subspace(data, args.rank, args.ridge)
    agent = load_policy(config, data['observation'].shape[1])
    novel_exclusions = excluded_seeds(config, {r['reset_seed'] for r in rows}, data)
    rng = np.random.default_rng(args.seed)
    random_bases = [np.linalg.qr(rng.normal(size=fitted['basis'].shape))[0] for _ in range(args.controls)]
    results, audit = [], {}
    with np.load(histories / 'matched_traces.npz', allow_pickle=False) as traces:
        for row in rows:
            if row['reset_seed'] in novel_exclusions:
                continue
            metrics = row['iq']
            primary = row['kind'] == 'loop' and metrics['different_unique_ig']
            same = (row['kind'] == 'rotation' and metrics['unresolved'] and metrics['same_belief']
                    and not metrics['identical_observation_history'])
            if not (primary or same):
                continue
            pair = reconstructed_pair(traces, row['pair_id'])
            h = np.array([traces[f'pair{row["pair_id"]}_{side}_hidden'] for side in ('a', 'b')])
            standardized = (h - fitted['mean']) / fitted['scale']
            for i, j in [(0, 1), (1, 0)]:
                base, donor = standardized[i], standardized[j]
                delta = donor_delta(base, donor, fitted['basis'])
                controls = np.array([donor_delta(base, donor, basis) for basis in random_bases])
                norms = np.linalg.norm(controls, axis=1)
                if np.linalg.norm(delta) > 1e-12 and np.any(norms < 1e-12):
                    raise ValueError('Cannot norm-match a degenerate random transfer.')
                controls *= np.linalg.norm(delta) / np.maximum(norms[:, None], 1e-12)
                edits = np.r_[delta[None], controls]
                edited = (base[None] + edits) * fitted['scale'] + fitted['mean']
                q = agent.q_net(torch.tensor(np.r_[h[i:i+1], edited], dtype=torch.float32)).numpy()
                if not np.allclose(q[0], pair[i]['q'], atol=1e-6):
                    raise ValueError('Saved hidden state no longer reproduces checkpoint Q values.')
                p = np.array([policy_probabilities(value, config.init_temp) for value in q])
                tv = .5 * np.abs(p[1:] - p[0]).sum(-1)
                item = dict(pair_id=row['pair_id'], reset_seed=row['reset_seed'], direction=i,
                            group='different_unique_ig' if primary else 'same_belief_control',
                            transfer_rms=float(np.sqrt(np.mean(delta**2))),
                            policy_tv=float(tv[0]), random_policy_tv=float(tv[1:].mean()),
                            excess_policy_tv=float(tv[0] - tv[1:].mean()))
                pose = pair[i]['positions'][-1]
                nuisance = np.r_[1., pose[:2] / (config.size - 1), np.eye(4)[int(pose[2])],
                                  (len(pair[i]['observations']) - 1) / 400.]
                residual_base = base - np.einsum('i,ij->j', nuisance, fitted['nuisance_x'])
                decoded = np.einsum('i,ij->j', residual_base, fitted['coefficients']) + np.einsum('i,ij->j', nuisance, fitted['nuisance_y'])
                edited_decode = decoded + np.einsum('i,ij->j', delta, fitted['coefficients'])
                target = np.array([1 / len(pair[j]['candidates']) if tuple(cell) in pair[j]['candidates'] else 0 for cell in data['cells']])
                item['donor_posterior_mse_reduction'] = float(np.mean((decoded-target)**2) - np.mean((edited_decode-target)**2))
                if primary:
                    source_action, target_action = int(pair[i]['ig'].argmax()), int(pair[j]['ig'].argmax())
                    margin = p[:, target_action] - p[:, source_action]
                    effect = margin[1:] - margin[0]
                    reference = np.asarray(row['reference_probabilities'])
                    item.update(directional_shift=float(effect[0]), random_directional_shift=float(effect[1:].mean()),
                                excess_directional_shift=float(effect[0] - effect[1:].mean()),
                                explicit_full_belief_transfer=float((reference[j,target_action]-reference[j,source_action]) -
                                                                   (reference[i,target_action]-reference[i,source_action])))
                results.append(item)
                audit[f'pair{row["pair_id"]}_direction{i}_edited_hidden'] = edited
                audit[f'pair{row["pair_id"]}_direction{i}_probabilities'] = p
    metrics = {}
    for group in ('different_unique_ig', 'same_belief_control'):
        subset = [r for r in results if r['group'] == group]
        keys = ['policy_tv', 'random_policy_tv', 'excess_policy_tv', 'transfer_rms', 'donor_posterior_mse_reduction']
        if group == 'different_unique_ig':
            keys += ['directional_shift', 'random_directional_shift', 'excess_directional_shift', 'explicit_full_belief_transfer']
        metrics[group] = {key: cluster_summary(subset, key) for key in keys}
    out.mkdir(parents=True)
    np.savez_compressed(out / 'subspace.npz', **fitted, random_bases=random_bases)
    np.savez_compressed(out / 'transfers.npz', **audit)
    (out / 'directions.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    report = dict(config=vars(args), checkpoint_sha256=summary['checkpoint_sha256'],
                  rollout_sha256=hashlib.sha256(probe_source.read_bytes()).hexdigest(),
                  comparison_sha256=hashlib.sha256((source / 'pairs.json').read_bytes()).hexdigest(),
                  manipulation_check='Reduction in raw linear decoder mean squared error to donor posterior; outputs are unconstrained scores, not normalized probabilities.',
                  excluded_reset_seeds=sorted(novel_exclusions), metrics=metrics, rank_selection=selection,
                  interpretation='Exploratory donor transfer, not proof of a causal Bayesian representation. No policy outputs used to learn directions or select rank.',
                  controls='20 by default: fixed random orthonormal subspaces of identical rank; each projected donor transfer rescaled to the belief-transfer norm. Average controls before reset-cluster bootstrap.',
                  limitations=['Ridge fixed; rank fixed or selected by validation MSE. Directions use only original training split.',
                               'Exploratory assay; independent fresh histories are needed for each newly selected procedure.',
                               'Linear nuisance residualization does not guarantee preservation of latent pose or other features.',
                               'Partial transfers may be off-manifold; no navigation rescue or necessity test.',
                               'Explicit control transfers the entire known posterior; does not validate learned subspace identification.',
                               'IG target tests information seeking, not all forms of belief-dependent behavior.'])
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    plot(report, out)
    print(json.dumps(metrics, indent=2))


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    group = report['metrics']['different_unique_ig']
    for ax, keys, labels in [
        (axes[0], ['directional_shift', 'random_directional_shift'], ['Belief subspace', 'Random control']),
        (axes[1], ['excess_directional_shift'], ['Effect beyond random control'])]:
        for i, (key, label) in enumerate(zip(keys, labels)):
            r = group[key]
            if r['mean'] is not None:
                ax.plot(i, r['mean'], 'o')
                ax.plot([i, i], r['ci95'])
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylabel('Change in action-probability margin')
    fig.suptitle('Partial donor-memory transfer; positive is the predicted effect')
    fig.savefig(out / 'belief_transfer.png', dpi=180)
    fig.savefig(out / 'belief_transfer.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', required=True)
    p.add_argument('--rollouts', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--selection', help='Frozen selection.json overrides --rank after provenance checks.')
    p.add_argument('--ridge', type=float, default=.01)
    p.add_argument('--controls', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.rank < 1 or args.controls < 1 or not np.isfinite(args.ridge) or args.ridge <= 0:
        p.error('Rank, control count and ridge must be positive.')
    run(args)


if __name__ == '__main__':
    main()
