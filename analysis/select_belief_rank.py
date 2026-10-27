"""Select posterior-predictive rank on validation reconstruction, never action effects."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.evaluate_belief_decoding import load_cached_rollouts, episode_weights
from analysis.transfer_belief_subspace import fit_subspace


def select_rank(data, ranks, ridge):
    ranks = sorted(set(ranks))
    if not ranks or ranks[0] != 0 or ranks[-1] < 1:
        raise ValueError('Include rank-zero nuisance baseline and at least one positive rank.')
    fitted = fit_subspace(data, max(ranks), ridge)
    valid = (data['split'] == 1) & (data['belief'].max(-1) < 1)
    if not valid.any():
        raise ValueError('No unresolved validation records.')
    nuisance = np.c_[np.ones(valid.sum()), data['pose'][valid], data['timestep'][valid] / 400.]
    x = (data['hidden'][valid] - fitted['mean']) / fitted['scale']
    x -= np.einsum('ij,jk->ik', nuisance, fitted['nuisance_x'])
    baseline = np.einsum('ij,jk->ik', nuisance, fitted['nuisance_y'])
    weights = episode_weights(data['episode'][valid])
    curve = []
    for rank in ranks:
        basis = fitted['basis'][:, :rank]
        coordinates = np.einsum('ni,ij->nj', x, basis)
        projected = np.einsum('nj,kj->nk', coordinates, basis)
        prediction = baseline + np.einsum('ni,ij->nj', projected, fitted['coefficients'])
        error = np.mean((prediction - data['belief'][valid])**2, axis=1)
        curve.append(dict(rank=rank, validation_mse=float(np.average(error, weights=weights))))
    selected = min(curve, key=lambda r: (r['validation_mse'], r['rank']))['rank']
    return dict(selected_rank=selected, curve=curve,
                validation_episodes=int(len(np.unique(data['episode'][valid]))),
                validation_steps=int(valid.sum())), fitted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollouts', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--ranks', type=int, nargs='+', default=[0, 2, 4, 8, 16, 32, 63])
    p.add_argument('--ridge', type=float, default=.01)
    args = p.parse_args()
    out, source = Path(args.output_dir), Path(args.rollouts)
    if out.exists():
        raise FileExistsError('Choose a new output directory.')
    if not np.isfinite(args.ridge) or args.ridge <= 0 or min(args.ranks) < 0:
        p.error('Need positive ridge and nonnegative ranks.')
    metadata = json.loads(source.with_name('summary.json').read_text())
    config = SimpleNamespace(**metadata['config'])
    config.cached_rollouts = str(source)
    config.split_seed = metadata['config'].get('split_seed', metadata['config']['probe_seed'])
    data = load_cached_rollouts(config)
    result, fitted = select_rank(data, args.ranks, args.ridge)
    result.update(config=vars(args), checkpoint_sha256=metadata['checkpoint_sha256'],
                  rollout_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  rule='Minimum episode-weighted unresolved validation posterior MSE; ties favor smaller rank. Zero rank means no transfer warranted. No action labels or held-out test targets used.',
                  limitation='Unconstrained linear posterior scores; selecting a higher rank does not establish selective belief representation or causal use.')
    out.mkdir(parents=True)
    (out / 'selection.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    np.savez_compressed(out / 'fit.npz', **fitted)
    plot(result, out)
    print(json.dumps(result, indent=2))


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve = result['curve']
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.plot([r['rank'] for r in curve], [r['validation_mse'] for r in curve], 'o-')
    ax.axhline(curve[0]['validation_mse'], linestyle='--', color='gray', label='Pose/time baseline (rank 0)')
    ax.set(xlabel='Posterior-predictive subspace rank', ylabel='Validation posterior MSE (lower is better)',
           title=f'Validation-selected rank: {result["selected_rank"]}')
    ax.legend()
    fig.savefig(out / 'rank_selection.png', dpi=180)
    fig.savefig(out / 'rank_selection.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
