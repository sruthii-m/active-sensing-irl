"""Known-mechanism calibration of posterior-predictive subspace transfers.

Synthetic frozen networks mix known posterior coordinates and independent noise.
This is an identification benchmark, not a trained IQ agent or navigation task.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.evaluate_belief_decoding import episode_weights, load_cached_rollouts
from analysis.transfer_belief_subspace import fit_subspace, donor_delta


def multiply(a, b):
    return np.einsum('...i,ij->...j', a, b)


def probabilities(hidden, rotation, weights, start=0):
    latent = multiply(hidden, rotation)
    logits = multiply(latent[..., start:start + weights.shape[0]], weights)
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(-1, keepdims=True)


def make_network(data, belief_dim, hidden_dim, seed):
    train = data['split'] == 0
    target = data['belief'].astype(float)
    weights = episode_weights(data['episode'][train])
    weights /= weights.sum()
    mean = np.average(target[train], axis=0, weights=weights)
    _, singular, vt = np.linalg.svd((target[train] - mean) * np.sqrt(weights[:, None]), full_matrices=False)
    if belief_dim > len(singular) or singular[belief_dim-1] < 1e-8:
        raise ValueError('Requested belief dimension exceeds training posterior rank.')
    if hidden_dim < belief_dim + 8:
        raise ValueError('Need at least eight independent nuisance coordinates.')
    components = vt[:belief_dim].T
    scales = singular[:belief_dim]
    coordinates = multiply(target - mean, components) / scales
    rng = np.random.default_rng(seed)
    rotation = np.linalg.qr(rng.normal(size=(hidden_dim, hidden_dim)))[0]
    latent = rng.normal(size=(len(target), hidden_dim))
    latent[:, :belief_dim] = coordinates
    hidden = multiply(latent, rotation.T)
    action_weights = rng.normal(size=(belief_dim, 7)) / np.sqrt(belief_dim)
    noise_weights = rng.normal(size=(8, 7)) / np.sqrt(8)
    return dict(hidden=hidden, rotation=rotation, components=components, scales=scales,
                posterior_mean=mean, action_weights=action_weights, noise_weights=noise_weights)


def oracle_transfer(source, donor, rotation, belief_dim):
    """Replace exactly the true belief block; leave nuisance coordinates intact."""
    delta = multiply(donor - source, rotation[:, :belief_dim])
    return source + multiply(delta, rotation[:, :belief_dim].T)


def heldout_pairs(data):
    """One unresolved source per test episode; circular donors are other episodes."""
    rng = np.random.default_rng(0)
    mask = (data['split'] == 2) & (data['belief'].max(-1) < 1)
    indices = [int(rng.choice(np.flatnonzero(mask & (data['episode'] == episode))))
               for episode in np.unique(data['episode'][mask])]
    if len(indices) < 2:
        raise ValueError('Need at least two unresolved held-out episodes.')
    indices = np.asarray(indices)
    return indices, np.roll(indices, 1)


def evaluate_method(network, fitted, basis, a, b, belief_dim):
    h = network['hidden']
    source, donor = h[a], h[b]
    x, y = (source - fitted['mean']) / fitted['scale'], (donor - fitted['mean']) / fitted['scale']
    delta = np.array([donor_delta(u, v, basis) for u, v in zip(x, y)])
    edited = source + delta * fitted['scale']
    return evaluate_hidden(network, source, donor, edited, belief_dim), edited, delta


def evaluate_hidden(network, source, donor, edited, belief_dim):
    r, w = network['rotation'], network['action_weights']
    before = probabilities(source, r, w)
    desired = probabilities(donor, r, w)
    after = probabilities(edited, r, w)
    before_tv = .5 * np.abs(before - desired).sum(-1)
    after_tv = .5 * np.abs(after - desired).sum(-1)
    before_noise = probabilities(source, r, network['noise_weights'], belief_dim)
    after_noise = probabilities(edited, r, network['noise_weights'], belief_dim)
    latent_error = multiply(edited - donor, r[:, :belief_dim])
    return dict(policy_tv_to_donor=float(after_tv.mean()),
                unedited_tv_to_donor=float(before_tv.mean()),
                tv_reduction=float((before_tv - after_tv).mean()),
                fraction_oracle_tv_reduction=float((before_tv - after_tv).mean() / before_tv.mean()) if before_tv.mean() > 1e-12 else None,
                belief_coordinate_mse=float(np.mean(latent_error**2)),
                nuisance_only_policy_tv=float((.5 * np.abs(after_noise - before_noise).sum(-1)).mean()))


def run(args):
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    source = Path(args.rollouts)
    metadata = json.loads(source.with_name('summary.json').read_text())
    cache_args = SimpleNamespace(**metadata['config'])
    cache_args.cached_rollouts = str(source)
    cache_args.split_seed = metadata['config'].get('split_seed', metadata['config']['probe_seed'])
    data = load_cached_rollouts(cache_args)
    a, b = heldout_pairs(data)
    out.mkdir(parents=True)
    full_rank = data['belief'].shape[1] - 1
    cases = [('compressed_rank8', 8, 8), ('full_rank8', full_rank, 8),
             ('full_matched_rank', full_rank, full_rank)]
    runs = []
    for case, belief_dim, rank in cases:
        for seed in args.seeds:
            print(f'{case}, seed {seed}', flush=True)
            network = make_network(data, belief_dim, args.hidden_dim, seed)
            synthetic = {**data, 'hidden': network['hidden']}
            fitted = fit_subspace(synthetic, rank, args.ridge)
            true_basis = np.linalg.qr(network['rotation'][:, :belief_dim] / fitted['scale'][:, None])[0]
            overlap = np.sum(multiply(fitted['basis'].T, true_basis)**2)
            record, edited, delta = evaluate_method(network, fitted, fitted['basis'], a, b, belief_dim)
            shuffled = data['belief'].copy()
            train = np.flatnonzero(data['split'] == 0)
            shuffled[train] = shuffled[np.random.default_rng(seed + 1000).permutation(train)]
            shuffled_fit = fit_subspace({**synthetic, 'belief': shuffled}, rank, args.ridge)
            placebo, placebo_hidden, _ = evaluate_method(network, shuffled_fit, shuffled_fit['basis'], a, b, belief_dim)
            rng = np.random.default_rng(seed + 2000)
            random_results = []
            source_h, donor_h = network['hidden'][a], network['hidden'][b]
            x, y = (source_h-fitted['mean'])/fitted['scale'], (donor_h-fitted['mean'])/fitted['scale']
            for _ in range(args.controls):
                basis = np.linalg.qr(rng.normal(size=fitted['basis'].shape))[0]
                random_delta = np.array([donor_delta(u, v, basis) for u, v in zip(x, y)])
                random_delta *= (np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(random_delta, axis=1), 1e-12))[:, None]
                random_results.append(evaluate_hidden(network, source_h, donor_h,
                                      source_h + random_delta * fitted['scale'], belief_dim))
            random_mean = {k: float(np.mean([r[k] for r in random_results])) for k in record}
            oracle = oracle_transfer(source_h, donor_h, network['rotation'], belief_dim)
            exact = evaluate_hidden(network, source_h, donor_h, oracle, belief_dim)
            result = dict(case=case, seed=seed, belief_dim=belief_dim, fitted_rank=rank,
                          overlap_per_fitted_dimension=float(overlap / rank),
                          fraction_true_subspace_recovered=float(overlap / belief_dim),
                          recovered=record, shuffled=placebo, random=random_mean, oracle=exact)
            runs.append(result)
            folder = out / f'{case}_seed{seed}'
            folder.mkdir()
            np.savez_compressed(folder / 'network.npz', **{k:v for k,v in network.items() if k != 'hidden'},
                                **{f'fit_{k}':v for k,v in fitted.items()}, shuffled_basis=shuffled_fit['basis'])
            np.savez_compressed(folder / 'transfers.npz', source_indices=a, donor_indices=b,
                                source=source_h, donor=donor_h, edited=edited, shuffled=placebo_hidden, oracle=oracle)
            (folder / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    report = dict(config=vars(args), rollout_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  checkpoint_sha256=metadata['checkpoint_sha256'], test_episodes=len(a),
                  design='Synthetic frozen linear mixing and softmax actors; posterior PCA, normalization and subspace fitting use training data only. Same held-out episodes across seeds; no independent-replicate confidence intervals.',
                  interpretation='Calibrates recovery for known linear belief encodings. Does not establish belief use in IQ, or validate recovery of nonlinear/compressed representations.',
                  cases='Eight retained posterior coordinates versus the full 63-dimensional simplex; rank-8 and rank-matched fits. PCA defines the planted mechanism, not a learned IQ representation.',
                  controls='Exact belief-block oracle, training-label shuffle, and 20 norm/rank-matched random transfers. A nuisance-only actor must remain unchanged under an ideal belief transfer.',
                  limitations=['Donor histories need not match physical pose; this is a synthetic internal-variable intervention, not navigation behavior.',
                               'Gaussian nuisance is simpler than learned task-correlated memory.',
                               'PCA coordinates and linear mixing create a deliberately favorable identification benchmark.',
                               'Previously used test layouts; exploratory calibration, not a fresh IQ confirmation.'], runs=runs)
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    plot(report, out)
    for case, _, _ in cases:
        selected = [r for r in runs if r['case'] == case]
        print(case, {name: round(float(np.mean([r[name]['fraction_oracle_tv_reduction'] for r in selected])), 4)
                     for name in ('recovered', 'shuffled', 'random', 'oracle')})


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = list(dict.fromkeys(r['case'] for r in report['runs']))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for i, case in enumerate(cases):
        selected = [r for r in report['runs'] if r['case'] == case]
        for j, name in enumerate(('recovered', 'shuffled', 'random', 'oracle')):
            for ax, metric in zip(axes, ('fraction_oracle_tv_reduction', 'nuisance_only_policy_tv')):
                ax.scatter([i + (j-1.5)*.16] * len(selected), [r[name][metric] for r in selected],
                           color=f'C{j}', label=name if i == 0 else None, alpha=.7)
    for ax in axes:
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(range(len(cases)), ['8 belief / rank 8', '63 belief / rank 8', '63 belief / rank 63'])
    axes[0].set_ylabel('Fraction of oracle policy change recovered')
    axes[1].set_ylabel('Unwanted change in nuisance-only policy (TV)')
    axes[0].legend()
    fig.suptitle('Known linear mechanism: each dot is a mixing seed, not an independent dataset')
    fig.savefig(out / 'recovery.png', dpi=180)
    fig.savefig(out / 'recovery.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollouts', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--ridge', type=float, default=.01)
    p.add_argument('--controls', type=int, default=20)
    args = p.parse_args()
    if args.controls < 1 or args.hidden_dim < 16 or args.ridge <= 0 or not np.isfinite(args.ridge):
        p.error('Need positive controls/ridge and sufficient hidden dimension.')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error('Seeds must be unique and nonnegative.')
    run(args)


if __name__ == '__main__':
    main()
