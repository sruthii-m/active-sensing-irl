"""Positive controls and repeated, parameter-budget-matched consistency probes.

The explicit reference actor uses Bayesian expected information gain. It is
queried on cached IQ histories, not trained or rolled out as a new expert.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from analysis.belief_reward_planner import BeliefRewardPlanner
from analysis.evaluate_belief_consistency import history_features, random_memory, update_errors
from analysis.evaluate_belief_decoding import fit_probe, load_cached_rollouts, paired_interval, summarize_scores


def budget_width(input_dim, output_dim, budget):
    """Nearest width for Linear-ReLU-Linear, including both biases."""
    if budget < input_dim + 2 * output_dim + 1:
        raise ValueError('Parameter budget cannot fit even one hidden unit.')
    return max(1, round((budget - output_dim) / (input_dim + output_dim + 1)))


def shuffled_beliefs(data, seed):
    """Marginal-distribution placebo, shuffled separately within each split."""
    rng = np.random.default_rng(seed)
    result = data['belief'].copy()
    for split in range(3):
        indices = np.flatnonzero(data['split'] == split)
        result[indices] = data['belief'][rng.permutation(indices)]
    return result


def information_policy(beliefs, poses, cells, planner, temperature=.25):
    """Seven-action softmax of expected entropy reduction; no IQ outputs.

Found-goal outcomes have zero entropy. If the goal remains unseen, condition
on unseen cells. Geometry is independent of actual hidden goal location.
"""
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError('Temperature must be finite and positive.')
    if tuple(map(tuple, cells)) != planner.cells:
        raise ValueError('Posterior cell order differs from planner geometry.')
    if np.any(beliefs < 0) or not np.allclose(beliefs.sum(-1), 1):
        raise ValueError('Expected normalized posterior distributions.')
    scores = np.zeros((len(beliefs), 7))
    for i, (belief, pose) in enumerate(zip(beliefs, poses)):
        x, y = np.rint(pose[:2] * (planner.size - 1)).astype(int)
        state = (x, y, int(pose[2:].argmax()))
        entropy = -(belief * np.log(np.maximum(belief, 1e-30))).sum()
        for action in range(7):
            nxt = planner.transition(state, action)
            revealed = set(planner.visibility[nxt])
            if action == 2 and nxt[:2] != state[:2]:
                revealed.add(planner.index[nxt[:2]])
            remaining = belief.copy()
            remaining[list(revealed)] = 0
            mass = remaining.sum()
            conditional = remaining / mass if mass else remaining
            expected_entropy = -mass * (conditional * np.log(np.maximum(conditional, 1e-30))).sum()
            scores[i, action] = max(0., entropy - expected_entropy)
    logits = scores / temperature
    logits -= logits.max(-1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(-1, keepdims=True), scores


def fit_record(features, targets, data, args, seed, folder, name, posterior=False):
    fitting = SimpleNamespace(**vars(args), probe_seed=seed,
                              width=budget_width(features.shape[1], targets.shape[1], args.parameter_budget))
    model_data = {**data, 'belief': targets}
    eligible = None if posterior else np.ones(len(targets), bool)
    fitted, ce, kl, prediction = fit_probe(features.astype(np.float32), model_data, fitting, eligible)
    torch.save(fitted, folder / f'{name}.pt')
    mask = (data['split'] == 2) & (data['belief'].max(-1) < 1)
    record = dict(width=fitting.width, parameters=sum(v.numel() for v in fitted['state_dict'].values()),
                  selected_epoch=fitted['selected_epoch'], epochs_run=len(fitted['history']),
                  selected_in_last_10_percent=fitted['selected_epoch'] >= .9 * args.epochs,
                  cross_entropy=summarize_scores(ce, data['episode'], mask),
                  kl=summarize_scores(kl, data['episode'], mask))
    return record, ce, kl, prediction


def summarize_repeats(records, key):
    values = np.array([r[key]['mean'] for r in records], dtype=float)
    return dict(mean=float(values.mean()), minimum=float(values.min()), maximum=float(values.max()),
                per_seed=values.tolist(),
                note='Same held-out layouts across fits; seed range is not a confidence interval.')


def run(args):
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    source = Path(args.rollouts)
    metadata = json.loads(source.with_name('summary.json').read_text())
    config = metadata['config']
    cache_args = SimpleNamespace(**config)
    cache_args.cached_rollouts = str(source)
    cache_args.split_seed = config.get('split_seed', config['probe_seed'])
    data = load_cached_rollouts(cache_args)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if metadata.get('rollout_sha256', digest) != digest:
        raise ValueError('Rollout file differs from its recorded fingerprint.')
    torch.set_num_threads(args.threads)
    out.mkdir(parents=True)
    common = np.c_[data['pose'], data['timestep'] / 400.]
    history = np.c_[data['observation'], common, history_features(data, config['size'])]
    planner = BeliefRewardPlanner(config['size'], config['view_size'])
    print('Computing explicit information-seeking reference policy', flush=True)
    reference, ig = information_policy(data['belief'], data['pose'], data['cells'], planner, args.reference_temperature)
    np.savez_compressed(out / 'reference_policy.npz', probabilities=reference, information_gain=ig,
                        episode=data['episode'], timestep=data['timestep'])
    mask = (data['split'] == 2) & (data['belief'].max(-1) < 1)
    if not mask.any():
        raise ValueError('No unresolved test records.')
    runs = []
    for seed in args.seeds:
        print(f'Fitting seed {seed}', flush=True)
        folder = out / f'seed{seed}'
        folder.mkdir()
        features = dict(history=history, history_belief=np.c_[history, data['belief']],
                        history_shuffled=np.c_[history, shuffled_beliefs(data, seed)])
        result = dict(seed=seed, behavior={}, decoding={}, updates={})
        predictions = {}
        for target_name, target in [('iq', np.eye(7)[data['action']]), ('explicit_ig', reference)]:
            ce_scores = {}
            result['behavior'][target_name] = {}
            for name, x in features.items():
                print(f'  {target_name}: {name}', flush=True)
                record, ce, _, pred = fit_record(x, target, data, args, seed, folder, f'{target_name}_{name}')
                result['behavior'][target_name][name] = record
                ce_scores[name] = ce
                predictions[f'{target_name}_{name}'] = pred[data['split'] == 2]
            for baseline in ('history', 'history_shuffled'):
                result[f'{target_name}_{baseline}_minus_belief'] = paired_interval(
                    ce_scores[baseline] - ce_scores['history_belief'], data, mask, 0)
        print('  Replaying untrained encoder', flush=True)
        random = random_memory(data, cache_args, seed)
        decoded_kl, errors = {}, {}
        for name, x in [('trained', data['hidden']), ('random', random), ('explicit', data['belief'])]:
            print(f'  Decoder: {name}', flush=True)
            record, _, kl, pred = fit_record(np.c_[x, common], data['belief'], data, args, seed,
                                            folder, f'decoder_{name}', posterior=True)
            result['decoding'][name] = record
            decoded_kl[name] = kl
            error, eligible, changed = update_errors(pred, data)
            errors[name] = error
            result['updates'][name] = {
                label: summarize_scores(error, data['episode'], (data['split'] == 2) & eligible & condition)
                for label, condition in [('new_evidence', changed), ('no_new_evidence', ~changed)]}
            predictions[f'decoder_{name}'] = pred[data['split'] == 2]
        stationary_error, eligible, changed = update_errors(np.zeros_like(data['belief']), data)
        update_mask = (data['split'] == 2) & eligible & changed
        for baseline, name in [('random', 'trained'), ('random', 'explicit')]:
            result[f'{baseline}_minus_{name}_kl'] = paired_interval(decoded_kl[baseline] - decoded_kl[name], data, mask, 0)
            result[f'{baseline}_minus_{name}_update'] = paired_interval(errors[baseline] - errors[name], data, update_mask, 0)
        result['stationary_minus_explicit_update'] = paired_interval(stationary_error - errors['explicit'], data, update_mask, 0)
        result['stationary_minus_trained_update'] = paired_interval(stationary_error - errors['trained'], data, update_mask, 0)
        np.savez_compressed(folder / 'test_predictions.npz', **predictions,
                            episode=data['episode'][data['split'] == 2],
                            timestep=data['timestep'][data['split'] == 2])
        (folder / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        runs.append(result)
    keys = [key for key in runs[0] if '_minus_' in key]
    summary = dict(config=vars(args), checkpoint_sha256=metadata['checkpoint_sha256'], rollout_sha256=digest,
                   design='Fixed splits and fitting budgets; approximate equal parameter counts. Per-fit intervals resample layouts, not fitting seeds.',
                   reference='Constructed Bayesian IG actor queried on cached IQ histories; soft action-distribution targets. No new teacher rollouts or training. Demonstrates analysis sensitivity, not emergent belief use.',
                   explicit_decoder='Learned MLP reads an explicit posterior memory; imperfect recovery tests decoder sensitivity beyond identity-copy sanity.',
                   limitations=['Previously inspected layouts; exploratory, not fresh confirmation.',
                                'Parameter counts approximately matched; widths differ and function classes are not identical.',
                                'Shuffled posterior is a marginal placebo, not a conditional randomization test.',
                                'Random encoder seed changes with fitting seed; these variance sources are not separated.',
                                'Known pose is supplied externally. No causal claim about IQ or biological agents.'],
                   reference_diagnostics=dict(unresolved_steps=int(mask.sum()),
                       action_sensitive_steps=int((mask & (np.ptp(ig, axis=1) > 1e-8)).sum())),
                   comparisons={key: summarize_repeats(runs, key) for key in keys}, runs=runs)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    plot(summary, out)
    print(json.dumps(summary['comparisons'], indent=2), flush=True)


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = ['iq_history_minus_belief', 'explicit_ig_history_minus_belief',
            'random_minus_trained_kl', 'random_minus_explicit_kl',
            'stationary_minus_trained_update', 'stationary_minus_explicit_update']
    labels = ['IQ: adding belief', 'Known IG actor: adding belief',
              'Trained vs random memory', 'Explicit vs random memory',
              'Trained vs stationary update', 'Explicit vs stationary update']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for panel, ax in enumerate(axes):
        for j in range(2):
            key = keys[panel * 2 + j]
            for i, run in enumerate(summary['runs']):
                r = run[key]
                if r['mean'] is None:
                    continue
                y = j + (i - (len(summary['runs']) - 1) / 2) * .12
                ax.plot(r['ci95'], [y, y], color=f'C{i}')
                ax.plot(r['mean'], y, 'o', color=f'C{i}', label=f'Seed {run["seed"]}' if j == 0 else None)
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_yticks([0, 1], labels[panel * 2:panel * 2 + 2])
        ax.set_xlabel(['Cross-entropy reduction (nats)', 'Posterior KL reduction (nats)', 'Update L1 error reduction'][panel])
    axes[0].legend()
    fig.suptitle('Positive controls and repeated fits: positive favors belief; layout-bootstrap 95% intervals')
    fig.savefig(out / 'validation.png', dpi=180)
    fig.savefig(out / 'validation.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollouts', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--parameter-budget', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--patience', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--reference-temperature', type=float, default=.25)
    args = p.parse_args()
    if min(args.epochs, args.parameter_budget, args.batch_size, args.threads) < 1 or args.patience < 0:
        p.error('Positive fitting settings required.')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error('Seeds must be unique and nonnegative.')
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.reference_temperature) or args.reference_temperature <= 0:
        p.error('Learning rate and reference temperature must be finite and positive.')
    run(args)


if __name__ == '__main__':
    main()
