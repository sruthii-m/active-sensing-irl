"""Held-out behavioral explanations and descriptive belief-update tracking.

Uses immutable belief-decoding rollouts. These are observational tests, not
causal interventions or a proof that the GRU implements Bayesian filtering.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from analysis.evaluate_belief_decoding import (
    fit_probe, load_cached_rollouts, paired_interval, summarize_scores,
)
from analysis.policy_io import load_policy
from imitation.recurrent_encoder import RecurrentEncoder


def history_features(data, size, lag=4):
    """Only past actions; visitation includes current pose. Reset each episode."""
    result = np.zeros((len(data['episode']), lag * 7 + size * size), np.float32)
    for episode in np.unique(data['episode']):
        indices = np.flatnonzero(data['episode'] == episode)
        if not np.array_equal(data['timestep'][indices], np.arange(len(indices))):
            raise ValueError('Rollouts must contain ordered complete episode prefixes.')
        visited = np.zeros(size * size, np.float32)
        past = []
        for i in indices:
            x, y = np.rint(data['pose'][i, :2] * (size - 1)).astype(int)
            visited[x * size + y] = 1
            for offset, action in enumerate(reversed(past[-lag:])):
                result[i, offset * 7 + action] = 1
            result[i, lag * 7:] = visited
            past.append(int(data['action'][i]))
    return result


@torch.no_grad()
def random_memory(data, args, seed):
    agent = load_policy(args, data['observation'].shape[1])
    if not hasattr(agent, 'encoder'):
        raise ValueError('A recurrent checkpoint is required.')
    trained = agent.encoder
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        encoder = RecurrentEncoder(data['observation'].shape[1],
                                   trained.gru.input_size, trained.rnn_hidden_dim,
                                   trained.grid_shape).eval()
    result = np.empty_like(data['hidden'])
    for episode in np.unique(data['episode']):
        hidden = None
        for i in np.flatnonzero(data['episode'] == episode):
            obs = torch.tensor(data['observation'][i:i+1], dtype=torch.float32)
            hidden, _ = encoder(obs, hidden)
            result[i] = hidden.numpy()[0]
    return result


def update_errors(prediction, data):
    """L1 error in posterior increments, excluding boundaries and resolved priors."""
    n = len(prediction)
    eligible = np.zeros(n, dtype=bool)
    eligible[1:] = ((data['episode'][1:] == data['episode'][:-1]) &
                    (data['timestep'][1:] == data['timestep'][:-1] + 1) &
                    (data['belief'][:-1].max(-1) < 1))
    delta = np.zeros_like(prediction)
    truth = np.zeros_like(prediction)
    delta[1:] = np.diff(prediction, axis=0)
    truth[1:] = np.diff(data['belief'], axis=0)
    changed = np.abs(truth).sum(-1) > 1e-7
    return np.abs(delta - truth).sum(-1), eligible, changed


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
    torch.set_num_threads(args.threads)
    history = history_features(data, config['size'])
    print('Replaying observations through an untrained encoder', flush=True)
    random = random_memory(data, cache_args, args.random_seed)
    common = np.c_[data['pose'], data['timestep'] / 400.]
    obs = np.c_[data['observation'], common]
    features = dict(observation=obs, history=np.c_[obs, history],
                    belief=np.c_[data['belief'], common],
                    history_belief=np.c_[obs, history, data['belief']])
    test = data['split'] == 2
    unresolved = data['belief'].max(-1) < 1
    out.mkdir(parents=True)
    summary = dict(config=vars(args), checkpoint_sha256=metadata['checkpoint_sha256'],
                   rollout_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   interpretation='Exploratory observational analysis, not causal use. Existing test layouts may have been inspected previously. Intervals omit training variability.',
                   behavior_target='Recorded greedy IQ action, not expert action or task success',
                   weighting='Equal episodes; paired confidence intervals resample initial layouts',
                   capacity='Same MLP width, different input dimensions; parameter counts reported. Capacity matching remains a follow-up.',
                   positive_control='Exact posterior input gives zero update error by construction: metric sanity check only, not an independently trained agent.',
                   behavior={}, decoding={}, updates={})
    scores, saved = {}, {}
    action_data = {**data, 'belief': np.eye(7)[data['action']]}
    for name, x in features.items():
        print(f'Fitting action model: {name}', flush=True)
        fitted, ce, _, prediction = fit_probe(x.astype(np.float32), action_data, args,
                                             validation_mask=np.ones(len(test), bool))
        scores[name] = ce
        summary['behavior'][name] = dict(
            test_nll=summarize_scores(ce, data['episode'], test),
            unresolved_nll=summarize_scores(ce, data['episode'], test & unresolved),
            test_accuracy=summarize_scores((prediction.argmax(-1) == data['action']).astype(float), data['episode'], test),
            selected_epoch=fitted['selected_epoch'],
            parameters=sum(v.numel() for v in fitted['state_dict'].values()))
        torch.save(fitted, out / f'action_{name}.pt')
        saved[f'action_{name}'] = prediction[test]
    summary['behavior']['history_minus_history_belief_nll'] = paired_interval(
        scores['history'] - scores['history_belief'], data, test & unresolved, args.probe_seed)
    decode_scores, update_scores = {}, {}
    for name, x in dict(observation=data['observation'], reset_hidden=data['reset_hidden'],
                        random_hidden=random, hidden=data['hidden']).items():
        print(f'Fitting posterior decoder: {name}', flush=True)
        fitted, _, kl, prediction = fit_probe(np.c_[x, common].astype(np.float32), data, args)
        decode_scores[name] = kl
        summary['decoding'][name] = dict(
            unresolved_kl=summarize_scores(kl, data['episode'], test & unresolved),
            selected_epoch=fitted['selected_epoch'],
            parameters=sum(v.numel() for v in fitted['state_dict'].values()))
        error, eligible, changed = update_errors(prediction, data)
        update_scores[name] = error
        summary['updates'][name] = {
            label: summarize_scores(error, data['episode'], test & eligible & mask)
            for label, mask in [('new_evidence', changed), ('no_new_evidence', ~changed)]}
        torch.save(fitted, out / f'belief_{name}.pt')
        saved[f'belief_{name}'] = prediction[test]
    summary['random_minus_trained_kl'] = paired_interval(
        decode_scores['random_hidden'] - decode_scores['hidden'], data, test & unresolved, args.probe_seed)
    summary['random_minus_trained_update_l1'] = paired_interval(
        update_scores['random_hidden'] - update_scores['hidden'], data,
        test & eligible & changed, args.probe_seed)
    # A stationary prediction is essential: small increments alone are not evidence of filtering.
    stationary = np.repeat(data['belief'][:1], len(test), axis=0)
    for name, prediction in [('stationary', stationary), ('exact_belief_sanity', data['belief'])]:
        error, eligible, changed = update_errors(prediction, data)
        summary['updates'][name] = {label: summarize_scores(error, data['episode'], test & eligible & mask)
                                   for label, mask in [('new_evidence', changed), ('no_new_evidence', ~changed)]}
    np.savez_compressed(out / 'test_predictions.npz', episode=data['episode'][test],
                        timestep=data['timestep'][test], belief=data['belief'][test],
                        action=data['action'][test], **saved)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    plot(summary, out)
    print(json.dumps({k: summary[k] for k in ('random_minus_trained_kl', 'random_minus_trained_update_l1')}, indent=2))
    print('Saved', out)


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    groups = [(summary['behavior'], 'unresolved_nll', 'Action NLL (nats)'),
              (summary['decoding'], 'unresolved_kl', 'Posterior KL (nats)'),
              (summary['updates'], 'new_evidence', 'Posterior increment L1 error')]
    for ax, (records, metric, title) in zip(axes, groups):
        names = [n for n, r in records.items() if metric in r]
        ax.bar(range(len(names)), [records[n][metric]['episode_mean'] if records[n][metric]['episode_mean'] is not None else np.nan for n in names])
        ax.set_xticks(range(len(names)), names, rotation=40, ha='right')
        ax.set_title(title + '\nLower is better')
    fig.suptitle('Held-out consistency tests: observational, not causal')
    fig.savefig(out / 'consistency.png', dpi=180)
    fig.savefig(out / 'consistency.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollouts', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--patience', type=int, default=0)
    p.add_argument('--probe-seed', type=int, default=0)
    p.add_argument('--random-seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    if min(args.epochs, args.width, args.batch_size, args.threads) < 1 or args.patience < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        p.error('Positive training settings and nonnegative patience required.')
    run(args)


if __name__ == '__main__':
    main()
