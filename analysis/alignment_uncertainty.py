"""Action-frequency reference and episode-cluster bootstrap for strict alignment.

The reference is the expected agreement after uniformly permuting actual actions
within fixed candidate-count bins of unresolved, unique-IG states. It preserves
all seven action frequencies, including non-navigation choices. This is a
marginal-frequency diagnostic, not a causal or optimal-policy null.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

BIN_LABELS = ('2-3', '4-7', '8-15', '16-31', '32-63', '64+')


def _true(value):
    if value not in ('True', 'False'):
        raise ValueError(f'Expected CSV boolean, got {value!r}')
    return value == 'True'


def _scores(matches, counts, action_counts, ig_counts):
    total = counts.sum()
    if total == 0:
        return None
    bin_counts = action_counts.sum(axis=-1)
    expected_matches = np.divide(
        (action_counts * ig_counts).sum(axis=-1), bin_counts,
        out=np.zeros_like(bin_counts, dtype=float), where=bin_counts > 0).sum()
    observed = matches.sum() / total
    reference = expected_matches / total
    return np.asarray([observed, reference, observed - reference])


def analyze(rows, resamples=5000, seed=0):
    if resamples < 1:
        raise ValueError('resamples must be positive')
    episodes = sorted({int(r['episode']) for r in rows})
    if not episodes:
        raise ValueError('CSV contains no episodes')
    index = {ep: i for i, ep in enumerate(episodes)}
    matches = np.zeros(len(episodes))
    counts = np.zeros(len(episodes))
    actions = np.zeros((len(episodes), len(BIN_LABELS), 7))
    targets = np.zeros_like(actions)
    for r in rows:
        if _true(r['resolved']) or _true(r['ig_tie']):
            continue
        n = int(r['candidates'])
        action, target = int(r['executed_action']), int(r['ig_argmax'])
        if n < 2 or action not in range(7) or target not in range(3):
            raise ValueError('Invalid unresolved action/candidate row')
        correct = action == target
        if correct != _true(r['strict_agreement']):
            raise ValueError('CSV strict agreement does not match executed action')
        e = index[int(r['episode'])]
        b = min(n.bit_length() - 2, len(BIN_LABELS) - 1)
        counts[e] += 1
        matches[e] += correct
        actions[e, b, action] += 1
        targets[e, b, target] += 1
    point = _scores(matches, counts, actions.sum(axis=0), targets.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = []
    if point is not None:
        for _ in range(resamples):
            # Whole episodes are selected together, including episodes with no
            # eligible states. Preserve the original step-weighted estimand.
            weights = rng.multinomial(len(episodes), np.full(len(episodes), 1 / len(episodes)))
            score = _scores(matches * weights, counts * weights,
                            np.einsum('e,eba->ba', weights, actions),
                            np.einsum('e,eba->ba', weights, targets))
            if score is not None:
                draws.append(score)
    keys = ('strict_agreement', 'action_frequency_reference', 'excess_agreement')
    result = dict(episodes=len(episodes), eligible_episodes=int(np.count_nonzero(counts)),
                  eligible_steps=int(counts.sum()), bootstrap_draws_requested=resamples,
                  bootstrap_draws_valid=len(draws))
    for i, key in enumerate(keys):
        result[key] = float(point[i]) if point is not None else None
        result[key + '_ci95'] = (np.quantile(np.asarray(draws)[:, i], [.025, .975]).tolist()
                                 if draws else None)
    result['bins'] = {}
    for b, label in enumerate(BIN_LABELS):
        a, g = actions[:, b].sum(axis=0), targets[:, b].sum(axis=0)
        total = a.sum()
        result['bins'][label] = dict(
            steps=int(total), episodes=int(np.count_nonzero(actions[:, b].sum(axis=-1))),
            action_counts=a.astype(int).tolist(), ig_argmax_counts=g.astype(int).tolist(),
            reference=float(np.dot(a, g) / total ** 2) if total else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/state_uncertainty')
    parser.add_argument('--resamples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='artifacts/analysis/alignment_uncertainty.json')
    args = parser.parse_args()
    paths = sorted(Path(args.root).glob('*/alignment.csv'))
    if not paths:
        parser.error('No */alignment.csv files found under --root')
    report = dict(config=vars(args), method={
        'eligible': 'unresolved states with a unique navigation IG maximum',
        'reference': 'sum_b (n_b/N) sum_a p_b(actual=a) p_b(IG argmax=a)',
        'bins': list(BIN_LABELS),
        'interval': '95% percentile bootstrap of complete episodes; refit frequencies each draw',
        'limitations': 'Exploratory, fixed trained policies, on-policy histories; no training-seed CI, p-values or multiplicity correction. Frequency reference does not condition on pose/geometry.'}, runs={})
    for path in paths:
        with path.open(newline='') as f:
            result = analyze(list(csv.DictReader(f)), args.resamples, args.seed)
        result['source'] = str(path)
        report['runs'][path.parent.name] = result
        if result['strict_agreement'] is None:
            print(f'{path.parent.name}: no eligible states')
        else:
            lo, hi = result['excess_agreement_ci95']
            print(f"{path.parent.name}: observed={result['strict_agreement']:.1%}, "
                  f"reference={result['action_frequency_reference']:.1%}, "
                  f"excess={result['excess_agreement'] * 100:.1f} pp "
                  f"[95% CI {lo * 100:.1f}, {hi * 100:.1f}]; "
                  f"{result['eligible_steps']} steps / {result['eligible_episodes']} eligible episodes")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
