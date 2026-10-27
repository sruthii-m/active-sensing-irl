"""Explicit-belief positive control on endpoint-matched forced histories.

Selection uses differing unique IG winners, never IQ outputs. Reference success
is a calibration check by construction, not evidence for emergent belief use.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.belief_reward_planner import BeliefRewardPlanner
from analysis.evaluate_matched_histories import pair_metrics, summarize
from analysis.validate_belief_consistency import information_policy


def reconstructed_pair(traces, pair_id):
    pair = []
    for label in ('a', 'b'):
        prefix = f'pair{pair_id}_{label}_'
        pair.append({key: traces[prefix + key] for key in
                     ('observations', 'positions', 'candidates', 'q', 'ig')})
    a, b = pair
    if (not np.array_equal(a['observations'][-1], b['observations'][-1])
            or not np.array_equal(a['positions'][-1], b['positions'][-1])
            or len(a['observations']) != len(b['observations'])):
        raise ValueError(f'Pair {pair_id} does not match observation, pose and duration.')
    for row in pair:
        row['candidates'] = tuple(map(tuple, row['candidates']))
    return pair


def reference_pair(pair, planner, temperature, iq_temperature):
    beliefs = np.zeros((2, len(planner.cells)))
    poses = []
    for i, row in enumerate(pair):
        candidates = row['candidates']
        for cell in candidates:
            beliefs[i, planner.index[cell]] = 1 / len(candidates)
        x, y, direction = row['positions'][-1]
        poses.append(np.r_[np.array([x, y]) / (planner.size - 1), np.eye(4)[direction]])
    probabilities, scores = information_policy(beliefs, np.asarray(poses), planner.cells,
                                               planner, temperature)
    for i, row in enumerate(pair):
        # Independent geometry implementation must reproduce the collector's IG.
        if not np.allclose(scores[i, :3], row['ig'], atol=1e-8):
            raise ValueError('Reference IG disagrees with collected endpoint scores.')
    # pair_metrics divides Q by the IQ temperature: scale reference scores so
    # this produces exactly the intended reference softmax, including no-ops.
    reference = [{**row, 'q': scores[i] * iq_temperature / temperature}
                 for i, row in enumerate(pair)]
    return reference, probabilities


def run(args):
    source, out = Path(args.histories), Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    metadata = json.loads((source / 'summary.json').read_text())
    config = metadata['config']
    planner = BeliefRewardPlanner(config['size'], config['view_size'])
    with (source / 'pairs.csv').open() as f:
        rows = list(csv.DictReader(f))
    iq_rows, reference_rows, saved = [], [], []
    with np.load(source / 'matched_traces.npz', allow_pickle=False) as traces:
        for row in rows:
            pair_id = int(row['pair_id'])
            pair = reconstructed_pair(traces, pair_id)
            reference, probabilities = reference_pair(pair, planner, args.reference_temperature, config['init_temp'])
            common = dict(pair_id=pair_id, reset_seed=int(row['reset_seed']), kind=row['kind'], distance=int(row['distance']))
            iq = {**common, **pair_metrics(*pair, config['init_temp'])}
            control = {**common, **pair_metrics(*reference, config['init_temp'])}
            if iq['different_unique_ig'] != control['different_unique_ig']:
                raise AssertionError('Policy changed pair eligibility.')
            iq_rows.append(iq)
            reference_rows.append(control)
            saved.append(dict(**common, iq=iq, explicit=control,
                              reference_probabilities=probabilities.tolist()))
    result = dict(config=vars(args), source_config=config,
                  checkpoint_sha256=metadata['checkpoint_sha256'],
                  source_hashes={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                 for name in ('summary.json', 'pairs.csv', 'matched_traces.npz')},
                  iq=summarize(iq_rows, args.bootstrap_seed),
                  explicit=summarize(reference_rows, args.bootstrap_seed),
                  interpretation='Reference IG agreement is guaranteed by eligible-pair construction and checks measurement sensitivity. IQ effects indicate history-dependent IG alignment, not causal use of an internal belief variable.',
                  limitations=['Forced histories may be off-policy.',
                               'New reset seeds are not guaranteed new layouts or unseen IQ-training states.',
                               'Same-belief rotation controls differ in movement and coverage from loop histories.',
                               'One-step IG is not reward-optimal planning; no between-group causal claim.',
                               'Reset-cluster intervals do not include policy-training variability.'])
    out.mkdir(parents=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (out / 'pairs.json').write_text(json.dumps(saved, indent=2, allow_nan=False))
    plot(result, out)
    print(json.dumps({name: result[name]['different_unique_ig'] for name in ('iq', 'explicit')}, indent=2))


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    panels = [('different_unique_ig', 'both_actions_match_ig', 'Both chosen actions match IG'),
              ('different_unique_ig', 'own_ig_support_shift', 'Probability shift toward own IG action'),
              ('same_belief_control', 'policy_tv', 'Same-belief controls: policy TV')]
    for ax, (group, key, title) in zip(axes, panels):
        for i, name in enumerate(('iq', 'explicit')):
            r = result[name][group][key]
            if r['mean'] is not None:
                ax.bar(i, r['mean'], color=f'C{i}')
                if r['ci95'] is not None:
                    ax.plot([i, i], r['ci95'], color='black')
        ax.set_xticks([0, 1], ['IQ', 'Explicit IG control'])
        ax.axhline(0, color='gray', linewidth=.5)
        ax.set_title(title)
    fig.suptitle('Endpoint-matched histories; reference is a constructed positive control')
    fig.savefig(out / 'matched_validation.png', dpi=180)
    fig.savefig(out / 'matched_validation.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--histories', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--reference-temperature', type=float, default=.25)
    p.add_argument('--bootstrap-seed', type=int, default=0)
    args = p.parse_args()
    if not np.isfinite(args.reference_temperature) or args.reference_temperature <= 0:
        p.error('Reference temperature must be finite and positive.')
    run(args)


if __name__ == '__main__':
    main()
