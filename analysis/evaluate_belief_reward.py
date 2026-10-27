"""Compare saved matched histories to finite-horizon Bayesian reward planning."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.belief_reward_planner import BeliefRewardPlanner, maximizers


def probabilities(q, temperature):
    z = np.asarray(q, dtype=float) / temperature
    p = np.exp(z - z.max())
    return p / p.sum()


def cluster_metric(rows, key):
    rows = [r for r in rows if r[key] is not None]
    seeds = sorted({r['reset_seed'] for r in rows})
    if not seeds:
        return dict(mean=None, ci95=None, pairs=0, reset_seeds=0)
    values = np.array([np.mean([r[key] for r in rows if r['reset_seed'] == seed]) for seed in seeds])
    draws = np.random.default_rng(0).integers(len(values), size=(2000, len(values)))
    return dict(mean=float(values.mean()),
                ci95=np.quantile(values[draws].mean(1), [.025, .975]).tolist() if len(seeds) > 1 else None,
                pairs=len(rows), reset_seeds=len(seeds))


def run(source, out, horizons):
    source = Path(source)
    original = json.loads((source / 'summary.json').read_text())
    config = original['config']
    planner = BeliefRewardPlanner(size=config['size'], view_size=config['view_size'],
                                 max_steps=config['max_steps'], gamma=config['gamma'])
    with (source / 'pairs.csv').open() as f:
        pairs = [r for r in csv.DictReader(f) if r['unresolved'] == 'True']
    results = []
    with np.load(source / 'matched_traces.npz', allow_pickle=False) as data:
        for i, pair in enumerate(pairs):
            states = []
            for label in ('a', 'b'):
                prefix = f"pair{pair['pair_id']}_{label}_"
                states.append(dict(pose=data[prefix+'positions'][-1],
                                   candidates=data[prefix+'candidates'],
                                   iq=data[prefix+'q'], ig=data[prefix+'ig'],
                                   obs=data[prefix+'observations'][-1],
                                   elapsed=len(data[prefix+'actions'])))
            if not (np.array_equal(states[0]['pose'], states[1]['pose'])
                    and np.array_equal(states[0]['obs'], states[1]['obs'])
                    and states[0]['elapsed'] == states[1]['elapsed']):
                raise ValueError('Source traces do not have matched endpoints.')
            for horizon in horizons:
                values = [planner.q_values(s['pose'], s['candidates'], s['elapsed'], horizon) for s in states]
                winners = [maximizers(v) for v in values]
                iq_actions = [int(s['iq'].argmax()) for s in states]
                eligible = all(len(w) == 1 for w in winners) and winners[0] != winners[1]
                probs = [probabilities(s['iq'], config['init_temp']) for s in states]
                shift = None
                if eligible:
                    a, b = winners[0][0], winners[1][0]
                    shift = float(.5 * (probs[0][a] - probs[1][a] + probs[1][b] - probs[0][b]))
                row = dict(pair_id=int(pair['pair_id']), reset_seed=int(pair['reset_seed']),
                           kind=pair['kind'], same_belief=pair['same_belief'] == 'True', horizon=horizon,
                           unique_both=all(len(w) == 1 for w in winners), different_unique_reward=eligible,
                           both_match_reward=all(a in w for a, w in zip(iq_actions, winners)) if eligible else None,
                           own_reward_support_shift=shift,
                           mean_action_regret=float(np.mean([v.max() - v[a] for v,a in zip(values, iq_actions)])),
                           all_tied_both=all(len(w) == 7 for w in winners))
                for label, value, winner, state, action in zip(('a','b'), values, winners, states, iq_actions):
                    ig_winner = maximizers(state['ig'])
                    row.update({f'{label}_reward_q': json.dumps(value.tolist()),
                                f'{label}_reward_maximizers': json.dumps(winner),
                                f'{label}_iq_action': action,
                                f'{label}_strict_unique_agreement': action == winner[0] if len(winner) == 1 else None,
                                f'{label}_reward_vs_ig_disagreement': winner[0] != ig_winner[0]
                                if len(winner) == len(ig_winner) == 1 else None})
                results.append(row)
            if (i+1) % 100 == 0:
                print(f'{source.name}: planned {i+1}/{len(pairs)} pairs', flush=True)
    out.mkdir(parents=True)
    with (out / 'pairs.csv').open('w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    groups = {}
    for horizon in horizons:
        group = {}
        for name, select in [('different_belief', lambda r: r['kind'] == 'loop' and not r['same_belief']),
                             ('same_belief_control', lambda r: r['kind'] == 'rotation' and r['same_belief'])]:
            selected = [r for r in results if r['horizon'] == horizon and select(r)]
            keys = ('unique_both', 'different_unique_reward', 'all_tied_both', 'mean_action_regret',
                    'both_match_reward', 'own_reward_support_shift',
                    'a_strict_unique_agreement', 'b_strict_unique_agreement',
                    'a_reward_vs_ig_disagreement', 'b_reward_vs_ig_disagreement')
            group[name] = {key: cluster_metric(selected, key) for key in keys}
        groups[str(horizon)] = group
    report = dict(source=str(source.resolve()), checkpoint_sha256=original['checkpoint_sha256'],
                  source_config=config, horizons=horizons, groups=groups,
                  objective='Expected discounted MiniGrid goal-entry reward (1 - 0.9 * arrival_step / max_steps); no information-gain bonus; seven actions.',
                  approximation='Exact uniform-posterior belief recursion within each finite horizon; zero value beyond horizon. Not infinite-horizon optimal. Known empty geometry and pose; hidden goal is not an input.',
                  interpretation='Reanalysis of controlled forced histories. No unique-winner claim on tied states. Probability shifts are distinct from executed-action agreement. Bootstrap clusters reset seeds; fixed checkpoints and reused layouts, exploratory only.')
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    plot(report, out)
    for h, groups_h in groups.items():
        g = groups_h['different_belief']
        print(source.name, 'horizon', h, 'different unique reward pairs',g['both_match_reward']['pairs'],
              'both match',g['both_match_reward']['mean'],'shift',g['own_reward_support_shift']['mean'])


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for i, horizon in enumerate(report['horizons']):
        g = report['groups'][str(horizon)]['different_belief']
        for ax, key in zip(axes, ['unique_both', 'both_match_reward', 'own_reward_support_shift']):
            value = g[key]
            if value['mean'] is not None:
                ax.plot(i, value['mean'], 'o', color='C0')
                if value['ci95']:
                    ax.plot([i,i],value['ci95'],color='C0')
            else:
                ax.text(i, .05, 'N/A', ha='center')
    for ax, title in zip(axes, ['Both histories have unique reward winners',
                                'Both IQ actions match different winners', 'Reward-directed probability shift']):
        ax.set(xticks=range(len(report['horizons'])), xticklabels=report['horizons'],
               xlabel='Planning horizon (actions)', title=title)
    axes[0].set_ylim(0,1)
    axes[1].set_ylim(0,1)
    axes[2].axhline(0,color='gray',linestyle='--')
    fig.suptitle('Bayesian reward reference on matched histories; 95% reset-cluster intervals')
    fig.savefig(out / 'belief_reward.png',dpi=180)
    fig.savefig(out / 'belief_reward.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', default=[f'artifacts/matched_histories/fov3_seed{s}_approach1' for s in (0,1)])
    parser.add_argument('--horizons', nargs='+', type=int, default=[3,5])
    parser.add_argument('--output-dir',default='artifacts/belief_reward')
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        parser.error('Output directory already exists; choose a new directory.')
    if any(h < 1 for h in args.horizons) or len(set(args.horizons)) != len(args.horizons):
        parser.error('Horizons must be positive and distinct.')
    for source in args.sources:
        if not (Path(source) / 'matched_traces.npz').is_file():
            parser.error(f'Missing traces in {source}')
    out.mkdir(parents=True)
    for i, source in enumerate(args.sources):
        run(source, out / f'policy{i}', args.horizons)


if __name__ == '__main__':
    main()
