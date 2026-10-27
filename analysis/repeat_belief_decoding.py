"""Repeat frozen-policy probes on fixed cached layouts; never select by test score."""
import argparse
import csv
import json
from pathlib import Path

from analysis.evaluate_belief_decoding import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', default=[
        'artifacts/belief_decoding/fov3_seed0_probe0',
        'artifacts/belief_decoding/fov3_seed1_probe0'])
    parser.add_argument('--probe-seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--output-dir', default='artifacts/belief_decoding/robustness_fixed_budget')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 0 or len(set(args.probe_seeds)) != len(args.probe_seeds):
        parser.error('Use positive epochs, nonnegative patience, and distinct probe seeds.')
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    sources = [Path(p).resolve() for p in args.sources]
    for source in sources:
        if not (source / 'summary.json').is_file() or not (source / 'rollouts.npz').is_file():
            raise FileNotFoundError(f'Missing completed source run: {source}')
    out.mkdir(parents=True)
    rows = []
    for index, source in enumerate(sources):
        original = json.loads((source / 'summary.json').read_text())
        for seed in args.probe_seeds:
            config = dict(original['config'])
            config.update(cached_rollouts=str(source / 'rollouts.npz'),
                          split_seed=config.get('split_seed', config['probe_seed']),
                          probe_seed=seed, epochs=args.epochs, patience=args.patience,
                          output_dir=str(out / f'policy{index}_probe{seed}'))
            print(f'Policy {index} ({source.name}), probe initialization {seed}', flush=True)
            run(argparse.Namespace(**config))
            summary = json.loads((Path(config['output_dir']) / 'summary.json').read_text())
            for baseline, comparison in summary['baseline_minus_hidden_kl'].items():
                rows.append(dict(policy=index, source=str(source), probe_seed=seed,
                                 baseline=baseline, improvement=comparison['mean'],
                                 ci_low=comparison['ci95'][0], ci_high=comparison['ci95'][1],
                                 baseline_kl=summary['probes'][baseline]['unresolved']['kl']['episode_mean'],
                                 hidden_kl=summary['probes']['hidden']['unresolved']['kl']['episode_mean'],
                                 baseline_epoch=summary['probes'][baseline]['selected_epoch'],
                                 hidden_epoch=summary['probes']['hidden']['selected_epoch'],
                                 baseline_epochs_run=summary['probes'][baseline]['epochs_run'],
                                 hidden_epochs_run=summary['probes']['hidden']['epochs_run']))
            # Incremental summary survives an interrupted repeat without implying completion.
            with (out / 'runs.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    report = dict(config=vars(args), status='complete', runs=rows,
                  interpretation='Every initialization is reported. Shared test layouts are not independent replications. Per-fit intervals describe layout sampling only; no pooled significance or causal claim.')
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    plot(rows, sources, out)
    print(f'Completed. Comparison: {out / "robustness.png"}', flush=True)


def plot(rows, sources, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(sources), squeeze=False,
                             figsize=(6 * len(sources), 4.5), layout='constrained')
    for index, source in enumerate(sources):
        ax = axes[0, index]
        selected = [r for r in rows if r['policy'] == index]
        for y, row in enumerate(selected):
            color = 'C0' if row['baseline'] == 'observation' else 'C1'
            ax.plot([row['ci_low'], row['ci_high']], [y, y], color=color)
            ax.plot(row['improvement'], y, 'o', color=color)
        ax.axvline(0, color='gray', linestyle='--')
        ax.set(yticks=range(len(selected)),
               yticklabels=[f"{r['baseline']} / probe {r['probe_seed']}" for r in selected],
               title=source.name, xlabel='Baseline KL − memory KL (nats); positive favors memory')
    fig.suptitle('Fixed splits, repeated decoder fits\n95% layout-bootstrap intervals per fit; not independent test sets')
    fig.savefig(out / 'robustness.png', dpi=180)
    fig.savefig(out / 'robustness.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(len(sources), 3, squeeze=False,
                             figsize=(13, 3.5 * len(sources)), layout='constrained')
    for index, source in enumerate(sources):
        seeds = sorted({r['probe_seed'] for r in rows if r['policy'] == index})
        for col, name in enumerate(('observation', 'reset_hidden', 'hidden')):
            ax = axes[index, col]
            for seed in seeds:
                history = json.loads((out / f'policy{index}_probe{seed}' /
                                      f'{name}_training.json').read_text())
                epochs = [h['epoch'] for h in history]
                values = [h['validation_unresolved_kl'] for h in history]
                line, = ax.plot(epochs, values, label=f'probe {seed}', alpha=.8)
                best = min(range(len(values)), key=values.__getitem__)
                ax.plot(epochs[best], values[best], 'o', color=line.get_color())
            ax.set(title=f'Policy {index}: {name}', xlabel='Decoder epoch',
                   ylabel='Validation unresolved KL (nats)')
            ax.legend()
    fig.suptitle('Validation curves; dots mark selected epochs (no test-based selection)')
    fig.savefig(out / 'validation_curves.png', dpi=180)
    fig.savefig(out / 'validation_curves.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
