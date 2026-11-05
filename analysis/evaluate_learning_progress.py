"""Evaluate saved learning checkpoints on fixed validation data and rollout seeds.

These are diagnostic curves, not final-test estimates or evidence of causality.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from analysis.training_curves import plot_training_curves


def evaluation_commands(config, checkpoint, demos, output, episodes, seed, expert=None):
    policy = ['--checkpoint', str(checkpoint), '--init-temp', str(config['init_temp']),
              '--gamma', str(config['gamma'])]
    if config.get('grid_shape'):
        policy += ['--grid-shape', *map(str, config['grid_shape'])]
    likelihood = [sys.executable, '-m', 'analysis.expert_likelihood', *policy,
                  '--heldout-demos', str(demos), '--output', str(output / 'likelihood.json')]
    if expert:
        likelihood += ['--expert-model', expert, '--expert-kind', 'recurrent']
    alignment = [sys.executable, '-m', 'analysis.evaluate_belief_alignment', *policy,
                 '--size', str(config['size']), '--view-size', str(config['view_size']),
                 '--episodes', str(episodes), '--seed', str(seed),
                 '--output', str(output / 'alignment.csv')]
    return likelihood, alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True, help='New directory for diagnostic curves.')
    parser.add_argument('--expert-model', help='Optional recurrent expert for paired likelihood.')
    parser.add_argument('--validation-demos', help='Override validation file location; hash must match training manifest.')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=10000)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    config = manifest['config']
    if config['env'] != 'partial_obs' or config['num_distractors'] != 0:
        parser.error('The Bayesian goal-location comparator requires partial observation with zero distractors.')
    if not config.get('validation_demo_path'):
        parser.error('This runner requires a separately generated validation file, not an automatic training-data split.')
    demos = Path(args.validation_demos or config['validation_demo_path'])
    if hashlib.sha256(demos.read_bytes()).hexdigest() != manifest['validation_sha256']:
        parser.error('Validation data do not match the training manifest.')
    records = manifest['evaluations']
    if args.episodes < 1 or not records:
        parser.error('Positive episode count and at least one snapshot required.')
    for record in records:
        if not Path(record.get('checkpoint', '')).is_file():
            parser.error(f"Missing checkpoint for step {record['step']}.")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(dict(arguments=vars(args), training_manifest=manifest,
        interpretation='Validation diagnostics; same reset seeds, but policy-dependent visited histories. Not a causal test.'), indent=2))
    rows = []
    for record in records:
        destination = out / f"step_{record['step']:09d}"
        destination.mkdir()
        commands = evaluation_commands(config, record['checkpoint'], demos, destination,
                                       args.episodes, args.seed, args.expert_model)
        for name, command in zip(('likelihood', 'alignment'), commands):
            print(f"Step {record['step']}: {name}", flush=True)
            with (destination / f'{name}.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        likelihood = json.loads((destination / 'likelihood.json').read_text())
        alignment = json.loads((destination / 'alignment.json').read_text())
        unresolved = alignment['unresolved']
        row = dict(step=record['step'], validation_episode_nll=record['episode_mean_nll'],
                   validation_step_nll=likelihood['negative_log_likelihood'],
                   validation_action_accuracy=likelihood['expert_top1_accuracy'],
                   strict_ig_agreement=unresolved['strict_agreement_unique_ig'],
                   unique_ig_steps=unresolved['unique_ig_steps'],
                   outside_navigation_fraction=unresolved['actual_outside_navigation_fraction'],
                   success_rate=alignment['success_rate'])
        for key in ('mean_likelihood_gap', 'mean_kl_expert_to_iq', 'greedy_policy_agreement'):
            row[key] = likelihood.get('paired', {}).get(key)
        rows.append(row)
        with (out / 'curves.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
    keys = ['validation_episode_nll', 'validation_action_accuracy', 'strict_ig_agreement', 'success_rate']
    curves = {key: [(r['step'], r[key]) for r in rows if r[key] is not None] for key in keys}
    plot_training_curves({k: v for k, v in curves.items() if v}, out / 'curves.png',
                         title='Expert imitation and information-gain alignment during learning')
    print(f"Saved diagnostic curves to {out}")


if __name__ == '__main__':
    main()
