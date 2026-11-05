"""Evaluate saved learning checkpoints on fixed validation data and rollout seeds.

These are diagnostic curves, not final-test estimates or evidence of causality.
Task-reward planning and information gain are separate reference policies.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from analysis.training_curves import plot_training_curves


def evaluation_commands(config, checkpoint, demos, output, episodes, seed, expert=None,
                        reward_horizons=(3, 5)):
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
                 '--reward-horizons', *map(str, reward_horizons),
                 '--output', str(output / 'alignment.csv')]
    # Recurrent trainer max_steps is a gradient-update cap, not an episode limit.
    if config.get('env_max_steps') is not None:
        alignment += ['--max-steps', str(config['env_max_steps'])]
    return likelihood, alignment


def learning_curve_row(record, likelihood, alignment, horizons):
    unresolved = alignment['unresolved']
    row = dict(step=record['step'], validation_episode_nll=record['episode_mean_nll'],
               validation_step_nll=likelihood['negative_log_likelihood'],
               validation_action_accuracy=likelihood['expert_top1_accuracy'],
               strict_ig_agreement=unresolved['strict_agreement_unique_ig'],
               unique_ig_steps=unresolved['unique_ig_steps'],
               outside_navigation_fraction=unresolved['actual_outside_navigation_fraction'],
               success_rate=alignment['success_rate'], unresolved_steps=unresolved['steps'])
    for horizon in horizons:
        reference = alignment['reward_planning']['by_horizon'][str(horizon)]['unresolved']
        row.update({f'strict_reward_agreement_h{horizon}': reference['strict_agreement_unique_reward'],
                    f'reward_action_regret_h{horizon}': reference['mean_action_regret'],
                    f'reward_action_regret_informative_h{horizon}': reference['mean_action_regret_informative'],
                    f'reward_q_in_maximizers_h{horizon}': reference['q_in_reward_maximizers']})
        for key in ('unique_reward_steps', 'unique_reward_episodes', 'reward_tie_steps',
                    'all_reward_tied_steps', 'informative_reward_steps', 'informative_reward_episodes'):
            row[f'{key}_h{horizon}'] = reference[key]
    for key in ('mean_likelihood_gap', 'mean_kl_expert_to_iq', 'greedy_policy_agreement'):
        row[key] = likelihood.get('paired', {}).get(key)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True, help='New directory for diagnostic curves.')
    parser.add_argument('--expert-model', help='Optional recurrent expert for paired likelihood.')
    parser.add_argument('--validation-demos', help='Override validation file location; hash must match training manifest.')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=900000,
                        help='Diagnostic rollout reset seed; keep this interval separate from final-test layouts (default: 900000).')
    parser.add_argument('--reward-horizons', type=int, nargs='+', default=[3, 5],
                        help='Positive action horizons for Bayesian task-reward planning (default: 3 5).')
    args = parser.parse_args()
    if any(h < 1 for h in args.reward_horizons) or len(set(args.reward_horizons)) != len(args.reward_horizons):
        parser.error('--reward-horizons must contain distinct positive integers')
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
        interpretation='Validation diagnostics; same reset seeds, but policy-dependent visited histories. Bayesian task-reward planning uses the simulator reward and a finite horizon, not the recovered IQ reward or an information-gain bonus. These curves are not a causal test or checkpoint-selection criteria.'), indent=2))
    rows = []
    for record in records:
        destination = out / f"step_{record['step']:09d}"
        destination.mkdir()
        commands = evaluation_commands(config, record['checkpoint'], demos, destination,
                                       args.episodes, args.seed, args.expert_model, args.reward_horizons)
        for name, command in zip(('likelihood', 'alignment'), commands):
            print(f"Step {record['step']}: {name}", flush=True)
            with (destination / f'{name}.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        likelihood = json.loads((destination / 'likelihood.json').read_text())
        alignment = json.loads((destination / 'alignment.json').read_text())
        row = learning_curve_row(record, likelihood, alignment, args.reward_horizons)
        rows.append(row)
        with (out / 'curves.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
    keys = ['validation_episode_nll', 'validation_action_accuracy']
    for horizon in args.reward_horizons:
        keys += [f'strict_reward_agreement_h{horizon}', f'reward_action_regret_informative_h{horizon}']
    keys += ['strict_ig_agreement', 'success_rate']
    curves = {key: [(r['step'], r[key]) for r in rows if r[key] is not None] for key in keys}
    plot_training_curves({k: v for k, v in curves.items() if v}, out / 'curves.png',
                         title='Expert imitation, Bayesian task-reward planning, and information gain during learning',
                         x_label='IQ-Learn gradient update')
    print(f"Saved diagnostic curves to {out}")


if __name__ == '__main__':
    main()
