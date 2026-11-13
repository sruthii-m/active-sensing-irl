"""Reward-policy rollouts with an external exact Bayesian comparator."""
import argparse
import csv
import json
from pathlib import Path

import torch

from environment import make_partial_obs_learning_env
from analysis.discrete_goal_belief import DiscreteGoalBelief, select_action, ACTIONS
from analysis.policy_io import add_policy_args, load_policy, representation
from analysis.alignment_metrics import (
    action_alignment, reward_action_alignment, summarize, summarize_reward_alignment,
    information_value_alignment, summarize_information_value,
)
from analysis.belief_reward_planner import BeliefRewardPlanner



def plot_information_value(summary, output):
    """Plot descriptive evidence and denominators; no causal or monotonicity claim."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = summary['information_value']['by_horizon']
    fig, axes = plt.subplots(len(results), 3, figsize=(16, 4 * len(results)), squeeze=False,
                             layout='constrained')
    for panel, (horizon, result) in zip(axes, results.items()):
        groups = list(result['unresolved'].values())
        x = np.arange(2)
        for offset, key, label in [(-.18, 'strict_ig_agreement', 'Actual agreement'),
                                   (.18, 'uncertainty_binned_action_frequency_reference', 'Action-frequency reference')]:
            values = [g[key]['mean'] if isinstance(g[key], dict) else g[key] for g in groups]
            panel[0].bar(x+offset, [np.nan if v is None else v for v in values], .36, label=label)
        panel[0].set_xticks(x, [f'{label}\nn={g["strict_ig_agreement"]["steps"]}'
                               for label, g in zip(('Positive value', 'Zero value'), groups)])
        panel[0].set(ylabel='Strict IG agreement', ylim=(0,1), title=f'Horizon {horizon}')
        panel[0].legend(fontsize=8)
        bins = list(result['by_candidate_count'])
        for label, name in [('Positive value','positive_visual_value'), ('Zero value','zero_visual_value')]:
            metrics = [result['by_candidate_count'][b][name]['strict_ig_agreement'] for b in bins]
            y = [np.nan if m['mean'] is None else m['mean'] for m in metrics]
            panel[1].plot(range(len(bins)), y, 'o-', label=label)
            for i, m in enumerate(metrics):
                if m['mean'] is not None:
                    panel[1].annotate(str(m['steps']), (i,m['mean']), fontsize=7)
        panel[1].set_xticks(range(len(bins)), bins, rotation=25)
        panel[1].set(ylabel='Strict IG agreement', xlabel='Candidate goal cells (numbers show steps)', ylim=(0,1))
        panel[1].legend(fontsize=8)
        metrics = [result['by_candidate_count'][b]['positive_visual_value']['controlled_q_value_correlation'] for b in bins]
        panel[2].plot(range(len(bins)), [np.nan if m['mean'] is None else m['mean'] for m in metrics], 'o-')
        panel[2].axhline(0, color='gray', linestyle='--')
        panel[2].set_xticks(range(len(bins)), bins, rotation=25)
        panel[2].set(ylabel='Mean residual Q–information-value correlation', xlabel='Candidate goal cells', ylim=(-1,1))
    fig.suptitle('Visual information value: exploratory association on fixed-policy rollouts')
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--view-size', type=int, default=7)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--reward-horizons', type=int, nargs='+',
                        help='Also compare with Bayesian task-reward planning at these positive action horizons.')
    parser.add_argument('--information-value', action='store_true', help='Compare reward planning with versus without future visual evidence; requires --reward-horizons.')
    parser.add_argument('--output', default='artifacts/analysis/alignment.csv')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    horizons = args.reward_horizons or []
    if args.information_value and not horizons:
        parser.error("--information-value requires --reward-horizons")
    if any(h < 1 for h in horizons) or len(set(horizons)) != len(horizons):
        parser.error('--reward-horizons must contain distinct positive integers')
    env = make_partial_obs_learning_env(size=args.size, num_distractors=0,
                                        agent_view_size=args.view_size, max_steps=args.max_steps)
    agent = load_policy(args, env.observation_space.shape[0])
    if not hasattr(agent, 'encoder'):
        raise ValueError('Alignment requires a recurrent partial-observation checkpoint.')
    planner = (BeliefRewardPlanner(size=args.size, view_size=args.view_size,
                                  max_steps=env.unwrapped.max_steps, gamma=args.gamma)
               if horizons else None)
    rows, successes = [], 0
    with torch.no_grad():
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            belief = DiscreteGoalBelief.from_env(env)
            hidden, done, timestep = None, False, 0
            previous_action = -1
            goal = next((x, y) for x in range(env.unwrapped.width)
                        for y in range(env.unwrapped.height)
                        if (o := env.unwrapped.grid.get(x, y)) is not None and o.type == 'goal')
            while not done:
                belief.observe(env)
                hidden = representation(agent, obs, hidden, previous_action)
                q = agent.q_net(hidden)[0].numpy()
                scores, new = belief.action_scores(env)
                ig_action, ties = select_action(scores)
                alignment = action_alignment(q, scores, ig_action, ties)
                reward_action = alignment['executed_action']  # seven-action policy
                row = dict(episode=episode, seed=args.seed + episode, timestep=timestep,
                           agent_x=int(env.unwrapped.agent_pos[0]), agent_y=int(env.unwrapped.agent_pos[1]),
                           direction=int(env.unwrapped.agent_dir), goal_x=goal[0], goal_y=goal[1],
                           goal_seen=belief.goal_seen, candidates=len(belief.candidates),
                           entropy_nats=belief.entropy, resolved=belief.resolved,
                           ig_argmax=ig_action,
                           **alignment)
                for a in range(len(q)):
                    row[f'q_{a}'] = float(q[a])
                for a in ACTIONS:
                    row[f'ig_{a}'] = scores[a]
                    row[f'new_cells_{a}'] = json.dumps(sorted(new[a]))
                if planner is not None:
                    row['belief_candidates'] = json.dumps(sorted(belief.candidates))
                    pose = (row['agent_x'], row['agent_y'], row['direction'])
                    for horizon in horizons:
                        values = planner.q_values(pose, belief.candidates,
                                                  env.unwrapped.step_count, horizon)
                        if args.information_value:
                            withheld = planner.q_values(pose, belief.candidates, env.unwrapped.step_count,
                                                        horizon, observe=False)
                            immediate = planner.q_values(pose, belief.candidates, env.unwrapped.step_count, 1)
                            diagnostic = information_value_alignment(q, values, withheld, immediate)
                            row.update({f'information_h{horizon}_{k}': v for k, v in diagnostic.items()})
                        metrics = reward_action_alignment(values, reward_action)
                        prefix = f'reward_h{horizon}_'
                        for key, value in metrics.items():
                            row[prefix + key] = json.dumps(value) if key == 'maximizers' else value
                        for a, value in enumerate(values):
                            row[f'{prefix}q_{a}'] = float(value)
                rows.append(row)
                obs, reward, terminated, truncated, _ = env.step(reward_action)
                previous_action = reward_action
                done = terminated or truncated
                successes += int(reward > 0)
                timestep += 1
    env.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(schema_version=2,
                   primary_metric='unresolved.strict_agreement_unique_ig',
                   config=vars(args), success_rate=successes / args.episodes, **summarize(rows))
    if planner is not None:
        summary['reward_planning'] = dict(
            horizons=horizons, gamma=planner.gamma, max_steps=planner.max_steps,
            tie_tolerance=1e-10, action_space=list(range(7)),
            objective='Expected discounted MiniGrid goal-entry reward (1 - 0.9 * arrival_step / max_steps); no information-gain bonus.',
            approximation='Exact recursion within each finite horizon; zero continuation value beyond horizon. Known empty geometry and pose; only the observed goal posterior is supplied, not the hidden goal.',
            aggregation='Step-weighted means within overall, unresolved, and resolved groups. Unique-winner agreement excludes all ties; maximizer membership and informative regret exclude states tied across all seven actions. Empty denominators are null.',
            by_horizon=summarize_reward_alignment(rows, horizons))
    if args.information_value:
        summary['information_value'] = dict(
            definition='Q with future visual evidence minus Q withholding all future visual evidence, conditional on the same first action. Goal-entry reward, survival feedback, motion, discount, deadline and horizon are identical.',
            limitations='Finite-horizon visual information value, not total information value or the value of only the next observation. Zero can reflect a short horizon. Associations compare policy-dependent histories; not causal evidence. Bootstrap intervals condition on this policy and resample episodes. Frequency-reference excess has no interval here. Controlled correlation removes linear withheld-Q and immediate-reward effects across seven actions; it does not control all confounds.',
            tolerance=1e-10, by_horizon=summarize_information_value(rows, horizons))
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    if args.information_value:
        plot_information_value(summary, out.with_suffix('.information_value.png'))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
