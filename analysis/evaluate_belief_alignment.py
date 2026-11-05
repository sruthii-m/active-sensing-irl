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
)
from analysis.belief_reward_planner import BeliefRewardPlanner


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
    parser.add_argument('--output', default='artifacts/analysis/alignment.csv')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    horizons = args.reward_horizons or []
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
            goal = next((x, y) for x in range(env.unwrapped.width)
                        for y in range(env.unwrapped.height)
                        if (o := env.unwrapped.grid.get(x, y)) is not None and o.type == 'goal')
            while not done:
                belief.observe(env)
                hidden = representation(agent, obs, hidden)
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
                        metrics = reward_action_alignment(values, reward_action)
                        prefix = f'reward_h{horizon}_'
                        for key, value in metrics.items():
                            row[prefix + key] = json.dumps(value) if key == 'maximizers' else value
                        for a, value in enumerate(values):
                            row[f'{prefix}q_{a}'] = float(value)
                rows.append(row)
                obs, reward, terminated, truncated, _ = env.step(reward_action)
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
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
