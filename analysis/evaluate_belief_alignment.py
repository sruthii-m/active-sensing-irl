"""Reward-policy rollouts with an external exact Bayesian comparator."""
import argparse
import csv
import json
from pathlib import Path

import torch

from environment import make_partial_obs_learning_env
from analysis.discrete_goal_belief import DiscreteGoalBelief, select_action, ACTIONS
from analysis.policy_io import add_policy_args, load_policy, representation
from analysis.alignment_metrics import action_alignment, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--view-size', type=int, default=7)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--output', default='artifacts/analysis/alignment.csv')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    env = make_partial_obs_learning_env(size=args.size, num_distractors=0,
                                        agent_view_size=args.view_size, max_steps=args.max_steps)
    agent = load_policy(args, env.observation_space.shape[0])
    if not hasattr(agent, 'encoder'):
        raise ValueError('Alignment requires a recurrent partial-observation checkpoint.')
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
                reward_action = int(q.argmax())  # seven-action policy
                row = dict(episode=episode, seed=args.seed + episode, timestep=timestep,
                           agent_x=int(env.unwrapped.agent_pos[0]), agent_y=int(env.unwrapped.agent_pos[1]),
                           direction=int(env.unwrapped.agent_dir), goal_x=goal[0], goal_y=goal[1],
                           goal_seen=belief.goal_seen, candidates=len(belief.candidates),
                           entropy_nats=belief.entropy, resolved=belief.resolved,
                           ig_argmax=ig_action,
                           ig_tie=len(ties) > 1, ig_all_tied=len(ties) == len(ACTIONS),
                           **action_alignment(q, scores, ig_action, ties),
                           executed_action=reward_action)
                for a in range(len(q)):
                    row[f'q_{a}'] = float(q[a])
                for a in ACTIONS:
                    row[f'ig_{a}'] = scores[a]
                    row[f'new_cells_{a}'] = json.dumps(sorted(new[a]))
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
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
