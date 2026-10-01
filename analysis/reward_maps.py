"""History-conditioned counterfactual rewards, aggregated at visited poses.

Each action branches the real simulator and the current recurrent history.
Terminal bootstrap is zero; time-limit truncation still bootstraps, matching
recorded demonstration dones and IQ loss. Unvisited cells remain missing.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment import make_full_obs_learning_env, make_partial_obs_learning_env
from analysis.policy_io import add_policy_args, load_policy, representation


@torch.no_grad()
def counterfactual_rewards(agent, env, h):
    q = agent.q_net(h)[0]
    rewards = []
    for action in range(env.action_space.n):
        branch = deepcopy(env)
        branch.unwrapped.render_mode = None
        obs, _, terminated, _, _ = branch.step(action)
        nxt = representation(agent, obs, h)
        value = agent.getV(nxt)[0, 0]
        rewards.append(float(q[action] - agent.gamma * (not terminated) * value))
    return np.asarray(rewards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument('--env', choices=['full_state', 'partial_obs'], required=True)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--view-size', type=int, default=7)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--output', default='artifacts/analysis/reward_map')
    args = parser.parse_args()
    factory = make_partial_obs_learning_env if args.env == 'partial_obs' else make_full_obs_learning_env
    env = factory(size=args.size, num_distractors=0, agent_view_size=args.view_size, max_steps=args.max_steps)
    obs, _ = env.reset(seed=args.seed)
    agent = load_policy(args, env.observation_space.shape[0])
    count = np.zeros((args.size, args.size), dtype=int)
    total = np.zeros((7, args.size, args.size))
    goal = next((x, y) for x in range(args.size) for y in range(args.size)
                if (o := env.unwrapped.grid.get(x, y)) is not None and o.type == 'goal')
    hidden, done = None, False
    samples = []
    with torch.no_grad():
        while not done:
            hidden = representation(agent, obs, hidden)
            rewards = counterfactual_rewards(agent, env, hidden)
            x, y = env.unwrapped.agent_pos
            total[:, y, x] += rewards
            count[y, x] += 1
            samples.append([int(x), int(y), int(env.unwrapped.agent_dir), *rewards.tolist()])
            action = int(agent.q_net(hidden).argmax())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
    maps = np.full_like(total, np.nan)
    np.divide(total, count[None], out=maps, where=count[None] > 0)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path) + '.npz', reward_by_action=maps, occupancy=count, goal=goal,
             samples=np.asarray(samples), config=json.dumps(vars(args)))
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    panels = [count, maps.mean(axis=0), *maps]
    labels = ['Occupancy', 'Mean over all 7 actions', 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
    lo, hi = float(np.nanmin(maps)), float(np.nanmax(maps))
    for i, (ax, panel, label) in enumerate(zip(axes.flat, panels, labels)):
        im = ax.imshow(panel, origin='upper', **({} if i == 0 else dict(vmin=lo, vmax=hi, cmap='coolwarm')))
        ax.scatter(*goal, marker='*', s=140, c='lime', edgecolors='black', label='goal')
        ax.set_title(label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, shrink=0.65)
    fig.suptitle('Recovered rewards at visited histories; blank cells unsampled')
    fig.savefig(str(path) + '.png', dpi=160)
    plt.close(fig)
    print(f'Saved {path}.png and {path}.npz')


if __name__ == '__main__':
    main()
