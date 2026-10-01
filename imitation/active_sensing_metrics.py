"""

Both are adapted from the entropy-reduction framework in Sharafeldin, Imam
& Choi, "Active Sensing with Predictive Coding and Uncertainty Minimization"
(Patterns, 2024). That paper's Score(a) = H(belief) - E[H(belief|a)]
motivates both metrics below, adapted to this task's belief (whether the
goal has entered the agent's field of view).

1) path shape vs. shortest-path baseline: BFS over the (x, y,
direction) state space (matching the real action cost of turning plus
moving, not raw Manhattan distance) gives the true minimum action count per
episode; compare against the trained policy's actual episode length.

2) turn-action rate before vs. after the goal is known: using
uses privileged simulator access (we always know true state in simulation,
per the plan) to track a monotonic "goal_known" flag per episode has the
goal object appeared in the agent's egocentric view yet. If active sensing
emerged, the agent should turn (scan) more before the goal is known and
move more directly once it is known, mirroring the reference paper's finding
that BAS fixates the informative location once it matters.
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import torch
from minigrid.core.constants import OBJECT_TO_IDX

from environment import make_partial_obs_learning_env
from imitation.recurrent_iq_learn_agent import RecurrentOfflineSoftQAgent, make_args

GOAL_IDX = OBJECT_TO_IDX["goal"]
TURN_ACTIONS = {0, 1}  # MiniGrid Actions.left, Actions.right
DIR_TO_DELTA = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def shortest_path_length(base_env) -> int:
    """BFS over (x, y, dir) with actions {left, right, forward}, matching
    the real action cost -- not Manhattan distance."""
    grid = base_env.grid
    goal_pos = None
    for y in range(base_env.height):
        for x in range(base_env.width):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                goal_pos = (x, y)
    if goal_pos is None:
        raise RuntimeError("No goal tile found in grid.")

    def passable(x, y):
        if not (0 <= x < base_env.width and 0 <= y < base_env.height):
            return False
        cell = grid.get(x, y)
        return cell is None or cell.can_overlap()

    start = (int(base_env.agent_pos[0]), int(base_env.agent_pos[1]), int(base_env.agent_dir))
    if (start[0], start[1]) == goal_pos:
        return 0

    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (x, y, d), dist = queue.popleft()
        neighbors = [(x, y, (d - 1) % 4), (x, y, (d + 1) % 4)]
        dx, dy = DIR_TO_DELTA[d]
        nx, ny = x + dx, y + dy
        if passable(nx, ny):
            neighbors.append((nx, ny, d))
        for state in neighbors:
            if state in visited:
                continue
            if (state[0], state[1]) == goal_pos:
                return dist + 1
            visited.add(state)
            queue.append((state, dist + 1))
    raise RuntimeError("Goal unreachable -- BFS exhausted without finding it.")


def run_episode(agent: RecurrentOfflineSoftQAgent, env, device: str):
    obs, _ = env.reset()
    base_env = env.unwrapped
    bfs_len = shortest_path_length(base_env)

    agent.reset_hidden()
    goal_known = (base_env.gen_obs()["image"][:, :, 0] == GOAL_IDX).any()
    terminated = truncated = False
    steps = 0
    turns_before, turns_after = 0, 0
    steps_before, steps_after = 0, 0
    episode_reward = 0.0

    while not (terminated or truncated):
        action = agent.act(torch.as_tensor(obs, dtype=torch.float32, device=device), deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        episode_reward += reward

        if goal_known:
            steps_after += 1
            turns_after += int(action in TURN_ACTIONS)
        else:
            steps_before += 1
            turns_before += int(action in TURN_ACTIONS)
            if (base_env.gen_obs()["image"][:, :, 0] == GOAL_IDX).any():
                goal_known = True

    return {
        "success": episode_reward > 0,
        "steps": steps,
        "bfs_len": bfs_len,
        "turns_before": turns_before,
        "steps_before": steps_before,
        "turns_after": turns_after,
        "steps_after": steps_after,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 4: active-sensing metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--grid-shape", type=int, nargs=3, default=[7, 7, 3])
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--rnn-hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_partial_obs_learning_env()
    obs_dim = env.observation_space.shape[0]
    agent = RecurrentOfflineSoftQAgent(
        obs_dim=obs_dim,
        action_dim=7,
        args=make_args(),
        latent_dim=args.latent_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        device=device,
        grid_shape=tuple(args.grid_shape),
    )
    agent.load(args.checkpoint)

    results = [run_episode(agent, env, device) for _ in range(args.episodes)]
    env.close()

    successes = [r for r in results if r["success"]]
    print(f"Episodes: {args.episodes}, successes: {len(successes)} ({len(successes)/args.episodes:.1%})")

    print("\n--- Metric 1: path shape vs. shortest-path baseline ---")
    ratios = [r["steps"] / r["bfs_len"] for r in successes if r["bfs_len"] > 0]
    print(f"Mean actual/shortest-path ratio: {np.mean(ratios):.2f} (1.0 = optimal, n={len(ratios)})")
    print(f"Median ratio: {np.median(ratios):.2f}")
    print(f"Mean BFS shortest length: {np.mean([r['bfs_len'] for r in successes]):.1f}")
    print(f"Mean actual length: {np.mean([r['steps'] for r in successes]):.1f}")

    print("\n--- Metric 2: turn-action rate, before vs. after goal is known ---")
    tb = sum(r["turns_before"] for r in successes)
    sb = sum(r["steps_before"] for r in successes)
    ta = sum(r["turns_after"] for r in successes)
    sa = sum(r["steps_after"] for r in successes)
    rate_before = tb / sb if sb else float("nan")
    rate_after = ta / sa if sa else float("nan")
    print(f"Turn-action rate before goal known: {rate_before:.1%} ({sb} steps across successes)")
    print(f"Turn-action rate after goal known:  {rate_after:.1%} ({sa} steps across successes)")
    print(f"Ratio (before/after): {rate_before / rate_after:.2f}" if rate_after else "after-rate undefined")


if __name__ == "__main__":
    main()
