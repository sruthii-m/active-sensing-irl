"""Exact finite-horizon reward planning for a uniform goal posterior in an empty room.

Known map and pose, deterministic MiniGrid observations, seven actions, no IG
bonus. Zero terminal value at the search horizon: NOT an infinite-horizon solver.
The actual hidden goal is never supplied to this planner.
"""
from functools import lru_cache

import numpy as np
from minigrid.core.grid import Grid

from analysis.discrete_goal_belief import visible_cells
from environment.forage_env import make_forage_env


class BeliefRewardPlanner:
    def __init__(self, size=10, view_size=3, max_steps=400, gamma=.99):
        if size < 4 or max_steps < 1 or not 0 < gamma <= 1:
            raise ValueError('Invalid room size, deadline, or discount.')
        self.size, self.max_steps, self.gamma = size, max_steps, gamma
        self.cells = tuple((x, y) for x in range(1, size - 1) for y in range(1, size - 1))
        self.index = {p: i for i, p in enumerate(self.cells)}
        # Construct goal-free geometry. Visibility is identical for empty cells
        # and the transparent, traversable goal; no real episode is inspected.
        env = make_forage_env(size=size, agent_view_size=view_size, num_distractors=0)
        env.reset(seed=0)
        env.grid = Grid(size, size)
        env.grid.wall_rect(0, 0, size, size)
        self.visibility = {}
        try:
            for x, y in self.cells:
                for direction in range(4):
                    env.agent_pos, env.agent_dir = (x, y), direction
                    self.visibility[x, y, direction] = frozenset(
                        self.index[p] for p in visible_cells(env) if p in self.index)
        finally:
            env.close()
        self.value = lru_cache(maxsize=200000)(self._value)

    def transition(self, pose, action):
        x, y, direction = pose
        if action == 0:
            return x, y, (direction - 1) % 4
        if action == 1:
            return x, y, (direction + 1) % 4
        if action == 2:
            dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[direction]
            if (x + dx, y + dy) in self.index:
                return x + dx, y + dy, direction
        return pose

    @staticmethod
    def turns(a, b):
        return min((a - b) % 4, (b - a) % 4)

    def goal_distance(self, pose, goal):
        x, y, direction = pose
        dx, dy = goal[0] - x, goal[1] - y
        segments = []
        if dx:
            segments.append((0 if dx > 0 else 2, abs(dx)))
        if dy:
            segments.append((1 if dy > 0 else 3, abs(dy)))
        if not segments:
            return 0
        costs = []
        for order in (segments, list(reversed(segments))):
            heading, cost = direction, 0
            for next_heading, distance in order:
                cost += self.turns(heading, next_heading) + distance
                heading = next_heading
            costs.append(cost)
        return min(costs)

    def known_goal_value(self, pose, goal, elapsed, horizon):
        distance = self.goal_distance(pose, goal)
        if not 1 <= distance <= min(horizon, self.max_steps - elapsed):
            return 0.
        return self.gamma ** (distance - 1) * (1 - .9 * (elapsed + distance) / self.max_steps)

    def _action_value(self, pose, candidates, elapsed, horizon, action, observe=True):
        if horizon <= 0 or elapsed >= self.max_steps:
            return 0.
        nxt = self.transition(pose, action)
        next_time = elapsed + 1
        remaining = set(candidates)
        total = 0.
        cell = self.index[nxt[:2]]
        if action == 2 and nxt[:2] != pose[:2] and cell in remaining:
            total += 1 - .9 * next_time / self.max_steps
            remaining.remove(cell)  # Goal entry terminates before continuation.
        if horizon > 1 and next_time < self.max_steps:
            revealed = remaining & self.visibility[nxt] if observe else set()
            for goal in revealed:
                total += self.gamma * self.known_goal_value(nxt, self.cells[goal], next_time, horizon - 1)
            unseen = tuple(sorted(remaining - revealed))
            if unseen:
                total += self.gamma * len(unseen) * self.value(nxt, unseen, next_time, horizon - 1, observe)
        return total / len(candidates)

    def _value(self, pose, candidates, elapsed, horizon, observe=True):
        if len(candidates) == 1:
            return self.known_goal_value(pose, self.cells[candidates[0]], elapsed, horizon)
        # Actions 3..6 are identical no-ops in this environment.
        return max(self._action_value(pose, candidates, elapsed, horizon, a, observe) for a in range(4))

    def q_values(self, pose, candidates, elapsed, horizon, observe=True):
        """With observe=False, withhold future visual evidence, retaining goal-entry/survival feedback."""
        pose = tuple(int(v) for v in pose)
        candidates = tuple(sorted({self.index[tuple(p)] for p in candidates}))
        if not candidates or self.index.get(pose[:2]) in candidates:
            raise ValueError('Need nonempty live belief with no goal under the agent.')
        if len(pose) != 3 or pose[2] not in range(4) or pose[:2] not in self.index:
            raise ValueError('Invalid pose.')
        if horizon < 1 or elapsed < 0:
            raise ValueError('Horizon must be positive and elapsed time nonnegative.')
        values = [self._action_value(pose, candidates, elapsed, horizon, a, observe) for a in range(4)]
        return np.array(values[:3] + [values[3]] * 4)


def maximizers(values, tolerance=1e-10):
    values = np.asarray(values)
    return tuple(np.flatnonzero(np.abs(values - values.max()) <= tolerance).tolist())
