"""Exact external goal-location posterior for the empty, known Forage room.

Pose/map are known through simulator access; only goal location is uncertain.
All entropy quantities exposed here are nats; generic baseline uses bits.
"""
from copy import deepcopy
from math import log

from analysis.entropy_score_baseline import categorical_entropy, entropy_score

ACTIONS = (0, 1, 2)  # MiniGrid left, right, forward


def visible_cells(env):
    """World cells selected by MiniGrid's actual occlusion mask."""
    env = env.unwrapped
    _, mask = env.gen_obs_grid()
    cells = set()
    for x in range(env.width):
        for y in range(env.height):
            vx, vy = env.get_view_coords(x, y)
            if 0 <= vx < mask.shape[0] and 0 <= vy < mask.shape[1] and mask[vx, vy]:
                cells.add((x, y))
    return cells


class DiscreteGoalBelief:
    def __init__(self, candidates):
        self.candidates = set(candidates)
        if not self.candidates:
            raise ValueError("Need at least one candidate.")
        self.seen = set()
        self.goal_seen = False

    @classmethod
    def from_env(cls, env):
        env = env.unwrapped
        if env.num_distractors != 0 or env.step_count != 0:
            raise ValueError("Initialize at reset with zero distractors.")
        return cls((x, y) for x in range(1, env.width - 1)
                   for y in range(1, env.height - 1) if (x, y) != tuple(env.agent_pos))

    @property
    def distribution(self):
        return {cell: 1 / len(self.candidates) for cell in self.candidates}

    @property
    def entropy(self):
        return categorical_entropy(self.distribution) * log(2)

    @property
    def resolved(self):
        return len(self.candidates) == 1

    def update(self, visible, goal=None):
        visible = set(visible)
        if goal is not None:
            if goal not in visible or goal not in self.candidates:
                raise ValueError("Goal observation contradicts belief or visibility.")
            remaining = {goal}
        else:
            remaining = self.candidates - visible
        if not remaining:
            raise ValueError("Observation eliminates all goal candidates.")
        self.candidates = remaining
        self.seen.update(visible)
        self.goal_seen |= goal is not None

    def observe(self, env):
        cells = visible_cells(env)
        # gen_obs_grid hides the tile under the agent. Goal entry is terminal
        # evidence; during live episodes the goal cannot be under the agent.
        goals = [p for p in cells if (obj := env.unwrapped.grid.get(*p)) is not None
                 and obj.type == "goal"]
        self.update(cells, goals[0] if goals else None)

    def information_gain(self, revealed):
        n = len(self.candidates)
        k = len(self.candidates & set(revealed))
        remaining = n - k
        return log(n) - (remaining / n * log(remaining) if remaining else 0.0)

    def enumerated_information_gain(self, revealed):
        """Reference enumeration using the existing generic entropy baseline."""
        found = self.candidates & set(revealed)
        rest = self.candidates - found
        outcomes = [(1 / len(self.candidates), {g: 1.0}) for g in found]
        if rest:
            outcomes.append((len(rest) / len(self.candidates), {g: 1 for g in rest}))
        return entropy_score(self.distribution, None, lambda *_: outcomes) * log(2)

    def action_scores(self, env):
        if env.unwrapped.num_distractors:
            raise ValueError("Exact geometry baseline requires zero distractors.")
        scores, newly_visible = {}, {}
        for action in ACTIONS:
            hypothetical = deepcopy(env.unwrapped)
            hypothetical.render_mode = None
            hypothetical.step(action)
            # Empty cells and Goal both overlap and transmit sight, so geometry
            # is independent of the hidden goal.
            cells = visible_cells(hypothetical) - self.seen
            newly_visible[action] = cells
            scores[action] = self.information_gain(cells)
        return scores, newly_visible


def select_action(scores, atol=1e-10):
    """Return deterministic first maximizer and the entire maximizing set."""
    best = max(scores.values())
    tied = tuple(a for a in sorted(scores) if abs(scores[a] - best) <= atol)
    return tied[0], tied
