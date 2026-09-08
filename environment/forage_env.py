"""Single-goal foraging task on MiniGrid: one goal object at a random position
per episode, plus a few reward-irrelevant distractor objects (different
colors/shapes) so downstream metrics can distinguish selective from generic
information-seeking.
"""

from __future__ import annotations

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Goal
from minigrid.minigrid_env import MiniGridEnv

_DISTRACTOR_SHAPES = (Ball, Box)
_DISTRACTOR_COLORS = ("red", "blue", "purple", "yellow", "grey")


class ForageEnv(MiniGridEnv):
    """Empty room, one Goal tile, `num_distractors` colored objects scattered
    at random empty cells. Episode ends on reaching the goal or timeout."""

    def __init__(self, size: int = 10, num_distractors: int = 3, max_steps: int | None = None, **kwargs):
        self.num_distractors = num_distractors
        mission_space = MissionSpace(mission_func=lambda: "get to the green goal square")
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps or 4 * size**2,
            see_through_walls=False,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.place_agent()
        self.place_obj(Goal())

        rng = self.np_random
        for _ in range(self.num_distractors):
            shape = _DISTRACTOR_SHAPES[rng.integers(len(_DISTRACTOR_SHAPES))]
            color = _DISTRACTOR_COLORS[rng.integers(len(_DISTRACTOR_COLORS))]
            self.place_obj(shape(color))

        self.mission = "get to the green goal square"


def make_forage_env(*, size: int = 10, num_distractors: int = 3, render_mode: str | None = None) -> ForageEnv:
    return ForageEnv(size=size, num_distractors=num_distractors, render_mode=render_mode)


def fraction_goal_initially_visible(num_episodes: int = 1000, **env_kwargs) -> float:
    """Fraction of resets where the goal is inside the agent's initial partial
    view. Low values mean the partial-obs expert never faces real uncertainty
    at the start of an episode — check this before trusting step 1."""
    from minigrid.core.constants import OBJECT_TO_IDX

    env = make_forage_env(**env_kwargs)
    goal_idx = OBJECT_TO_IDX["goal"]
    visible = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        if (obs["image"][:, :, 0] == goal_idx).any():
            visible += 1
    env.close()
    return visible / num_episodes
