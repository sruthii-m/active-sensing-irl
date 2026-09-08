"""Full-state and partial-obs wrappers around the same ForageEnv, so the two
experts train on identical episodes and only observability differs."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from minigrid.core.world_object import Goal
from minigrid.wrappers import FullyObsWrapper

from .forage_env import make_forage_env


def make_full_obs(**env_kwargs):
    return FullyObsWrapper(make_forage_env(**env_kwargs))


def make_partial_obs(**env_kwargs):
    return make_forage_env(**env_kwargs)


class SymbolicGridVectorWrapper(gym.ObservationWrapper):
    """Convert MiniGrid's symbolic image and direction into a small vector.

    This excludes the mission text and any convenience state features.  It lets
    the full and partial experts use SB3's MLP policies while differing only in
    whether the image is global or egocentric.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        image_space = env.observation_space["image"]
        vector_size = int(np.prod(image_space.shape)) + 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(vector_size,), dtype=np.float32
        )

    def observation(self, observation):
        image = np.asarray(observation["image"], dtype=np.float32).reshape(-1) / 10.0
        direction = np.asarray([observation["direction"] / 3.0], dtype=np.float32)
        return np.concatenate((image, direction), dtype=np.float32)


class FullStateFeatureWrapper(gym.ObservationWrapper):
    """Privileged compact state for the full-observation learnability gate.

    The baseline receives agent position/direction and goal position, all
    normalized to ``[0, 1]``.  It is intentionally used *only* for the
    full-state PPO gate; the partial expert never receives these features.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

    def observation(self, observation):
        del observation  # This view is derived from the privileged simulator state.
        base_env = self.unwrapped
        agent_x, agent_y = base_env.agent_pos
        goal_x = goal_y = None
        for y in range(base_env.height):
            for x in range(base_env.width):
                if isinstance(base_env.grid.get(x, y), Goal):
                    goal_x, goal_y = x, y
                    break
            if goal_x is not None:
                break
        if goal_x is None:
            raise RuntimeError("ForageEnv reset without a goal tile.")
        scale = base_env.width - 1
        return np.asarray(
            [agent_x / scale, agent_y / scale, base_env.agent_dir / 3.0, goal_x / scale, goal_y / scale],
            dtype=np.float32,
        )


def make_full_obs_learning_env(**env_kwargs):
    """Privileged-state environment for the full-state PPO baseline."""
    return FullStateFeatureWrapper(make_forage_env(**env_kwargs))


def make_partial_obs_learning_env(**env_kwargs):
    """Egocentric vector environment for the recurrent expert."""
    return SymbolicGridVectorWrapper(make_partial_obs(**env_kwargs))
