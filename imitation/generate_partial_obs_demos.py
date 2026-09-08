"""Roll out the partial-observation expert and save IQ-Learn trajectories.

The serialization format matches :mod:`generate_full_state_demos`.
"""

from __future__ import annotations


def generate(model_path: str, env_kwargs: dict, num_episodes: int, out_path: str):
    raise NotImplementedError
