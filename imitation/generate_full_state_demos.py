"""Roll out the trained full-state expert and save trajectories in the format
iq_learn/dataset/expert_dataset.py expects, for train_iq.py's `demo:` config.

Gate before moving to step 3: a policy trained against the recovered reward
reproduces expert performance (reward correlation alone is a weak check given
this task's sparse terminal reward).
"""

from __future__ import annotations


def generate(model_path: str, env_kwargs: dict, num_episodes: int, out_path: str):
    raise NotImplementedError
