"""Auxiliary head z_t -> reward-relevant features (e.g. food location),
trained with privileged full-state labels available only at train time.
Train-time-only: never feed this into the partial-obs policy's inputs, or it
leaks full state through the back door.
"""

from __future__ import annotations

import torch.nn as nn


class RewardRelevantAuxHead(nn.Module):
    def __init__(self, latent_dim: int, feature_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, z):
        raise NotImplementedError


def evaluate_sufficiency(model, dataloader) -> dict:
    """Accuracy of aux head predictions against ground-truth reward-relevant
    features — the actual validation gate, not world-model prediction loss."""
    raise NotImplementedError
