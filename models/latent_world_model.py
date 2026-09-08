"""Action-conditioned VAE-style latent world model, replacing the plain
encoder in imitation/recurrent_encoder.py: posterior q(z_t | h_t),
transition p(z_{t+1} | z_t, a_t). A stochastic latent gives calibrated entropy
for free, unlike a deterministic embedding.

Target is reward-sufficiency, not prediction quality:
    r(s_t, a_t) ~= E[r(s_t, a_t) | z_t], small residual variance
Validate with aux_head.py rather than assuming good frame prediction implies a
good IRL substrate.
"""

from __future__ import annotations

import torch.nn as nn


class LatentWorldModel(nn.Module):
    def __init__(self, obs_shape, action_dim, latent_dim: int):
        super().__init__()
        raise NotImplementedError

    def encode(self, obs, hidden_state):
        raise NotImplementedError

    def predict_next(self, z_t, action):
        raise NotImplementedError
