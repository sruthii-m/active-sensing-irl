"""Encoder (CNN/MLP over the grid, trained jointly, not frozen) + GRU feeding
iq_learn's Q-network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentEncoder(nn.Module):
    """CNN/MLP + GRUCell producing a history-conditioned latent for the Q-head.

    If grid_shape is set, reshapes the leading part of obs to (H, W, C) and
    runs it through a small CNN before the GRU.
    Any leftover scalar features (e.g. direction) get concatenated in after.
    """

    def __init__(self, obs_shape, latent_dim: int, rnn_hidden_dim: int, grid_shape: tuple[int, int, int] | None = None):
        super().__init__()
        obs_dim = obs_shape[0] if isinstance(obs_shape, (tuple, list)) else obs_shape
        self.rnn_hidden_dim = rnn_hidden_dim
        self.grid_shape = grid_shape

        if grid_shape is not None:
            h, w, c = grid_shape
            self.grid_flat_dim = h * w * c
            self.extra_dim = obs_dim - self.grid_flat_dim
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            cnn_out_dim = 32 * h * w
            self.mlp = nn.Sequential(nn.Linear(cnn_out_dim + self.extra_dim, latent_dim), nn.ReLU())
        else:
            self.grid_flat_dim = None
            self.mlp = nn.Sequential(nn.Linear(obs_dim, latent_dim), nn.ReLU())

        self.gru = nn.GRUCell(latent_dim, rnn_hidden_dim)

    def _featurize(self, obs: torch.Tensor) -> torch.Tensor:
        if self.grid_shape is None:
            return self.mlp(obs)
        h, w, c = self.grid_shape
        grid = obs[:, : self.grid_flat_dim].view(-1, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        conv_out = self.cnn(grid)
        if self.extra_dim > 0:
            conv_out = torch.cat([conv_out, obs[:, self.grid_flat_dim :]], dim=1)
        return self.mlp(conv_out)

    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor | None):
        latent = self._featurize(obs)
        if hidden_state is None:
            hidden_state = torch.zeros(obs.shape[0], self.rnn_hidden_dim, device=obs.device)
        new_hidden = self.gru(latent, hidden_state)
        return new_hidden, new_hidden
