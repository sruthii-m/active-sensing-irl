"""Jointly-trained encoder (CNN/MLP over the symbolic grid, not frozen
pretrained) + GRU/LSTM feeding iq_learn's Q-network (see
iq_learn/agent/softq_models.py for the interface it must match) — the Q-function
needs observation history, not single frames.

Training this off IQ-Learn's replay buffer inherits recurrent-off-policy issues
(stale hidden states, burn-in, sequence-chunked sampling). Validate the
recurrent Q-network on a partially-observed but memory-solvable toy task
(exercises the recurrence) before trusting it on the full pipeline.
"""

from __future__ import annotations

import torch.nn as nn


class RecurrentEncoder(nn.Module):
    """Encoder + GRU/LSTM producing a history-conditioned latent for the Q-head."""

    def __init__(self, obs_shape, latent_dim: int, rnn_hidden_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, obs, hidden_state):
        raise NotImplementedError
