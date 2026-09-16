"""Offline IQ-Learn for the partial-obs foraging task. Same objective as
iq_learn_agent.py, but the Q-head reads a RecurrentEncoder belief state
instead of the raw observation, since one egocentric frame isn't Markov.

Reuses the same SimpleQNetwork Q-head as the full-state agent, fed
rnn_hidden_dim vs. raw obs size. Only the Q-head gets a periodic
frozen target copy, matching softq.py.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from iq_learn.agent.softq_models import SimpleQNetwork
from iq_learn.iq import iq_loss
from imitation.iq_learn_agent import make_args
from imitation.recurrent_encoder import RecurrentEncoder


class RecurrentOfflineSoftQAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        args,
        latent_dim: int = 64,
        rnn_hidden_dim: int = 128,
        device: str = "cpu",
        lr: float = 3e-4,
        grid_shape: tuple[int, int, int] | None = None,
    ):
        self.args = args
        self.gamma = args.gamma
        self.device = torch.device(device)
        self.log_alpha = torch.log(torch.tensor(float(args.agent.init_temp), device=self.device))
        self.encoder = RecurrentEncoder(obs_dim, latent_dim, rnn_hidden_dim, grid_shape=grid_shape).to(self.device)
        self.q_net = SimpleQNetwork(rnn_hidden_dim, action_dim, args, device=device).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = Adam(list(self.encoder.parameters()) + list(self.q_net.parameters()), lr=lr)
        self._hidden = None  # persistent hidden state for act() during a rollout

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def getV(self, h: torch.Tensor) -> torch.Tensor:
        q = self.q_net(h)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def get_targetV(self, h: torch.Tensor) -> torch.Tensor:
        q = self.target_net(h)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def critic(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q = self.q_net(h)
        return q.gather(1, action.long())

    def encode_episode(self, states: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """states/next_states: [T, obs_dim] for one episode. Returns [T+1, H]
        hiddens: hiddens[t] is the belief after states[t], hiddens[T] is the
        belief after the final next_states[-1] (reached but never acted from)."""
        hidden = None
        hiddens = []
        for t in range(states.shape[0]):
            hidden, _ = self.encoder(states[t : t + 1], hidden)
            hiddens.append(hidden)
        final_hidden, _ = self.encoder(next_states[-1:], hidden)
        hiddens.append(final_hidden)
        return torch.cat(hiddens, dim=0)

    def _encode_batch(self, episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        """Loop over time, not (episode, time) pairs: one GRUCell call per
        step handles the whole batch at once. Looping per-episode like
        encode_episode does is ~batch_size times more overhead for nothing."""
        lengths = [ep[0].shape[0] for ep in episodes]
        max_len = max(lengths)
        batch_size = len(episodes)
        obs_dim = episodes[0][0].shape[1]
        device = episodes[0][0].device

        padded_states = torch.zeros(max_len, batch_size, obs_dim, device=device)
        for i, (states, *_rest) in enumerate(episodes):
            padded_states[: lengths[i], i] = states

        # The featurizer (CNN/MLP over one frame) has no temporal dependency,
        # so run it once on every (t, episode) pair batched together instead
        # of once per timestep in the loop below. With a CNN front-end,
        # re-running convolutions max_len separate times was the difference
        # between ~1 minute and never finishing.
        flat_obs = padded_states.reshape(max_len * batch_size, obs_dim)
        flat_latent = self.encoder._featurize(flat_obs)
        latents = flat_latent.view(max_len, batch_size, -1)

        hidden = torch.zeros(batch_size, self.encoder.rnn_hidden_dim, device=device)
        all_h = []
        for t in range(max_len):
            hidden = self.encoder.gru(latents[t], hidden)
            all_h.append(hidden)
        all_h = torch.stack(all_h, dim=0)  # [max_len, B, H]

        last_idx = torch.tensor([length - 1 for length in lengths], device=device)
        batch_arange = torch.arange(batch_size, device=device)
        hidden_before_final = all_h[last_idx, batch_arange]  # [B, H]
        final_next_obs = torch.stack([ep[1][-1] for ep in episodes], dim=0)  # [B, obs_dim]
        final_hidden, _ = self.encoder(final_next_obs, hidden_before_final)  # [B, H]

        all_curr, all_next, all_actions, all_dones = [], [], [], []
        for i, (_states, _next_states, actions, dones) in enumerate(episodes):
            length = lengths[i]
            curr = all_h[:length, i]
            if length > 1:
                nxt = torch.cat([all_h[1:length, i], final_hidden[i : i + 1]], dim=0)
            else:
                nxt = final_hidden[i : i + 1]
            all_curr.append(curr)
            all_next.append(nxt)
            all_actions.append(actions)
            all_dones.append(dones)

        return (
            torch.cat(all_curr, dim=0),
            torch.cat(all_next, dim=0),
            torch.cat(all_actions, dim=0),
            torch.cat(all_dones, dim=0),
        )

    def update(self, episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], step: int, target_update_freq: int) -> dict:
        obs, next_obs, action, done = self._encode_batch(episodes)
        next_obs = next_obs.detach()

        is_expert = torch.ones(obs.shape[0], 1, dtype=torch.bool, device=self.device)
        env_reward = torch.zeros(obs.shape[0], 1, device=self.device)
        batch = (obs, next_obs, action, env_reward, done, is_expert)

        current_Q = self.critic(obs, action)
        current_V = self.getV(obs)
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)

        loss, loss_dict = iq_loss(self, current_Q, current_V, next_V, batch)
        loss_dict["hidden_std"] = obs.std().item()
        loss_dict["q_std"] = current_Q.std().item()
        with torch.no_grad():
            all_actions_q = self.q_net(obs)
        loss_dict["q_spread_across_actions"] = (all_actions_q.max(dim=1).values - all_actions_q.min(dim=1).values).mean().item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss_dict

    def reset_hidden(self):
        self._hidden = None

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> int:
        with torch.no_grad():
            hidden, _ = self.encoder(obs.unsqueeze(0), self._hidden)
            self._hidden = hidden
            q = self.q_net(hidden)
            probs = F.softmax(q / self.alpha, dim=1)
        if deterministic:
            return int(torch.argmax(probs, dim=1).item())
        return int(Categorical(probs).sample().item())

    def save(self, path: str):
        torch.save({"encoder": self.encoder.state_dict(), "q_net": self.q_net.state_dict()}, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(state["encoder"])
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["q_net"])


__all__ = ["RecurrentOfflineSoftQAgent", "make_args"]
