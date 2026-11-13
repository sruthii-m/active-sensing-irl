"""Offline IQ-Learn for the partial-obs foraging task.

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
        condition_on_action: bool = True,
        terminal_semantics: str = "recorded",
    ):
        self.args = args
        self.gamma = args.gamma
        self.device = torch.device(device)
        self.log_alpha = torch.log(torch.tensor(float(args.agent.init_temp), device=self.device))
        self.config = dict(obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim,
                           rnn_hidden_dim=rnn_hidden_dim, grid_shape=grid_shape,
                           condition_on_action=condition_on_action, terminal_semantics=terminal_semantics,
                           gamma=float(args.gamma), init_temp=float(args.agent.init_temp),
                           action_convention='previous_action_reset_minus_one' if condition_on_action else 'observation_only')
        self.encoder = RecurrentEncoder(obs_dim, latent_dim, rnn_hidden_dim, grid_shape=grid_shape,
                                        action_dim=action_dim if condition_on_action else None).to(self.device)
        self.q_net = SimpleQNetwork(rnn_hidden_dim, action_dim, args, device=device).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = Adam(list(self.encoder.parameters()) + list(self.q_net.parameters()), lr=lr)
        self._hidden = None  # persistent hidden state for act() during a rollout
        self._previous_action = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def q_values(self, h: torch.Tensor) -> torch.Tensor:
        return self.q_net(h)

    def getV(self, h: torch.Tensor) -> torch.Tensor:
        q = self.q_values(h)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def get_targetV(self, h: torch.Tensor) -> torch.Tensor:
        q = self.target_net(h)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def critic(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q = self.q_values(h)
        return q.gather(1, action.long())

    def encode_episode(self, states: torch.Tensor, next_states: torch.Tensor, actions=None) -> torch.Tensor:
        """states/next_states: [T, obs_dim] for one episode. Returns [T+1, H]
        hiddens: hiddens[t] is the belief after states[t], hiddens[T] is the
        belief after the final next_states[-1] (reached but never acted from)."""
        if len(states) == 0 or len(next_states) != len(states):
            raise ValueError('Need a nonempty complete episode with matching states/next_states.')
        if self.config['condition_on_action'] and (actions is None or len(actions) != len(states)):
            raise ValueError('Action-conditioned episode encoding requires one action per transition.')
        hidden = None
        hiddens = []
        for t in range(states.shape[0]):
            previous = -1 if t == 0 else (actions[t - 1] if actions is not None else None)
            hidden, _ = self.encoder(states[t : t + 1], hidden, previous)
            hiddens.append(hidden)
        final_hidden, _ = self.encoder(next_states[-1:], hidden, actions[-1] if actions is not None else None)
        hiddens.append(final_hidden)
        return torch.cat(hiddens, dim=0)

    def _encode_batch(self, episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        """Loop over time, not (episode, time) pairs: one GRUCell call per
        step handles the whole batch at once. Looping per-episode like
        encode_episode does is ~batch_size times more overhead for nothing."""
        lengths = [ep[0].shape[0] for ep in episodes]
        if not lengths or min(lengths) < 1 or any(any(len(v) != len(ep[0]) for v in ep) for ep in episodes):
            raise ValueError('Need nonempty complete episodes with matching transition arrays.')
        max_len = max(lengths)
        batch_size = len(episodes)
        obs_dim = episodes[0][0].shape[1]
        device = episodes[0][0].device

        padded_states = torch.zeros(max_len, batch_size, obs_dim, device=device)
        previous_actions = torch.full((max_len, batch_size), -1, device=device, dtype=torch.long)
        for i, (states, *_rest) in enumerate(episodes):
            padded_states[: lengths[i], i] = states
            previous_actions[1:lengths[i], i] = episodes[i][2][:-1].reshape(-1)

        # featurizer (CNN/MLP) is over one frame with no temporal dependency, meaning we run it once
        # every (t,episode) pair batched together vs. once per timestep
        flat_obs = padded_states.reshape(max_len * batch_size, obs_dim)
        flat_latent = self.encoder._featurize(flat_obs)
        latents = flat_latent.view(max_len, batch_size, -1)

        hidden = torch.zeros(batch_size, self.encoder.rnn_hidden_dim, device=device)
        all_h = []
        for t in range(max_len):
            hidden = self.encoder.recurrent_step(latents[t], hidden, previous_actions[t])
            all_h.append(hidden)
        all_h = torch.stack(all_h, dim=0)  # [max_len, B, H]

        last_idx = torch.tensor([length - 1 for length in lengths], device=device)
        batch_arange = torch.arange(batch_size, device=device)
        hidden_before_final = all_h[last_idx, batch_arange]  # [B, H]
        final_next_obs = torch.stack([ep[1][-1] for ep in episodes], dim=0)  # [B, obs_dim]
        final_actions = torch.stack([ep[2][-1].reshape(()) for ep in episodes])
        final_hidden, _ = self.encoder(final_next_obs, hidden_before_final, final_actions)  # [B, H]

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
         
        # investigative metrics
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
        self._previous_action = None

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> int:
        with torch.no_grad():
            hidden, _ = self.encoder(obs.unsqueeze(0), self._hidden, self._previous_action)
            self._hidden = hidden
            q = self.q_values(hidden)
            probs = F.softmax(q / self.alpha, dim=1)
        action = int(torch.argmax(probs, dim=1).item()) if deterministic else int(Categorical(probs).sample().item())
        self._previous_action = action
        return action

    def save(self, path: str):
        torch.save({"format_version": 2, "config": self.config,
                    "encoder": self.encoder.state_dict(), "q_net": self.q_net.state_dict()}, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        stored = state.get('config')
        if stored is None:
            if self.config['condition_on_action'] or self.config['terminal_semantics'] != 'recorded':
                raise ValueError('Legacy checkpoints require observation-only memory and recorded terminal semantics; use from_checkpoint().')
        elif state.get('format_version') != 2 or stored != self.config:
            raise ValueError('Checkpoint configuration differs from this agent; use from_checkpoint().')
        self.encoder.load_state_dict(state["encoder"])
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["q_net"])
        self.reset_hidden()

    @classmethod
    def from_checkpoint(cls, path, *, obs_dim, args, grid_shape=None, device='cpu'):
        """Load both explicit v2 protocols and legacy observation-only weights."""
        state = torch.load(path, map_location=device, weights_only=True)
        config = state.get('config')
        if config is not None:
            if state.get('format_version') != 2:
                raise ValueError('Unsupported recurrent checkpoint version.')
            if (config['obs_dim'] != obs_dim or config['gamma'] != float(args.gamma)
                    or config['init_temp'] != float(args.agent.init_temp)):
                raise ValueError('Observation size, gamma, or temperature disagrees with checkpoint metadata.')
            if grid_shape is not None and tuple(config['grid_shape'] or ()) != tuple(grid_shape):
                raise ValueError('Grid shape disagrees with checkpoint metadata.')
            agent = cls(obs_dim, config['action_dim'], args, latent_dim=config['latent_dim'],
                        rnn_hidden_dim=config['rnn_hidden_dim'], grid_shape=config['grid_shape'],
                        condition_on_action=config['condition_on_action'],
                        terminal_semantics=config['terminal_semantics'], device=device)
        else:
            enc = state['encoder']
            if ('cnn.0.weight' in enc) != (grid_shape is not None):
                raise ValueError('Legacy CNN checkpoints require the original --grid-shape.')
            agent = cls(obs_dim, state['q_net']['fc3.weight'].shape[0], args,
                        latent_dim=enc['gru.weight_ih'].shape[1],
                        rnn_hidden_dim=enc['gru.weight_hh'].shape[1], grid_shape=grid_shape,
                        condition_on_action=False, device=device)
        agent.load(path)
        return agent


__all__ = ["RecurrentOfflineSoftQAgent", "make_args"]
