"""Offline IQ-Learn agent for the full-state foraging task, reimplementing
iq_learn/agent/softq.py's update logic (no Hydra, no online replay buffer —
this project's offline recipe trains purely on expert demonstrations, matching
IQ-Learn's own scripts/run_offline.sh: method.loss=value_expert, method.chi=True).

r(s,a) = Q(s,a) - gamma * E[V*(s')] is the IRL-recovered reward (see infer_reward).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from iq_learn.agent.softq_models import SimpleQNetwork
from iq_learn.iq import iq_loss


def _namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**{k: _namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


def make_args(
    *,
    gamma: float = 0.99,
    alpha: float = 0.5,
    init_temp: float = 0.001,
    critic_target_update_frequency: int = 4,
    tanh: bool = False,
) -> SimpleNamespace:
    """Config object shaped like softq_models.py/iq.py expect (subset of the
    upstream Hydra config actually read by those two vendored files)."""
    return _namespace(
        {
            "gamma": gamma,
            "q_net": {"_target_": "agent.softq_models.SimpleQNetwork"},
            "method": {
                "tanh": tanh,
                "loss": "value_expert",
                "chi": True,
                "div": None,
                "alpha": alpha,
                "grad_pen": False,
                "regularize": False,
            },
            "agent": {"init_temp": init_temp, "critic_target_update_frequency": critic_target_update_frequency},
        }
    )


class OfflineSoftQAgent:
    """Adapted from iq_learn/agent/softq.py: same Q/V definitions and target
    network, but built directly (no hydra.utils.instantiate) and with no
    dependence on an online policy replay buffer."""

    def __init__(self, obs_dim: int, action_dim: int, args: SimpleNamespace, device: str = "cpu", lr: float = 3e-4):
        self.args = args
        self.gamma = args.gamma
        self.device = torch.device(device)
        self.log_alpha = torch.log(torch.tensor(float(args.agent.init_temp), device=self.device))
        self.q_net = SimpleQNetwork(obs_dim, action_dim, args, device=device).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.q_net

    def getV(self, obs: torch.Tensor) -> torch.Tensor:
        q = self.q_net(obs)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def get_targetV(self, obs: torch.Tensor) -> torch.Tensor:
        q = self.target_net(obs)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q = self.q_net(obs)
        return q.gather(1, action.long())

    def policy_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = self.q_net(obs)
            return F.softmax(q / self.alpha, dim=1)

    def choose_action(self, obs: torch.Tensor, deterministic: bool = True) -> int:
        probs = self.policy_action_probs(obs.unsqueeze(0))
        if deterministic:
            return int(torch.argmax(probs, dim=1).item())
        return int(Categorical(probs).sample().item())

    def infer_reward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """r(s,a) = Q(s,a) - gamma * V(s') -- the IRL-recovered reward."""
        with torch.no_grad():
            return self.critic(obs, action) - self.gamma * self.getV(next_obs)

    def update(self, obs, next_obs, action, done, step: int, target_update_freq: int) -> dict:
        is_expert = torch.ones(obs.shape[0], 1, dtype=torch.bool, device=self.device)
        # env_reward is unused by iq_loss -- the whole point of IQ-Learn is
        # recovering a reward from the Q-function's structure, not from the
        # ground-truth env reward. `done` still matters: it stops V(s') from
        # bootstrapping past a terminal (goal-reached) transition.
        env_reward = torch.zeros(obs.shape[0], 1, device=self.device)
        batch = (obs, next_obs, action, env_reward, done, is_expert)

        current_Q = self.critic(obs, action)
        current_V = self.getV(obs)
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)

        loss, loss_dict = iq_loss(self, current_Q, current_V, next_V, batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss_dict

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state)
        self.target_net.load_state_dict(state)
