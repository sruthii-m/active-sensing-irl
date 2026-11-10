"""Matched 128,128 MLPs and numerically stable objectives."""
import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim=71, output_dim=7):
    return nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128),
                         nn.ReLU(), nn.Linear(128, output_dim))


def soft_value(q, temperature):
    if temperature <= 0:
        raise ValueError("Positive temperature required")
    return temperature * torch.logsumexp(q / temperature, dim=-1)


def iq_loss(q, action, next_q, terminal, gamma=.99, temperature=.001, chi_lambda=.5):
    if chi_lambda <= 0 or not 0 <= gamma <= 1:
        raise ValueError("Invalid IQ parameters")
    chosen = q.gather(1, action[:, None]).squeeze(1)
    target = gamma * (~terminal).float() * soft_value(next_q.detach(), temperature)
    imitation = soft_value(q, temperature) - chosen
    penalty = (chosen - target).square() / (4 * chi_lambda)
    return (imitation + penalty).mean(), dict(imitation=imitation.mean().item(), chi=penalty.mean().item())


def airl_logits(base, potential, next_potential, terminal, log_pi, gamma=.99):
    return base + gamma * (~terminal).float() * next_potential - potential - log_pi


def airl_loss(expert_logits, generator_logits):
    return F.softplus(-expert_logits).mean() + F.softplus(generator_logits).mean()
