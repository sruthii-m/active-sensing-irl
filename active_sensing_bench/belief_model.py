"""Bayesian filtering and joint observation/feedback transition enumeration."""
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .state_views import BELIEF_DIM, HORIZON, _prefix, validate_decision_state


def probabilities(value, shape=(BELIEF_DIM,)):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Invalid probability vector")
    if not np.isclose(value.sum(), 1, atol=1e-6):
        raise ValueError("Probabilities must sum to one")
    return value


def bayes_update(prior, likelihoods):
    prior = probabilities(prior)
    likelihoods = np.asarray(likelihoods, dtype=np.float64)
    if likelihoods.shape != prior.shape or not np.isfinite(likelihoods).all() or ((likelihoods < 0) | (likelihoods > 1)).any():
        raise ValueError("Invalid observation likelihoods")
    weights = prior * likelihoods
    if weights.sum() <= 0:
        raise ValueError("Observation has zero probability under the supplied belief")
    return (weights / weights.sum()).astype(np.float32)


def enumerate_observation_updates(prior, likelihoods_by_observation):
    prior = probabilities(prior)
    likelihoods = np.asarray(list(likelihoods_by_observation.values()), dtype=float)
    if likelihoods.ndim != 2 or likelihoods.shape[1] != BELIEF_DIM or not np.isfinite(likelihoods).all() or (likelihoods < 0).any() or not np.allclose(likelihoods.sum(0), 1):
        raise ValueError("Observation kernel must normalize for each candidate goal")
    return [(float(prior @ values), key, bayes_update(prior, values))
            for key, values in likelihoods_by_observation.items() if prior @ values > 0]


@dataclass
class BeliefTransitionModel:
    """Known-model callbacks enumerate hypotheses, never the sampled goal.

    Public next pose and goal-entry feedback are jointly conditioned with the
    image. The deadline masks continuation but carries no goal evidence.
    """
    transition: Callable
    observation_model: Callable[..., Mapping[object, float]]

    def successors(self, x, action):
        x = validate_decision_state(x)
        if int(action) != action or not 0 <= action < 7:
            raise ValueError("Invalid action")
        pose, t = decode_state(x)
        if t >= HORIZON:
            raise ValueError("No decisions after the intrinsic deadline")
        branches = {}
        for index in np.flatnonzero(x[7:]):
            goal = (int(index // 8 + 1), int(index % 8 + 1))
            next_pose, reached = self.transition(pose, int(action), goal)
            outcomes = self.observation_model(pose, int(action), goal)
            probabilities(list(outcomes.values()), (len(outcomes),))
            for observation, likelihood in outcomes.items():
                if likelihood == 0:
                    continue
                key = (tuple(next_pose), bool(reached), observation)
                weights = branches.setdefault(key, np.zeros(BELIEF_DIM, dtype=np.float64))
                weights[index] += float(x[7 + index]) * likelihood
        merged = {}
        for (next_pose, reached, _), weights in branches.items():
            mass = float(weights.sum())
            posterior = (weights / mass).astype(np.float32)
            nxt = np.concatenate((_prefix(next_pose, t + 1), posterior))
            terminal = reached or t + 1 == HORIZON
            key = (nxt.tobytes(), terminal)
            if key in merged:
                merged[key][0] += mass
            else:
                merged[key] = [mass, nxt, terminal]
        result = [tuple(branch) for branch in merged.values()]
        if not np.isclose(sum(p for p, _, _ in result), 1, atol=1e-6):
            raise ValueError("Successor probabilities do not sum to one")
        return result


def decode_state(x):
    pose = (int(round(x[0] * 9)), int(round(x[1] * 9)), int(np.argmax(x[2:6])))
    return pose, int(round(float(x[6]) * HORIZON))
