"""Uncertainty-minimization action score, kept as a reference to correlate
against post hoc — never fused into the reward-only objective, or the
correlation stops being a real test.

    score(a) = H[q_phi(s|h_t)] - E_{o_{t+1}}[H[q_phi(s|h_t,a,o_{t+1})]]

Pattern reference: AbdoSharaf98/active-sensing-paper BAS.py:score_action /
agents/cmc_explorer.py:compute_entropy_score (MC-estimated entropy difference
over a discrete action grid). Reference only — their interfaces don't match
this task.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from math import isfinite, log2
from typing import Any


def categorical_entropy(distribution: Any) -> float:
    """Return Shannon entropy in bits for a categorical belief.

    ``distribution`` may be a mapping from latent state to mass, a one-
    dimensional sequence of masses, or a tensor/array with ``tolist``.  Masses
    need not already sum to one; normalizing here avoids silent differences
    between exact and Monte-Carlo belief implementations.
    """
    if hasattr(distribution, "detach"):
        distribution = distribution.detach().cpu()
    if hasattr(distribution, "tolist"):
        distribution = distribution.tolist()
    masses = list(distribution.values()) if isinstance(distribution, Mapping) else list(distribution)
    if not masses:
        return 0.0
    if any(not isfinite(float(mass)) or float(mass) < 0 for mass in masses):
        raise ValueError("Belief masses must be finite and non-negative.")
    total = sum(float(mass) for mass in masses)
    if total <= 0:
        raise ValueError("A belief must have positive total mass.")
    return -sum(
        probability * log2(probability)
        for mass in masses
        if (probability := float(mass) / total) > 0
    )


def _outcomes(result: Any) -> list[tuple[float, Any]]:
    """Normalize supported transition-update return formats.

    The updater may return ``[(p(o), posterior), ...]``, a mapping from
    observations to ``(p(o), posterior)``, or ``(probabilities, posteriors)``.
    A bare posterior means a deterministic observation.
    """
    if isinstance(result, Mapping):
        values = list(result.values())
        if values and all(isinstance(value, Sequence) and len(value) == 2 for value in values):
            return [(float(probability), posterior) for probability, posterior in values]
        return [(1.0, result)]

    if isinstance(result, tuple) and len(result) == 2:
        probabilities, posteriors = result
        if isinstance(probabilities, Sequence) and not isinstance(probabilities, (str, bytes)):
            return [(float(probability), posterior) for probability, posterior in zip(probabilities, posteriors, strict=True)]

    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if not result:
            raise ValueError("Belief updater returned no possible observations.")
        if all(isinstance(outcome, Sequence) and len(outcome) == 2 for outcome in result):
            return [(float(probability), posterior) for probability, posterior in result]
    return [(1.0, result)]


def expected_posterior_entropy(belief_or_latent_dist: Any, action: Any, transition_and_belief_update_fn: Callable[[Any, Any], Any]) -> float:
    """Return expected posterior entropy after taking ``action``.

    The callback is the only environment/model-specific part of this baseline:
    it must enumerate or sample possible next observations and provide the
    posterior belief for each one.  It is deliberately not an environment
    reward callback, preserving this as a reward-free null hypothesis.
    """
    outcomes = _outcomes(transition_and_belief_update_fn(belief_or_latent_dist, action))
    probabilities = [probability for probability, _ in outcomes]
    if any(not isfinite(probability) or probability < 0 for probability in probabilities):
        raise ValueError("Observation probabilities must be finite and non-negative.")
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("Observation probabilities must have positive total mass.")
    return sum((probability / total) * categorical_entropy(posterior) for probability, posterior in outcomes)


def entropy_score(belief_or_latent_dist, action, transition_and_belief_update_fn) -> float:
    """Expected information gain for one action, measured in bits.

    This is the standalone null baseline ``H[b] - E[H[b' | o]]``.  Do not add
    this score to IQ-Learn's reward objective; compare scores with the
    reward-only policy only after training.
    """
    return categorical_entropy(belief_or_latent_dist) - expected_posterior_entropy(
        belief_or_latent_dist, action, transition_and_belief_update_fn
    )


def select_entropy_action(belief_or_latent_dist: Any, actions: Iterable[Any], transition_and_belief_update_fn: Callable[[Any, Any], Any]) -> tuple[Any, dict[Any, float]]:
    """Select the highest-information action and return all scores.

    Ties retain the first action supplied, making comparisons reproducible.
    """
    action_list = list(actions)
    if not action_list:
        raise ValueError("At least one action is required.")
    scores = {
        action: entropy_score(belief_or_latent_dist, action, transition_and_belief_update_fn)
        for action in action_list
    }
    return max(action_list, key=scores.__getitem__), scores
