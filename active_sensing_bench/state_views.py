"""The common 71-dimensional decision-state contract.

The full and exact-belief conditions intentionally have identical dimensions:
pose (x, y, heading), intrinsic time, and one 64-cell information vector.
Only the source of the final vector differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

GRID_SIZE = 10
INTERIOR_SIZE = GRID_SIZE - 2
BELIEF_DIM = INTERIOR_SIZE * INTERIOR_SIZE
ACTION_DIM = 7
HORIZON = 400
DECISION_DIM = 2 + 4 + 1 + BELIEF_DIM


def _pose(pose: Any) -> tuple[int, int, int]:
    value = np.asarray(pose).reshape(-1)
    if value.size != 3:
        raise ValueError("pose must contain x, y, and direction")
    if not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
        raise ValueError("pose must contain integers")
    x, y, direction = (int(v) for v in value)
    if not (1 <= x <= 8 and 1 <= y <= 8 and 0 <= direction < 4):
        raise ValueError(f"invalid pose {(x, y, direction)}")
    return x, y, direction


def _prefix(pose: Any, t: int) -> np.ndarray:
    x, y, direction = _pose(pose)
    if int(t) != t or t < 0 or t > HORIZON:
        raise ValueError(f"time must be in [0, {HORIZON}], got {t}")
    heading = np.zeros(4, dtype=np.float32)
    heading[direction] = 1.0
    return np.concatenate(
        (np.asarray([x / 9.0, y / 9.0], dtype=np.float32), heading,
         np.asarray([t / HORIZON], dtype=np.float32))
    )


def goal_index(goal: Any) -> int:
    value = np.asarray(goal)
    if value.shape != (2,) or not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
        raise ValueError("goal must contain two integers")
    x, y = value.astype(int)
    if not (1 <= x <= 8 and 1 <= y <= 8):
        raise ValueError(f"goal must be an interior cell, got {(x, y)}")
    return (x - 1) * INTERIOR_SIZE + (y - 1)


def goal_one_hot(goal: Any) -> np.ndarray:
    result = np.zeros(BELIEF_DIM, dtype=np.float32)
    result[goal_index(goal)] = 1.0
    return result


def validate_decision_state(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (DECISION_DIM,):
        raise ValueError(f"decision state must have shape ({DECISION_DIM},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("decision state contains non-finite values")
    pose = result[:2] * 9
    if not np.allclose(pose, np.rint(pose), atol=1e-5) or (pose < 1 - 1e-5).any() or (pose > 8 + 1e-5).any():
        raise ValueError("decision state has invalid pose")
    if not np.array_equal(result[2:6], np.eye(4)[result[2:6].argmax()]):
        raise ValueError("heading must be one-hot")
    if not 0 <= result[6] <= 1 or not np.isclose(result[6] * HORIZON, round(float(result[6]) * HORIZON), atol=1e-4):
        raise ValueError("invalid intrinsic time")
    if (result[7:] < 0).any() or not np.isclose(result[7:].sum(), 1, atol=1e-6):
        raise ValueError("goal posterior must normalize")
    return result


@dataclass
class FullStateView:
    """Build the privileged full-state input used only in the full condition."""

    def reset(self, observation: Any, public_context: Mapping[str, Any]) -> np.ndarray:
        del observation
        return self._make(public_context)

    def step(self, action: int, observation: Any, public_feedback: Mapping[str, Any]) -> np.ndarray:
        del action, observation
        return self._make(public_feedback)

    @staticmethod
    def _make(context: Mapping[str, Any]) -> np.ndarray:
        if "goal" not in context:
            raise ValueError("the full-state view requires an evaluator-side goal")
        return validate_decision_state(
            np.concatenate((_prefix(context["pose"], context["t"]), goal_one_hot(context["goal"])))
        )


@dataclass
class BeliefStateView:
    """Build exact Bayesian inputs without accepting a sampled hidden goal."""

    model: Any = None
    belief: Any = None

    def reset(self, observation: Any, public_context: Mapping[str, Any]) -> np.ndarray:
        self._public_only(public_context)
        if self.model is None:
            return self._make(public_context)
        self.belief = np.ones(BELIEF_DIM, dtype=np.float32) / (BELIEF_DIM - 1)
        self.belief[goal_index(public_context["pose"][:2])] = 0
        self.belief = self.model.observe(self.belief, public_context["pose"], observation, False)
        return self._make({**public_context, "belief": self.belief})

    def step(self, action: int, observation: Any, public_feedback: Mapping[str, Any]) -> np.ndarray:
        self._public_only(public_feedback)
        if self.model is None:
            return self._make(public_feedback)
        if self.belief is None:
            raise RuntimeError("Reset belief before stepping")
        self.belief = self.model.observe(self.belief, public_feedback["pose"], observation,
                                         public_feedback["goal_reached"])
        return self._make({**public_feedback, "belief": self.belief})

    @staticmethod
    def _public_only(context):
        if set(context) - {"pose", "t", "belief", "goal_reached"}:
            raise ValueError("Belief view accepts public context only")

    @staticmethod
    def _make(context: Mapping[str, Any]) -> np.ndarray:
        if "belief" not in context:
            raise ValueError("the belief view requires a posterior, not a hidden goal")
        belief = np.asarray(context["belief"], dtype=np.float32).copy()
        if belief.shape != (BELIEF_DIM,) or not np.isfinite(belief).all() or np.any(belief < 0) or not np.isclose(belief.sum(), 1.0, atol=1e-6):
            raise ValueError("belief must be a normalized nonnegative 64-cell posterior")
        belief = np.maximum(belief, 0.0)
        belief /= belief.sum()
        return validate_decision_state(np.concatenate((_prefix(context["pose"], context["t"]), belief)))
