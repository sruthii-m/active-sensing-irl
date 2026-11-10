"""Shared contracts for the active-sensing IRL comparison.

The package deliberately contains representation and data semantics separately
from any particular IRL algorithm.  IQ-Learn, AIRL, BC, and the exact MCE
calibration should consume the same decision-state and episode contracts.
"""

from .state_views import BELIEF_DIM, DECISION_DIM, FullStateView, BeliefStateView
from .data import validate_episode

__all__ = [
    "BELIEF_DIM",
    "DECISION_DIM",
    "FullStateView",
    "BeliefStateView",
    "validate_episode",
]
