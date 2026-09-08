"""Pre-registered quantitative metrics for active sensing.

The functions accept either a sequence of visited states/positions, a sequence
of transition dictionaries containing ``position`` or ``agent_pos``, or a
trajectory dictionary containing ``positions`` or ``states``.  Positions are
normally MiniGrid ``(x, y)`` coordinates.  This deliberately keeps analysis
independent of a particular expert or replay-buffer format.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from math import log2
from typing import Any


def _freeze_state(state: Any) -> Any:
    """Convert common array-like states into stable, countable values."""
    if hasattr(state, "tolist"):
        state = state.tolist()
    if isinstance(state, list):
        return tuple(_freeze_state(item) for item in state)
    if isinstance(state, tuple):
        return tuple(_freeze_state(item) for item in state)
    if isinstance(state, Mapping):
        return tuple(sorted((key, _freeze_state(value)) for key, value in state.items()))
    return state


def _trajectory_states(trajectory: Any) -> list[Any]:
    """Extract an ordered state sequence from the supported trajectory forms."""
    if isinstance(trajectory, Mapping):
        for key in ("positions", "agent_positions", "states"):
            if key in trajectory:
                return list(trajectory[key])
        raise ValueError("Trajectory mapping needs 'positions', 'agent_positions', or 'states'.")

    states = list(trajectory)
    if not states:
        return []
    if isinstance(states[0], Mapping):
        extracted = []
        for transition in states:
            for key in ("position", "agent_pos", "state"):
                if key in transition:
                    extracted.append(transition[key])
                    break
            else:
                raise ValueError("Each transition needs 'position', 'agent_pos', or 'state'.")
        return extracted
    return states


def _in_region(position: Any, region: Any) -> bool:
    """Test membership in a callable, a set of cells, or an inclusive rectangle.

    Rectangle syntax is ``(x_min, y_min, x_max, y_max)``.  A region object may
    also provide ``contains(position)``; this makes it easy to use task-specific
    visibility/fixation regions without changing the metric.
    """
    if callable(region):
        return bool(region(position))
    if hasattr(region, "contains"):
        return bool(region.contains(position))
    if isinstance(region, Mapping):
        return (
            region["x_min"] <= position[0] <= region["x_max"]
            and region["y_min"] <= position[1] <= region["y_max"]
        )
    if isinstance(region, (set, frozenset)):
        return _freeze_state(position) in {_freeze_state(cell) for cell in region}
    if isinstance(region, Sequence) and len(region) == 4:
        x_min, y_min, x_max, y_max = region
        return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max
    raise TypeError("A region must be a callable, containable object, cell set, mapping, or (x_min, y_min, x_max, y_max).")


def _region_summary(positions: list[Any], region: Any) -> dict[str, int | float | None]:
    visits = [_in_region(position, region) for position in positions]
    first = next((index for index, inside in enumerate(visits) if inside), None)
    dwell_steps = sum(visits)
    return {
        "time_to_first_fixation": first,
        "dwell_steps": dwell_steps,
        "dwell_fraction": dwell_steps / len(positions) if positions else 0.0,
    }


def visited_state_entropy(trajectory) -> float:
    """Return Shannon entropy (bits) of the empirical visited-state distribution.

    A high value means the agent spread time over many states; it does *not* by
    itself establish task-selective sensing, so interpret it with fixation data.
    """
    states = [_freeze_state(state) for state in _trajectory_states(trajectory)]
    if not states:
        return 0.0
    total = len(states)
    return -sum((count / total) * log2(count / total) for count in Counter(states).values())


def path_length_vs_shortest_path(trajectory, shortest_path_length: int) -> float:
    """Return executed transition count divided by the shortest-path length.

    ``1.0`` is shortest-path efficient; larger values quantify navigation
    overhead.  Repeated positions still count as executed steps, which captures
    turns/look actions in an action-level trajectory.
    """
    if shortest_path_length < 0:
        raise ValueError("shortest_path_length must be non-negative.")
    actual_length = max(0, len(_trajectory_states(trajectory)) - 1)
    if shortest_path_length == 0:
        return 1.0 if actual_length == 0 else float("inf")
    return actual_length / shortest_path_length


def fixation_time_goal_vs_distractors(trajectory, goal_region, distractor_regions) -> dict:
    """Compare goal fixation with fixation on reward-irrelevant distractors.

    ``distractor_regions`` can be a mapping of labels to regions or a sequence
    of regions.  Time is in trajectory indices, with index 0 denoting spawn.
    Distance-from-spawn matching belongs in the experiment's aggregation layer:
    call this metric per episode, then compare matched episodes/regions.
    """
    positions = _trajectory_states(trajectory)
    if isinstance(distractor_regions, Mapping):
        labelled_regions = dict(distractor_regions)
    else:
        labelled_regions = {f"distractor_{index}": region for index, region in enumerate(distractor_regions)}

    per_distractor = {
        label: _region_summary(positions, region)
        for label, region in labelled_regions.items()
    }
    any_distractor = _region_summary(
        positions,
        lambda position: any(_in_region(position, region) for region in labelled_regions.values()),
    )
    return {
        "goal": _region_summary(positions, goal_region),
        "distractors": per_distractor,
        "any_distractor": any_distractor,
    }
