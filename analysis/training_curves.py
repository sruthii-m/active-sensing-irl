"""Generic, file-based plots for scalar training and evaluation curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _series_points(series: Any, shared_steps: Sequence[float] | None) -> tuple[list[float], list[float]]:
    """Normalize convenient curve formats to aligned x/y points."""
    if isinstance(series, Mapping):
        if "steps" in series and "values" in series:
            steps, values = series["steps"], series["values"]
        elif "x" in series and "y" in series:
            steps, values = series["x"], series["y"]
        else:
            steps, values = zip(*series.items()) if series else ([], [])
    else:
        values = list(series)
        if values and isinstance(values[0], Sequence) and len(values[0]) == 2:
            steps, values = zip(*values)
        else:
            steps = shared_steps if shared_steps is not None else range(len(values))

    x, y = list(steps), list(values)
    if len(x) != len(y):
        raise ValueError("Each curve needs equally many steps and values.")
    if not x:
        raise ValueError("Cannot plot an empty curve.")
    return [float(point) for point in x], [float(point) for point in y]


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    if window < 1:
        raise ValueError("smoothing_window must be at least 1.")
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        smoothed.append(sum(values[start : index + 1]) / (index - start + 1))
    return smoothed


def plot_training_curves(
    curves: Mapping[str, Any],
    output_path: str | Path,
    *,
    steps: Sequence[float] | None = None,
    title: str | None = None,
    smoothing_window: int = 1,
    columns: int = 2,
) -> Path:
    """Save one panel per named scalar curve and return the output path.

    Examples::

        curves = {
            "eval/success_rate": [(0, 0.06), (100_000, 0.42)],
            "eval/mean_return": {"steps": [0, 100_000], "values": [0.05, 0.4]},
            "train/policy_loss": [-0.2, -0.1, -0.05],
        }
        plot_training_curves(curves, "artifacts/plots/full_state.png", steps=[...])

    Curves without explicit x-coordinates use ``steps``; if it is omitted,
    their sample index is used.  A trailing moving average can make noisy
    episodic returns readable without changing the raw data on disk.
    """
    if not curves:
        raise ValueError("Provide at least one named curve.")
    if columns < 1:
        raise ValueError("columns must be at least 1.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Plotting requires matplotlib. Install perception_irl/requirements.txt."
        ) from error

    num_curves = len(curves)
    rows = (num_curves + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 3.8 * rows), squeeze=False)
    for axis, (name, series) in zip(axes.flat, curves.items()):
        x, y = _series_points(series, steps)
        axis.plot(x, _moving_average(y, smoothing_window), linewidth=2)
        axis.set_title(name)
        axis.set_xlabel("training timestep")
        axis.set_ylabel(name.rsplit("/", maxsplit=1)[-1].replace("_", " "))
        axis.grid(alpha=0.3)
    for axis in list(axes.flat)[num_curves:]:
        axis.set_visible(False)
    if title:
        figure.suptitle(title)
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination
