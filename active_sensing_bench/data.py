"""Versioned neutral episodes, layout splits, and evaluation-sidecar exclusion."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .state_views import HORIZON, validate_decision_state

REQUIRED_KEYS = {"episode_id", "layout_id", "expert_id", "condition", "observations",
                 "actions", "terminated", "truncated", "task_terminal", "length"}
CONDITION_KEYS = {"representation", "view_size", "horizon", "collection_limit"}
SCHEMA = "empty-room71-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_episode(episode):
    if set(episode) != REQUIRED_KEYS:
        raise ValueError(f"Neutral episode fields must be exactly {sorted(REQUIRED_KEYS)}; evaluation sidecars are forbidden")
    condition = episode["condition"]
    if set(condition) != CONDITION_KEYS or condition["horizon"] != HORIZON or condition["representation"] not in ("full", "belief"):
        raise ValueError("Invalid benchmark condition")
    if any(not isinstance(episode[k], str) or not episode[k] for k in ("episode_id", "layout_id", "expert_id")):
        raise ValueError("Nonempty episode/layout/expert IDs required")
    length = episode["length"]
    if not isinstance(length, (int, np.integer)) or not 1 <= length <= HORIZON:
        raise ValueError("Invalid episode length")
    observations = np.asarray(episode["observations"])
    actions = np.asarray(episode["actions"])
    flags = [np.asarray(episode[k]) for k in ("terminated", "truncated", "task_terminal")]
    if observations.shape != (length + 1, 71) or not np.isfinite(observations).all():
        raise ValueError("observations must have shape [length + 1, 71]")
    for x in observations:
        validate_decision_state(x)
    if not np.allclose(observations[:, 6] * HORIZON, np.arange(length + 1), atol=1e-4):
        raise ValueError("Episodes must be ordered, complete prefixes starting at reset")
    if condition["representation"] == "full" and not np.isin(observations[:, 7:], [0, 1]).all():
        raise ValueError("Full-state goal input must be one-hot")
    if actions.shape != (length,) or actions.dtype.kind not in "iu" or ((actions < 0) | (actions >= 7)).any():
        raise ValueError("Actions must be integer MiniGrid indices")
    if any(f.shape != (length,) or f.dtype != np.bool_ for f in flags):
        raise ValueError("Terminal flags must be boolean arrays of length T")
    terminated, truncated, terminal = flags
    deadline = np.arange(1, length + 1) == HORIZON
    if not np.array_equal(terminal, terminated | deadline) or np.any(deadline & ~truncated):
        raise ValueError("task_terminal must equal goal termination OR intrinsic deadline")
    if (terminated[:-1] | truncated[:-1] | terminal[:-1]).any() or not (terminated[-1] or truncated[-1]):
        raise ValueError("Episodes must end at the first task end or collection cutoff")
    result = dict(episode)
    result.update(observations=observations.astype(np.float32), actions=actions.astype(np.int64))
    return result


def save_episodes(path, episodes):
    episodes = [validate_episode(ep) for ep in episodes]
    if not episodes or len({ep["episode_id"] for ep in episodes}) != len(episodes):
        raise ValueError("Need nonempty episodes with unique IDs")
    fields = ("observations", "actions", "terminated", "truncated", "task_terminal")
    metadata, arrays = [], {}
    for i, episode in enumerate(episodes):
        metadata.append({k: v for k, v in episode.items() if k not in fields})
        for field in fields:
            arrays[f"{i}_{field}"] = episode[field]
    arrays["metadata"] = np.asarray(json.dumps({"schema": SCHEMA, "episodes": metadata}))
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_episodes(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        if metadata["schema"] != SCHEMA:
            raise ValueError("Unsupported episode schema")
        result = []
        for i, meta in enumerate(metadata["episodes"]):
            episode = dict(meta)
            for field in ("observations", "actions", "terminated", "truncated", "task_terminal"):
                episode[field] = saved[f"{i}_{field}"]
            result.append(validate_episode(episode))
        return result


def load_split(directory, name):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or name not in ("train", "validation", "test"):
        raise ValueError("Unsupported manifest or split")
    seen_ids, seen_layouts = set(), set()
    for split in ("train", "validation", "test"):
        entry = manifest["splits"][split]
        if seen_ids.intersection(entry["episode_ids"]) or seen_layouts.intersection(entry["layout_ids"]):
            raise ValueError("Episode or complete-layout leakage across splits")
        seen_ids.update(entry["episode_ids"])
        seen_layouts.update(entry["layout_ids"])
    entry = manifest["splits"][name]
    path = directory / f"{name}.npz"
    if digest(path) != entry["sha256"]:
        raise ValueError("Dataset differs from frozen manifest")
    episodes = load_episodes(path)
    if [e["episode_id"] for e in episodes] != entry["episode_ids"] or [e["layout_id"] for e in episodes] != entry["layout_ids"]:
        raise ValueError("Episode provenance differs from manifest")
    if any(e["condition"] != manifest["condition"] or e["expert_id"] != manifest["expert_id"] for e in episodes):
        raise ValueError("Mixed conditions or expert IDs")
    return episodes, manifest


def transition_arrays(episodes, weighting="transition"):
    if weighting not in ("transition", "episode"):
        raise ValueError("Choose transition or episode weighting")
    episodes = [validate_episode(ep) for ep in episodes]
    lengths = np.asarray([ep["length"] for ep in episodes])
    arrays = dict(
        x=np.concatenate([e["observations"][:-1] for e in episodes]),
        next_x=np.concatenate([e["observations"][1:] for e in episodes]),
        action=np.concatenate([e["actions"] for e in episodes]),
        terminal=np.concatenate([e["task_terminal"] for e in episodes]),
        episode=np.repeat(np.arange(len(episodes)), lengths),
    )
    weight = np.ones(lengths.sum()) if weighting == "transition" else np.repeat(1 / lengths, lengths)
    arrays["sampling_probability"] = weight / weight.sum()
    return arrays


def layout_id(layout):
    return hashlib.sha256(json.dumps([int(v) for v in layout]).encode()).hexdigest()


def split_layouts(counts=(400, 100, 100), seed=0):
    """Assign complete reset specifications before any trajectories are sampled."""
    if min(counts) < 1 or sum(counts) > 64 * 63 * 4:
        raise ValueError("Invalid layout split sizes")
    rng, used, result = np.random.default_rng(seed), set(), {}
    for name, count in zip(("train", "validation", "test"), counts):
        layouts = []
        while len(layouts) < count:
            x, y, gx, gy = rng.integers(1, 9, 4)
            layout = (int(x), int(y), int(rng.integers(4)), int(gx), int(gy))
            if (x, y) != (gx, gy) and layout not in used:
                used.add(layout)
                layouts.append(layout)
        result[name] = layouts
    return result
