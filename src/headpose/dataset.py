"""Bounded, user-disjoint BOXRR training-set materialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import motion_features
from .quaternion import inverse, log_map, multiply
from .resample import contiguous_slices, uniform_resample
from .splits import partition_user_hashes
from .windows import make_window_indices, materialize_window


@dataclass(frozen=True)
class WindowDataset:
    features: np.ndarray
    targets: np.ndarray
    user_indices: np.ndarray
    users: tuple[str, ...]


def tangent_targets(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    """Express every future orientation relative to the last observed pose."""
    return log_map(multiply(inverse(history[-1]), future)).astype(np.float32)


def materialize_boxrr_splits(
    root: Path,
    context: int = 50,
    horizon: int = 20,
    stride: int = 20,
    split_seed: str = "xr-headpose-v1",
    max_windows: dict[str, int] | None = None,
) -> tuple[dict[str, WindowDataset], dict[str, str]]:
    """Stream processed recordings into bounded arrays without raw-data copies."""
    manifest = json.loads((root / "pilot-manifest.json").read_text())
    user_hashes = [entry["user_hash"] for entry in manifest["users"]]
    assignments = partition_user_hashes(user_hashes, split_seed)
    limits = max_windows or {"train": 120_000, "validation": 20_000, "test": 20_000}
    rng = np.random.default_rng(2027)
    buffers: dict[str, list[tuple[np.ndarray, np.ndarray, int]]] = {
        split: [] for split in limits
    }
    seen = {split: 0 for split in limits}
    split_users = {
        split: tuple(user for user in user_hashes if assignments[user] == split)
        for split in limits
    }
    user_to_index = {
        split: {user: index for index, user in enumerate(split_users[split])}
        for split in limits
    }
    for user in user_hashes:
        split = assignments[user]
        if split not in limits:
            continue
        for path in sorted((root / user).glob("*.npz")):
            with np.load(path) as recording:
                timestamps = recording["timestamps"]
                poses = recording["quaternions"]
            for segment_number, segment in enumerate(contiguous_slices(timestamps)):
                try:
                    _, uniform = uniform_resample(
                        timestamps[segment], poses[segment], sampling_hz=100.0
                    )
                except ValueError:
                    continue
                indices = make_window_indices(
                    f"{path.stem}-{segment_number}",
                    len(uniform),
                    context,
                    horizon,
                    stride,
                )
                for index in indices:
                    history, future = materialize_window(uniform, index)
                    item = (
                        motion_features(history),
                        tangent_targets(history, future),
                        user_to_index[split][user],
                    )
                    seen[split] += 1
                    if len(buffers[split]) < limits[split]:
                        buffers[split].append(item)
                    else:
                        replacement = int(rng.integers(0, seen[split]))
                        if replacement < limits[split]:
                            buffers[split][replacement] = item
    datasets: dict[str, WindowDataset] = {}
    for split, items in buffers.items():
        if not items:
            raise RuntimeError(f"split {split} contains no usable windows")
        datasets[split] = WindowDataset(
            features=np.stack([item[0] for item in items]),
            targets=np.stack([item[1] for item in items]),
            user_indices=np.asarray([item[2] for item in items], dtype=np.int64),
            users=split_users[split],
        )
    return datasets, assignments
