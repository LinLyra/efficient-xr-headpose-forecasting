"""Causal-safe preprocessing for timestamped quaternion streams."""

from __future__ import annotations

import numpy as np

from .quaternion import enforce_sign_continuity, slerp


def contiguous_slices(
    timestamps: np.ndarray, max_gap_seconds: float = 0.1, min_samples: int = 2
) -> list[slice]:
    """Partition a timestamp stream at non-positive or overly large gaps."""
    times = np.asarray(timestamps, dtype=float)
    if times.ndim != 1 or min_samples < 2 or max_gap_seconds <= 0.0:
        raise ValueError("invalid timestamp stream or segmentation settings")
    if len(times) < min_samples:
        return []
    gaps = np.diff(times)
    boundaries = np.flatnonzero(
        ~np.isfinite(gaps) | (gaps <= 0.0) | (gaps > max_gap_seconds)
    ) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(times)]))
    return [
        slice(int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
        if stop - start >= min_samples
    ]


def uniform_resample(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    sampling_hz: float = 100.0,
    max_gap_seconds: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """SLERP a strictly increasing stream onto a uniform inclusive grid."""
    times = np.asarray(timestamps, dtype=float)
    poses = enforce_sign_continuity(quaternions)
    if times.ndim != 1 or len(times) != len(poses) or len(times) < 2:
        raise ValueError("timestamps and quaternions must describe one stream")
    gaps = np.diff(times)
    if np.any(~np.isfinite(times)) or np.any(gaps <= 0.0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if np.any(gaps > max_gap_seconds):
        raise ValueError("stream contains a gap larger than max_gap_seconds")
    if sampling_hz <= 0.0:
        raise ValueError("sampling_hz must be positive")

    period = 1.0 / sampling_hz
    grid = np.arange(times[0], times[-1] + period * 0.5, period)
    indices = np.searchsorted(times, grid, side="right") - 1
    indices = np.clip(indices, 0, len(times) - 2)
    denominator = times[indices + 1] - times[indices]
    fraction = np.clip((grid - times[indices]) / denominator, 0.0, 1.0)
    result = slerp(poses[indices], poses[indices + 1], fraction)
    return grid, result
