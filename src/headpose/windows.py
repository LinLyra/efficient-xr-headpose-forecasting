"""Strictly session-local causal window indexing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowIndex:
    session_hash: str
    context_start: int
    prediction_start: int
    prediction_stop: int


def make_window_indices(
    session_hash: str,
    sample_count: int,
    context_samples: int,
    horizon_samples: int,
    stride: int = 1,
) -> list[WindowIndex]:
    if not session_hash:
        raise ValueError("session_hash must be non-empty")
    if min(sample_count, context_samples, horizon_samples, stride) < 1:
        raise ValueError("sample counts and stride must be positive")
    width = context_samples + horizon_samples
    if sample_count < width:
        return []
    return [
        WindowIndex(
            session_hash=session_hash,
            context_start=start,
            prediction_start=start + context_samples,
            prediction_stop=start + width,
        )
        for start in range(0, sample_count - width + 1, stride)
    ]


def materialize_window(
    poses: np.ndarray, index: WindowIndex
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(poses, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("poses must have shape [time, 4]")
    if index.prediction_stop > len(values):
        raise ValueError("window exceeds session boundary")
    return (
        values[index.context_start : index.prediction_start],
        values[index.prediction_start : index.prediction_stop],
    )
