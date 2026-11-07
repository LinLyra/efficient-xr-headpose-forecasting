"""Deterministic SO(3) trajectories for pipeline and baseline validation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .quaternion import exp_map, multiply, normalize


def integrate_angular_steps(steps: np.ndarray) -> np.ndarray:
    """Integrate body-frame axis-angle increments from identity."""
    increments = np.asarray(steps, dtype=float)
    if increments.ndim != 2 or increments.shape[1] != 3:
        raise ValueError("steps must have shape [time, 3]")
    poses = [np.array([1.0, 0.0, 0.0, 0.0])]
    for increment in increments:
        poses.append(normalize(multiply(poses[-1], exp_map(increment))))
    return np.stack(poses)


def make_trajectory(regime: str, samples: int = 160) -> np.ndarray:
    """Create a simple trajectory with a transition after the context boundary."""
    if samples < 40:
        raise ValueError("samples must be at least 40")
    base = np.radians(0.5)
    time = np.arange(samples - 1, dtype=float)
    if regime == "stationary":
        speed = np.zeros_like(time)
    elif regime == "steady":
        speed = np.full_like(time, base)
    elif regime == "accelerating":
        speed = base * np.clip(time / 100.0, 0.0, 2.0)
    elif regime == "decelerating":
        speed = base * np.clip(1.5 - time / 120.0, 0.0, None)
    elif regime == "reversal":
        speed = np.where(time < 100.0, base, -base)
    else:
        raise ValueError(f"unknown regime: {regime}")
    steps = np.zeros((samples - 1, 3), dtype=float)
    steps[:, 1] = speed
    return integrate_angular_steps(steps)


def sliding_windows(
    trajectory: np.ndarray, context_samples: int, horizon_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(trajectory, dtype=float)
    width = context_samples + horizon_samples
    if poses.ndim != 2 or poses.shape[1] != 4 or len(poses) < width:
        raise ValueError("trajectory is shorter than one complete window")
    contexts = []
    targets = []
    for start in range(len(poses) - width + 1):
        contexts.append(poses[start : start + context_samples])
        targets.append(poses[start + context_samples : start + width])
    return np.stack(contexts), np.stack(targets)


def forecast_windows(
    predictor: Callable[[np.ndarray, int], np.ndarray],
    contexts: np.ndarray,
    horizon_samples: int,
) -> np.ndarray:
    return np.stack([predictor(context, horizon_samples) for context in contexts])
