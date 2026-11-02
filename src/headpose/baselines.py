"""Non-learned forecasting baselines required for credible evaluation."""

from __future__ import annotations

import numpy as np

from .quaternion import inverse, multiply, normalize


def _validate_context(context: np.ndarray) -> np.ndarray:
    values = normalize(np.asarray(context, dtype=float))
    if values.ndim != 2 or values.shape[1] != 4 or len(values) < 2:
        raise ValueError("context must have shape [time>=2, 4]")
    return values


def constant_pose(context: np.ndarray, horizon: int = 20) -> np.ndarray:
    values = _validate_context(context)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return np.repeat(values[-1][None, :], horizon, axis=0)


def constant_velocity(context: np.ndarray, horizon: int = 20) -> np.ndarray:
    """Extrapolate the most recent relative rotation at every future step."""
    values = _validate_context(context)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    delta = normalize(multiply(inverse(values[-2]), values[-1]))
    forecasts: list[np.ndarray] = []
    current = values[-1]
    for _ in range(horizon):
        current = normalize(multiply(current, delta))
        forecasts.append(current)
    return np.stack(forecasts)

