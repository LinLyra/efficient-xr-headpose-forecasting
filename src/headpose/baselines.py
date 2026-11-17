"""Non-learned forecasting baselines required for credible evaluation."""

from __future__ import annotations

import numpy as np

from .quaternion import exp_map, inverse, log_map, multiply, normalize


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


def damped_velocity(
    context: np.ndarray, horizon: int = 20, damping: float = 0.98
) -> np.ndarray:
    """Exponentially damp the latest angular displacement over the horizon."""
    values = _validate_context(context)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0.0 <= damping <= 1.0:
        raise ValueError("damping must be in [0, 1]")
    velocity = log_map(multiply(inverse(values[-2]), values[-1]))
    forecasts: list[np.ndarray] = []
    current = values[-1]
    for step in range(horizon):
        current = normalize(multiply(current, exp_map((damping**step) * velocity)))
        forecasts.append(current)
    return np.stack(forecasts)


def constant_acceleration(
    context: np.ndarray, horizon: int = 20, acceleration_limit: float = 0.05
) -> np.ndarray:
    """Extrapolate angular velocity and a clipped per-sample acceleration."""
    values = _validate_context(context)
    if len(values) < 3:
        raise ValueError("constant acceleration requires at least 3 samples")
    if horizon < 1 or acceleration_limit <= 0.0:
        raise ValueError("invalid horizon or acceleration limit")
    previous = log_map(multiply(inverse(values[-3]), values[-2]))
    velocity = log_map(multiply(inverse(values[-2]), values[-1]))
    acceleration = velocity - previous
    norm = np.linalg.norm(acceleration)
    if norm > acceleration_limit:
        acceleration *= acceleration_limit / norm
    forecasts: list[np.ndarray] = []
    current = values[-1]
    for _ in range(horizon):
        velocity = velocity + acceleration
        current = normalize(multiply(current, exp_map(velocity)))
        forecasts.append(current)
    return np.stack(forecasts)


def smoothed_velocity(
    context: np.ndarray, horizon: int = 20, history: int = 5
) -> np.ndarray:
    """Extrapolate a robust median of recent tangent-space increments."""
    values = _validate_context(context)
    if horizon < 1 or history < 1:
        raise ValueError("horizon and history must be positive")
    tail = values[-min(len(values), history + 1) :]
    increments = log_map(multiply(inverse(tail[:-1]), tail[1:]))
    velocity = np.median(increments, axis=0)
    forecasts: list[np.ndarray] = []
    current = values[-1]
    for _ in range(horizon):
        current = normalize(multiply(current, exp_map(velocity)))
        forecasts.append(current)
    return np.stack(forecasts)


def windowed_velocity(
    context: np.ndarray, horizon: int = 20, history: int = 10
) -> np.ndarray:
    """Estimate one average body-frame increment over a longer recent window."""
    values = _validate_context(context)
    if horizon < 1 or history < 1:
        raise ValueError("horizon and history must be positive")
    lag = min(history, len(values) - 1)
    total_rotation = log_map(multiply(inverse(values[-lag - 1]), values[-1]))
    increment = total_rotation / lag
    forecasts: list[np.ndarray] = []
    current = values[-1]
    for _ in range(horizon):
        current = normalize(multiply(current, exp_map(increment)))
        forecasts.append(current)
    return np.stack(forecasts)


def linear_velocity_trend(
    context: np.ndarray,
    horizon: int = 20,
    history: int = 10,
    acceleration_limit: float = 0.01,
) -> np.ndarray:
    """Fit a causal linear trend to recent body-frame angular increments."""
    values = _validate_context(context)
    if horizon < 1 or history < 2 or acceleration_limit <= 0.0:
        raise ValueError("invalid horizon, history or acceleration limit")
    tail = values[-min(len(values), history + 1) :]
    increments = log_map(multiply(inverse(tail[:-1]), tail[1:]))
    time = np.arange(len(increments), dtype=float)
    design = np.column_stack([np.ones_like(time), time])
    coefficients = np.linalg.lstsq(design, increments, rcond=None)[0]
    velocity = design[-1] @ coefficients
    acceleration = coefficients[1]
    norm = np.linalg.norm(acceleration)
    if norm > acceleration_limit:
        acceleration *= acceleration_limit / norm
    forecasts: list[np.ndarray] = []
    current = values[-1]
    for _ in range(horizon):
        velocity = velocity + acceleration
        current = normalize(multiply(current, exp_map(velocity)))
        forecasts.append(current)
    return np.stack(forecasts)


def so3_error_state_kalman(
    context: np.ndarray,
    horizon: int = 20,
    *,
    dt: float = 0.01,
    orientation_process_noise: float = 1e-4,
    angular_acceleration_noise: float = 0.5,
    measurement_noise: float = 2e-3,
) -> np.ndarray:
    """Forecast with a causal SO(3) error-state constant-velocity filter.

    The nominal state is orientation and body-frame angular velocity. The
    six-dimensional covariance covers their local errors. Quaternion
    measurements update the nominal orientation through a tangent-space
    innovation; no future measurement is used during forecasting.
    """
    values = _validate_context(context)
    if horizon < 1 or dt <= 0.0:
        raise ValueError("horizon and dt must be positive")
    noises = (
        orientation_process_noise,
        angular_acceleration_noise,
        measurement_noise,
    )
    if any(value <= 0.0 for value in noises):
        raise ValueError("Kalman noise parameters must be positive")

    orientation = values[0]
    velocity = log_map(multiply(inverse(values[0]), values[1])) / dt
    covariance = np.eye(6, dtype=float) * measurement_noise**2
    transition = np.eye(6, dtype=float)
    transition[:3, 3:] = np.eye(3) * dt
    measurement = np.zeros((3, 6), dtype=float)
    measurement[:, :3] = np.eye(3)
    measurement_covariance = np.eye(3) * measurement_noise**2
    process_covariance = np.zeros((6, 6), dtype=float)
    process_covariance[:3, :3] = (
        np.eye(3) * orientation_process_noise**2 * dt
    )
    process_covariance[3:, 3:] = (
        np.eye(3) * angular_acceleration_noise**2 * dt
    )
    identity = np.eye(6, dtype=float)

    for observed in values[1:]:
        orientation = normalize(multiply(orientation, exp_map(velocity * dt)))
        covariance = transition @ covariance @ transition.T + process_covariance
        innovation = log_map(multiply(inverse(orientation), observed))
        innovation_covariance = (
            measurement @ covariance @ measurement.T + measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance, measurement @ covariance
        ).T
        correction = gain @ innovation
        orientation = normalize(multiply(orientation, exp_map(correction[:3])))
        velocity = velocity + correction[3:]
        residual = identity - gain @ measurement
        covariance = (
            residual @ covariance @ residual.T
            + gain @ measurement_covariance @ gain.T
        )

    forecasts: list[np.ndarray] = []
    for _ in range(horizon):
        orientation = normalize(multiply(orientation, exp_map(velocity * dt)))
        forecasts.append(orientation)
    return np.stack(forecasts)

