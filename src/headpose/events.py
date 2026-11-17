"""Training-fitted, frozen motion-event definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .quaternion import enforce_sign_continuity, inverse, log_map, multiply


@dataclass(frozen=True)
class EventThresholds:
    speed_low: float
    speed_high: float
    acceleration_high: float
    jerk_high: float
    dt: float = 0.01
    recent_samples: int = 10

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _kinematics(context: np.ndarray, dt: float) -> tuple[np.ndarray, ...]:
    poses = enforce_sign_continuity(context)
    if len(poses) < 4 or dt <= 0.0:
        raise ValueError("events require at least four poses and positive dt")
    omega = log_map(multiply(inverse(poses[:-1]), poses[1:])) / dt
    alpha = np.diff(omega, axis=0) / dt
    jerk = np.diff(alpha, axis=0) / dt
    return omega, alpha, jerk


def _recent_rms(values: np.ndarray, samples: int) -> float:
    tail = values[-min(len(values), samples) :]
    return float(np.sqrt(np.mean(np.sum(tail * tail, axis=-1))))


def fit_event_thresholds(
    training_contexts: np.ndarray,
    *,
    dt: float = 0.01,
    recent_samples: int = 10,
) -> EventThresholds:
    """Fit event quantiles from training contexts only."""
    contexts = np.asarray(training_contexts, dtype=float)
    if contexts.ndim != 3 or contexts.shape[-1] != 4 or contexts.shape[1] < 4:
        raise ValueError("training_contexts must have shape [windows, time>=4, 4]")
    if recent_samples < 2:
        raise ValueError("recent_samples must be at least two")
    summaries = []
    for context in contexts:
        omega, alpha, jerk = _kinematics(context, dt)
        summaries.append(
            (
                _recent_rms(omega, recent_samples),
                _recent_rms(alpha, recent_samples),
                _recent_rms(jerk, recent_samples),
            )
        )
    values = np.asarray(summaries)
    return EventThresholds(
        speed_low=float(np.quantile(values[:, 0], 0.25)),
        speed_high=float(np.quantile(values[:, 0], 0.75)),
        acceleration_high=float(np.quantile(values[:, 1], 0.75)),
        jerk_high=float(np.quantile(values[:, 2], 0.75)),
        dt=dt,
        recent_samples=recent_samples,
    )


def classify_motion_event(context: np.ndarray, thresholds: EventThresholds) -> str:
    """Assign one exclusive frozen-protocol event label."""
    omega, alpha, jerk = _kinematics(context, thresholds.dt)
    samples = thresholds.recent_samples
    recent_omega = omega[-min(len(omega), samples) :]
    speeds = np.linalg.norm(recent_omega, axis=-1)
    speed_rms = _recent_rms(omega, samples)
    acceleration_rms = _recent_rms(alpha, samples)
    jerk_rms = _recent_rms(jerk, samples)

    last = omega[-1]
    previous = omega[-2]
    denominator = np.linalg.norm(last) * np.linalg.norm(previous)
    cosine = 1.0 if denominator <= 1e-12 else float(np.dot(last, previous) / denominator)
    if (
        cosine < -0.5
        and np.linalg.norm(last) > thresholds.speed_low
        and np.linalg.norm(previous) > thresholds.speed_low
    ):
        return "reversal"
    peak = float(np.max(speeds))
    if peak > thresholds.speed_high and speeds[-1] <= 0.5 * peak:
        return "braking"
    if (
        acceleration_rms > thresholds.acceleration_high
        or jerk_rms > thresholds.jerk_high
    ):
        return "acceleration_onset"
    if speed_rms > thresholds.speed_high:
        return "sustained_fast_turn"
    if (
        speed_rms <= thresholds.speed_low
        and acceleration_rms < thresholds.acceleration_high
    ):
        return "steady"
    return "other"


def is_high_dynamic(label: str) -> bool:
    return label in {
        "reversal",
        "braking",
        "acceleration_onset",
        "sustained_fast_turn",
    }
