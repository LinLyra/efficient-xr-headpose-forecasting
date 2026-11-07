"""Evaluation primitives matching the public challenge metric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quaternion import geodesic_degrees, inverse, log_map, multiply


@dataclass(frozen=True)
class ErrorSummary:
    mean: float
    median: float
    p90: float
    p95: float
    p99: float
    count: int


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    clusters: int


def summarize_errors(errors: np.ndarray) -> ErrorSummary:
    values = np.asarray(errors, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("no finite errors")
    return ErrorSummary(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p90=float(np.percentile(values, 90)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        count=len(values),
    )


def evaluate_horizon(predictions: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    predicted = np.asarray(predictions, dtype=float)
    expected = np.asarray(targets, dtype=float)
    if predicted.shape != expected.shape or predicted.ndim != 3 or predicted.shape[-1] != 4:
        raise ValueError("predictions and targets must share shape [windows, horizon, 4]")
    errors = geodesic_degrees(predicted, expected)
    return {
        "primary": summarize_errors(errors[:, -1]),
        "per_step_mean": np.mean(errors, axis=0),
        "all_steps": summarize_errors(errors),
    }


def motion_regime(context: np.ndarray, threshold_deg: float = 1.0) -> str:
    """Deterministic coarse regime label for stratified error analysis."""
    values = np.asarray(context, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) < 4:
        raise ValueError("context must have shape [time>=4, 4]")
    deltas = multiply(inverse(values[:-1]), values[1:])
    velocities = np.degrees(log_map(deltas))
    speed = np.linalg.norm(velocities, axis=-1)
    acceleration = np.diff(velocities, axis=0)
    recent_speed = float(np.mean(speed[-3:]))
    recent_acceleration = float(np.mean(np.linalg.norm(acceleration[-2:], axis=-1)))
    if recent_speed < threshold_deg:
        return "stationary"
    if recent_acceleration < threshold_deg * 0.5:
        return "steady"
    if np.dot(velocities[-1], velocities[-2]) < 0.0:
        return "reversal"
    if np.linalg.norm(velocities[-1]) < np.linalg.norm(velocities[-2]):
        return "decelerating"
    return "accelerating"


def clustered_bootstrap_mean(
    errors: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 2027,
) -> ConfidenceInterval:
    """Bootstrap users, not correlated windows, for a mean-error interval."""
    values = np.asarray(errors, dtype=float).reshape(-1)
    clusters = np.asarray(cluster_ids).reshape(-1)
    if len(values) != len(clusters) or len(values) == 0:
        raise ValueError("errors and cluster_ids must be non-empty and aligned")
    if not 0.0 < confidence < 1.0 or resamples < 100:
        raise ValueError("invalid confidence or resample count")
    unique = np.unique(clusters)
    cluster_means = np.array([np.mean(values[clusters == item]) for item in unique])
    generator = np.random.default_rng(seed)
    draws = generator.choice(cluster_means, size=(resamples, len(unique)), replace=True)
    bootstrap = np.mean(draws, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        estimate=float(np.mean(cluster_means)),
        lower=float(np.quantile(bootstrap, alpha)),
        upper=float(np.quantile(bootstrap, 1.0 - alpha)),
        confidence=confidence,
        clusters=len(unique),
    )


def paired_clustered_bootstrap_difference(
    candidate_errors: np.ndarray,
    reference_errors: np.ndarray,
    cluster_ids: np.ndarray,
    **kwargs: object,
) -> ConfidenceInterval:
    """CI for candidate minus reference; negative values favor candidate."""
    candidate = np.asarray(candidate_errors, dtype=float)
    reference = np.asarray(reference_errors, dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError("paired error arrays must have identical shapes")
    return clustered_bootstrap_mean(candidate - reference, cluster_ids, **kwargs)
