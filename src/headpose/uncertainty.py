"""Validation-only scaling and diagnostics for diagonal Gaussian forecasts."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


def fit_variance_scale(
    predicted_variance: np.ndarray,
    residual_error: np.ndarray,
    *,
    minimum: float = 1e-6,
    maximum: float = 1e6,
) -> float:
    """Fit one positive variance multiplier by Gaussian maximum likelihood."""
    variance = np.asarray(predicted_variance, dtype=float)
    error = np.asarray(residual_error, dtype=float)
    if variance.shape != error.shape or variance.size == 0:
        raise ValueError("predicted_variance and residual_error must align")
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("predicted_variance must be finite and positive")
    scale = float(np.mean(np.square(error) / variance))
    return float(np.clip(scale, minimum, maximum))


def gaussian_nll(
    predicted_variance: np.ndarray,
    residual_error: np.ndarray,
    *,
    variance_scale: float = 1.0,
) -> float:
    """Mean diagonal-Gaussian negative log likelihood, including constants."""
    variance = np.asarray(predicted_variance, dtype=float) * variance_scale
    error = np.asarray(residual_error, dtype=float)
    if variance.shape != error.shape or variance.size == 0:
        raise ValueError("predicted_variance and residual_error must align")
    if variance_scale <= 0.0 or np.any(variance <= 0.0):
        raise ValueError("calibrated variance must be positive")
    values = 0.5 * (
        np.log(2.0 * np.pi * variance) + np.square(error) / variance
    )
    return float(np.mean(values))


def interval_calibration(
    predicted_variance: np.ndarray,
    residual_error: np.ndarray,
    *,
    variance_scale: float = 1.0,
    levels: tuple[float, ...] = (0.5, 0.68, 0.8, 0.9, 0.95),
) -> dict[str, object]:
    """Report component-wise central interval coverage and mean absolute gap."""
    variance = np.asarray(predicted_variance, dtype=float) * variance_scale
    error = np.asarray(residual_error, dtype=float)
    if variance.shape != error.shape or variance.size == 0:
        raise ValueError("predicted_variance and residual_error must align")
    if variance_scale <= 0.0 or np.any(variance <= 0.0):
        raise ValueError("calibrated variance must be positive")
    standardized = np.abs(error) / np.sqrt(variance)
    normal = NormalDist()
    coverage: dict[str, float] = {}
    gaps: list[float] = []
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError("levels must lie strictly between zero and one")
        threshold = normal.inv_cdf((1.0 + level) / 2.0)
        observed = float(np.mean(standardized <= threshold))
        coverage[f"{int(round(level * 100))}%"] = observed
        gaps.append(abs(observed - level))
    return {
        "component_coverage": coverage,
        "mean_absolute_coverage_error": float(np.mean(gaps)),
    }
