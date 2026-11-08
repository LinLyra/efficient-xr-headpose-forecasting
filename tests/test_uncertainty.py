import numpy as np
import pytest

from headpose.uncertainty import (
    fit_variance_scale,
    gaussian_nll,
    interval_calibration,
)


def test_variance_scale_recovers_known_miscalibration() -> None:
    generator = np.random.default_rng(2027)
    variance = np.full((50_000, 3), 0.25)
    error = generator.normal(scale=1.0, size=variance.shape)
    scale = fit_variance_scale(variance, error)
    assert scale == pytest.approx(4.0, rel=0.02)


def test_scaling_improves_nll_and_interval_calibration() -> None:
    generator = np.random.default_rng(7)
    variance = np.full((50_000, 3), 0.25)
    error = generator.normal(scale=1.0, size=variance.shape)
    scale = fit_variance_scale(variance, error)
    assert gaussian_nll(variance, error, variance_scale=scale) < gaussian_nll(
        variance, error
    )
    report = interval_calibration(variance, error, variance_scale=scale)
    assert report["component_coverage"]["95%"] == pytest.approx(0.95, abs=0.01)
    assert report["mean_absolute_coverage_error"] < 0.01


def test_uncertainty_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        fit_variance_scale(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="positive"):
        gaussian_nll(np.ones(2), np.ones(2), variance_scale=0.0)
