import numpy as np
import pytest

from headpose.evaluation import evaluate_horizon, motion_regime
from headpose.quaternion import exp_map, geodesic_degrees
from headpose.resample import contiguous_slices, uniform_resample


def test_uniform_resample_preserves_constant_rate_motion() -> None:
    times = np.array([0.0, 0.02])
    poses = np.stack([exp_map(np.zeros(3)), exp_map(np.array([0.0, 0.0, 0.2]))])
    grid, resampled = uniform_resample(times, poses, sampling_hz=100.0)
    assert grid == pytest.approx([0.0, 0.01, 0.02])
    assert geodesic_degrees(resampled[1], exp_map(np.array([0.0, 0.0, 0.1]))) < 1e-6


def test_uniform_resample_vectorizes_multiple_intervals() -> None:
    times = np.array([0.0, 0.02, 0.04])
    angles = np.array([0.0, 0.2, 0.4])
    poses = exp_map(np.column_stack([np.zeros(3), np.zeros(3), angles]))
    grid, resampled = uniform_resample(times, poses, sampling_hz=100.0)
    assert len(grid) == len(resampled) == 5
    assert geodesic_degrees(resampled[3], exp_map(np.array([0.0, 0.0, 0.3]))) < 1e-6


def test_uniform_resample_rejects_large_gap() -> None:
    times = np.array([0.0, 0.2])
    poses = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
    with pytest.raises(ValueError, match="gap"):
        uniform_resample(times, poses)


def test_contiguous_slices_partition_large_and_nonpositive_gaps() -> None:
    times = np.array([0.0, 0.01, 0.02, 0.5, 0.51, 0.50, 0.52])
    slices = contiguous_slices(times, max_gap_seconds=0.1, min_samples=2)
    assert [(item.start, item.stop) for item in slices] == [(0, 3), (3, 5), (5, 7)]


def test_evaluate_horizon_uses_final_step_as_primary() -> None:
    targets = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 3, 1))
    predictions = targets.copy()
    predictions[:, -1] = exp_map(np.array([0.0, 0.0, np.radians(10.0)]))
    report = evaluate_horizon(predictions, targets)
    assert report["primary"].mean == pytest.approx(10.0)
    assert report["per_step_mean"] == pytest.approx([0.0, 0.0, 10.0])


def test_motion_regime_detects_stationary() -> None:
    context = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1))
    assert motion_regime(context) == "stationary"
