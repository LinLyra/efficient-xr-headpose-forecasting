import numpy as np
import pytest

from headpose.dataset import WindowDataset, tangent_targets
from headpose.physical_benchmark import (
    dataset_fingerprint,
    endpoint_errors,
    tune_physical_baselines,
)
from headpose.quaternion import exp_map


def constant_rate_dataset() -> WindowDataset:
    contexts = []
    targets = []
    for _ in range(4):
        angles = np.arange(8, dtype=float) * 0.01
        poses = exp_map(np.column_stack([np.zeros(8), np.zeros(8), angles]))
        contexts.append(poses[:5])
        targets.append(tangent_targets(poses[:5], poses[5:]))
    return WindowDataset(
        contexts=np.asarray(contexts, dtype=np.float32),
        features=np.zeros((4, 3, 8), dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        user_indices=np.array([0, 0, 1, 1]),
        users=("a", "b"),
    )


def test_paired_endpoint_errors_use_absolute_target_orientation() -> None:
    dataset = constant_rate_dataset()
    errors = endpoint_errors("constant_velocity", {}, dataset, horizon=3)
    assert errors.max() < 1e-5


def test_validation_tuning_selects_constant_velocity_on_constant_rate() -> None:
    dataset = constant_rate_dataset()
    result = tune_physical_baselines(
        dataset,
        horizon=3,
        search_spaces={"constant_pose": [{}], "constant_velocity": [{}]},
    )
    assert result["best_method"] == "constant_velocity"
    assert len(result["validation_fingerprint"]) == 64


def test_dataset_fingerprint_changes_with_window_content() -> None:
    first = constant_rate_dataset()
    second = constant_rate_dataset()
    second.contexts[0, 0] = exp_map(np.array([0.1, 0.0, 0.0]))
    assert dataset_fingerprint(first) != dataset_fingerprint(second)


def test_endpoint_errors_reject_unknown_baseline() -> None:
    with pytest.raises(ValueError, match="unknown"):
        endpoint_errors("invented", {}, constant_rate_dataset(), horizon=3)
