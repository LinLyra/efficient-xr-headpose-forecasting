import numpy as np

from headpose.baselines import constant_pose, constant_velocity
from headpose.evaluation import evaluate_horizon
from headpose.synthetic import forecast_windows, make_trajectory, sliding_windows


def test_constant_velocity_is_exact_on_steady_synthetic_motion() -> None:
    contexts, targets = sliding_windows(make_trajectory("steady"), 100, 20)
    predictions = forecast_windows(constant_velocity, contexts, 20)
    assert evaluate_horizon(predictions, targets)["primary"].mean < 1e-8


def test_constant_pose_is_exact_on_stationary_synthetic_motion() -> None:
    contexts, targets = sliding_windows(make_trajectory("stationary"), 100, 20)
    predictions = forecast_windows(constant_pose, contexts, 20)
    assert evaluate_horizon(predictions, targets)["primary"].mean < 1e-8


def test_reversal_trajectory_changes_direction() -> None:
    trajectory = make_trajectory("reversal")
    assert not np.allclose(trajectory[90], trajectory[120])
