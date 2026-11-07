import numpy as np
import pytest

from headpose.baselines import (
    constant_acceleration,
    constant_pose,
    constant_velocity,
    damped_velocity,
    linear_velocity_trend,
    smoothed_velocity,
    windowed_velocity,
)
from headpose.quaternion import (
    enforce_sign_continuity,
    exp_map,
    geodesic_degrees,
    inverse,
    log_map,
    multiply,
)


def test_geodesic_is_sign_invariant() -> None:
    quaternion = np.array([0.5, 0.5, 0.5, 0.5])
    assert geodesic_degrees(quaternion, -quaternion) == pytest.approx(0.0)


def test_inverse_product_is_identity() -> None:
    quaternion = np.array([0.9238795, 0.0, 0.3826834, 0.0])
    product = multiply(quaternion, inverse(quaternion))
    assert product == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_constant_pose_shape() -> None:
    context = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    assert constant_pose(context, 20).shape == (20, 4)


def test_constant_velocity_advances_rotation() -> None:
    context = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.9999619, 0.0, 0.0087265, 0.0]]
    )
    forecast = constant_velocity(context, 2)
    assert geodesic_degrees(forecast[0], context[-1]) > 0
    assert forecast.shape == (2, 4)


def test_exp_log_round_trip() -> None:
    vector = np.array([0.1, -0.2, 0.3])
    assert log_map(exp_map(vector)) == pytest.approx(vector, abs=1e-10)


def test_sign_continuity_removes_flips() -> None:
    poses = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    fixed = enforce_sign_continuity(poses)
    assert np.dot(fixed[0], fixed[1]) > 0.0


def test_damped_velocity_moves_less_than_constant_velocity() -> None:
    context = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.9999619, 0.0, 0.0087265, 0.0]]
    )
    constant = constant_velocity(context, 5)
    damped = damped_velocity(context, 5, damping=0.5)
    assert geodesic_degrees(damped[-1], context[-1]) < geodesic_degrees(
        constant[-1], context[-1]
    )


def test_constant_acceleration_requires_three_samples() -> None:
    context = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        constant_acceleration(context)


def test_robust_velocity_baselines_have_expected_shape() -> None:
    steps = np.arange(12, dtype=float) * np.radians(0.5)
    context = np.stack(
        [np.cos(steps / 2.0), np.zeros(12), np.sin(steps / 2.0), np.zeros(12)],
        axis=-1,
    )
    assert smoothed_velocity(context, 20).shape == (20, 4)
    assert windowed_velocity(context, 20).shape == (20, 4)
    assert linear_velocity_trend(context, 20).shape == (20, 4)

