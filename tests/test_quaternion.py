import numpy as np
import pytest

from headpose.baselines import constant_pose, constant_velocity
from headpose.quaternion import geodesic_degrees, inverse, multiply


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

