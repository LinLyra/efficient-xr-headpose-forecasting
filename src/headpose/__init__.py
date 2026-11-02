"""Efficient causal head-pose forecasting."""

from .baselines import constant_pose, constant_velocity
from .quaternion import geodesic_degrees, inverse, multiply, normalize

__all__ = [
    "constant_pose",
    "constant_velocity",
    "geodesic_degrees",
    "inverse",
    "multiply",
    "normalize",
]

