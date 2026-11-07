"""Efficient causal head-pose forecasting."""

from .baselines import (
    constant_pose,
    constant_velocity,
    linear_velocity_trend,
    smoothed_velocity,
    windowed_velocity,
)
from .boxrr import HeadTrajectory, extract_head_trajectory
from .quaternion import geodesic_degrees, inverse, multiply, normalize

__all__ = [
    "HeadTrajectory",
    "constant_pose",
    "constant_velocity",
    "linear_velocity_trend",
    "smoothed_velocity",
    "windowed_velocity",
    "extract_head_trajectory",
    "geodesic_degrees",
    "inverse",
    "multiply",
    "normalize",
]

