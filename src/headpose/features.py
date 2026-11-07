"""Rotation-invariant causal motion features for learned predictors."""

from __future__ import annotations

import numpy as np

from .quaternion import enforce_sign_continuity, inverse, log_map, multiply


def motion_features(context: np.ndarray) -> np.ndarray:
    """Return [omega, alpha, speed, acceleration norm] at each valid step."""
    poses = enforce_sign_continuity(context)
    if len(poses) < 3:
        raise ValueError("motion features require at least three poses")
    increments = log_map(multiply(inverse(poses[:-1]), poses[1:]))
    acceleration = np.diff(increments, axis=0)
    velocity = increments[1:]
    return np.concatenate(
        [
            velocity,
            acceleration,
            np.linalg.norm(velocity, axis=-1, keepdims=True),
            np.linalg.norm(acceleration, axis=-1, keepdims=True),
        ],
        axis=-1,
    ).astype(np.float32)
