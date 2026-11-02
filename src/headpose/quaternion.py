"""Numerically stable quaternion operations using scalar-first order."""

from __future__ import annotations

import numpy as np


def normalize(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("zero-norm quaternion")
    return values / norm


def inverse(quaternion: np.ndarray) -> np.ndarray:
    values = normalize(quaternion)
    result = values.copy()
    result[..., 1:] *= -1
    return result


def multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def geodesic_degrees(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return sign-invariant SO(3) geodesic error in degrees."""
    predicted = normalize(prediction)
    expected = normalize(target)
    dot = np.sum(predicted * expected, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))

