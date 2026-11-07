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


def enforce_sign_continuity(quaternions: np.ndarray) -> np.ndarray:
    """Choose equivalent signs so adjacent samples stay in one hemisphere."""
    values = normalize(np.asarray(quaternions, dtype=float))
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("quaternions must have shape [time, 4]")
    result = values.copy()
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def exp_map(rotation_vectors: np.ndarray) -> np.ndarray:
    """Map axis-angle vectors in so(3) to scalar-first unit quaternions."""
    vectors = np.asarray(rotation_vectors, dtype=float)
    if vectors.shape[-1] != 3:
        raise ValueError("rotation vectors must have final dimension 3")
    angle = np.linalg.norm(vectors, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.empty_like(angle)
    small = angle < 1e-8
    scale[small] = 0.5 - (angle[small] ** 2) / 48.0
    scale[~small] = np.sin(half[~small]) / angle[~small]
    return normalize(np.concatenate([np.cos(half), vectors * scale], axis=-1))


def log_map(quaternions: np.ndarray) -> np.ndarray:
    """Map unit quaternions to principal axis-angle vectors in so(3)."""
    values = normalize(np.asarray(quaternions, dtype=float))
    if values.shape[-1] != 4:
        raise ValueError("quaternions must have final dimension 4")
    values = np.where(values[..., :1] < 0.0, -values, values)
    vector = values[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(values[..., :1], 0.0, 1.0))
    scale = np.empty_like(vector_norm)
    small = vector_norm < 1e-8
    scale[small] = 2.0
    scale[~small] = angle[~small] / vector_norm[~small]
    return vector * scale


def slerp(left: np.ndarray, right: np.ndarray, fraction: np.ndarray | float) -> np.ndarray:
    """Spherical interpolation using the shortest rotation arc."""
    start = normalize(np.asarray(left, dtype=float))
    end = normalize(np.asarray(right, dtype=float))
    if start.shape[-1] != 4 or end.shape[-1] != 4:
        raise ValueError("slerp endpoints must have final dimension 4")
    start, end = np.broadcast_arrays(start, end)
    delta = multiply(inverse(start), end)
    vector = log_map(delta)
    fractions = np.asarray(fraction, dtype=float)
    increments = exp_map(fractions[..., None] * vector)
    return normalize(multiply(start, increments))


def geodesic_degrees(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return sign-invariant SO(3) geodesic error in degrees."""
    predicted = normalize(prediction)
    expected = normalize(target)
    dot = np.sum(predicted * expected, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))

