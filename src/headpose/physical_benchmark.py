"""Fair, paired tuning and evaluation of physical forecasting baselines."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterable, Mapping

import numpy as np

from .baselines import (
    constant_acceleration,
    constant_pose,
    constant_velocity,
    damped_velocity,
    linear_velocity_trend,
    so3_error_state_kalman,
    windowed_velocity,
)
from .dataset import WindowDataset
from .quaternion import exp_map, geodesic_degrees, multiply


BASELINE_SEARCH_SPACES: dict[str, list[dict[str, float | int]]] = {
    "constant_pose": [{}],
    "constant_velocity": [{}],
    "windowed_velocity": [
        {"history": history} for history in (3, 5, 10, 20, 40)
    ],
    "damped_velocity": [
        {"damping": damping} for damping in (0.5, 0.7, 0.85, 0.95, 0.98, 0.995)
    ],
    "constant_acceleration": [
        {"acceleration_limit": limit} for limit in (0.001, 0.003, 0.01, 0.03)
    ],
    "linear_velocity_trend": [
        {"history": history, "acceleration_limit": limit}
        for history, limit in itertools.product((5, 10, 20), (0.001, 0.003, 0.01))
    ],
    "so3_error_state_kalman": [
        {
            "angular_acceleration_noise": acceleration_noise,
            "measurement_noise": measurement_noise,
        }
        for acceleration_noise, measurement_noise in itertools.product(
            (0.05, 0.5, 2.0), (0.0005, 0.002, 0.01)
        )
    ],
}


def dataset_fingerprint(dataset: WindowDataset) -> str:
    """Hash paired window contents without exposing user identifiers."""
    digest = hashlib.sha256()
    for array in (dataset.contexts, dataset.targets, dataset.user_indices):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def endpoint_errors(
    name: str,
    parameters: Mapping[str, float | int],
    dataset: WindowDataset,
    *,
    horizon: int = 20,
) -> np.ndarray:
    """Evaluate one physical baseline on the dataset's exact paired windows."""
    functions = {
        "constant_pose": constant_pose,
        "constant_velocity": constant_velocity,
        "windowed_velocity": windowed_velocity,
        "damped_velocity": damped_velocity,
        "constant_acceleration": constant_acceleration,
        "linear_velocity_trend": linear_velocity_trend,
        "so3_error_state_kalman": so3_error_state_kalman,
    }
    if name not in functions:
        raise ValueError(f"unknown baseline: {name}")
    if dataset.contexts.shape[0] != dataset.targets.shape[0]:
        raise ValueError("contexts and targets must contain identical windows")
    target_relative = exp_map(dataset.targets[:, horizon - 1])
    target_absolute = multiply(dataset.contexts[:, -1], target_relative)
    function = functions[name]
    predictions = np.stack(
        [
            function(context, horizon=horizon, **parameters)[-1]
            for context in dataset.contexts
        ]
    )
    return geodesic_degrees(predictions, target_absolute)


def user_macro_mean(errors: np.ndarray, user_indices: np.ndarray) -> float:
    values = np.asarray(errors, dtype=float).reshape(-1)
    users = np.asarray(user_indices).reshape(-1)
    if len(values) != len(users) or len(values) == 0:
        raise ValueError("errors and user_indices must be non-empty and aligned")
    return float(np.mean([values[users == user].mean() for user in np.unique(users)]))


def tune_physical_baselines(
    validation: WindowDataset,
    *,
    search_spaces: Mapping[str, Iterable[Mapping[str, float | int]]] | None = None,
    horizon: int = 20,
) -> dict[str, object]:
    """Select every method on validation user-macro error only."""
    spaces = BASELINE_SEARCH_SPACES if search_spaces is None else search_spaces
    methods: dict[str, object] = {}
    for name, configurations in spaces.items():
        trials = []
        for configuration in configurations:
            parameters = dict(configuration)
            errors = endpoint_errors(name, parameters, validation, horizon=horizon)
            trials.append(
                {
                    "parameters": parameters,
                    "user_macro_mean_degrees": user_macro_mean(
                        errors, validation.user_indices
                    ),
                    "window_mean_degrees": float(errors.mean()),
                }
            )
        if not trials:
            raise ValueError(f"baseline {name} has an empty search space")
        best = min(trials, key=lambda trial: trial["user_macro_mean_degrees"])
        methods[name] = {"selected": best, "trials": trials}
    best_name = min(
        methods,
        key=lambda name: methods[name]["selected"]["user_macro_mean_degrees"],
    )
    return {
        "selection_metric": "validation_user_macro_mean_degrees_at_200ms",
        "validation_fingerprint": dataset_fingerprint(validation),
        "best_method": best_name,
        "methods": methods,
    }
