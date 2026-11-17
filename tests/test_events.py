import numpy as np

from headpose.events import EventThresholds, classify_motion_event, fit_event_thresholds
from headpose.quaternion import exp_map, multiply


def poses_from_increments(increments: np.ndarray) -> np.ndarray:
    poses = [np.array([1.0, 0.0, 0.0, 0.0])]
    for increment in increments:
        poses.append(multiply(poses[-1], exp_map(increment)))
    return np.stack(poses)


def test_event_thresholds_are_training_fitted_and_serializable() -> None:
    contexts = []
    for speed in (0.001, 0.002, 0.01, 0.02):
        increments = np.tile(np.array([0.0, 0.0, speed]), (12, 1))
        contexts.append(poses_from_increments(increments))
    thresholds = fit_event_thresholds(np.stack(contexts), recent_samples=5)
    assert thresholds.speed_low < thresholds.speed_high
    assert thresholds.acceleration_high >= 0.0


def test_event_classifier_detects_reversal_before_acceleration() -> None:
    increments = np.tile(np.array([0.0, 0.0, 0.02]), (11, 1))
    increments[-1] *= -1.0
    context = poses_from_increments(increments)
    thresholds = EventThresholds(
        speed_low=0.5,
        speed_high=1.5,
        acceleration_high=10.0,
        jerk_high=100.0,
        recent_samples=5,
    )
    assert classify_motion_event(context, thresholds) == "reversal"


def test_event_classifier_detects_steady_motion() -> None:
    context = poses_from_increments(np.zeros((12, 3)))
    thresholds = EventThresholds(
        speed_low=0.1,
        speed_high=1.0,
        acceleration_high=1.0,
        jerk_high=10.0,
        recent_samples=5,
    )
    assert classify_motion_event(context, thresholds) == "steady"
