#!/usr/bin/env python3
"""Smoke benchmark for the complete geometry-to-evaluation path."""

from __future__ import annotations

import json
from dataclasses import asdict

from headpose.baselines import (
    constant_acceleration,
    constant_pose,
    constant_velocity,
    damped_velocity,
)
from headpose.evaluation import evaluate_horizon
from headpose.synthetic import forecast_windows, make_trajectory, sliding_windows


def main() -> None:
    predictors = {
        "constant_pose": constant_pose,
        "constant_velocity": constant_velocity,
        "damped_velocity": damped_velocity,
        "constant_acceleration": constant_acceleration,
    }
    report: dict[str, dict[str, dict[str, float | int]]] = {}
    for regime in ("stationary", "steady", "accelerating", "decelerating", "reversal"):
        contexts, targets = sliding_windows(make_trajectory(regime), 100, 20)
        report[regime] = {}
        for name, predictor in predictors.items():
            predictions = forecast_windows(predictor, contexts, 20)
            summary = evaluate_horizon(predictions, targets)["primary"]
            report[regime][name] = asdict(summary)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
