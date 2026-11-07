"""Audit a processed BOXRR pilot and run non-learned real-data baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from headpose.baselines import (
    constant_pose,
    constant_velocity,
    damped_velocity,
    smoothed_velocity,
    windowed_velocity,
)
from headpose.evaluation import summarize_errors
from headpose.quaternion import geodesic_degrees
from headpose.resample import contiguous_slices, uniform_resample
from headpose.windows import make_window_indices, materialize_window

BASELINES = {
    "constant_pose": constant_pose,
    "constant_velocity": constant_velocity,
    "damped_velocity": damped_velocity,
    "smoothed_velocity": smoothed_velocity,
    "windowed_velocity_100ms": windowed_velocity,
}


def bootstrap_user_mean(values: np.ndarray, seed: int = 2027) -> dict[str, float]:
    """Percentile interval from resampling users as independent clusters."""
    user_values = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    draws = generator.choice(
        user_values, size=(5000, len(user_values)), replace=True
    ).mean(axis=1)
    return {
        "estimate": float(np.mean(user_values)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "users": int(len(user_values)),
    }


def audit(root: Path, context: int, horizon: int, stride: int) -> dict[str, object]:
    manifest = json.loads((root / "pilot-manifest.json").read_text())
    sample_rates: list[float] = []
    durations: list[float] = []
    raw_frames = 0
    usable_recordings = 0
    skipped_gaps = 0
    errors: dict[str, list[float]] = {name: [] for name in BASELINES}
    user_errors: dict[str, dict[str, list[float]]] = {}
    regime_errors: dict[str, dict[str, list[float]]] = {
        "low": {name: [] for name in BASELINES},
        "medium": {name: [] for name in BASELINES},
        "high": {name: [] for name in BASELINES},
    }
    for user in manifest["users"]:
        user_hash = user["user_hash"]
        user_errors[user_hash] = {name: [] for name in BASELINES}
        for path in sorted((root / user_hash).glob("*.npz")):
            with np.load(path) as recording:
                timestamps = recording["timestamps"]
                poses = recording["quaternions"]
            raw_frames += len(timestamps)
            gaps = np.diff(timestamps)
            sample_rates.append(float(1.0 / np.median(gaps)))
            durations.append(float(timestamps[-1] - timestamps[0]))
            segments = contiguous_slices(timestamps, max_gap_seconds=0.1, min_samples=8)
            if len(segments) > 1:
                skipped_gaps += len(segments) - 1
            recording_used = False
            for segment_index, segment in enumerate(segments):
                try:
                    _, uniform = uniform_resample(
                        timestamps[segment], poses[segment], sampling_hz=100.0
                    )
                except ValueError:
                    continue
                indices = make_window_indices(
                    f"{path.stem}-{segment_index}",
                    len(uniform),
                    context,
                    horizon,
                    stride,
                )
                if indices:
                    recording_used = True
                for index in indices:
                    history, target = materialize_window(uniform, index)
                    recent_change = float(geodesic_degrees(history[-11], history[-1]))
                    regime = (
                        "low"
                        if recent_change < 1.0
                        else "medium"
                        if recent_change < 5.0
                        else "high"
                    )
                    for name, predictor in BASELINES.items():
                        prediction = predictor(history, horizon=horizon)
                        error = float(geodesic_degrees(prediction[-1], target[-1]))
                        errors[name].append(error)
                        user_errors[user_hash][name].append(error)
                        regime_errors[regime][name].append(error)
            usable_recordings += int(recording_used)

    baseline_report: dict[str, object] = {}
    macro_by_method: dict[str, np.ndarray] = {}
    for name, values in errors.items():
        summary = asdict(summarize_errors(np.asarray(values)))
        per_user = np.asarray([
            float(np.mean(user_errors[user][name]))
            for user in user_errors
            if user_errors[user][name]
        ])
        summary["user_macro_mean"] = float(np.mean(per_user))
        summary["user_bootstrap_95"] = bootstrap_user_mean(per_user)
        baseline_report[name] = summary
        macro_by_method[name] = per_user
    reference = macro_by_method["constant_pose"]
    for name, per_user in macro_by_method.items():
        baseline_report[name]["paired_delta_vs_constant_pose"] = bootstrap_user_mean(
            per_user - reference
        )
    regime_report = {
        regime: {
            name: asdict(summarize_errors(np.asarray(values)))
            for name, values in methods.items()
            if values
        }
        for regime, methods in regime_errors.items()
    }
    return {
        "status": "pilot_only_not_a_final_benchmark",
        "users": len(manifest["users"]),
        "recordings": sum(user["recordings"] for user in manifest["users"]),
        "usable_recordings": usable_recordings,
        "gap_boundaries_segmented": skipped_gaps,
        "raw_frames": raw_frames,
        "duration_hours": float(np.sum(durations) / 3600.0),
        "native_hz_median": float(np.median(sample_rates)),
        "native_hz_p05": float(np.percentile(sample_rates, 5)),
        "native_hz_p95": float(np.percentile(sample_rates, 95)),
        "context_samples": context,
        "horizon_samples": horizon,
        "stride": stride,
        "baselines_step_20_degrees": baseline_report,
        "motion_regimes_by_recent_100ms_rotation": regime_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--context", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.context, args.horizon, args.stride)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
