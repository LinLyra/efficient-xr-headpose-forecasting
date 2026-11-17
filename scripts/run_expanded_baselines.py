"""Freeze physical baselines on validation, then evaluate test exactly once."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from headpose.dataset import materialize_boxrr_splits
from headpose.evaluation import (
    paired_clustered_bootstrap_difference,
    paired_cluster_permutation_test,
    summarize_errors,
)
from headpose.events import (
    EventThresholds,
    classify_motion_event,
    fit_event_thresholds,
)
from headpose.physical_benchmark import (
    dataset_fingerprint,
    endpoint_errors,
    tune_physical_baselines,
    user_macro_mean,
)

PROTOCOL_VERSION = "expanded-1.0"
SPLIT_SEED = "xr-headpose-expanded-v1"
RESERVOIR_SEED = 2027


def split_counts(arguments: argparse.Namespace) -> dict[str, int]:
    return {
        "train": arguments.train_users,
        "validation": arguments.validation_users,
        "test": arguments.test_users,
    }


def common_materialization(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        "context": arguments.context,
        "horizon": arguments.horizon,
        "stride": arguments.stride,
        "split_seed": SPLIT_SEED,
        "split_counts": split_counts(arguments),
        "reservoir_seed": RESERVOIR_SEED,
    }


def tune(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir(parents=True, exist_ok=True)
    limits = {
        "train": arguments.train_windows,
        "validation": arguments.validation_windows,
    }
    datasets, assignments = materialize_boxrr_splits(
        arguments.root,
        max_windows=limits,
        **common_materialization(arguments),
    )
    thresholds = fit_event_thresholds(datasets["train"].contexts)
    selection = tune_physical_baselines(
        datasets["validation"], horizon=arguments.horizon
    )
    frozen = {
        "protocol_version": PROTOCOL_VERSION,
        "split_seed": SPLIT_SEED,
        "reservoir_seed": RESERVOIR_SEED,
        "cohort_counts": split_counts(arguments),
        "window_limits": {
            **limits,
            "test": arguments.test_windows,
        },
        "context": arguments.context,
        "horizon": arguments.horizon,
        "stride": arguments.stride,
        "event_thresholds": asdict(thresholds),
        "physical_selection": selection,
    }
    (arguments.output / "frozen-selection.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )
    (arguments.output / "split-manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "split_seed": SPLIT_SEED,
                "assignments": assignments,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(frozen, indent=2))


def summary(errors: np.ndarray, users: np.ndarray) -> dict[str, object]:
    values = summarize_errors(errors)
    return {
        **asdict(values),
        "user_macro_mean": user_macro_mean(errors, users),
    }


def event_breakdown(
    errors: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
) -> dict[str, object]:
    result = {}
    for label in np.unique(labels):
        mask = labels == label
        result[str(label)] = {
            **summary(errors[mask], users[mask]),
            "users": int(len(np.unique(users[mask]))),
            "confirmatory_support": bool(len(np.unique(users[mask])) >= 20),
        }
    dynamic = np.isin(
        labels,
        ("reversal", "braking", "acceleration_onset", "sustained_fast_turn"),
    )
    if dynamic.any():
        result["high_dynamic_union"] = {
            **summary(errors[dynamic], users[dynamic]),
            "users": int(len(np.unique(users[dynamic]))),
            "confirmatory_support": bool(len(np.unique(users[dynamic])) >= 20),
        }
    return result


def test(arguments: argparse.Namespace) -> None:
    frozen = json.loads(arguments.selection.read_text())
    if frozen.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("selection protocol version does not match this evaluator")
    expected_shape = {
        "cohort_counts": split_counts(arguments),
        "context": arguments.context,
        "horizon": arguments.horizon,
        "stride": arguments.stride,
    }
    for key, expected in expected_shape.items():
        if frozen.get(key) != expected:
            raise ValueError(f"frozen {key} differs from requested evaluation")
    requested_limits = {
        "train": arguments.train_windows,
        "validation": arguments.validation_windows,
        "test": arguments.test_windows,
    }
    if frozen.get("window_limits") != requested_limits:
        raise ValueError("frozen window limits differ from requested evaluation")
    arguments.output.mkdir(parents=True, exist_ok=True)
    destination = arguments.output / "physical-baseline-test-report.json"
    if destination.exists() and not arguments.allow_test_overwrite:
        raise FileExistsError(
            "test report exists; refusing a repeated test look without explicit override"
        )
    datasets, _ = materialize_boxrr_splits(
        arguments.root,
        max_windows={
            "validation": arguments.validation_windows,
            "test": arguments.test_windows,
        },
        **common_materialization(arguments),
    )
    expected_fingerprint = frozen["physical_selection"]["validation_fingerprint"]
    if dataset_fingerprint(datasets["validation"]) != expected_fingerprint:
        raise RuntimeError("validation windows differ from the frozen tuning run")

    test_data = datasets["test"]
    thresholds = EventThresholds(**frozen["event_thresholds"])
    labels = np.asarray(
        [classify_motion_event(context, thresholds) for context in test_data.contexts]
    )
    method_reports = {}
    method_errors = {}
    for name, method in frozen["physical_selection"]["methods"].items():
        parameters = method["selected"]["parameters"]
        errors = endpoint_errors(
            name, parameters, test_data, horizon=arguments.horizon
        )
        method_errors[name] = errors
        method_reports[name] = {
            "parameters": parameters,
            "overall": summary(errors, test_data.user_indices),
            "events": event_breakdown(errors, labels, test_data.user_indices),
        }

    best_name = frozen["physical_selection"]["best_method"]
    reference_name = "constant_pose"
    comparison = None
    if best_name != reference_name:
        interval = paired_clustered_bootstrap_difference(
            method_errors[best_name],
            method_errors[reference_name],
            test_data.user_indices,
            resamples=10_000,
        )
        permutation = paired_cluster_permutation_test(
            method_errors[best_name],
            method_errors[reference_name],
            test_data.user_indices,
            alternative="less",
            resamples=10_000,
        )
        comparison = {
            "candidate": best_name,
            "reference": reference_name,
            "bootstrap": asdict(interval),
            "permutation": asdict(permutation),
        }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_test_evaluation",
        "test_fingerprint": dataset_fingerprint(test_data),
        "windows": int(len(test_data.contexts)),
        "users": int(len(np.unique(test_data.user_indices))),
        "best_physical_selected_on_validation": best_name,
        "event_thresholds_fitted_on_training": asdict(thresholds),
        "methods": method_reports,
        "best_vs_constant_pose": comparison,
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--train-users", type=int, default=160)
    parser.add_argument("--validation-users", type=int, default=40)
    parser.add_argument("--test-users", type=int, default=50)
    parser.add_argument("--train-windows", type=int, default=120_000)
    parser.add_argument("--validation-windows", type=int, default=20_000)
    parser.add_argument("--test-windows", type=int, default=20_000)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    tune_parser = subparsers.add_parser("tune")
    add_common(tune_parser)
    test_parser = subparsers.add_parser("test")
    add_common(test_parser)
    test_parser.add_argument("--selection", type=Path, required=True)
    test_parser.add_argument("--allow-test-overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "tune":
        tune(arguments)
    else:
        test(arguments)


if __name__ == "__main__":
    main()
