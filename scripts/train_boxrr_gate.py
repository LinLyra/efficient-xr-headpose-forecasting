"""Train and evaluate the protected SO(3) residual gate on processed BOXRR data."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from headpose.dataset import WindowDataset, materialize_boxrr_splits
from headpose.models import (
    ResidualGatedTCN,
    gated_prediction,
    trainable_parameter_count,
    uncertainty_aware_loss,
)
from headpose.quaternion import exp_map, geodesic_degrees
from headpose.uncertainty import (
    fit_variance_scale,
    gaussian_nll,
    interval_calibration,
)


def user_bootstrap(values: np.ndarray, seed: int = 2027) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "users": int(len(values)),
    }


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = np.argsort(np.argsort(left))
    right_rank = np.argsort(np.argsort(right))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def risk_coverage(
    uncertainty: np.ndarray, errors: np.ndarray
) -> dict[str, float]:
    order = np.argsort(uncertainty)
    return {
        f"{int(coverage * 100)}%": float(
            errors[order[: max(1, int(len(order) * coverage))]].mean()
        )
        for coverage in (1.0, 0.8, 0.5, 0.2)
    }


def evaluate(
    model: nn.Module,
    dataset: WindowDataset,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    batch_size: int,
    variance_scale: float = 1.0,
) -> dict[str, object]:
    model.eval()
    predictions: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    normalized = (dataset.features - feature_mean) / feature_std
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            features = torch.from_numpy(normalized[start : start + batch_size]).to(device)
            residual, log_variance, gate = model(features)
            predictions.append(gated_prediction(residual, gate).cpu().numpy())
            variances.append(log_variance.exp().cpu().numpy())
            gates.append(gate.squeeze(-1).cpu().numpy())
    prediction = np.concatenate(predictions)
    variance = np.concatenate(variances)
    uncertainty = variance.mean(axis=-1)
    gate = np.concatenate(gates)
    final_errors = geodesic_degrees(
        exp_map(prediction[:, -1]), exp_map(dataset.targets[:, -1])
    )
    pose_errors = geodesic_degrees(
        exp_map(np.zeros_like(dataset.targets[:, -1])),
        exp_map(dataset.targets[:, -1]),
    )
    user_ids = np.unique(dataset.user_indices)
    per_user_model = np.asarray([
        final_errors[dataset.user_indices == index].mean() for index in user_ids
    ])
    per_user_pose = np.asarray([
        pose_errors[dataset.user_indices == index].mean() for index in user_ids
    ])
    residual_error = prediction[:, -1] - dataset.targets[:, -1]
    squared_error = np.square(residual_error).mean(axis=-1)
    final_variance = variance[:, -1]
    return {
        "windows": int(len(final_errors)),
        "users": int(len(user_ids)),
        "model": {
            "mean": float(final_errors.mean()),
            "median": float(np.median(final_errors)),
            "p95": float(np.quantile(final_errors, 0.95)),
            "p99": float(np.quantile(final_errors, 0.99)),
            "user_macro_95": user_bootstrap(per_user_model),
        },
        "constant_pose": {
            "mean": float(pose_errors.mean()),
            "p95": float(np.quantile(pose_errors, 0.95)),
            "user_macro_95": user_bootstrap(per_user_pose),
        },
        "paired_delta_model_minus_pose_95": user_bootstrap(
            per_user_model - per_user_pose
        ),
        "gate_mean_final": float(gate[:, -1].mean()),
        "uncertainty_error_correlation_final": float(
            np.corrcoef(uncertainty[:, -1], squared_error)[0, 1]
        ),
        "uncertainty_error_rank_correlation_final": rank_correlation(
            uncertainty[:, -1], final_errors
        ),
        "risk_coverage_mean_degrees": risk_coverage(
            uncertainty[:, -1], final_errors
        ),
        "uncertainty_calibration": {
            "variance_scale": variance_scale,
            "gaussian_nll": gaussian_nll(
                final_variance, residual_error, variance_scale=variance_scale
            ),
            **interval_calibration(
                final_variance, residual_error, variance_scale=variance_scale
            ),
        },
    }


def fit_validation_variance_scale(
    model: nn.Module,
    dataset: WindowDataset,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    normalized = (dataset.features - feature_mean) / feature_std
    predictions: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            features = torch.from_numpy(normalized[start : start + batch_size]).to(device)
            residual, log_variance, gate = model(features)
            predictions.append(gated_prediction(residual, gate)[:, -1].cpu().numpy())
            variances.append(log_variance.exp()[:, -1].cpu().numpy())
    prediction = np.concatenate(predictions)
    predicted_variance = np.concatenate(variances)
    residual_error = prediction - dataset.targets[:, -1]
    return fit_variance_scale(predicted_variance, residual_error)


class OnnxForecaster(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual, log_variance, gate = self.model(features)
        return gated_prediction(residual, gate), log_variance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--train-windows", type=int, default=120_000)
    parser.add_argument("--validation-windows", type=int, default=20_000)
    parser.add_argument("--test-windows", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--max-correction-degrees", type=float)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2027)
    np.random.seed(2027)
    started = time.perf_counter()
    datasets, assignments = materialize_boxrr_splits(
        args.root,
        context=args.context,
        horizon=args.horizon,
        stride=args.stride,
        max_windows={
            "train": args.train_windows,
            "validation": args.validation_windows,
            "test": args.test_windows,
        },
    )
    (args.output_dir / "split-manifest.json").write_text(
        json.dumps({"seed": "xr-headpose-v1", "assignments": assignments}, indent=2)
        + "\n"
    )
    feature_mean = datasets["train"].features.mean(axis=(0, 1), keepdims=True)
    feature_std = datasets["train"].features.std(axis=(0, 1), keepdims=True) + 1e-6
    train_features = (datasets["train"].features - feature_mean) / feature_std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features), torch.from_numpy(datasets["train"].targets)
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    max_residual_radians = (
        None
        if args.max_correction_degrees is None
        else float(np.radians(args.max_correction_degrees))
    )
    model = ResidualGatedTCN(
        horizon=args.horizon, max_residual_radians=max_residual_radians
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    history: list[dict[str, float]] = []
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for features, targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            residual, log_variance, gate = model(features)
            loss = uncertainty_aware_loss(residual, log_variance, gate, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(
            model,
            datasets["validation"],
            feature_mean,
            feature_std,
            device,
            args.batch_size,
        )
        validation_mean = float(validation["model"]["mean"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_mean_degrees": validation_mean,
        })
        print(json.dumps(history[-1]), flush=True)
        if validation_mean < best_validation - 1e-4:
            best_validation = validation_mean
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break
    model.load_state_dict(best_state)
    variance_scale = fit_validation_variance_scale(
        model,
        datasets["validation"],
        feature_mean,
        feature_std,
        device,
        args.batch_size,
    )
    test_report = evaluate(
        model,
        datasets["test"],
        feature_mean,
        feature_std,
        device,
        args.batch_size,
        variance_scale,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "context": args.context,
            "horizon": args.horizon,
            "max_residual_radians": max_residual_radians,
            "variance_scale": variance_scale,
        },
        args.output_dir / "residual-gated-tcn.pt",
    )
    model_cpu = model.cpu().eval()
    torch.onnx.export(
        OnnxForecaster(model_cpu),
        torch.zeros(1, args.context - 2, 8),
        args.output_dir / "residual-gated-tcn.onnx",
        input_names=["motion_features"],
        output_names=["tangent_prediction", "log_variance"],
        dynamic_axes={"motion_features": {0: "batch"}},
        opset_version=17,
    )
    report = {
        "status": "preliminary_benchmark_not_final_population_estimate",
        "device": str(device),
        "parameters": trainable_parameter_count(model),
        "max_correction_degrees": args.max_correction_degrees,
        "variance_scale_fitted_on_validation": variance_scale,
        "elapsed_seconds": time.perf_counter() - started,
        "cohort_counts": {
            split: sum(value == split for value in assignments.values())
            for split in ("train", "validation", "test")
        },
        "window_counts": {split: len(data.features) for split, data in datasets.items()},
        "history": history,
        "test": test_report,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
