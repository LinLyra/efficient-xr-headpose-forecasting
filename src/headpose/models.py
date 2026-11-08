"""Compact operator-friendly residual forecasting models."""

from __future__ import annotations

import torch
from torch import nn


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.receptive_padding = 2 * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=self.receptive_padding,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.normalization = nn.BatchNorm1d(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(values)
        values = values[..., : -self.receptive_padding]
        values = self.pointwise(values)
        values = self.normalization(values)
        return residual + self.activation(values)


class ResidualGatedTCN(nn.Module):
    """Predict SO(3) residual means, diagonal uncertainty and blend gates."""

    def __init__(
        self,
        feature_dim: int = 8,
        hidden_dim: int = 48,
        horizon: int = 20,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        max_residual_radians: float | None = None,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.horizon = horizon
        if max_residual_radians is not None and max_residual_radians <= 0.0:
            raise ValueError("max_residual_radians must be positive")
        self.max_residual_radians = max_residual_radians
        self.input_projection = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[CausalDepthwiseBlock(hidden_dim, dilation) for dilation in dilations]
        )
        self.residual_head = nn.Linear(hidden_dim, horizon * 3)
        self.uncertainty_head = nn.Linear(hidden_dim, horizon * 3)
        self.gate_head = nn.Linear(hidden_dim, horizon)
        self._initialize_safe_residual()

    def _initialize_safe_residual(self) -> None:
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must have shape [batch, time, feature_dim]")
        values = features.transpose(1, 2)
        values = self.input_projection(values)
        values = self.blocks(values)
        state = values[..., -1]
        residual = self.residual_head(state).reshape(-1, self.horizon, 3)
        if self.max_residual_radians is not None:
            residual = self.max_residual_radians * torch.tanh(residual)
        log_variance = self.uncertainty_head(state).reshape(-1, self.horizon, 3)
        gate = torch.sigmoid(self.gate_head(state)).reshape(-1, self.horizon, 1)
        return residual, log_variance, gate


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def gated_prediction(
    residual: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Blend a learned tangent-space correction with the zero-motion fallback."""
    if residual.shape[:-1] != gate.shape[:-1] or gate.shape[-1] != 1:
        raise ValueError("gate must match residual batch/horizon dimensions")
    return residual * gate


def uncertainty_aware_loss(
    residual: torch.Tensor,
    log_variance: torch.Tensor,
    gate: torch.Tensor,
    target: torch.Tensor,
    gate_penalty: float = 1e-3,
) -> torch.Tensor:
    """Diagonal tangent Gaussian NLL with a weak protected-fallback prior."""
    if residual.shape != log_variance.shape or residual.shape != target.shape:
        raise ValueError("residual, log_variance and target shapes must match")
    prediction = gated_prediction(residual, gate)
    bounded_log_variance = torch.clamp(log_variance, min=-8.0, max=6.0)
    squared_error = (prediction - target).square()
    nll = 0.5 * (
        torch.exp(-bounded_log_variance) * squared_error + bounded_log_variance
    )
    return nll.mean() + gate_penalty * gate.mean()
