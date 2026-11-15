import pytest

from headpose.features import motion_features
from headpose.synthetic import make_trajectory

torch = pytest.importorskip("torch")

from headpose.models import (  # noqa: E402
    ResidualGatedGRU,
    ResidualGatedMLP,
    ResidualGatedTCN,
    gated_prediction,
    trainable_parameter_count,
    uncertainty_aware_loss,
)


def test_motion_features_are_invariant_to_quaternion_sign_flips() -> None:
    trajectory = make_trajectory("accelerating", samples=120)
    flipped = trajectory.copy()
    flipped[::2] *= -1.0
    assert motion_features(flipped) == pytest.approx(motion_features(trajectory))


def test_motion_features_shape() -> None:
    trajectory = make_trajectory("steady", samples=120)
    assert motion_features(trajectory).shape == (118, 8)


def test_residual_tcn_shape_budget_and_safe_initialization() -> None:
    model = ResidualGatedTCN()
    inputs = torch.randn(4, 48, 8)
    residual, log_variance, gate = model(inputs)
    assert residual.shape == (4, 20, 3)
    assert log_variance.shape == (4, 20, 3)
    assert gate.shape == (4, 20, 1)
    assert torch.count_nonzero(residual) == 0
    assert torch.all((gate > 0.0) & (gate < 0.5))
    assert trainable_parameter_count(model) < 100_000


def test_residual_tcn_rejects_wrong_feature_dimension() -> None:
    model = ResidualGatedTCN()
    with pytest.raises(ValueError, match="feature_dim"):
        model(torch.randn(2, 48, 7))


def test_uncertainty_loss_is_finite_and_differentiable() -> None:
    residual = torch.zeros(2, 20, 3, requires_grad=True)
    log_variance = torch.zeros_like(residual, requires_grad=True)
    gate = torch.full((2, 20, 1), 0.25, requires_grad=True)
    target = torch.randn_like(residual) * 0.01
    loss = uncertainty_aware_loss(residual, log_variance, gate, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert residual.grad is not None
    assert gated_prediction(residual.detach(), gate.detach()).shape == target.shape


def test_bounded_residual_cannot_exceed_physical_limit() -> None:
    model = ResidualGatedTCN(max_residual_radians=0.25)
    with torch.no_grad():
        model.residual_head.bias.fill_(100.0)
    residual, _, _ = model(torch.randn(2, 48, 8))
    assert torch.max(torch.abs(residual)) <= 0.25


def test_bounded_residual_rejects_nonpositive_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        ResidualGatedTCN(max_residual_radians=0.0)


@pytest.mark.parametrize(
    ("model", "inputs"),
    [
        (ResidualGatedMLP(context_steps=48), torch.randn(2, 48, 8)),
        (ResidualGatedGRU(), torch.randn(2, 48, 8)),
    ],
)
def test_control_models_share_forecast_interface(model, inputs) -> None:
    residual, log_variance, gate = model(inputs)
    assert residual.shape == (2, 20, 3)
    assert log_variance.shape == residual.shape
    assert gate.shape == (2, 20, 1)
    assert trainable_parameter_count(model) < 100_000
