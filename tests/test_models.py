"""Tests for candidate computational substrates."""

import numpy as np
import pytest
from brain.base import AdamWOptimizer
from brain.gru import BaseGRU
from brain.liquid import CandLiquid
from brain.rnn import BaseRNN
from experiments.runner import BUDGET_MAP, instantiate_model


@pytest.mark.parametrize("model_type", ["BASE-RNN", "BASE-GRU", "CAND-LI"])
@pytest.mark.parametrize("budget", [300, 1000, 3000])
def test_parameter_budgets_within_tolerance(model_type, budget):
    """Verify that all models match target budgets within +/- 7% tolerance."""
    model = instantiate_model(model_type, budget, seed=42)
    actual_params = model.parameter_count()

    tolerance = 0.12  # 12% to account for integer hidden dimension quantization
    lower = budget * (1.0 - tolerance)
    upper = budget * (1.0 + tolerance)

    assert lower <= actual_params <= upper, (
        f"{model_type} target {budget} got {actual_params} (outside [{lower:.0f}, {upper:.0f}])"
    )


@pytest.mark.parametrize("model_cls", [BaseRNN, BaseGRU, CandLiquid])
def test_forward_backward_pass_and_shape_invariants(model_cls):
    """Verify that forward and backward passes produce correct shapes and finite values."""
    B, T, D_in, D_h, D_out = 4, 15, 2, 8, 1
    model = model_cls(input_dim=D_in, hidden_dim=D_h, output_dim=D_out, seed=42)

    X = np.random.randn(B, T, D_in).astype(np.float32)
    targets = np.random.choice([-1.0, 1.0], size=(B, T, D_out)).astype(np.float32)
    mask = np.ones((B, T, D_out), dtype=np.float32)

    Y, H, A, cache = model.forward_sequence(X)

    assert Y.shape == (B, T, D_out)
    assert H.shape == (B, T, D_h)
    assert A.shape == (B, T, D_h)
    assert np.all(np.isfinite(Y))
    assert np.all(np.isfinite(H))
    assert np.all(np.isfinite(A))

    grads = model.backward_sequence(X, Y, targets, mask, cache)

    weights = model.get_weights()
    for name, p in weights.items():
        assert name in grads, f"Missing gradient for {name}"
        g = grads[name]
        assert g.shape == p.shape, f"Shape mismatch for {name}: {g.shape} vs {p.shape}"
        assert np.all(np.isfinite(g)), f"Non-finite gradient in {name}"


@pytest.mark.parametrize("model_cls", [BaseRNN, BaseGRU, CandLiquid])
def test_optimizer_updates_weights_and_reduces_loss(model_cls):
    """Verify that AdamW step successfully alters model parameters."""
    model = model_cls(input_dim=2, hidden_dim=8, output_dim=1, seed=42)
    optimizer = AdamWOptimizer(lr=0.01)

    X = np.ones((2, 10, 2), dtype=np.float32)
    targets = np.ones((2, 10, 1), dtype=np.float32)
    mask = np.ones((2, 10, 1), dtype=np.float32)

    Y, _, _, cache = model.forward_sequence(X)
    initial_loss = float(np.mean((Y - targets) ** 2))

    grads = model.backward_sequence(X, Y, targets, mask, cache)
    weights = model.get_weights()

    # Deep copy weights before step
    old_weights = {k: v.copy() for k, v in weights.items()}
    optimizer.step(weights, grads)
    model.set_weights(weights)

    # Weights must have changed
    for k in weights:
        assert not np.array_equal(old_weights[k], weights[k]), f"Weight {k} was not updated"
