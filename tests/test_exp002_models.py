"""Unit tests for Phase 2 ablation models: DISC-LRNN and BASE-GRU-MemBias."""

import numpy as np
import pytest
from brain.discrete_leaky_rnn import DiscreteLeakyRNN
from brain.gru import BaseGRUMemBias, sigmoid
from experiments.runner import instantiate_model


@pytest.mark.parametrize("budget", [300, 1000, 3000])
def test_disc_lrnn_budgets(budget):
    model = instantiate_model("DISC-LRNN", budget, seed=42)
    p = model.parameter_count()
    assert budget * 0.88 <= p <= budget * 1.12


@pytest.mark.parametrize("budget", [300, 1000, 3000])
def test_gru_membias_budgets(budget):
    model = instantiate_model("BASE-GRU-MemBias", budget, seed=42)
    p = model.parameter_count()
    assert budget * 0.88 <= p <= budget * 1.12


def test_gru_membias_initial_gate_bias():
    """Verify BASE-GRU-MemBias initializes b_z = -2.0, enforcing high memory retention."""
    model = BaseGRUMemBias(input_dim=2, hidden_dim=9, output_dim=1, seed=42)
    assert np.allclose(model.b_z, -2.0)

    # Initial update gate z_t should be ~= 0.119
    init_z = sigmoid(model.b_z)
    assert np.all(init_z < 0.15)
    # Retention factor (1 - z) should be >= 0.85
    assert np.all((1.0 - init_z) > 0.85)


def test_disc_lrnn_gradients_and_learning():
    """Verify DISC-LRNN backward pass computes valid gradients and reduces loss."""
    model = DiscreteLeakyRNN(input_dim=2, hidden_dim=8, output_dim=1, seed=42)

    X = np.random.randn(4, 15, 2).astype(np.float32)
    targets = np.random.choice([-1.0, 1.0], size=(4, 15, 1)).astype(np.float32)
    mask = np.ones((4, 15, 1), dtype=np.float32)

    Y, H, A, cache = model.forward_sequence(X)
    grads = model.backward_sequence(X, Y, targets, mask, cache)

    assert "W_xh" in grads
    assert "W_hh" in grads
    assert "rho" in grads
    assert np.all(np.isfinite(grads["rho"]))
    assert np.all(np.isfinite(grads["W_hh"]))


def test_observable_state_exposure():
    """Verify observable_state exposes actual runtime variables for all models."""
    lrnn = DiscreteLeakyRNN(input_dim=2, hidden_dim=6, output_dim=1, seed=42)
    lrnn.step(np.array([[0.5, -0.5]]))
    obs_lrnn = lrnn.observable_state()
    assert "alpha" in obs_lrnn
    assert "candidate_state" in obs_lrnn
    assert "previous_state" in obs_lrnn

    gru_mem = BaseGRUMemBias(input_dim=2, hidden_dim=6, output_dim=1, seed=42)
    gru_mem.step(np.array([[0.5, -0.5]]))
    obs_gru = gru_mem.observable_state()
    assert "gate_z" in obs_gru
    assert "gate_r" in obs_gru
    assert "candidate_state" in obs_gru
