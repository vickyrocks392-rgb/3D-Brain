"""Unit and integration tests for online plasticity rules and PlasticRNN."""

import numpy as np
import pytest

from brain.plasticity import (
    FrozenPlasticity,
    HebbianPlasticity,
    HomeostaticPlasticity,
    ModulatedPlasticity,
    OjaPlasticity,
    PlasticRNN,
)


def test_frozen_plasticity_zero_delta():
    rule = FrozenPlasticity()
    pre = np.ones(5, dtype=np.float32)
    post = np.ones(5, dtype=np.float32)
    weights = np.ones((5, 5), dtype=np.float32)

    dw, metrics = rule.compute_delta(pre, post, weights)
    assert np.all(dw == 0.0)
    assert metrics["dw_norm"] == 0.0


def test_hebbian_plasticity_correlation_and_decay():
    rule = HebbianPlasticity(lr=0.01, weight_decay=0.001)
    pre = np.array([1.0, -1.0], dtype=np.float32)
    post = np.array([1.0, 1.0], dtype=np.float32)
    weights = np.zeros((2, 2), dtype=np.float32)

    dw, metrics = rule.compute_delta(pre, post, weights)
    # outer product: [[1, 1], [-1, -1]]
    expected_dw = 0.01 * np.outer(pre, post)
    np.testing.assert_allclose(dw, expected_dw, atol=1e-5)
    assert metrics["dw_norm"] > 0


def test_oja_plasticity_stabilization_and_normalization():
    rule = OjaPlasticity(lr=0.05)
    hidden_dim = 4
    weights = np.ones((hidden_dim, hidden_dim), dtype=np.float32) * 0.5
    rng = np.random.default_rng(42)

    # Step repeatedly with stationary inputs
    for _ in range(200):
        pre = rng.normal(0.0, 1.0, hidden_dim).astype(np.float32)
        post = np.tanh(np.dot(pre, weights))
        dw, _ = rule.compute_delta(pre, post, weights)
        weights += dw

    # Weights must remain bounded (no explosion)
    norm = np.linalg.norm(weights)
    assert not np.isnan(norm)
    assert not np.isinf(norm)
    assert norm < 10.0


def test_modulated_plasticity_gating():
    rule = ModulatedPlasticity(lr=0.02)
    pre = np.ones(3, dtype=np.float32)
    post = np.ones(3, dtype=np.float32)
    weights = np.zeros((3, 3), dtype=np.float32)

    # Modulation = 0 -> zero update
    dw_zero, _ = rule.compute_delta(pre, post, weights, modulation=0.0)
    assert np.all(dw_zero == 0.0)

    # Modulation = 1.0 -> positive update
    dw_active, m_active = rule.compute_delta(pre, post, weights, modulation=1.0)
    assert np.all(dw_active > 0.0)
    assert m_active["modulation"] == 1.0


def test_homeostatic_plasticity_activity_regulation():
    target_act = 0.2
    rule = HomeostaticPlasticity(lr=0.01, target_activity=target_act, gamma=0.01)
    weights = np.ones((4, 4), dtype=np.float32) * 0.5

    # Simulate hyperactive neuron (mean act = 0.9 > target)
    pre = np.ones(4, dtype=np.float32) * 0.9
    post = np.ones(4, dtype=np.float32) * 0.9

    for _ in range(50):
        dw, metrics = rule.compute_delta(pre, post, weights)
        weights += dw

    # Activity above target should induce downward scaling
    assert metrics["mean_running_act"] > target_act
    assert np.all(np.isfinite(weights))


def test_plastic_rnn_step_updates_weights():
    model = PlasticRNN(input_dim=2, hidden_dim=8, output_dim=1, plasticity_rule=OjaPlasticity(lr=0.02), seed=42)
    w_initial = model.W_hh.copy()

    x = np.array([[1.0, 0.5]], dtype=np.float32)
    y = model.step(x)

    assert y.shape == (1, 1)
    # Recurrent weights must have changed
    assert not np.allclose(model.W_hh, w_initial)
    assert model.plastic_updates_count == 1
    assert model.observable_state()["mean_abs_dw"] > 0


def test_plastic_rnn_frozen_mode():
    model = PlasticRNN(input_dim=2, hidden_dim=8, output_dim=1, plasticity_rule=FrozenPlasticity(), seed=42)
    w_initial = model.W_hh.copy()

    x = np.array([[1.0, 0.5]], dtype=np.float32)
    model.step(x)

    # Frozen rule must keep weights identical
    np.testing.assert_allclose(model.W_hh, w_initial)
    assert model.observable_state()["mean_abs_dw"] == 0.0
