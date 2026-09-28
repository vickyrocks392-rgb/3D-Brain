"""Unit tests for PredictiveDendriticGating (Phase 11)."""

import numpy as np
import pytest

from brain.predictive_gating import GateMode, PredictiveDendriticGating


def test_predictive_gating_initialization():
    engine = PredictiveDendriticGating(
        num_neurons=20,
        hidden_dim=20,
        gate_mode=GateMode.GATE_ON,
        disinhibition_strength=0.85,
        seed=42,
    )
    assert engine.num_neurons == 20
    assert engine.gate_mode == GateMode.GATE_ON
    assert engine.disinhibition_strength == 0.85
    assert engine.last_gate_vector.shape == (20,)
    assert engine.last_disinhibition_factor.shape == (20,)


def test_gate_modes():
    # 1. GATE_ON: Produces non-zero sigmoid gating vector
    engine_on = PredictiveDendriticGating(num_neurons=10, hidden_dim=10, gate_mode=GateMode.GATE_ON, seed=42)
    s = np.ones(10, dtype=np.float32) * 2.0
    g_on = engine_on.compute_gate(s)
    assert np.all(g_on >= 0.0) and np.all(g_on <= 1.0)
    assert np.any(g_on > 0.1)

    # 2. GATE_OFF: Gating vector clamped to zero
    engine_off = PredictiveDendriticGating(num_neurons=10, hidden_dim=10, gate_mode=GateMode.GATE_OFF, seed=42)
    g_off = engine_off.compute_gate(s)
    assert np.all(g_off == 0.0)

    # 3. GATE_REVERSED: Gating vector inverted (1 - g_raw)
    engine_rev = PredictiveDendriticGating(num_neurons=10, hidden_dim=10, gate_mode=GateMode.GATE_REVERSED, seed=42)
    g_rev = engine_rev.compute_gate(s)
    assert np.allclose(g_rev, 1.0 - g_on, atol=1e-5)

    # 4. GATE_SHUFFLED: Same values, permuted order
    engine_shuf = PredictiveDendriticGating(num_neurons=10, hidden_dim=10, gate_mode=GateMode.GATE_SHUFFLED, seed=42)
    g_shuf = engine_shuf.compute_gate(s)
    assert np.isclose(np.mean(g_shuf), np.mean(g_on), atol=1e-4)


def test_dendritic_disinhibition_application():
    engine = PredictiveDendriticGating(
        num_neurons=5,
        hidden_dim=5,
        gate_mode=GateMode.GATE_ON,
        disinhibition_strength=0.80,
        seed=42,
    )
    # Raw inhibitory current is negative
    i_syn_i = np.full(5, -10.0, dtype=np.float32)
    # Manual gate: neuron 0 fully gated (G=1.0), neuron 1 ungated (G=0.0)
    gate = np.array([1.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)

    gated_i = engine.apply_disinhibition(i_syn_i, gate_vector=gate)

    # For G=1.0: factor = 1.0 - 0.8 * 1.0 = 0.2 -> current = -10 * 0.2 = -2.0 (80% disinhibition!)
    assert np.isclose(gated_i[0], -2.0, atol=1e-4)
    # For G=0.0: factor = 1.0 -> current = -10.0 (full inhibition)
    assert np.isclose(gated_i[1], -10.0, atol=1e-4)
    # For G=0.5: factor = 0.6 -> current = -6.0
    assert np.isclose(gated_i[2], -6.0, atol=1e-4)


def test_gate_selectivity_metric():
    engine = PredictiveDendriticGating(num_neurons=6, hidden_dim=6, gate_mode=GateMode.GATE_ON, seed=42)
    relevant_mask = np.array([True, True, True, False, False, False], dtype=bool)

    s = np.zeros(6, dtype=np.float32)
    engine.compute_gate(s, relevant_mask=relevant_mask)
    # Selectivity must be a valid float in [-1.0, 1.0]
    assert -1.0 <= engine.last_gate_selectivity <= 1.0
