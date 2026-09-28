"""Unit tests for TargetedSpatialFeedbackEngine (Phase 10)."""

import numpy as np
import pytest

from brain.targeted_feedback import (
    FeedbackMode,
    DendriticEligibilityMode,
    TargetedSpatialFeedbackEngine,
)


def test_feedback_engine_modes():
    for mode in [FeedbackMode.SPATIAL_LOCAL, FeedbackMode.RANDOM_ALIGNMENT, FeedbackMode.GLOBAL_SCALAR, FeedbackMode.FROZEN]:
        engine = TargetedSpatialFeedbackEngine(
            num_neurons=20,
            output_dim=1,
            feedback_mode=mode,
            seed=42,
        )
        m = engine.compute_targeted_feedback(reward=1.0)
        assert len(m) == 20
        if mode == FeedbackMode.GLOBAL_SCALAR:
            assert np.allclose(m, m[0])
        elif mode == FeedbackMode.FROZEN:
            assert np.allclose(m, 0.0)


def test_dendritic_eligibility_accumulation_and_decay():
    engine = TargetedSpatialFeedbackEngine(
        num_neurons=10,
        tau_eligibility_ms=50.0,
        eligibility_mode=DendriticEligibilityMode.VOLTAGE_GATED,
        seed=42,
    )
    pre_spikes = np.array([True, False, True, False, False, False, False, False, False, False])
    v_dend = np.full(10, -50.0, dtype=np.float32)  # Depolarized above -70
    adj_mask = np.ones((10, 10), dtype=bool)

    engine.update_dendritic_eligibility(pre_spikes, v_dend, v_rest=-70.0, adjacency_mask=adj_mask)
    # Pre-synaptic neurons 0 and 2 fired onto depolarized dendrites
    assert np.all(engine.E[0, :] > 0.0)
    assert np.all(engine.E[2, :] > 0.0)
    assert np.all(engine.E[1, :] == 0.0)

    # Next step with no spikes: decay by lambda_e
    zero_spikes = np.zeros(10, dtype=bool)
    e_before = engine.E.copy()
    engine.update_dendritic_eligibility(zero_spikes, v_dend, v_rest=-70.0, adjacency_mask=adj_mask)
    expected_decay = float(np.exp(-1.0 / 50.0))
    assert np.allclose(engine.E, e_before * expected_decay, atol=1e-4)


def test_dale_principle_bounds_on_plasticity():
    is_inh = np.array([False, False, True, True])
    engine = TargetedSpatialFeedbackEngine(
        num_neurons=4,
        is_inhibitory=is_inh,
        w_max_e=2.0,
        w_max_i=3.0,
        seed=42,
    )
    weights = np.array([
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [-1.0, -1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0, -1.0],
    ], dtype=np.float32)
    adj_mask = np.ones((4, 4), dtype=bool)

    # Pre-populate eligibility
    engine.E.fill(1.0)
    m_vector = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    new_w, updates, metrics = engine.apply_targeted_plasticity(
        weights=weights,
        adjacency_mask=adj_mask,
        m_vector=m_vector,
        step=1,
    )
    # Excitatory rows (0, 1) must remain >= 0
    assert np.all(new_w[:2, :] >= 0.0)
    # Inhibitory rows (2, 3) must remain <= 0
    assert np.all(new_w[2:, :] <= 0.0)


def test_credit_specificity_index_csi_calculation():
    engine = TargetedSpatialFeedbackEngine(num_neurons=4, seed=42)
    weights = np.full((4, 4), 0.5, dtype=np.float32)
    adj_mask = np.ones((4, 4), dtype=bool)
    engine.E.fill(0.1)

    # Relevant mask: only (0, 0) and (0, 1) are relevant
    rel_mask = np.zeros((4, 4), dtype=bool)
    rel_mask[0, :2] = True

    m_vector = np.array([2.0, 2.0, 0.0, 0.0], dtype=np.float32)

    new_w, updates, metrics = engine.apply_targeted_plasticity(
        weights=weights,
        adjacency_mask=adj_mask,
        m_vector=m_vector,
        relevant_mask=rel_mask,
        step=1,
    )
    # Since m_vector is concentrated on neurons 0 and 1, precision and CSI must be positive
    assert metrics["csi"] > 0.0
    assert metrics["precision"] > 0.0
    assert metrics["leakage"] <= 1.0
