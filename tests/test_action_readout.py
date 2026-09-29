"""Tests for ActionReadout, Target-Specific Dendritic Learning, and Retrograde Signaling."""

import numpy as np
import pytest

from brain.action_readout import (
    ActionReadout,
    InputRepresentationType,
    LearningMechanism,
    TeachingSignalType,
    RetrogradeSignalType,
    DendriticSTDPType,
)


def test_action_readout_initialization():
    readout = ActionReadout(input_dim=20, num_actions=2, seed=42)
    assert readout.weights.shape == (20, 2)
    assert readout.eligibility.shape == (20, 2)
    assert len(readout.synapse_records) == 40
    assert not readout.is_frozen
    assert not readout.is_lesioned


def test_action_readout_forward_and_inputs():
    readout = ActionReadout(input_dim=16, num_actions=2, seed=42)
    substrate_state = {
        "spikes": np.ones(16),
        "spike_counts": np.full(16, 5.0),
        "bursts": np.array([2.0, 1.0]),
        "temporal_population": np.full(16, 0.5),
        "phase_features": np.zeros(16),
        "resonance_features": np.full(16, 0.8),
        "context_vector": np.ones(16),
    }

    # Test each input format
    for inp_type in InputRepresentationType:
        readout.input_type = inp_type
        drives = readout.forward(substrate_state)
        assert drives.shape == (2,)
        assert np.all(np.isfinite(drives))


def test_target_specific_teaching_signals():
    readout = ActionReadout(input_dim=10, num_actions=2, seed=42)
    probs = np.array([0.4, 0.6], dtype=np.float32)

    # G0: Global scalar
    readout.teaching_signal_type = TeachingSignalType.G0_GLOBAL_SCALAR
    m_g0 = readout.compute_teaching_signals(target_action=0, reward=1.0, action_probs=probs)
    assert np.allclose(m_g0, [1.0, 1.0])

    # G1: Action-target scalar
    readout.teaching_signal_type = TeachingSignalType.G1_ACTION_TARGET_SCALAR
    m_g1 = readout.compute_teaching_signals(target_action=0, reward=1.0, action_probs=probs)
    assert m_g1[0] > 0.0 and m_g1[1] < 0.0

    # G3: Dendritic target signal
    readout.teaching_signal_type = TeachingSignalType.G3_DENDRITIC_TARGET_SIGNAL
    m_g3 = readout.compute_teaching_signals(target_action=0, reward=1.0, action_probs=probs)
    assert m_g3[0] > 0.0  # (1 - 0.4) > 0
    assert m_g3[1] < 0.0  # (0 - 0.6) < 0

    # G4: Shuffled target
    readout.teaching_signal_type = TeachingSignalType.G4_SHUFFLED_TARGET
    m_g4 = readout.compute_teaching_signals(target_action=0, reward=1.0, action_probs=probs)
    assert np.all(np.isfinite(m_g4))

    # G5: Sign-flipped
    readout.teaching_signal_type = TeachingSignalType.G5_SIGN_FLIPPED
    m_g5 = readout.compute_teaching_signals(target_action=0, reward=1.0, action_probs=probs)
    assert m_g5[0] < 0.0  # -(1 - 0.4) < 0


def test_retrograde_signaling():
    readout = ActionReadout(input_dim=10, num_actions=2, seed=42)
    act = np.array([1.5, 0.2], dtype=np.float32)

    # R0: None
    readout.retrograde_type = RetrogradeSignalType.R0_NONE
    assert np.allclose(readout.compute_retrograde_signals(act), [0.0, 0.0])

    # R1: Scalar
    readout.retrograde_type = RetrogradeSignalType.R1_SCALAR
    r1 = readout.compute_retrograde_signals(act)
    assert np.allclose(r1, [np.mean(act), np.mean(act)])

    # R2: Action-specific
    readout.retrograde_type = RetrogradeSignalType.R2_ACTION_SPECIFIC
    r2 = readout.compute_retrograde_signals(act)
    assert np.allclose(r2, act)


def test_dendritic_stdp_modes():
    readout = ActionReadout(input_dim=10, num_actions=2, seed=42)
    pre = np.ones(10)
    post = np.array([0.8, 0.2])
    teaching = np.array([1.0, -1.0])

    for stdp_mode in DendriticSTDPType:
        readout.stdp_type = stdp_mode
        # Step twice to populate history
        _ = readout.compute_stdp_update(pre, post, teaching)
        stdp = readout.compute_stdp_update(pre * 1.1, post * 0.9, teaching)
        assert stdp.shape == (10, 2)
        assert np.all(np.isfinite(stdp))


def test_learning_mechanisms_and_credit_localization():
    readout = ActionReadout(input_dim=10, num_actions=2, lr=0.05, seed=42)
    substrate_state = {"context_vector": np.ones(10)}

    # D1 Frozen
    readout.learning_mech = LearningMechanism.D1_FROZEN
    w_before = readout.weights.copy()
    _ = readout.step_plasticity(
        substrate_state=substrate_state,
        target_action=0,
        reward=1.0,
        action_probs=np.array([0.5, 0.5]),
        action_activations=np.array([0.5, 0.5]),
    )
    assert np.allclose(readout.weights, w_before)

    # D8 Temporal target dendritic learning
    readout.learning_mech = LearningMechanism.D8_TEMPORAL_TARGET_DENDRITIC
    readout.teaching_signal_type = TeachingSignalType.G3_DENDRITIC_TARGET_SIGNAL
    telemetry = readout.step_plasticity(
        substrate_state=substrate_state,
        target_action=0,
        reward=1.0,
        action_probs=np.array([0.3, 0.7]),
        action_activations=np.array([0.5, 0.5]),
    )
    assert "credit_specificity_index" in telemetry
    assert "credit_precision" in telemetry
    assert "credit_leakage" in telemetry
    assert "gini_concentration" in telemetry
    assert 0.0 <= telemetry["credit_specificity_index"] <= 1.0


def test_freeze_lesion_scramble_replace():
    readout = ActionReadout(input_dim=8, num_actions=2, seed=42)
    w_orig = readout.weights.copy()

    # Freeze
    readout.freeze(True)
    assert readout.is_frozen
    readout.freeze(False)
    assert not readout.is_frozen

    # Lesion
    readout.lesion(action_target=0)
    assert np.all(readout.lesion_mask[:, 0] == 0.0)
    assert np.all(readout.lesion_mask[:, 1] == 1.0)

    # Scramble
    readout.scramble()
    assert readout.weights.shape == w_orig.shape

    # Replace
    new_w = np.zeros_like(w_orig)
    readout.replace_weights(new_w)
    assert np.allclose(readout.weights, 0.0)
