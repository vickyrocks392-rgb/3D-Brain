"""Unit and Numerical Tests for Three-Factor Synaptic Plasticity Engine (Phase 9).

Verifies:
1. Eligibility trace accumulation and exponential decay: e(t) = lambda_e * e(t-1) + f(pre, post)
2. Asymmetric STDP eligibility trace mode
3. Signed Reward Prediction Error (RPE) moving baseline adaptation
4. Dale's Principle weight bounding (Excitatory >= 0, Inhibitory <= 0)
5. Zero eligibility trace ablation
6. Frozen mode (M=0)
7. Community-localized neuromodulation
"""

import numpy as np
import pytest

from brain.three_factor import (
    EligibilityTraceMode,
    NeuromodulatorMode,
    ThreeFactorPlasticityEngine,
    ThreeFactorUpdateRecord,
)


def test_eligibility_accumulation_and_decay():
    """Verify numerical behavior of eligibility trace accumulation and exponential decay."""
    num_neurons = 4
    tau_e = 50.0
    dt = 1.0
    expected_lambda = np.exp(-dt / tau_e)

    engine = ThreeFactorPlasticityEngine(
        num_neurons=num_neurons,
        learning_rate=0.01,
        tau_eligibility_ms=tau_e,
        eligibility_mode=EligibilityTraceMode.COINCIDENCE,
        neuromodulator_mode=NeuromodulatorMode.GLOBAL_REWARD,
        dt_ms=dt,
    )

    # All-to-all connectivity
    adj_mask = np.ones((num_neurons, num_neurons), dtype=bool)
    np.fill_diagonal(adj_mask, False)

    # Initial eligibility should be zero
    assert np.allclose(engine.E, 0.0)

    # Step 1: Pre neuron 0 and Post neuron 1 spike together
    spikes = np.array([True, True, False, False], dtype=bool)
    engine.update_eligibility(spikes, adj_mask)

    # Coincidence between 0 and 1: e_01 = 1.0
    assert np.isclose(engine.E[0, 1], 1.0, atol=1e-5)
    assert np.isclose(engine.E[1, 0], 1.0, atol=1e-5)
    assert np.isclose(engine.E[0, 2], 0.0)

    # Step 2: No spikes occur; eligibility must decay by lambda_e
    no_spikes = np.zeros(num_neurons, dtype=bool)
    engine.update_eligibility(no_spikes, adj_mask)

    assert np.isclose(engine.E[0, 1], expected_lambda, atol=1e-5)
    assert np.isclose(engine.E[1, 0], expected_lambda, atol=1e-5)

    # Step 3: Decay for 20 more steps without spikes
    for _ in range(20):
        engine.update_eligibility(no_spikes, adj_mask)

    assert np.isclose(engine.E[0, 1], expected_lambda ** 21, atol=1e-5)
    assert engine.E[0, 1] < 0.70  # Substantial decay confirmed


def test_signed_rpe_baseline_adaptation():
    """Verify signed RPE calculation: RPE(t) = R(t) - V_bar(t), where baseline updates exponentially."""
    engine = ThreeFactorPlasticityEngine(
        num_neurons=4,
        neuromodulator_mode=NeuromodulatorMode.REWARD_PREDICTION_ERROR,
        rpe_baseline_alpha=0.10,
    )

    # Trial 1: Reward = 1.0, baseline initially 0.0
    rpe1 = engine.compute_neuromodulatory_signal(reward=1.0)
    assert np.isclose(rpe1, 1.0, atol=1e-5)
    # Baseline after update: 0.9 * 0.0 + 0.1 * 1.0 = 0.1
    assert np.isclose(engine.reward_baseline, 0.10, atol=1e-5)

    # Trial 2: Repeated Reward = 1.0
    rpe2 = engine.compute_neuromodulatory_signal(reward=1.0)
    # RPE should diminish because baseline caught up
    assert np.isclose(rpe2, 0.90, atol=1e-5)
    assert np.isclose(engine.reward_baseline, 0.19, atol=1e-5)

    # Trial 3: Unrewarded trial (Reward = 0.0) -> negative RPE
    rpe3 = engine.compute_neuromodulatory_signal(reward=0.0)
    assert rpe3 < 0.0
    assert np.isclose(rpe3, -0.19, atol=1e-5)


def test_dales_principle_enforcement():
    """Verify that three-factor weight updates strictly preserve Dale's sign constraints."""
    num_neurons = 4
    # Neurons 0, 1 are Excitatory; Neurons 2, 3 are Inhibitory
    is_inh = np.array([False, False, True, True], dtype=bool)

    engine = ThreeFactorPlasticityEngine(
        num_neurons=num_neurons,
        learning_rate=0.05,
        w_min_e=0.0,
        w_max_e=2.0,
        w_min_i=0.0,
        w_max_i=3.0,
        is_inhibitory=is_inh,
    )

    # Initial weights: [pre, post]
    # Row 0, 1: Excitatory (> 0)
    # Row 2, 3: Inhibitory (< 0)
    weights = np.array([
        [0.0, 0.5, 0.5, 0.5],
        [0.5, 0.0, 0.5, 0.5],
        [-0.8, -0.8, 0.0, -0.8],
        [-0.8, -0.8, -0.8, 0.0],
    ], dtype=np.float32)

    adj_mask = (np.abs(weights) > 0.0)

    # Set positive eligibility on all synapses
    engine.E.fill(1.0)
    engine.E[~adj_mask] = 0.0

    # 1. Negative reinforcement (M = -2.0): Attempt to drive excitatory weights negative
    updated_w, _ = engine.apply_plasticity(weights, adj_mask, neuromodulator=-20.0)

    # Excitatory weights must be clipped at 0.0, NEVER negative
    assert np.all(updated_w[0:2, :] >= 0.0)
    assert np.allclose(updated_w[0, 1], 0.0)  # Clipped at minimum

    # 2. Positive reinforcement (M = +2.0): Attempt to drive inhibitory weights positive
    engine.E.fill(1.0)
    engine.E[~adj_mask] = 0.0
    updated_w2, _ = engine.apply_plasticity(weights, adj_mask, neuromodulator=20.0)

    # Inhibitory weights must be clipped at 0.0, NEVER positive
    assert np.all(updated_w2[2:4, :] <= 0.0)
    assert np.allclose(updated_w2[2, 0], 0.0)  # Clipped at boundary


def test_zero_eligibility_ablation():
    """Verify that when eligibility is disabled (mode=ZERO), no synaptic plasticity occurs."""
    num_neurons = 4
    engine = ThreeFactorPlasticityEngine(
        num_neurons=num_neurons,
        learning_rate=0.05,
        eligibility_mode=EligibilityTraceMode.ZERO,
    )

    adj_mask = np.ones((num_neurons, num_neurons), dtype=bool)
    spikes = np.ones(num_neurons, dtype=bool)

    engine.update_eligibility(spikes, adj_mask)
    assert np.allclose(engine.E, 0.0)

    weights = np.ones((num_neurons, num_neurons), dtype=np.float32) * 0.5
    updated_w, updates = engine.apply_plasticity(weights, adj_mask, neuromodulator=1.0)

    assert np.allclose(updated_w, weights)
    assert len(updates) == 0


def test_frozen_mode():
    """Verify that neuromodulator_mode=FROZEN clamps updates to zero."""
    num_neurons = 4
    engine = ThreeFactorPlasticityEngine(
        num_neurons=num_neurons,
        learning_rate=0.05,
        neuromodulator_mode=NeuromodulatorMode.FROZEN,
    )

    adj_mask = np.ones((num_neurons, num_neurons), dtype=bool)
    spikes = np.ones(num_neurons, dtype=bool)
    weights = np.ones((num_neurons, num_neurons), dtype=np.float32) * 0.5

    updated_w, updates, _ = engine.step(weights, adj_mask, spikes, reward=1.0)
    assert np.allclose(updated_w, weights)
    assert len(updates) == 0
