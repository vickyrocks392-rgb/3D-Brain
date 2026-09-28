"""Unit tests for Phase 12 Learnable Context-to-Action Interface.

Validates:
1. Interface initialization across sources and architectures.
2. Competitive action population dynamics and winner selection.
3. Local readout learning rules (R1 supervised, R2 Hebbian, R3 three-factor, R4 pred-error, R5 delayed reward).
4. Isolated interface eligibility tracking and zero recurrent leakage.
5. Action competition modes (no inhibition, weak, baseline, strong, no recurrence).
6. Causal interventions (scramble, swap, lesion, irrelevant, label permutation, freeze).
7. Phase 11 lesion paradox replication and disinhibition mechanism.
8. Representation source extraction (somatic, dendritic, NMDA, attractor, concatenated, PCA, random).
9. Credit localization metrics (CSI, precision, leakage, Gini concentration).
10. Phase12Network integration and observable state contract.
"""

import numpy as np
import pytest

from brain.context_action_interface import (
    ContextActionInterface,
    RepresentationSource,
    ReadoutArchitecture,
    ActionCompetitionMode,
)
from brain.phase12_network import Phase12Network
from benchmarks.pure_context_decode import PureContextDecodeTask


def test_interface_initialization():
    """Test 1: ContextActionInterface initializes with proper dimensions and shapes."""
    iface = ContextActionInterface(
        hidden_dim=40,
        num_actions=4,
        source=RepresentationSource.ATTRACTOR,
        architecture=ReadoutArchitecture.R3_THREE_FACTOR,
        seed=42,
    )
    assert iface.context_dim == 40
    assert iface.num_actions == 4
    assert iface.W_ctx_to_act.shape == (40, 4)
    assert iface.eligibility.shape == (40, 4)
    assert iface.v_action.shape == (4,)
    assert len(iface.last_action_probs) == 4


def test_action_population_winner_selection():
    """Test 2: Competitive action population produces valid probabilities, margin, and winner."""
    iface = ContextActionInterface(hidden_dim=20, num_actions=4, seed=42)
    # Set explicit positive drive to Action 2
    ctx = np.ones(20, dtype=np.float32)
    iface.W_ctx_to_act[:, 2] = 2.0

    v_act, sel_act, margin = iface.step(ctx, dt_ms=1.0)
    assert sel_act == 2
    assert iface.last_selected_action == 2
    assert margin > 0.0
    assert np.isclose(np.sum(iface.last_action_probs), 1.0, atol=1e-5)
    assert iface.last_action_probs[2] > iface.last_action_probs[0]


def test_local_readout_learning_r1_supervised():
    """Test 3: Linear supervised readout (R1) updates weights in direction of target."""
    iface = ContextActionInterface(
        hidden_dim=10,
        num_actions=2,
        architecture=ReadoutArchitecture.R1_LINEAR_SUPERVISED,
        learning_rate=0.1,
        seed=42,
    )
    ctx = np.ones(10, dtype=np.float32)
    w_before = iface.W_ctx_to_act.copy()

    # Step with target_action = 1 on choice step
    iface.step(ctx, target_action=1, is_choice_step=True)
    w_after = iface.W_ctx_to_act.copy()

    # Weights toward target 1 should increase, weights toward 0 should decrease
    assert np.mean(w_after[:, 1]) > np.mean(w_before[:, 1])
    assert np.mean(w_after[:, 0]) < np.mean(w_before[:, 0])


def test_isolated_eligibility_and_delayed_reward_r5():
    """Test 4: Delayed reward (R5) only updates synapses when reward arrives."""
    iface = ContextActionInterface(
        hidden_dim=10,
        num_actions=2,
        architecture=ReadoutArchitecture.R5_DELAYED_REWARD,
        learning_rate=0.1,
        seed=42,
    )
    ctx = np.ones(10, dtype=np.float32)

    # 1. Step during delay (no reward) -> weights should not change
    w_init = iface.W_ctx_to_act.copy()
    for _ in range(5):
        iface.step(ctx, is_choice_step=False, is_reward_step=False)
    assert np.allclose(iface.W_ctx_to_act, w_init)
    assert np.any(np.abs(iface.eligibility) > 1e-4)

    # 2. Step on reward delivery -> weights update based on accumulated eligibility
    iface.step(ctx, reward=1.0, is_reward_step=True)
    assert not np.allclose(iface.W_ctx_to_act, w_init)


def test_credit_localization_zero_recurrent_leakage():
    """Test 5: Interface credit assignment has strictly 0% recurrent leakage."""
    iface = ContextActionInterface(
        hidden_dim=20,
        num_actions=2,
        architecture=ReadoutArchitecture.R3_THREE_FACTOR,
        learning_rate=0.05,
        seed=42,
    )
    ctx = np.ones(20, dtype=np.float32)
    iface.step(ctx, target_action=0, is_choice_step=True)

    summary = iface.get_credit_summary()
    assert summary["recurrent_leakage"] == 0.0
    assert summary["interface_credit"] > 0.0
    assert summary["precision"] > 0.0


def test_action_competition_modes():
    """Test 6: Competition modes configure intra-action excitation and lateral inhibition."""
    modes = [
        (ActionCompetitionMode.NO_INHIBITION, 0.0),
        (ActionCompetitionMode.WEAK_INHIBITION, 0.20),
        (ActionCompetitionMode.BASELINE_INHIBITION, 0.50),
        (ActionCompetitionMode.STRONG_INHIBITION, 1.00),
        (ActionCompetitionMode.NO_RECURRENCE, 0.0),
    ]
    for mode, expected_cross in modes:
        iface = ContextActionInterface(hidden_dim=10, num_actions=2, competition_mode=mode)
        assert np.isclose(iface.w_cross, expected_cross)


def test_causal_interventions():
    """Test 7: Causal intervention suite (scramble, swap, lesion, freeze)."""
    iface = ContextActionInterface(hidden_dim=10, num_actions=2, seed=42)
    ctx = np.arange(10, dtype=np.float32)

    # Scramble
    iface.apply_context_scramble(True)
    c_scrambled = iface.extract_context({"v_soma": ctx, "v_dend": ctx, "g_nmda": ctx, "activations": ctx})
    assert not np.array_equal(c_scrambled, ctx)
    iface.apply_context_scramble(False)

    # Swap
    iface.apply_context_swap(True)
    c_swapped = iface.extract_context({"v_soma": ctx, "v_dend": ctx, "g_nmda": ctx, "activations": ctx})
    assert np.array_equal(c_swapped[:5], ctx[5:])
    assert np.array_equal(c_swapped[5:], ctx[:5])
    iface.apply_context_swap(False)

    # Lesion
    w_pre = iface.W_ctx_to_act.copy()
    n_lesioned = iface.apply_readout_lesion(0.5)
    assert n_lesioned == 10
    assert np.sum(iface.W_ctx_to_act == 0.0) >= 10
    iface.restore_lesion()
    assert np.allclose(iface.W_ctx_to_act, w_pre)

    # Freeze plasticity
    iface.freeze_plasticity(True)
    assert not iface.plasticity_enabled
    iface.freeze_plasticity(False)
    assert iface.plasticity_enabled


def test_phase11_lesion_paradox_configuration():
    """Test 8: Lesion paradox configurations reproduce structural lesions and disinhibition."""
    iface = ContextActionInterface(
        hidden_dim=10,
        num_actions=2,
        architecture=ReadoutArchitecture.R7_PHASE11_ROUTING,
        seed=42,
    )
    # Relevant lesion zeroes first half to Action 0
    iface.configure_lesion_paradox_condition("RELEVANT_LESION")
    assert np.all(iface.W_ctx_to_act[:5, 0] == 0.0)

    # Irrelevant lesion zeroes second half to Action 1
    iface.configure_lesion_paradox_condition("IRRELEVANT_LESION")
    assert np.all(iface.W_ctx_to_act[5:, 1] == 0.0)

    # Cross inhibition disabled
    iface.configure_lesion_paradox_condition("CROSS_INH_DISABLED")
    assert iface.w_cross == 0.0

    # Cross inhibition reversed
    iface.configure_lesion_paradox_condition("CROSS_INH_REVERSED")
    assert iface.w_cross < 0.0


def test_representation_source_extraction():
    """Test 9: Representation sources correctly extract designated state variables."""
    state = {
        "v_soma": np.ones(10, dtype=np.float32) * -65.0,
        "v_dend": np.ones(10, dtype=np.float32) * -60.0,
        "g_nmda": np.ones(10, dtype=np.float32) * 0.25,
        "activations": np.ones(10, dtype=np.float32) * 0.8,
    }

    for src, expected_dim in [
        (RepresentationSource.SOMATIC, 10),
        (RepresentationSource.DENDRITIC, 10),
        (RepresentationSource.NMDA, 10),
        (RepresentationSource.ATTRACTOR, 10),
        (RepresentationSource.CONCATENATED, 40),
        (RepresentationSource.PCA, 8),
        (RepresentationSource.RANDOM_PROJ, 12),
    ]:
        iface = ContextActionInterface(
            hidden_dim=10,
            num_actions=2,
            source=src,
            pca_dim=8,
            random_proj_dim=12,
            seed=42,
        )
        ctx = iface.extract_context(state)
        assert len(ctx) == expected_dim


def test_phase12_network_integration():
    """Test 10: Phase12Network integrates biophysical substrate with learnable interface."""
    net = Phase12Network(
        input_dim=4,
        hidden_dim=12,
        output_dim=2,
        architecture=ReadoutArchitecture.R3_THREE_FACTOR,
        seed=42,
    )
    assert net.parameter_count() > 0
    x = np.ones(4, dtype=np.float32)

    v_act, sel_act = net.step(x, target_action=0, is_choice_step=True)
    assert len(v_act) == 2
    assert sel_act in (0, 1)

    obs = net.observable_state()
    assert "v_soma" in obs
    assert "interface" in obs
    assert obs["interface"]["recurrent_leakage"] == 0.0
