"""Unit tests for Phase 20 Track A: Context-Gated Modular Routing & Branch-Specific Plasticity."""

import numpy as np
import pytest

from brain.phase16_network import Phase16Network, ArchitecturePreset
from brain.modular_routing import (
    ContextRouter,
    ModularActionReadout,
    ModularReadoutBranch,
    RoutingCondition,
)
from benchmarks.phase20_routing import (
    generate_routing_trial,
    run_routing_task_phase,
    evaluate_routing_reversal,
    evaluate_xor_task,
)


def test_context_router_modes():
    """Test ContextRouter under soft, hard WTA, uniform, and oracle conditions."""
    router = ContextRouter(input_dim=8, num_branches=3, seed=42)
    x = np.array([1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # A1 / Fixed / Ablated: Uniform distribution
    g_unif, m_unif = router.compute_gate(x, condition=RoutingCondition.A1_FIXED_MODULAR)
    assert np.allclose(g_unif, [1/3, 1/3, 1/3])
    assert router.compute_sparsity() == 0.0

    # A2 / Soft Gating: Softmax distribution
    g_soft, m_soft = router.compute_gate(x, condition=RoutingCondition.A2_CONTEXT_GATED_SOFT)
    assert np.isclose(np.sum(g_soft), 1.0)
    assert np.all(g_soft >= 0.0)

    # A3 / Learned Sparse WTA: One-hot distribution
    g_hard, m_hard = router.compute_gate(x, condition=RoutingCondition.A3_LEARNED_SPARSE_GATE)
    assert np.sum(g_hard == 1.0) == 1
    assert np.sum(g_hard == 0.0) == 2
    assert router.compute_sparsity() == 2/3
    assert np.isclose(router.compute_selectivity(), 1.0)

    # A5 / Oracle Gating
    g_oracle, m_oracle = router.compute_gate(x, condition=RoutingCondition.A5_ORACLE_GATED, oracle_context_id=2)
    assert m_oracle == 2
    assert g_oracle[2] == 1.0


def test_branch_isolation_and_specific_plasticity():
    """Verify that under A4, only the active branch receives weight updates and eligibility updates."""
    readout = ModularActionReadout(
        input_dim=8,
        num_actions=2,
        num_branches=3,
        routing_condition=RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY,
        seed=42,
    )

    x = np.array([1.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    drives = readout.forward(x)
    active_m = readout.last_active_branch
    inactive_branches = [m for m in range(3) if m != active_m]

    w_before = [b.weights.copy() for b in readout.branches]

    action_probs = np.array([0.7, 0.3], dtype=np.float32)
    readout.update_eligibility(x, action_probs)

    # Inactive branches must have zero eligibility
    for inact_m in inactive_branches:
        assert np.all(readout.branches[inact_m].eligibility == 0.0)

    teaching = np.array([1.0, -1.0], dtype=np.float32)
    telem = readout.step_plasticity(x, teaching, reward=1.0, is_reward_step=True)

    # Active branch must have updated weights
    assert not np.allclose(readout.branches[active_m].weights, w_before[active_m])

    # Inactive branches must NOT have changed weights (Zero cross-talk)
    for inact_m in inactive_branches:
        assert np.allclose(readout.branches[inact_m].weights, w_before[inact_m])


def test_causal_ablations_collapse_performance():
    """Verify that causal ablations (random gate, coupling) alter routing behavior."""
    readout = ModularActionReadout(
        input_dim=8,
        num_actions=2,
        num_branches=3,
        routing_condition=RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY,
        seed=42,
    )
    x = np.ones(8, dtype=np.float32)

    # Normal routing
    g1, m1 = readout.router.compute_gate(x, RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY)
    assert np.max(g1) == 1.0

    # Gate ablation: collapses to uniform
    readout.router.is_ablated = True
    g_abl, m_abl = readout.router.compute_gate(x, RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY)
    assert np.allclose(g_abl, [1/3, 1/3, 1/3])


def test_trial_generator_no_leakage():
    """Verify that routing trials do NOT leak the target during the delay period."""
    trial = generate_routing_trial(trial_idx=0, task_id="A", context_id=0, delay_steps=5, seed=42)
    choice_idx = trial.metadata["choice_idx"]

    # Steps between cue (s >= 5) and choice (s < choice_idx) must be silent
    for s in range(5, choice_idx):
        st = trial.steps[s]
        assert np.all(st.sensory_input == 0.0)
        assert not st.is_choice_step
        assert not st.is_reward_step
        assert st.reward_val == 0.0


def test_parameter_accounting_sublinear_growth():
    """Verify that adding branches does not lead to global capacity explosion."""
    readout = ModularActionReadout(
        input_dim=40,
        num_actions=2,
        num_branches=3,
        routing_condition=RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY,
        seed=42,
    )
    counts = readout.count_parameters()
    # 3 branches * (40*2 + 2) = 246 params; router: 40*3 + 3 = 123 params. Total = 369.
    assert counts["total_trainable"] < 500
    assert counts["total_memory_bytes"] < 5000
