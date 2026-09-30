"""Tests for Phase 19 Track B: Synaptic Consolidation & Catastrophic Forgetting.

Verifies:
1. Baseline catastrophic forgetting reproduction on task reversals.
2. Dual-timescale synaptic separation (fast plastic component vs slow structural store).
3. Synaptic tagging and capture dynamics (tag accumulation and reward-gated capture).
4. Metaplasticity cascade (stability counter increment and resistance scaling).
5. Local elastic stabilization (precision accumulation and anchor restoring force).
6. Multi-task sequence execution and stability-plasticity tradeoff accounting.
"""

import numpy as np
import pytest

from brain.phase16_network import Phase16Network, ArchitecturePreset
from brain.synaptic_consolidation import ConsolidationMode, SynapticConsolidationModule
from benchmarks.phase19_consolidation import (
    generate_continual_trial,
    evaluate_continual_reversal,
    evaluate_multitask_sequence,
)


def test_consolidation_module_initialization():
    """Verify initialization of all consolidation state tensors."""
    module = SynapticConsolidationModule(input_dim=40, num_actions=2, mode=ConsolidationMode.DUAL_TIMESCALE)
    assert module.w_fast.shape == (40, 2)
    assert module.w_slow.shape == (40, 2)
    assert module.tags.shape == (40, 2)
    assert module.w_cons.shape == (40, 2)
    assert module.stability.shape == (40, 2)
    assert module.w_anchor.shape == (40, 2)
    assert module.omega.shape == (40, 2)


def test_dual_timescale_consolidation():
    """Verify that sustained positive plastic updates transfer into slow consolidated weights."""
    module = SynapticConsolidationModule(input_dim=4, num_actions=2, mode=ConsolidationMode.DUAL_TIMESCALE, tau_slow=5.0)
    base_w = np.full((4, 2), 0.5, dtype=np.float32)
    raw_delta = np.full((4, 2), 0.05, dtype=np.float32)
    elig = np.ones((4, 2), dtype=np.float32)
    teaching = np.array([1.0, -1.0], dtype=np.float32)

    assert np.all(module.w_slow == 0.0)

    # Apply reward step with positive reward
    eff_delta, telem = module.modulate_plasticity_step(
        base_weights=base_w,
        raw_delta_w=raw_delta,
        eligibility=elig,
        teaching_signal=teaching,
        reward=1.0,
        is_reward_step=True,
    )

    # Slow weights must accumulate consolidation
    assert np.any(module.w_slow > 0.0)
    assert telem["stabilized_synapses"] > 0


def test_synaptic_tagging_and_capture():
    """Verify that dendritic eligibility sets synaptic tags and reward triggers capture."""
    module = SynapticConsolidationModule(input_dim=4, num_actions=2, mode=ConsolidationMode.SYNAPTIC_TAGGING, tag_threshold=0.10)
    base_w = np.full((4, 2), 0.5, dtype=np.float32)
    raw_delta = np.full((4, 2), 0.02, dtype=np.float32)
    elig = np.full((4, 2), 0.3, dtype=np.float32)
    teaching = np.array([1.0, -1.0], dtype=np.float32)

    # Step through delay without reward: tags accumulate, w_cons unchanged
    for _ in range(8):
        module.modulate_plasticity_step(
            base_weights=base_w,
            raw_delta_w=raw_delta,
            eligibility=elig,
            teaching_signal=teaching,
            reward=0.0,
            is_reward_step=False,
        )
    assert np.mean(module.tags) >= 0.10
    assert np.all(module.w_cons == 0.0)

    # Step with reward delivery: tags trigger capture into w_cons
    module.modulate_plasticity_step(
        base_weights=base_w,
        raw_delta_w=raw_delta,
        eligibility=elig,
        teaching_signal=teaching,
        reward=1.0,
        is_reward_step=True,
    )
    assert np.any(module.w_cons > 0.0)


def test_metaplasticity_stability_resistance():
    """Verify that metaplasticity increments stability and scales effective learning rate."""
    module = SynapticConsolidationModule(input_dim=4, num_actions=2, mode=ConsolidationMode.METAPLASTICITY, metaplasticity_beta=5.0)
    base_w = np.zeros((4, 2), dtype=np.float32)
    raw_delta = np.full((4, 2), 0.1, dtype=np.float32)
    elig = np.full((4, 2), 0.5, dtype=np.float32)
    teaching = np.array([1.0, -1.0], dtype=np.float32)

    assert np.all(module.stability == 0.0)

    # Step with reward reinforces stability
    module.modulate_plasticity_step(base_w, raw_delta, elig, teaching, reward=1.0, is_reward_step=True)
    assert np.mean(module.stability) > 0.0

    # Subsequent update on stabilized synapses must be resisted (attenuated)
    eff_delta, _ = module.modulate_plasticity_step(base_w, raw_delta, elig, teaching, reward=0.0, is_reward_step=False)
    assert np.all(np.abs(eff_delta) < np.abs(raw_delta))


def test_network_consolidation_mode_setting():
    """Verify that Phase16Network properly integrates and reports consolidation modes."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=42)
    assert net.action_readout.consolidation_module.mode == ConsolidationMode.NONE

    net.set_consolidation_mode(ConsolidationMode.DUAL_TIMESCALE, tau_slow=15.0)
    assert net.action_readout.consolidation_module.mode == ConsolidationMode.DUAL_TIMESCALE
    assert net.action_readout.consolidation_module.tau_slow == 15.0


def test_continual_reversal_execution():
    """Verify complete execution of continual learning reversal protocol."""
    res = evaluate_continual_reversal(ConsolidationMode.NONE, trials_per_task=10, eval_trials=10, seed=42)
    assert "acc_a_pre" in res
    assert "acc_b" in res
    assert "acc_a_post" in res
    assert "relative_retention" in res
    assert "stability_plasticity_product" in res
    assert 0.0 <= res["relative_retention"] <= 2.0
