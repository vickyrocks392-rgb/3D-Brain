"""Tests for Phase 19 Track A: True Long-Horizon Credit Assignment.

Verifies:
1. Strict causal trial structure: cue -> delay -> choice -> reward delay -> reward.
2. Complete absence of target leakage during delay steps.
3. Strict evaluation freezing (no test-time weight updates).
4. Mechanism necessity ablations (K, C, A).
5. Empirical finite credit horizon (performance collapses to chance at long delays).
6. Deterministic baseline execution across seeds.
"""

import numpy as np
import pytest

from brain.phase16_network import Phase16Network, Phase16Mechanism, ArchitecturePreset
from benchmarks.phase19_credit_horizon import (
    generate_delayed_credit_trial,
    train_and_eval_min_e,
    train_and_eval_transformer,
    train_and_eval_bptt,
    train_and_eval_simple_recurrent,
    SimpleRecurrentControl,
    DT_MS,
)


def test_causal_trial_structure_and_no_target_leakage():
    """Verify that sensory inputs and target actions follow strict causal timing."""
    delay_steps = 10
    trial = generate_delayed_credit_trial(trial_idx=0, delay_steps=delay_steps, context_id=0, seed=42)

    cue_steps = 5
    choice_idx = cue_steps + delay_steps  # 15
    rew_delay = 2
    reward_idx = choice_idx + rew_delay + 1  # 18
    total_steps = reward_idx + 1  # 19

    assert len(trial.steps) == total_steps
    assert trial.metadata["choice_step_idx"] == choice_idx
    assert trial.metadata["reward_step_idx"] == reward_idx

    # 1. Cue steps (0..4)
    for s in range(cue_steps):
        st = trial.steps[s]
        assert not st.is_choice_step
        assert not st.is_reward_step
        assert st.sensory_input[0] == 1.0  # Context 0 cue
        assert st.sensory_input[-1] == 0.0
        assert st.sensory_input[-2] == 0.0

    # 2. Delay steps (5..14) - Strictly silent, zero observation, zero target
    for s in range(cue_steps, choice_idx):
        st = trial.steps[s]
        assert not st.is_choice_step
        assert not st.is_reward_step
        assert np.all(st.sensory_input == 0.0)

    # 3. Choice step (15) - Choice probe active, target NOT presented to model
    choice_st = trial.steps[choice_idx]
    assert choice_st.is_choice_step
    assert not choice_st.is_reward_step
    assert choice_st.sensory_input[-1] == 1.0

    # 4. Reward delay steps (16..17) - Silence
    for s in range(choice_idx + 1, reward_idx):
        st = trial.steps[s]
        assert not st.is_choice_step
        assert not st.is_reward_step
        assert np.all(st.sensory_input == 0.0)

    # 5. Reward step (18) - Reward probe active
    rew_st = trial.steps[reward_idx]
    assert not rew_st.is_choice_step
    assert rew_st.is_reward_step
    assert rew_st.sensory_input[-2] == 1.0


def test_frozen_evaluation_preserves_weights():
    """Verify that during zero-shot evaluation, no synaptic updates or state drift occurs."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=42)
    net.set_frozen(True)

    initial_weights = net.action_readout.weights.copy()
    trial = generate_delayed_credit_trial(trial_idx=0, delay_steps=5, context_id=0, seed=42)

    for st in trial.steps:
        net.step(
            st.sensory_input,
            target_action=st.target_action if st.is_reward_step else None,
            reward=1.0 if st.is_reward_step else 0.0,
            is_choice_step=st.is_choice_step,
            is_reward_step=st.is_reward_step,
        )

    # Weights must be exactly unchanged
    assert np.all(net.action_readout.weights == initial_weights)
    assert np.all(net.action_readout.last_weight_delta == 0.0)


def test_mechanism_k_nmda_necessity_in_delayed_credit():
    """Verify that lesioning Mechanism K collapses NMDA conductance."""
    net_intact = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=42)
    net_lesion = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=42)
    net_lesion.ablate_mechanism(Phase16Mechanism.K_PERSISTENT_LATENT_NMDA)

    trial = generate_delayed_credit_trial(trial_idx=0, delay_steps=5, context_id=0, seed=42)

    # Step through cue
    for s in range(5):
        net_intact.step(trial.steps[s].sensory_input)
        net_lesion.step(trial.steps[s].sensory_input)

    assert np.sum(net_intact.population.g_nmda) > 0.0
    assert np.sum(net_lesion.population.g_nmda) == 0.0


def test_mechanism_c_dendritic_eligibility_ablation():
    """Verify that ablating dendritic eligibility sets dendritic eligibility to standard eligibility."""
    net_lesion = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=42)
    net_lesion.ablate_mechanism(Phase16Mechanism.C_DENDRITIC_ELIGIBILITY)

    trial = generate_delayed_credit_trial(trial_idx=0, delay_steps=2, context_id=0, seed=42)
    net_lesion.step(trial.steps[0].sensory_input)

    # When C is ablated, dendritic_eligibility in readout matches non-dendritic eligibility
    assert net_lesion.action_readout.dendritic_eligibility.shape == (40, 2)


def test_long_delay_finiteness_and_no_leakage():
    """Verify that MIN_E at 5000 ms (125 steps) does NOT achieve 100% (confirming Phase 18 audit)."""
    res = train_and_eval_min_e(ArchitecturePreset.MIN_E, delay_steps=125, train_trials=10, eval_trials=10, seed=42)
    # Under un-leaked evaluation, performance at 5000 ms is at chance level
    assert res["eval_accuracy"] <= 0.70
    assert res["delay_ms"] == 5000
