"""Tests for Phase 17: Generalization, Transfer & Task-Independent Validation."""

import json
from pathlib import Path
import pytest
import numpy as np

from brain.phase17_suite import (
    Phase17Network,
    TaskFamilyA_DelayedChoice,
    TaskFamilyB_SequencePrediction,
    TaskFamilyC_EmbodiedSpatial,
    TaskFamilyD_ContinualLearning,
    TaskFamilyE_CompositionalContext,
    TaskFamilyF_DistractorInterference,
    TaskStep,
    TaskTrial,
    evaluate_task_trials,
)
from brain.phase16_network import Phase16Mechanism, ArchitecturePreset
from observability.brainlog import BrainLogValidator
from observability.events import ComputationalEvent, EventType


def test_brainlog_phase17_validation():
    """Verify that the generated Phase 17 BrainLog is valid under schema v3.3."""
    bpath = Path(__file__).resolve().parent.parent / "experiments" / "results" / "brainlogs" / "EXP017_generalization.brainlog"
    assert bpath.exists(), f"BrainLog file not found at {bpath}"
    valid, reason = BrainLogValidator.validate(bpath)
    assert valid, f"Phase 17 BrainLog failed validation: {reason}"


def test_exp017_results_json_structure():
    """Verify that EXP_017_RESULTS.json exists and contains all required stages."""
    rpath = Path(__file__).resolve().parent.parent / "experiments" / "results" / "EXP_017_RESULTS.json"
    assert rpath.exists(), f"Results JSON not found at {rpath}"
    with open(rpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert data["metadata"]["architecture"] == "MIN_E"
    assert "stages" in data
    stages = data["stages"]

    expected_stages = [
        "stage_a_reproduction",
        "stage_b_temporal_extrapolation",
        "stage_c_transfer_matrix",
        "stage_d_continual_learning",
        "stage_e_compositional",
        "stage_f_distractor_robustness",
        "stage_g_mechanism_generality",
        "stage_h_parameter_scaling",
        "stage_i_scorecard",
    ]
    for stg in expected_stages:
        assert stg in stages, f"Missing required stage: {stg}"


def test_task_family_generators():
    """Verify all 6 task family generators produce valid TaskTrials."""
    # Task A
    gen_a = TaskFamilyA_DelayedChoice()
    tr_a = gen_a.generate_trial(0, delay_steps=5, context_id=0)
    assert isinstance(tr_a, TaskTrial)
    assert len(tr_a.steps) > 5
    assert any(s.is_choice_step for s in tr_a.steps)

    # Task B
    gen_b = TaskFamilyB_SequencePrediction()
    tr_b = gen_b.generate_trial(0, pattern_id=1)
    assert isinstance(tr_b, TaskTrial)
    assert tr_b.target_action == 1

    # Task C
    gen_c = TaskFamilyC_EmbodiedSpatial()
    tr_c = gen_c.generate_trial(0, target_landmark_id=0)
    assert isinstance(tr_c, TaskTrial)
    assert tr_c.target_action == 0

    # Task D
    gen_d = TaskFamilyD_ContinualLearning()
    tr_d = gen_d.generate_trial(0, regime="REGIME_B", context_id=1)
    assert isinstance(tr_d, TaskTrial)
    assert tr_d.target_action == 0  # Inverted mapping: 1-1=0

    # Task E
    gen_e = TaskFamilyE_CompositionalContext()
    tr_e = gen_e.generate_trial(0, ctx_a=1, ctx_b=1)
    assert isinstance(tr_e, TaskTrial)
    assert tr_e.target_action == 0  # 1 XOR 1 = 0

    # Task F
    gen_f = TaskFamilyF_DistractorInterference()
    tr_f = gen_f.generate_trial(0, context_id=1, interference_level="HIGH")
    assert isinstance(tr_f, TaskTrial)
    assert tr_f.target_action == 1


def test_phase17_network_freezing_and_events():
    """Verify Phase17Network freezing controls and explicit event creation."""
    net = Phase17Network(preset=ArchitecturePreset.MIN_E, seed=42)
    assert not net.is_frozen

    # Test freezing
    net.set_frozen(True)
    assert net.is_frozen
    assert net.action_readout.is_frozen

    # Test unfreezing
    net.set_frozen(False)
    assert not net.is_frozen
    assert not net.action_readout.is_frozen

    # Test Phase 17 Event Creation
    evt = net.record_phase17_event(
        event_type=EventType.TASK_STARTED,
        step=1,
        time_ms=40.0,
        payload={"task_name": "TASK_A"},
    )
    assert evt.event_type == EventType.TASK_STARTED
    assert len(net.phase17_events) == 1


def test_phase17_mechanism_ablations():
    """Verify that mechanism ablations alter active mechanism lists."""
    net = Phase17Network(preset=ArchitecturePreset.MIN_E, seed=42)
    initial_mechs = set(net.get_active_mechanisms())
    assert "K_PERSISTENT_LATENT_NMDA" in initial_mechs

    net.ablate_mechanism(Phase16Mechanism.K_PERSISTENT_LATENT_NMDA)
    assert "K_PERSISTENT_LATENT_NMDA" not in net.get_active_mechanisms()
    assert "K_PERSISTENT_LATENT_NMDA" in net.get_removed_mechanisms()

    net.restore_mechanism(Phase16Mechanism.K_PERSISTENT_LATENT_NMDA)
    assert "K_PERSISTENT_LATENT_NMDA" in net.get_active_mechanisms()
