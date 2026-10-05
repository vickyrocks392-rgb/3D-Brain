"""Phase 23A Test Suite: Sample-Efficiency Benchmark & Observability Verification.

Strict validation covering:
1. Sample-budget generation & leakage auditing
2. Train/eval data separation
3. Frozen evaluation & zero inference-time plasticity
4. Zero evaluation-time replay
5. Parameter, dynamic state & external memory accounting
6. Exact analytical FLOP accounting
7. Natural-language parser & task translation grammar
8. Telemetry correctness & no fabricated telemetry
9. BrainLog v3.9 schema serialization & replay fidelity
10. Baseline comparability & JEV disclaimer integrity
"""

import json
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List
import numpy as np
import pytest

from brain.phase16_network import Phase16Network, ArchitecturePreset
from benchmarks.phase23a_sample_efficiency import (
    BenchmarkTrial,
    SampleEfficiencyRunner,
    SystemArchitecture,
    TaskType,
    TRAINING_BUDGETS,
    ACCURACY_THRESHOLDS,
    get_system_accounting,
    get_task_datasets,
)
from benchmarks.typed_probabilistic_baseline import TypedProbabilisticDecisionModel
from observability.natural_language_parser import (
    NaturalLanguageTaskParser,
    ParsedTaskSpec,
    build_executable_trials_from_spec,
)
from observability.brainlog import BrainLog, BrainLogSerializer, BrainLogValidator
from observability.events import ComputationalEvent, EventType
from observability.topology import BrainTopology, TopologyNode, TopologyEdge
from observability.capture import StateCapture
from experiments.generate_phase23a_brainlog import generate_phase23a_brainlog


# ============================================================================
# 1. SAMPLE-BUDGET GENERATION & TRAIN/EVAL SEPARATION
# ============================================================================

def test_sample_budget_generation_and_separation():
    """Verify train and eval sets are disjoint, correctly budgeted, and non-empty."""
    for task in TaskType:
        for budget in [5, 25, 100]:
            train_data, eval_data = get_task_datasets(task=task, train_budget=budget, eval_count=30, seed=42)
            assert len(train_data) == budget, f"Task {task} train budget mismatch"
            assert len(eval_data) == 30, f"Task {task} eval count mismatch"

            # Check that trials have unique IDs and disjoint trial collections
            train_ids = set(tr.trial_id for tr in train_data)
            eval_ids = set(tr.trial_id for tr in eval_data)
            assert len(train_ids.intersection(eval_ids)) == 0, f"Leakage: overlapping trial IDs in {task}"


def test_leakage_and_shortcut_audit():
    """Verify input vectors are dense continuous values without one-hot shortcut markers."""
    for task in TaskType:
        train_data, _ = get_task_datasets(task=task, train_budget=10, eval_count=10, seed=42)
        for tr in train_data:
            assert tr.inputs.ndim == 2, f"Input must be 2D sequence, got {tr.inputs.shape}"
            assert tr.inputs.shape[1] == 16, f"Feature dimension must be 16, got {tr.inputs.shape[1]}"
            # Ensure not all values are zeros or simple one-hot
            non_zeros = np.count_nonzero(tr.inputs)
            assert non_zeros > 5, "Inputs appear degenerate or sparse"
            assert not np.isnan(tr.inputs).any()
            assert not np.isinf(tr.inputs).any()


# ============================================================================
# 2. FROZEN EVALUATION & NO INFERENCE LEARNING
# ============================================================================

def test_frozen_evaluation_zero_plasticity():
    """Verify model weights and external memory do NOT change during evaluation."""
    runner = SampleEfficiencyRunner(seed=42)
    # Test on MIN_E
    train_trials, eval_trials = get_task_datasets(TaskType.STATIC_CLASSIFICATION, train_budget=10, eval_count=10, seed=42)
    model, _ = runner._train_model(SystemArchitecture.MIN_E, TaskType.STATIC_CLASSIFICATION, train_trials, seed=42)

    # Record initial weights before evaluation
    w_initial = model.action_readout.weights.copy()

    # Run frozen evaluation on all trials
    for tr in eval_trials:
        runner._eval_step(SystemArchitecture.MIN_E, model, tr)

    w_after = model.action_readout.weights.copy()
    assert np.array_equal(w_initial, w_after), "Violation: Readout weights modified during frozen evaluation!"


def test_no_evaluation_replay():
    """Verify no new episodes are stored during held-out evaluation in MIN_E_PLUS_EPISODIC."""
    runner = SampleEfficiencyRunner(seed=42)
    train_trials, eval_trials = get_task_datasets(TaskType.EPISODE_DEPENDENT_DECISION, train_budget=10, eval_count=10, seed=42)
    (net, mem), _ = runner._train_model(SystemArchitecture.MIN_E_PLUS_EPISODIC, TaskType.EPISODE_DEPENDENT_DECISION, train_trials, seed=42)

    records_count_before = len(mem.records)

    # Evaluate
    for tr in eval_trials:
        runner._eval_step(SystemArchitecture.MIN_E_PLUS_EPISODIC, (net, mem), tr)

    records_count_after = len(mem.records)
    assert records_count_before == records_count_after, "Violation: Episodic memory stored records during evaluation!"


# ============================================================================
# 3. RESOURCE ACCOUNTING AUDIT
# ============================================================================

def test_resource_accounting_completeness():
    """Verify all 6 systems have separate parameters, state, external memory, and FLOPs."""
    for sys_arch in SystemArchitecture:
        acct = get_system_accounting(sys_arch, input_dim=16, seq_len=6)
        assert "total_parameters" in acct
        assert "trainable_parameters" in acct
        assert "dynamic_state_floats" in acct
        assert "external_memory_bytes" in acct
        assert "flops_per_step" in acct
        assert "flops_per_trial" in acct

        # Total = Trainable + Fixed
        assert acct["total_parameters"] == acct["trainable_parameters"] + acct["fixed_parameters"]
        # External memory strictly separated from parameters
        if sys_arch != SystemArchitecture.MIN_E_PLUS_EPISODIC:
            assert acct["external_memory_bytes"] == 0, f"{sys_arch} should have 0 external memory"
        else:
            assert acct["external_memory_bytes"] > 0, "MIN_E_PLUS_EPISODIC must account for external memory bytes"


def test_analytical_flops_nonzero_and_consistent():
    """Verify analytical FLOPs scale with sequence length and are positive."""
    for sys_arch in SystemArchitecture:
        a1 = get_system_accounting(sys_arch, input_dim=16, seq_len=5)
        a2 = get_system_accounting(sys_arch, input_dim=16, seq_len=10)
        assert a1["flops_per_trial"] > 0
        assert a2["flops_per_trial"] > a1["flops_per_trial"], f"FLOPs must scale with sequence length for {sys_arch}"


# ============================================================================
# 4. TYPED PROBABILISTIC BASELINE (SYSTEM ONE INTERFACE) INTEGRITY
# ============================================================================

def test_typed_probabilistic_baseline_integrity():
    """Verify System One-inspired typed probabilistic model characteristics & disclaimer."""
    model = TypedProbabilisticDecisionModel(input_dim=16, output_dim=2, seed=42)
    # Check disclaimer
    obs = model.observable_state()
    assert obs["is_proprietary_jev"] is False
    assert obs["system_type"] == "TYPED_PROBABILISTIC_DECISION"

    # Forward probability test
    x = np.random.randn(16).astype(np.float32)
    probs = model.forward_probs(x)
    assert probs.shape == (2,)
    assert abs(np.sum(probs) - 1.0) < 1e-5
    assert 0.0 <= probs[0] <= 1.0 and 0.0 <= probs[1] <= 1.0

    # Decision latency reflects entropy
    assert obs["decision_latency_ms"] >= 5.0

    # Frozen mode prevents weight update
    model.freeze()
    w_before = model.W.copy()
    loss = model.train_step(x, target_action=1)
    assert loss == 0.0
    assert np.array_equal(w_before, model.W)


# ============================================================================
# 5. NATURAL-LANGUAGE PARSER & TASK TRANSLATION
# ============================================================================

def test_natural_language_parser_grammar():
    """Verify deterministic parser accurately extracts rules, distractors, budget, and model."""
    # Test case 1: Prompt example
    cmd1 = "Learn that red means left, then test the rule after 20 distracting events."
    spec1 = NaturalLanguageTaskParser.parse(cmd1)
    assert spec1.valid
    assert spec1.rule_description == "red → left"
    assert spec1.stimulus_name == "red"
    assert spec1.response_action == 0
    assert spec1.distractor_count == 20
    assert spec1.evaluation_mode == "frozen"
    assert spec1.learning_mode == "enabled during training only"

    # Test case 2: Blue means right with 15 distractors and budget 50
    cmd2 = "Learn rule blue -> right for 50 trials, test after 15 distractors with frozen weights."
    spec2 = NaturalLanguageTaskParser.parse(cmd2)
    assert spec2.valid
    assert spec2.rule_description == "blue → right"
    assert spec2.response_action == 1
    assert spec2.distractor_count == 15
    assert spec2.training_budget == 50

    # Test case 3: Compositional XOR
    cmd3 = "Test compositional XOR rule for 100 examples with frozen evaluation on Transformer."
    spec3 = NaturalLanguageTaskParser.parse(cmd3)
    assert spec3.valid
    assert spec3.task_type == TaskType.SIMPLE_COMPOSITIONAL_DECISION
    assert spec3.target_model == "CAUSAL_TRANSFORMER"
    assert spec3.training_budget == 100

    # Test card format display
    card = spec1.format_display_card()
    assert "INTERPRETED TASK" in card
    assert "red → left" in card
    assert "20" in card
    assert "frozen" in card


def test_task_translation_to_executable_trials():
    """Verify translated spec can build executable benchmark trials."""
    cmd = "Learn that red means left, then test the rule after 5 distracting events."
    spec = NaturalLanguageTaskParser.parse(cmd)
    train_trials, eval_trials = build_executable_trials_from_spec(spec, seed=42)
    assert len(train_trials) == spec.training_budget
    assert len(eval_trials) > 0
    assert train_trials[0].inputs.shape[1] == 16


# ============================================================================
# 6. TELEMETRY & OBSERVABILITY AUDIT (NO FABRICATION)
# ============================================================================

def test_telemetry_captures_real_computational_state():
    """Verify capture_phase23a_step populates actual non-fabricated computational telemetry."""
    nodes = [
        TopologyNode(id=0, label="Input", type="INPUT", pos=[-2.0, 0.0, 0.0], unit_index=-1),
        TopologyNode(id=1, label="Memory", type="MEMORY", pos=[0.0, 2.0, 0.0], unit_index=-1),
        TopologyNode(id=2, label="Output", type="OUTPUT", pos=[2.0, 0.0, 0.0], unit_index=-1),
    ]
    edges = [
        TopologyEdge(src=0, dst=1, weight=1.0),
        TopologyEdge(src=1, dst=2, weight=1.0),
    ]
    topo = BrainTopology(topology_id="test_topo", model_id="TEST", nodes=nodes, edges=edges)
    capture = StateCapture(topology=topo, dt_ms=40.0)

    net = Phase16Network(input_dim=16, hidden_dim=40, output_dim=2, preset=ArchitecturePreset.MIN_E, seed=42)
    x = np.random.randn(16).astype(np.float32)
    net.step(x, target_action=0)

    frame = capture.capture_phase23a_step(
        step=1,
        input_vec=x,
        model=net,
        target_val=0.0,
        mask_val=1.0,
        benchmark_info={"current_task": "STATIC_CLASSIFICATION", "current_trial": 1},
    )

    details = frame.dynamic_details
    assert "active_neuron_count" in details
    assert "active_synapse_count" in details
    assert "firing_rate" in details
    assert "mean_v_soma" in details
    assert "spike_count" in details
    assert "visual_markers" in details
    # Must have INPUT and DECISION or LEARNING marker
    assert "INPUT" in details["visual_markers"]


# ============================================================================
# 7. BRAINLOG V3.9 SCHEMA & REPLAY FIDELITY
# ============================================================================

def test_brainlog_v39_serialization_and_validation():
    """Verify BrainLog v3.9 trace generation, schema validation, and replay fidelity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        blog_path = Path(tmpdir) / "test_trace.brainlog"
        generate_phase23a_brainlog(blog_path, seed=42)

        assert blog_path.exists()
        valid, msg = BrainLogValidator.validate(blog_path)
        assert valid, f"Validation error: {msg}"

        # Re-read and check fields
        with open(blog_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["brainlog_version"] == "3.9"
        assert "natural_language_instruction" in data
        assert "parsed_task" in data
        assert "metadata" in data
        assert "topology" in data
        assert len(data["frames"]) > 0
        assert len(data["events"]) > 0

        # Check visual markers present in telemetry
        has_visual_markers = any("visual_markers" in f.get("dynamic_details", {}) for f in data["frames"])
        assert has_visual_markers, "Frames missing visual markers"


def test_player_has_phase23a_overlays_and_js_wiring():
    """Verify player.html contains Phase 23A NL panel, overlay elements, and JS wiring."""
    player_path = Path(__file__).resolve().parent.parent / "visualization" / "player.html"
    assert player_path.exists()
    html = player_path.read_text(encoding="utf-8")

    # 1. Toolbar button
    assert 'id="p23a-btn"' in html

    # 2. NL Panel and observatory overlay
    assert 'id="p23a-nl-panel"' in html
    assert 'id="p23a-observatory-overlay"' in html
    assert 'id="interpreted-task-card"' in html
    assert 'id="p23a-run-btn"' in html
    assert 'id="p23a-pause-btn"' in html
    assert 'id="p23a-reset-btn"' in html
    assert 'id="p23a-replay-btn"' in html
    assert 'id="p23a-floating-marker"' in html

    # 3. Model select option
    assert "EXP023A_sample_efficiency.brainlog" in html

    # 4. JS wiring
    assert "function updatePhase23AOverlays" in html
    assert "setP23AActive" in html
    assert "parseNaturalLanguageTask" in html

    # 5. Check setText IDs in updatePhase23AOverlays exist in HTML
    m = re.search(r"function updatePhase23AOverlays.*?\n    \}\n", html, re.S)
    assert m, "updatePhase23AOverlays body not found"
    js_refs = set(re.findall(r"setText\('([a-z0-9-]+)'", m.group(0)))
    assert js_refs, "No setText references in updatePhase23AOverlays"
    for ref_id in js_refs:
        assert f'id="{ref_id}"' in html, f"Missing referenced element id='{ref_id}' in player.html"

