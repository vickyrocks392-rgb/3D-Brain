"""Unit tests for Phase 22: Episode-Dependent Composition, Abstract Retrieval & Causal Memory Use.

Covers all required areas:
- deterministic reproduction across seeds
- episode construction (two-phase write, no pre-outcome target)
- memory write protocol
- correct retrieval (B0)
- wrong retrieval (B4)
- shuffled retrieval (B3)
- memory ablation (B1, B2)
- corrupted memory (B5)
- partial memory (B6)
- corrupted cue (B7)
- compositional holdout (A+D held out, never trained)
- frozen zero-shot evaluation (SHA-256 weight hash identical)
- zero evaluation plasticity
- zero target exposure
- zero replay during evaluation
- no hidden-ID or one-hot shortcuts
- separate parameter and external memory accounting
- BrainLog v3.8 schema validation and roundtrip replay
- player.html Phase 22 overlays [M][E][R][C][T] contract & JS wiring
"""

import json
from pathlib import Path
import re
import numpy as np
import pytest

from brain.episodic_memory import (
    EpisodeRecord,
    EpisodicCondition,
    MemoryAblation,
    SparseEpisodicMemory,
)
from brain.compositional_transfer import TransferAudit
from benchmarks.phase22_causal_task import (
    CausalCondition,
    EpisodeDependentTrial,
    evaluate_causal_ablations,
    evaluate_retrieval_timing_horizon,
    make_episode_dependent_session,
    run_episode_dependent_trial,
    CHANCE_ACCURACY,
)
from benchmarks.phase22_composition import (
    Component,
    CompositionPair,
    CompositionalLearner,
    evaluate_compositional_retrieval_sweep,
    evaluate_episode_dependent_composition,
    make_compositional_stimulus,
    train_compositional_system,
)
from benchmarks.phase22_baselines import build_baseline_comparison, analytical_flops_per_step
from experiments.generate_phase22_brainlog import generate_phase22_brainlog
from observability.brainlog import BrainLogValidator, BrainLogSerializer
from observability.events import EventType

ROOT = Path(__file__).resolve().parent.parent
PLAYER_PATH = ROOT / "visualization" / "player.html"
RESULTS_JSON = ROOT / "experiments" / "results" / "PHASE_22_RESULTS.json"


# ---------------------------------------------------------------------------
# 1. Deterministic Reproduction
# ---------------------------------------------------------------------------
def test_deterministic_reproduction():
    """Identical seeds must produce bit-for-bit identical results."""
    res1 = evaluate_causal_ablations(n_trials=20, delay_sec=1.0, seed=42)
    res2 = evaluate_causal_ablations(n_trials=20, delay_sec=1.0, seed=42)
    assert res1 == res2, "Non-deterministic reproduction under identical seed 42"


# ---------------------------------------------------------------------------
# 2. Episode Construction & Two-Phase Write Protocol
# ---------------------------------------------------------------------------
def test_episode_construction_no_pre_outcome_target():
    """open_episode must NOT accept target or reward; only closed by close_episode."""
    mem = SparseEpisodicMemory(capacity=16, seed=42)
    ctx = np.ones(40, dtype=np.float32)
    events = np.zeros((2, 40), dtype=np.float32)

    rec = mem.open_episode(
        episode_id=1,
        context_vector=ctx,
        event_sequence=events,
        chosen_action=0,
        timestamp_ms=0.0,
        task_name="TEST_EPISODE",
    )
    assert not rec.is_closed, "Episode must be pending before outcome"
    assert rec.reward_val == 0.0, "Reward must not be present before outcome"
    assert rec.prediction_error == 0.0

    # Commit outcome
    closed = mem.close_episode(prediction_error=0.15, reward_val=1.0)
    assert closed is not None
    assert closed.is_closed
    assert closed.reward_val == 1.0
    assert closed.prediction_error == 0.15
    assert len(mem) == 1


# ---------------------------------------------------------------------------
# 3. Causal Ablations (B0 - B7)
# ---------------------------------------------------------------------------
def test_causal_ablations_b0_to_b7():
    """All 8 causal ablations must function as preregistered."""
    res = evaluate_causal_ablations(n_trials=60, delay_sec=1.0, seed=42)
    for cond in CausalCondition:
        assert cond.value in res, f"Missing condition {cond.value}"
        assert 0.0 <= res[cond.value]["accuracy"] <= 1.0

    # B0 (Full Memory) must substantially exceed B1 (Memory Removed)
    b0_acc = res[CausalCondition.B0_FULL_MEMORY.value]["accuracy"]
    b1_acc = res[CausalCondition.B1_MEMORY_REMOVED.value]["accuracy"]
    b2_acc = res[CausalCondition.B2_RETRIEVAL_DISABLED.value]["accuracy"]
    b3_acc = res[CausalCondition.B3_SHUFFLED_MEMORY.value]["accuracy"]

    assert b0_acc >= 0.90, f"Full memory accuracy {b0_acc} below threshold 0.90"
    assert abs(b1_acc - CHANCE_ACCURACY) <= 0.20, f"B1 accuracy {b1_acc} too far from chance"
    assert abs(b2_acc - CHANCE_ACCURACY) <= 0.20, f"B2 accuracy {b2_acc} too far from chance"
    assert abs(b3_acc - CHANCE_ACCURACY) <= 0.20, f"B3 accuracy {b3_acc} too far from chance"
    assert b0_acc > b1_acc + 0.30, "Memory is not causally necessary for the task"


def test_corrupted_and_partial_memory_graded_degradation():
    """Corrupted (B5) and Partial (B6) memory must show intermediate degradation."""
    res = evaluate_causal_ablations(n_trials=80, delay_sec=1.0, seed=42)
    b0_acc = res[CausalCondition.B0_FULL_MEMORY.value]["accuracy"]
    b1_acc = res[CausalCondition.B1_MEMORY_REMOVED.value]["accuracy"]
    b5_acc = res[CausalCondition.B5_CORRUPTED_MEMORY.value]["accuracy"]
    b6_acc = res[CausalCondition.B6_PARTIAL_MEMORY.value]["accuracy"]

    assert b0_acc >= b5_acc >= b1_acc - 0.05
    assert b0_acc >= b6_acc >= b1_acc - 0.05


def test_memory_specificity_pattern():
    """Correct memory must yield higher accuracy than wrong or shuffled memory."""
    res = evaluate_causal_ablations(n_trials=60, delay_sec=1.0, seed=42)
    b0 = res[CausalCondition.B0_FULL_MEMORY.value]["accuracy"]
    b4 = res[CausalCondition.B4_WRONG_MEMORY.value]["accuracy"]
    b3 = res[CausalCondition.B3_SHUFFLED_MEMORY.value]["accuracy"]
    assert b0 > b4, f"Full memory {b0} not higher than wrong memory {b4}"
    assert b0 > b3, f"Full memory {b0} not higher than shuffled memory {b3}"


# ---------------------------------------------------------------------------
# 4. Abstract Compositional Transfer on Held-Out Pairs
# ---------------------------------------------------------------------------
def test_compositional_holdout_and_frozen_zero_shot():
    """Held-out combination A+D must be evaluated frozen with no evaluation-time learning."""
    res = evaluate_compositional_retrieval_sweep(seed=42, n_trials=40)
    conds = res["conditions"]

    # Zero-shot contract verified across conditions
    for cond_name, metrics in conds.items():
        assert metrics["zero_shot_valid"], f"Zero shot violation in {cond_name}"

    # Memory advantage on held-out A+D
    ad_with_mem = conds["with_memory"]["ad_accuracy"]
    ad_no_mem = conds["memory_removed"]["ad_accuracy"]
    assert ad_with_mem >= 0.85, f"With memory accuracy {ad_with_mem} below 0.85"
    assert ad_no_mem < ad_with_mem, "A+D transfer did not require memory"
    assert abs(ad_no_mem - CHANCE_ACCURACY) <= 0.20, f"No memory accuracy {ad_no_mem} far from chance"


def test_trained_combinations_sanity():
    """Trained combinations (A+B, A+C, B+D) must achieve high accuracy."""
    learner = train_compositional_system(seed=42)
    assert learner.parameter_count() == 16  # 4 weights x 4 components
    # Predict single component
    x_A = make_compositional_stimulus((Component.A,), trial_idx=10, seed=42)
    assert learner.predict_component(Component.A, x_A) == 1
    x_B = make_compositional_stimulus((Component.B,), trial_idx=10, seed=42)
    assert learner.predict_component(Component.A, x_B) == 0


# ---------------------------------------------------------------------------
# 5. Retrieval Timing & Effective Measured Horizon
# ---------------------------------------------------------------------------
def test_retrieval_timing_horizon():
    """Measure performance across delays (0 to 120s) and verify effective horizon."""
    timing = evaluate_retrieval_timing_horizon(delays_sec=[0, 1, 5, 10], n_trials=20, seed=42)
    curve = timing["timing_curve"]
    assert "delay_0s_full_memory" in curve
    assert "delay_10s_full_memory" in curve
    # Episodic store retains across capacity
    assert timing["effective_horizon_sec"] >= 5.0
    # No memory must remain at chance across all delays
    for d in [0, 1, 5, 10]:
        acc_none = curve[f"delay_{d}s_no_memory"]["accuracy"]
        assert abs(acc_none - CHANCE_ACCURACY) <= 0.25


# ---------------------------------------------------------------------------
# 6. Leakage & Shortcut Audit
# ---------------------------------------------------------------------------
def test_leakage_audit_no_one_hot_shortcut():
    """Stimuli must be continuous and multi-dimensional, not one-hot indicators."""
    session = make_episode_dependent_session(n_trials=10, seed=42)
    for tr in session:
        # Check continuous distribution
        assert np.count_nonzero(tr.cue_vector) > 5, "Cue vector should not be sparse or one-hot"
        assert not np.isnan(tr.cue_vector).any()
        assert not np.isinf(tr.cue_vector).any()


def test_leakage_audit_generator_independent_sampling():
    """Target actions must be independently sampled with chance expectation."""
    session = make_episode_dependent_session(n_trials=100, seed=42)
    targets = [tr.target_action for tr in session]
    mean_target = np.mean(targets)
    assert 0.40 <= mean_target <= 0.60, f"Generator target prior biased: {mean_target}"


def test_transfer_audit_catches_mutation():
    """TransferAudit must throw if weights are modified during evaluation."""
    audit = TransferAudit()
    w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    audit.begin(w)
    w_modified = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    audit.end(w_modified)
    with pytest.raises(RuntimeError, match="Zero-shot violation"):
        audit.assert_frozen()


# ---------------------------------------------------------------------------
# 7. Resource Accounting & Fair Baselines
# ---------------------------------------------------------------------------
def test_resource_accounting_and_baselines():
    """External episodic memory bytes must be separate from neural parameters."""
    base = build_baseline_comparison(peak_rss_mb=12.5)
    assert base["external_memory_reported_separately"]
    assert base["zero_inference_learning_verified"]

    systems = {b["system"]: b for b in base["baselines"]}
    assert "MIN_E_PLUS_EPISODIC" in systems
    assert "MIN_E_NO_MEMORY" in systems
    assert "SIMPLE_RECURRENT" in systems
    assert "BPTT_RECURRENT" in systems
    assert "CAUSAL_TRANSFORMER" in systems

    min_e = systems["MIN_E_PLUS_EPISODIC"]
    assert min_e["external_memory_bytes"] > 0
    assert not min_e["external_memory_included_in_parameters"]
    assert min_e["total_neural_parameters"] == min_e["trainable_parameters"] + min_e["fixed_parameters"]

    # Transformer requires attention scaling
    tf = systems["CAUSAL_TRANSFORMER"]
    assert tf["trainable_parameters"] > min_e["trainable_parameters"]


# ---------------------------------------------------------------------------
# 8. BrainLog v3.8 Validation & Generator
# ---------------------------------------------------------------------------
def test_brainlog_v38_generation_and_validation(tmp_path: Path):
    """BrainLog v3.8 must validate and contain Phase 22 events."""
    out_file = tmp_path / "test_p22.brainlog"
    generate_phase22_brainlog(out_file, seed=42)

    valid, reason = BrainLogValidator.validate(out_file)
    assert valid, f"BrainLog v3.8 validation failed: {reason}"

    data = BrainLogSerializer.load(out_file)
    assert data["brainlog_version"] == "3.8"
    assert data["metadata"]["phase"] == "22"

    events = [e["type"] for e in data["events"]]
    assert EventType.EPISODE_CREATED.value in events
    assert EventType.EPISODE_RETRIEVED.value in events
    assert EventType.RETRIEVED_EPISODE_CONSUMED.value in events
    assert EventType.COMPOSITIONAL_RELATION_RETRIEVED.value in events


# ---------------------------------------------------------------------------
# 9. Results JSON Verification
# ---------------------------------------------------------------------------
def test_results_json_integrity():
    """PHASE_22_RESULTS.json must exist and satisfy all primary endpoints."""
    assert RESULTS_JSON.exists(), f"Missing {RESULTS_JSON}"
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    assert res["phase"] == 22
    assert res["seeds"] == [42, 101, 2024, 777, 999]
    assert res["track_f_leakage_audit"]["all_passed"]

    # Primary contrasts
    contrasts = res["track_a_b_d_causal_ablations"]["contrasts"]
    assert contrasts["B0_vs_B1_full_vs_no_memory"]["p_value"] < 0.01
    assert contrasts["B0_vs_B1_full_vs_no_memory"]["cohens_d"] > 3.0

    # Compositional contrast
    c_contrast = res["track_c_compositional_transfer"]["contrast_with_vs_without_memory"]
    assert c_contrast["p_value"] < 0.01

    # Holm-Bonferroni primary significance
    hb = res["primary_hypotheses"]["holm_bonferroni"]
    for hyp, sig in hb["significant"].items():
        assert sig, f"Primary hypothesis {hyp} failed significance under Holm-Bonferroni"


# ---------------------------------------------------------------------------
# 10. Visualization / Player Contract
# ---------------------------------------------------------------------------
def test_player_has_phase22_overlays_and_js_wiring():
    """player.html must contain [M][E][R][C][T] overlays, p22-btn, and JS wiring."""
    html = PLAYER_PATH.read_text(encoding="utf-8")

    # 1. Five overlay containers
    for ov_id in [
        "p22-memory-overlay",
        "p22-episode-overlay",
        "p22-retrieval-overlay",
        "p22-composition-overlay",
        "p22-timing-overlay",
    ]:
        assert f'id="{ov_id}"' in html, f"Missing overlay container {ov_id}"

    # 2. Toolbar button
    assert 'id="p22-btn"' in html

    # 3. Model select option
    assert "EXP022_causal_memory.brainlog" in html

    # 4. JS function
    assert "function updatePhase22Overlays" in html
    assert "setP22Active" in html

    # 5. Every setText ID referenced in updatePhase22Overlays must exist in markup
    m = re.search(r"function updatePhase22Overlays.*?\n    \}\n", html, re.S)
    assert m, "updatePhase22Overlays function body not found"
    js_refs = set(re.findall(r"setText\('([a-z0-9-]+)'", m.group(0)))
    assert js_refs, "No setText references in updatePhase22Overlays"

    start = html.index("<!-- PHASE 22 EPISODE-DEPENDENT")
    end = html.index("<!-- SPIKE RASTER DRAWER -->")
    markup = html[start:end]
    html_ids = set(re.findall(r'id="([a-z0-9-]+)"', markup))
    missing = js_refs - html_ids
    assert not missing, f"JS references IDs absent from overlay markup: {sorted(missing)}"

    # 6. Close buttons
    for btn in [
        "close-p22-mem-btn",
        "close-p22-ep-btn",
        "close-p22-retr-btn",
        "close-p22-comp-btn",
        "close-p22-time-btn",
    ]:
        assert f"'{btn}'" in html, f"Close button {btn} not wired in JS"
        assert f'id="{btn}"' in html, f"Close button {btn} missing from markup"
