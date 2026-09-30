"""Unit tests for Phase 21: episodic memory, replay consolidation & task transfer.

Covers the five benchmark tracks:
  Track A - episodic memory & episode discrimination
  Track B - replay consolidation (forgeting sequence) & content audit
  Track C - compositional zero-shot transfer & retrieval dependence
  Track D - integrated architecture sweep (D-conditions)
  Track E - fair baseline comparison & compute accounting
plus the statistics helpers, the master results file, the BrainLog v3.7
generator, and the player overlay wiring.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from brain.episodic_memory import EpisodicCondition
from benchmarks.phase21_episodic import (
    evaluate_episode_discrimination,
    evaluate_memory_ablations,
)
from benchmarks.phase21_consolidation import (
    evaluate_forgetting_sequence,
    evaluate_replay_content_audit,
    evaluate_transfer_without_replay,
)
from benchmarks.phase21_transfer import (
    Combination,
    ComponentRule,
    evaluate_compositional_transfer,
    evaluate_transfer_retrieval_dependence,
    train_components,
    evaluate_component_rules,
)
from benchmarks.phase21_integrated import D_CONDITIONS, evaluate_integrated_condition
from benchmarks.phase21_baselines import (
    BASELINE_SPECS,
    analytical_flops_per_step,
    build_baseline_comparison,
)
from experiments.run_phase21_experiments import (
    cohens_d,
    holm_bonferroni,
    paired_t_pvalue,
    summarize,
    _betainc_reg,
    _t_two_sided_pvalue,
)
from brain.replay_consolidation import (
    ReplayConsolidationCondition,
    ReplayContent,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "experiments" / "results" / "PHASE_21_RESULTS.json"
PLAYER_PATH = REPO_ROOT / "visualization" / "player.html"


# ---------------------------------------------------------------------------
# Track A: episodic memory & episode discrimination
# ---------------------------------------------------------------------------
def test_sparse_content_retrieves_no_memory_does_not():
    """P1: A3 (sparse content-addressed) must retrieve; A0 must retrieve nothing."""
    a3 = evaluate_episode_discrimination(EpisodicCondition.A3_SPARSE_CONTENT, n_episodes=60, seed=42)
    a0 = evaluate_episode_discrimination(EpisodicCondition.A0_NO_MEMORY, n_episodes=60, seed=42)

    assert a3["retrieval_accuracy"] >= 0.9
    assert a0["retrieval_accuracy"] == 0.0
    assert a3["memory_bytes"] > 0
    # A0 never retrieves: latency/sparsity stats stay empty (records may
    # still exist, but no retrieval path is taken).
    assert a0["retrieval_latency_us"] == 0.0
    assert a0["retrieval_sparsity"] == 0.0
    # Contextual confusion: retrieval must be driven by order, not the
    # (identical) initial observation.
    assert a3["contextual_confusion"] < 0.2
    # Sparsity reported and below half the dictionary.
    assert 0.0 <= a3["retrieval_sparsity"] < 0.5


def test_memory_ablations_collapse_retrieval():
    """Causal ablations: removing or corrupting memory destroys retrieval."""
    abl = evaluate_memory_ablations(n_episodes=60, seed=42)

    assert abl["NONE"]["retrieval_accuracy"] >= 0.9
    assert abl["REMOVE_MEMORY"]["retrieval_accuracy"] == 0.0
    # Randomized retrieval drives accuracy to (near) chance with maximal
    # confusion about which episode was actually retrieved.
    assert abl["RANDOMIZE_RETRIEVAL"]["retrieval_accuracy"] <= 0.1
    assert abl["RANDOMIZE_RETRIEVAL"]["contextual_confusion"] > 0.9


def test_memory_accounting_is_explicit():
    """Parameters, memory bytes, and records are reported separately."""
    r = evaluate_episode_discrimination(EpisodicCondition.A3_SPARSE_CONTENT, n_episodes=40, seed=42)

    assert r["memory_bytes"] > 0
    assert r["dictionary_parameters"] > 0
    assert r["records_stored"] == 40
    assert r["condition"] == "A3_SPARSE_CONTENT"


# ---------------------------------------------------------------------------
# Track B: replay consolidation & content audit
# ---------------------------------------------------------------------------
def test_interference_protocol_b0_forgets_b6_retains():
    """Primary Track B endpoint: disjoint-prototype interference under B0,
    retention under B6 (the pre-registered per-seed expectation for seed 42)."""
    r0 = evaluate_forgetting_sequence(
        ReplayConsolidationCondition.B0_NO_REPLAY, seed=42, epochs=30)
    r6 = evaluate_forgetting_sequence(
        ReplayConsolidationCondition.B6_BOTH, seed=42, epochs=30)

    # Task A is actually learned before interference in both conditions.
    assert r0["acc_a_pre"] >= 0.8
    assert r6["acc_a_pre"] >= 0.8

    # No replay happens under B0; replay events occur under B6.
    assert r0["replay_events"] == 0
    assert r6["replay_events"] > 0

    # Forgetting is real under B0 and absent (or smaller) under B6 for this seed.
    assert r0["forgetting_abs"] > 0.0
    assert r6["forgetting_abs"] <= r0["forgetting_abs"]
    assert r6["retention_ratio"] >= r0["retention_ratio"]

    # Memory accounting reported.
    assert r0["memory_bytes"] == 0
    assert r6["memory_bytes"] > 0

    # The slow pool changed or captures ran when consolidation was enabled.
    assert r6["slow_pool_norm_post"] != r6["slow_pool_norm_pre"] or r6["captures"] > 0


def test_replay_content_audit_covers_all_seven_protocols():
    """All preregistered R1-R7 content protocols are evaluated under B6."""
    audit = evaluate_replay_content_audit(seed=42, epochs=10)

    expected = {c.value for c in ReplayContent}
    assert set(audit.keys()) == expected
    for content_key, row in audit.items():
        assert row["condition"] == ReplayConsolidationCondition.B6_BOTH.value
        assert row["content"] == content_key
        assert 0.0 <= row["acc_a_post"] <= 1.0
        assert row["replay_events"] > 0
        assert row["captures"] > 0


def test_transfer_without_replay_has_four_protocols():
    """Spec 6.3: four pre-registered protocols isolating replay's contribution."""
    out = evaluate_transfer_without_replay(seed=42, dim=40)

    assert set(out.keys()) == {
        "learn_test_no_replay",
        "learn_rest_replay_test",
        "learnA_replayA_learnB_testA",
        "learnA_noreplayA_learnB_testA",
    }
    for row in out.values():
        assert set(row.keys()) == {"acc_a_post", "forgetting_abs"}
        assert 0.0 <= row["acc_a_post"] <= 1.0
        assert row["forgetting_abs"] >= 0.0


# ---------------------------------------------------------------------------
# Track C: compositional zero-shot transfer
# ---------------------------------------------------------------------------
def test_component_rules_learned_in_isolation():
    """Each component rule is learnable on its own to (near) ceiling."""
    ro = train_components(seed=42, epochs=60)
    acc = evaluate_component_rules(ro, seed=42)

    assert set(acc.keys()) == {r.value for r in ComponentRule}
    for rule, a in acc.items():
        assert a >= 0.95, f"rule {rule} not learned: {a}"


def test_zero_shot_transfer_is_frozen_and_above_chance():
    """P5 + frozen-evaluation audit: zero-shot composition above chance with
    a strictly frozen model, and a shuffled-label control at chance."""
    r = evaluate_compositional_transfer(seed=42)

    audit = r["zero_shot_audit"]
    assert audit["zero_shot_valid"] is True
    assert audit["weights_unchanged"] is True
    assert audit["plasticity_calls"] == 0
    assert audit["replay_calls"] == 0
    assert audit["target_access"] == 0

    assert r["chance"] == 0.5
    assert r["mean_combination_accuracy"] > 0.9
    assert r["shuffled_label_accuracy"] <= 0.65
    assert r["linear_control_probe"] >= r["chance"]

    # Per-combination detail includes every preregistered combination.
    assert set(r["combinations"].keys()) == {c.value for c in Combination}
    for combo, row in r["combinations"].items():
        assert row["chance"] == 0.5
        assert row["n_trials"] == 80

    # External memory is accounted separately from parameters.
    assert r["parameters"] > 0
    assert r["episodic_memory_bytes"] > 0
    assert r["episodic_memory_records"] > 0


def test_transfer_does_not_depend_on_retrieval():
    """H-C2 honest null: ablating memory does not change zero-shot transfer."""
    dep = evaluate_transfer_retrieval_dependence(seed=42)

    assert set(dep.keys()) == {
        "with_memory", "memory_removed", "retrieval_randomized", "no_memory"}
    for row in dep.values():
        assert 0.0 <= row["mean_combination_accuracy"] <= 1.0

    wm = dep["with_memory"]["mean_combination_accuracy"]
    for tag in ("memory_removed", "retrieval_randomized", "no_memory"):
        assert abs(dep[tag]["mean_combination_accuracy"] - wm) <= 0.1, (
            f"{tag} deviates from with_memory beyond the pre-registered null band")


# ---------------------------------------------------------------------------
# Track D: integrated sweep
# ---------------------------------------------------------------------------
def test_d_condition_table_is_complete_and_well_formed():
    """Six D conditions covering the full factorial of optional mechanisms."""
    assert set(D_CONDITIONS.keys()) == {
        "D0_PHASE20_BASELINE",
        "D1_EPISODIC_MEMORY",
        "D2_REPLAY_CONSOLIDATION",
        "D3_MEMORY_REPLAY",
        "D4_MEMORY_REPLAY_CONSOLIDATION",
        "D5_FULL_ROUTING",
    }
    for cond_id, flags in D_CONDITIONS.items():
        assert set(flags.keys()) == {"memory", "replay", "consolidation", "routing"}
        assert all(isinstance(v, bool) for v in flags.values()), cond_id
        # Routing is only meaningful with the full stack enabled.
        if flags["routing"]:
            assert flags["memory"] and flags["replay"] and flags["consolidation"]

    assert D_CONDITIONS["D0_PHASE20_BASELINE"] == {
        "memory": False, "replay": False, "consolidation": False, "routing": False}
    assert D_CONDITIONS["D4_MEMORY_REPLAY_CONSOLIDATION"] == {
        "memory": True, "replay": True, "consolidation": True, "routing": False}


def test_integrated_d4_not_worse_than_d0():
    """P7 endpoint at the registered seed: D4 retention >= D0 retention."""
    d0 = evaluate_integrated_condition("D0_PHASE20_BASELINE", seed=42, epochs=30)
    d4 = evaluate_integrated_condition("D4_MEMORY_REPLAY_CONSOLIDATION", seed=42, epochs=30)

    assert d0["acc_a_pre"] >= 0.8
    assert d4["acc_a_pre"] >= 0.8
    assert d4["acc_a_post"] >= d0["acc_a_post"]
    assert d4["forgetting_abs"] <= d0["forgetting_abs"]
    assert d4["config"]["memory"] and d4["config"]["replay"] and d4["config"]["consolidation"]
    assert not d0["config"]["memory"] and not d0["config"]["replay"]


def test_accounting_never_folds_external_memory_into_parameters():
    """Every accounting row must state that external memory is NOT parameters."""
    d0 = evaluate_integrated_condition("D0_PHASE20_BASELINE", seed=42, epochs=5)
    d1 = evaluate_integrated_condition("D1_EPISODIC_MEMORY", seed=42, epochs=5)

    for row in (d0, d1):
        acc = row["accounting"]
        assert acc["external_memory_included_in_parameters"] is False
        assert acc["neural_parameters_trainable"] > 0
        assert acc["neural_parameters_fixed"] > 0
        assert acc["computational_workspace_peak_bytes"] > 0

    # Memory-enabled condition reports episodic bytes; disabled reports none.
    assert d1["accounting"]["episodic_memory_bytes"] > 0
    assert d1["accounting"]["total_external_memory_bytes"] >= d1["accounting"]["episodic_memory_bytes"]
    assert d0["accounting"]["episodic_memory_bytes"] == 0


# ---------------------------------------------------------------------------
# Track E: fair baselines
# ---------------------------------------------------------------------------
def test_analytical_flops_accounting_for_all_systems():
    for spec in BASELINE_SPECS:
        flops = analytical_flops_per_step(spec["system"])
        assert flops["total"] > 0
        assert flops["total"] >= flops["feedforward"]

    phase21 = analytical_flops_per_step("MIN_E_PLUS_PHASE21")
    # Phase 21 system pays explicitly for retrieval and replay.
    assert phase21["retrieval"] > 0
    assert phase21["replay"] > 0
    assert phase21["total"] == phase21["feedforward"] + phase21["retrieval"] + phase21["replay"]

    with pytest.raises(ValueError):
        analytical_flops_per_step("NOT_A_SYSTEM")


def test_baseline_comparison_keeps_memory_separate():
    cmp = build_baseline_comparison(peak_rss_mb=123.4)

    assert cmp["memory_accounting_is_separate"] is True
    assert "No claim is made" in cmp["note"]
    assert len(cmp["baselines"]) == len(BASELINE_SPECS)

    for row in cmp["baselines"]:
        assert row["external_memory_included_in_parameters"] is False
        assert row["total_parameters"] == row["trainable_parameters"] + row["fixed_parameters"]
        assert row["flops_per_trial"] == row["flops_per_step"] * row["sequence_length"]
        assert (
            row["total_training_flops"]
            == row["flops_per_step"] * row["optimization_steps"] * row["sequence_length"]
        )
        assert row["peak_rss_mb"] == 123.4

    phase21_row = next(r for r in cmp["baselines"] if r["system"] == "MIN_E_PLUS_PHASE21")
    # External memory exists but is NOT inside any parameter count.
    assert phase21_row["external_memory_bytes"] > 0


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def test_summarize_reports_mean_sd_ci_and_n():
    s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])

    assert s["n"] == 5
    assert s["mean"] == 3.0
    assert s["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1), abs=1e-6)
    lo, hi = s["ci95"]
    assert lo < s["mean"] < hi
    assert s["values"] == [1.0, 2.0, 3.0, 4.0, 5.0]

    assert summarize([])["n"] == 0
    assert summarize([])["mean"] is None


def test_cohens_d_properties():
    assert cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    # Paired d over differences: constant difference has zero sd -> 0.0.
    assert cohens_d([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) == 0.0
    # Variable differences with positive mean -> positive d.
    assert cohens_d([2.0, 3.5, 4.0, 6.0], [1.0, 2.0, 3.0, 4.0]) > 0.0
    assert cohens_d([1.0, 2.0, 3.0, 4.0], [2.0, 3.5, 4.0, 6.0]) < 0.0
    with pytest.raises(ValueError):
        cohens_d([1.0, 2.0], [1.0])


def test_t_pvalue_matches_reference_table():
    """Validate the incomplete-beta t CDF against published t-table values."""
    # Two-sided 5% critical values: t(0.975, df).
    assert _t_two_sided_pvalue(2.776, 4) == pytest.approx(0.05, abs=1e-3)
    assert _t_two_sided_pvalue(12.706, 1) == pytest.approx(0.05, abs=1e-3)
    # Two-sided 1% critical value: t(0.995, 4) = 4.604.
    assert _t_two_sided_pvalue(4.604, 4) == pytest.approx(0.01, abs=1e-3)
    # Monotonicity and boundary behaviour.
    assert _t_two_sided_pvalue(0.0, 4) == 1.0
    assert _t_two_sided_pvalue(1.0, 4) > _t_two_sided_pvalue(2.0, 4)
    assert _t_two_sided_pvalue(np.inf, 4) == 0.0


def test_paired_t_pvalue_behaviour():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert paired_t_pvalue(a, a) == 1.0                            # no difference
    # Consistent positive differences (small variance) -> significant.
    assert paired_t_pvalue(a, [0.0, 1.0, 2.0, 3.0, 3.0]) < 0.1
    assert paired_t_pvalue([1.0], [0.0]) == 1.0                    # n < 2 guard
    # Constant non-zero difference -> zero variance -> degenerate but defined.
    assert paired_t_pvalue([1.0, 2.0], [0.0, 1.0]) == 0.0


def test_incomplete_beta_reference_values():
    assert _betainc_reg(1, 1, 0.5) == pytest.approx(0.5, abs=1e-12)
    assert _betainc_reg(2, 1, 0.5) == pytest.approx(0.25, abs=1e-12)
    assert _betainc_reg(1, 2, 0.5) == pytest.approx(0.75, abs=1e-12)
    # I_x(1, b) = 1 - (1 - x)^b
    assert _betainc_reg(1, 3, 0.3) == pytest.approx(1 - 0.7 ** 3, abs=1e-12)
    assert _betainc_reg(2, 3, 0.0) == 0.0
    assert _betainc_reg(2, 3, 1.0) == 1.0


def test_holm_bonferroni_known_example():
    hb = holm_bonferroni({"A": 0.01, "B": 0.03, "C": 0.04}, alpha=0.05)

    # Sorted: A(0.01)->3*0.01=0.03, B(0.03)->max(0.03, 2*0.03=0.06)=0.06,
    # C(0.04)->max(0.06, 1*0.04=0.04)=0.06.
    assert hb["corrected"]["A"] == pytest.approx(0.03, abs=1e-6)
    assert hb["corrected"]["B"] == pytest.approx(0.06, abs=1e-6)
    assert hb["corrected"]["C"] == pytest.approx(0.06, abs=1e-6)
    assert hb["significant"]["A"] is True
    assert hb["significant"]["B"] is False
    assert hb["significant"]["C"] is False
    assert hb["alpha"] == 0.05
    # Adjusted p-values are monotone in sorted order and capped at 1.
    assert hb["corrected"]["A"] <= hb["corrected"]["B"] <= hb["corrected"]["C"]
    assert holm_bonferroni({"X": 1.5})["corrected"]["X"] <= 1.0


# ---------------------------------------------------------------------------
# Master results file (committed artifact)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="PHASE_21_RESULTS.json not generated")
def test_primary_endpoints_and_leakage_audit_in_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        master = json.load(f)

    assert master["phase"] == 21
    assert master["preregistration"] == "PREREG-PHASE21-001"
    assert master["seeds"] == [42, 101, 2024, 777, 999]
    assert master["chance_accuracy"] == 0.5

    p = master["primary_endpoints"]
    for key in (
        "P1_retrieval_accuracy_A3_vs_A0",
        "P2_discrimination_A3_vs_A1",
        "P3_post_rest_retention_B6_vs_B0",
        "P4_exact_vs_shuffled_replay",
        "P5_zero_shot_combination_accuracy",
        "P6_transfer_with_memory_ablated",
        "P7_integrated_retention_D4_vs_D0",
    ):
        assert key in p, f"missing pre-registered endpoint {key}"

    # Holm-Bonferroni verdicts: P1/P2/P5/P7 significant, P3/P4/P6 honest nulls.
    sig = p["holm_bonferroni"]["significant"]
    assert sig["P1"] is True and sig["P2"] is True
    assert sig["P5"] is True and sig["P7"] is True
    assert sig["P3"] is False and sig["P4"] is False and sig["P6"] is False

    # Leakage checklist fully passed.
    audit = master["leakage_audit"]
    assert audit["all_passed"] is True
    assert all(v["status"] == "PASS" for v in audit["checks"].values())


# ---------------------------------------------------------------------------
# BrainLog v3.7 generator
# ---------------------------------------------------------------------------
def test_generated_phase21_brainlog_validates_and_replays(tmp_path):
    from experiments.generate_phase21_brainlog import generate_phase21_brainlog
    from observability.brainlog import BrainLogValidator
    from observability.replay import ReplayEngine

    out = generate_phase21_brainlog(tmp_path / "EXP021_test.brainlog", seed=42)
    assert out.exists()

    valid, reason = BrainLogValidator.validate(out)
    assert valid, reason

    with open(out, encoding="utf-8") as f:
        data = json.load(f)

    assert data["brainlog_version"] == "3.7"
    assert data["metadata"]["phase"] == 21
    assert data["metadata"]["overlays"] == [
        "E_EPISODIC", "R_REPLAY", "C_CONSOLIDATION", "T_TRANSFER"]

    # Activation vector must match the topology node count exactly.
    n_nodes = len(data["topology"]["nodes"])
    for frame in data["frames"]:
        assert len(frame["activations"]) == n_nodes

    # All four telemetry sections appear across the trace.
    sections = set()
    for frame in data["frames"]:
        sections.update(k for k in frame["dynamic_details"] if k != "phase")
    assert {"episodic", "replay", "consolidation", "transfer"} <= sections

    # Distinct phases for each preregistered stage.
    phases = {f["dynamic_details"].get("phase") for f in data["frames"]}
    assert "EPISODE_CREATION" in phases
    assert "QUIESCENT_REST" in phases
    assert "NOVEL_COMBINATION" in phases
    assert "ZERO_SHOT_VERIFICATION" in phases

    # Phase 21 event types present.
    event_types = {e["type"] for e in data["events"]}
    for et in ("EPISODE_CREATED", "EPISODE_CLOSED", "EPISODE_RETRIEVED",
               "REPLAY_SELECTED", "CONSOLIDATION_TAG_SET",
               "COMPONENT_RULE_LEARNED", "NOVEL_COMBINATION_EVALUATED",
               "ZERO_SHOT_VERIFIED", "MEMORY_ACCOUNTING_REPORTED"):
        assert et in event_types, f"missing event type {et}"

    # Replayable: load, seek, step, and inspect node telemetry.
    engine = ReplayEngine()
    engine.load(out)
    assert engine.current_step == 0
    engine.seek(len(data["frames"]) - 1)
    assert engine.current_step == len(data["frames"]) - 1
    engine.seek(0)
    assert engine.step_forward()["step"] == 1
    assert engine.inspect_node(0) is not None



# ---------------------------------------------------------------------------
# Player overlay wiring (static contract)
# ---------------------------------------------------------------------------
def test_player_has_phase21_overlays_and_js_wiring():
    html = PLAYER_PATH.read_text(encoding="utf-8")

    # Four overlay containers.
    for overlay_id in ("episodic-overlay", "p21-replay-overlay",
                       "p21-consolidation-overlay", "p21-transfer-overlay"):
        assert f'id="{overlay_id}"' in html, f"missing overlay {overlay_id}"

    # Toolbar toggle + EXP021 default selection.
    assert 'id="p21-btn"' in html
    assert "EXP021_episodic_transfer.brainlog" in html
    assert "selected>EXP021" in html

    # JS update function and toggle logic present.
    assert "function updatePhase21Overlays" in html
    assert "setP21Active" in html

    # Every setText id referenced by the Phase 21 JS must exist in the
    # overlay markup (guards against typos between HTML and JS).
    m = re.search(r"function updatePhase21Overlays.*?\n    \}\n", html, re.S)
    assert m, "updatePhase21Overlays not found"
    js_refs = set(re.findall(r"setText\('([a-z0-9-]+)'", m.group(0)))
    assert js_refs, "no setText references found in updatePhase21Overlays"

    start = html.index("<!-- PHASE 21 EPISODIC")
    end = html.index("<!-- SPIKE RASTER DRAWER -->")
    markup = html[start:end]
    html_ids = set(re.findall(r'id="([a-z0-9-]+)"', markup))
    missing = js_refs - html_ids
    assert not missing, f"JS references ids absent from overlay markup: {sorted(missing)}"

    # Close buttons wired for all four overlays.
    for btn in ("close-episodic-btn", "close-p21-replay-btn",
                "close-p21-consol-btn", "close-p21-transfer-btn"):
        assert f"'{btn}'" in html, f"close button {btn} not wired in JS"
        assert f'id="{btn}"' in html, f"close button {btn} missing from markup"

    # Metric ids populated by JS cover all four telemetry sections.
    for prefix in ("ep-", "r21-", "c21-", "t21-"):
        assert any(ref.startswith(prefix) for ref in js_refs), f"no JS refs for {prefix}"







