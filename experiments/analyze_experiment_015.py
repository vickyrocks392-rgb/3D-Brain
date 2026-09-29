"""Experiment 015 Analysis Suite: Statistical Diagnostics, Credit Localization & Bottleneck Diagnosis."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


def analyze_experiment_015(results_path: Optional[Path] = None) -> Dict[str, Any]:
    """Parse and generate quantitative diagnostics from EXP_015_RESULTS.json."""
    if results_path is None:
        results_path = Path(__file__).resolve().parent / "results" / "EXP_015_RESULTS.json"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    primary = data.get("primary_conditions", {})
    offline = data.get("offline_readout_controls", {})
    bottleneck = data.get("bottleneck_decision_tree", {})
    causal = data.get("causal_action_interventions", {})
    info_audit = data.get("information_measurement_audit", {})

    print("=" * 70)
    print("EXPERIMENT 015 ANALYSIS REPORT")
    print("=" * 70)

    # 1. Primary Conditions Summary
    print("\n--- 1. Primary Conditions Performance & Credit Localization ---")
    print(f"{'Condition':<40} | {'Mean Acc':<9} | {'CSI':<7} | {'Leakage':<8} | {'Precision':<9} | {'Margin':<7}")
    print("-" * 88)
    for name, stats in primary.items():
        print(
            f"{name:<40} | {stats['mean_accuracy']*100:>7.1f}% | "
            f"{stats['credit_specificity_index']:>7.3f} | {stats['credit_leakage']:>8.3f} | "
            f"{stats['credit_precision']:>9.3f} | {stats['mean_margin']:>7.3f}"
        )

    # 2. Offline vs Online Comparison
    print("\n--- 2. Representation vs Readout vs Autonomous Behavior ---")
    offline_linear = offline.get("O0_LINEAR_CLASSIFIER", 0.0)
    offline_centroid = offline.get("O1_NONLINEAR_CENTROID", 0.0)
    online_baseline = primary.get("COND_A_PHASE14_BASELINE", {}).get("mean_accuracy", 0.0)
    online_integrated = primary.get("COND_N_INTEGRATED_CANDIDATE", {}).get("mean_accuracy", 0.0)
    bptt = bottleneck.get("CASE_D_BPTT_REFERENCE", 0.0)
    oracle = bottleneck.get("CASE_E_CONTINUOUS_ORACLE", 0.0)

    print(f"Continuous Subthreshold Oracle (Ceiling): {oracle*100:5.1f}%")
    print(f"Offline Linear Decodability (Ridge):     {offline_linear*100:5.1f}%")
    print(f"Offline Nonlinear Decodability:          {offline_centroid*100:5.1f}%")
    print(f"Analytical BPTT Reference:               {bptt*100:5.1f}%")
    print(f"Phase 14 Online Baseline:                {online_baseline*100:5.1f}%")
    print(f"Phase 15 Integrated Online Candidate:    {online_integrated*100:5.1f}%")

    # 3. Credit Localization Gain
    base_csi = primary.get("COND_A_PHASE14_BASELINE", {}).get("credit_specificity_index", 0.5)
    int_csi = primary.get("COND_N_INTEGRATED_CANDIDATE", {}).get("credit_specificity_index", 0.5)
    base_leak = primary.get("COND_A_PHASE14_BASELINE", {}).get("credit_leakage", 0.5)
    int_leak = primary.get("COND_N_INTEGRATED_CANDIDATE", {}).get("credit_leakage", 0.5)
    print(f"\nCredit Specificity Index (CSI): {base_csi:5.3f} -> {int_csi:5.3f} (+{int_csi - base_csi:5.3f})")
    print(f"Credit Leakage Reduction:      {base_leak:5.3f} -> {int_leak:5.3f} ({int_leak - base_leak:5.3f})")

    # 4. Critical Causal Interventions
    print("\n--- 3. Critical Causal Action Interventions ---")
    intact_acc = causal.get("INTACT", {}).get("accuracy", 0.0)
    print(f"Intact Accuracy: {intact_acc*100:5.1f}%")
    for k, v in causal.items():
        if k == "INTACT":
            continue
        print(f"  {k:<35} | Acc: {v['accuracy']*100:5.1f}% | Impact: {v.get('delta_acc', 0.0):+5.1%}")

    analysis = {
        "offline_linear": offline_linear,
        "online_baseline": online_baseline,
        "online_integrated": online_integrated,
        "csi_gain": int_csi - base_csi,
        "leakage_reduction": base_leak - int_leak,
    }
    return analysis


if __name__ == "__main__":
    analyze_experiment_015()
