"""Analyze Experiment 010 Results and Generate Summary Tables for Phase 10 Reports."""

import json
from pathlib import Path
from typing import Any, Dict


def analyze_exp010() -> Dict[str, Any]:
    metrics_path = Path("experiments/results/exp010_metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Please run experiments/run_experiment_010.py first.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conditions = data["conditions"]

    print("=" * 85)
    print("PHASE 10 EXPERIMENT 010 — MAIN ARCHITECTURE COMPARISON TABLE")
    print("=" * 85)
    print(f"{'Condition':<36} | {'Accuracy':<10} | {'CSI':<8} | {'Precision':<10} | {'Leakage':<8}")
    print("-" * 85)

    for name, c in conditions.items():
        acc = f"{c['accuracy_mean']*100:.1f}%"
        csi = f"{c['csi_mean']:+.3f}"
        prec = f"{c['precision_mean']*100:.1f}%"
        leak = f"{c['leakage_mean']*100:.1f}%"
        print(f"{name:<36} | {acc:<10} | {csi:<8} | {prec:<10} | {leak:<8}")

    print("=" * 85)

    # Hypothesis evaluations
    p9_base = conditions["COND_01_POINT_SPIKING_BASELINE"]["accuracy_mean"]
    two_comp_target = conditions["COND_09_DENDRITIC_TARGETED_SPATIAL"]["accuracy_mean"]
    two_comp_csi = conditions["COND_09_DENDRITIC_TARGETED_SPATIAL"]["csi_mean"]
    two_comp_prec = conditions["COND_09_DENDRITIC_TARGETED_SPATIAL"]["precision_mean"]

    print("\nHYPOTHESIS EVALUATIONS:")
    # H1
    h1_diff = abs(two_comp_target - p9_base)
    print(f"H1 (Compartmentalized Computation): {'SUPPORTED' if two_comp_target > p9_base else 'NOT SUPPORTED'} (Two-Comp Acc={two_comp_target*100:.1f}% vs Point Acc={p9_base*100:.1f}%)")
    # H2
    mem = data["memory_retention"]
    tau100 = mem.get("tau_100ms", {})
    tau1 = mem.get("no_nmda", {})
    r2_100_tau100 = tau100.get("100", 0.0)
    r2_100_tau1 = tau1.get("100", 0.0)
    print(f"H2 (Slow Conductance Memory): {'SUPPORTED' if r2_100_tau100 > r2_100_tau1 else 'NOT SUPPORTED'} (R2@100={r2_100_tau100:.3f} with NMDA vs {r2_100_tau1:.3f} without)")
    # H3
    print(f"H3 (Targeted Credit Specificity): {'SUPPORTED' if two_comp_csi > 0.20 else 'PARTIALLY SUPPORTED'} (CSI={two_comp_csi:+.3f}, Precision={two_comp_prec*100:.1f}%)")
    # H4
    attractor = data["attractor_persistence"]
    print(f"H4 (Working Memory Attractor): {'SUPPORTED' if attractor['attractor_stable'] else 'PARTIALLY SUPPORTED'} (Drift={attractor['drift_per_step']}, Switches={attractor['spontaneous_switches']})")

    return data


if __name__ == "__main__":
    analyze_exp010()
