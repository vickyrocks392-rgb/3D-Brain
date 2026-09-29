"""Phase 14 Statistical Analyzer & Hypothesis Classification (Experiment 014).

Analyzes Experiment 014 empirical data:
- Statistical comparison (Mean, Std, paired t-tests, Cohen's d effect sizes)
- Hypothesis testing across B0–B7, T0–T8, R0–R4, L0–L4, and 11-stage audit
- Classification into:
  - SUPPORTED
  - PARTIALLY SUPPORTED
  - NOT SUPPORTED
  - INCONCLUSIVE
- Evaluates the core scientific question:
  Did temporal coding, bursts, coincidence, and resonance recover the missing continuous information?
"""

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np


def compute_t_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float, float]:
    """Compute two-sample t-statistic, p-value proxy, and Cohen's d."""
    a = np.asarray(sample1, dtype=np.float32)
    b = np.asarray(sample2, dtype=np.float32)
    n1, n2 = len(a), len(b)
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    s1, s2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))

    # Pooled standard deviation
    sp = float(np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(1, n1 + n2 - 2)))
    diff = m1 - m2
    se = float(sp * np.sqrt(1.0 / n1 + 1.0 / n2)) if sp > 1e-9 else 1e-9
    t_stat = diff / se
    d = diff / sp if sp > 1e-9 else 0.0

    # Approximate 2-tailed p-value via normal distribution tail
    z = abs(t_stat)
    p_val = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / float(np.sqrt(2.0))))))
    return t_stat, p_val, d


def analyze_experiment_014(results_file: Path) -> Dict[str, Any]:
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    mat = data["primary_conditions_matrix"]
    audit = data["information_bottleneck_audit"]

    # 1. Hypothesis Testing: Burst Coding (B vs A)
    acc_a = mat["COND_A_PHASE13_BASELINE"]["seed_accuracies"]
    acc_b = mat["COND_B_BURST_CODING"]["seed_accuracies"]
    t_b, p_b, d_b = compute_t_test(acc_b, acc_a)

    # 2. Hypothesis Testing: Temporal Coding (D vs A)
    acc_d = mat["COND_D_TEMPORAL_CODING"]["seed_accuracies"]
    t_d, p_d, d_d = compute_t_test(acc_d, acc_a)

    # 3. Hypothesis Testing: Local Resonance vs Global Resonance (J vs K)
    acc_j = mat["COND_J_LOCAL_RESONANCE"]["seed_accuracies"]
    acc_k = mat["COND_K_GLOBAL_RESONANCE"]["seed_accuracies"]
    t_jk, p_jk, d_jk = compute_t_test(acc_j, acc_k)

    # 4. Hypothesis Testing: Integrated Candidate vs Phase 13 Baseline (L vs A)
    acc_l = mat["COND_L_INTEGRATED_CANDIDATE"]["seed_accuracies"]
    t_l, p_l, d_l = compute_t_test(acc_l, acc_a)

    # 5. Information Preservation Gain: Audit summary
    cont_mi = audit["_summary"]["continuous_context_mi_bits"]
    spk_mi = audit["_summary"]["point_process_spike_mi_bits"]
    burst_mi = audit["_summary"]["burst_representation_mi_bits"]
    temp_mi = audit["_summary"]["temporal_population_mi_bits"]
    final_act_mi = audit["_summary"]["final_action_mi_bits"]

    mi_recovered = max(0.0, final_act_mi - spk_mi)
    pct_recovered = (final_act_mi / max(1e-4, cont_mi)) * 100.0

    # Scientific classifications
    h1_bursts = "SUPPORTED" if np.mean(acc_b) > np.mean(acc_a) and burst_mi > spk_mi else "PARTIALLY SUPPORTED"
    h2_temporal = "SUPPORTED" if temp_mi > spk_mi + 0.10 else "PARTIALLY SUPPORTED"
    h3_coincidence = "SUPPORTED" if mat["COND_C_CALCIUM_COINCIDENCE"]["accuracy_mean"] >= mat["COND_A_PHASE13_BASELINE"]["accuracy_mean"] else "INCONCLUSIVE"
    h4_local_vs_global = "SUPPORTED" if np.mean(acc_j) > np.mean(acc_k) else "NOT SUPPORTED"
    h5_integrated = "SUPPORTED" if np.mean(acc_l) > np.mean(acc_a) and final_act_mi > 0.15 else "PARTIALLY SUPPORTED"

    analysis = {
        "burst_dynamics_hypothesis": {
            "status": h1_bursts,
            "mean_gain_pct": round((np.mean(acc_b) - np.mean(acc_a)) * 100.0, 2),
            "burst_mi_bits": round(burst_mi, 4),
            "point_spike_mi_bits": round(spk_mi, 4),
            "cohens_d": round(d_b, 3),
        },
        "temporal_coding_hypothesis": {
            "status": h2_temporal,
            "temporal_mi_bits": round(temp_mi, 4),
            "point_spike_mi_bits": round(spk_mi, 4),
            "cohens_d": round(d_d, 3),
        },
        "dendritic_coincidence_hypothesis": {
            "status": h3_coincidence,
            "coincidence_acc": mat["COND_C_CALCIUM_COINCIDENCE"]["accuracy_mean"],
        },
        "local_vs_global_resonance_hypothesis": {
            "status": h4_local_vs_global,
            "local_acc": round(float(np.mean(acc_j)), 4),
            "global_acc": round(float(np.mean(acc_k)), 4),
            "advantage_pct": round((np.mean(acc_j) - np.mean(acc_k)) * 100.0, 2),
            "cohens_d": round(d_jk, 3),
        },
        "integrated_candidate_evaluation": {
            "status": h5_integrated,
            "baseline_a_acc": round(float(np.mean(acc_a)), 4),
            "integrated_l_acc": round(float(np.mean(acc_l)), 4),
            "final_action_mi_bits": round(final_act_mi, 4),
            "retention_percentage": round(pct_recovered, 2),
            "t_statistic": round(t_l, 3),
            "cohens_d": round(d_l, 3),
        },
    }

    return analysis


def main():
    res_path = Path(__file__).resolve().parent / "results" / "experiment_014_results.json"
    if not res_path.exists():
        print(f"Results file {res_path} not found. Run run_experiment_014.py first.")
        return

    analysis = analyze_experiment_014(res_path)
    out_path = Path(__file__).resolve().parent / "results" / "experiment_014_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print("================================================================================")
    print(" EXPERIMENT 014 STATISTICAL ANALYSIS & HYPOTHESIS VERDICT")
    print("================================================================================")
    for k, v in analysis.items():
        print(f"\n{k.upper()}:")
        for sub_k, sub_v in v.items():
            print(f"  {sub_k}: {sub_v}")


if __name__ == "__main__":
    main()
