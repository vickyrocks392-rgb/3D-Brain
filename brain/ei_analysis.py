"""Quantitative Excitation/Inhibition (E/I) Balance Analysis Engine (Phase 8).

Implements rigorous biophysical current accounting, E/I ratio distributions,
temporal balance tracking correlation, and population stability diagnostics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_ei_balance_metrics(
    i_e_history: np.ndarray,
    i_i_history: np.ndarray,
    eps: float = 1e-5,
) -> Dict[str, Any]:
    """Compute comprehensive E/I balance statistics across time and neuron populations.
    
    Args:
        i_e_history: Array of shape [T, N] containing excitatory input currents (>= 0)
        i_i_history: Array of shape [T, N] containing inhibitory input currents (<= 0)
        eps: Small numerical constant for division stability
        
    Returns:
        Dictionary of population means, medians, variances, percentiles, and tracking correlation.
    """
    T, N = i_e_history.shape
    if T == 0 or N == 0:
        return {
            "mean_i_e": 0.0,
            "mean_i_i": 0.0,
            "mean_i_net": 0.0,
            "mean_ei_ratio": 1.0,
            "median_ei_ratio": 1.0,
            "ei_tracking_corr": 0.0,
            "balance_regime": "QUIESCENT",
        }

    i_e_abs = np.maximum(0.0, i_e_history)
    i_i_abs = np.abs(i_i_history)
    i_net = i_e_abs - i_i_abs
    ei_ratio = i_e_abs / (i_i_abs + eps)

    # 1. Population-wide scalar metrics
    mean_e = float(np.mean(i_e_abs))
    mean_i = float(np.mean(i_i_abs))
    mean_net = float(np.mean(i_net))

    ratio_values = ei_ratio.flatten()
    mean_ratio = float(np.mean(ratio_values))
    median_ratio = float(np.median(ratio_values))
    var_ratio = float(np.var(ratio_values))

    # 2. Percentiles of E/I ratio
    p5 = float(np.percentile(ratio_values, 5))
    p25 = float(np.percentile(ratio_values, 25))
    p50 = float(np.percentile(ratio_values, 50))
    p75 = float(np.percentile(ratio_values, 75))
    p95 = float(np.percentile(ratio_values, 95))

    # 3. Temporal tracking correlation: Corr(I_E(t), |I_I(t)|)
    # A signature of tight E/I balance is that inhibitory currents tightly track excitatory fluctuations.
    pop_e = np.mean(i_e_abs, axis=1)  # [T]
    pop_i = np.mean(i_i_abs, axis=1)  # [T]

    if np.std(pop_e) > 1e-6 and np.std(pop_i) > 1e-6:
        r_matrix = np.corrcoef(pop_e, pop_i)
        tracking_corr = float(r_matrix[0, 1]) if not np.isnan(r_matrix[0, 1]) else 0.0
    else:
        tracking_corr = 0.0

    # 4. Balance regime classification
    if mean_e < 1e-4 and mean_i < 1e-4:
        regime = "QUIESCENT"
    elif tracking_corr >= 0.70 and abs(mean_net) <= 0.5 * (mean_e + mean_i):
        regime = "TIGHTLY_BALANCED"
    elif 0.30 <= tracking_corr < 0.70:
        regime = "MODERATELY_BALANCED"
    elif mean_e > 2.0 * mean_i:
        regime = "EXCITATION_DOMINATED"
    elif mean_i > 2.0 * mean_e:
        regime = "INHIBITION_DOMINATED"
    else:
        regime = "LOOSELY_BALANCED"

    return {
        "mean_i_e": round(mean_e, 4),
        "mean_i_i": round(mean_i, 4),
        "mean_i_net": round(mean_net, 4),
        "mean_ei_ratio": round(mean_ratio, 3),
        "median_ei_ratio": round(median_ratio, 3),
        "var_ei_ratio": round(var_ratio, 4),
        "p5_ei_ratio": round(p5, 3),
        "p25_ei_ratio": round(p25, 3),
        "p50_ei_ratio": round(p50, 3),
        "p75_ei_ratio": round(p75, 3),
        "p95_ei_ratio": round(p95, 3),
        "ei_tracking_corr": round(tracking_corr, 4),
        "balance_regime": regime,
    }
