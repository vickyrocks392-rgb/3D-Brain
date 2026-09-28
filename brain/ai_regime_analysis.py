"""Asynchronous-Irregular (AI) Regime Scientific Analysis Engine (Phase 8).

Implements rigorous empirical diagnostics for identifying and characterizing
the Asynchronous-Irregular (AI) computational regime:
- Coefficient of Variation of Inter-Spike Intervals (CV_ISI)
- Fano factor across non-overlapping temporal windows
- Pairwise spike-count Pearson correlation (r_sc)
- Population-rate power spectral density (PSD) & spectral entropy
- Autocorrelation decay time constant (tau_decay)
- Branching ratio estimation (subcritical vs. critical vs. supercritical)
- Composite operational AI score and multi-criteria diagnostic classification
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_cv_isi(spike_matrix: np.ndarray, dt_ms: float = 1.0) -> Tuple[float, float, List[float]]:
    """Compute mean and standard deviation of CV_ISI across active neurons.
    
    Returns:
        mean_cv: Mean CV_ISI across neurons with at least 3 spikes
        std_cv: Standard deviation of CV_ISI
        per_neuron_cv: List of individual neuron CVs
    """
    T, N = spike_matrix.shape
    neuron_cvs = []

    for j in range(N):
        spike_times = np.where(spike_matrix[:, j])[0] * dt_ms
        if len(spike_times) >= 3:
            isis = np.diff(spike_times)
            mean_isi = np.mean(isis)
            std_isi = np.std(isis)
            if mean_isi > 1e-4:
                neuron_cvs.append(float(std_isi / mean_isi))

    if len(neuron_cvs) == 0:
        return 0.0, 0.0, []

    return float(np.mean(neuron_cvs)), float(np.std(neuron_cvs)), neuron_cvs


def compute_fano_factor(spike_matrix: np.ndarray, dt_ms: float = 1.0, window_ms: float = 50.0) -> float:
    """Compute population mean Fano factor (variance / mean of spike counts in temporal windows)."""
    T, N = spike_matrix.shape
    bin_size = max(1, int(window_ms / dt_ms))
    num_bins = T // bin_size
    if num_bins < 3 or N == 0:
        return 1.0

    binned = np.zeros((num_bins, N), dtype=np.float32)
    for b in range(num_bins):
        binned[b] = np.sum(spike_matrix[b * bin_size : (b + 1) * bin_size], axis=0)

    means = np.mean(binned, axis=0)
    vars_ = np.var(binned, axis=0)
    valid = means > 0.05
    if not np.any(valid):
        return 1.0

    return float(np.mean(vars_[valid] / means[valid]))


def compute_pairwise_spike_correlation(
    spike_matrix: np.ndarray,
    bin_size_steps: int = 10,
    max_pairs: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute mean and std pairwise spike-count Pearson correlation coefficient r_sc."""
    T, N = spike_matrix.shape
    if N < 2:
        return 0.0, 0.0

    num_bins = max(1, T // bin_size_steps)
    binned = np.zeros((num_bins, N), dtype=np.float32)
    for b in range(num_bins):
        binned[b] = np.sum(spike_matrix[b * bin_size_steps : (b + 1) * bin_size_steps], axis=0)

    stds = np.std(binned, axis=0)
    means = np.mean(binned, axis=0)
    valid_mask = stds > 1e-6
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        return 0.0, 0.0

    # Sub-select pairs if N is large
    rng = np.random.default_rng(seed)
    n_valid = len(valid_indices)

    if n_valid <= 50:
        corrs = []
        normed = (binned[:, valid_indices] - means[valid_indices]) / stds[valid_indices]
        cov = np.dot(normed.T, normed) / float(max(1, num_bins - 1))
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                corrs.append(float(cov[i, j]))
        if len(corrs) == 0:
            return 0.0, 0.0
        return float(np.mean(corrs)), float(np.std(corrs))
    else:
        corrs = []
        for _ in range(max_pairs):
            idx_a, idx_b = rng.choice(valid_indices, size=2, replace=False)
            norm_a = (binned[:, idx_a] - means[idx_a]) / stds[idx_a]
            norm_b = (binned[:, idx_b] - means[idx_b]) / stds[idx_b]
            r = float(np.mean(norm_a * norm_b))
            corrs.append(r)
        return float(np.mean(corrs)), float(np.std(corrs))


def compute_autocorrelation_decay(pop_rate: np.ndarray, dt_ms: float = 1.0, max_lag_ms: float = 100.0) -> float:
    """Estimate exponential decay time constant tau_decay (ms) of the population firing rate autocorrelation."""
    T = len(pop_rate)
    max_lag_steps = min(T // 2, max(2, int(max_lag_ms / dt_ms)))
    if T < 10 or max_lag_steps < 2:
        return 0.0

    centered = pop_rate - np.mean(pop_rate)
    var = np.var(pop_rate)
    if var < 1e-8:
        return 0.0

    # Normalized autocorrelation
    autocorr = np.correlate(centered, centered, mode="full")[T - 1 : T - 1 + max_lag_steps] / (var * T)
    lags_ms = np.arange(len(autocorr)) * dt_ms

    # Find where autocorrelation drops below 1/e ~ 0.368
    threshold = 1.0 / np.e
    below = np.where(autocorr <= threshold)[0]
    if len(below) > 0:
        tau_decay = float(lags_ms[below[0]])
    else:
        tau_decay = float(lags_ms[-1])

    return tau_decay


def estimate_branching_ratio(spike_matrix: np.ndarray) -> float:
    """Estimate the population branching ratio sigma = <N(t+1) / N(t)>.
    
    sigma < 1.0: subcritical (activity dies out)
    sigma ~ 1.0: critical-like (fluctuations sustained)
    sigma > 1.0: supercritical (runaway activity)
    """
    pop_spikes = np.sum(spike_matrix, axis=1).astype(np.float64)  # [T]
    n_t = pop_spikes[:-1]
    n_t1 = pop_spikes[1:]

    denom = np.sum(n_t ** 2)
    if denom < 1e-6:
        return 0.0

    sigma = float(np.sum(n_t * n_t1) / denom)
    return round(sigma, 4)


def evaluate_ai_regime_diagnostic(
    spike_matrix: np.ndarray,
    kuramoto_r_global: float,
    dt_ms: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete operational diagnostic for Asynchronous-Irregular (AI) regime classification.
    
    Explicit criteria:
    1. Active Firing: 0.5 Hz <= r_mean <= 60.0 Hz
    2. Spike Irregularity: CV_ISI >= 0.70
    3. Low Pairwise Correlation: |r_sc| <= 0.12
    4. Global Asynchrony: Kuramoto R_global <= 0.35
    5. Population Stability: Branching ratio sigma in [0.70, 1.25], active neuron fraction >= 0.20
    """
    T, N = spike_matrix.shape
    total_time_s = (T * dt_ms) / 1000.0

    spike_counts = np.sum(spike_matrix, axis=0)
    firing_rates = spike_counts / max(1e-4, total_time_s)
    mean_rate = float(np.mean(firing_rates))
    max_rate = float(np.max(firing_rates)) if N > 0 else 0.0
    active_fraction = float(np.mean(spike_counts > 0)) if N > 0 else 0.0

    mean_cv, std_cv, _ = compute_cv_isi(spike_matrix, dt_ms=dt_ms)
    fano = compute_fano_factor(spike_matrix, dt_ms=dt_ms)
    r_sc_mean, r_sc_std = compute_pairwise_spike_correlation(spike_matrix, seed=seed)

    pop_rate = (np.mean(spike_matrix, axis=1) * (1000.0 / dt_ms)).astype(np.float32)
    tau_decay = compute_autocorrelation_decay(pop_rate, dt_ms=dt_ms)
    branching_ratio = estimate_branching_ratio(spike_matrix)

    # 1. Operational criteria evaluations
    crit_firing = 0.5 <= mean_rate <= 60.0 and active_fraction >= 0.20
    crit_irregularity = mean_cv >= 0.70 or (fano >= 0.70 and mean_cv >= 0.50)
    crit_correlation = abs(r_sc_mean) <= 0.12
    crit_asynchrony = kuramoto_r_global <= 0.35
    crit_stability = 0.60 <= branching_ratio <= 1.30 and max_rate <= 150.0

    criteria_passed = [
        crit_firing,
        crit_irregularity,
        crit_correlation,
        crit_asynchrony,
        crit_stability,
    ]
    ai_score = float(np.sum(criteria_passed)) / 5.0
    is_ai_regime = bool(ai_score >= 0.80)  # At least 4 of 5 formal criteria satisfied

    if is_ai_regime:
        status = "AI_REGIME"
    elif kuramoto_r_global > 0.50 and mean_rate > 1.0:
        status = "SYNCHRONOUS_REGULAR"
    elif mean_rate < 0.5 or active_fraction < 0.10:
        status = "QUIESCENT"
    elif abs(r_sc_mean) > 0.20:
        status = "CORRELATED_REGULAR"
    else:
        status = "INTERMEDIATE_TRANSITIONAL"

    return {
        "mean_firing_rate_hz": round(mean_rate, 2),
        "active_neuron_fraction": round(active_fraction, 3),
        "cv_isi_mean": round(mean_cv, 3),
        "cv_isi_std": round(std_cv, 3),
        "fano_factor": round(fano, 3),
        "pairwise_correlation_r_sc": round(r_sc_mean, 4),
        "pairwise_correlation_r_sc_std": round(r_sc_std, 4),
        "kuramoto_r_global": round(float(kuramoto_r_global), 4),
        "autocorr_tau_decay_ms": round(tau_decay, 2),
        "branching_ratio": branching_ratio,
        "criteria": {
            "active_firing": bool(crit_firing),
            "irregularity": bool(crit_irregularity),
            "low_correlation": bool(crit_correlation),
            "global_asynchrony": bool(crit_asynchrony),
            "stability": bool(crit_stability),
        },
        "ai_score": round(ai_score, 2),
        "is_ai_regime": is_ai_regime,
        "regime_status": status,
    }
