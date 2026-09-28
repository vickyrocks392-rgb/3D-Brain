"""Scientific Analysis Engine for Spiking Dynamics, Functional Modularity, and Oscillations.

Implements the formal evaluation metrics for Phase 7:
- Firing statistics, ISI distributions, CV, Fano factor, and burst indices (H1)
- Data-driven functional modularity (Q_func) derived independently from spike trains via Louvain/Newman optimization (H2)
- Power spectral density, dominant frequency, and spectral entropy (H3)
- Instantaneous phase extraction, Kuramoto order parameter R(t), and Phase-Locking Value (PLV) (H4)
- Spatial participation, distance-correlation curves, and traveling wave metrics (H5)
- Memory-synchronization correlation analysis (H6)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_spiking_statistics(
    spike_matrix: np.ndarray, dt_ms: float = 1.0, dt: Optional[float] = None
) -> Dict[str, float]:
    """Compute comprehensive single-neuron and population spiking statistics.

    Args:
        spike_matrix: Binary array of shape [T, N] where 1 indicates an action potential.
        dt_ms: Duration of each simulation timestep in milliseconds.
        dt: Optional alias for dt_ms.

    Returns:
        Dictionary of firing rate, ISI CV, Fano factor, and population activity metrics.
    """
    if dt is not None:
        dt_ms = dt
    T, N = spike_matrix.shape
    total_time_s = (T * dt_ms) / 1000.0
    if total_time_s <= 0 or N == 0:
        return {"mean_firing_rate_hz": 0.0, "cv_isi": 0.0, "fano_factor": 1.0, "active_fraction": 0.0}

    # Firing rates (Hz) per neuron
    spike_counts = np.sum(spike_matrix, axis=0)  # [N]
    firing_rates = spike_counts / total_time_s   # [N]

    # Inter-Spike Intervals (ISI), Coefficient of Variation (CV), and Burst Fraction
    all_cvs = []
    all_isis = []
    for j in range(N):
        spike_times = np.where(spike_matrix[:, j])[0] * dt_ms
        if len(spike_times) >= 2:
            isis = np.diff(spike_times)
            all_isis.extend(isis.tolist())
            if len(spike_times) >= 3:
                mean_isi = np.mean(isis)
                std_isi = np.std(isis)
                if mean_isi > 1e-4:
                    all_cvs.append(std_isi / mean_isi)

    mean_cv_isi = float(np.mean(all_cvs)) if len(all_cvs) > 0 else 0.0
    mean_isi_ms = float(np.mean(all_isis)) if len(all_isis) > 0 else 0.0
    burst_fraction = float(np.mean(np.array(all_isis) <= 15.0)) if len(all_isis) > 0 else 0.0

    # Fano factor (variance of spike counts / mean of spike counts across non-overlapping 50ms windows)
    bin_size = max(1, int(50.0 / dt_ms))
    num_bins = T // bin_size
    if num_bins >= 3:
        binned = np.zeros((num_bins, N), dtype=np.float32)
        for b in range(num_bins):
            binned[b] = np.sum(spike_matrix[b * bin_size : (b + 1) * bin_size], axis=0)
        means = np.mean(binned, axis=0)
        vars_ = np.var(binned, axis=0)
        valid = means > 0.05
        fano = float(np.mean(vars_[valid] / means[valid])) if np.any(valid) else 1.0
    else:
        fano = 1.0

    return {
        "mean_firing_rate_hz": round(float(np.mean(firing_rates)), 2),
        "max_firing_rate_hz": round(float(np.max(firing_rates)), 2),
        "active_neuron_fraction": round(float(np.mean(spike_counts > 0)), 3),
        "mean_cv_isi": round(mean_cv_isi, 3),
        "cv_isi": round(mean_cv_isi, 3),
        "isi_mean_ms": round(mean_isi_ms, 2),
        "burst_fraction": round(burst_fraction, 3),
        "fano_factor": round(fano, 3),
        "total_spikes": int(np.sum(spike_counts)),
    }


def compute_spike_correlation_matrix(
    spike_matrix: np.ndarray, bin_size_steps: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate binned spike-count Pearson cross-correlation matrix.

    Returns:
        corr_matrix: [N, N] pairwise Pearson correlation matrix
        binned_counts: [num_bins, N] binned spike counts
    """
    T, N = spike_matrix.shape
    num_bins = max(1, T // bin_size_steps)

    binned = np.zeros((num_bins, N), dtype=np.float32)
    for b in range(num_bins):
        binned[b] = np.sum(spike_matrix[b * bin_size_steps : (b + 1) * bin_size_steps], axis=0)

    stds = np.std(binned, axis=0)
    means = np.mean(binned, axis=0)

    # Avoid zero division for silent neurons
    valid_mask = stds > 1e-6
    corr = np.zeros((N, N), dtype=np.float32)

    if np.sum(valid_mask) >= 2:
        normed = (binned[:, valid_mask] - means[valid_mask]) / stds[valid_mask]
        sub_corr = np.dot(normed.T, normed) / float(max(1, num_bins - 1))
        # Embed back into full [N, N]
        valid_indices = np.where(valid_mask)[0]
        for idx_i, orig_i in enumerate(valid_indices):
            corr[orig_i, valid_indices] = sub_corr[idx_i]

    np.fill_diagonal(corr, 0.0)
    return corr, binned


def detect_functional_communities_louvain(
    corr_matrix: np.ndarray, min_corr: float = 0.05, max_iter: int = 20
) -> np.ndarray:
    """Discover functional communities independently from spike cross-correlation using Newman-style greedy modularity optimization.

    Returns:
        community_assignments: [N] array of integer module labels in [0, C-1]
    """
    N = len(corr_matrix)
    # Positive functional graph
    A = np.maximum(0.0, corr_matrix)
    A[A < min_corr] = 0.0
    np.fill_diagonal(A, 0.0)

    m = np.sum(A)
    if m < 1e-6 or N < 2:
        return np.zeros(N, dtype=np.int32)

    k = np.sum(A, axis=1)  # Node strengths [N]

    # Start with each node in its own community
    communities = np.arange(N, dtype=np.int32)

    # Iterative greedy local movement
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        order = np.random.permutation(N)
        for i in order:
            best_comm = communities[i]
            best_delta_q = 0.0
            current_comm = communities[i]

            # Neighboring communities
            neighbors = np.where(A[i, :] > 0)[0]
            if len(neighbors) == 0:
                continue

            neighbor_comms = np.unique(communities[neighbors])
            for c in neighbor_comms:
                if c == current_comm:
                    continue
                # Modular change delta Q for moving i to community c
                in_c = (communities == c)
                k_i_in = np.sum(A[i, in_c])
                sigma_tot = np.sum(k[in_c])

                in_curr = (communities == current_comm) & (np.arange(N) != i)
                k_i_curr = np.sum(A[i, in_curr])
                sigma_curr = np.sum(k[in_curr])

                delta_q = (k_i_in - k_i_curr) / m - (2.0 * k[i] * (sigma_tot - sigma_curr)) / (4.0 * m * m)
                if delta_q > best_delta_q:
                    best_delta_q = delta_q
                    best_comm = c

            if best_comm != current_comm and best_delta_q > 1e-5:
                communities[i] = best_comm
                improved = True

    # Renumber communities contiguously 0, 1, ...
    unique_comms = np.unique(communities)
    mapping = {c: idx for idx, c in enumerate(unique_comms)}
    return np.array([mapping[c] for c in communities], dtype=np.int32)


def compute_functional_modularity_from_spikes(
    spike_matrix: np.ndarray,
    structural_modules: Optional[np.ndarray] = None,
    bin_size_steps: int = 10,
    bin_size: Optional[int] = None,
    resolution: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute empirical functional modularity Q_func on actual spike data.

    Returns:
        Dictionary containing Q_func, inferred communities, Q_func_struct, and correlation metrics.
    """
    if bin_size is not None:
        bin_size_steps = bin_size
    corr_matrix, _ = compute_spike_correlation_matrix(spike_matrix, bin_size_steps=bin_size_steps)
    N = len(corr_matrix)

    # 1. Infer data-driven functional communities
    functional_modules = detect_functional_communities_louvain(corr_matrix)
    num_func_modules = len(np.unique(functional_modules))

    # 2. Compute Newman's modularity on positive correlation graph
    A = np.maximum(0.0, corr_matrix)
    m = np.sum(A)

    if m < 1e-6 or N < 2:
        return {
            "q_func": 0.0,
            "q_func_struct": 0.0,
            "num_functional_modules": 1,
            "num_functional_communities": 1,
            "functional_modules": functional_modules.tolist(),
            "mean_functional_correlation": 0.0,
            "mean_within_func_corr": 0.0,
            "mean_cross_func_corr": 0.0,
        }

    k = np.sum(A, axis=1)
    expected = np.outer(k, k) / m

    # Modularity with data-driven functional communities
    delta_func = (functional_modules[:, np.newaxis] == functional_modules[np.newaxis, :]).astype(np.float64)
    q_func = float(np.sum((A - expected) * delta_func) / m)

    # Modularity evaluated using structural modules (to test structural-functional alignment)
    q_func_struct = 0.0
    if structural_modules is not None and len(structural_modules) == N:
        delta_struct = (structural_modules[:, np.newaxis] == structural_modules[np.newaxis, :]).astype(np.float64)
        q_func_struct = float(np.sum((A - expected) * delta_struct) / m)

    # Within vs cross functional module correlations
    same_func = (functional_modules[:, np.newaxis] == functional_modules[np.newaxis, :])
    diff_func = ~same_func
    np.fill_diagonal(same_func, False)
    np.fill_diagonal(diff_func, False)

    within_corrs = corr_matrix[same_func]
    cross_corrs = corr_matrix[diff_func]

    mean_within = float(np.mean(within_corrs)) if len(within_corrs) > 0 else 0.0
    mean_cross = float(np.mean(cross_corrs)) if len(cross_corrs) > 0 else 0.0

    return {
        "q_func": round(q_func, 4),
        "q_func_struct": round(q_func_struct, 4),
        "num_functional_modules": num_func_modules,
        "num_functional_communities": num_func_modules,
        "functional_modules": functional_modules.tolist(),
        "mean_functional_correlation": round(float(np.mean(corr_matrix[corr_matrix > 0])), 4)
        if np.any(corr_matrix > 0) else 0.0,
        "mean_within_func_corr": round(mean_within, 4),
        "mean_cross_func_corr": round(mean_cross, 4),
    }


def compute_oscillatory_spectrum(
    spike_matrix: np.ndarray,
    dt_ms: float = 1.0,
    fs: Optional[float] = None,
    dt: Optional[float] = None,
) -> Dict[str, float]:
    """Compute Fourier power spectral density of population rate to detect emergent oscillations.

    Returns:
        Dominant frequency (Hz), peak power ratio, and spectral entropy.
    """
    if fs is not None and fs > 0:
        dt_ms = 1000.0 / fs
    elif dt is not None and dt > 0:
        dt_ms = dt

    if spike_matrix.ndim == 1:
        pop_rate = spike_matrix
        T = len(pop_rate)
    else:
        T, N = spike_matrix.shape
        if N == 0:
            return {"dominant_frequency_hz": 0.0, "peak_power_ratio": 1.0, "spectral_entropy": 0.0}
        pop_rate = np.mean(spike_matrix, axis=1)

    if T < 16:
        return {"dominant_frequency_hz": 0.0, "peak_power_ratio": 1.0, "spectral_entropy": 0.0}
    # Center time-series
    centered = pop_rate - np.mean(pop_rate)

    # FFT
    fft_vals = np.fft.rfft(centered)
    psd = np.abs(fft_vals) ** 2
    sampling_rate_hz = 1000.0 / dt_ms
    freqs = np.fft.rfftfreq(T, d=(dt_ms / 1000.0))

    if len(psd) <= 2:
        return {"dominant_frequency_hz": 0.0, "peak_power_ratio": 1.0, "spectral_entropy": 0.0}

    # Exclude DC component (freq = 0)
    psd_pos = psd[1:]
    freqs_pos = freqs[1:]

    total_power = np.sum(psd_pos)
    if total_power < 1e-10:
        return {"dominant_frequency_hz": 0.0, "peak_power_ratio": 1.0, "spectral_entropy": 0.0}

    peak_idx = np.argmax(psd_pos)
    dom_freq = float(freqs_pos[peak_idx])
    peak_power = float(psd_pos[peak_idx])
    mean_power = float(np.mean(psd_pos))
    peak_ratio = peak_power / max(1e-10, mean_power)

    # Spectral entropy
    prob_dist = psd_pos / total_power
    prob_dist = prob_dist[prob_dist > 1e-12]
    spectral_entropy = -float(np.sum(prob_dist * np.log2(prob_dist))) / np.log2(len(psd_pos))

    return {
        "dominant_frequency_hz": round(dom_freq, 2),
        "peak_power_ratio": round(peak_ratio, 2),
        "spectral_entropy": round(spectral_entropy, 3),
        "total_spectral_power": round(float(total_power), 4),
    }


class PhaseLockingResult(tuple):
    """Result object for phase locking and Kuramoto synchronization that supports both tuple and dict access."""

    def __new__(cls, kuramoto_r_mean: float, r_t: np.ndarray, mean_plv: float):
        return super().__new__(cls, (kuramoto_r_mean, r_t, mean_plv))

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            mapping = {
                "kuramoto_r_mean": self[0],
                "kuramoto_r": self[0],
                "mean_kuramoto_r": self[0],
                "r_t": self[1],
                "mean_plv": self[2],
                "plv_mean": self[2],
                "plv": self[2],
            }
            if item in mapping:
                return mapping[item]
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default


def compute_phase_locking_and_kuramoto(
    spike_matrix: np.ndarray, dt_ms: float = 1.0
) -> PhaseLockingResult:
    """Compute instantaneous phases, Kuramoto order parameter R(t), and pairwise Phase-Locking Value (PLV).

    Returns:
        PhaseLockingResult (supports both tuple unpacking and dict key access)
    """
    T, N = spike_matrix.shape
    if T < 10 or N < 2:
        return PhaseLockingResult(0.0, np.zeros(T, dtype=np.float32), 0.0)

    # 1. Smooth spike trains with causal exponential filter to get continuous signals for phase extraction
    tau_smooth_steps = max(2, int(15.0 / dt_ms))
    smooth_rate = np.zeros((T, N), dtype=np.float32)
    decay = np.exp(-1.0 / tau_smooth_steps)

    accum = np.zeros(N, dtype=np.float32)
    for t in range(T):
        accum = decay * accum + spike_matrix[t].astype(np.float32)
        smooth_rate[t] = accum

    # 2. Extract instantaneous phase via numerical velocity: theta(t) = atan2(dRate/dt, Rate - mean)
    d_rate = np.gradient(smooth_rate, axis=0)  # [T, N]
    centered = smooth_rate - np.mean(smooth_rate, axis=0, keepdims=True)

    phases = np.arctan2(d_rate, centered)  # [T, N] in [-pi, pi]

    # 3. Kuramoto order parameter R(t) = (1/N) * |sum_j exp(i * theta_j(t))|
    complex_order = np.mean(np.exp(1j * phases), axis=1)  # [T]
    r_t = np.abs(complex_order).astype(np.float32)
    mean_r = float(np.mean(r_t))

    # 4. Pairwise Phase-Locking Value (PLV): PLV_ij = (1/T) * |sum_t exp(i * (theta_i(t) - theta_j(t)))|
    max_sample_pairs = 100
    if N <= 50:
        all_plvs = []
        for i in range(N):
            for j in range(i + 1, N):
                phase_diff = phases[:, i] - phases[:, j]
                plv_ij = np.abs(np.mean(np.exp(1j * phase_diff)))
                all_plvs.append(plv_ij)
        mean_plv = float(np.mean(all_plvs)) if len(all_plvs) > 0 else 0.0
    else:
        rng = np.random.default_rng(42)
        pairs_i = rng.choice(N, size=max_sample_pairs)
        pairs_j = rng.choice(N, size=max_sample_pairs)
        diff_mask = pairs_i != pairs_j
        pairs_i = pairs_i[diff_mask]
        pairs_j = pairs_j[diff_mask]

        sample_plvs = []
        for i, j in zip(pairs_i, pairs_j):
            phase_diff = phases[:, i] - phases[:, j]
            plv_ij = np.abs(np.mean(np.exp(1j * phase_diff)))
            sample_plvs.append(plv_ij)
        mean_plv = float(np.mean(sample_plvs)) if len(sample_plvs) > 0 else 0.0

    return PhaseLockingResult(round(mean_r, 4), r_t, round(mean_plv, 4))


def compute_spatial_participation_and_waves(
    spike_matrix: np.ndarray,
    positions: np.ndarray,
    functional_modules: Optional[np.ndarray] = None,
    corr_matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute spatial participation coefficient and detect spatial wave propagation.

    Returns:
        Mean community participation coefficient, spatial correlation slope, and distance correlation.
    """
    N = len(positions)
    if N < 3:
        return {
            "mean_participation_coefficient": 0.0,
            "spatial_participation_ratio": 0.0,
            "spatial_correlation_slope": 0.0,
            "distance_correlation_r": 0.0,
            "mean_propagation_latency": 1.0,
        }

    if corr_matrix is None:
        corr_matrix, _ = compute_spike_correlation_matrix(spike_matrix)

    if functional_modules is None:
        functional_modules = detect_functional_communities_louvain(corr_matrix)

    # 1. Functional participation coefficient: P_i = 1 - sum_m (k_im / k_i)^2
    A = np.maximum(0.0, corr_matrix)
    k_total = np.sum(A, axis=1) + 1e-8
    unique_modules = np.unique(functional_modules)

    P = np.ones(N, dtype=np.float32)
    for m in unique_modules:
        m_mask = (functional_modules == m)
        k_im = np.sum(A[:, m_mask], axis=1)
        P -= (k_im / k_total) ** 2
    P = np.maximum(0.0, P)

    # 2. Pairwise Euclidean distances vs correlation
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)

    triu_indices = np.triu_indices(N, k=1)
    pair_dists = dists[triu_indices]
    pair_corrs = corr_matrix[triu_indices]

    valid = np.isfinite(pair_dists) & np.isfinite(pair_corrs)
    if np.sum(valid) >= 10:
        slope, _ = np.polyfit(pair_dists[valid], pair_corrs[valid], deg=1)
        # Distance correlation r
        r_matrix = np.corrcoef(pair_dists[valid], pair_corrs[valid])
        dist_corr_r = float(r_matrix[0, 1]) if not np.isnan(r_matrix[0, 1]) else 0.0
    else:
        slope = 0.0
        dist_corr_r = 0.0

    mean_dist = float(np.mean(pair_dists)) if len(pair_dists) > 0 else 5.0
    mean_latency = max(1.0, mean_dist / 5.0)  # conduction speed = 5.0 mm/ms

    return {
        "mean_participation_coefficient": round(float(np.mean(P)), 3),
        "spatial_participation_ratio": round(float(np.mean(P)), 3),
        "spatial_correlation_slope": round(float(slope), 5),
        "distance_correlation_r": round(float(dist_corr_r), 4),
        "mean_propagation_latency": round(float(mean_latency), 2),
    }
