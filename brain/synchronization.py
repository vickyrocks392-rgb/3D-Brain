"""Multi-Scale Synchronization Analysis Engine (Phase 8).

Implements multi-scale spatial and community synchronization profiling:
- Extracts instantaneous phases from spike trains via causal exponential rate smoothing
- Computes Kuramoto order parameter R and Phase-Locking Value (PLV) across:
  1. Local spatial neighborhoods (k-nearest neighbors in 3D metric coordinates)
  2. Functional communities (data-driven Louvain clusters)
  3. Mesoscale spatial regions (structural modules or spatial octants)
  4. Global whole-brain scale
- Produces the comprehensive spatial synchronization profile: scale -> synchrony
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from brain.spiking_analysis import detect_functional_communities_louvain, compute_spike_correlation_matrix


def extract_instantaneous_phases(
    spike_matrix: np.ndarray, dt_ms: float = 1.0, tau_smooth_ms: float = 15.0
) -> np.ndarray:
    """Extract instantaneous phase angle theta_i(t) in [-pi, pi] for each neuron."""
    T, N = spike_matrix.shape
    if T < 5 or N == 0:
        return np.zeros((T, N), dtype=np.float32)

    # 1. Causal exponential smoothing
    tau_steps = max(2, int(tau_smooth_ms / dt_ms))
    decay = np.exp(-1.0 / tau_steps)
    smooth_rate = np.zeros((T, N), dtype=np.float32)
    accum = np.zeros(N, dtype=np.float32)

    for t in range(T):
        accum = decay * accum + spike_matrix[t].astype(np.float32)
        smooth_rate[t] = accum

    # 2. Phase from rate derivative and centered amplitude
    d_rate = np.gradient(smooth_rate, axis=0)
    centered = smooth_rate - np.mean(smooth_rate, axis=0, keepdims=True)
    phases = np.arctan2(d_rate, centered)  # [T, N]
    return phases.astype(np.float32)


def compute_group_kuramoto_and_plv(phases: np.ndarray) -> Tuple[float, float]:
    """Compute mean Kuramoto R and pairwise PLV for a subset of neurons with phases [T, K]."""
    T, K = phases.shape
    if T == 0 or K < 2:
        return 1.0, 1.0

    # Kuramoto R(t) = |(1/K) sum_j exp(i * theta_j(t))|
    complex_order = np.mean(np.exp(1j * phases), axis=1)  # [T]
    mean_r = float(np.mean(np.abs(complex_order)))

    # Pairwise PLV
    if K <= 20:
        plvs = []
        for i in range(K):
            for j in range(i + 1, K):
                diff = phases[:, i] - phases[:, j]
                plv = float(np.abs(np.mean(np.exp(1j * diff))))
                plvs.append(plv)
        mean_plv = float(np.mean(plvs)) if len(plvs) > 0 else 0.0
    else:
        # Sample 50 pairs
        rng = np.random.default_rng(42)
        idx_a = rng.choice(K, size=50)
        idx_b = rng.choice(K, size=50)
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]
        plvs = []
        for a, b in zip(idx_a, idx_b):
            diff = phases[:, a] - phases[:, b]
            plvs.append(float(np.abs(np.mean(np.exp(1j * diff)))))
        mean_plv = float(np.mean(plvs)) if len(plvs) > 0 else 0.0

    return mean_r, mean_plv


def analyze_multiscale_synchronization(
    spike_matrix: np.ndarray,
    positions_3d: np.ndarray,
    functional_modules: Optional[np.ndarray] = None,
    structural_modules: Optional[np.ndarray] = None,
    k_neighbors: int = 5,
    dt_ms: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute multi-scale synchronization metrics across local, community, mesoscale, and global scales."""
    T, N = spike_matrix.shape
    if T < 10 or N < 3:
        return {
            "r_global": 0.0,
            "r_local": 0.0,
            "r_community": 0.0,
            "r_meso": 0.0,
            "plv_global": 0.0,
            "plv_local": 0.0,
            "plv_community": 0.0,
            "synchrony_gradient": 0.0,
        }

    phases = extract_instantaneous_phases(spike_matrix, dt_ms=dt_ms)

    # 1. Global Synchrony
    r_global, plv_global = compute_group_kuramoto_and_plv(phases)

    # 2. Local Neighborhood Synchrony (k-nearest neighbors in 3D space)
    # Compute Euclidean distance matrix
    diff = positions_3d[:, np.newaxis, :] - positions_3d[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))  # [N, N]
    np.fill_diagonal(dists, np.inf)

    local_rs = []
    local_plvs = []
    actual_k = min(k_neighbors, N - 1)

    for i in range(N):
        nearest_neighbors = np.argsort(dists[i])[:actual_k]
        group = np.append(nearest_neighbors, i)
        r_loc, plv_loc = compute_group_kuramoto_and_plv(phases[:, group])
        local_rs.append(r_loc)
        local_plvs.append(plv_loc)

    r_local = float(np.mean(local_rs))
    plv_local = float(np.mean(local_plvs))

    # 3. Functional Community Synchrony
    if functional_modules is None:
        corr_matrix, _ = compute_spike_correlation_matrix(spike_matrix, bin_size_steps=max(2, int(10.0 / dt_ms)))
        functional_modules = detect_functional_communities_louvain(corr_matrix)

    comm_rs = []
    comm_plvs = []
    for mod_id in np.unique(functional_modules):
        mod_units = np.where(functional_modules == mod_id)[0]
        if len(mod_units) >= 2:
            r_c, plv_c = compute_group_kuramoto_and_plv(phases[:, mod_units])
            comm_rs.append(r_c)
            comm_plvs.append(plv_c)

    r_community = float(np.mean(comm_rs)) if len(comm_rs) > 0 else r_global
    plv_community = float(np.mean(comm_plvs)) if len(comm_plvs) > 0 else plv_global

    # 4. Mesoscale Structural Synchrony (structural modules or 3D spatial octants)
    if structural_modules is None:
        # Spatial octant grouping from 3D coordinates relative to center
        center = np.mean(positions_3d, axis=0)
        signs = (positions_3d >= center).astype(int)
        structural_modules = signs[:, 0] * 4 + signs[:, 1] * 2 + signs[:, 2]

    meso_rs = []
    for s_id in np.unique(structural_modules):
        s_units = np.where(structural_modules == s_id)[0]
        if len(s_units) >= 2:
            r_m, _ = compute_group_kuramoto_and_plv(phases[:, s_units])
            meso_rs.append(r_m)

    r_meso = float(np.mean(meso_rs)) if len(meso_rs) > 0 else r_global

    # 5. Synchrony Spatial Gradient: R_local - R_global (positive indicates local coherence with global asynchrony)
    sync_gradient = r_local - r_global

    return {
        "r_global": round(r_global, 4),
        "r_local": round(r_local, 4),
        "r_community": round(r_community, 4),
        "r_meso": round(r_meso, 4),
        "plv_global": round(plv_global, 4),
        "plv_local": round(plv_local, 4),
        "plv_community": round(plv_community, 4),
        "synchrony_gradient": round(sync_gradient, 4),
        "has_local_coherence": bool(r_local > r_global + 0.05),
    }


def compute_kuramoto_order_parameter(spike_matrix: np.ndarray, dt_ms: float = 1.0) -> float:
    """Compute global Kuramoto order parameter R from spike matrix [T, N]."""
    phases = extract_instantaneous_phases(spike_matrix, dt_ms=dt_ms)
    r, _ = compute_group_kuramoto_and_plv(phases)
    return float(r)


def compute_phase_locking_value(spike_matrix: np.ndarray, dt_ms: float = 1.0) -> float:
    """Compute global Phase-Locking Value (PLV) from spike matrix [T, N]."""
    phases = extract_instantaneous_phases(spike_matrix, dt_ms=dt_ms)
    _, plv = compute_group_kuramoto_and_plv(phases)
    return float(plv)


class MultiScaleSynchronyAnalyzer:
    """Class wrapper for computing multi-scale synchronization profiles."""

    def __init__(
        self,
        num_neurons: int,
        positions: Optional[np.ndarray] = None,
        module_ids: Optional[np.ndarray] = None,
        dt_ms: float = 1.0,
    ):
        self.num_neurons = num_neurons
        if positions is None:
            self.positions = np.zeros((num_neurons, 3), dtype=np.float32)
        else:
            self.positions = np.asarray(positions, dtype=np.float32)
        self.module_ids = module_ids
        self.dt_ms = dt_ms

    def compute_multiscale_synchrony(self, spike_matrix: np.ndarray) -> Dict[str, Any]:
        return analyze_multiscale_synchronization(
            spike_matrix=spike_matrix,
            positions_3d=self.positions,
            structural_modules=self.module_ids,
            dt_ms=self.dt_ms,
        )
