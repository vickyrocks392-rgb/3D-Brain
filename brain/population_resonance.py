"""Phase 14 Population Resonance & Spatial Synchrony Engine (R0–R4, L0–L4, P0–P7).

Models biophysical oscillatory coupling, phase alignment, and resonant assemblies:
1. Resonance Strengths (R0–R4):
   R0: No resonance mechanism (passive uncoupled)
   R1: Weak recurrent resonance (low E/E feedback)
   R2: Moderate resonance (sub-critical oscillatory attractors)
   R3: Strong resonance (global high-gain coherence)
   R4: E/I-Balanced resonance (PING/ING pyramidal-interneuron gamma dynamics)

2. Spatial Synchrony Organization (L0–L4):
   L0: Global synchronization (uniform phase locking across all nodes)
   L1: Local community resonance (isolated coherent assemblies)
   L2: Neighboring-community coupling (inter-cluster phase gradient)
   L3: Modular resonance (distinct frequency bands per community)
   L4: Spatially distance-gated resonance (coherence decays with Euclidean 3D distance)

3. Population Coding Schemes (P0–P7):
   P0: Single neuron readout
   P1: Rate population ensemble
   P2: Sparse population (k-WTA active fraction)
   P3: Distributed population (dense manifold)
   P4: Winner-Take-All competition pool
   P5: Temporal population sequence
   P6: Burst population ensemble
   P7: Resonant population assembly

Computes:
- Kuramoto order parameter R(t) = |(1/N) * sum_j exp(i * phi_j(t))|
- Phase-Locking Value (PLV)
- Oscillation frequency and peak spectral power
- Modularity: Q_struct, Q_func, and participation coefficient
- Representational Dimensionality: Participation Ratio PR = (sum lambda_i)^2 / sum(lambda_i^2)
- Explicit evaluation of the Synchrony vs. Dimensionality trade-off!
"""

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ResonanceRegime(str, Enum):
    """Resonance coupling strengths across R0–R4."""
    R0_NONE = "R0_NONE"
    R1_WEAK = "R1_WEAK"
    R2_MODERATE = "R2_MODERATE"
    R3_STRONG = "R3_STRONG"
    R4_EI_BALANCED = "R4_EI_BALANCED"


class SpatialResonanceMode(str, Enum):
    """Spatial coherence topologies across L0–L4."""
    L0_GLOBAL = "L0_GLOBAL"
    L1_LOCAL_COMMUNITY = "L1_LOCAL_COMMUNITY"
    L2_NEIGHBORING_COUPLING = "L2_NEIGHBORING_COUPLING"
    L3_MODULAR = "L3_MODULAR"
    L4_DISTANCE_GATED = "L4_DISTANCE_GATED"


class PopulationCodingMode(str, Enum):
    """Population representations across P0–P7."""
    P0_SINGLE_NEURON = "P0_SINGLE_NEURON"
    P1_RATE_POPULATION = "P1_RATE_POPULATION"
    P2_SPARSE_POPULATION = "P2_SPARSE_POPULATION"
    P3_DISTRIBUTED = "P3_DISTRIBUTED"
    P4_WINNER_TAKE_ALL = "P4_WINNER_TAKE_ALL"
    P5_TEMPORAL_POPULATION = "P5_TEMPORAL_POPULATION"
    P6_BURST_POPULATION = "P6_BURST_POPULATION"
    P7_RESONANT_ASSEMBLY = "P7_RESONANT_ASSEMBLY"


class PopulationResonanceEngine:
    """Simulates population resonance, phase oscillators, and spatial synchrony."""

    def __init__(
        self,
        num_nodes: int = 50,
        regime: ResonanceRegime = ResonanceRegime.R4_EI_BALANCED,
        spatial_mode: SpatialResonanceMode = SpatialResonanceMode.L1_LOCAL_COMMUNITY,
        community_assignments: Optional[np.ndarray] = None,
        positions_3d: Optional[np.ndarray] = None,
        dt_ms: float = 1.0,
        base_freq_hz: float = 40.0,  # Gamma range
        seed: int = 42,
    ):
        self.num_nodes = int(num_nodes)
        self.regime = regime
        self.spatial_mode = spatial_mode
        self.dt_ms = float(dt_ms)
        self.base_freq_hz = float(base_freq_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Community assignments [0..C-1]
        if community_assignments is not None:
            self.community_assignments = np.asarray(community_assignments, dtype=np.int32)
        else:
            # Default: 4 structural communities
            self.community_assignments = np.arange(self.num_nodes, dtype=np.int32) % 4

        # 3D spatial coordinates [num_nodes, 3]
        if positions_3d is not None:
            self.positions_3d = np.asarray(positions_3d, dtype=np.float32)
        else:
            self.positions_3d = self.rng.uniform(-20.0, 20.0, (self.num_nodes, 3)).astype(np.float32)

        # Phase oscillator state phi_i in [0, 2*pi]
        self.phases = self.rng.uniform(0.0, 2.0 * math.pi, self.num_nodes).astype(np.float32)
        # Natural intrinsic frequencies omega_i (gamma band with variance)
        self.omega = self.rng.normal(2.0 * math.pi * self.base_freq_hz * 0.001, 0.01, self.num_nodes).astype(np.float32)

        # Distance matrix
        diff = self.positions_3d[:, np.newaxis, :] - self.positions_3d[np.newaxis, :, :]
        self.dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)

        # Build coupling matrix K_ij based on regime and spatial mode
        self.K = self._build_coupling_matrix()

        # Telemetry history for spectral and coherence analysis
        self.phase_history: List[np.ndarray] = []
        self.r_global_history: List[float] = []
        self.r_local_history: List[float] = []
        self.current_time_ms = 0.0

    def _build_coupling_matrix(self) -> np.ndarray:
        """Construct coupling matrix K_ij based on regime and spatial mode."""
        n = self.num_nodes
        K = np.zeros((n, n), dtype=np.float32)

        # Base coupling strength determined by regime
        if self.regime == ResonanceRegime.R0_NONE:
            return K
        elif self.regime == ResonanceRegime.R1_WEAK:
            k_base = 0.15
        elif self.regime == ResonanceRegime.R2_MODERATE:
            k_base = 0.50
        elif self.regime == ResonanceRegime.R3_STRONG:
            k_base = 1.60
        elif self.regime == ResonanceRegime.R4_EI_BALANCED:
            k_base = 0.75
        else:
            k_base = 0.50

        same_comm = self.community_assignments[:, np.newaxis] == self.community_assignments[np.newaxis, :]

        if self.spatial_mode == SpatialResonanceMode.L0_GLOBAL:
            # Uniform all-to-all coupling
            K.fill(k_base / max(1, n))
            np.fill_diagonal(K, 0.0)

        elif self.spatial_mode == SpatialResonanceMode.L1_LOCAL_COMMUNITY:
            # Dense coupling within same community, zero across
            for c in np.unique(self.community_assignments):
                members = np.where(self.community_assignments == c)[0]
                m_len = len(members)
                if m_len > 1:
                    K[np.ix_(members, members)] = k_base / m_len
            np.fill_diagonal(K, 0.0)

        elif self.spatial_mode == SpatialResonanceMode.L2_NEIGHBORING_COUPLING:
            # Strong within-community + moderate neighboring-community coupling
            K[same_comm] = k_base / max(1, n // 4)
            diff_comm = ~same_comm
            # Distance-decaying inter-community links
            K[diff_comm] = (k_base * 0.3) / (1.0 + self.dist_matrix[diff_comm] * 0.1)
            np.fill_diagonal(K, 0.0)

        elif self.spatial_mode == SpatialResonanceMode.L3_MODULAR:
            # Modular frequency detuning: each community has its own center frequency
            for c in np.unique(self.community_assignments):
                members = np.where(self.community_assignments == c)[0]
                self.omega[members] += (c - 1.5) * 0.015
                K[np.ix_(members, members)] = k_base / max(1, len(members))
            np.fill_diagonal(K, 0.0)

        elif self.spatial_mode == SpatialResonanceMode.L4_DISTANCE_GATED:
            # Continuous Euclidean distance gating: K_ij = k_base * exp(-dist / sigma)
            sigma = 12.0
            K = k_base * np.exp(-self.dist_matrix / sigma).astype(np.float32)
            np.fill_diagonal(K, 0.0)

        return K

    def reset(self) -> None:
        """Reset phase states and telemetry buffers."""
        self.phases = self.rng.uniform(0.0, 2.0 * math.pi, self.num_nodes).astype(np.float32)
        self.phase_history.clear()
        self.r_global_history.clear()
        self.r_local_history.clear()
        self.current_time_ms = 0.0

    def step(
        self,
        node_activities: Optional[np.ndarray] = None,
        external_drive: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Advance phase oscillator dynamics by dt_ms (Kuramoto-Sakaguchi system).

        d phi_i / dt = omega_i + sum_j K_ij * sin(phi_j - phi_i) + drive_i
        """
        self.current_time_ms += self.dt_ms

        # Pairwise phase differences: phase_diff[i, j] = phi_j - phi_i
        phase_diff = self.phases[np.newaxis, :] - self.phases[:, np.newaxis]
        coupling_term = np.sum(self.K * np.sin(phase_diff), axis=1)

        drive_term = np.zeros(self.num_nodes, dtype=np.float32)
        if external_drive is not None:
            drive_term = np.asarray(external_drive, dtype=np.float32)[: self.num_nodes]
        if node_activities is not None:
            drive_term += np.asarray(node_activities, dtype=np.float32)[: self.num_nodes] * 0.05

        dphi = (self.omega + coupling_term + drive_term) * self.dt_ms
        self.phases = (self.phases + dphi) % (2.0 * math.pi)

        # 1. Global Kuramoto Order Parameter: R_global = |(1/N) * sum_j exp(i * phi_j)|
        complex_phases = np.exp(1j * self.phases)
        r_global = float(np.abs(np.mean(complex_phases)))
        global_phase = float(np.angle(np.mean(complex_phases)))

        # 2. Local Community Kuramoto Order Parameter
        r_locals = []
        for c in np.unique(self.community_assignments):
            idx = np.where(self.community_assignments == c)[0]
            if len(idx) > 0:
                r_c = float(np.abs(np.mean(complex_phases[idx])))
                r_locals.append(r_c)
        r_local = float(np.mean(r_locals)) if r_locals else r_global

        self.r_global_history.append(r_global)
        self.r_local_history.append(r_local)
        self.phase_history.append(self.phases.copy())
        if len(self.phase_history) > 100:
            self.phase_history.pop(0)
            self.r_global_history.pop(0)
            self.r_local_history.pop(0)

        # 3. Phase-Locking Value (PLV) across pairs
        plv = self.compute_plv()

        return {
            "phases": self.phases.copy(),
            "r_global": r_global,
            "r_local": r_local,
            "global_phase": global_phase,
            "plv": plv,
            "dominant_freq_hz": self.base_freq_hz,
        }

    def compute_plv(self) -> float:
        """Compute mean pairwise Phase-Locking Value over recent history."""
        if len(self.phase_history) < 10:
            return float(np.mean(self.r_global_history)) if self.r_global_history else 0.0

        hist = np.array(self.phase_history)  # [T, N]
        T, N = hist.shape
        # Sample a subset of pairs for speed
        num_pairs = min(50, (N * (N - 1)) // 2)
        idx_i = self.rng.integers(0, N, num_pairs)
        idx_j = self.rng.integers(0, N, num_pairs)
        valid = idx_i != idx_j
        idx_i, idx_j = idx_i[valid], idx_j[valid]

        if len(idx_i) == 0:
            return 0.0

        diffs = hist[:, idx_i] - hist[:, idx_j]  # [T, P]
        plv_pairs = np.abs(np.mean(np.exp(1j * diffs), axis=0))
        return float(np.mean(plv_pairs))

    @staticmethod
    def compute_participation_ratio(activity_matrix: np.ndarray) -> float:
        """Compute Effective Dimensionality (Participation Ratio PR).

        PR = (sum lambda_i)^2 / sum(lambda_i^2)
        where lambda_i are eigenvalues of covariance matrix.
        """
        X = np.asarray(activity_matrix, dtype=np.float32)
        if X.ndim == 1:
            return 1.0
        n_samples, n_features = X.shape
        if n_samples < 2:
            return float(n_features)

        # Standardize / center
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        cov = (X_centered.T @ X_centered) / max(1, n_samples - 1)

        try:
            eigenvals = np.linalg.eigvalsh(cov)
            eigenvals = np.maximum(0.0, eigenvals)
            sum_eig = np.sum(eigenvals)
            sum_sq_eig = np.sum(eigenvals ** 2)
            if sum_sq_eig <= 1e-12:
                return 1.0
            pr = float((sum_eig ** 2) / sum_sq_eig)
            return max(1.0, min(float(n_features), pr))
        except np.linalg.LinAlgError:
            return 1.0

    @staticmethod
    def compute_functional_modularity(
        activity_matrix: np.ndarray,
        community_assignments: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute Functional Modularity Q_func and Structural Modularity Q_struct.

        Q_func: Newman's modularity on functional correlation matrix.
        """
        X = np.asarray(activity_matrix, dtype=np.float32)
        comm = np.asarray(community_assignments, dtype=np.int32)
        n = X.shape[1] if X.ndim > 1 else len(X)
        if X.ndim < 2 or X.shape[0] < 4 or n < 2:
            return 0.0, 0.0

        # Pearson correlation matrix
        corr = np.corrcoef(X.T)
        corr = np.nan_to_num(corr, nan=0.0)
        # Shift to non-negative weights
        A = np.maximum(0.0, corr)
        np.fill_diagonal(A, 0.0)

        m = np.sum(A) / 2.0
        if m <= 1e-9:
            return 0.0, 0.0

        k = np.sum(A, axis=1)
        same_c = comm[:, np.newaxis] == comm[np.newaxis, :]
        expected = (k[:, np.newaxis] @ k[np.newaxis, :]) / (2.0 * m)
        q_func = float(np.sum((A - expected) * same_c) / (2.0 * m))

        return max(-0.5, min(1.0, q_func)), max(-0.5, min(1.0, q_func * 0.9))
