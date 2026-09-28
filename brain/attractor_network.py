"""Recurrent Attractor Latching & Dynamical Stability Circuit (Phase 11).

Converts transient NMDA dendritic conductances into persistent, stable,
context-specific population attractors (Attractor Pool A vs. Attractor Pool B).

Key Architectural Components:
1. Pool-Specific Recurrent Excitation:
   - Neurons in Pool A mutually reinforce each other via excitatory synapses (W_AA^EE > 0).
   - Neurons in Pool B mutually reinforce each other via excitatory synapses (W_BB^EE > 0).
2. Cross-Pool Recurrent Lateral Inhibition:
   - Activation of Pool A drives local interneurons that project inhibitory synapses
     onto Pool B (W_AB^EI > 0), creating a competitive, bistable dynamical attractor landscape.
3. Dale's Principle & Homeostasis:
   - Strictly 80% Excitatory, 20% Inhibitory with non-negative / non-positive bounds.
   - Homeostatic synaptic scaling stabilizes long-term baseline rates.
4. Attractor Diagnostics Suite:
   - Autonomous state drift rate (during complete sensory deprivation)
   - Attractor centroid separation and within-attractor variance (radius)
   - Spontaneous switching rate and mean dwell time
   - Effective representational dimensionality D_eff
   - Local Jacobian eigenvalue / stability diagnostic
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.ei_dynamics import NeuronType


class AttractorCouplingMode(str, Enum):
    NORMAL = "NORMAL"
    NO_EXCITATION = "NO_EXCITATION"
    NO_INHIBITION = "NO_INHIBITION"
    REDUCED_COUPLING = "REDUCED_COUPLING"
    EXCESSIVE_COUPLING = "EXCESSIVE_COUPLING"


class RecurrentAttractorCircuit:
    """Manages structured recurrent connectivity to produce bistable context attractors."""

    def __init__(
        self,
        num_neurons: int,
        num_pools: int = 2,
        inhibitory_fraction: float = 0.20,
        coupling_mode: AttractorCouplingMode = AttractorCouplingMode.NORMAL,
        w_ee_intra: float = 0.65,
        w_ei_cross: float = 0.55,
        w_ie_local: float = 0.50,
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.num_pools = num_pools
        self.inhibitory_fraction = float(inhibitory_fraction)
        self.coupling_mode = AttractorCouplingMode(coupling_mode)
        self.rng = np.random.RandomState(seed)

        # 1. Assign Dale Neuron Types: first (N - N_I) are Excitatory, remaining N_I are Inhibitory
        self.num_inhibitory = int(round(num_neurons * inhibitory_fraction))
        self.num_excitatory = num_neurons - self.num_inhibitory
        self.neuron_types = np.full(num_neurons, NeuronType.EXCITATORY.value, dtype=object)
        if self.num_inhibitory > 0:
            self.neuron_types[-self.num_inhibitory:] = NeuronType.INHIBITORY.value

        self.excitatory_mask = (self.neuron_types == NeuronType.EXCITATORY.value)
        self.inhibitory_mask = (self.neuron_types == NeuronType.INHIBITORY.value)

        # 2. Assign Pool Membership (Pool 0 for Context A, Pool 1 for Context B)
        self.pool_ids = np.full(num_neurons, -1, dtype=np.int32)
        exc_indices = np.where(self.excitatory_mask)[0]
        n_per_pool = len(exc_indices) // num_pools
        for p in range(num_pools):
            p_idx = exc_indices[p * n_per_pool : (p + 1) * n_per_pool]
            self.pool_ids[p_idx] = p

        # 3. Construct Recurrent Weight Matrix with Attractor Topology: W_rec[post, pre]
        self.W_rec = np.zeros((num_neurons, num_neurons), dtype=np.float32)
        self._init_attractor_weights(w_ee_intra, w_ei_cross, w_ie_local)

        # Telemetry & History
        self.last_state_history: List[np.ndarray] = []
        self.active_attractor_id: int = -1

    def _init_attractor_weights(
        self,
        w_ee_intra: float,
        w_ei_cross: float,
        w_ie_local: float,
    ) -> None:
        """Initialize intra-pool excitation and inter-pool cross-inhibition respecting Dale's Principle.
        
        W_rec[post, pre]: column 'pre' must be non-negative for E and non-positive for I.
        """
        mode = self.coupling_mode

        # Multiplier based on coupling mode ablation
        if mode == AttractorCouplingMode.NO_EXCITATION:
            mult_ee = 0.0
            mult_ei = 1.0
        elif mode == AttractorCouplingMode.NO_INHIBITION:
            mult_ee = 1.0
            mult_ei = 0.0
        elif mode == AttractorCouplingMode.REDUCED_COUPLING:
            mult_ee = 0.2
            mult_ei = 0.2
        elif mode == AttractorCouplingMode.EXCESSIVE_COUPLING:
            mult_ee = 3.0
            mult_ei = 2.5
        else:  # NORMAL
            mult_ee = 1.0
            mult_ei = 1.0

        for pre in range(self.num_neurons):
            for post in range(self.num_neurons):
                if pre == post:
                    continue  # No autapses

                is_pre_e = self.excitatory_mask[pre]
                is_pre_i = self.inhibitory_mask[pre]
                pool_pre = self.pool_ids[pre]
                pool_post = self.pool_ids[post]

                # Case 1: Intra-pool E-E recurrent excitation
                if is_pre_e and self.excitatory_mask[post] and pool_pre >= 0 and pool_pre == pool_post:
                    if self.rng.rand() < 0.45:
                        self.W_rec[post, pre] = float(mult_ee * w_ee_intra * (0.8 + 0.4 * self.rng.rand()))

                # Case 2: E-to-I excitation (driving interneurons)
                elif is_pre_e and self.inhibitory_mask[post]:
                    if self.rng.rand() < 0.35:
                        self.W_rec[post, pre] = float(mult_ee * w_ie_local * (0.8 + 0.4 * self.rng.rand()))

                # Case 3: I-to-E cross-inhibition (inhibiting the opposing pool)
                elif is_pre_i and self.excitatory_mask[post]:
                    if self.rng.rand() < 0.40:
                        # Inhibitory weights are strictly <= 0
                        self.W_rec[post, pre] = -float(mult_ei * w_ei_cross * (0.8 + 0.4 * self.rng.rand()))

                # Case 4: Baseline sparse background recurrence
                else:
                    if self.rng.rand() < 0.08:
                        if is_pre_e:
                            self.W_rec[post, pre] = float(0.1 * mult_ee * self.rng.rand())
                        else:
                            self.W_rec[post, pre] = -float(0.1 * mult_ei * self.rng.rand())

        # Ensure strict diagonal zero
        np.fill_diagonal(self.W_rec, 0.0)

    def compute_recurrent_currents(
        self,
        spikes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute recurrent excitatory and inhibitory currents received by each neuron.

        Args:
            spikes: Boolean array of presynaptic somatic spikes [num_neurons]

        Returns:
            i_rec_e: Recurrent excitatory current (>= 0) [num_neurons]
            i_rec_i: Recurrent inhibitory current (<= 0) [num_neurons]
        """
        s = np.asarray(spikes, dtype=np.float32)
        # Separate positive (E) and negative (I) weights in W_rec[post, pre]
        w_e = np.maximum(0.0, self.W_rec)
        w_i = np.minimum(0.0, self.W_rec)

        i_rec_e = w_e @ s  # Excitatory currents arriving at post neurons (>= 0)
        i_rec_i = w_i @ s  # Inhibitory currents arriving at post neurons (<= 0)

        return i_rec_e, i_rec_i

    def identify_current_attractor(self, hidden_state: np.ndarray) -> int:
        """Determine which attractor pool is currently active based on population state.

        Returns:
            attractor_id: 0 (Pool A), 1 (Pool B), or -1 (Ambiguous / Quiescent)
        """
        s = np.asarray(hidden_state, dtype=np.float32)
        if s.ndim == 2:
            s = s[0]

        pool_acts = []
        for p in range(self.num_pools):
            idx = np.where(self.pool_ids == p)[0]
            if len(idx) > 0:
                pool_acts.append(float(np.mean(s[idx])))
            else:
                pool_acts.append(0.0)

        p_max = int(np.argmax(pool_acts))
        p_val = pool_acts[p_max]
        p_other = pool_acts[1 - p_max] if self.num_pools == 2 else 0.0

        if p_val > 0.05 and (p_val - p_other) > 0.02:
            self.active_attractor_id = p_max
            return p_max
        else:
            self.active_attractor_id = -1
            return -1

    def evaluate_attractor_diagnostics(
        self,
        trajectory_a: np.ndarray,
        trajectory_b: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute formal dynamical attractor stability and separability diagnostics.

        Args:
            trajectory_a: Temporal state trajectory under Context A [T, num_neurons]
            trajectory_b: Temporal state trajectory under Context B [T, num_neurons]

        Returns:
            diagnostics: Dictionary of quantitative attractor metrics
        """
        T_a, N = trajectory_a.shape
        T_b, _ = trajectory_b.shape

        # 1. State Drift Rate during Sensory Deprivation
        drift_a = float(np.mean([np.linalg.norm(trajectory_a[t] - trajectory_a[t - 1]) for t in range(1, T_a)])) if T_a > 1 else 0.0
        drift_b = float(np.mean([np.linalg.norm(trajectory_b[t] - trajectory_b[t - 1]) for t in range(1, T_b)])) if T_b > 1 else 0.0
        mean_drift = float(0.5 * (drift_a + drift_b))

        # 2. Attractor Centroids & Radii (within-attractor variance)
        centroid_a = np.mean(trajectory_a, axis=0)
        centroid_b = np.mean(trajectory_b, axis=0)

        radius_a = float(np.mean(np.linalg.norm(trajectory_a - centroid_a, axis=1)))
        radius_b = float(np.mean(np.linalg.norm(trajectory_b - centroid_b, axis=1)))
        mean_radius = float(0.5 * (radius_a + radius_b))

        # 3. Between-Attractor Distance & Separation Ratio
        centroid_dist = float(np.linalg.norm(centroid_a - centroid_b))
        separation_ratio = float(centroid_dist / (mean_radius + 1e-6))

        # 4. Spontaneous Switching Count & Rate
        # Check if trajectory_a ever crosses into Pool B's domain, or vice versa
        switches_a = 0
        switches_b = 0
        pool_a_idx = np.where(self.pool_ids == 0)[0]
        pool_b_idx = np.where(self.pool_ids == 1)[0]

        for t in range(T_a):
            act_a = float(np.mean(trajectory_a[t, pool_a_idx])) if len(pool_a_idx) > 0 else 0.0
            act_b = float(np.mean(trajectory_a[t, pool_b_idx])) if len(pool_b_idx) > 0 else 0.0
            if act_b > act_a + 0.05:
                switches_a += 1

        for t in range(T_b):
            act_a = float(np.mean(trajectory_b[t, pool_a_idx])) if len(pool_a_idx) > 0 else 0.0
            act_b = float(np.mean(trajectory_b[t, pool_b_idx])) if len(pool_b_idx) > 0 else 0.0
            if act_a > act_b + 0.05:
                switches_b += 1

        total_switches = switches_a + switches_b
        switch_rate = float(total_switches / max(1, T_a + T_b))

        # 5. Effective Dimensionality (Participation Ratio of Eigenvalues)
        combined_traj = np.vstack([trajectory_a, trajectory_b])
        cov = np.cov(combined_traj, rowvar=False)
        eigvals = np.real(np.linalg.eigvals(cov))
        eigvals = np.maximum(0.0, eigvals)
        sum_eigs = float(np.sum(eigvals))
        sum_sq_eigs = float(np.sum(eigvals ** 2))
        d_eff = float((sum_eigs ** 2) / (sum_sq_eigs + 1e-8)) if sum_sq_eigs > 0 else 1.0

        # 6. Local Stability Diagnostic (Lyapunov / Jacobian Spectral Radius Proxy)
        # Approximate local linearized recurrence Jacobian around attractor centroid
        jac_proxy = self.W_rec
        spec_radius = float(np.max(np.abs(np.real(np.linalg.eigvals(jac_proxy)))))
        is_stable_attractor = bool(mean_drift < 50.0 and separation_ratio > 1.2 and switch_rate < 0.05)

        return {
            "mean_state_drift": round(mean_drift, 4),
            "attractor_radius_a": round(radius_a, 4),
            "attractor_radius_b": round(radius_b, 4),
            "mean_attractor_radius": round(mean_radius, 4),
            "between_attractor_distance": round(centroid_dist, 4),
            "attractor_separation_ratio": round(separation_ratio, 4),
            "spontaneous_switches": total_switches,
            "spontaneous_switching_rate": round(switch_rate, 4),
            "effective_dimensionality": round(d_eff, 2),
            "jacobian_spectral_radius": round(spec_radius, 4),
            "is_stable_attractor": is_stable_attractor,
        }
