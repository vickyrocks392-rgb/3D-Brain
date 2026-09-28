"""Targeted Spatial Feedback and Dendritic Credit Assignment Engine for 3D-Brain (Phase 10).

Solves the Credit Diffusion Problem discovered in Phase 9:
- Rather than broadcasting an identical scalar RPE to all recurrent synapses, feedback is
  targeted to post-synaptic dendritic compartments using spatial feedback alignment.
- Synapses terminating on dendrite j receive modulatory signal M_j(t) projected via
  topographically structured or random feedback alignment pathways.
- Dendritic eligibility traces combine pre-synaptic action potentials with dendritic
  membrane depolarization: e_ij(t) = lambda_e * e_ij(t-1) + S_i(t) * phi(V_dend,j).
- Rigorously computes Credit Specificity Index (CSI), Credit Precision, Credit Leakage,
  and Eligibility Concentration to prove spatial localization of learning.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class FeedbackMode(str, Enum):
    SPATIAL_LOCAL = "SPATIAL_LOCAL"          # Distance-decayed projection aligned to 3D geometry
    RANDOM_ALIGNMENT = "RANDOM_ALIGNMENT"    # Fixed random feedback weights (Lillicrap et al.)
    COMMUNITY_LOCAL = "COMMUNITY_LOCAL"      # Modular community-targeted feedback
    GLOBAL_SCALAR = "GLOBAL_SCALAR"          # Phase 9 scalar broadcast (comparison control)
    SHUFFLED_FEEDBACK = "SHUFFLED_FEEDBACK"  # Permuted feedback alignment (ablation)
    FROZEN = "FROZEN"                        # Plasticity disabled


class DendriticEligibilityMode(str, Enum):
    VOLTAGE_GATED = "VOLTAGE_GATED"          # e_ij += S_i * max(0, V_dend_j - V_rest)
    SPIKE_COINCIDENCE = "COINCIDENCE"        # e_ij += S_i * S_j
    ZERO = "ZERO"                            # e_ij = 0 (ablation control)


@dataclass
class TargetedUpdateRecord:
    """Detailed record of a spatially targeted synaptic update."""
    step: int
    source: int
    target: int
    eligibility: float
    neuromodulator: float
    delta_w: float
    new_weight: float
    is_relevant: bool = False
    target_community: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "src": int(self.source),
            "dst": int(self.target),
            "eligibility": round(float(self.eligibility), 6),
            "neuromodulator": round(float(self.neuromodulator), 5),
            "dw": round(float(self.delta_w), 6),
            "w": round(float(self.new_weight), 5),
            "relevant": self.is_relevant,
        }


class TargetedSpatialFeedbackEngine:
    """Computes neuron-specific dendritic modulatory feedback and updates synaptic weights."""

    def __init__(
        self,
        num_neurons: int,
        output_dim: int = 1,
        learning_rate: float = 0.005,
        tau_eligibility_ms: float = 50.0,
        feedback_mode: FeedbackMode = FeedbackMode.SPATIAL_LOCAL,
        eligibility_mode: DendriticEligibilityMode = DendriticEligibilityMode.VOLTAGE_GATED,
        positions: Optional[np.ndarray] = None,          # [N, 3] 3D coordinates
        output_positions: Optional[np.ndarray] = None,   # [output_dim, 3] readout coordinates
        module_ids: Optional[np.ndarray] = None,         # [N] community cluster indices
        w_min_e: float = 0.0,
        w_max_e: float = 2.5,
        w_min_i: float = 0.0,
        w_max_i: float = 4.0,
        is_inhibitory: Optional[np.ndarray] = None,
        rpe_baseline_alpha: float = 0.05,
        sigma_feedback: float = 6.0,                     # Spatial decay length (mm)
        dt_ms: float = 1.0,
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.output_dim = output_dim
        self.eta = float(learning_rate)
        self.tau_e = max(1.0, float(tau_eligibility_ms))
        self.feedback_mode = feedback_mode
        self.eligibility_mode = eligibility_mode
        self.dt_ms = float(dt_ms)
        self.sigma_fb = max(0.1, float(sigma_feedback))
        self.rpe_baseline_alpha = float(np.clip(rpe_baseline_alpha, 0.001, 1.0))
        self.rng = np.random.default_rng(seed)

        self.lambda_e = float(np.exp(-self.dt_ms / self.tau_e))

        # Dale boundaries
        self.w_min_e = float(w_min_e)
        self.w_max_e = float(w_max_e)
        self.w_min_i = float(w_min_i)
        self.w_max_i = float(w_max_i)

        if is_inhibitory is None:
            self.is_inhibitory = np.zeros(num_neurons, dtype=bool)
        else:
            self.is_inhibitory = np.asarray(is_inhibitory, dtype=bool)
        self.is_excitatory = ~self.is_inhibitory

        self.module_ids = np.asarray(module_ids, dtype=np.int32) if module_ids is not None else None

        # 3D spatial layout
        self.positions = positions if positions is not None else np.zeros((num_neurons, 3), dtype=np.float32)
        if output_positions is not None:
            self.output_positions = output_positions
        else:
            # Default output readout at z = +8.0
            self.output_positions = np.zeros((output_dim, 3), dtype=np.float32)
            self.output_positions[:, 2] = 8.0

        # Construct Feedback Alignment Projection Matrix B [num_neurons, output_dim]
        self.B_fb = np.zeros((num_neurons, output_dim), dtype=np.float32)
        self._initialize_feedback_matrix(seed)

        # Synaptic eligibility trace matrix E [pre, post]
        self.E = np.zeros((num_neurons, num_neurons), dtype=np.float32)

        # Moving reward baseline for signed RPE
        self.reward_baseline = 0.0
        self.last_rpe = 0.0
        self.last_raw_reward = 0.0

        # Telemetry & analysis metrics
        self.last_modulator_vector = np.zeros(num_neurons, dtype=np.float32)
        self.recent_updates: List[TargetedUpdateRecord] = []
        self.last_mean_eligibility = 0.0
        self.last_max_eligibility = 0.0
        self.last_total_weight_change = 0.0

        # Predefined Credit Specificity Metrics
        self.last_credit_precision = 0.0
        self.last_credit_leakage = 0.0
        self.last_credit_specificity_index = 0.0
        self.last_eligibility_concentration = 0.0

    def _initialize_feedback_matrix(self, seed: int) -> None:
        """Initialize feedback projection matrix B based on feedback mode."""
        rng = np.random.default_rng(seed)

        if self.feedback_mode == FeedbackMode.SPATIAL_LOCAL:
            # Topographically structured feedback decaying with 3D Euclidean distance
            for k in range(self.output_dim):
                out_pos = self.output_positions[k]
                dists = np.linalg.norm(self.positions - out_pos, axis=-1)  # [N]
                weights = np.exp(-dists / self.sigma_fb).astype(np.float32)
                # Randomize orientation sign (+1 / -1) per neuron so both positive and negative signals propagate
                signs = rng.choice([-1.0, 1.0], size=self.num_neurons).astype(np.float32)
                self.B_fb[:, k] = weights * signs

        elif self.feedback_mode == FeedbackMode.RANDOM_ALIGNMENT:
            # Lillicrap Random Feedback Alignment (fixed random weights)
            self.B_fb = rng.normal(0.0, 1.0, size=(self.num_neurons, self.output_dim)).astype(np.float32)

        elif self.feedback_mode == FeedbackMode.COMMUNITY_LOCAL:
            # Community targeted: each community receives directional feedback
            self.B_fb = np.zeros((self.num_neurons, self.output_dim), dtype=np.float32)
            if self.module_ids is not None:
                num_mods = max(1, int(np.max(self.module_ids)) + 1)
                mod_signs = rng.choice([-1.0, 1.0], size=(num_mods, self.output_dim)).astype(np.float32)
                for i in range(self.num_neurons):
                    c = self.module_ids[i]
                    self.B_fb[i, :] = mod_signs[c % num_mods]
            else:
                self.B_fb = rng.normal(0.0, 1.0, size=(self.num_neurons, self.output_dim)).astype(np.float32)

        elif self.feedback_mode == FeedbackMode.SHUFFLED_FEEDBACK:
            # Spatial coordinates shuffled, destroying topological alignment
            for k in range(self.output_dim):
                out_pos = self.output_positions[k]
                dists = np.linalg.norm(self.positions - out_pos, axis=-1)
                weights = np.exp(-dists / self.sigma_fb).astype(np.float32)
                rng.shuffle(weights)
                signs = rng.choice([-1.0, 1.0], size=self.num_neurons).astype(np.float32)
                self.B_fb[:, k] = weights * signs

        elif self.feedback_mode == FeedbackMode.GLOBAL_SCALAR:
            # Scalar broadcast: all neurons have uniform feedback weight 1.0
            self.B_fb.fill(1.0)

        elif self.feedback_mode == FeedbackMode.FROZEN:
            self.B_fb.fill(0.0)

    def reset(self) -> None:
        """Reset internal traces, baselines, and telemetry."""
        self.E.fill(0.0)
        self.reward_baseline = 0.0
        self.last_rpe = 0.0
        self.last_raw_reward = 0.0
        self.last_modulator_vector.fill(0.0)
        self.recent_updates.clear()
        self.last_mean_eligibility = 0.0
        self.last_max_eligibility = 0.0
        self.last_total_weight_change = 0.0
        self.last_credit_precision = 0.0
        self.last_credit_leakage = 0.0
        self.last_credit_specificity_index = 0.0
        self.last_eligibility_concentration = 0.0

    def compute_targeted_feedback(
        self,
        reward: float = 0.0,
        error_vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the post-synaptic neuron-specific feedback vector M [N].

        Args:
            reward: Scalar teaching reward signal R(t)
            error_vector: Output error / signed RPE vector delta [output_dim]

        Returns:
            m_vector: Modulatory signal arriving at each neuron's dendritic compartment [N]
        """
        self.last_raw_reward = float(reward)

        if self.feedback_mode == FeedbackMode.FROZEN:
            self.last_modulator_vector.fill(0.0)
            return self.last_modulator_vector

        # Calculate signed RPE
        rpe = float(reward) - self.reward_baseline
        self.reward_baseline = (
            (1.0 - self.rpe_baseline_alpha) * self.reward_baseline
            + self.rpe_baseline_alpha * float(reward)
        )
        self.last_rpe = rpe

        if error_vector is not None:
            err = np.asarray(error_vector, dtype=np.float32)
        else:
            err = np.full(self.output_dim, rpe, dtype=np.float32)

        if self.feedback_mode == FeedbackMode.GLOBAL_SCALAR:
            # Phase 9 broadcast: exact scalar across all neurons
            m_vector = np.full(self.num_neurons, float(rpe), dtype=np.float32)
        else:
            # Spatially targeted projection: M = B_fb * delta
            m_vector = np.dot(self.B_fb, err).astype(np.float32)

        self.last_modulator_vector = m_vector.copy()
        return m_vector

    def update_dendritic_eligibility(
        self,
        pre_spikes: np.ndarray,      # [N] boolean spikes of pre-synaptic neurons
        v_dend: np.ndarray,          # [N] dendritic membrane potential of post-synaptic neurons
        v_rest: float = -70.0,
        post_spikes: Optional[np.ndarray] = None,
        adjacency_mask: Optional[np.ndarray] = None, # [pre, post]
    ) -> np.ndarray:
        """Update dendritic synaptic eligibility traces E[pre, post].

        Biological Mechanism:
        A pre-synaptic spike delivers glutamate to dendritic spine. If the post-synaptic
        dendrite is simultaneously depolarized (opening NMDA channels / triggering Ca2+ influx),
        an active synaptic eligibility tag is set.
        """
        if self.eligibility_mode == DendriticEligibilityMode.ZERO:
            self.E.fill(0.0)
            self.last_mean_eligibility = 0.0
            self.last_max_eligibility = 0.0
            return self.E

        # 1. Exponential decay of existing eligibility traces
        self.E *= self.lambda_e

        spk_pre = pre_spikes.astype(np.float32)  # [N]

        if self.eligibility_mode == DendriticEligibilityMode.VOLTAGE_GATED:
            # Depolarization factor: phi(V_dend) = max(0, V_dend - V_rest) / 20.0
            dend_depol = np.maximum(0.0, (v_dend - float(v_rest)) / 20.0).astype(np.float32)  # [post]
            tag_increment = np.outer(spk_pre, dend_depol)  # [pre, post]
            if adjacency_mask is not None:
                np.add(self.E, tag_increment, out=self.E, where=adjacency_mask)
            else:
                self.E += tag_increment

        elif self.eligibility_mode == DendriticEligibilityMode.SPIKE_COINCIDENCE:
            if post_spikes is not None:
                spk_post = post_spikes.astype(np.float32)
                tag_increment = np.outer(spk_pre, spk_post)
                if adjacency_mask is not None:
                    np.add(self.E, tag_increment, out=self.E, where=adjacency_mask)
                else:
                    self.E += tag_increment

        # Mask non-synaptic pairs
        if adjacency_mask is not None:
            self.E[~adjacency_mask] = 0.0

        # Calculate metrics
        active = self.E[adjacency_mask] if adjacency_mask is not None else self.E.ravel()
        if len(active) > 0 and np.any(np.abs(active) > 1e-7):
            abs_act = np.abs(active)
            self.last_mean_eligibility = float(np.mean(abs_act))
            self.last_max_eligibility = float(np.max(abs_act))
            # Normalized Gini coefficient for eligibility concentration
            sorted_e = np.sort(abs_act)
            n = len(sorted_e)
            idx = np.arange(1, n + 1)
            sum_e = np.sum(sorted_e)
            gini = (2.0 * np.sum(idx * sorted_e) / (n * sum_e) - (n + 1.0) / n) if sum_e > 1e-9 else 0.0
            self.last_eligibility_concentration = float(np.clip(gini, 0.0, 1.0))
        else:
            self.last_mean_eligibility = 0.0
            self.last_max_eligibility = 0.0
            self.last_eligibility_concentration = 0.0

        return self.E

    def apply_targeted_plasticity(
        self,
        weights: np.ndarray,               # [pre, post]
        adjacency_mask: np.ndarray,        # [pre, post]
        m_vector: np.ndarray,              # [post]
        relevant_mask: Optional[np.ndarray] = None, # [pre, post] boolean mask of causally relevant synapses
        step: int = 0,
    ) -> Tuple[np.ndarray, List[TargetedUpdateRecord], Dict[str, float]]:
        """Apply targeted feedback modulatory signal to update synaptic weights.

        Equation:
            Delta w_ij(t) = eta * M_j(t) * e_ij(t)
            w_ij(t) = Clip(w_ij(t-1) + Delta w_ij(t), Dale_bounds)
        """
        if self.feedback_mode == FeedbackMode.FROZEN or abs(float(np.mean(np.abs(m_vector)))) < 1e-9:
            return weights, [], {
                "step": step,
                "mean_eligibility": self.last_mean_eligibility,
                "total_dw": 0.0,
                "csi": 0.0,
                "precision": 0.0,
                "leakage": 0.0,
            }

        W = weights.copy()

        # M_matrix is broadcast across post-synaptic neurons: M[j] modulates all incoming synapses to neuron j
        # m_vector is [post] -> M_matrix is [pre, post]
        M_matrix = np.tile(m_vector, (self.num_neurons, 1))

        delta_w = self.eta * M_matrix * self.E
        delta_w[~adjacency_mask] = 0.0

        # Enforce Dale's Principle:
        # Pre-synaptic excitatory: W >= 0, W <= w_max_e
        # Pre-synaptic inhibitory: W <= 0, W >= -w_max_i
        pre_is_exc = self.is_excitatory[:, np.newaxis]
        pre_is_inh = self.is_inhibitory[:, np.newaxis]

        # Candidate new weights
        W_new = W + delta_w
        W_new = np.where(pre_is_exc, np.clip(W_new, self.w_min_e, self.w_max_e), W_new)
        W_new = np.where(pre_is_inh, np.clip(W_new, -self.w_max_i, -self.w_min_i), W_new)
        W_new[~adjacency_mask] = 0.0

        effective_dw = W_new - W

        # Credit Specificity Analysis:
        # Defined BEFORE experiments:
        # Credit Precision = sum_{relevant} |dw| / sum_{all} |dw|
        # Credit Leakage = sum_{irrelevant} |dw| / sum_{all} |dw|
        # CSI = (mean_relevant |dw| - mean_irrelevant |dw|) / (mean_relevant |dw| + mean_irrelevant |dw| + eps)
        abs_dw = np.abs(effective_dw[adjacency_mask])
        total_abs_dw = float(np.sum(abs_dw))
        self.last_total_weight_change = total_abs_dw

        if relevant_mask is not None and total_abs_dw > 1e-9:
            rel_dw = np.abs(effective_dw[relevant_mask & adjacency_mask])
            irrel_mask = (~relevant_mask) & adjacency_mask
            irrel_dw = np.abs(effective_dw[irrel_mask])

            sum_rel = float(np.sum(rel_dw))
            sum_irrel = float(np.sum(irrel_dw))

            precision = sum_rel / (total_abs_dw + 1e-12)
            leakage = sum_irrel / (total_abs_dw + 1e-12)

            mean_rel = float(np.mean(rel_dw)) if len(rel_dw) > 0 else 0.0
            mean_irrel = float(np.mean(irrel_dw)) if len(irrel_dw) > 0 else 0.0

            csi = (mean_rel - mean_irrel) / (mean_rel + mean_irrel + 1e-9)

            self.last_credit_precision = precision
            self.last_credit_leakage = leakage
            self.last_credit_specificity_index = float(np.clip(csi, -1.0, 1.0))
        else:
            self.last_credit_precision = 0.0
            self.last_credit_leakage = 0.0
            self.last_credit_specificity_index = 0.0

        # Construct records
        updates: List[TargetedUpdateRecord] = []
        sig_indices = np.where(np.abs(effective_dw) > 1e-6)
        if len(sig_indices[0]) > 0:
            top_k = min(50, len(sig_indices[0]))
            sorted_idx = np.argsort(-np.abs(effective_dw[sig_indices]))[:top_k]
            for idx in sorted_idx:
                src = int(sig_indices[0][idx])
                dst = int(sig_indices[1][idx])
                dw_val = float(effective_dw[src, dst])
                w_val = float(W_new[src, dst])
                is_rel = bool(relevant_mask[src, dst]) if relevant_mask is not None else False
                comm_id = int(self.module_ids[dst]) if self.module_ids is not None else None
                updates.append(
                    TargetedUpdateRecord(
                        step=step,
                        source=src,
                        target=dst,
                        eligibility=float(self.E[src, dst]),
                        neuromodulator=float(m_vector[dst]),
                        delta_w=dw_val,
                        new_weight=w_val,
                        is_relevant=is_rel,
                        target_community=comm_id,
                    )
                )

        self.recent_updates = updates
        metrics = {
            "step": step,
            "mean_eligibility": self.last_mean_eligibility,
            "max_eligibility": self.last_max_eligibility,
            "total_dw": total_abs_dw,
            "csi": self.last_credit_specificity_index,
            "precision": self.last_credit_precision,
            "leakage": self.last_credit_leakage,
            "concentration": self.last_eligibility_concentration,
        }
        return W_new, updates, metrics
