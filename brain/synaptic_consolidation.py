"""Synaptic Consolidation & Continual Learning Subsystem for 3D-Brain.

Implements biologically grounded, synapse-local consolidation mechanisms to protect
previously learned context-action representations from catastrophic forgetting:

1. Baseline (NONE): Standard single-timescale plastic readout (Phase 18 baseline).
2. Dual-Timescale Synapses (DUAL_TIMESCALE): Fast plastic component (AMPA/early-LTP)
   coupled to a slow consolidated component (structural/late-LTP).
3. Synaptic Tagging and Capture (SYNAPTIC_TAGGING): Local tags set by dendritic eligibility;
   systemic reward triggers synthesis/capture into a protected consolidated matrix.
4. Metaplasticity Cascade (METAPLASTICITY): Multi-state stability cascade where
   reinforced synapses become progressively more resistant to overwriting.
5. Local Elastic Stabilization (LOCAL_ELASTIC): Synapse-local precision-weighted anchor
   restoring forces without global gradient computation.

All mechanisms are strictly local to individual synapses and require NO backpropagation,
NO global task ID oracle, and NO explicit dataset replay buffers.
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple
import numpy as np


class ConsolidationMode(str, Enum):
    """Preregistered candidate synaptic consolidation mechanisms."""
    NONE = "NONE"
    DUAL_TIMESCALE = "DUAL_TIMESCALE"
    SYNAPTIC_TAGGING = "SYNAPTIC_TAGGING"
    METAPLASTICITY = "METAPLASTICITY"
    LOCAL_ELASTIC = "LOCAL_ELASTIC"


class SynapticConsolidationModule:
    """Manages synaptic consolidation state and plasticity modulation for ActionReadout."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        mode: ConsolidationMode = ConsolidationMode.NONE,
        tau_slow: float = 20.0,
        fast_decay: float = 0.002,
        tag_threshold: float = 0.15,
        tag_decay: float = 0.85,
        tag_transfer_rate: float = 0.25,
        metaplasticity_beta: float = 6.0,
        stability_increment: float = 0.40,
        elastic_lambda: float = 0.08,
        seed: int = 42,
    ):
        self.input_dim = int(input_dim)
        self.num_actions = int(num_actions)
        self.mode = mode
        self.tau_slow = float(tau_slow)
        self.fast_decay = float(fast_decay)
        self.tag_threshold = float(tag_threshold)
        self.tag_decay = float(tag_decay)
        self.tag_transfer_rate = float(tag_transfer_rate)
        self.metaplasticity_beta = float(metaplasticity_beta)
        self.stability_increment = float(stability_increment)
        self.elastic_lambda = float(elastic_lambda)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # State tensors (shape: [input_dim, num_actions])
        self.w_fast = np.zeros((self.input_dim, self.num_actions), dtype=np.float32)
        self.w_slow = np.zeros((self.input_dim, self.num_actions), dtype=np.float32)
        self.tags = np.zeros((self.input_dim, self.num_actions), dtype=np.float32)
        self.w_cons = np.zeros((self.input_dim, self.num_actions), dtype=np.float32)
        self.stability = np.zeros((self.input_dim, self.num_actions), dtype=np.float32)
        self.w_anchor = np.zeros((self.input_dim, self.num_actions), dtype=np.float32)
        self.omega = np.zeros((self.input_dim, self.num_actions), dtype=np.float32)

        # Inspection and telemetry tracking
        self.cumulative_turnover = 0.0
        self.last_turnover = 0.0
        self.last_stabilized_count = 0
        self.last_mean_stability = 0.0

    def reset(self) -> None:
        """Reset transient dynamic registers per trial while preserving consolidated state."""
        self.tags.fill(0.0)
        self.last_turnover = 0.0

    def reset_all(self) -> None:
        """Complete reset of all consolidation matrices."""
        self.w_fast.fill(0.0)
        self.w_slow.fill(0.0)
        self.tags.fill(0.0)
        self.w_cons.fill(0.0)
        self.stability.fill(0.0)
        self.w_anchor.fill(0.0)
        self.omega.fill(0.0)
        self.cumulative_turnover = 0.0
        self.last_turnover = 0.0
        self.last_stabilized_count = 0
        self.last_mean_stability = 0.0

    def compute_effective_weights(self, base_weights: np.ndarray) -> np.ndarray:
        """Combine base synaptic weights with consolidated components for forward computation."""
        if self.mode == ConsolidationMode.NONE:
            return base_weights

        elif self.mode == ConsolidationMode.DUAL_TIMESCALE:
            # W_eff = base_weights (fast) + W_slow (consolidated structural)
            return base_weights + self.w_slow

        elif self.mode == ConsolidationMode.SYNAPTIC_TAGGING:
            # W_eff = base_weights (plastic) + W_cons (tag-consolidated)
            return base_weights + self.w_cons

        elif self.mode in (ConsolidationMode.METAPLASTICITY, ConsolidationMode.LOCAL_ELASTIC):
            return base_weights

        return base_weights

    def modulate_plasticity_step(
        self,
        base_weights: np.ndarray,
        raw_delta_w: np.ndarray,
        eligibility: np.ndarray,
        teaching_signal: np.ndarray,
        reward: float,
        is_reward_step: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply consolidation-specific dynamics to synaptic weight updates."""
        effective_delta_w = raw_delta_w.copy()
        stabilized_synapses = 0
        mean_stab = 0.0

        if self.mode == ConsolidationMode.NONE:
            # Standard single-timescale plasticity (no modification)
            pass

        elif self.mode == ConsolidationMode.DUAL_TIMESCALE:
            # Fast weights receive plastic updates with slight passive decay
            effective_delta_w = raw_delta_w - self.fast_decay * base_weights
            
            # Slow weights consolidate sustained base weights
            # Delta W_slow = (1 / tau_slow) * base_weights * (reward > 0)
            if is_reward_step and reward > 0.0:
                delta_slow = (1.0 / self.tau_slow) * base_weights * float(reward)
                self.w_slow += delta_slow
                self.w_slow = np.clip(self.w_slow, -1.8, 1.8)

            stabilized_synapses = int(np.sum(np.abs(self.w_slow) >= 0.05))
            mean_stab = float(np.mean(np.abs(self.w_slow)))

        elif self.mode == ConsolidationMode.SYNAPTIC_TAGGING:
            # 1. Update local synaptic tags based on dendritic eligibility
            self.tags = self.tag_decay * self.tags + (1.0 - self.tag_decay) * np.abs(eligibility)
            
            # 2. Reward-gated capture into consolidated store
            if is_reward_step and reward > 0.0:
                tagged_mask = (self.tags >= self.tag_threshold).astype(np.float32)
                # Transfer fraction of plastic weights to consolidated store
                delta_cons = self.tag_transfer_rate * reward * tagged_mask * base_weights
                self.w_cons += delta_cons
                self.w_cons = np.clip(self.w_cons, -2.0, 2.0)
                # Reset captured tags
                self.tags *= (1.0 - tagged_mask * 0.7)

            # Consolidated weights resist opposing sign plastic updates
            resistance = 1.0 / (1.0 + 3.0 * np.abs(self.w_cons))
            effective_delta_w = raw_delta_w * resistance
            
            stabilized_synapses = int(np.sum(np.abs(self.w_cons) > 0.10))
            mean_stab = float(np.mean(np.abs(self.w_cons)))

        elif self.mode == ConsolidationMode.METAPLASTICITY:
            # Multi-state cascade / stability counter
            if is_reward_step and reward > 0.0:
                # Reinforce stability where eligibility was active and outcome was positive
                reinforcement = np.abs(eligibility) * float(reward)
                self.stability += self.stability_increment * reinforcement * 5.0
                self.stability = np.clip(self.stability, 0.0, 30.0)

            # Scale effective learning rate inversely by stability
            # eta_eff = eta_0 / (1 + beta * stability)
            resistance_factor = 1.0 / (1.0 + self.metaplasticity_beta * self.stability)
            effective_delta_w = raw_delta_w * resistance_factor
            
            stabilized_synapses = int(np.sum(self.stability > 1.0))
            mean_stab = float(np.mean(self.stability))

        elif self.mode == ConsolidationMode.LOCAL_ELASTIC:
            # Precision accumulator: omega += (delta_w)^2
            if is_reward_step and reward > 0.0:
                self.omega += np.square(raw_delta_w) * float(reward)
                self.w_anchor = 0.90 * self.w_anchor + 0.10 * base_weights

            # Local restoring force back to anchor
            restoring_force = -self.elastic_lambda * self.omega * (base_weights - self.w_anchor)
            effective_delta_w = raw_delta_w + restoring_force
            
            stabilized_synapses = int(np.sum(self.omega > 0.05))
            mean_stab = float(np.mean(self.omega))

        # Turnover calculation
        turnover = float(np.sum(np.abs(effective_delta_w)))
        self.last_turnover = turnover
        self.cumulative_turnover += turnover
        self.last_stabilized_count = stabilized_synapses
        self.last_mean_stability = mean_stab

        telemetry = {
            "mode": self.mode.value,
            "stabilized_synapses": stabilized_synapses,
            "mean_stability": mean_stab,
            "last_turnover": turnover,
            "cumulative_turnover": float(self.cumulative_turnover),
            "max_slow_weight": float(np.max(np.abs(self.w_slow))) if self.mode == ConsolidationMode.DUAL_TIMESCALE else 0.0,
            "max_cons_weight": float(np.max(np.abs(self.w_cons))) if self.mode == ConsolidationMode.SYNAPTIC_TAGGING else 0.0,
            "mean_tag": float(np.mean(self.tags)),
        }
        return effective_delta_w, telemetry

    def observable_state(self) -> Dict[str, Any]:
        """Return inspectable consolidation state dictionary for telemetry and BrainLog."""
        return {
            "mode": self.mode.value,
            "stabilized_synapses": self.last_stabilized_count,
            "mean_stability": round(self.last_mean_stability, 4),
            "cumulative_turnover": round(self.cumulative_turnover, 4),
            "w_slow": self.w_slow.tolist(),
            "w_cons": self.w_cons.tolist(),
            "stability": self.stability.tolist(),
            "tags": self.tags.tolist(),
        }
