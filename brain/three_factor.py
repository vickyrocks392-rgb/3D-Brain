"""Three-Factor Synaptic Plasticity Engine for 3D-Brain (Phase 9).

Implements the formal mathematical substrate for credit assignment in recurrent spiking networks:
- Local eligibility traces: e_ij(t) = lambda_e * e_ij(t-1) + f(pre_i, post_j)
- Neuromodulatory teaching factor: M(t) in [Global Reward, Prediction Error, RPE, Local/Spatial]
- Synaptic weight update: Delta w_ij(t) = eta * M(t) * e_ij(t)
- Strict Dale's Principle enforcement: W_E >= 0, W_I <= 0, with upper bounds
- Dual Hebbian / STDP-based eligibility accumulation options
- Spatial community-specific neuromodulation routing
- Comprehensive telemetry capturing pre, post, eligibility, neuromodulator, and weight updates
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class NeuromodulatorMode(str, Enum):
    GLOBAL_REWARD = "GLOBAL_REWARD"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    REWARD_PREDICTION_ERROR = "REWARD_PREDICTION_ERROR"  # Signed RPE: M(t) = R(t) - V_bar(t)
    LOCAL_COMMUNITY = "LOCAL_COMMUNITY"                  # Community-localized modulation
    CONSTANT = "CONSTANT"                                # M(t) = 1.0 (eligibility-only control)
    FROZEN = "FROZEN"                                    # M(t) = 0.0 (no plasticity)


class EligibilityTraceMode(str, Enum):
    COINCIDENCE = "COINCIDENCE"      # f(pre, post) = pre * post
    STDP_ASYNCHRONOUS = "STDP"       # f(pre, post) = A+ * x_pre * post - A- * y_post * pre
    ZERO = "ZERO"                    # e_ij = 0 (ablation: no eligibility)


@dataclass
class ThreeFactorUpdateRecord:
    """Detailed record of a three-factor synaptic weight update for verification and replay."""
    step: int
    source: int
    target: int
    eligibility: float
    neuromodulator: float
    delta_w: float
    new_weight: float
    rule: str = "THREE_FACTOR"
    community_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "step": int(self.step),
            "src": int(self.source),
            "dst": int(self.target),
            "eligibility": round(float(self.eligibility), 6),
            "neuromodulator": round(float(self.neuromodulator), 5),
            "dw": round(float(self.delta_w), 6),
            "w": round(float(self.new_weight), 5),
            "rule": self.rule,
        }
        if self.community_id is not None:
            d["community_id"] = int(self.community_id)
        return d


class ThreeFactorPlasticityEngine:
    """Three-Factor Synaptic Plasticity Engine implementing local eligibility and neuromodulation."""

    def __init__(
        self,
        num_neurons: int,
        learning_rate: float = 0.005,
        tau_eligibility_ms: float = 40.0,
        eligibility_mode: EligibilityTraceMode = EligibilityTraceMode.COINCIDENCE,
        neuromodulator_mode: NeuromodulatorMode = NeuromodulatorMode.REWARD_PREDICTION_ERROR,
        rpe_baseline_alpha: float = 0.05,
        w_min_e: float = 0.0,
        w_max_e: float = 2.5,
        w_min_i: float = 0.0,
        w_max_i: float = 4.0,
        is_inhibitory: Optional[np.ndarray] = None,
        module_ids: Optional[np.ndarray] = None,
        dt_ms: float = 1.0,
        track_updates: bool = True,
        max_tracked_updates: int = 50,
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.eta = float(learning_rate)
        self.tau_e = max(1.0, float(tau_eligibility_ms))
        self.eligibility_mode = eligibility_mode
        self.neuromodulator_mode = neuromodulator_mode
        self.rpe_baseline_alpha = float(np.clip(rpe_baseline_alpha, 0.001, 1.0))
        self.dt_ms = float(dt_ms)
        self.track_updates = track_updates
        self.max_tracked_updates = max_tracked_updates
        self.rng = np.random.default_rng(seed)

        # Decay factor per simulation step: lambda_e = exp(-dt / tau_e)
        self.lambda_e = float(np.exp(-self.dt_ms / self.tau_e))

        # Boundaries respecting Dale's Principle
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

        # Full eligibility trace matrix: E[src, dst] or E[pre, post]
        self.E = np.zeros((num_neurons, num_neurons), dtype=np.float32)

        # Fast pre- and post-synaptic traces for STDP-derived eligibility
        self.x_pre = np.zeros(num_neurons, dtype=np.float32)
        self.y_post = np.zeros(num_neurons, dtype=np.float32)
        self.tau_plus = 20.0
        self.tau_minus = 25.0
        self.lambda_plus = float(np.exp(-self.dt_ms / self.tau_plus))
        self.lambda_minus = float(np.exp(-self.dt_ms / self.tau_minus))
        self.a_plus = 0.02
        self.a_minus = 0.024

        # Moving reward baseline for signed RPE: RPE(t) = R(t) - V_bar(t)
        self.reward_baseline = 0.0
        self.last_rpe = 0.0
        self.last_raw_reward = 0.0

        # Telemetry
        self.recent_updates: List[ThreeFactorUpdateRecord] = []
        self.total_plastic_events = 0
        self.last_mean_eligibility = 0.0
        self.last_max_eligibility = 0.0
        self.last_total_weight_change = 0.0

    def reset(self) -> None:
        """Reset internal traces and baselines."""
        self.E.fill(0.0)
        self.x_pre.fill(0.0)
        self.y_post.fill(0.0)
        self.reward_baseline = 0.0
        self.last_rpe = 0.0
        self.last_raw_reward = 0.0
        self.recent_updates.clear()
        self.total_plastic_events = 0
        self.last_mean_eligibility = 0.0
        self.last_max_eligibility = 0.0
        self.last_total_weight_change = 0.0

    def compute_neuromodulatory_signal(
        self,
        reward: float = 0.0,
        prediction_error: float = 0.0,
        community_rewards: Optional[Dict[int, float]] = None,
    ) -> Union[float, np.ndarray]:
        """Compute the effective third-factor neuromodulatory signal M(t).

        Returns:
            Scalar float for global modulation, or [N] array for community-localized modulation.
        """
        self.last_raw_reward = float(reward)

        if self.neuromodulator_mode == NeuromodulatorMode.FROZEN:
            return 0.0

        if self.neuromodulator_mode == NeuromodulatorMode.CONSTANT:
            return 1.0

        if self.neuromodulator_mode == NeuromodulatorMode.GLOBAL_REWARD:
            return float(reward)

        if self.neuromodulator_mode == NeuromodulatorMode.PREDICTION_ERROR:
            # Positive prediction error drives plasticity: error signal
            return float(prediction_error)

        if self.neuromodulator_mode == NeuromodulatorMode.REWARD_PREDICTION_ERROR:
            # Signed RPE: delta = R(t) - V_bar(t)
            rpe = float(reward) - self.reward_baseline
            # Update running baseline: V_bar <- (1 - alpha) * V_bar + alpha * R
            self.reward_baseline = (
                (1.0 - self.rpe_baseline_alpha) * self.reward_baseline
                + self.rpe_baseline_alpha * float(reward)
            )
            self.last_rpe = rpe
            return rpe

        if self.neuromodulator_mode == NeuromodulatorMode.LOCAL_COMMUNITY:
            # Vector of modulation values per neuron based on its community
            m_vec = np.full(self.num_neurons, float(reward), dtype=np.float32)
            if community_rewards is not None and self.module_ids is not None:
                for c_id, c_rew in community_rewards.items():
                    mask = (self.module_ids == c_id)
                    m_vec[mask] = float(c_rew)
            return m_vec

        return float(reward)

    def update_eligibility(
        self,
        spikes: np.ndarray,
        adjacency_mask: np.ndarray,  # [pre, post] boolean mask
    ) -> np.ndarray:
        """Update local synaptic eligibility traces e_ij(t).

        Equation:
            e_ij(t) = lambda_e * e_ij(t-1) + f(pre_i, post_j)
        where f is gated strictly by structural synaptic connectivity.
        """
        spk_float = spikes.astype(np.float32)  # [N]

        if self.eligibility_mode == EligibilityTraceMode.ZERO:
            self.E.fill(0.0)
            self.last_mean_eligibility = 0.0
            self.last_max_eligibility = 0.0
            return self.E

        # 1. Exponential decay of existing eligibility traces
        self.E *= self.lambda_e

        # 2. Activity-dependent increment f(pre, post)
        if self.eligibility_mode == EligibilityTraceMode.COINCIDENCE:
            # Outer product of pre- and post-synaptic spike events
            coincidence = np.outer(spk_float, spk_float)  # [pre, post]
            np.add(self.E, coincidence, out=self.E, where=adjacency_mask)

        elif self.eligibility_mode == EligibilityTraceMode.STDP_ASYNCHRONOUS:
            # Update pre- and post-synaptic traces
            self.x_pre = self.lambda_plus * self.x_pre + spk_float
            self.y_post = self.lambda_minus * self.y_post + spk_float

            # LTP increment: post-spike arrives while pre trace is high
            ltp_increment = self.a_plus * np.outer(self.x_pre, spk_float)
            # LTD increment: pre-spike arrives while post trace is high
            ltd_increment = self.a_minus * np.outer(spk_float, self.y_post)

            stdp_step = ltp_increment - ltd_increment
            np.add(self.E, stdp_step, out=self.E, where=adjacency_mask)

        # Mask disconnected synapses to strictly prevent non-local phantom traces
        self.E[~adjacency_mask] = 0.0

        # Telemetry metrics
        active_traces = self.E[adjacency_mask]
        if len(active_traces) > 0:
            self.last_mean_eligibility = float(np.mean(np.abs(active_traces)))
            self.last_max_eligibility = float(np.max(np.abs(active_traces)))
        else:
            self.last_mean_eligibility = 0.0
            self.last_max_eligibility = 0.0

        return self.E

    def apply_plasticity(
        self,
        weights: np.ndarray,        # [pre, post] synaptic weight matrix
        adjacency_mask: np.ndarray, # [pre, post] boolean mask
        neuromodulator: Union[float, np.ndarray],
        step: int = 0,
    ) -> Tuple[np.ndarray, List[ThreeFactorUpdateRecord]]:
        """Apply the third-factor neuromodulatory signal to update synaptic weights.

        Equation:
            Delta w_ij(t) = eta * M(t) * e_ij(t)
            w_ij(t) = Clip(w_ij(t-1) + Delta w_ij(t), Dale_bounds)

        Returns:
            updated_weights: [pre, post] modified weights
            records: list of recent updates
        """
        if self.neuromodulator_mode == NeuromodulatorMode.FROZEN or abs(float(np.mean(neuromodulator))) < 1e-9:
            return weights, []

        W = weights.copy()
        updates: List[ThreeFactorUpdateRecord] = []

        # Compute Delta W = eta * M * E
        if isinstance(neuromodulator, np.ndarray):
            # Neuromodulator varies by post-synaptic neuron: M is [post] or [pre, post]
            if neuromodulator.ndim == 1:
                M_matrix = np.tile(neuromodulator, (self.num_neurons, 1))
            else:
                M_matrix = neuromodulator
            delta_w = self.eta * M_matrix * self.E
        else:
            delta_w = (self.eta * float(neuromodulator)) * self.E

        # Only modify existing synapses
        delta_w[~adjacency_mask] = 0.0

        # Dale's Principle Constraints:
        # Excitatory pre-synaptic neurons (is_excitatory[pre] == True):
        #   weights must stay >= 0, bounded by [0, w_max_e]
        # Inhibitory pre-synaptic neurons (is_inhibitory[pre] == True):
        #   weights must stay <= 0, bounded by [-w_max_i, 0]
        pre_is_exc = self.is_excitatory[:, np.newaxis]
        pre_is_inh = self.is_inhibitory[:, np.newaxis]

        # Apply updates
        new_W = W + delta_w

        # Clip Excitatory synapses
        new_W = np.where(
            pre_is_exc & adjacency_mask,
            np.clip(new_W, self.w_min_e, self.w_max_e),
            new_W,
        )
        # Clip Inhibitory synapses
        new_W = np.where(
            pre_is_inh & adjacency_mask,
            np.clip(new_W, -self.w_max_i, -self.w_min_i),
            new_W,
        )

        effective_dw = new_W - W
        self.last_total_weight_change = float(np.sum(np.abs(effective_dw)))

        # Track representative updates
        if self.track_updates:
            active_srcs, active_dsts = np.where(np.abs(effective_dw) > 1e-6)
            num_updates = len(active_srcs)
            self.total_plastic_events += num_updates

            if num_updates > 0:
                # Sample up to max_tracked_updates
                indices = np.arange(num_updates)
                if num_updates > self.max_tracked_updates:
                    indices = self.rng.choice(num_updates, size=self.max_tracked_updates, replace=False)

                for idx in indices:
                    s = int(active_srcs[idx])
                    d = int(active_dsts[idx])
                    mod_val = (
                        float(neuromodulator[d])
                        if isinstance(neuromodulator, np.ndarray) and neuromodulator.ndim == 1
                        else float(np.mean(neuromodulator))
                    )
                    comm = int(self.module_ids[d]) if self.module_ids is not None else None
                    rec = ThreeFactorUpdateRecord(
                        step=step,
                        source=s,
                        target=d,
                        eligibility=float(self.E[s, d]),
                        neuromodulator=mod_val,
                        delta_w=float(effective_dw[s, d]),
                        new_weight=float(new_W[s, d]),
                        community_id=comm,
                    )
                    updates.append(rec)

        self.recent_updates = updates
        return new_W, updates

    def step(
        self,
        weights: np.ndarray,        # [pre, post]
        adjacency_mask: np.ndarray, # [pre, post]
        spikes: np.ndarray,         # [N]
        reward: float = 0.0,
        prediction_error: float = 0.0,
        community_rewards: Optional[Dict[int, float]] = None,
        step: int = 0,
    ) -> Tuple[np.ndarray, List[ThreeFactorUpdateRecord], Dict[str, Any]]:
        """Full single-step execution: advance eligibility, compute neuromodulator, apply plasticity.

        Returns:
            updated_weights: [pre, post]
            updates: recorded weight changes
            telemetry: step telemetry dictionary
        """
        # 1. Update eligibility traces from current pre/post spikes
        self.update_eligibility(spikes, adjacency_mask)

        # 2. Compute neuromodulatory signal M(t)
        M = self.compute_neuromodulatory_signal(
            reward=reward,
            prediction_error=prediction_error,
            community_rewards=community_rewards,
        )

        # 3. Apply credit-assignment weight update
        updated_weights, updates = self.apply_plasticity(
            weights=weights,
            adjacency_mask=adjacency_mask,
            neuromodulator=M,
            step=step,
        )

        telemetry = {
            "step": step,
            "mean_eligibility": self.last_mean_eligibility,
            "max_eligibility": self.last_max_eligibility,
            "neuromodulator": float(np.mean(M)),
            "raw_reward": self.last_raw_reward,
            "rpe": self.last_rpe,
            "rpe_baseline": self.reward_baseline,
            "total_weight_change": self.last_total_weight_change,
            "num_synaptic_updates": len(updates),
        }

        return updated_weights, updates, telemetry
