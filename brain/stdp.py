"""Spike-Timing-Dependent Plasticity (STDP) and Homeostatic Synaptic Regulation.

Implements the formal spike-based plasticity mechanisms for Phase 7:
- Asymmetric pair-based STDP (pre-before-post LTP, post-before-pre LTD)
- Exponential pre- and post-synaptic eligibility traces
- Hard and soft bounded weights to guarantee numerical and biological stability
- Homeostatic synaptic scaling and target firing rate regulation
- Neuromodulated STDP gated by novelty/error signals
- Multiple plasticity control regimes (Frozen, STDP, STDP+Homeostasis, STDP+Neuromodulation)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class PlasticityMode(str, Enum):
    FROZEN = "FROZEN"
    STDP = "STDP"
    STDP_HOMEOSTASIS = "STDP_HOMEOSTASIS"
    STDP_NEUROMODULATION = "STDP_NEUROMODULATION"


@dataclass
class STDPUpdateRecord:
    """Detailed record of an individual synaptic weight change for verification."""
    step: int
    source: int
    target: int
    delta_w: float
    new_weight: float
    rule: str
    dt_pair: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "step": int(self.step),
            "src": int(self.source),
            "dst": int(self.target),
            "dw": round(float(self.delta_w), 5),
            "w": round(float(self.new_weight), 4),
            "rule": self.rule,
        }
        if self.dt_pair is not None:
            d["dt_pair"] = round(float(self.dt_pair), 2)
        return d


class STDPEngine:
    """High-performance vectorized STDP engine with homeostatic and neuromodulatory controls."""

    def __init__(
        self,
        num_neurons: int,
        mode: PlasticityMode = PlasticityMode.STDP,
        a_plus: float = 0.015,         # Maximum potentiation step
        a_minus: float = 0.018,        # Maximum depression step (a_minus > a_plus ensures stability)
        tau_plus: float = 20.0,        # Pre-synaptic trace decay time constant (ms)
        tau_minus: float = 25.0,       # Post-synaptic trace decay time constant (ms)
        w_min: float = 0.0,            # Minimum synaptic weight
        w_max: float = 2.0,            # Maximum synaptic weight
        target_rate: float = 0.05,     # Target spikes/step for homeostatic regulation
        tau_homeo: float = 200.0,      # Moving average window for firing rate estimation
        eta_homeo: float = 0.001,      # Rate of homeostatic weight adjustment
        max_sum_weight: float = 10.0,  # Max sum of incoming weights per post-synaptic neuron
        track_updates: bool = True,
    ):
        self.num_neurons = num_neurons
        self.mode = mode
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.tau_plus = max(1.0, float(tau_plus))
        self.tau_minus = max(1.0, float(tau_minus))
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.target_rate = float(target_rate)
        self.tau_homeo = max(10.0, float(tau_homeo))
        self.eta_homeo = float(eta_homeo)
        self.max_sum_weight = float(max_sum_weight)
        self.track_updates = track_updates

        # Synaptic traces [N]
        self.x_pre = np.zeros(num_neurons, dtype=np.float32)
        self.y_post = np.zeros(num_neurons, dtype=np.float32)

        # Homeostatic moving average firing rates [N]
        self.rate_estimates = np.full(num_neurons, target_rate, dtype=np.float32)

        # Telemetry records
        self.recent_updates: List[STDPUpdateRecord] = []
        self.total_plastic_events = 0

    def reset(self) -> None:
        self.x_pre.fill(0.0)
        self.y_post.fill(0.0)
        self.rate_estimates.fill(self.target_rate)
        self.recent_updates.clear()
        self.total_plastic_events = 0

    def step_plasticity(
        self,
        weights: np.ndarray,            # [N, N] dense or masked adjacency weights
        adjacency_mask: np.ndarray,     # [N, N] boolean mask of existing synaptic connections
        spikes: np.ndarray,             # [N] boolean spikes emitted at current step
        step: int,
        neuromodulator: float = 1.0,    # Neuromodulatory gain M(t) in [0.0, 2.0]
        dt: float = 1.0,
    ) -> Tuple[np.ndarray, List[STDPUpdateRecord]]:
        """Advance eligibility traces, apply STDP pair updates, and enforce homeostasis.

        Returns:
            updated_weights: [N, N] new weight matrix
            updates: list of recorded weight changes
        """
        if self.mode == PlasticityMode.FROZEN:
            # Synaptic plasticity completely disabled (control condition)
            return weights, []

        # 1. Exponential decay of pre- and post-synaptic traces
        decay_pre = np.exp(-dt / self.tau_plus)
        decay_post = np.exp(-dt / self.tau_minus)
        self.x_pre *= decay_pre
        self.y_post *= decay_post

        # 2. Update homeostatic moving average rate
        decay_rate = np.exp(-dt / self.tau_homeo)
        self.rate_estimates = decay_rate * self.rate_estimates + (1.0 - decay_rate) * spikes.astype(np.float32)

        new_weights = weights.copy()
        current_records: List[STDPUpdateRecord] = []

        mod_factor = 1.0
        if self.mode == PlasticityMode.STDP_NEUROMODULATION:
            mod_factor = float(np.clip(neuromodulator, 0.0, 3.0))

        fired_indices = np.where(spikes)[0]

        # 3. Depression (LTD): Pre-synaptic spike occurs -> depressed by target post-trace y_post
        for pre in fired_indices:
            # Outgoing connections from this pre-synaptic unit
            post_targets = np.where(adjacency_mask[pre, :])[0]
            if len(post_targets) > 0:
                y_vals = self.y_post[post_targets]
                # Soft-bounded multiplicative LTD: delta_w = -A_- * y * (W - w_min)
                dw = -self.a_minus * y_vals * (new_weights[pre, post_targets] - self.w_min) * mod_factor
                new_weights[pre, post_targets] = np.clip(
                    new_weights[pre, post_targets] + dw, self.w_min, self.w_max
                )
                if self.track_updates:
                    for idx, post in enumerate(post_targets):
                        if abs(dw[idx]) > 1e-4:
                            rec = STDPUpdateRecord(
                                step=step,
                                source=int(pre),
                                target=int(post),
                                delta_w=float(dw[idx]),
                                new_weight=float(new_weights[pre, post]),
                                rule="LTD",
                            )
                            current_records.append(rec)

        # 4. Potentiation (LTP): Post-synaptic spike occurs -> potentiated by source pre-trace x_pre
        for post in fired_indices:
            # Incoming connections into this post-synaptic unit
            pre_sources = np.where(adjacency_mask[:, post])[0]
            if len(pre_sources) > 0:
                x_vals = self.x_pre[pre_sources]
                # Soft-bounded multiplicative LTP: delta_w = +A_+ * x * (w_max - W)
                dw = self.a_plus * x_vals * (self.w_max - new_weights[pre_sources, post]) * mod_factor
                new_weights[pre_sources, post] = np.clip(
                    new_weights[pre_sources, post] + dw, self.w_min, self.w_max
                )
                if self.track_updates:
                    for idx, pre in enumerate(pre_sources):
                        if abs(dw[idx]) > 1e-4:
                            rec = STDPUpdateRecord(
                                step=step,
                                source=int(pre),
                                target=int(post),
                                delta_w=float(dw[idx]),
                                new_weight=float(new_weights[pre, post]),
                                rule="LTP",
                            )
                            current_records.append(rec)

        # 5. Increment traces on spike
        self.x_pre[fired_indices] += 1.0
        self.y_post[fired_indices] += 1.0

        # 6. Homeostatic Synaptic Regulation
        if self.mode == PlasticityMode.STDP_HOMEOSTASIS:
            # Target rate regulation on post-synaptic units
            rate_diff = self.target_rate - self.rate_estimates  # Positive if firing too low -> scale up
            homeo_dw = self.eta_homeo * rate_diff  # [N]
            # Apply to incoming connections: W[:, j] += homeo_dw[j]
            for j in range(self.num_neurons):
                in_conn = adjacency_mask[:, j]
                if np.any(in_conn):
                    new_weights[in_conn, j] = np.clip(
                        new_weights[in_conn, j] + homeo_dw[j], self.w_min, self.w_max
                    )

            # Synaptic scaling normalization: ensure sum of incoming weights does not exceed max_sum_weight
            in_sums = np.sum(new_weights * adjacency_mask, axis=0)  # [N]
            over_budget = in_sums > self.max_sum_weight
            for j in np.where(over_budget)[0]:
                scale = self.max_sum_weight / max(1e-6, in_sums[j])
                new_weights[:, j] *= scale

        # Ensure exact zeros on non-existent connections
        new_weights *= adjacency_mask

        self.total_plastic_events += len(current_records)
        self.recent_updates = current_records[-100:]  # Keep latest 100 updates for telemetry
        return new_weights, current_records
