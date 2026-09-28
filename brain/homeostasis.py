"""Homeostatic Synaptic Scaling Engine for 3D-Brain (Phase 8).

Implements biologically grounded, activity-dependent multiplicative synaptic scaling (Turrigiano et al.):
- Maintains target firing rate distributions across excitatory and inhibitory populations
- Operates on slow homeostatic timescales (tau_homeo >> tau_membrane)
- Scales synaptic weights multiplicatively while strictly preserving Dale's sign constraint and boundaries
- Supports both local (neuron-specific) and global (population-averaged) scaling modes for causal ablations
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class HomeostaticSynapticScaling:
    """Multiplicative homeostatic synaptic scaling engine."""

    def __init__(
        self,
        num_neurons: int,
        target_rate_hz: float = 10.0,
        tau_homeo_ms: float = 200.0,
        eta_homeo: float = 0.005,
        w_max_e: float = 3.0,
        w_max_i: float = 5.0,
        dt_ms: float = 1.0,
        is_inhibitory: Optional[np.ndarray] = None,
        mode: str = "local",  # "local" or "global"
        enabled: bool = True,
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.target_rate_hz = float(target_rate_hz)
        self.target_rate_per_step = (self.target_rate_hz * dt_ms) / 1000.0
        self.tau_homeo_ms = max(10.0, float(tau_homeo_ms))
        self.eta_homeo = float(eta_homeo)
        self.w_max_e = float(w_max_e)
        self.w_max_i = float(w_max_i)
        self.dt_ms = float(dt_ms)
        self.mode = mode.lower()
        self.enabled = bool(enabled)
        self.rng = np.random.default_rng(seed)

        if is_inhibitory is None:
            self.is_inhibitory = np.zeros(num_neurons, dtype=bool)
        else:
            self.is_inhibitory = np.asarray(is_inhibitory, dtype=bool)
        self.is_excitatory = ~self.is_inhibitory

        # Exponential moving average filter factor
        self.alpha = float(np.clip(self.dt_ms / self.tau_homeo_ms, 1e-4, 1.0))

        # Filtered estimated firing rates per step [N]
        self.estimated_rate = np.full(num_neurons, self.target_rate_per_step, dtype=np.float32)
        # Current multiplicative scale factor per neuron
        self.scaling_factors = np.ones(num_neurons, dtype=np.float32)

        self.total_updates = 0
        self.last_mean_scale = 1.0

    def reset(self) -> None:
        self.estimated_rate.fill(self.target_rate_per_step)
        self.scaling_factors.fill(1.0)
        self.total_updates = 0
        self.last_mean_scale = 1.0

    def update(
        self,
        spikes: np.ndarray,
        W_rec: np.ndarray,
        step: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Update firing rate estimates and apply multiplicative homeostatic scaling to incoming weights.
        
        Args:
            spikes: Boolean or binary array of emitted spikes [N]
            W_rec: Recurrent synaptic matrix [N, N] where W_rec[dst, src] is synaptic weight
            step: Current simulation step
            
        Returns:
            W_rec_scaled: Updated synaptic weight matrix
            telemetry: Dictionary containing homeostatic metrics
        """
        if not self.enabled:
            return W_rec, {
                "enabled": False,
                "mean_rate_hz": float(np.mean(spikes)) * (1000.0 / self.dt_ms),
                "mean_scale_factor": 1.0,
                "max_scale_factor": 1.0,
                "min_scale_factor": 1.0,
            }

        spike_float = spikes.astype(np.float32)
        # Update estimated firing rate via causal leaky filter
        self.estimated_rate = (1.0 - self.alpha) * self.estimated_rate + self.alpha * spike_float
        current_rates_hz = self.estimated_rate * (1000.0 / self.dt_ms)

        denom = max(self.target_rate_per_step, 1e-6)
        if self.mode == "global":
            # Global scaling: compute population-level discrepancy
            pop_rate = float(np.mean(self.estimated_rate))
            rel_err = (self.target_rate_per_step - pop_rate) / denom
            scale_delta = self.eta_homeo * rel_err
            mult_factors = np.full(self.num_neurons, float(np.clip(1.0 + scale_delta, 0.5, 2.0)))
        else:
            # Local scaling: neuron-specific scaling based on post-synaptic activity
            rel_errs = (self.target_rate_per_step - self.estimated_rate) / denom
            scale_deltas = self.eta_homeo * rel_errs
            mult_factors = np.clip(1.0 + scale_deltas, 0.5, 2.0).astype(np.float32)

        self.scaling_factors = mult_factors
        self.last_mean_scale = float(np.mean(mult_factors))
        self.total_updates += 1

        # Apply scaling to incoming weights: row dst is scaled by mult_factors[dst]
        # W_rec[dst, :] *= mult_factors[dst]
        W_scaled = W_rec * mult_factors[:, np.newaxis]

        # Enforce bounds and strictly preserve Dale's signs
        for src in range(self.num_neurons):
            if self.is_inhibitory[src]:
                # Inhibitory columns must remain non-positive in [-w_max_i, 0]
                W_scaled[:, src] = np.clip(W_scaled[:, src], -self.w_max_i, 0.0)
            else:
                # Excitatory columns must remain non-negative in [0, w_max_e]
                W_scaled[:, src] = np.clip(W_scaled[:, src], 0.0, self.w_max_e)

        telemetry = {
            "enabled": True,
            "mode": self.mode,
            "mean_rate_hz": round(float(np.mean(current_rates_hz)), 2),
            "target_rate_hz": self.target_rate_hz,
            "mean_scale_factor": round(float(np.mean(mult_factors)), 6),
            "min_scale_factor": round(float(np.min(mult_factors)), 6),
            "max_scale_factor": round(float(np.max(mult_factors)), 6),
            "e_rate_hz": round(float(np.mean(current_rates_hz[self.is_excitatory])), 2) if np.any(self.is_excitatory) else 0.0,
            "i_rate_hz": round(float(np.mean(current_rates_hz[self.is_inhibitory])), 2) if np.any(self.is_inhibitory) else 0.0,
        }

        return W_scaled, telemetry
