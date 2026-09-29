"""Biophysically Plausible Spike Encoding & Action Population Mechanisms (Phase 13).

Provides biophysical mechanisms to convert continuous subthreshold contextual states
into discrete, stable, self-sustaining action spike patterns:

S0: Phase 12 Baseline (Analog accumulator / Poisson baseline)
S1: Adaptive Threshold (Neuron-specific dynamic spike threshold)
S2: Adaptive Gain (Activity-dependent membrane & synaptic gain control)
S3: Combined Threshold + Gain Adaptation
S4: Rate Population Coding (Heterogeneous population tuning curves & rate distribution)
S5: Temporal Spike Coding (First-to-spike latency & relative spike phase coding)
S6: Population + Temporal Coding (Spatio-temporal synchronous burst detection)
S7: Winner-Take-All Spiking Competition (Dale-constrained E/I lateral inhibition)
S8: Persistent Action Attractor (Recurrent self-excitation & hysteresis lockout)
S9: Full Biophysical Interface (Integrated best-supported model)
"""

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class SpikeEncodingStrategy(str, Enum):
    """Spike encoding strategies for converting context to action spike representations."""
    S0_BASELINE = "S0_BASELINE"
    S1_ADAPTIVE_THRESHOLD = "S1_ADAPTIVE_THRESHOLD"
    S2_ADAPTIVE_GAIN = "S2_ADAPTIVE_GAIN"
    S3_THRESHOLD_PLUS_GAIN = "S3_THRESHOLD_PLUS_GAIN"
    S4_RATE_POPULATION = "S4_RATE_POPULATION"
    S5_TEMPORAL_CODING = "S5_TEMPORAL_CODING"
    S6_POPULATION_TEMPORAL = "S6_POPULATION_TEMPORAL"
    S7_WTA_COMPETITION = "S7_WTA_COMPETITION"
    S8_PERSISTENT_ATTRACTOR = "S8_PERSISTENT_ATTRACTOR"
    S9_FULL_BIOPHYSICAL = "S9_FULL_BIOPHYSICAL"


class HomeostasisMode(str, Enum):
    """Homeostatic regulation modes for action populations."""
    H0_NONE = "H0_NONE"
    H1_FIRING_RATE = "H1_FIRING_RATE"
    H2_THRESHOLD = "H2_THRESHOLD"
    H3_GAIN = "H3_GAIN"
    H4_COMBINED = "H4_COMBINED"


class BiophysicalSpikeEncoder:
    """Encodes continuous input drives into action population spikes and states.

    Models:
    - Two-compartment somatic integration
    - Axon initial segment threshold adaptation:
        tau_theta * dDelta_theta/dt = -Delta_theta + alpha_theta * spike
        theta_eff = theta_0 + Delta_theta
    - Activity-dependent adaptive gain:
        tau_gain * dg/dt = -g + g_0 / (1 + beta_gain * mean_recent_activity)
    - Population coding with M neurons per action pool
    - Temporal / latency-to-first-spike coding
    - Dale-constrained E/I winner-take-all competition
    - Persistent recurrent attractor dynamics
    """

    def __init__(
        self,
        num_actions: int = 2,
        neurons_per_action: int = 16,
        strategy: SpikeEncodingStrategy = SpikeEncodingStrategy.S9_FULL_BIOPHYSICAL,
        homeostasis: HomeostasisMode = HomeostasisMode.H0_NONE,
        dt_ms: float = 1.0,
        tau_soma_ms: float = 15.0,
        tau_theta_ms: float = 40.0,
        theta_base_mv: float = -50.0,
        theta_increment_mv: float = 5.0,
        v_rest_mv: float = -65.0,
        v_reset_mv: float = -68.0,
        gain_base: float = 1.0,
        tau_gain_ms: float = 50.0,
        w_intra_exc: float = 0.40,
        w_inter_inh: float = 0.55,
        target_rate_hz: float = 20.0,
        seed: int = 42,
    ):
        self.num_actions = max(2, int(num_actions))
        self.neurons_per_action = max(1, int(neurons_per_action))
        self.total_action_neurons = self.num_actions * self.neurons_per_action
        self.strategy = strategy
        self.homeostasis = homeostasis
        self.dt_ms = float(dt_ms)
        self.tau_soma_ms = float(tau_soma_ms)
        self.tau_theta_ms = float(tau_theta_ms)
        self.theta_base_mv = float(theta_base_mv)
        self.theta_increment_mv = float(theta_increment_mv)
        self.v_rest_mv = float(v_rest_mv)
        self.v_reset_mv = float(v_reset_mv)
        self.gain_base = float(gain_base)
        self.tau_gain_ms = float(tau_gain_ms)
        self.w_intra_exc = float(w_intra_exc)
        self.w_inter_inh = float(w_inter_inh)
        self.target_rate_hz = float(target_rate_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Decay constants
        self.alpha_soma = float(np.exp(-self.dt_ms / max(1.0, self.tau_soma_ms)))
        self.alpha_theta = float(np.exp(-self.dt_ms / max(1.0, self.tau_theta_ms)))
        self.alpha_gain = float(np.exp(-self.dt_ms / max(1.0, self.tau_gain_ms)))

        # Heterogeneous tuning: intrinsic thresholds and gains per neuron
        # Action populations [num_actions, neurons_per_action]
        shape = (self.num_actions, self.neurons_per_action)
        self.v_soma = np.full(shape, self.v_rest_mv, dtype=np.float32)
        self.theta_base = self.rng.normal(self.theta_base_mv, 1.5, shape).astype(np.float32)
        self.delta_theta = np.zeros(shape, dtype=np.float32)
        self.gain = np.full(shape, self.gain_base, dtype=np.float32)
        self.last_spikes = np.zeros(shape, dtype=np.float32)
        self.smoothed_rate = np.zeros(shape, dtype=np.float32)

        # Spike history & timing
        self.spike_counts = np.zeros(shape, dtype=np.int32)
        self.time_to_first_spike = np.full(shape, np.inf, dtype=np.float32)
        self.current_time_ms = 0.0

        # Action pool outputs
        self.pool_firing_rates = np.zeros(self.num_actions, dtype=np.float32)
        self.pool_evidence = np.zeros(self.num_actions, dtype=np.float32)
        self.pool_persistence = np.zeros(self.num_actions, dtype=np.float32)
        self.selected_action = 0
        self.winner_margin = 0.0

        # Causal intervention perturbation flags
        self.shuffle_spike_timing = False
        self.shuffle_spike_count = False
        self.randomize_threshold = False
        self.randomize_gain = False
        self.competition_disabled = False
        self.persistence_disabled = False

    def reset(self) -> None:
        """Reset membrane states, thresholds, and timers."""
        self.v_soma.fill(self.v_rest_mv)
        self.delta_theta.fill(0.0)
        self.gain.fill(self.gain_base)
        self.last_spikes.fill(0.0)
        self.smoothed_rate.fill(0.0)
        self.spike_counts.fill(0)
        self.time_to_first_spike.fill(np.inf)
        self.current_time_ms = 0.0
        self.pool_firing_rates.fill(0.0)
        self.pool_evidence.fill(0.0)
        self.pool_persistence.fill(0.0)
        self.selected_action = 0
        self.winner_margin = 0.0

    def step(
        self,
        input_drives: np.ndarray,
        lateral_override: Optional[float] = None,
        recurrent_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute one simulation step of the biophysical spike encoding interface.

        Args:
            input_drives: Continuous feedforward drive per action pool [num_actions]
            lateral_override: Optional override for w_inter_inh
            recurrent_override: Optional override for w_intra_exc

        Returns:
            Dict containing spikes, voltages, thresholds, gains, pool rates, and selected action.
        """
        self.current_time_ms += self.dt_ms
        drives = np.asarray(input_drives, dtype=np.float32)
        if len(drives) != self.num_actions:
            raise ValueError(f"Expected input_drives of length {self.num_actions}, got {len(drives)}")

        # Configure competition weights based on strategy and overrides
        w_exc = self.w_intra_exc if recurrent_override is None else float(recurrent_override)
        w_inh = self.w_inter_inh if lateral_override is None else float(lateral_override)

        if self.competition_disabled:
            w_inh = 0.0
        if self.persistence_disabled:
            w_exc = 0.0

        # Enable features according to strategy
        has_adaptive_theta = self.strategy in (
            SpikeEncodingStrategy.S1_ADAPTIVE_THRESHOLD,
            SpikeEncodingStrategy.S3_THRESHOLD_PLUS_GAIN,
            SpikeEncodingStrategy.S6_POPULATION_TEMPORAL,
            SpikeEncodingStrategy.S8_PERSISTENT_ATTRACTOR,
            SpikeEncodingStrategy.S9_FULL_BIOPHYSICAL,
        )
        has_adaptive_gain = self.strategy in (
            SpikeEncodingStrategy.S2_ADAPTIVE_GAIN,
            SpikeEncodingStrategy.S3_THRESHOLD_PLUS_GAIN,
            SpikeEncodingStrategy.S6_POPULATION_TEMPORAL,
            SpikeEncodingStrategy.S9_FULL_BIOPHYSICAL,
        )
        has_wta_competition = self.strategy in (
            SpikeEncodingStrategy.S7_WTA_COMPETITION,
            SpikeEncodingStrategy.S8_PERSISTENT_ATTRACTOR,
            SpikeEncodingStrategy.S9_FULL_BIOPHYSICAL,
        )
        has_persistence_attractor = self.strategy in (
            SpikeEncodingStrategy.S8_PERSISTENT_ATTRACTOR,
            SpikeEncodingStrategy.S9_FULL_BIOPHYSICAL,
        )

        # Baseline S0: simple linear accumulator with Poisson generation
        if self.strategy == SpikeEncodingStrategy.S0_BASELINE:
            self.v_soma = self.v_soma * self.alpha_soma + (1.0 - self.alpha_soma) * (
                self.v_rest_mv + drives[:, np.newaxis] * 10.0
            )
            # Poisson spike generation
            rate_proxy = np.clip((self.v_soma - self.theta_base_mv) * 2.0, 0.0, 100.0)
            prob_spike = 1.0 - np.exp(-rate_proxy * (self.dt_ms / 1000.0))
            spikes = (self.rng.random(self.v_soma.shape) < prob_spike).astype(np.float32)
            self.last_spikes = spikes
            self.pool_firing_rates = np.mean(spikes, axis=1) / (self.dt_ms / 1000.0)
            self.pool_evidence = drives.copy()
            self.selected_action = int(np.argmax(self.pool_firing_rates))
            sorted_rates = np.sort(self.pool_firing_rates)[::-1]
            self.winner_margin = float(sorted_rates[0] - sorted_rates[1]) if len(sorted_rates) > 1 else float(sorted_rates[0])

            return {
                "spikes": spikes,
                "v_soma": self.v_soma.copy(),
                "theta_eff": self.theta_base.copy(),
                "gain": self.gain.copy(),
                "pool_firing_rates": self.pool_firing_rates.copy(),
                "pool_evidence": self.pool_evidence.copy(),
                "selected_action": self.selected_action,
                "winner_margin": self.winner_margin,
            }

        # 1. Update Gain Adaptation (Divisive normalization & sensitivity control)
        if has_adaptive_gain and not self.randomize_gain:
            # Gain scales inversely with smoothed activity
            target_gain = self.gain_base / (1.0 + 0.05 * self.smoothed_rate)
            self.gain = self.gain * self.alpha_gain + (1.0 - self.alpha_gain) * target_gain
        elif self.randomize_gain:
            self.gain = self.rng.uniform(0.1, 2.5, self.gain.shape).astype(np.float32)

        # 2. Compute Synaptic Drives (Feedforward + Recurrent intra-pool + Lateral cross-inhibition)
        # Feedforward: drives broadcast across neurons within each action pool
        ff_current = drives[:, np.newaxis] * self.gain * 12.0

        # Recurrent intra-pool excitation (Persistent attractor dynamics)
        pool_recent_activity = np.mean(self.last_spikes, axis=1)  # [num_actions]
        if has_persistence_attractor and w_exc > 0.0:
            recurrent_current = (w_exc * pool_recent_activity)[:, np.newaxis] * 15.0
        else:
            recurrent_current = np.zeros_like(ff_current)

        # Lateral inter-pool inhibition (Winner-Take-All competition)
        if has_wta_competition and w_inh > 0.0:
            total_act = np.sum(pool_recent_activity)
            lateral_inhibition_per_pool = w_inh * (total_act - pool_recent_activity)  # [num_actions]
            inh_current = (lateral_inhibition_per_pool[:, np.newaxis]) * 18.0
        else:
            inh_current = np.zeros_like(ff_current)

        total_synaptic_current = ff_current + recurrent_current - inh_current

        # 3. Somatic Membrane Integration
        i_leak = -(self.v_soma - self.v_rest_mv)
        dv = (i_leak + total_synaptic_current) * (self.dt_ms / self.tau_soma_ms)
        self.v_soma += dv

        # 4. Axon Initial Segment (AIS) Adaptive Threshold
        if has_adaptive_theta and not self.randomize_threshold:
            # Decay adaptive threshold increment toward zero
            self.delta_theta *= self.alpha_theta
        elif self.randomize_threshold:
            self.delta_theta = self.rng.uniform(0.0, 15.0, self.delta_theta.shape).astype(np.float32)

        theta_eff = self.theta_base + self.delta_theta

        # 5. Spike Generation & Reset
        spikes = (self.v_soma >= theta_eff).astype(np.float32)

        # Causal timing / count shuffle perturbations
        if self.shuffle_spike_timing:
            # Randomly permute spike array across time/neurons
            flat = spikes.flatten()
            self.rng.shuffle(flat)
            spikes = flat.reshape(spikes.shape)
        elif self.shuffle_spike_count:
            # Shuffle spike counts across action pools
            idx = self.rng.permutation(self.num_actions)
            spikes = spikes[idx]

        # Reset membrane potential on spike & increment adaptive threshold
        self.v_soma[spikes > 0.0] = self.v_reset_mv
        if has_adaptive_theta and not self.randomize_threshold:
            self.delta_theta[spikes > 0.0] += self.theta_increment_mv

        # Track spike timing & counts
        for a in range(self.num_actions):
            for n in range(self.neurons_per_action):
                if spikes[a, n] > 0.0:
                    self.spike_counts[a, n] += 1
                    if math.isinf(self.time_to_first_spike[a, n]):
                        self.time_to_first_spike[a, n] = self.current_time_ms

        self.last_spikes = spikes

        # Smooth firing rates (tau = 20 ms)
        alpha_rate = float(np.exp(-self.dt_ms / 20.0))
        inst_rate = spikes / (self.dt_ms / 1000.0)
        self.smoothed_rate = self.smoothed_rate * alpha_rate + (1.0 - alpha_rate) * inst_rate

        # 6. Apply Homeostatic Control if configured
        self._apply_homeostasis()

        # 7. Action Decoding Strategy
        self.pool_firing_rates = np.mean(self.smoothed_rate, axis=1)

        if self.strategy == SpikeEncodingStrategy.S5_TEMPORAL_CODING:
            # Latency coding: action with shortest time-to-first-spike or highest early spike density
            latencies = np.min(self.time_to_first_spike, axis=1)
            # Evidence inversely proportional to latency
            self.pool_evidence = np.where(
                np.isinf(latencies),
                0.0,
                np.exp(-latencies / 50.0).astype(np.float32)
            )
            # Fallback to rate if no spikes yet
            if np.all(self.pool_evidence == 0.0):
                self.pool_evidence = self.pool_firing_rates.copy()

        elif self.strategy in (
            SpikeEncodingStrategy.S6_POPULATION_TEMPORAL,
            SpikeEncodingStrategy.S9_FULL_BIOPHYSICAL,
        ):
            # Combined population rate and synchronous burst density
            # Population synchronous bursts: fraction of pool neurons spiking simultaneously
            coincidence = np.sum(spikes, axis=1) / self.neurons_per_action
            self.pool_evidence = (
                0.6 * (self.pool_firing_rates / max(1.0, np.max(self.pool_firing_rates) + 1e-6))
                + 0.4 * coincidence
            )
        else:
            # Rate population or standard rate
            self.pool_evidence = self.pool_firing_rates.copy()

        # 8. Action Selection & Decision Margin
        self.selected_action = int(np.argmax(self.pool_evidence))
        sorted_ev = np.sort(self.pool_evidence)[::-1]
        self.winner_margin = float(sorted_ev[0] - sorted_ev[1]) if len(sorted_ev) > 1 else float(sorted_ev[0])

        return {
            "spikes": spikes,
            "v_soma": self.v_soma.copy(),
            "theta_eff": theta_eff.copy(),
            "gain": self.gain.copy(),
            "pool_firing_rates": self.pool_firing_rates.copy(),
            "pool_evidence": self.pool_evidence.copy(),
            "selected_action": self.selected_action,
            "winner_margin": self.winner_margin,
            "spike_counts": self.spike_counts.copy(),
            "time_to_first_spike": self.time_to_first_spike.copy(),
        }

    def _apply_homeostasis(self) -> None:
        """Apply homeostatic plastic regulation on thresholds or gains."""
        if self.homeostasis == HomeostasisMode.H0_NONE:
            return

        rate_error = (self.smoothed_rate - self.target_rate_hz)  # positive if too active

        if self.homeostasis in (HomeostasisMode.H1_FIRING_RATE, HomeostasisMode.H2_THRESHOLD, HomeostasisMode.H4_COMBINED):
            # Target firing rate: shift base threshold up if firing too fast, down if too slow
            d_theta = 0.001 * rate_error
            self.theta_base += d_theta
            self.theta_base = np.clip(self.theta_base, -60.0, -35.0)

        if self.homeostasis in (HomeostasisMode.H3_GAIN, HomeostasisMode.H4_COMBINED):
            # Divisive gain homeostasis: decrease gain if firing too fast
            d_gain = -0.0005 * rate_error
            self.gain = np.clip(self.gain + d_gain, 0.2, 3.0)
