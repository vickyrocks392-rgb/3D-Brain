"""Phase 14 Burst Dynamics Engine: Biologically Grounded Burst Mechanisms (B0–B7).

Provides biophysical mechanisms to convert sustained dendritic and somatic drives
into structured temporal burst events rather than Poisson point-process spikes:

B0: Point-Process Baseline (Standard LIF, 1 spike per threshold crossing)
B1: Simple Burst-Capable Neuron (Doublet/triplet burst trigger mode)
B2: Calcium-Dependent Burst Generation (Intracellular [Ca2+] accumulation drives burst state)
B3: Dendritic Plateau-Triggered Burst (Dendritic depolarization > theta_plateau injects sustained somatic drive)
B4: NMDA + Calcium Coincidence Burst (Simultaneous high NMDA slow conductance + dendritic Ca2+)
B5: Adaptive Burst Threshold (Activity-dependent burst threshold adaptation preventing runaway)
B6: Burst + Population Coding (Bursting ensembles integrated across action pools)
B7: Full Burst/Temporal Population Mechanism (Integrated candidate: coincidence + plateau + adaptive burst + ensemble)

Records structured burst metrics:
- burst onset
- burst length (ms)
- inter-spike interval (ISI ms)
- spikes per burst
- burst frequency (Hz)
- burst amplitude
- dendritic trigger state
- somatic state
- action population receiving burst
"""

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class BurstStrategy(str, Enum):
    """Burst generation strategies across B0–B7."""
    B0_POINT_PROCESS = "B0_POINT_PROCESS"
    B1_SIMPLE_BURST = "B1_SIMPLE_BURST"
    B2_CALCIUM_DEPENDENT = "B2_CALCIUM_DEPENDENT"
    B3_DENDRITIC_PLATEAU = "B3_DENDRITIC_PLATEAU"
    B4_NMDA_CALCIUM_COINCIDENCE = "B4_NMDA_CALCIUM_COINCIDENCE"
    B5_ADAPTIVE_BURST_THRESHOLD = "B5_ADAPTIVE_BURST_THRESHOLD"
    B6_BURST_POPULATION = "B6_BURST_POPULATION"
    B7_FULL_BURST_TEMPORAL = "B7_FULL_BURST_TEMPORAL"


class BurstRecord:
    """Record of a structured temporal burst event."""

    def __init__(
        self,
        pool_id: int,
        neuron_id: int,
        onset_time_ms: float,
        dendritic_trigger_v: float,
        somatic_v: float,
        calcium_level: float,
        nmda_level: float,
    ):
        self.pool_id = pool_id
        self.neuron_id = neuron_id
        self.onset_time_ms = onset_time_ms
        self.dendritic_trigger_v = dendritic_trigger_v
        self.somatic_v = somatic_v
        self.calcium_level = calcium_level
        self.nmda_level = nmda_level
        self.spike_times_ms: List[float] = [onset_time_ms]
        self.isis_ms: List[float] = []
        self.completed = False
        self.length_ms = 0.0
        self.amplitude = 1.0

    def add_spike(self, time_ms: float) -> None:
        last_t = self.spike_times_ms[-1]
        isi = time_ms - last_t
        self.isis_ms.append(isi)
        self.spike_times_ms.append(time_ms)
        self.length_ms = time_ms - self.spike_times_ms[0]

    def finalize(self, current_time_ms: float) -> None:
        self.completed = True
        if len(self.spike_times_ms) > 1:
            self.length_ms = self.spike_times_ms[-1] - self.spike_times_ms[0]
        else:
            self.length_ms = 1.0  # minimal single-spike event duration

    @property
    def spike_count(self) -> int:
        return len(self.spike_times_ms)

    @property
    def mean_isi_ms(self) -> float:
        return float(np.mean(self.isis_ms)) if self.isis_ms else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "neuron_id": self.neuron_id,
            "onset_time_ms": round(float(self.onset_time_ms), 2),
            "length_ms": round(float(self.length_ms), 2),
            "spike_count": self.spike_count,
            "mean_isi_ms": round(self.mean_isi_ms, 2),
            "dendritic_trigger_v": round(float(self.dendritic_trigger_v), 2),
            "somatic_v": round(float(self.somatic_v), 2),
            "calcium_level": round(float(self.calcium_level), 4),
            "nmda_level": round(float(self.nmda_level), 4),
            "amplitude": round(float(self.amplitude), 3),
        }


class BurstDynamicsEngine:
    """Biophysical burst generation and modulation engine for action pools.

    Models:
    - Intracellular calcium accumulation [Ca2+]_i with exponential clearance
    - Dendritic plateau potentials triggering somatic high-frequency bursts
    - Coincidence between NMDA conductance and calcium concentration
    - Adaptive burst thresholds to prevent pathological runaway synchrony
    - Multi-neuron action population pooling
    """

    def __init__(
        self,
        num_actions: int = 2,
        neurons_per_action: int = 16,
        strategy: BurstStrategy = BurstStrategy.B7_FULL_BURST_TEMPORAL,
        dt_ms: float = 1.0,
        tau_soma_ms: float = 15.0,
        tau_ca_ms: float = 80.0,
        alpha_ca: float = 0.25,
        ca_burst_threshold: float = 0.60,
        v_rest_mv: float = -65.0,
        v_reset_mv: float = -68.0,
        theta_base_mv: float = -50.0,
        theta_plateau_mv: float = -42.0,
        burst_refractory_ms: float = 8.0,
        target_burst_rate_hz: float = 5.0,
        seed: int = 42,
    ):
        self.num_actions = max(2, int(num_actions))
        self.neurons_per_action = max(1, int(neurons_per_action))
        self.strategy = strategy
        self.dt_ms = float(dt_ms)
        self.tau_soma_ms = float(tau_soma_ms)
        self.tau_ca_ms = float(tau_ca_ms)
        self.alpha_ca = float(alpha_ca)
        self.ca_burst_threshold = float(ca_burst_threshold)
        self.v_rest_mv = float(v_rest_mv)
        self.v_reset_mv = float(v_reset_mv)
        self.theta_base_mv = float(theta_base_mv)
        self.theta_plateau_mv = float(theta_plateau_mv)
        self.burst_refractory_ms = float(burst_refractory_ms)
        self.target_burst_rate_hz = float(target_burst_rate_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Decay constants
        self.alpha_soma = float(np.exp(-self.dt_ms / max(1.0, self.tau_soma_ms)))
        self.alpha_ca = float(np.exp(-self.dt_ms / max(1.0, self.tau_ca_ms)))

        shape = (self.num_actions, self.neurons_per_action)
        self.v_soma = np.full(shape, self.v_rest_mv, dtype=np.float32)
        self.theta_base = self.rng.normal(self.theta_base_mv, 1.2, shape).astype(np.float32)
        self.delta_burst_theta = np.zeros(shape, dtype=np.float32)

        # Calcium variable [Ca2+]_i
        self.calcium = np.zeros(shape, dtype=np.float32)

        # Active burst tracking
        self.burst_active = np.zeros(shape, dtype=bool)
        self.burst_remaining_spikes = np.zeros(shape, dtype=np.int32)
        self.burst_timer_ms = np.zeros(shape, dtype=np.float32)
        self.time_since_last_burst_ms = np.full(shape, 1000.0, dtype=np.float32)

        # Internal clock and events
        self.current_time_ms = 0.0
        self.active_burst_records: Dict[Tuple[int, int], BurstRecord] = {}
        self.recent_bursts: List[BurstRecord] = []
        self.total_burst_count = 0
        self.total_spike_count = 0

        # Causal intervention flags
        self.burst_ablation_mode = "none"  # "single_spike", "timing_shuffle", "length_shuffle"

    def reset(self) -> None:
        """Reset all membrane states, calcium levels, and burst trackers."""
        self.v_soma.fill(self.v_rest_mv)
        self.delta_burst_theta.fill(0.0)
        self.calcium.fill(0.0)
        self.burst_active.fill(False)
        self.burst_remaining_spikes.fill(0)
        self.burst_timer_ms.fill(0.0)
        self.time_since_last_burst_ms.fill(1000.0)
        self.current_time_ms = 0.0
        self.active_burst_records.clear()
        self.recent_bursts.clear()
        self.total_burst_count = 0
        self.total_spike_count = 0

    def step(
        self,
        input_drives: np.ndarray,
        v_dend: Optional[np.ndarray] = None,
        g_nmda: Optional[np.ndarray] = None,
        lateral_inhibition: float = 0.40,
        recurrent_self_excitation: float = 0.35,
    ) -> Dict[str, Any]:
        """Execute one simulation step of burst-based spike generation.

        Args:
            input_drives: Feedforward drive per action pool [num_actions]
            v_dend: Optional dendritic potentials [num_actions] or [num_actions, neurons]
            g_nmda: Optional slow NMDA conductances [num_actions] or [num_actions, neurons]
            lateral_inhibition: Cross-pool lateral inhibition strength
            recurrent_self_excitation: Intra-pool recurrent excitation strength

        Returns:
            Dict containing spikes, bursts, calcium levels, pool rates, and burst metrics.
        """
        self.current_time_ms += self.dt_ms
        self.time_since_last_burst_ms += self.dt_ms

        drives = np.asarray(input_drives, dtype=np.float32)
        shape = (self.num_actions, self.neurons_per_action)

        # Broadcast dendritic potential and NMDA if provided as pool vectors
        if v_dend is None:
            v_dend_arr = np.full(shape, -60.0, dtype=np.float32)
        elif v_dend.shape == shape:
            v_dend_arr = v_dend.astype(np.float32)
        elif len(v_dend) == self.num_actions:
            v_dend_arr = np.repeat(v_dend[:, np.newaxis], self.neurons_per_action, axis=1).astype(np.float32)
        else:
            v_dend_arr = np.full(shape, float(np.mean(v_dend)), dtype=np.float32)

        if g_nmda is None:
            g_nmda_arr = np.zeros(shape, dtype=np.float32)
        elif g_nmda.shape == shape:
            g_nmda_arr = g_nmda.astype(np.float32)
        elif len(g_nmda) == self.num_actions:
            g_nmda_arr = np.repeat(g_nmda[:, np.newaxis], self.neurons_per_action, axis=1).astype(np.float32)
        else:
            g_nmda_arr = np.full(shape, float(np.mean(g_nmda)), dtype=np.float32)

        # 1. Update Intracellular Calcium Dynamics: tau_Ca dCa/dt = -Ca
        self.calcium *= self.alpha_ca

        # 2. Update Adaptive Burst Thresholds (B5, B7)
        if self.strategy in (BurstStrategy.B5_ADAPTIVE_BURST_THRESHOLD, BurstStrategy.B7_FULL_BURST_TEMPORAL):
            self.delta_burst_theta *= float(np.exp(-self.dt_ms / 60.0))

        eff_threshold = self.theta_base + self.delta_burst_theta

        # 3. Compute Synaptic Currents
        ff_current = drives[:, np.newaxis] * 12.0
        # Dendritic plateau sustained current injection
        plateau_mask = v_dend_arr >= self.theta_plateau_mv
        dend_current = np.where(plateau_mask, (v_dend_arr - self.theta_plateau_mv) * 4.0, 0.0)

        # Lateral inhibition and recurrent excitation
        recent_act = np.mean(self.calcium, axis=1)  # pooled calcium activity
        total_act = np.sum(recent_act)
        inh_per_pool = lateral_inhibition * (total_act - recent_act)
        recurrent_per_pool = recurrent_self_excitation * recent_act

        inh_current = inh_per_pool[:, np.newaxis] * 14.0
        rec_current = recurrent_per_pool[:, np.newaxis] * 10.0

        i_syn = ff_current + dend_current + rec_current - inh_current

        # 4. Somatic Integration
        i_leak = -(self.v_soma - self.v_rest_mv)
        dv = (i_leak + i_syn) * (self.dt_ms / self.tau_soma_ms)
        self.v_soma += dv

        # 5. Evaluate Burst Trigger Conditions
        spikes = np.zeros(shape, dtype=np.float32)
        burst_started = np.zeros(shape, dtype=bool)

        for a in range(self.num_actions):
            for n in range(self.neurons_per_action):
                # If currently inside an active burst: emit subsequent burst spike
                if self.burst_active[a, n]:
                    self.burst_timer_ms[a, n] += self.dt_ms
                    # Fast burst ISI typically 2-3 ms
                    if self.burst_timer_ms[a, n] >= 2.0 and self.burst_remaining_spikes[a, n] > 0:
                        spikes[a, n] = 1.0
                        self.burst_remaining_spikes[a, n] -= 1
                        self.burst_timer_ms[a, n] = 0.0
                        self.v_soma[a, n] = self.v_reset_mv
                        self.calcium[a, n] += self.alpha_ca * 0.4

                        # Update active burst record
                        rec_key = (a, n)
                        if rec_key in self.active_burst_records:
                            self.active_burst_records[rec_key].add_spike(self.current_time_ms)

                        if self.burst_remaining_spikes[a, n] <= 0:
                            self.burst_active[a, n] = False
                            if rec_key in self.active_burst_records:
                                rec = self.active_burst_records.pop(rec_key)
                                rec.finalize(self.current_time_ms)
                                self.recent_bursts.append(rec)
                                if len(self.recent_bursts) > 100:
                                    self.recent_bursts.pop(0)
                    continue

                # Threshold crossing check
                somatic_threshold_crossed = self.v_soma[a, n] >= eff_threshold[a, n]
                if not somatic_threshold_crossed:
                    continue

                # Neuronal spike occurred
                spikes[a, n] = 1.0
                self.v_soma[a, n] = self.v_reset_mv
                self.calcium[a, n] += self.alpha_ca * 1.0

                # Determine whether this event promotes to a BURST
                trigger_burst = False
                burst_len = 2  # default doublet

                if self.strategy == BurstStrategy.B0_POINT_PROCESS:
                    trigger_burst = False

                elif self.strategy == BurstStrategy.B1_SIMPLE_BURST:
                    # Simple burst triggered if drive is high
                    if drives[a] > 0.4:
                        trigger_burst = True
                        burst_len = 2 if drives[a] < 0.8 else 3

                elif self.strategy == BurstStrategy.B2_CALCIUM_DEPENDENT:
                    # Calcium elevation triggers burst
                    if self.calcium[a, n] >= self.ca_burst_threshold:
                        trigger_burst = True
                        burst_len = int(np.clip(2 + int(self.calcium[a, n] * 2.0), 2, 4))

                elif self.strategy == BurstStrategy.B3_DENDRITIC_PLATEAU:
                    # Dendritic plateau depolarization triggers prolonged burst
                    if v_dend_arr[a, n] >= self.theta_plateau_mv:
                        trigger_burst = True
                        burst_len = 3

                elif self.strategy == BurstStrategy.B4_NMDA_CALCIUM_COINCIDENCE:
                    # Coincidence between NMDA slow conductance and dendritic Ca2+
                    if g_nmda_arr[a, n] > 0.25 and self.calcium[a, n] > 0.40:
                        trigger_burst = True
                        burst_len = 3

                elif self.strategy in (
                    BurstStrategy.B5_ADAPTIVE_BURST_THRESHOLD,
                    BurstStrategy.B6_BURST_POPULATION,
                    BurstStrategy.B7_FULL_BURST_TEMPORAL,
                ):
                    # Integrated conditions: Ca2+ + plateau/NMDA drive with refractory gate
                    refractory_ok = self.time_since_last_burst_ms[a, n] >= self.burst_refractory_ms
                    coincidence_ok = (
                        (v_dend_arr[a, n] >= self.theta_plateau_mv)
                        or (g_nmda_arr[a, n] > 0.20 and self.calcium[a, n] > 0.35)
                        or (drives[a] > 0.55)
                    )
                    if refractory_ok and coincidence_ok:
                        trigger_burst = True
                        # Burst length scales with combined evidence
                        burst_len = 2 if drives[a] < 0.7 else 3
                        # Adapt burst threshold to prevent runaway
                        self.delta_burst_theta[a, n] += 4.0

                if trigger_burst:
                    burst_started[a, n] = True
                    self.burst_active[a, n] = True
                    self.burst_remaining_spikes[a, n] = burst_len - 1  # 1st spike emitted now
                    self.burst_timer_ms[a, n] = 0.0
                    self.time_since_last_burst_ms[a, n] = 0.0
                    self.total_burst_count += 1

                    # Record burst event
                    rec = BurstRecord(
                        pool_id=a,
                        neuron_id=n,
                        onset_time_ms=self.current_time_ms,
                        dendritic_trigger_v=float(v_dend_arr[a, n]),
                        somatic_v=float(self.v_soma[a, n]),
                        calcium_level=float(self.calcium[a, n]),
                        nmda_level=float(g_nmda_arr[a, n]),
                    )
                    self.active_burst_records[(a, n)] = rec

        # 6. Apply Causal Burst Interventions if active
        if self.burst_ablation_mode == "single_spike":
            # Strip extra spikes from burst, reducing everything to point processes
            self.burst_active.fill(False)
            self.burst_remaining_spikes.fill(0)
        elif self.burst_ablation_mode == "timing_shuffle":
            flat = spikes.flatten()
            self.rng.shuffle(flat)
            spikes = flat.reshape(spikes.shape)

        self.total_spike_count += int(np.sum(spikes))

        # 7. Action Pool Rates and Burst Evidence
        pool_spike_rates = np.mean(spikes, axis=1) / (self.dt_ms / 1000.0)
        pool_burst_counts = np.sum(burst_started, axis=1)
        pool_calcium = np.mean(self.calcium, axis=1)

        # In B6 and B7, burst density provides direct high-gain evidence
        if self.strategy in (BurstStrategy.B6_BURST_POPULATION, BurstStrategy.B7_FULL_BURST_TEMPORAL):
            pool_evidence = (
                0.40 * (pool_spike_rates / max(1.0, np.max(pool_spike_rates) + 1e-5))
                + 0.40 * (pool_burst_counts / max(1.0, np.max(pool_burst_counts) + 1e-5))
                + 0.20 * (pool_calcium / max(1.0, np.max(pool_calcium) + 1e-5))
            )
        else:
            pool_evidence = pool_spike_rates / max(1.0, np.max(pool_spike_rates) + 1e-5)

        selected_action = int(np.argmax(pool_evidence))
        sorted_ev = np.sort(pool_evidence)[::-1]
        winner_margin = float(sorted_ev[0] - sorted_ev[1]) if len(sorted_ev) > 1 else float(sorted_ev[0])

        return {
            "spikes": spikes,
            "v_soma": self.v_soma.copy(),
            "calcium": self.calcium.copy(),
            "burst_started": burst_started,
            "burst_active": self.burst_active.copy(),
            "pool_spike_rates": pool_spike_rates,
            "pool_burst_counts": pool_burst_counts,
            "pool_evidence": pool_evidence,
            "selected_action": selected_action,
            "winner_margin": winner_margin,
            "total_burst_count": self.total_burst_count,
            "total_spike_count": self.total_spike_count,
        }

    def get_recent_burst_statistics(self) -> Dict[str, float]:
        """Compute aggregated statistics over recently recorded bursts."""
        if not self.recent_bursts:
            return {
                "burst_count": 0,
                "mean_burst_length_ms": 0.0,
                "mean_spikes_per_burst": 0.0,
                "mean_isi_ms": 0.0,
                "burst_frequency_hz": 0.0,
            }

        lengths = [b.length_ms for b in self.recent_bursts]
        spk_counts = [b.spike_count for b in self.recent_bursts]
        isis = [b.mean_isi_ms for b in self.recent_bursts if b.isis_ms]

        total_time_s = max(0.001, self.current_time_ms / 1000.0)
        burst_freq = len(self.recent_bursts) / total_time_s

        return {
            "burst_count": len(self.recent_bursts),
            "mean_burst_length_ms": float(np.mean(lengths)),
            "mean_spikes_per_burst": float(np.mean(spk_counts)),
            "mean_isi_ms": float(np.mean(isis)) if isis else 0.0,
            "burst_frequency_hz": float(burst_freq),
        }
