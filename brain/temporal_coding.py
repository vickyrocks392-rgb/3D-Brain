"""Phase 14 Temporal Coding Engine: Spike Timing, Burst Structure & Temporal Ensembles (T0–T8).

Explicitly implements and compares 9 temporal coding paradigms:
T0: Instantaneous spike snapshot (single-step binary vector)
T1: Spike count over temporal integration window
T2: Latency-to-first-spike (first spike timing relative to cue/epoch onset)
T3: Inter-spike interval (ISI) code (distribution of intervals between consecutive spikes)
T4: Burst structure (burst counts, spikes-per-burst, intra-burst frequency)
T5: Relative spike timing (rank-order and pairwise temporal precedence across neurons)
T6: Phase-of-firing code (spike timing relative to population oscillatory cycle [0, 2*pi])
T7: Population temporal sequence (spatiotemporal raster trajectory across window)
T8: Combined burst + phase + population temporal code

Provides decoding tools, mutual information estimation, linear/nonlinear probe classification,
temporal window sweeps (1 to 100 steps), timing jitter perturbations, and spike dropout robustness.
"""

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class TemporalCodingScheme(str, Enum):
    """Temporal coding representations across T0–T8."""
    T0_INSTANTANEOUS = "T0_INSTANTANEOUS"
    T1_SPIKE_COUNT = "T1_SPIKE_COUNT"
    T2_LATENCY = "T2_LATENCY"
    T3_ISI = "T3_ISI"
    T4_BURST_STRUCTURE = "T4_BURST_STRUCTURE"
    T5_RELATIVE_TIMING = "T5_RELATIVE_TIMING"
    T6_PHASE_CODE = "T6_PHASE_CODE"
    T7_POPULATION_SEQUENCE = "T7_POPULATION_SEQUENCE"
    T8_COMBINED = "T8_COMBINED"


class TemporalSpikeDecoder:
    """Decodes representations from spike rasters and extracts temporal features."""

    def __init__(
        self,
        num_neurons: int = 32,
        default_window_steps: int = 10,
        dt_ms: float = 1.0,
        seed: int = 42,
    ):
        self.num_neurons = int(num_neurons)
        self.default_window_steps = int(default_window_steps)
        self.dt_ms = float(dt_ms)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def extract_features(
        self,
        spike_raster: np.ndarray,
        scheme: TemporalCodingScheme,
        window_steps: Optional[int] = None,
        oscillation_phase: Optional[float] = None,
        burst_data: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Extract feature representation from spatiotemporal raster [time_steps, num_neurons].

        Args:
            spike_raster: Binary or float array of shape [T, N]
            scheme: Selected TemporalCodingScheme (T0..T8)
            window_steps: Integration window (defaults to self.default_window_steps)
            oscillation_phase: Current population phase in [0, 2*pi]
            burst_data: Optional dictionary containing burst metrics

        Returns:
            1D feature vector for linear/nonlinear decoding.
        """
        raster = np.asarray(spike_raster, dtype=np.float32)
        if raster.ndim == 1:
            raster = raster[np.newaxis, :]  # [1, N]
        T, N = raster.shape

        W = window_steps if window_steps is not None else min(T, self.default_window_steps)
        W = max(1, min(T, W))
        recent_raster = raster[-W:, :]  # [W, N]

        # T0: Instantaneous spike state (last step only)
        if scheme == TemporalCodingScheme.T0_INSTANTANEOUS:
            return recent_raster[-1, :].copy()

        # T1: Spike count over fixed window
        elif scheme == TemporalCodingScheme.T1_SPIKE_COUNT:
            return np.sum(recent_raster, axis=0)  # [N]

        # T2: Latency-to-first-spike
        elif scheme == TemporalCodingScheme.T2_LATENCY:
            latencies = np.full(N, float(W), dtype=np.float32)
            for n in range(N):
                spk_idx = np.where(recent_raster[:, n] > 0.0)[0]
                if len(spk_idx) > 0:
                    latencies[n] = float(spk_idx[0])
            # Invert latency so early spikes have high positive evidence
            return np.exp(-latencies / max(1.0, W / 2.0))

        # T3: Inter-spike interval (ISI) code
        elif scheme == TemporalCodingScheme.T3_ISI:
            mean_isis = np.zeros(N, dtype=np.float32)
            for n in range(N):
                spk_idx = np.where(recent_raster[:, n] > 0.0)[0]
                if len(spk_idx) >= 2:
                    diffs = np.diff(spk_idx)
                    mean_isis[n] = float(np.mean(diffs))
                elif len(spk_idx) == 1:
                    mean_isis[n] = float(W)
                else:
                    mean_isis[n] = 0.0
            return mean_isis

        # T4: Burst structure (burst counts and intra-burst density)
        elif scheme == TemporalCodingScheme.T4_BURST_STRUCTURE:
            burst_counts = np.zeros(N, dtype=np.float32)
            intra_burst_density = np.zeros(N, dtype=np.float32)
            for n in range(N):
                spk_idx = np.where(recent_raster[:, n] > 0.0)[0]
                if len(spk_idx) >= 2:
                    diffs = np.diff(spk_idx)
                    # Bursts characterized by short ISIs <= 3 steps
                    burst_spikes = np.sum(diffs <= 3)
                    burst_counts[n] = float(burst_spikes)
                    intra_burst_density[n] = float(burst_spikes) / max(1, len(spk_idx))
            return np.concatenate([burst_counts, intra_burst_density])

        # T5: Relative spike timing (rank-order coding across neurons)
        elif scheme == TemporalCodingScheme.T5_RELATIVE_TIMING:
            first_spikes = np.full(N, 999.0, dtype=np.float32)
            for n in range(N):
                spk_idx = np.where(recent_raster[:, n] > 0.0)[0]
                if len(spk_idx) > 0:
                    first_spikes[n] = float(spk_idx[0])
            # Rank order normalized: earliest spike has rank 1.0, latest 0.0
            ranks = np.zeros(N, dtype=np.float32)
            valid = first_spikes < 999.0
            if np.any(valid):
                sorted_idx = np.argsort(first_spikes)
                for r, idx in enumerate(sorted_idx):
                    if valid[idx]:
                        ranks[idx] = 1.0 - (r / max(1, N))
            return ranks

        # T6: Phase code (timing relative to oscillation cycle)
        elif scheme == TemporalCodingScheme.T6_PHASE_CODE:
            phase = float(oscillation_phase) if oscillation_phase is not None else 0.0
            phase_vec = np.zeros(N, dtype=np.float32)
            for n in range(N):
                spk_idx = np.where(recent_raster[:, n] > 0.0)[0]
                if len(spk_idx) > 0:
                    # Effective phase of last spike
                    last_step = spk_idx[-1]
                    relative_time = (W - 1 - last_step) * self.dt_ms
                    spk_phase = (phase - (relative_time * 0.1)) % (2.0 * math.pi)
                    phase_vec[n] = math.cos(spk_phase)
            return phase_vec

        # T7: Population temporal sequence (spatiotemporal raster trajectory)
        elif scheme == TemporalCodingScheme.T7_POPULATION_SEQUENCE:
            # Flatten recent spatiotemporal raster [W * N]
            return recent_raster.flatten()

        # T8: Combined burst + phase + population code
        elif scheme == TemporalCodingScheme.T8_COMBINED:
            f_count = np.sum(recent_raster, axis=0) / max(1.0, float(W))
            # Latencies
            lat = np.full(N, float(W), dtype=np.float32)
            for n in range(N):
                spk_idx = np.where(recent_raster[:, n] > 0.0)[0]
                if len(spk_idx) > 0:
                    lat[n] = float(spk_idx[0])
            f_lat = np.exp(-lat / max(1.0, W / 2.0))

            # Bursts
            b_counts = np.zeros(N, dtype=np.float32)
            for n in range(N):
                spk_idx = np.where(recent_raster[:, n] > 0.0)[0]
                if len(spk_idx) >= 2:
                    b_counts[n] = float(np.sum(np.diff(spk_idx) <= 3))

            phase = float(oscillation_phase) if oscillation_phase is not None else 0.0
            f_phase = np.zeros(N, dtype=np.float32)
            for n in range(N):
                spk_idx = np.where(recent_raster[:, n] > 0.0)[0]
                if len(spk_idx) > 0:
                    spk_phase = (phase - (W - 1 - spk_idx[-1]) * 0.1) % (2.0 * math.pi)
                    f_phase[n] = math.cos(spk_phase)

            return np.concatenate([f_count, f_lat, b_counts, f_phase])

        return recent_raster[-1, :].copy()

    def apply_jitter(self, spike_raster: np.ndarray, jitter_fraction: float) -> np.ndarray:
        """Introduce temporal jitter by shifting spike times by +/- dt * jitter_fraction."""
        if jitter_fraction <= 0.0:
            return spike_raster.copy()

        raster = np.asarray(spike_raster, dtype=np.float32)
        T, N = raster.shape
        jittered = np.zeros_like(raster)

        max_shift = max(1, int(round(T * jitter_fraction)))
        for n in range(N):
            spk_indices = np.where(raster[:, n] > 0.0)[0]
            for t in spk_indices:
                shift = self.rng.integers(-max_shift, max_shift + 1)
                new_t = np.clip(t + shift, 0, T - 1)
                jittered[new_t, n] = 1.0

        return jittered

    def apply_dropout(self, spike_raster: np.ndarray, dropout_fraction: float) -> np.ndarray:
        """Randomly drop spikes with probability dropout_fraction."""
        if dropout_fraction <= 0.0:
            return spike_raster.copy()

        raster = np.asarray(spike_raster, dtype=np.float32).copy()
        mask = self.rng.random(raster.shape) < dropout_fraction
        raster[mask] = 0.0
        return raster
