"""Spiking Neural Dynamics, Neuron Populations, and Delayed Spike Propagation.

Implements the formal spiking computational substrate for Phase 7:
- Canonical Spike Event representation with causal traceability
- Leaky Integrate-and-Fire (LIF) neuron population with explicit membrane dynamics, threshold crossing, and refractory period
- Izhikevich (IZH) two-variable neuron population supporting regular spiking, fast spiking, and bursting regimes
- SpikePropagationQueue: Circular ring-buffer delivering discrete synaptic weight pulses with spatial conduction delays
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class SpikingNeuronModel(str, Enum):
    LIF = "LIF"
    IZHIKEVICH = "IZHIKEVICH"


@dataclass
class SpikeEvent:
    """Canonical discrete action potential event record."""
    source: int
    target: int
    emission_step: int
    arrival_step: int
    weight: float
    delay: int
    distance: float = 1.0
    spike_id: int = 0

    def progress(self, current_step: int) -> float:
        """Normalized spatial propagation progress along axon in [0.0, 1.0]."""
        if self.delay <= 0:
            return 1.0
        elapsed = current_step - self.emission_step
        return float(np.clip(elapsed / self.delay, 0.0, 1.0))

    def to_dict(self, current_step: int) -> dict:
        return {
            "src": int(self.source),
            "dst": int(self.target),
            "emission_step": int(self.emission_step),
            "arrival_step": int(self.arrival_step),
            "weight": round(float(self.weight), 4),
            "delay": int(self.delay),
            "distance": round(float(self.distance), 2),
            "progress": round(self.progress(current_step), 2),
            "spike_id": int(self.spike_id),
        }


class LIFNeuronPopulation:
    """Leaky Integrate-and-Fire (LIF) neuron population with explicit biophysical variables."""

    def __init__(
        self,
        num_neurons: int,
        v_rest: float = -70.0,      # Resting membrane potential (mV)
        v_reset: float = -75.0,     # Post-spike reset potential (mV)
        v_thresh: float = -50.0,    # Action potential threshold (mV)
        tau_m: float = 20.0,        # Membrane time constant (ms)
        r_m: float = 1.0,           # Membrane resistance (MOhm)
        refractory_period: int = 3, # Absolute refractory period (timesteps / ms)
        noise_std: float = 0.5,     # Subthreshold membrane voltage noise
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.v_thresh = float(v_thresh)
        self.tau_m = max(1.0, float(tau_m))
        self.r_m = float(r_m)
        self.refractory_period = max(0, int(refractory_period))
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

        # State arrays [N]
        self.v = np.full(num_neurons, self.v_rest, dtype=np.float32)
        self.refractory_remaining = np.zeros(num_neurons, dtype=np.int32)
        self.spikes = np.zeros(num_neurons, dtype=bool)
        self.last_spike_step = np.full(num_neurons, -1000, dtype=np.int32)
        self.spike_counts = np.zeros(num_neurons, dtype=np.int64)

    def reset(self) -> None:
        """Reset population membrane states to resting potential."""
        self.v.fill(self.v_rest)
        self.refractory_remaining.fill(0)
        self.spikes.fill(False)
        self.last_spike_step.fill(-1000)
        self.spike_counts.fill(0)

    def step(self, i_syn: np.ndarray, dt: float = 1.0, current_step: int = 0) -> np.ndarray:
        """Advance LIF membrane state by one simulation timestep dt.

        Args:
            i_syn: Synaptic input current vector of shape [N] (nA or arbitrary current units)
            dt: Simulation timestep duration (ms)
            current_step: Current global simulation timestep integer

        Returns:
            spikes: Boolean array of shape [N], True where a spike was emitted
        """
        # 1. Decrement refractory period timer
        in_refractory = self.refractory_remaining > 0
        self.refractory_remaining[in_refractory] -= 1

        # Clamped voltage for refractory neurons
        self.v[in_refractory] = self.v_reset

        # 2. Integrate passive leak and synaptic drive for non-refractory neurons
        non_refractory = ~in_refractory
        if np.any(non_refractory):
            noise = self.rng.normal(0.0, self.noise_std, size=self.num_neurons).astype(np.float32)
            # dv/dt = -(v - v_rest) / tau_m + (r_m * i_syn) / tau_m
            leak = (self.v_rest - self.v[non_refractory]) / self.tau_m
            drive = (self.r_m * i_syn[non_refractory]) / self.tau_m
            self.v[non_refractory] += dt * (leak + drive) + noise[non_refractory]

        # 3. Detect threshold crossings
        fired = non_refractory & (self.v >= self.v_thresh)

        # 4. Apply action potential reset and enter refractory state
        if np.any(fired):
            self.v[fired] = self.v_reset
            self.refractory_remaining[fired] = self.refractory_period
            self.last_spike_step[fired] = current_step
            self.spike_counts[fired] += 1

        self.spikes = fired
        return fired

    def get_state_dict(self) -> Dict[str, Any]:
        """Return verifiable snapshot of membrane potential and refractory states."""
        return {
            "model": "LIF",
            "v_mean": float(np.mean(self.v)),
            "v_std": float(np.std(self.v)),
            "refractory_count": int(np.sum(self.refractory_remaining > 0)),
            "spikes_active": int(np.sum(self.spikes)),
            "total_spikes": int(np.sum(self.spike_counts)),
        }


class IzhikevichNeuronPopulation:
    """Izhikevich two-variable dynamical system neuron population.

    dv/dt = 0.04 v^2 + 5 v + 140 - u + I
    du/dt = a (b v - u)
    if v >= 30 mV, then v <- c, u <- u + d
    """

    def __init__(
        self,
        num_neurons: int,
        a: float = 0.02,        # Time scale of recovery variable u
        b: float = 0.20,        # Sensitivity of recovery variable u to subthreshold fluctuations
        c: float = -65.0,       # After-spike reset value of membrane potential v (mV)
        d: float = 8.0,         # After-spike reset increment of recovery variable u
        v_peak: float = 30.0,   # Spike cutoff threshold (mV)
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.v_peak = float(v_peak)
        self.rng = np.random.default_rng(seed)

        # State arrays [N]
        self.v = np.full(num_neurons, self.c, dtype=np.float32)
        self.u = (self.b * self.v).astype(np.float32)
        self.spikes = np.zeros(num_neurons, dtype=bool)
        self.last_spike_step = np.full(num_neurons, -1000, dtype=np.int32)
        self.spike_counts = np.zeros(num_neurons, dtype=np.int64)

    def reset(self) -> None:
        self.v.fill(self.c)
        self.u = (self.b * self.v).astype(np.float32)
        self.spikes.fill(False)
        self.last_spike_step.fill(-1000)
        self.spike_counts.fill(0)

    def step(self, i_syn: np.ndarray, dt: float = 1.0, current_step: int = 0) -> np.ndarray:
        """Advance Izhikevich population state by one timestep using 0.5ms sub-stepping."""
        # Two sub-steps per ms for numerical stability of quadratic term
        half_dt = 0.5 * dt
        for _ in range(2):
            dv = 0.04 * (self.v ** 2) + 5.0 * self.v + 140.0 - self.u + i_syn
            self.v += half_dt * dv
            du = self.a * (self.b * self.v - self.u)
            self.u += half_dt * du

        # Detect spike cutoff
        fired = self.v >= self.v_peak
        if np.any(fired):
            self.v[fired] = self.c
            self.u[fired] += self.d
            self.last_spike_step[fired] = current_step
            self.spike_counts[fired] += 1

        self.spikes = fired
        return fired

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "model": "IZHIKEVICH",
            "v_mean": float(np.mean(self.v)),
            "v_std": float(np.std(self.v)),
            "u_mean": float(np.mean(self.u)),
            "spikes_active": int(np.sum(self.spikes)),
            "total_spikes": int(np.sum(self.spike_counts)),
        }


class SpikePropagationQueue:
    """Circular ring buffer queue delivering discrete spike arrivals with conduction latencies."""

    def __init__(self, num_neurons: int, max_delay: int = 50, track_active_spikes: bool = True):
        self.num_neurons = num_neurons
        self.max_delay = max(1, max_delay)
        self.buffer_size = self.max_delay + 2
        self.track_active_spikes = track_active_spikes

        # Ring buffer accumulating post-synaptic current pulses arriving at future simulation steps
        # Shape: [buffer_size, num_neurons]
        self.ring_buffer = np.zeros((self.buffer_size, num_neurons), dtype=np.float32)
        self.current_step = 0
        self._spike_counter = 0

        # Detailed in-transit spike records for 3D visualization and observability
        self.active_spikes: List[SpikeEvent] = []

    def reset(self) -> None:
        self.ring_buffer.fill(0.0)
        self.current_step = 0
        self._spike_counter = 0
        self.active_spikes.clear()

    def advance_step(self) -> np.ndarray:
        """Advance time by one simulation step and return arrived synaptic currents."""
        slot = self.current_step % self.buffer_size
        arrived_currents = self.ring_buffer[slot].copy()
        # Clear consumed slot for future circular cycles
        self.ring_buffer[slot].fill(0.0)
        self.current_step += 1

        # Prune delivered in-transit spikes
        if self.track_active_spikes:
            self.active_spikes = [s for s in self.active_spikes if s.arrival_step >= self.current_step]

        return arrived_currents

    def enqueue_spikes(
        self,
        src_indices: np.ndarray,
        dst_indices: np.ndarray,
        weights: np.ndarray,
        delays: np.ndarray,
        distances: Optional[np.ndarray] = None,
    ) -> List[SpikeEvent]:
        """Enqueue discrete spike pulses originating from firing pre-synaptic units.

        Args:
            src_indices: Pre-synaptic firing neuron IDs [K]
            dst_indices: Post-synaptic target neuron IDs [K]
            weights: Synaptic connection weights [K]
            delays: Axonal propagation delays in simulation steps [K]
            distances: Pairwise physical distances [K]
        """
        if len(src_indices) == 0:
            return []

        arrival_steps = self.current_step + delays
        slots = arrival_steps % self.buffer_size

        # Vectorized accumulation into circular ring buffer
        np.add.at(self.ring_buffer, (slots, dst_indices), weights)

        new_events: List[SpikeEvent] = []
        if self.track_active_spikes:
            for k in range(len(src_indices)):
                self._spike_counter += 1
                dist = float(distances[k]) if distances is not None else 1.0
                event = SpikeEvent(
                    source=int(src_indices[k]),
                    target=int(dst_indices[k]),
                    emission_step=self.current_step,
                    arrival_step=int(arrival_steps[k]),
                    weight=float(weights[k]),
                    delay=int(delays[k]),
                    distance=dist,
                    spike_id=self._spike_counter,
                )
                self.active_spikes.append(event)
                new_events.append(event)

        return new_events

    def get_active_spikes(self, max_count: int = 150) -> List[SpikeEvent]:
        """Return subset of currently propagating spikes for inspection and visualization."""
        return self.active_spikes[:max_count]
