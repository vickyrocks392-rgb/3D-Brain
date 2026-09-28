"""Excitation/Inhibition (E/I) Dynamics, Dale's Principle, and Fast Inhibitory Interneurons.

Implements the formal biophysical and structural substrate for Phase 8:
- NeuronType: Strict cell-type identification (EXCITATORY vs. INHIBITORY)
- Dale's Principle: Cell-specific synaptic signs (E neurons emit positive drive, I neurons emit negative drive)
- Fast-Spiking Inhibitory Interneuron dynamics (shorter tau_m, lower threshold, faster refractory recovery)
- EILIFPopulation and EIIzhikevichPopulation supporting heterogeneous cellular dynamics
- EISpikePropagationQueue: Separately accumulating and accounting for excitatory (I_E) and inhibitory (I_I) currents
- Controlled E->E, E->I, I->E, I->I connection matrices
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.spiking import SpikeEvent


class NeuronType(str, Enum):
    EXCITATORY = "EXCITATORY"
    INHIBITORY = "INHIBITORY"


class EILIFPopulation:
    """Heterogeneous Leaky Integrate-and-Fire population respecting Dale's cell typing.
    
    Excitatory neurons: standard regular integration (tau_m ~ 20ms, refr ~ 3ms)
    Inhibitory interneurons: fast-spiking basket-cell dynamics (tau_m ~ 10ms, refr ~ 1ms, sharper response)
    """

    def __init__(
        self,
        num_neurons: int,
        inhibitory_fraction: float = 0.20,
        e_tau_m: float = 20.0,
        i_tau_m: float = 10.0,
        v_rest: float = -70.0,
        v_reset: float = -75.0,
        e_v_thresh: float = -50.0,
        i_v_thresh: float = -52.0,
        e_refractory: int = 3,
        i_refractory: int = 1,
        r_m: float = 1.0,
        noise_std: float = 0.5,
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.inhibitory_fraction = float(np.clip(inhibitory_fraction, 0.0, 1.0))
        self.num_inhibitory = int(round(num_neurons * self.inhibitory_fraction))
        self.num_excitatory = num_neurons - self.num_inhibitory

        self.rng = np.random.default_rng(seed)

        # Assign neuron types: first (N - N_I) are Excitatory, remaining N_I are Inhibitory
        self.neuron_types = np.full(num_neurons, NeuronType.EXCITATORY.value, dtype=object)
        if self.num_inhibitory > 0:
            self.neuron_types[-self.num_inhibitory :] = NeuronType.INHIBITORY.value

        self.is_inhibitory = (self.neuron_types == NeuronType.INHIBITORY.value)
        self.is_excitatory = ~self.is_inhibitory

        # Cell-specific biophysical parameters
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.r_m = float(r_m)
        self.noise_std = float(noise_std)

        self.tau_m = np.where(self.is_inhibitory, float(i_tau_m), float(e_tau_m)).astype(np.float32)
        self.v_thresh = np.where(self.is_inhibitory, float(i_v_thresh), float(e_v_thresh)).astype(np.float32)
        self.refractory_period = np.where(self.is_inhibitory, int(i_refractory), int(e_refractory)).astype(np.int32)

        # State arrays
        self.v = np.full(num_neurons, self.v_rest, dtype=np.float32)
        self.refractory_remaining = np.zeros(num_neurons, dtype=np.int32)
        self.spikes = np.zeros(num_neurons, dtype=bool)
        self.last_spike_step = np.full(num_neurons, -1000, dtype=np.int32)
        self.spike_counts = np.zeros(num_neurons, dtype=np.int64)

    def reset(self) -> None:
        """Reset membrane states to rest."""
        self.v.fill(self.v_rest)
        self.refractory_remaining.fill(0)
        self.spikes.fill(False)
        self.last_spike_step.fill(-1000)
        self.spike_counts.fill(0)

    def step(self, i_syn: np.ndarray, dt: float = 1.0, current_step: int = 0) -> np.ndarray:
        """Step membrane integration with cell-type-specific constants."""
        # 1. Decrement refractory timer
        in_refractory = self.refractory_remaining > 0
        self.refractory_remaining[in_refractory] -= 1
        self.v[in_refractory] = self.v_reset

        # 2. Integrate passive leak and drive
        non_refractory = ~in_refractory
        if np.any(non_refractory):
            noise = self.rng.normal(0.0, self.noise_std, size=self.num_neurons).astype(np.float32)
            leak = (self.v_rest - self.v[non_refractory]) / self.tau_m[non_refractory]
            drive = (self.r_m * i_syn[non_refractory]) / self.tau_m[non_refractory]
            self.v[non_refractory] += dt * (leak + drive) + noise[non_refractory]

        # 3. Threshold crossing
        fired = non_refractory & (self.v >= self.v_thresh)

        # 4. Action potential reset
        if np.any(fired):
            self.v[fired] = self.v_reset
            self.refractory_remaining[fired] = self.refractory_period[fired]
            self.last_spike_step[fired] = current_step
            self.spike_counts[fired] += 1

        self.spikes = fired
        return fired

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "model": "EI_LIF",
            "v_mean": float(np.mean(self.v)),
            "v_std": float(np.std(self.v)),
            "e_v_mean": float(np.mean(self.v[self.is_excitatory])) if np.any(self.is_excitatory) else 0.0,
            "i_v_mean": float(np.mean(self.v[self.is_inhibitory])) if np.any(self.is_inhibitory) else 0.0,
            "spikes_active": int(np.sum(self.spikes)),
            "e_spikes_active": int(np.sum(self.spikes[self.is_excitatory])),
            "i_spikes_active": int(np.sum(self.spikes[self.is_inhibitory])),
            "total_spikes": int(np.sum(self.spike_counts)),
        }


class EIIzhikevichPopulation:
    """Heterogeneous Izhikevich population: Regular Spiking (RS) for E, Fast Spiking (FS) for I."""

    def __init__(
        self,
        num_neurons: int,
        inhibitory_fraction: float = 0.20,
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.inhibitory_fraction = float(np.clip(inhibitory_fraction, 0.0, 1.0))
        self.num_inhibitory = int(round(num_neurons * self.inhibitory_fraction))
        self.num_excitatory = num_neurons - self.num_inhibitory

        self.rng = np.random.default_rng(seed)

        self.neuron_types = np.full(num_neurons, NeuronType.EXCITATORY.value, dtype=object)
        if self.num_inhibitory > 0:
            self.neuron_types[-self.num_inhibitory :] = NeuronType.INHIBITORY.value

        self.is_inhibitory = (self.neuron_types == NeuronType.INHIBITORY.value)
        self.is_excitatory = ~self.is_inhibitory

        # Regular Spiking (E): a=0.02, b=0.20, c=-65.0, d=8.0
        # Fast Spiking (I): a=0.10, b=0.20, c=-65.0, d=2.0
        self.a = np.where(self.is_inhibitory, 0.10, 0.02).astype(np.float32)
        self.b = np.where(self.is_inhibitory, 0.20, 0.20).astype(np.float32)
        self.c = np.where(self.is_inhibitory, -65.0, -65.0).astype(np.float32)
        self.d = np.where(self.is_inhibitory, 2.0, 8.0).astype(np.float32)
        self.v_peak = 30.0

        # State arrays
        self.v = self.c.copy()
        self.u = (self.b * self.v).astype(np.float32)
        self.spikes = np.zeros(num_neurons, dtype=bool)
        self.last_spike_step = np.full(num_neurons, -1000, dtype=np.int32)
        self.spike_counts = np.zeros(num_neurons, dtype=np.int64)

    def reset(self) -> None:
        self.v = self.c.copy()
        self.u = (self.b * self.v).astype(np.float32)
        self.spikes.fill(False)
        self.last_spike_step.fill(-1000)
        self.spike_counts.fill(0)

    def step(self, i_syn: np.ndarray, dt: float = 1.0, current_step: int = 0) -> np.ndarray:
        half_dt = 0.5 * dt
        for _ in range(2):
            dv = 0.04 * (self.v ** 2) + 5.0 * self.v + 140.0 - self.u + i_syn
            self.v += half_dt * dv
            du = self.a * (self.b * self.v - self.u)
            self.u += half_dt * du

        fired = self.v >= self.v_peak
        if np.any(fired):
            self.v[fired] = self.c[fired]
            self.u[fired] += self.d[fired]
            self.last_spike_step[fired] = current_step
            self.spike_counts[fired] += 1

        self.spikes = fired
        return fired

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "model": "EI_IZHIKEVICH",
            "v_mean": float(np.mean(self.v)),
            "v_std": float(np.std(self.v)),
            "u_mean": float(np.mean(self.u)),
            "spikes_active": int(np.sum(self.spikes)),
            "e_spikes_active": int(np.sum(self.spikes[self.is_excitatory])),
            "i_spikes_active": int(np.sum(self.spikes[self.is_inhibitory])),
            "total_spikes": int(np.sum(self.spike_counts)),
        }


class EISpikePropagationQueue:
    """Circular ring buffer queue with separate Excitatory (I_E) and Inhibitory (I_I) current accounting."""

    def __init__(self, num_neurons: int, max_delay: int = 50, track_active_spikes: bool = True):
        self.num_neurons = num_neurons
        self.max_delay = max(1, max_delay)
        self.buffer_size = self.max_delay + 2
        self.track_active_spikes = track_active_spikes

        # Separate buffers for positive (excitatory) and negative (inhibitory) arriving currents
        self.e_buffer = np.zeros((self.buffer_size, num_neurons), dtype=np.float32)
        self.i_buffer = np.zeros((self.buffer_size, num_neurons), dtype=np.float32)

        self.current_step = 0
        self._spike_counter = 0
        self.active_spikes: List[SpikeEvent] = []

    def reset(self) -> None:
        self.e_buffer.fill(0.0)
        self.i_buffer.fill(0.0)
        self.current_step = 0
        self._spike_counter = 0
        self.active_spikes.clear()

    def advance_step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance time by one simulation step.
        
        Returns:
            i_e: Excitatory currents arriving at each neuron [N] (>= 0)
            i_i: Inhibitory currents arriving at each neuron [N] (<= 0)
            i_net: Net current = i_e + i_i [N]
        """
        slot = self.current_step % self.buffer_size
        arrived_e = self.e_buffer[slot].copy()
        arrived_i = self.i_buffer[slot].copy()

        # Clear consumed slot
        self.e_buffer[slot].fill(0.0)
        self.i_buffer[slot].fill(0.0)

        self.current_step += 1

        if self.track_active_spikes:
            self.active_spikes = [s for s in self.active_spikes if s.arrival_step >= self.current_step]

        net_current = arrived_e + arrived_i
        return arrived_e, arrived_i, net_current

    def enqueue_spikes(
        self,
        src_indices: np.ndarray,
        dst_indices: np.ndarray,
        weights: np.ndarray,
        delays: np.ndarray,
        is_inhibitory_source: np.ndarray,
        distances: Optional[np.ndarray] = None,
    ) -> List[SpikeEvent]:
        """Enqueue spikes into separate E and I ring buffers based on source neuron type."""
        if len(src_indices) == 0:
            return []

        arrival_steps = self.current_step + delays
        slots = arrival_steps % self.buffer_size

        new_events: List[SpikeEvent] = []
        for k in range(len(src_indices)):
            slot = slots[k]
            dst = dst_indices[k]
            w = weights[k]

            if is_inhibitory_source[k]:
                # Inhibitory weight should be negative or sign-enforced <= 0
                val = -abs(w)
                self.i_buffer[slot, dst] += val
            else:
                # Excitatory weight >= 0
                val = abs(w)
                self.e_buffer[slot, dst] += val

            if self.track_active_spikes:
                self._spike_counter += 1
                dist = float(distances[k]) if distances is not None else 1.0
                event = SpikeEvent(
                    source=int(src_indices[k]),
                    target=int(dst),
                    emission_step=self.current_step,
                    arrival_step=int(arrival_steps[k]),
                    weight=float(val),
                    delay=int(delays[k]),
                    distance=dist,
                    spike_id=self._spike_counter,
                )
                self.active_spikes.append(event)
                new_events.append(event)

        return new_events

    def get_active_spikes(self, max_count: int = 150) -> List[SpikeEvent]:
        return self.active_spikes[:max_count]


def build_dale_connectivity_matrix(
    num_neurons: int,
    inhibitory_fraction: float = 0.20,
    density_ee: float = 0.15,
    density_ei: float = 0.25,
    density_ie: float = 0.35,
    density_ii: float = 0.20,
    weight_scale_e: float = 1.0,
    weight_scale_i: float = 2.0,
    seed: int = 42,
    break_dale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct an explicit Dale-compliant recurrent connectivity matrix W_rec [N, N].
    
    In W_rec[dst, src], column src determines the synaptic sign:
    - If src is Excitatory: W_rec[:, src] >= 0
    - If src is Inhibitory: W_rec[:, src] <= 0
    
    Args:
        num_neurons: Total units N
        inhibitory_fraction: Fraction of inhibitory interneurons (e.g. 0.20)
        density_ee, density_ei, density_ie, density_ii: Subpopulation connection probabilities
        weight_scale_e: Base magnitude for excitatory connections
        weight_scale_i: Base magnitude for inhibitory connections (relative strength g)
        seed: Random seed
        break_dale: Causal ablation flag to break sign constraint while preserving topology
        
    Returns:
        W_rec: [N, N] signed synaptic matrix
        neuron_types: [N] array of strings ("EXCITATORY" or "INHIBITORY")
    """
    rng = np.random.default_rng(seed)
    num_i = int(round(num_neurons * inhibitory_fraction))
    num_e = num_neurons - num_i

    neuron_types = np.full(num_neurons, NeuronType.EXCITATORY.value, dtype=object)
    if num_i > 0:
        neuron_types[-num_i:] = NeuronType.INHIBITORY.value

    is_i = (neuron_types == NeuronType.INHIBITORY.value)
    is_e = ~is_i

    e_indices = np.where(is_e)[0]
    i_indices = np.where(is_i)[0]

    W = np.zeros((num_neurons, num_neurons), dtype=np.float32)

    # 1. E -> E connections
    for src in e_indices:
        for dst in e_indices:
            if src != dst and rng.random() < density_ee:
                W[dst, src] = float(rng.uniform(0.1, 1.0) * weight_scale_e)

    # 2. E -> I connections
    for src in e_indices:
        for dst in i_indices:
            if src != dst and rng.random() < density_ei:
                W[dst, src] = float(rng.uniform(0.1, 1.0) * weight_scale_e)

    # 3. I -> E connections
    for src in i_indices:
        for dst in e_indices:
            if src != dst and rng.random() < density_ie:
                W[dst, src] = float(-rng.uniform(0.1, 1.0) * weight_scale_i)

    # 4. I -> I connections
    for src in i_indices:
        for dst in i_indices:
            if src != dst and rng.random() < density_ii:
                W[dst, src] = float(-rng.uniform(0.1, 1.0) * weight_scale_i)

    if break_dale:
        # Causal ablation: Randomize signs across both E and I columns, violating Dale's constraint
        sign_flip = rng.choice([-1.0, 1.0], size=W.shape)
        W = W * sign_flip

    return W, neuron_types
