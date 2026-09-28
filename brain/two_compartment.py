"""Two-Compartment Biophysical Neuron Population with Slow Conductances and Active Dendrites.

Implements the formal biophysical substrate for Phase 10:
- Distinct dendritic compartment (V_dend) and somatic compartment (V_soma)
- Slow NMDA-like conductance (g_nmda) with decay time constant tau_nmda and magnesium unblock
- Nonlinear dendritic plateau / calcium-like depolarization event (g_nl, theta_dend)
- Inter-compartment coupling current: I_coupling = g_c * (V_dend - V_soma)
- Somatic spike threshold, reset, and absolute refractory period
- Backpropagating action potential (bAP) injecting transient depolarization into dendrite
- Exposes full biophysical telemetry: V_soma, V_dend, g_nmda, I_coupling, plateaus, spikes
- Strictly adheres to Dale's Principle: Excitatory pyramidal-like cells and Inhibitory interneurons
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.ei_dynamics import NeuronType


class TwoCompartmentPopulation:
    """Heterogeneous Two-Compartment Neuron Population respecting Dale's cell typing.
    
    Each neuron consists of:
    1. Somatic compartment:
       - Leaky integrate-and-fire dynamics
       - Emits discrete action potentials when V_soma >= V_thresh
       - Enters refractory period after firing
       - Receives coupling current I_c from dendrite and optional somatic background drive
    2. Dendritic compartment:
       - Receives excitatory and inhibitory synaptic inputs (I_syn_e, I_syn_i)
       - Integrates slow NMDA conductance: dg_nmda/dt = -g_nmda/tau_nmda + alpha_nmda * I_syn_e
       - Voltage-dependent magnesium unblock sigmoid sigma(V_dend)
       - Nonlinear active dendritic plateau current when V_dend >= theta_dend
       - Receives backpropagating action potentials (bAP) from soma upon somatic spike
    3. Coupling:
       - Resistor coupling I_coupling = g_c * (V_dend - V_soma)
       - Depolarizes soma while repolarizing dendrite
    """

    def __init__(
        self,
        num_neurons: int,
        inhibitory_fraction: float = 0.20,
        # Somatic parameters
        v_rest: float = -70.0,
        v_reset: float = -75.0,
        e_v_thresh: float = -50.0,
        i_v_thresh: float = -52.0,
        e_tau_soma: float = 15.0,
        i_tau_soma: float = 10.0,
        e_refractory: int = 3,
        i_refractory: int = 1,
        # Dendritic parameters
        tau_dend: float = 25.0,
        g_coupling: float = 0.7,            # Inter-compartment coupling conductance
        tau_nmda: float = 100.0,            # Slow NMDA decay time constant (ms)
        alpha_nmda: float = 0.08,           # NMDA conductance increment per excitatory unit
        e_nmda_rev: float = 0.0,            # NMDA reversal potential (mV)
        nmda_v_half: float = -40.0,         # Half-activation potential for Mg2+ unblock (mV)
        nmda_k: float = 5.0,                # Slope factor for Mg2+ unblock
        # Nonlinear dendritic plateau
        nonlinear_dendrite: bool = True,    # Enable active dendritic plateau events
        theta_dend: float = -45.0,          # Threshold for dendritic plateau trigger (mV)
        g_nl: float = 0.45,                 # Plateau conductance
        e_ca_rev: float = 20.0,             # Calcium / plateau reversal potential (mV)
        plateau_duration_ms: float = 5.0,   # Duration of dendritic plateau (ms)
        # Backpropagating action potential
        bap_amplitude: float = 12.0,        # bAP depolarization added to dendrite upon spike (mV)
        # Noise
        noise_std: float = 0.5,
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.inhibitory_fraction = float(np.clip(inhibitory_fraction, 0.0, 1.0))
        self.num_inhibitory = int(round(num_neurons * self.inhibitory_fraction))
        self.num_excitatory = num_neurons - self.num_inhibitory

        self.rng = np.random.default_rng(seed)

        # Cell typing: first (N - N_I) are Excitatory, remaining N_I are Inhibitory
        self.neuron_types = np.full(num_neurons, NeuronType.EXCITATORY.value, dtype=object)
        if self.num_inhibitory > 0:
            self.neuron_types[-self.num_inhibitory :] = NeuronType.INHIBITORY.value

        self.is_inhibitory = (self.neuron_types == NeuronType.INHIBITORY.value)
        self.is_excitatory = ~self.is_inhibitory

        # Somatic biophysical parameters
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.v_thresh = np.where(self.is_inhibitory, float(i_v_thresh), float(e_v_thresh)).astype(np.float32)
        self.tau_soma = np.where(self.is_inhibitory, float(i_tau_soma), float(e_tau_soma)).astype(np.float32)
        self.refractory_period = np.where(self.is_inhibitory, int(i_refractory), int(e_refractory)).astype(np.int32)
        self.noise_std = float(noise_std)

        # Dendritic parameters
        self.tau_dend = float(max(1.0, tau_dend))
        self.g_coupling = float(max(0.0, g_coupling))
        self.tau_nmda = float(max(1.0, tau_nmda))
        self.alpha_nmda = float(alpha_nmda)
        self.e_nmda_rev = float(e_nmda_rev)
        self.nmda_v_half = float(nmda_v_half)
        self.nmda_k = float(max(0.1, nmda_k))

        # Nonlinear dendritic plateau parameters
        self.nonlinear_dendrite = bool(nonlinear_dendrite)
        self.theta_dend = float(theta_dend)
        self.g_nl = float(g_nl)
        self.e_ca_rev = float(e_ca_rev)
        self.plateau_duration_steps = int(max(1, round(plateau_duration_ms)))
        self.bap_amplitude = float(bap_amplitude)

        # Dynamic State Variables
        self.v_soma = np.full(num_neurons, self.v_rest, dtype=np.float32)
        self.v_dend = np.full(num_neurons, self.v_rest, dtype=np.float32)
        self.g_nmda = np.zeros(num_neurons, dtype=np.float32)
        self.plateau_remaining = np.zeros(num_neurons, dtype=np.int32)
        self.refractory_remaining = np.zeros(num_neurons, dtype=np.int32)
        self.spikes = np.zeros(num_neurons, dtype=bool)
        self.last_spike_step = np.full(num_neurons, -1000, dtype=np.int32)
        self.spike_counts = np.zeros(num_neurons, dtype=np.int64)

        # Telemetry cache
        self.last_i_coupling = np.zeros(num_neurons, dtype=np.float32)
        self.last_i_nmda = np.zeros(num_neurons, dtype=np.float32)
        self.last_i_plateau = np.zeros(num_neurons, dtype=np.float32)
        self.last_i_dend_e = np.zeros(num_neurons, dtype=np.float32)
        self.last_i_dend_i = np.zeros(num_neurons, dtype=np.float32)

    def reset(self) -> None:
        """Reset all somatic and dendritic membrane states to baseline resting values."""
        self.v_soma.fill(self.v_rest)
        self.v_dend.fill(self.v_rest)
        self.g_nmda.fill(0.0)
        self.plateau_remaining.fill(0)
        self.refractory_remaining.fill(0)
        self.spikes.fill(False)
        self.last_spike_step.fill(-1000)
        self.spike_counts.fill(0)
        self.last_i_coupling.fill(0.0)
        self.last_i_nmda.fill(0.0)
        self.last_i_plateau.fill(0.0)
        self.last_i_dend_e.fill(0.0)
        self.last_i_dend_i.fill(0.0)

    def step(
        self,
        i_syn_e: np.ndarray,
        i_syn_i: np.ndarray,
        i_soma_ext: Optional[np.ndarray] = None,
        dt: float = 1.0,
        current_step: int = 0,
    ) -> np.ndarray:
        """Advance two-compartment biophysical state by one simulation timestep dt (ms).

        Args:
            i_syn_e: Excitatory synaptic current arrived at dendrites [N] (>= 0)
            i_syn_i: Inhibitory synaptic current arrived at dendrites [N] (<= 0)
            i_soma_ext: Optional external somatic current drive [N]
            dt: Simulation timestep (ms)
            current_step: Global step counter

        Returns:
            spikes: Boolean array of somatic action potentials [N]
        """
        dt_float = float(dt)
        i_e = np.asarray(i_syn_e, dtype=np.float32)
        i_i = np.asarray(i_syn_i, dtype=np.float32)
        i_ext = np.asarray(i_soma_ext, dtype=np.float32) if i_soma_ext is not None else np.zeros(self.num_neurons, dtype=np.float32)

        self.last_i_dend_e = i_e.copy()
        self.last_i_dend_i = i_i.copy()

        # -------------------------------------------------------------
        # 1. Update Slow NMDA Conductance:
        # dg_nmda/dt = -g_nmda / tau_nmda + alpha_nmda * max(0, I_syn_e)
        # -------------------------------------------------------------
        decay_nmda = float(np.exp(-dt_float / self.tau_nmda))
        self.g_nmda = self.g_nmda * decay_nmda + self.alpha_nmda * np.maximum(0.0, i_e)

        # Voltage-dependent magnesium unblock factor:
        # sigma(V) = 1 / (1 + exp(-(V_dend - V_half) / k))
        mg_unblock = 1.0 / (1.0 + np.exp(-(self.v_dend - self.nmda_v_half) / self.nmda_k))
        i_nmda = self.g_nmda * mg_unblock * (self.e_nmda_rev - self.v_dend)
        self.last_i_nmda = i_nmda.copy()

        # -------------------------------------------------------------
        # 2. Active Dendritic Plateau Events (Nonlinear Calcium Spikes):
        # -------------------------------------------------------------
        i_plateau = np.zeros(self.num_neurons, dtype=np.float32)
        if self.nonlinear_dendrite:
            # Trigger new plateaus where V_dend crosses dendritic threshold and not currently in plateau
            trigger_mask = (self.v_dend >= self.theta_dend) & (self.plateau_remaining == 0) & self.is_excitatory
            self.plateau_remaining[trigger_mask] = self.plateau_duration_steps

            active_plateau = self.plateau_remaining > 0
            if np.any(active_plateau):
                i_plateau[active_plateau] = self.g_nl * (self.e_ca_rev - self.v_dend[active_plateau])
                self.plateau_remaining[active_plateau] -= 1

        self.last_i_plateau = i_plateau.copy()

        # -------------------------------------------------------------
        # 3. Inter-Compartment Coupling Current:
        # I_coupling = g_c * (V_dend - V_soma)
        # Positive I_coupling depolarizes soma while repolarizing dendrite.
        # -------------------------------------------------------------
        i_coupling = self.g_coupling * (self.v_dend - self.v_soma)
        self.last_i_coupling = i_coupling.copy()

        # -------------------------------------------------------------
        # 4. Integrate Dendritic Membrane Potential (V_dend):
        # C_d * dV_d/dt = -(V_d - E_L)/tau_d - I_coupling + I_syn_e + I_syn_i + I_nmda + I_plateau
        # -------------------------------------------------------------
        leak_dend = -(self.v_dend - self.v_rest) / self.tau_dend
        dv_dend = (
            leak_dend
            - i_coupling
            + i_e
            + i_i
            + i_nmda
            + i_plateau
        ) * dt_float

        self.v_dend = np.clip(self.v_dend + dv_dend, -90.0, 30.0)

        # -------------------------------------------------------------
        # 5. Integrate Somatic Membrane Potential (V_soma):
        # C_s * dV_s/dt = -(V_s - E_L)/tau_s + I_coupling + I_soma_ext + noise
        # (Only integrate neurons not in absolute refractory state)
        # -------------------------------------------------------------
        in_refractory = self.refractory_remaining > 0
        self.refractory_remaining[in_refractory] -= 1

        integrating = ~in_refractory
        if np.any(integrating):
            leak_soma = -(self.v_soma[integrating] - self.v_rest) / self.tau_soma[integrating]
            noise = (
                self.rng.normal(0.0, self.noise_std, size=int(np.sum(integrating))).astype(np.float32)
                if self.noise_std > 0.0
                else 0.0
            )
            dv_soma = (
                leak_soma
                + i_coupling[integrating]
                + i_ext[integrating]
                + noise
            ) * dt_float

            self.v_soma[integrating] += dv_soma

        # -------------------------------------------------------------
        # 6. Action Potential Generation & Reset:
        # If V_soma >= V_thresh -> Fire spike S=1, V_soma <- V_reset, enter refractory
        # Backpropagating action potential (bAP) adds depolarization to dendrite
        # -------------------------------------------------------------
        fired = (self.v_soma >= self.v_thresh) & integrating
        self.spikes = fired

        if np.any(fired):
            self.v_soma[fired] = self.v_reset
            self.refractory_remaining[fired] = self.refractory_period[fired]
            self.last_spike_step[fired] = current_step
            self.spike_counts[fired] += 1

            # Backpropagating action potential (bAP) to dendrite:
            # Active somatic spike injects depolarizing voltage pulse into dendrite
            if self.bap_amplitude > 0.0:
                self.v_dend[fired] += self.bap_amplitude
                # bAP can trigger dendritic calcium plateau if combined with dendritic depolarization
                if self.nonlinear_dendrite:
                    bap_triggers = (self.v_dend[fired] >= self.theta_dend) & (self.plateau_remaining[fired] == 0) & self.is_excitatory[fired]
                    fired_indices = np.where(fired)[0]
                    for idx in fired_indices[bap_triggers]:
                        self.plateau_remaining[idx] = self.plateau_duration_steps

        # Clamp resting values for non-integrating refractory neurons
        self.v_soma[in_refractory] = self.v_reset

        return self.spikes

    def get_state(self) -> Dict[str, np.ndarray]:
        """Return a snapshot of all compartment state variables."""
        return {
            "v_soma": self.v_soma.copy(),
            "v_dend": self.v_dend.copy(),
            "g_nmda": self.g_nmda.copy(),
            "spikes": self.spikes.copy(),
            "i_coupling": self.last_i_coupling.copy(),
            "i_nmda": self.last_i_nmda.copy(),
            "i_plateau": self.last_i_plateau.copy(),
            "plateau_active": (self.plateau_remaining > 0),
            "refractory_active": (self.refractory_remaining > 0),
        }
