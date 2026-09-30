"""Phase 16 Network: Minimal Sufficient Brain & Causal Architecture Pruning.

Implements rigorous, programmatic mechanism pruning and ablation controls to isolate
the Minimal Sufficient Brain:

  SENSORY / CONTEXT INPUT (Stage 1)
                │
                ▼
  PERSISTENT LATENT / NMDA STATE (Stage 2) [Mechanism K]
                │
                ▼
  TEMPORAL POPULATION REPRESENTATION (Stage 3) [Mechanism J]
                │
                ▼
  DENDRITIC REPRESENTATION (Stage 4) [Mechanisms B, C]
                │
                ▼
  ACTION INTERFACE & LOCAL CREDIT (Stage 5) [Mechanisms A, D, E, L]
                │
                ▼
  ACTION ASSEMBLIES / COMPETITION (Stage 6) [Mechanisms F, G, H, I]
                │
                ▼
  FINAL AUTONOMOUS ACTION (Stage 7)

Supports:
- Individual ablations (A through L)
- Pairwise ablations
- Multi-timescale assembly configurations (A0 to A5, and direct continuous readout)
- Retrograde signaling modes (R0 None, R1 Diffuse, R2 Target-Specific, R3 Compartment)
- Teaching signal modes (G0 Global, G1 Action-Target, G2 Population, G3 Dendritic, G4 Shuffled, G5 Inverted)
- 7-Stage continuous Shannon mutual information telemetry
- Causal action interventions (C1 to C12)
- Real BrainLog computational events for architecture selection and mechanism pruning
"""

from collections import defaultdict
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from brain.phase15_network import Phase15Network
from brain.action_readout import (
    ActionReadout,
    InputRepresentationType,
    LearningMechanism,
    TeachingSignalType,
    RetrogradeSignalType,
    DendriticSTDPType,
)
from brain.multi_timescale_assembly import (
    MultiTimescaleAssembly,
    AssemblyTimescaleMode,
)
from observability.events import ComputationalEvent, EventType


class Phase16Mechanism(str, Enum):
    """Explicit architectural mechanisms subject to causal pruning."""
    A_TARGET_SPECIFIC_TEACHING = "A_TARGET_SPECIFIC_TEACHING"
    B_DENDRITIC_TEACHING = "B_DENDRITIC_TEACHING"
    C_DENDRITIC_ELIGIBILITY = "C_DENDRITIC_ELIGIBILITY"
    D_DENDRITIC_STDP = "D_DENDRITIC_STDP"
    E_RETROGRADE_SIGNALING = "E_RETROGRADE_SIGNALING"
    F_FAST_ACTION_ASSEMBLY = "F_FAST_ACTION_ASSEMBLY"
    G_MEDIUM_ACTION_ASSEMBLY = "G_MEDIUM_ACTION_ASSEMBLY"
    H_SLOW_ACTION_ASSEMBLY = "H_SLOW_ACTION_ASSEMBLY"
    I_MUTUAL_INHIBITION = "I_MUTUAL_INHIBITION"
    J_TEMPORAL_POPULATION = "J_TEMPORAL_POPULATION"
    K_PERSISTENT_LATENT_NMDA = "K_PERSISTENT_LATENT_NMDA"
    L_TARGET_ACTION_ROUTING = "L_TARGET_ACTION_ROUTING"


class ArchitecturePreset(str, Enum):
    """Named architecture configurations for evaluation."""
    FULL_P15 = "FULL_P15"
    MIN_A = "MIN_A"  # Temporal Pop + Target Teaching + Direct Readout (No Assembly)
    MIN_B = "MIN_B"  # Temporal Pop + Target Teaching + Single Slow Assembly
    MIN_C = "MIN_C"  # Temporal Pop + Target Teaching + Fast/Slow Assembly
    MIN_D = "MIN_D"  # Temporal Pop + Dendritic Teaching + Dendritic Eligibility + Slow Assembly
    MIN_E = "MIN_E"  # Minimal Sufficient Candidate


class Phase16Network(Phase15Network):
    """Phase 16 Minimal Sufficient Brain Architecture with programmatic mechanism pruning."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 40,
        output_dim: int = 2,
        preset: ArchitecturePreset = ArchitecturePreset.FULL_P15,
        readout_lr: float = 0.02,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            readout_lr=readout_lr,
            seed=seed,
            **kwargs,
        )

        self.architecture_name = preset.value
        self.preset = preset

        # Mechanism state registry: True = active, False = ablated
        self.mechanism_registry: Dict[str, bool] = {
            m.value: True for m in Phase16Mechanism
        }

        # Specific control states
        self.retrograde_regime: str = "R2_ACTION_SPECIFIC"
        self.teaching_signal_regime: str = "G3_DENDRITIC_TARGET_SIGNAL"
        self.stdp_regime: str = "S4_TARGET_SPECIFIC_STDP"
        self.assembly_regime: str = "A5_FAST_MEDIUM_SLOW"
        self.input_regime: str = "I6_COMBINED_TEMPORAL"
        self.direct_readout: bool = False  # Bypasses assembly if True

        # Recorded 7-stage representation buffer for continuous Shannon MI audit
        self.stage_history_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.context_labels: List[int] = []
        self.target_labels: List[int] = []
        self.action_labels: List[int] = []

        # Audit events
        self.recent_computational_events: List[ComputationalEvent] = []

        # Apply preset configuration
        self.apply_preset(preset)

    def apply_preset(self, preset: ArchitecturePreset) -> None:
        """Configure active mechanisms according to standard presets."""
        self.preset = preset
        self.architecture_name = preset.value

        # Reset all to active first
        for k in self.mechanism_registry:
            self.mechanism_registry[k] = True

        if preset == ArchitecturePreset.FULL_P15:
            # All mechanisms active
            self.direct_readout = False
            self.retrograde_regime = "R2_ACTION_SPECIFIC"
            self.teaching_signal_regime = "G3_DENDRITIC_TARGET_SIGNAL"
            self.stdp_regime = "S4_TARGET_SPECIFIC_STDP"
            self.assembly_regime = "A5_FAST_MEDIUM_SLOW"
            self.input_regime = "I6_COMBINED_TEMPORAL"

        elif preset == ArchitecturePreset.MIN_A:
            # Minimal Direct Readout: No retrograde, no STDP, no assemblies
            self.ablate_mechanism(Phase16Mechanism.D_DENDRITIC_STDP.value)
            self.ablate_mechanism(Phase16Mechanism.E_RETROGRADE_SIGNALING.value)
            self.ablate_mechanism(Phase16Mechanism.F_FAST_ACTION_ASSEMBLY.value)
            self.ablate_mechanism(Phase16Mechanism.G_MEDIUM_ACTION_ASSEMBLY.value)
            self.ablate_mechanism(Phase16Mechanism.H_SLOW_ACTION_ASSEMBLY.value)
            self.direct_readout = True
            self.teaching_signal_regime = "G1_ACTION_TARGET_SCALAR"

        elif preset == ArchitecturePreset.MIN_B:
            # Single Slow Assembly: No retrograde, no STDP, no fast/med assembly
            self.ablate_mechanism(Phase16Mechanism.D_DENDRITIC_STDP.value)
            self.ablate_mechanism(Phase16Mechanism.E_RETROGRADE_SIGNALING.value)
            self.ablate_mechanism(Phase16Mechanism.F_FAST_ACTION_ASSEMBLY.value)
            self.ablate_mechanism(Phase16Mechanism.G_MEDIUM_ACTION_ASSEMBLY.value)
            self.assembly_regime = "A2_SLOW_ONLY"
            self.teaching_signal_regime = "G1_ACTION_TARGET_SCALAR"
            self.direct_readout = False

        elif preset == ArchitecturePreset.MIN_C:
            # Fast + Slow Two-Timescale Assembly: No retrograde, no STDP, no medium assembly
            self.ablate_mechanism(Phase16Mechanism.D_DENDRITIC_STDP.value)
            self.ablate_mechanism(Phase16Mechanism.E_RETROGRADE_SIGNALING.value)
            self.ablate_mechanism(Phase16Mechanism.G_MEDIUM_ACTION_ASSEMBLY.value)
            self.assembly_regime = "A3_FAST_MEDIUM"  # or customized two-timescale
            self.teaching_signal_regime = "G1_ACTION_TARGET_SCALAR"
            self.direct_readout = False

        elif preset == ArchitecturePreset.MIN_D:
            # Dendritic Target Signal + Dendritic Eligibility + Slow Assembly
            self.ablate_mechanism(Phase16Mechanism.D_DENDRITIC_STDP.value)
            self.ablate_mechanism(Phase16Mechanism.E_RETROGRADE_SIGNALING.value)
            self.ablate_mechanism(Phase16Mechanism.F_FAST_ACTION_ASSEMBLY.value)
            self.ablate_mechanism(Phase16Mechanism.G_MEDIUM_ACTION_ASSEMBLY.value)
            self.assembly_regime = "A2_SLOW_ONLY"
            self.teaching_signal_regime = "G3_DENDRITIC_TARGET_SIGNAL"
            self.direct_readout = False

        elif preset == ArchitecturePreset.MIN_E:
            # Empirically discovered Minimal Sufficient Brain:
            # Persistent Latent NMDA (K) + Temporal Population (J)
            # + Target-Specific Action Routing (L) + Target-Specific Reward Teaching (A)
            # + Dendritic Eligibility (C) + Single Slow Assembly with Mutual Inhibition (H, I)
            # Pruned: STDP (D), Retrograde (E), Fast Assembly (F), Medium Assembly (G), Diffuse Teaching (B)
            self.ablate_mechanism(Phase16Mechanism.D_DENDRITIC_STDP.value)
            self.ablate_mechanism(Phase16Mechanism.E_RETROGRADE_SIGNALING.value)
            self.ablate_mechanism(Phase16Mechanism.F_FAST_ACTION_ASSEMBLY.value)
            self.ablate_mechanism(Phase16Mechanism.G_MEDIUM_ACTION_ASSEMBLY.value)
            self.assembly_regime = "A2_SLOW_ONLY"
            self.teaching_signal_regime = "G1_ACTION_TARGET_SCALAR"
            self.direct_readout = False

        self._synchronize_readout_subsystems()

    def ablate_mechanism(self, mechanism_key: Union[Phase16Mechanism, str]) -> None:
        """Ablate a specific biophysical mechanism."""
        k = mechanism_key.value if isinstance(mechanism_key, Phase16Mechanism) else str(mechanism_key)
        if k in self.mechanism_registry:
            self.mechanism_registry[k] = False
            self._synchronize_readout_subsystems()

    def restore_mechanism(self, mechanism_key: Union[Phase16Mechanism, str]) -> None:
        """Restore a previously ablated mechanism."""
        k = mechanism_key.value if isinstance(mechanism_key, Phase16Mechanism) else str(mechanism_key)
        if k in self.mechanism_registry:
            self.mechanism_registry[k] = True
            self._synchronize_readout_subsystems()

    def is_mechanism_active(self, mechanism_key: Union[Phase16Mechanism, str]) -> bool:
        """Check if a mechanism is currently active."""
        k = mechanism_key.value if isinstance(mechanism_key, Phase16Mechanism) else str(mechanism_key)
        return self.mechanism_registry.get(k, False)

    def get_active_mechanisms(self) -> List[str]:
        """Return list of all currently active mechanism names."""
        return [k for k, v in self.mechanism_registry.items() if v]

    def get_removed_mechanisms(self) -> List[str]:
        """Return list of all currently removed/ablated mechanism names."""
        return [k for k, v in self.mechanism_registry.items() if not v]

    def set_consolidation_mode(self, mode: Any, **kwargs) -> None:
        """Set active synaptic consolidation mechanism on action readout."""
        self.action_readout.set_consolidation_mode(mode, **kwargs)

    def set_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze synaptic plasticity."""
        self.is_frozen = frozen
        self.action_readout.is_frozen = frozen
        if frozen:
            self.action_readout.learning_mech = LearningMechanism.D1_FROZEN
        else:
            self._synchronize_readout_subsystems()

    def _synchronize_readout_subsystems(self) -> None:
        """Synchronize internal submodules with current mechanism registry."""
        # 1. Teaching Signal (Mechanisms A & B)
        if not self.is_mechanism_active(Phase16Mechanism.A_TARGET_SPECIFIC_TEACHING):
            # Fallback to global scalar reward
            self.action_readout.teaching_signal_type = TeachingSignalType.G0_GLOBAL_SCALAR
            self.action_readout.learning_mech = LearningMechanism.D4_GLOBAL_REWARD
        elif not self.is_mechanism_active(Phase16Mechanism.B_DENDRITIC_TEACHING):
            # Target-specific scalar reward (G1)
            self.action_readout.teaching_signal_type = TeachingSignalType.G1_ACTION_TARGET_SCALAR
            self.action_readout.learning_mech = LearningMechanism.D5_TARGET_SPECIFIC_REWARD
        else:
            # Dendritic target signal (G3)
            self.action_readout.teaching_signal_type = TeachingSignalType.G3_DENDRITIC_TARGET_SIGNAL
            self.action_readout.learning_mech = LearningMechanism.D7_DENDRITIC_TEACHING

        # 2. Retrograde Signaling (Mechanism E)
        if not self.is_mechanism_active(Phase16Mechanism.E_RETROGRADE_SIGNALING):
            self.action_readout.retrograde_type = RetrogradeSignalType.R0_NONE
        else:
            if self.retrograde_regime == "R1_DIFFUSE":
                self.action_readout.retrograde_type = RetrogradeSignalType.R1_SCALAR
            elif self.retrograde_regime == "R3_COMPARTMENT":
                self.action_readout.retrograde_type = RetrogradeSignalType.R3_COMPARTMENT_SPECIFIC
            else:
                self.action_readout.retrograde_type = RetrogradeSignalType.R2_ACTION_SPECIFIC

        # 3. Dendritic STDP (Mechanism D)
        if not self.is_mechanism_active(Phase16Mechanism.D_DENDRITIC_STDP):
            self.action_readout.stdp_type = DendriticSTDPType.S0_NONE
        else:
            self.action_readout.stdp_type = DendriticSTDPType.S4_TARGET_SPECIFIC_STDP

        # 4. Input Representation (Mechanism J)
        if not self.is_mechanism_active(Phase16Mechanism.J_TEMPORAL_POPULATION):
            self.action_readout.input_type = InputRepresentationType.I0_INSTANTANEOUS_SPIKES
        else:
            self.action_readout.input_type = InputRepresentationType.I6_COMBINED_TEMPORAL

        # 5. Multi-Timescale Assemblies (Mechanisms F, G, H)
        f_act = self.is_mechanism_active(Phase16Mechanism.F_FAST_ACTION_ASSEMBLY)
        g_act = self.is_mechanism_active(Phase16Mechanism.G_MEDIUM_ACTION_ASSEMBLY)
        h_act = self.is_mechanism_active(Phase16Mechanism.H_SLOW_ACTION_ASSEMBLY)

        if f_act and g_act and h_act:
            self.multi_timescale_assembly.set_mode(AssemblyTimescaleMode.A5_FAST_MEDIUM_SLOW)
        elif g_act and h_act and not f_act:
            self.multi_timescale_assembly.set_mode(AssemblyTimescaleMode.A4_MEDIUM_SLOW)
        elif f_act and g_act and not h_act:
            self.multi_timescale_assembly.set_mode(AssemblyTimescaleMode.A3_FAST_MEDIUM)
        elif h_act and not f_act and not g_act:
            self.multi_timescale_assembly.set_mode(AssemblyTimescaleMode.A2_SLOW_ONLY)
        elif g_act and not f_act and not h_act:
            self.multi_timescale_assembly.set_mode(AssemblyTimescaleMode.A1_MEDIUM_ONLY)
        elif f_act and not g_act and not h_act:
            self.multi_timescale_assembly.set_mode(AssemblyTimescaleMode.A0_FAST_ONLY)
        else:
            # Fallback if all ablated: direct readout
            self.direct_readout = True

        # 6. Mutual Inhibition (Mechanism I)
        if not self.is_mechanism_active(Phase16Mechanism.I_MUTUAL_INHIBITION):
            self.multi_timescale_assembly.w_cross_inh = 0.0
        else:
            self.multi_timescale_assembly.w_cross_inh = 0.50

    def reset_audit_buffers(self) -> None:
        """Clear recorded representation buffers for Shannon MI auditing."""
        self.stage_history_buffers.clear()
        self.context_labels.clear()
        self.target_labels.clear()
        self.action_labels.clear()
        self.recent_computational_events.clear()

    def get_complexity_stats(self) -> Dict[str, Any]:
        """Compute complete architectural complexity metrics for the current configuration."""
        active_mechs = self.get_active_mechanisms()
        num_mechs = len(active_mechs)

        # Neurons
        neurons = self.hidden_dim + (0 if self.direct_readout else self.output_dim)

        # Synapses
        # Upstream: recurrent + input
        weights_dict = self.get_weights()
        w_rec = weights_dict.get("W_rec", np.zeros((self.hidden_dim, self.hidden_dim)))
        w_in = weights_dict.get("W_in", np.zeros((self.hidden_dim, self.input_dim)))
        upstream_synapses = int(np.sum(np.abs(w_rec) > 1e-6)) + int(np.sum(np.abs(w_in) > 1e-6))
        readout_synapses = self.readout_input_dim * self.output_dim
        total_synapses = upstream_synapses + readout_synapses

        # Plastic parameters: readout weights
        plastic_params = readout_synapses if not self.action_readout.is_frozen else 0

        # State variables
        # Somatic (40) + Dendritic (40) + NMDA (40 if K active else 0)
        state_vars = self.hidden_dim * 2
        if self.is_mechanism_active(Phase16Mechanism.K_PERSISTENT_LATENT_NMDA):
            state_vars += self.hidden_dim  # NMDA conductance

        # Action interface states: eligibility traces
        state_vars += self.readout_input_dim * self.output_dim  # standard eligibility
        if self.is_mechanism_active(Phase16Mechanism.C_DENDRITIC_ELIGIBILITY):
            state_vars += self.readout_input_dim * self.output_dim  # dendritic eligibility

        # Assemblies
        if not self.direct_readout:
            if self.is_mechanism_active(Phase16Mechanism.F_FAST_ACTION_ASSEMBLY):
                state_vars += self.output_dim
            if self.is_mechanism_active(Phase16Mechanism.G_MEDIUM_ACTION_ASSEMBLY):
                state_vars += self.output_dim
            if self.is_mechanism_active(Phase16Mechanism.H_SLOW_ACTION_ASSEMBLY):
                state_vars += self.output_dim

        return {
            "architecture_name": self.architecture_name,
            "preset": self.preset.value,
            "mechanism_count": num_mechs,
            "active_mechanisms": active_mechs,
            "removed_mechanisms": self.get_removed_mechanisms(),
            "neurons": neurons,
            "synapses": total_synapses,
            "readout_synapses": readout_synapses,
            "trainable_parameters": plastic_params,
            "plastic_parameters": plastic_params,
            "state_variables": state_vars,
            "direct_readout": self.direct_readout,
        }

    def step(
        self,
        x: np.ndarray,
        target_action: Optional[int] = None,
        context_id: Optional[int] = None,
        reward: float = 0.0,
        is_choice_step: bool = False,
        is_reward_step: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Execute one complete biophysical step with Phase 16 pruning and 7-stage representation tracking."""
        effective_target = target_action
        if self.action_causal_interventions.get("C5_LABEL_PERMUTATION", False) and target_action is not None:
            effective_target = (target_action + 1) % self.output_dim

        # 1. Handle Causal Context Interventions
        ctx_val = context_id if context_id is not None else 0
        if self.action_causal_interventions.get("C1_CONTEXT_SCRAMBLE", False):
            ctx_val = int(self.rng.choice([0, 1]))
        elif self.action_causal_interventions.get("C2_CONTEXT_SWAP", False):
            ctx_val = 1 - ctx_val

        # Mechanism K Ablation: silence NMDA conductances
        if not self.is_mechanism_active(Phase16Mechanism.K_PERSISTENT_LATENT_NMDA):
            self.population.g_nmda.fill(0.0)

        # 2. Step Upstream Biophysical Substrate
        probs_phase14, sel_phase14 = super().step(
            x=x,
            target_action=effective_target,
            context_id=ctx_val,
            reward=reward,
            is_choice_step=is_choice_step,
            is_reward_step=is_reward_step,
        )

        # Re-enforce NMDA ablation if K ablated
        if not self.is_mechanism_active(Phase16Mechanism.K_PERSISTENT_LATENT_NMDA):
            self.population.g_nmda.fill(0.0)

        # 3. Extract 7 Canonical Information Stages
        soma_v = self.population.v_soma.copy()
        dend_v = self.population.v_dend.copy()
        g_nmda = self.population.g_nmda.copy()
        spikes = self.last_spikes.copy()
        bursts = self.last_11stage_audit.get("burst_representation", np.zeros(self.output_dim)).copy()
        temp_features = self.last_11stage_audit.get("population_temporal", np.zeros(self.readout_input_dim)).copy()

        # Context vector
        ctx_vector = self.action_interface.extract_context({
            "v_soma": soma_v,
            "v_dend": dend_v,
            "g_nmda": g_nmda,
            "activations": self.activations,
        })
        if self.action_causal_interventions.get("C4_IRRELEVANT_CONTEXT", False):
            ctx_vector = self.rng.normal(0.0, 1.0, size=ctx_vector.shape).astype(np.float32)

        # Stage 1: Sensory/Context Input
        stage1_input = ctx_vector.copy()

        # Stage 2: Persistent Latent State (NMDA + slow conductances)
        stage2_persistent = g_nmda.copy() if self.is_mechanism_active(Phase16Mechanism.K_PERSISTENT_LATENT_NMDA) else soma_v.copy()

        # Stage 3: Temporal Population Representation
        stage3_temporal = temp_features.copy() if self.is_mechanism_active(Phase16Mechanism.J_TEMPORAL_POPULATION) else spikes.copy()

        # Stage 4: Dendritic Representation
        stage4_dendritic = dend_v.copy()

        # Substrate state packaging for readout
        raster_arr = np.array(self.action_raster_history) if len(self.action_raster_history) > 0 else np.zeros((1, len(spikes)))
        substrate_state = {
            "context_vector": ctx_vector,
            "v_soma": soma_v,
            "v_dend": dend_v,
            "g_nmda": g_nmda,
            "spikes": spikes,
            "spike_counts": np.sum(raster_arr[-min(10, len(raster_arr)):], axis=0) if len(raster_arr) > 0 else spikes,
            "bursts": bursts,
            "temporal_population": stage3_temporal,
            "phase_features": np.full(self.readout_input_dim, float(self.last_11stage_audit.get("r_global", 0.5)), dtype=np.float32),
            "resonance_features": np.full(self.readout_input_dim, float(self.last_11stage_audit.get("r_local", 0.5)), dtype=np.float32),
        }

        # 4. Action Readout Forward Pass
        action_dend_v = np.array([np.mean(dend_v[:len(dend_v)//2]), np.mean(dend_v[len(dend_v)//2:])])[:self.output_dim]
        readout_drives = self.action_readout.forward(substrate_state, v_dend=action_dend_v)

        # Target Routing Ablation (Mechanism L)
        if not self.is_mechanism_active(Phase16Mechanism.L_TARGET_ACTION_ROUTING):
            mean_drive = float(np.mean(readout_drives))
            readout_drives = np.full_like(readout_drives, mean_drive)

        if self.action_causal_interventions.get("C7_ACTION_POP_SWAP", False):
            readout_drives = readout_drives[::-1]

        # Stage 5: Action Interface Representation (Readout Drives)
        stage5_readout = readout_drives.copy()

        # 5. Action Assembly / Direct Readout
        if self.direct_readout:
            # Continuous softmax readout without assembly dynamics
            shifted = readout_drives - np.max(readout_drives)
            exp_c = np.exp(np.clip(shifted, -20.0, 20.0))
            final_probs = exp_c / (np.sum(exp_c) + 1e-12)
            final_selected = int(np.argmax(final_probs))
            final_activations = readout_drives.copy()
            margin = float(np.sort(final_probs)[-1] - np.sort(final_probs)[-2]) if len(final_probs) > 1 else 1.0
            confidence = float(np.max(final_probs))
            switches = 0
            stage6_assembly = final_activations.copy()
        else:
            assembly_out = self.multi_timescale_assembly.step(readout_drives, target_action=effective_target)
            final_probs = assembly_out["action_probs"]
            final_selected = assembly_out["selected_action"]
            final_activations = assembly_out["combined_activations"]
            margin = assembly_out["action_margin"]
            confidence = assembly_out["decision_confidence"]
            switches = assembly_out["total_switches"]
            stage6_assembly = final_activations.copy()

        # Stage 7: Final Action (One-hot or probability vector)
        stage7_action = np.zeros(self.output_dim, dtype=np.float32)
        stage7_action[final_selected] = 1.0

        # Buffer stage vectors for information measurement audit
        if is_choice_step:
            self.stage_history_buffers["1_SENSORY_CONTEXT"].append(stage1_input)
            self.stage_history_buffers["2_PERSISTENT_LATENT"].append(stage2_persistent)
            self.stage_history_buffers["3_TEMPORAL_POPULATION"].append(stage3_temporal)
            self.stage_history_buffers["4_DENDRITIC_REPRESENTATION"].append(stage4_dendritic)
            self.stage_history_buffers["5_ACTION_INTERFACE"].append(stage5_readout)
            self.stage_history_buffers["6_ACTION_ASSEMBLY"].append(stage6_assembly)
            self.stage_history_buffers["7_FINAL_ACTION"].append(stage7_action)
            self.context_labels.append(ctx_val)
            self.target_labels.append(effective_target if effective_target is not None else 0)
            self.action_labels.append(final_selected)

        # 6. Plasticity Step
        effective_reward = reward
        if is_choice_step and effective_target is not None:
            effective_reward = 1.0 if (final_selected == effective_target) else -0.5

        # Handle Dendritic Eligibility Ablation (Mechanism C)
        if not self.is_mechanism_active(Phase16Mechanism.C_DENDRITIC_ELIGIBILITY):
            action_dend_v = None

        credit_telemetry = self.action_readout.step_plasticity(
            substrate_state=substrate_state,
            target_action=effective_target,
            reward=effective_reward,
            action_probs=final_probs,
            action_activations=final_activations,
            v_dend=action_dend_v,
            is_choice_step=is_choice_step,
            is_reward_step=is_reward_step,
        )

        # 7. Update Telemetry
        self.last_phase15_telemetry = {
            "selected_action": final_selected,
            "action_probs": final_probs.copy(),
            "decision_confidence": confidence,
            "action_margin": margin,
            "action_entropy": float(-np.sum(final_probs * np.log(final_probs + 1e-12))),
            "total_switches": switches,
            "act_fast": self.multi_timescale_assembly.act_fast.copy() if not self.direct_readout else np.zeros(self.output_dim),
            "act_med": self.multi_timescale_assembly.act_med.copy() if not self.direct_readout else np.zeros(self.output_dim),
            "act_slow": self.multi_timescale_assembly.act_slow.copy() if not self.direct_readout else np.zeros(self.output_dim),
            "credit_specificity_index": credit_telemetry["credit_specificity_index"],
            "credit_precision": credit_telemetry["credit_precision"],
            "credit_leakage": credit_telemetry["credit_leakage"],
            "gini_concentration": credit_telemetry["gini_concentration"],
            "max_abs_update": credit_telemetry["max_abs_update"],
            "teaching_signals": self.action_readout.last_teaching_signals.copy(),
            "retrograde_signals": self.action_readout.last_retrograde_signals.copy(),
            "architecture_name": self.architecture_name,
            "active_mechanisms_count": len(self.get_active_mechanisms()),
        }

        self.last_action_probs = final_probs.copy()
        self.last_selected_action = final_selected

        return final_probs, final_selected
