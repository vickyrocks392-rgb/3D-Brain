"""Phase 15 Integrated Network: Target-Specific Dendritic Learning, Retrograde Signaling & Multi-Timescale Assemblies.

Solves the action readout bottleneck:
  TEMPORAL CONTEXT REPRESENTATION
                 │
                 ▼
        ACTION READOUT SYNAPSES
    (Target-Specific Dendritic Plasticity,
     Retrograde Signaling & Dendritic STDP)
                 │
                 ▼
       ACTION POPULATIONS (A0, A1, ...)
                 │
                 ▼
     MULTI-TIMESCALE COMPETITION (Fast / Med / Slow)
                 │
                 ▼
         PERSISTENT ACTION

Features:
- Complete biophysical upstream substrate (Phase 10/14: two-compartment, NMDA, coincidence, bursts, resonance)
- Dedicated, isolatable ActionReadout component (freeze, lesion, scramble, replace, inspect)
- Target-specific dendritic learning (D1 - D8)
- Target-specific teaching signals (G0 - G5)
- Retrograde signaling (R0 - R4)
- Dendritic STDP (S0 - S4)
- Multi-timescale action assemblies (A0 - A5)
- Input representation routing (I0 - I6)
- Comprehensive Causal Action Intervention Suite (C1 - C12)
- Diagnostic Decision Tree modes (Cases A - E)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.phase14_network import Phase14Network
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


class Phase15Network(Phase14Network):
    """Integrated Phase 15 Network addressing the temporal-representation-to-action readout bottleneck."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 50,
        output_dim: int = 2,
        # Readout & Plasticity configurations
        input_type: InputRepresentationType = InputRepresentationType.I6_COMBINED_TEMPORAL,
        learning_mech: LearningMechanism = LearningMechanism.D8_TEMPORAL_TARGET_DENDRITIC,
        teaching_signal_type: TeachingSignalType = TeachingSignalType.G3_DENDRITIC_TARGET_SIGNAL,
        retrograde_type: RetrogradeSignalType = RetrogradeSignalType.R2_ACTION_SPECIFIC,
        stdp_type: DendriticSTDPType = DendriticSTDPType.S4_TARGET_SPECIFIC_STDP,
        assembly_mode: AssemblyTimescaleMode = AssemblyTimescaleMode.A5_FAST_MEDIUM_SLOW,
        readout_lr: float = 0.02,
        tau_eligibility_ms: float = 300.0,
        # Substrate configurations (Phase 14)
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seed=seed,
            **kwargs,
        )

        self.phase15_input_type = input_type
        self.phase15_learning_mech = learning_mech
        self.phase15_teaching_type = teaching_signal_type
        self.phase15_retrograde_type = retrograde_type
        self.phase15_stdp_type = stdp_type
        self.phase15_assembly_mode = assembly_mode
        self.readout_lr = float(readout_lr)

        # Dimension of input fed to readout: based on hidden_dim
        self.readout_input_dim = hidden_dim

        # Dedicated Action Readout component (Section 3 & 4)
        self.action_readout = ActionReadout(
            input_dim=self.readout_input_dim,
            num_actions=self.output_dim,
            input_type=self.phase15_input_type,
            learning_mech=self.phase15_learning_mech,
            teaching_signal_type=self.phase15_teaching_type,
            retrograde_type=self.phase15_retrograde_type,
            stdp_type=self.phase15_stdp_type,
            lr=self.readout_lr,
            tau_eligibility_ms=tau_eligibility_ms,
            seed=seed,
        )

        # Multi-Timescale Action Assembly (Section 8)
        self.multi_timescale_assembly = MultiTimescaleAssembly(
            num_actions=self.output_dim,
            mode=self.phase15_assembly_mode,
            seed=seed,
        )

        # Causal Action Interventions C1 to C12
        self.action_causal_interventions: Dict[str, bool] = {
            "C1_CONTEXT_SCRAMBLE": False,
            "C2_CONTEXT_SWAP": False,
            "C3_READOUT_LESION": False,
            "C4_IRRELEVANT_CONTEXT": False,
            "C5_LABEL_PERMUTATION": False,
            "C6_FREEZE_READOUT": False,
            "C7_ACTION_POP_SWAP": False,
            "C8_TARGET_SIGNAL_SHUFFLE": False,
            "C9_TARGET_SIGNAL_SIGN_FLIP": False,
            "C10_RETROGRADE_REMOVAL": False,
            "C11_ELIGIBILITY_REMOVAL": False,
            "C12_TEMPORAL_REP_REMOVAL": False,
        }

        # Telemetry tracking for Phase 15
        self.last_phase15_telemetry: Dict[str, Any] = {}
        self.credit_history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset substrate, readout, assemblies, and history."""
        super().reset()
        self.action_readout.reset_state()
        self.multi_timescale_assembly.reset()
        self.credit_history.clear()
        self.last_phase15_telemetry.clear()

    def set_action_causal_intervention(self, intervention_id: str, enable: bool = True) -> None:
        """Enable or disable causal action interventions C1 through C12."""
        for key in list(self.action_causal_interventions.keys()):
            if key.startswith(intervention_id) or intervention_id in key:
                self.action_causal_interventions[key] = enable

                # Dynamic adjustments
                if "C3" in key:
                    if enable:
                        self.action_readout.lesion(fraction=0.8)
                    else:
                        self.action_readout.lesion_mask.fill(1.0)
                elif "C6" in key:
                    self.action_readout.freeze(enable)
                elif "C8" in key and enable:
                    self.action_readout.teaching_signal_type = TeachingSignalType.G4_SHUFFLED_TARGET
                elif "C9" in key and enable:
                    self.action_readout.teaching_signal_type = TeachingSignalType.G5_SIGN_FLIPPED
                elif "C10" in key and enable:
                    self.action_readout.retrograde_type = RetrogradeSignalType.R0_NONE
                elif "C11" in key and enable:
                    self.action_readout.eligibility.fill(0.0)
                    self.action_readout.dendritic_eligibility.fill(0.0)
                elif "C12" in key and enable:
                    self.action_readout.input_type = InputRepresentationType.I0_INSTANTANEOUS_SPIKES

    @property
    def num_neurons(self) -> int:
        return self.hidden_dim

    def step(
        self,
        x: np.ndarray,
        target_action: Optional[int] = None,
        context_id: Optional[int] = None,
        reward: float = 0.0,
        is_choice_step: bool = False,
        is_reward_step: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Execute one complete biophysical step through upstream substrate, readout, and assemblies."""
        effective_target = target_action
        if self.action_causal_interventions.get("C5_LABEL_PERMUTATION", False) and target_action is not None:
            effective_target = (target_action + 1) % self.output_dim

        # 1. Step Upstream Biophysical Substrate (Phase 10/14 dynamics)
        # Gating, Dale recurrence, NMDA, plateaus, bursts, resonance
        # Handle context interventions
        ctx_val = context_id if context_id is not None else 0
        if self.action_causal_interventions.get("C1_CONTEXT_SCRAMBLE", False):
            ctx_val = int(self.rng.choice([0, 1]))
        elif self.action_causal_interventions.get("C2_CONTEXT_SWAP", False):
            ctx_val = 1 - ctx_val

        # Execute substrate step
        probs_phase14, sel_phase14 = super().step(
            x=x,
            target_action=effective_target,
            context_id=ctx_val,
            reward=reward,
            is_choice_step=is_choice_step,
            is_reward_step=is_reward_step,
        )

        # 2. Extract Complete Biophysical Substrate State
        soma_v = self.population.v_soma.copy()
        dend_v = self.population.v_dend.copy()
        g_nmda = self.population.g_nmda.copy()
        spikes = self.last_spikes.copy()
        bursts = self.last_11stage_audit.get("burst_representation", np.zeros(self.output_dim)).copy()
        
        # Temporal and resonance features from Phase 14 step
        temp_features = self.last_11stage_audit.get("population_temporal", np.zeros(self.readout_input_dim)).copy()
        r_local = float(self.last_11stage_audit.get("r_local", 0.5))
        r_global = float(self.last_11stage_audit.get("r_global", 0.5))

        ctx_vector = self.action_interface.extract_context({
            "v_soma": soma_v,
            "v_dend": dend_v,
            "g_nmda": g_nmda,
            "activations": self.activations,
        })

        if self.action_causal_interventions.get("C4_IRRELEVANT_CONTEXT", False):
            ctx_vector = self.rng.normal(0.0, 1.0, size=ctx_vector.shape).astype(np.float32)

        raster_arr = np.array(self.action_raster_history) if len(self.action_raster_history) > 0 else np.zeros((1, len(spikes)))
        substrate_state = {
            "context_vector": ctx_vector,
            "v_soma": soma_v,
            "v_dend": dend_v,
            "g_nmda": g_nmda,
            "spikes": spikes,
            "spike_counts": np.sum(raster_arr[-min(10, len(raster_arr)):], axis=0) if len(raster_arr) > 0 else spikes,
            "bursts": bursts,
            "temporal_population": temp_features,
            "phase_features": np.full(self.readout_input_dim, r_global, dtype=np.float32),
            "resonance_features": np.full(self.readout_input_dim, r_local, dtype=np.float32),
        }

        # 3. Dedicated Action Readout Forward Pass
        # Pass dendritic potentials of the target action pools
        action_dend_v = np.array([np.mean(dend_v[:len(dend_v)//2]), np.mean(dend_v[len(dend_v)//2:])])[:self.output_dim]
        readout_drives = self.action_readout.forward(substrate_state, v_dend=action_dend_v)

        if self.action_causal_interventions.get("C7_ACTION_POP_SWAP", False):
            readout_drives = readout_drives[::-1]

        # 4. Multi-Timescale Action Assembly Integration
        assembly_out = self.multi_timescale_assembly.step(readout_drives, target_action=effective_target)
        final_probs = assembly_out["action_probs"]
        final_selected = assembly_out["selected_action"]
        final_activations = assembly_out["combined_activations"]

        # 5. Target-Specific Dendritic Learning Step
        effective_reward = reward
        if is_choice_step and effective_target is not None:
            # Task reward signal
            effective_reward = 1.0 if (final_selected == effective_target) else -0.5

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

        # 6. Assemble Comprehensive Phase 15 Telemetry
        self.last_phase15_telemetry = {
            "selected_action": final_selected,
            "action_probs": final_probs.copy(),
            "decision_confidence": assembly_out["decision_confidence"],
            "action_margin": assembly_out["action_margin"],
            "action_entropy": assembly_out["action_entropy"],
            "total_switches": assembly_out["total_switches"],
            "act_fast": assembly_out["act_fast"].copy(),
            "act_med": assembly_out["act_med"].copy(),
            "act_slow": assembly_out["act_slow"].copy(),
            "credit_specificity_index": credit_telemetry["credit_specificity_index"],
            "credit_precision": credit_telemetry["credit_precision"],
            "credit_leakage": credit_telemetry["credit_leakage"],
            "gini_concentration": credit_telemetry["gini_concentration"],
            "max_abs_update": credit_telemetry["max_abs_update"],
            "teaching_signals": self.action_readout.last_teaching_signals.copy(),
            "retrograde_signals": self.action_readout.last_retrograde_signals.copy(),
        }

        # Update 11-stage audit data with Phase 15 persistent action state
        self.last_11stage_audit["action_population"] = readout_drives.copy()
        self.last_11stage_audit["persistent_action"] = final_activations.copy()
        self.last_11stage_audit["selected_action"] = final_selected
        self.last_11stage_audit["decision_confidence"] = assembly_out["decision_confidence"]

        self.last_action_probs = final_probs.copy()
        self.last_selected_action = final_selected

        return final_probs, final_selected
