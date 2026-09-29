"""Phase 13 Integrated Network: Biophysical Spike Encoding, Action Persistence & Multi-Stage Observability.

Extends Phase 12 by integrating:
1. Frozen Phase 11/12 Upstream Biophysical Substrate:
   - Two-compartment pyramidal neurons with slow NMDA conductances
   - Predictive dendritic disinhibition (top-down context gating)
   - Dale-constrained E/I recurrent attractor pools (strictly frozen: Delta W_rec = 0)
   - 3D spatial topology with conduction delays
2. Learnable Context-to-Action Synaptic Projection:
   - Pre-synaptic context evidence extraction (somatic, dendritic, NMDA, concatenated, etc.)
   - Synapse-isolated eligibility traces & delayed reward credit assignment
   - Local plasticity rules (R1 supervised, R3 three-factor, R5 delayed reward)
3. Biophysical Spike-Encoding Engine (S0–S9):
   - Adaptive Axon Initial Segment (AIS) thresholds
   - Activity-dependent adaptive gains
   - Rate population coding (M neurons per action pool)
   - Temporal latency-to-first-spike coding
   - Dale-constrained E/I Winner-Take-All competition
4. Action Persistence Engine:
   - Dynamic hysteresis switching barrier
   - Refractory action lockout period
   - Recurrent action attractor latching
   - Chatter, false transition, and stability tracking
5. Multi-Stage Information-Preservation Pipeline:
   - Instruments every stage: context -> dendrite -> soma -> threshold -> spike -> action
   - Zero recurrent credit leakage
6. Complete Causal Intervention Suite (C1–C12):
   - Scramble, swap, lesion, irrelevant injection, label permutation, freeze
   - Spike timing shuffle, spike count shuffle, threshold randomization, gain randomization
   - Competition disabled, persistence disabled
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.spatial import TopologyFamily
from brain.predictive_gating import GateMode
from brain.attractor_network import AttractorCouplingMode
from brain.phase12_network import Phase12Network
from brain.context_action_interface import (
    ContextActionInterface,
    RepresentationSource,
    ReadoutArchitecture,
    ActionCompetitionMode,
)
from brain.spike_encoding import (
    BiophysicalSpikeEncoder,
    SpikeEncodingStrategy,
    HomeostasisMode,
)
from brain.action_persistence import (
    ActionPersistenceController,
    ActionPersistenceState,
)


class Phase13Network(Phase12Network):
    """Integrated Phase 13 Network combining frozen biophysical substrate with biophysical action spike encoding."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 50,
        output_dim: int = 2,
        neurons_per_action: int = 16,
        topology_type: TopologyFamily = TopologyFamily.MODULAR,
        inhibitory_fraction: float = 0.20,
        propagation_speed: float = 8.0,
        # Two-Compartment Biophysics
        tau_dend: float = 25.0,
        g_coupling: float = 0.7,
        tau_nmda: float = 100.0,
        alpha_nmda: float = 0.08,
        nonlinear_dendrite: bool = True,
        theta_dend: float = -45.0,
        g_nl: float = 0.45,
        bap_amplitude: float = 12.0,
        # Predictive Gating
        gate_mode: GateMode = GateMode.GATE_ON,
        disinhibition_strength: float = 0.85,
        gate_delay_steps: int = 5,
        # Recurrent Attractor (FROZEN)
        coupling_mode: AttractorCouplingMode = AttractorCouplingMode.NORMAL,
        w_ee_intra: float = 0.65,
        w_ei_cross: float = 0.55,
        # Action Interface & Plasticity
        source: RepresentationSource = RepresentationSource.SOMATIC,
        architecture: ReadoutArchitecture = ReadoutArchitecture.R3_THREE_FACTOR,
        learning_rate: float = 0.02,
        tau_eligibility_ms: float = 50.0,
        # Phase 13 Spike Encoding Strategy & Homeostasis
        spike_strategy: SpikeEncodingStrategy = SpikeEncodingStrategy.S9_FULL_BIOPHYSICAL,
        homeostasis_mode: HomeostasisMode = HomeostasisMode.H0_NONE,
        tau_theta_ms: float = 40.0,
        theta_increment_mv: float = 5.0,
        tau_gain_ms: float = 50.0,
        w_intra_exc: float = 0.40,
        w_inter_inh: float = 0.55,
        # Phase 13 Action Persistence Parameters
        tau_persistence_ms: float = 120.0,
        lockout_duration_ms: float = 30.0,
        hysteresis_margin: float = 0.15,
        dt_ms: float = 1.0,
        seed: int = 42,
    ):
        # Initialize Phase 12 base network
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            topology_type=topology_type,
            inhibitory_fraction=inhibitory_fraction,
            propagation_speed=propagation_speed,
            tau_dend=tau_dend,
            g_coupling=g_coupling,
            tau_nmda=tau_nmda,
            alpha_nmda=alpha_nmda,
            nonlinear_dendrite=nonlinear_dendrite,
            theta_dend=theta_dend,
            g_nl=g_nl,
            bap_amplitude=bap_amplitude,
            gate_mode=gate_mode,
            disinhibition_strength=disinhibition_strength,
            gate_delay_steps=gate_delay_steps,
            coupling_mode=coupling_mode,
            w_ee_intra=w_ee_intra,
            w_ei_cross=w_ei_cross,
            source=source,
            architecture=architecture,
            learning_rate=learning_rate,
            tau_eligibility_ms=tau_eligibility_ms,
            dt_ms=dt_ms,
            seed=seed,
        )

        self.neurons_per_action = max(1, int(neurons_per_action))
        self.spike_strategy = spike_strategy
        self.homeostasis_mode = homeostasis_mode

        # 1. Biophysical Spike Encoder (S0..S9)
        self.spike_encoder = BiophysicalSpikeEncoder(
            num_actions=output_dim,
            neurons_per_action=self.neurons_per_action,
            strategy=spike_strategy,
            homeostasis=homeostasis_mode,
            dt_ms=dt_ms,
            tau_theta_ms=tau_theta_ms,
            theta_increment_mv=theta_increment_mv,
            tau_gain_ms=tau_gain_ms,
            w_intra_exc=w_intra_exc,
            w_inter_inh=w_inter_inh,
            seed=seed,
        )

        # 2. Action Persistence Controller
        self.persistence = ActionPersistenceController(
            num_actions=output_dim,
            dt_ms=dt_ms,
            tau_persistence_ms=tau_persistence_ms,
            lockout_duration_ms=lockout_duration_ms,
            hysteresis_margin=hysteresis_margin,
            seed=seed,
        )

        # Multi-stage biophysical audit records
        self.last_stage_audit: Dict[str, Any] = {}

    def reset(self) -> None:
        """Reset dynamic states across all biophysical compartments."""
        super().reset()
        self.spike_encoder.reset()
        self.persistence.reset()
        self.last_stage_audit.clear()

    def set_context_id(self, context_id: int) -> None:
        """Notify network and persistence controller of context identity."""
        self.persistence.set_context(context_id)

    def step(
        self,
        x: np.ndarray,
        target_action: Optional[int] = None,
        reward: float = 0.0,
        context_id: Optional[int] = None,
        relevant_mask: Optional[np.ndarray] = None,
        is_choice_step: bool = False,
        is_reward_step: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Execute one simulation step across the complete biophysical pathway."""
        if context_id is not None:
            self.set_context_id(context_id)

        # 1. Step upstream biophysical substrate and extract context
        # Advance axonal spike queue
        i_e_arrived, i_i_arrived, _ = self.queue.advance_step()

        x_arr = np.asarray(x, dtype=np.float32)
        if len(x_arr) < self.input_dim:
            padded = np.zeros(self.input_dim, dtype=np.float32)
            padded[: len(x_arr)] = x_arr
            x_arr = padded
        elif len(x_arr) > self.input_dim:
            x_arr = x_arr[: self.input_dim]

        # Sensory currents to dendrite
        i_ext_dend = np.dot(self.W_in, x_arr)
        i_e_dend = i_e_arrived + np.maximum(0.0, i_ext_dend)
        i_i_dend = i_i_arrived

        # Predictive gating
        current_state = np.maximum(0.0, self.population.v_soma - self.population.v_rest)
        gate_vector = self.gating_engine.compute_gate(
            attractor_state=current_state,
            relevant_mask=relevant_mask,
        )
        i_i_gated = self.gating_engine.apply_disinhibition(i_i_dend, gate_vector)

        # Two-compartment step
        spikes = self.population.step(
            i_syn_e=i_e_dend,
            i_syn_i=i_i_gated,
            i_soma_ext=self.b_soma,
            dt=self.dt_ms,
            current_step=self.current_step,
        )
        self.last_spikes = spikes
        self.history_spikes.append(spikes.copy())
        self.history_states.append(self.population.v_soma.copy())
        self.activations = self.activations * self.act_decay + (1.0 - self.act_decay) * spikes.astype(np.float32)

        # Enqueue spikes
        fired_idx = np.where(spikes)[0]
        if len(fired_idx) > 0:
            src_list, dst_list, w_list, del_list, dist_list, is_inh_list = [], [], [], [], [], []
            for src in fired_idx:
                post_targets = np.where(self.adjacency_mask[:, src])[0]
                if len(post_targets) > 0:
                    weights = self.W_rec[post_targets, src]
                    delays = self.delay_matrix[src, post_targets]
                    dists = self.distance_matrix[src, post_targets]
                    src_inh = self.is_inhibitory[src]
                    src_list.extend([src] * len(post_targets))
                    dst_list.extend(post_targets.tolist())
                    w_list.extend(weights.tolist())
                    del_list.extend(delays.tolist())
                    dist_list.extend(dists.tolist())
                    is_inh_list.extend([src_inh] * len(post_targets))

            if len(src_list) > 0:
                self.queue.enqueue_spikes(
                    src_indices=np.array(src_list, dtype=np.int32),
                    dst_indices=np.array(dst_list, dtype=np.int32),
                    weights=np.array(w_list, dtype=np.float32),
                    delays=np.array(del_list, dtype=np.int32),
                    is_inhibitory_source=np.array(is_inh_list, dtype=bool),
                    distances=np.array(dist_list, dtype=np.float32),
                )

        # 2. Extract Context Representation from Substrate
        substrate_state = {
            "v_soma": self.population.v_soma.copy(),
            "v_dend": self.population.v_dend.copy(),
            "g_nmda": self.population.g_nmda.copy(),
            "activations": self.activations.copy(),
        }
        ctx_vector = self.action_interface.extract_context(substrate_state)

        # 3. Feedforward Evidence Transmission: z_j = W^T * ctx + b
        raw_evidence = ctx_vector @ self.action_interface.W_ctx_to_act + self.action_interface.b_action

        # 4. Biophysical Spike Encoding (S0..S9)
        encoder_out = self.spike_encoder.step(input_drives=raw_evidence)
        action_spikes = encoder_out["spikes"]
        theta_eff = encoder_out["theta_eff"]
        gain = encoder_out["gain"]
        pool_firing_rates = encoder_out["pool_firing_rates"]
        pool_evidence = encoder_out["pool_evidence"]

        # 5. Action Persistence & Attractor Switching
        persistence_out = self.persistence.step(
            raw_evidence=pool_evidence,
            spikes=action_spikes,
            target_action=target_action,
        )
        selected_action = persistence_out["selected_action"]
        persistence_activations = persistence_out["persistence_activations"]

        # Compute Action Probabilities from persistence activations
        p_shifted = persistence_activations - np.max(persistence_activations)
        exp_p = np.exp(np.clip(p_shifted, -20.0, 20.0))
        self.last_action_probs = (exp_p / (np.sum(exp_p) + 1e-12)).astype(np.float32)
        self.last_action_values = persistence_activations
        self.last_selected_action = selected_action

        sorted_p = np.sort(persistence_activations)[::-1]
        self.last_winner_margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0])

        # 6. Isolated Interface Synaptic Plasticity (Strictly 0% Recurrent Leakage)
        pre_ctx = ctx_vector[:, np.newaxis]
        post_act = self.last_action_probs[np.newaxis, :]
        self.action_interface.eligibility = (
            self.action_interface.eligibility * self.action_interface.alpha_eligibility
            + (1.0 - self.action_interface.alpha_eligibility) * (pre_ctx @ post_act)
        )

        if self.action_interface.plasticity_enabled:
            self.action_interface._apply_interface_plasticity(
                x_ctx=ctx_vector,
                target_action=target_action,
                reward=reward,
                context_id=context_id,
                is_choice_step=is_choice_step,
                is_reward_step=is_reward_step,
            )

        # 7. Multi-Stage Audit Record
        self.last_stage_audit = {
            "context_vector": ctx_vector.copy(),
            "v_dend": self.population.v_dend.copy(),
            "g_nmda": self.population.g_nmda.copy(),
            "v_soma": self.population.v_soma.copy(),
            "theta_eff": theta_eff.copy(),
            "gain": gain.copy(),
            "action_spikes": action_spikes.copy(),
            "action_pool_rates": pool_firing_rates.copy(),
            "persistent_action_state": persistence_activations.copy(),
            "selected_action": selected_action,
            "margin": self.last_winner_margin,
            "persistence_state": persistence_out["state"],
            "persistence_duration_ms": persistence_out["duration_ms"],
        }

        self.current_step += 1
        return self.last_action_probs.copy(), self.last_selected_action

    # -------------------------------------------------------------------------
    # Causal Interventions Suite (C1–C12)
    # -------------------------------------------------------------------------
    def apply_causal_intervention(self, intervention_id: str, enable: bool = True) -> None:
        """Apply or clear designated causal intervention."""
        # Upstream & interface interventions
        if intervention_id == "C1_CONTEXT_SCRAMBLE":
            self.action_interface.apply_context_scramble(enable)
        elif intervention_id == "C2_CONTEXT_SWAP":
            self.action_interface.apply_context_swap(enable)
        elif intervention_id.startswith("C3_LESION"):
            if enable:
                fraction = 0.50
                if "25" in intervention_id:
                    fraction = 0.25
                elif "50" in intervention_id:
                    fraction = 0.50
                elif "75" in intervention_id:
                    fraction = 0.75
                elif "100" in intervention_id:
                    fraction = 1.00
                self.action_interface.apply_readout_lesion(fraction)
            else:
                self.action_interface.restore_lesion()
        elif intervention_id == "C4_IRRELEVANT_CONTEXT":
            self.action_interface.apply_irrelevant_context(enable)
        elif intervention_id == "C5_LABEL_PERMUTATION":
            self.action_interface.apply_label_permutation(enable)
        elif intervention_id == "C6_FREEZE_PLASTICITY":
            self.action_interface.freeze_plasticity(enable)

        # Spike encoding & persistence causal interventions
        elif intervention_id == "C7_SPIKE_TIMING_SHUFFLE":
            self.spike_encoder.shuffle_spike_timing = enable
        elif intervention_id == "C8_SPIKE_COUNT_SHUFFLE":
            self.spike_encoder.shuffle_spike_count = enable
        elif intervention_id == "C9_THRESHOLD_RANDOMIZATION":
            self.spike_encoder.randomize_threshold = enable
        elif intervention_id == "C10_GAIN_RANDOMIZATION":
            self.spike_encoder.randomize_gain = enable
        elif intervention_id == "C11_COMPETITION_DISABLED":
            self.spike_encoder.competition_disabled = enable
        elif intervention_id == "C12_PERSISTENCE_DISABLED":
            self.spike_encoder.persistence_disabled = enable
            if enable:
                self.persistence.hysteresis_margin = 0.0
                self.persistence.lockout_duration_ms = 0.0

    def observable_state(self) -> Dict[str, Any]:
        """Return complete observable state compliant with BrainLog and Observatory."""
        base_obs = super().observable_state()
        base_obs["spike_encoder"] = {
            "strategy": self.spike_encoder.strategy.value,
            "pool_rates": self.spike_encoder.pool_firing_rates.tolist(),
            "pool_evidence": self.spike_encoder.pool_evidence.tolist(),
            "theta_eff_mean": float(np.mean(self.spike_encoder.theta_base + self.spike_encoder.delta_theta)),
            "gain_mean": float(np.mean(self.spike_encoder.gain)),
        }
        base_obs["persistence"] = {
            "state": self.persistence.state.value,
            "current_action": self.persistence.current_action,
            "duration_ms": self.persistence.current_action_duration_ms,
            "lockout_remaining_ms": self.persistence.lockout_remaining_ms,
            "persistence_activations": self.persistence.persistence_activations.tolist(),
        }
        return base_obs
