"""Phase 14 Integrated Network: Temporal Neural Coding, Burst Dynamics, Dendritic Coincidence & Population Resonance.

Unifies:
1. Frozen Phase 11/12/13 Upstream Biophysical Substrate:
   - Two-compartment pyramidal neurons with slow NMDA conductances
   - Top-down predictive dendritic disinhibition
   - Dale-constrained E/I recurrent attractor pools (Delta W_rec = 0)
   - 3D spatial Euclidean topology with axonal conduction delays
2. Dendritic Calcium Coincidence Detection:
   - Temporal coincidence window Delta t in [0, 1, 2, 5, 10, 20, 50] ms
   - Larkum-style bAP + apical input coincidence
   - Supralinear dendritic plateau initiation
3. Burst Dynamics Engine (B0–B7):
   - Calcium [Ca2+] accumulation and extrusion
   - Plateau-triggered doublet/triplet bursts
   - Adaptive burst thresholds
4. Temporal Coding Engine (T0–T8):
   - Multi-step spike raster tracking
   - Latency-to-first-spike, ISI, burst structure, and phase codes
5. Population Resonance Engine (R0–R4, L0–L4, P0–P7):
   - Kuramoto order parameters (R_global, R_local)
   - Phase-locking values (PLV)
   - Modularity Q_func and Participation Ratio dimensionality
6. 11-Stage Information Bottleneck Pipeline:
   - Context -> Dendrite -> NMDA -> Calcium -> Soma -> Threshold -> Spikes -> Bursts -> Temporal Pop -> Action Pop -> Persistent Action
7. Causal Interventions Suite (C1–C12):
   - Selectively ablates bursts, coincidence, NMDA, temporal integration, timing, phase, population identity, spatial organization, resonance, and interface plasticity.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.spatial import TopologyFamily
from brain.predictive_gating import GateMode
from brain.attractor_network import AttractorCouplingMode
from brain.phase13_network import Phase13Network
from brain.context_action_interface import (
    ContextActionInterface,
    RepresentationSource,
    ReadoutArchitecture,
    ActionCompetitionMode,
)
from brain.spike_encoding import SpikeEncodingStrategy, HomeostasisMode
from brain.action_persistence import ActionPersistenceController
from brain.burst_dynamics import BurstDynamicsEngine, BurstStrategy
from brain.dendritic_coincidence import DendriticCoincidenceDetector
from brain.temporal_coding import TemporalSpikeDecoder, TemporalCodingScheme
from brain.population_resonance import (
    PopulationResonanceEngine,
    ResonanceRegime,
    SpatialResonanceMode,
    PopulationCodingMode,
)
from brain.information_bottleneck import InformationBottleneckAuditor


class Phase14Network(Phase13Network):
    """Integrated Phase 14 Network combining temporal coding, burst dynamics, coincidence, and resonance."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 50,
        output_dim: int = 2,
        neurons_per_action: int = 16,
        topology_type: TopologyFamily = TopologyFamily.MODULAR,
        inhibitory_fraction: float = 0.20,
        propagation_speed: float = 8.0,
        # Two-Compartment & Slow NMDA
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
        # Frozen Attractor
        coupling_mode: AttractorCouplingMode = AttractorCouplingMode.NORMAL,
        w_ee_intra: float = 0.65,
        w_ei_cross: float = 0.55,
        # Action Interface
        source: RepresentationSource = RepresentationSource.SOMATIC,
        architecture: ReadoutArchitecture = ReadoutArchitecture.R3_THREE_FACTOR,
        learning_rate: float = 0.02,
        tau_eligibility_ms: float = 50.0,
        # Phase 14 Burst Dynamics (B0–B7)
        burst_strategy: BurstStrategy = BurstStrategy.B7_FULL_BURST_TEMPORAL,
        tau_ca_ms: float = 80.0,
        ca_burst_thresh: float = 0.60,
        # Phase 14 Dendritic Coincidence
        delta_t_coincidence_ms: float = 10.0,
        supralinear_factor: float = 2.4,
        # Phase 14 Temporal Coding (T0–T8)
        temporal_scheme: TemporalCodingScheme = TemporalCodingScheme.T8_COMBINED,
        temporal_window_steps: int = 10,
        # Phase 14 Resonance (R0–R4, L0–L4, P0–P7)
        resonance_regime: ResonanceRegime = ResonanceRegime.R4_EI_BALANCED,
        spatial_resonance_mode: SpatialResonanceMode = SpatialResonanceMode.L1_LOCAL_COMMUNITY,
        population_coding_mode: PopulationCodingMode = PopulationCodingMode.P7_RESONANT_ASSEMBLY,
        # Persistence
        tau_persistence_ms: float = 120.0,
        lockout_duration_ms: float = 30.0,
        hysteresis_margin: float = 0.15,
        dt_ms: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            neurons_per_action=neurons_per_action,
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

        self.burst_strategy = burst_strategy
        self.delta_t_coincidence_ms = float(delta_t_coincidence_ms)
        self.temporal_scheme = temporal_scheme
        self.temporal_window_steps = int(temporal_window_steps)
        self.resonance_regime = resonance_regime
        self.spatial_resonance_mode = spatial_resonance_mode
        self.population_coding_mode = population_coding_mode

        # 1. Burst Dynamics Engine (B0..B7)
        self.burst_engine = BurstDynamicsEngine(
            num_actions=output_dim,
            neurons_per_action=self.neurons_per_action,
            strategy=burst_strategy,
            dt_ms=dt_ms,
            tau_ca_ms=tau_ca_ms,
            ca_burst_threshold=ca_burst_thresh,
            seed=seed,
        )

        # 2. Dendritic Coincidence Detector
        self.coincidence_detector = DendriticCoincidenceDetector(
            num_compartments=self.hidden_dim,
            delta_t_ms=delta_t_coincidence_ms,
            dt_ms=dt_ms,
            supralinear_factor=supralinear_factor,
            seed=seed,
        )

        # 3. Temporal Spike Decoder (T0..T8)
        self.temporal_decoder = TemporalSpikeDecoder(
            num_neurons=self.output_dim * self.neurons_per_action,
            default_window_steps=temporal_window_steps,
            dt_ms=dt_ms,
            seed=seed,
        )

        # 4. Population Resonance Engine (R0..R4, L0..L4)
        pos_3d = getattr(self, "positions", None)
        comm_ids = getattr(self, "module_ids", None)
        if comm_ids is None and hasattr(self, "attractor_circuit"):
            comm_ids = getattr(self.attractor_circuit, "pool_ids", None)
        self.resonance_engine = PopulationResonanceEngine(
            num_nodes=self.hidden_dim,
            regime=resonance_regime,
            spatial_mode=spatial_resonance_mode,
            community_assignments=comm_ids,
            positions_3d=pos_3d,
            dt_ms=dt_ms,
            seed=seed,
        )

        # Action raster history across time [T, total_action_neurons]
        self.action_raster_history: List[np.ndarray] = []

        # 11-Stage Information Pipeline Audit
        self.last_11stage_audit: Dict[str, Any] = {}

        # Causal intervention flags
        self.causal_intervention_active: Dict[str, bool] = {
            f"C{i}": False for i in range(1, 13)
        }

    def reset(self) -> None:
        """Reset all network states and auxiliary Phase 14 engines."""
        super().reset()
        self.burst_engine.reset()
        self.coincidence_detector.reset()
        self.resonance_engine.reset()
        self.action_raster_history.clear()
        self.last_11stage_audit.clear()

    def set_coincidence_window(self, delta_t_ms: float) -> None:
        """Update dendritic coincidence window (for parametric sweep)."""
        self.delta_t_coincidence_ms = float(delta_t_ms)
        self.coincidence_detector.set_delta_t(delta_t_ms)

    def set_temporal_window(self, window_steps: int) -> None:
        """Update temporal integration window (for parametric sweep)."""
        self.temporal_window_steps = max(1, int(window_steps))
        self.temporal_decoder.default_window_steps = self.temporal_window_steps

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
        """Execute one simulation step across the complete 11-stage pathway."""
        if context_id is not None:
            self.set_context_id(context_id)

        # 1. Advance Axonal Queue
        i_e_arrived, i_i_arrived, _ = self.queue.advance_step()

        x_arr = np.asarray(x, dtype=np.float32)
        if len(x_arr) < self.input_dim:
            padded = np.zeros(self.input_dim, dtype=np.float32)
            padded[: len(x_arr)] = x_arr
            x_arr = padded
        elif len(x_arr) > self.input_dim:
            x_arr = x_arr[: self.input_dim]

        # Stage 1: Context input vector
        context_vector_raw = x_arr.copy()

        # 2. Sensory Currents & Dendritic Coincidence Processing
        i_ext_dend = np.dot(self.W_in, x_arr)
        i_e_dend = i_e_arrived + np.maximum(0.0, i_ext_dend)
        i_i_dend = i_i_arrived

        # C3 Ablation: Remove NMDA
        if self.causal_intervention_active.get("C3", False):
            self.population.g_nmda.fill(0.0)

        # Apply Dendritic Coincidence Detection (unless C2 active)
        if not self.causal_intervention_active.get("C2", False):
            coinc_out = self.coincidence_detector.step(
                synaptic_inputs=i_e_dend,
                v_dend=self.population.v_dend,
                v_soma=self.population.v_soma,
                g_nmda=self.population.g_nmda,
                calcium=np.mean(self.burst_engine.calcium, axis=0)
                if self.burst_engine.calcium.shape[1] == self.hidden_dim
                else np.zeros(self.hidden_dim, dtype=np.float32),
                somatic_spikes=self.last_spikes if hasattr(self, "last_spikes") else None,
            )
            i_e_dend = coinc_out["effective_drive"]

        # Predictive Gating
        current_state = np.maximum(0.0, self.population.v_soma - self.population.v_rest)
        gate_vector = self.gating_engine.compute_gate(
            attractor_state=current_state,
            relevant_mask=relevant_mask,
        )
        i_i_gated = self.gating_engine.apply_disinhibition(i_i_dend, gate_vector)

        # Two-Compartment Population Step
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

        # Advance Population Resonance Dynamics (unless C11 active)
        if self.causal_intervention_active.get("C11", False):
            self.resonance_engine.phases = self.rng.uniform(0.0, 2.0 * np.pi, self.hidden_dim).astype(np.float32)
            res_out = {
                "phases": self.resonance_engine.phases.copy(),
                "r_global": 0.05,
                "r_local": 0.05,
                "global_phase": 0.0,
                "plv": 0.05,
                "dominant_freq_hz": 40.0,
            }
        else:
            res_out = self.resonance_engine.step(node_activities=self.activations)

        # Enqueue spikes into spatial axonal queue
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

        # Extract context vector from upstream substrate
        substrate_state = {
            "v_soma": self.population.v_soma.copy(),
            "v_dend": self.population.v_dend.copy(),
            "g_nmda": self.population.g_nmda.copy(),
            "activations": self.activations.copy(),
        }
        ctx_vector = self.action_interface.extract_context(substrate_state)

        # Feedforward evidence transmission: z_j = W^T * ctx + b
        raw_evidence = ctx_vector @ self.action_interface.W_ctx_to_act + self.action_interface.b_action

        # C12 Ablation: Freeze interface plasticity
        if self.causal_intervention_active.get("C12", False):
            self.action_interface.freeze_plasticity(True)

        # 3. Burst Dynamics & Spike Generation (Stage 7 & Stage 8)
        # Configure burst ablation if C1 active
        if self.causal_intervention_active.get("C1", False):
            self.burst_engine.burst_ablation_mode = "single_spike"
        elif self.causal_intervention_active.get("C6", False):
            self.burst_engine.burst_ablation_mode = "timing_shuffle"
        else:
            self.burst_engine.burst_ablation_mode = "none"

        burst_out = self.burst_engine.step(
            input_drives=raw_evidence,
            v_dend=np.array([np.mean(self.population.v_dend)] * self.output_dim),
            g_nmda=np.array([np.mean(self.population.g_nmda)] * self.output_dim),
        )
        action_spikes = burst_out["spikes"]  # [num_actions, neurons_per_action]
        action_calcium = burst_out["calcium"]

        # Track raster history
        flat_action_spikes = action_spikes.flatten()

        # C5 Ablation: Shuffle spike timing
        if self.causal_intervention_active.get("C5", False):
            self.rng.shuffle(flat_action_spikes)
            action_spikes = flat_action_spikes.reshape(action_spikes.shape)

        self.action_raster_history.append(flat_action_spikes.copy())
        if len(self.action_raster_history) > 100:
            self.action_raster_history.pop(0)

        # 4. Temporal Population Decoding (Stage 9)
        raster_arr = np.array(self.action_raster_history)  # [T, N_total]

        # C4 Ablation: Remove temporal integration (window = 1)
        win = 1 if self.causal_intervention_active.get("C4", False) else self.temporal_window_steps
        curr_phase = res_out["global_phase"]
        # C8 Ablation: Destroy phase information
        if self.causal_intervention_active.get("C8", False):
            curr_phase = float(self.rng.uniform(0.0, 2.0 * np.pi))

        temporal_features = self.temporal_decoder.extract_features(
            spike_raster=raster_arr,
            scheme=self.temporal_scheme,
            window_steps=win,
            oscillation_phase=curr_phase,
        )

        # 5. Action Persistence & Competition (Stage 10 & 11)
        pool_evidence = burst_out["pool_evidence"]

        # C7 Ablation: Shuffle spike counts / pool evidence
        if self.causal_intervention_active.get("C7", False):
            pool_evidence = self.rng.permutation(pool_evidence)

        persistence_out = self.persistence.step(
            raw_evidence=pool_evidence,
            spikes=action_spikes,
            target_action=target_action,
        )
        selected_action = persistence_out["selected_action"]
        persistence_activations = persistence_out["persistence_activations"]

        p_shifted = persistence_activations - np.max(persistence_activations)
        exp_p = np.exp(np.clip(p_shifted, -20.0, 20.0))
        self.last_action_probs = (exp_p / (np.sum(exp_p) + 1e-12)).astype(np.float32)
        self.last_action_values = persistence_activations
        self.last_selected_action = selected_action

        sorted_p = np.sort(persistence_activations)[::-1]
        self.last_winner_margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0])

        # Synaptic Plasticity on Interface
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

        # Record 11-Stage Audit Data
        self.last_11stage_audit = {
            "context_vector": ctx_vector.copy(),
            "v_dend": self.population.v_dend.copy(),
            "g_nmda": self.population.g_nmda.copy(),
            "calcium": action_calcium.flatten().copy(),
            "v_soma": self.population.v_soma.copy(),
            "threshold_gain": np.concatenate([
                self.spike_encoder.theta_base.flatten(),
                self.spike_encoder.gain.flatten(),
            ]),
            "spikes": flat_action_spikes.copy(),
            "burst_representation": burst_out["pool_burst_counts"].copy(),
            "population_temporal": temporal_features.copy(),
            "action_population": pool_evidence.copy(),
            "persistent_action": persistence_activations.copy(),
            "selected_action": selected_action,
            "margin": self.last_winner_margin,
            "r_global": res_out["r_global"],
            "r_local": res_out["r_local"],
            "plv": res_out["plv"],
        }

        self.current_step += 1
        return self.last_action_probs.copy(), self.last_selected_action

    def apply_causal_intervention(self, intervention_id: str, enable: bool = True) -> None:
        """Enable or disable causal intervention C1 through C12."""
        key = intervention_id.split("_")[0]
        if key in self.causal_intervention_active:
            self.causal_intervention_active[key] = enable

        # Also apply standard Phase 13 interventions where mapped
        super().apply_causal_intervention(intervention_id, enable)

    def observable_state(self) -> Dict[str, Any]:
        """Return full observable telemetry compliant with BrainLog and Observatory."""
        base_obs = super().observable_state()
        base_obs["temporal_coding"] = {
            "scheme": self.temporal_scheme.value,
            "window_steps": self.temporal_window_steps,
            "total_bursts": self.burst_engine.total_burst_count,
            "burst_stats": self.burst_engine.get_recent_burst_statistics(),
            "r_global": round(float(self.resonance_engine.r_global_history[-1]), 4)
            if self.resonance_engine.r_global_history else 0.0,
            "r_local": round(float(self.resonance_engine.r_local_history[-1]), 4)
            if self.resonance_engine.r_local_history else 0.0,
            "plv": round(float(self.resonance_engine.compute_plv()), 4),
            "dominant_frequency_hz": self.resonance_engine.base_freq_hz,
        }
        return base_obs
