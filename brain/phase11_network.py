"""Phase 11 Spatially Embedded Network: Predictive Disinhibition, Recurrent Attractors & Action Routing.

Integrates the full Phase 11 architecture:
1. Two-Compartment biophysical neurons with slow NMDA conductances and dendritic plateaus.
2. Predictive Dendritic Disinhibition (top-down predictive gating attenuating dendritic inhibition on relevant branches).
3. Recurrent Attractor Latching (intra-pool excitation, inter-pool lateral cross-inhibition, Dale 80/20 constraints).
4. Selective Action Routing Circuit (competitive mutual inhibition between downstream action populations).
5. Targeted Spatial Feedback Alignment (credit assignment with gate & routing selectivity).
6. Comprehensive Attractor & Routing Diagnostics (state drift, radius, separation, switching, mutual information).
7. Causal Pathway Lesioning Suite (relevant vs. matched irrelevant pathway lesions).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.base import BrainModel
from brain.spatial import SpatialGeometry, TopologyFamily
from brain.ei_dynamics import (
    NeuronType,
    EISpikePropagationQueue,
    build_dale_connectivity_matrix,
)
from brain.homeostasis import HomeostaticSynapticScaling
from brain.two_compartment import TwoCompartmentPopulation
from brain.targeted_feedback import (
    DendriticEligibilityMode,
    FeedbackMode,
    TargetedSpatialFeedbackEngine,
    TargetedUpdateRecord,
)
from brain.predictive_gating import GateMode, PredictiveDendriticGating
from brain.attractor_network import AttractorCouplingMode, RecurrentAttractorCircuit
from brain.action_routing import ActionSelectionCircuit


class Phase11SpatialNetwork(BrainModel):
    """Integrated Phase 11 Network with predictive gating, recurrent attractor latching, and action routing."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 50,
        output_dim: int = 2,
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
        # Recurrent Attractor
        coupling_mode: AttractorCouplingMode = AttractorCouplingMode.NORMAL,
        w_ee_intra: float = 0.65,
        w_ei_cross: float = 0.55,
        # Action Routing
        mutual_inhibition: float = 0.50,
        # Plasticity & Feedback
        learning_rate: float = 0.005,
        tau_eligibility_ms: float = 50.0,
        feedback_mode: FeedbackMode = FeedbackMode.SPATIAL_LOCAL,
        eligibility_mode: DendriticEligibilityMode = DendriticEligibilityMode.VOLTAGE_GATED,
        # Homeostasis
        homeostasis_enabled: bool = True,
        dt_ms: float = 1.0,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.topology_type = topology_type
        self.inhibitory_fraction = float(inhibitory_fraction)
        self.propagation_speed = max(0.1, float(propagation_speed))
        self.dt_ms = float(dt_ms)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 1. 3D Spatial Geometry
        if topology_type in (TopologyFamily.MODULAR, TopologyFamily.LOCAL):
            self.positions, self.module_ids = SpatialGeometry.clustered_gaussian(
                n=hidden_dim, num_clusters=4, seed=seed
            )
        else:
            self.positions, self.module_ids = SpatialGeometry.uniform_sphere(n=hidden_dim, radius=4.0, seed=seed)

        diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        self.distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)
        self.delay_matrix = np.maximum(
            1, np.round(self.distance_matrix / (self.propagation_speed * self.dt_ms))
        ).astype(np.int32)

        # 2. Recurrent Attractor Circuit (Structured E-E recurrence + E-I cross inhibition)
        self.attractor_circuit = RecurrentAttractorCircuit(
            num_neurons=hidden_dim,
            num_pools=2,
            inhibitory_fraction=inhibitory_fraction,
            coupling_mode=coupling_mode,
            w_ee_intra=w_ee_intra,
            w_ei_cross=w_ei_cross,
            seed=seed,
        )
        self.neuron_types = self.attractor_circuit.neuron_types
        self.is_inhibitory = self.attractor_circuit.inhibitory_mask
        self.is_excitatory = self.attractor_circuit.excitatory_mask
        self.num_inhibitory = self.attractor_circuit.num_inhibitory
        self.num_excitatory = self.attractor_circuit.num_excitatory

        # Recurrent weights from attractor circuit
        self.W_rec = self.attractor_circuit.W_rec.copy()
        self.initial_W_rec = self.W_rec.copy()
        self.adjacency_mask = (np.abs(self.W_rec) > 1e-4).astype(bool)

        # 3. Two-Compartment Population
        self.population = TwoCompartmentPopulation(
            num_neurons=hidden_dim,
            inhibitory_fraction=inhibitory_fraction,
            tau_dend=tau_dend,
            g_coupling=g_coupling,
            tau_nmda=tau_nmda,
            alpha_nmda=alpha_nmda,
            nonlinear_dendrite=nonlinear_dendrite,
            theta_dend=theta_dend,
            g_nl=g_nl,
            bap_amplitude=bap_amplitude,
            seed=seed,
        )
        # Inherit matched cell types
        self.population.neuron_types = self.neuron_types.copy()
        self.population.is_inhibitory = self.is_inhibitory.copy()
        self.population.is_excitatory = self.is_excitatory.copy()

        # 4. Axonal Spike Delay Queue
        max_delay = int(np.max(self.delay_matrix)) + 2
        self.queue = EISpikePropagationQueue(num_neurons=hidden_dim, max_delay=max_delay)

        # 5. Predictive Dendritic Disinhibition Engine
        self.gating_engine = PredictiveDendriticGating(
            num_neurons=hidden_dim,
            hidden_dim=hidden_dim,
            gate_mode=gate_mode,
            disinhibition_strength=disinhibition_strength,
            gate_delay_steps=gate_delay_steps,
            seed=seed,
        )

        # 6. Action-Selection Routing Circuit
        self.action_circuit = ActionSelectionCircuit(
            num_hidden=hidden_dim,
            num_actions=output_dim,
            mutual_inhibition=mutual_inhibition,
            seed=seed,
        )

        # 7. Targeted Spatial Feedback Alignment Engine
        self.output_positions = np.zeros((output_dim, 3), dtype=np.float32)
        self.output_positions[:, 2] = 8.0
        self.feedback_engine = TargetedSpatialFeedbackEngine(
            num_neurons=hidden_dim,
            output_dim=output_dim,
            positions=self.positions,
            output_positions=self.output_positions,
            learning_rate=learning_rate,
            tau_eligibility_ms=tau_eligibility_ms,
            feedback_mode=feedback_mode,
            eligibility_mode=eligibility_mode,
            is_inhibitory=self.is_inhibitory,
            dt_ms=dt_ms,
            seed=seed,
        )

        # 8. Homeostatic Synaptic Scaling
        self.homeostasis = HomeostaticSynapticScaling(
            num_neurons=hidden_dim,
            is_inhibitory=self.is_inhibitory,
            enabled=homeostasis_enabled,
            dt_ms=dt_ms,
            seed=seed,
        )

        # 9. Sensory Projections
        scale_in = 1.2 / np.sqrt(input_dim)
        self.W_in = self.rng.normal(0.0, scale_in, (hidden_dim, input_dim)).astype(np.float32)
        # Route Context A (inputs 0..half) to Pool A, Context B to Pool B
        half_in = input_dim // 2
        pool_a_idx = np.where(self.attractor_circuit.pool_ids == 0)[0]
        pool_b_idx = np.where(self.attractor_circuit.pool_ids == 1)[0]
        self.W_in[pool_a_idx, :half_in] += 0.8
        self.W_in[pool_b_idx, half_in:] += 0.8

        self.b_soma = np.full(hidden_dim, 0.5, dtype=np.float32)

        # Dynamic State Tracking
        self.current_step = 0
        self.last_spikes = np.zeros(hidden_dim, dtype=bool)
        self.last_plastic_updates: List[TargetedUpdateRecord] = []
        self.last_telemetry: Dict[str, Any] = {}
        self.history_spikes: List[np.ndarray] = []
        self.history_states: List[np.ndarray] = []
        self.activations = np.zeros(hidden_dim, dtype=np.float32)
        self.act_decay = float(np.exp(-self.dt_ms / 20.0))
        self.last_action_values = np.zeros(output_dim, dtype=np.float32)
        self.last_selected_action = 0

    def reset(self) -> None:
        """Reset dynamic states while preserving synaptic weights."""
        self.population.reset()
        self.queue.reset()
        self.gating_engine.reset()
        self.action_circuit.reset()
        self.feedback_engine.reset()
        self.homeostasis.reset()
        self.current_step = 0
        self.last_spikes.fill(False)
        self.last_plastic_updates.clear()
        self.history_spikes.clear()
        self.history_states.clear()
        self.activations.fill(0.0)
        self.last_action_values.fill(0.0)
        self.last_selected_action = 0

    def step(
        self,
        x: np.ndarray,
        reward: float = 0.0,
        target_action: Optional[int] = None,
        context_id: Optional[int] = None,
        relevant_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """Advance network by one simulation timestep dt.

        Args:
            x: Input sensory vector [input_dim]
            reward: Teaching reward signal R(t)
            target_action: Desired action index (0 or 1)
            context_id: Current contextual cue ID
            relevant_mask: Boolean mask [hidden_dim] of causally relevant units

        Returns:
            action_values: Continuous action accumulator values [output_dim]
            selected_action: Winning action index (0 or 1)
        """
        x_arr = np.asarray(x, dtype=np.float32)
        if len(x_arr) < self.input_dim:
            padded = np.zeros(self.input_dim, dtype=np.float32)
            padded[: len(x_arr)] = x_arr
            x_arr = padded
        elif len(x_arr) > self.input_dim:
            x_arr = x_arr[: self.input_dim]

        # 1. Advance axonal spike queue
        i_e_arrived, i_i_arrived, _ = self.queue.advance_step()

        # 2. Sensory input currents arriving at dendrites
        i_ext_dend = np.dot(self.W_in, x_arr)
        i_e_dend = i_e_arrived + np.maximum(0.0, i_ext_dend)
        i_i_dend = i_i_arrived  # Negative (<= 0)

        # 3. Compute Top-Down Predictive Gating & Apply Dendritic Disinhibition
        current_state = np.maximum(0.0, self.population.v_soma - self.population.v_rest)
        gate_vector = self.gating_engine.compute_gate(
            attractor_state=current_state,
            relevant_mask=relevant_mask,
        )
        i_i_gated = self.gating_engine.apply_disinhibition(i_i_dend, gate_vector)

        # 4. Two-Compartment Biophysical Update
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

        # Update smoothed activations
        self.activations = self.activations * self.act_decay + (1.0 - self.act_decay) * spikes.astype(np.float32)

        # 5. Enqueue Action Potentials with Axonal Transmission Delays
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

        # 6. Action-Selection Routing Circuit Step
        # Feedforward routing of attractor state into competitive action populations
        v_act, sel_act, margin = self.action_circuit.step(self.activations, dt=self.dt_ms)
        self.last_action_values = v_act
        self.last_selected_action = sel_act

        # 7. Update Dendritic Eligibility Traces
        adj_pre_post = self.adjacency_mask.T
        self.feedback_engine.update_dendritic_eligibility(
            pre_spikes=spikes,
            v_dend=self.population.v_dend,
            v_rest=self.population.v_rest,
            post_spikes=spikes,
            adjacency_mask=adj_pre_post,
        )

        # 8. Compute Targeted Spatial Feedback & Apply Plasticity
        # Prediction error vector on action outputs
        error_vec = np.zeros(self.output_dim, dtype=np.float32)
        if target_action is not None:
            one_hot = np.zeros(self.output_dim, dtype=np.float32)
            one_hot[target_action] = 1.0
            error_vec = self.action_circuit.last_action_probs - one_hot

        # Transform 1D relevant mask into [pre, post] matrix if needed
        rel_mat = None
        if relevant_mask is not None:
            rel_arr = np.asarray(relevant_mask, dtype=bool)
            rel_mat = np.outer(rel_arr, rel_arr) & adj_pre_post

        m_vec = self.feedback_engine.compute_targeted_feedback(reward=reward, error_vector=error_vec)
        self.last_m_vector = m_vec.copy()

        new_w_rec_t, updates, metrics = self.feedback_engine.apply_targeted_plasticity(
            weights=self.W_rec.T,
            adjacency_mask=adj_pre_post,
            m_vector=m_vec,
            relevant_mask=rel_mat,
            step=self.current_step,
        )
        self.W_rec = new_w_rec_t.T
        self.last_plastic_updates = updates

        # 9. Homeostasis
        if self.homeostasis.enabled:
            self.W_rec, _ = self.homeostasis.update(spikes, self.W_rec, step=self.current_step)

        # 10. Record Telemetry
        active_attractor = self.attractor_circuit.identify_current_attractor(self.activations)
        routing_metrics = {}
        if target_action is not None and context_id is not None:
            routing_metrics = self.action_circuit.compute_routing_metrics(target_action, context_id)

        self.last_telemetry = {
            "step": self.current_step,
            "spikes_count": int(np.sum(spikes)),
            "active_attractor": active_attractor,
            "gate_selectivity": self.gating_engine.last_gate_selectivity,
            "selected_action": sel_act,
            "winner_margin": margin,
            **metrics,
            **routing_metrics,
        }

        self.current_step += 1
        return self.last_action_values, self.last_selected_action

    def forward_sequence(
        self,
        X: np.ndarray,
        hidden_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Process batch trajectory X [B, T, D] sequentially."""
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        B, T, D = X.shape
        Y = np.zeros((B, T, self.output_dim), dtype=np.float32)
        H = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        A = np.zeros((B, T, self.hidden_dim), dtype=np.float32)

        for b in range(B):
            self.reset()
            for t in range(T):
                y_act, _ = self.step(X[b, t, :])
                Y[b, t, :] = y_act
                H[b, t, :] = self.population.v_soma.copy()
                A[b, t, :] = self.activations.copy()

        return Y, H, A, {"X": X, "Y": Y, "H": H, "A": A}

    def backward_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        cache: dict,
    ) -> Dict[str, np.ndarray]:
        """Local biologically grounded learning interface; returns zero dummy gradients."""
        return {}

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            "W_in": self.W_in.copy(),
            "W_rec": self.W_rec.copy(),
            "W_route": self.action_circuit.W_route.copy(),
            "W_pred": self.gating_engine.W_pred.copy(),
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        if "W_in" in weights:
            self.W_in = weights["W_in"].copy()
        if "W_rec" in weights:
            self.W_rec = weights["W_rec"].copy()
            self.adjacency_mask = (np.abs(self.W_rec) > 1e-4).astype(bool)
        if "W_route" in weights:
            self.action_circuit.W_route = weights["W_route"].copy()
        if "W_pred" in weights:
            self.gating_engine.W_pred = weights["W_pred"].copy()

    def parameter_count(self) -> int:
        return (
            int(self.W_in.size)
            + int(self.W_rec.size)
            + int(self.action_circuit.W_route.size)
            + int(self.gating_engine.W_pred.size)
        )

    def observable_state(self) -> Dict[str, Any]:
        return {
            "step": self.current_step,
            "v_soma": self.population.v_soma.tolist(),
            "v_dend": self.population.v_dend.tolist(),
            "g_nmda": self.population.g_nmda.tolist(),
            "gate_vector": self.gating_engine.last_gate_vector.tolist(),
            "disinhibition_factor": self.gating_engine.last_disinhibition_factor.tolist(),
            "active_attractor": self.attractor_circuit.active_attractor_id,
            "pool_ids": self.attractor_circuit.pool_ids.tolist(),
            "action_values": self.last_action_values.tolist(),
            "selected_action": self.last_selected_action,
            "activations": self.activations.tolist(),
            "telemetry": self.last_telemetry,
        }
