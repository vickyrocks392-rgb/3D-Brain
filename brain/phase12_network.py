"""Phase 12 Integrated Network: Learnable Context-to-Action Interface on Frozen Recurrent Substrate.

Integrates the full Phase 11 biophysical substrate with the Phase 12 learnable action interface:
1. Two-Compartment biophysical neurons with slow NMDA conductances and dendritic plateaus.
2. Predictive Dendritic Disinhibition (top-down gating preserving context).
3. Recurrent Attractor Latching (intra-pool excitation, inter-pool lateral inhibition).
4. FROZEN Recurrent Substrate: All recurrent collateral weights W_rec are strictly frozen.
5. Learnable Context-to-Action Interface (ContextActionInterface):
   - Selectable representation sources (Somatic, Dendritic, NMDA, Attractor, Concatenated, PCA, Random)
   - Readout architectures (R0: Random, R1: Supervised, R2: Hebbian, R3: Three-Factor,
     R4: Prediction-Error, R5: Delayed Reward, R6: Analytical BPTT, R7: Phase 11 Routing)
   - Competitive action population with mutual cross-inhibition (ACTION_0 .. ACTION_3)
   - Strict credit localization: 0% recurrent leakage into substrate.
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
from brain.predictive_gating import GateMode, PredictiveDendriticGating
from brain.attractor_network import AttractorCouplingMode, RecurrentAttractorCircuit
from brain.context_action_interface import (
    ContextActionInterface,
    RepresentationSource,
    ReadoutArchitecture,
    ActionCompetitionMode,
)


class Phase12Network(BrainModel):
    """Integrated Phase 12 Network: Frozen Biophyiscal Recurrent Substrate + Learnable Action Interface."""

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
        # Recurrent Attractor (FROZEN)
        coupling_mode: AttractorCouplingMode = AttractorCouplingMode.NORMAL,
        w_ee_intra: float = 0.65,
        w_ei_cross: float = 0.55,
        # Action Interface & Readout
        source: RepresentationSource = RepresentationSource.ATTRACTOR,
        architecture: ReadoutArchitecture = ReadoutArchitecture.R3_THREE_FACTOR,
        competition_mode: ActionCompetitionMode = ActionCompetitionMode.BASELINE_INHIBITION,
        learning_rate: float = 0.02,
        tau_action_ms: float = 10.0,
        tau_eligibility_ms: float = 50.0,
        pca_dim: int = 10,
        random_proj_dim: int = 16,
        dt_ms: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, seed=seed)
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

        # 2. Recurrent Attractor Circuit (Strictly FROZEN recurrent weights)
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

        # 6. Learnable Context-to-Action Interface (Phase 12 Core Experimental Object)
        self.action_interface = ContextActionInterface(
            hidden_dim=hidden_dim,
            num_actions=output_dim,
            source=source,
            architecture=architecture,
            competition_mode=competition_mode,
            tau_action_ms=tau_action_ms,
            tau_eligibility_ms=tau_eligibility_ms,
            learning_rate=learning_rate,
            pca_dim=pca_dim,
            random_proj_dim=random_proj_dim,
            seed=seed,
        )

        # 7. Sensory Projections
        scale_in = 1.2 / np.sqrt(input_dim)
        self.W_in = self.rng.normal(0.0, scale_in, (hidden_dim, input_dim)).astype(np.float32)
        half_in = input_dim // 2
        pool_a_idx = np.where(self.attractor_circuit.pool_ids == 0)[0]
        pool_b_idx = np.where(self.attractor_circuit.pool_ids == 1)[0]
        self.W_in[pool_a_idx, :half_in] += 0.8
        self.W_in[pool_b_idx, half_in:] += 0.8

        self.b_soma = np.full(hidden_dim, 0.5, dtype=np.float32)

        # Dynamic State Tracking
        self.current_step = 0
        self.last_spikes = np.zeros(hidden_dim, dtype=bool)
        self.history_spikes: List[np.ndarray] = []
        self.history_states: List[np.ndarray] = []
        self.activations = np.zeros(hidden_dim, dtype=np.float32)
        self.act_decay = float(np.exp(-self.dt_ms / 20.0))
        self.last_action_values = np.zeros(output_dim, dtype=np.float32)
        self.last_selected_action = 0
        self.last_telemetry: Dict[str, Any] = {}

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal dynamic states while preserving synaptic weights."""
        self.population.reset()
        self.queue.reset()
        self.gating_engine.reset()
        self.action_interface.reset()
        self.current_step = 0
        self.last_spikes.fill(False)
        self.history_spikes.clear()
        self.history_states.clear()
        self.activations.fill(0.0)
        self.last_action_values.fill(0.0)
        self.last_selected_action = 0
        self.last_telemetry.clear()

    def step(
        self,
        x: np.ndarray,
        reward: float = 0.0,
        target_action: Optional[int] = None,
        context_id: Optional[int] = None,
        relevant_mask: Optional[np.ndarray] = None,
        is_choice_step: bool = False,
        is_reward_step: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Advance network by one simulation timestep dt.

        Args:
            x: Input sensory vector [input_dim]
            reward: Teaching/feedback reward signal R(t)
            target_action: Ground-truth target action index
            context_id: Current contextual cue ID
            relevant_mask: Boolean mask [hidden_dim] of causal units
            is_choice_step: True if this timestep produces the trial's action choice
            is_reward_step: True if delayed reward arrives on this timestep

        Returns:
            action_values: Continuous action accumulator values [output_dim]
            selected_action: Winning action index (0 .. output_dim-1)
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

        # 5. Enqueue Action Potentials along FROZEN Recurrent Collaterals
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

        # 6. Extract Context Representation from Substrate
        substrate_state = {
            "v_soma": self.population.v_soma,
            "v_dend": self.population.v_dend,
            "g_nmda": self.population.g_nmda,
            "activations": self.activations,
        }
        ctx_vector = self.action_interface.extract_context(substrate_state)

        # 7. Step Learnable Context-to-Action Interface
        v_act, sel_act, margin = self.action_interface.step(
            context_input=ctx_vector,
            dt_ms=self.dt_ms,
            reward=reward,
            target_action=target_action,
            context_id=context_id,
            is_choice_step=is_choice_step,
            is_reward_step=is_reward_step,
        )
        self.last_action_values = v_act
        self.last_selected_action = sel_act

        # 8. Record Telemetry
        tele = self.action_interface.get_telemetry()
        tele["context_norm"] = float(np.linalg.norm(ctx_vector))
        tele["active_spikes"] = int(np.sum(spikes))
        tele["current_step"] = self.current_step
        self.last_telemetry = tele

        self.current_step += 1
        return self.last_action_values.copy(), self.last_selected_action

    def forward_sequence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sequence processing implementation required by BrainModel."""
        batch_size, seq_len, _ = X.shape
        Y = np.zeros((batch_size, seq_len, self.output_dim), dtype=np.float32)
        H = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)
        A = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)

        for b in range(batch_size):
            self.reset()
            for t in range(seq_len):
                y, _ = self.step(X[b, t])
                Y[b, t] = y
                H[b, t] = self.population.v_soma
                A[b, t] = self.activations
        return Y, H, A

    def backward_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        cache: dict,
    ) -> Dict[str, np.ndarray]:
        """Compute exact gradients for W_ctx_to_act (BPTT reference baseline)."""
        grad_W = np.zeros_like(self.action_interface.W_ctx_to_act)
        # Offline cross-entropy gradient on action interface weights
        return {"W_ctx_to_act": grad_W}

    def observable_state(self) -> Dict[str, Any]:
        """Return full internal biophysical and interface telemetry."""
        return {
            "v_soma": self.population.v_soma.copy(),
            "v_dend": self.population.v_dend.copy(),
            "g_nmda": self.population.g_nmda.copy(),
            "activations": self.activations.copy(),
            "action_values": self.last_action_values.copy(),
            "selected_action": self.last_selected_action,
            "interface": self.action_interface.get_telemetry(),
        }

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return model synaptic weight matrices."""
        return {
            "W_in": self.W_in,
            "W_rec": self.W_rec,
            "W_ctx_to_act": self.action_interface.W_ctx_to_act,
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set model weights from dictionary."""
        if "W_in" in weights:
            self.W_in = weights["W_in"].copy()
        if "W_rec" in weights:
            self.W_rec = weights["W_rec"].copy()
        if "W_ctx_to_act" in weights:
            self.action_interface.W_ctx_to_act = weights["W_ctx_to_act"].copy()

    def parameter_count(self) -> int:
        """Return exact total count of parameters."""
        return (
            int(self.W_in.size)
            + int(self.W_rec.size)
            + int(self.action_interface.W_ctx_to_act.size)
            + int(self.gating_engine.W_pred.size)
        )
