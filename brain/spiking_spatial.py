"""Integrated Spiking Spatial Substrates and Spiking-Latent Memory Architectures.

Implements the formal spatial spiking network and spiking-latent hybrid architecture for Phase 7:
- SpikingSpatialNetwork: 3D metric geometry, LIF or Izhikevich populations, discrete delayed spike queue, and STDP
- SpikingLatentSpatialNetwork: Two-tier coupling uniting fast discrete spiking spatial dynamics with slow persistent latent memory
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.base import BrainModel
from brain.latent_memory import PersistentLatentModule
from brain.spatial import (
    SpatialConnectivityBuilder,
    SpatialGeometry,
    SpatialNeuron,
    SpatialSynapse,
    TopologyFamily,
)
from brain.spiking import (
    IzhikevichNeuronPopulation,
    LIFNeuronPopulation,
    SpikeEvent,
    SpikePropagationQueue,
    SpikingNeuronModel,
)
from brain.stdp import PlasticityMode, STDPEngine, STDPUpdateRecord


class SpikingSpatialNetwork(BrainModel):
    """Spatially embedded spiking neural substrate with metric conduction delays and STDP."""

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 40,
        output_dim: int = 1,
        neuron_model: SpikingNeuronModel = SpikingNeuronModel.LIF,
        topology_type: TopologyFamily = TopologyFamily.MODULAR,
        target_edges: Optional[int] = None,
        propagation_speed: float = 5.0,  # mm / ms (distance units per step)
        plasticity_mode: PlasticityMode = PlasticityMode.STDP,
        dt_ms: float = 1.0,
        seed: int = 42,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        dt: Optional[float] = None,
    ):
        if input_dim is None and obs_dim is not None:
            input_dim = obs_dim
        if input_dim is None:
            input_dim = 8
        if dt is not None:
            dt_ms = dt
        self.input_dim = input_dim
        self.action_dim = action_dim if action_dim is not None else 4
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.neuron_model_type = neuron_model
        self.topology_type = topology_type
        self.propagation_speed = propagation_speed
        self.dt_ms = dt_ms
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 1. Generate 3D metric coordinates and module IDs
        if topology_type == TopologyFamily.MODULAR:
            self.positions, self.module_ids = SpatialGeometry.modular_spatial(
                n=hidden_dim, num_modules=4, seed=seed
            )
        elif topology_type == TopologyFamily.LOCAL:
            self.positions, self.module_ids = SpatialGeometry.clustered_gaussian(
                n=hidden_dim, num_clusters=4, seed=seed
            )
        else:
            self.positions, self.module_ids = SpatialGeometry.uniform_sphere(
                n=hidden_dim, radius=10.0, seed=seed
            )

        # 2. Build connectivity, pairwise distances, and discrete delays
        max_possible_edges = hidden_dim * (hidden_dim - 1)
        if target_edges is None:
            # Average out-degree = 8
            target_edges = min(max_possible_edges, hidden_dim * 8)
        self.target_edges = target_edges

        self.adjacency_mask, self.distances, self.delays = SpatialConnectivityBuilder.build_topology(
            positions=self.positions,
            module_ids=self.module_ids,
            topology_type=topology_type,
            target_edges=target_edges,
            propagation_speed=propagation_speed,
            seed=seed,
        )

        # 3. Initialize synaptic weight matrix [N, N]
        scale_w = 0.8 / np.sqrt(max(1, target_edges // hidden_dim))
        initial_weights = self.rng.uniform(0.1, 0.5, size=(hidden_dim, hidden_dim)).astype(np.float32)
        self.weights = np.where(self.adjacency_mask, initial_weights, 0.0).astype(np.float32)
        self.initial_weights = self.weights.copy()

        # 4. Spiking neuron population
        if neuron_model == SpikingNeuronModel.IZHIKEVICH:
            self.population = IzhikevichNeuronPopulation(num_neurons=hidden_dim, seed=seed)
        else:
            self.population = LIFNeuronPopulation(num_neurons=hidden_dim, seed=seed)

        # 5. Axonal spike propagation queue
        max_delay = int(np.max(self.delays)) if len(self.delays) > 0 else 10
        self.queue = SpikePropagationQueue(
            num_neurons=hidden_dim, max_delay=max_delay + 5, track_active_spikes=True
        )

        # 6. STDP plasticity engine
        self.stdp = STDPEngine(num_neurons=hidden_dim, mode=plasticity_mode)

        # 7. Input projection and linear readout
        scale_in = 1.0 / np.sqrt(max(1, input_dim))
        self.W_in = self.rng.normal(0.0, scale_in, (input_dim, hidden_dim)).astype(np.float32) * 5.0
        self.b_soma = self.rng.uniform(0.0, 1.0, hidden_dim).astype(np.float32)

        scale_out = 1.0 / np.sqrt(max(1, hidden_dim))
        self.W_out = self.rng.normal(0.0, scale_out, (hidden_dim, output_dim)).astype(np.float32)
        self.b_out = np.zeros(output_dim, dtype=np.float32)

        # Continuous post-synaptic trace for linear decoding and readout
        # a_i(t) = decay * a_i(t-1) + (1 - decay) * S_i(t)
        self.trace_decay = float(np.exp(-self.dt_ms / 20.0))
        self.activations = np.zeros(hidden_dim, dtype=np.float32)

        # Simulation step counter and telemetry buffers
        self.current_step = 0
        self.last_spikes = np.zeros(hidden_dim, dtype=bool)
        self.last_input = np.zeros(input_dim, dtype=np.float32)
        self.last_output = np.zeros(output_dim, dtype=np.float32)
        self.last_new_spikes: List[SpikeEvent] = []
        self.last_plastic_updates: List[STDPUpdateRecord] = []

    @property
    def W_rec(self) -> np.ndarray:
        return self.weights

    @W_rec.setter
    def W_rec(self, val: np.ndarray) -> None:
        self.weights = val.astype(np.float32)

    @property
    def neurons(self) -> Any:
        return self.population

    @property
    def spatial_geometry(self) -> Any:
        return type("SpatialGeom", (), {"coordinates": self.positions, "module_ids": self.module_ids})()

    def reset(self, batch_size: int = 1) -> None:
        """Reset network dynamic states and queues."""
        self.population.reset()
        self.queue.reset()
        self.stdp.reset()
        self.activations.fill(0.0)
        self.current_step = 0
        self.last_spikes.fill(False)
        self.last_input.fill(0.0)
        self.last_output.fill(0.0)
        self.last_new_spikes.clear()
        self.last_plastic_updates.clear()

    def forward_sequence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process sequence X of shape [batch_size, seq_len, input_dim]."""
        batch_size, seq_len, _ = X.shape
        Y = np.zeros((batch_size, seq_len, self.output_dim), dtype=np.float32)
        H = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)
        A = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)

        for b in range(batch_size):
            self.reset()
            for t in range(seq_len):
                y_t = self.step(X[b, t, :])
                Y[b, t, :] = y_t
                H[b, t, :] = self.population.v if hasattr(self.population, "v") else np.zeros(self.hidden_dim)
                A[b, t, :] = self.activations

        return Y, H, A

    def backward_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        cache: dict,
    ) -> Dict[str, np.ndarray]:
        """Spiking networks train online via STDP; returns empty gradient dict."""
        return {
            "W_in": np.zeros_like(self.W_in),
            "W_out": np.zeros_like(self.W_out),
            "b_out": np.zeros_like(self.b_out),
        }

    def parameter_count(self) -> int:
        """Return total parameter count."""
        return self.W_in.size + int(np.sum(self.adjacency_mask)) + self.W_out.size + self.b_out.size

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            "W_in": self.W_in.copy(),
            "W_xh": self.W_in.copy(),
            "weights": self.weights.copy(),
            "W_hh": self.weights.copy(),
            "W_out": self.W_out.copy(),
            "b_out": self.b_out.copy(),
            "b_soma": self.b_soma.copy(),
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        if "W_in" in weights:
            self.W_in = weights["W_in"].copy()
        if "weights" in weights:
            self.weights = weights["weights"].copy() * self.adjacency_mask
        if "W_out" in weights:
            self.W_out = weights["W_out"].copy()
        if "b_out" in weights:
            self.b_out = weights["b_out"].copy()
        if "b_soma" in weights:
            self.b_soma = weights["b_soma"].copy()

    def observable_state(self) -> Dict[str, Any]:
        """Return standardized observability snapshot for telemetry frames and .brainlog."""
        pop_dict = self.population.get_state_dict()
        v_mem = self.population.v if hasattr(self.population, "v") else self.population.c
        return {
            "state": np.expand_dims(v_mem, axis=0),
            "activation": np.expand_dims(self.activations, axis=0),
            "spikes": self.last_spikes.tolist(),
            "population_summary": pop_dict,
            "in_transit_spikes": len(self.queue.active_spikes),
            "plastic_updates_last_step": len(self.last_plastic_updates),
        }


    def step(
        self,
        x: np.ndarray,
        i_latent: Optional[np.ndarray] = None,
        neuromodulator: float = 1.0,
    ) -> np.ndarray:
        """Execute a single simulation step dt.

        Args:
            x: Input sensory vector [input_dim]
            i_latent: Optional subthreshold modulation current from persistent latent module [hidden_dim]
            neuromodulator: Scalar neuromodulatory novelty factor M(t)

        Returns:
            y: Output prediction vector [output_dim]
        """
        x_flat = np.asarray(x, dtype=np.float32).flatten()
        if len(x_flat) != self.input_dim:
            if len(x_flat) < self.input_dim:
                x_padded = np.zeros(self.input_dim, dtype=np.float32)
                x_padded[: len(x_flat)] = x_flat
                x_flat = x_padded
            else:
                x_flat = x_flat[: self.input_dim]

        self.last_input = x_flat

        # 1. Advance spike propagation queue and retrieve arrived synaptic currents
        i_syn = self.queue.advance_step()

        # 2. Compute sensory input drive: I_in = x · W_in
        i_in = np.dot(x_flat, self.W_in)

        # 3. Total somatic current
        i_total = i_in + i_syn + self.b_soma
        if i_latent is not None:
            i_total += i_latent

        # 4. Advance spiking neuron population
        spikes = self.population.step(i_syn=i_total, dt=self.dt_ms, current_step=self.current_step)
        self.last_spikes = spikes

        # 5. Enqueue emitted spikes into axonal propagation ring-buffer
        fired_indices = np.where(spikes)[0]
        if len(fired_indices) > 0:
            src_list = []
            dst_list = []
            weight_list = []
            delay_list = []
            dist_list = []

            for pre in fired_indices:
                post_targets = np.where(self.adjacency_mask[pre, :])[0]
                for post in post_targets:
                    src_list.append(pre)
                    dst_list.append(post)
                    weight_list.append(self.weights[pre, post])
                    delay_list.append(self.delays[pre, post])
                    dist_list.append(self.distances[pre, post])

            if len(src_list) > 0:
                new_spikes = self.queue.enqueue_spikes(
                    src_indices=np.array(src_list, dtype=np.int32),
                    dst_indices=np.array(dst_list, dtype=np.int32),
                    weights=np.array(weight_list, dtype=np.float32),
                    delays=np.array(delay_list, dtype=np.int32),
                    distances=np.array(dist_list, dtype=np.float32),
                )
                self.last_new_spikes = new_spikes
            else:
                self.last_new_spikes = []
        else:
            self.last_new_spikes = []

        # 6. Apply STDP synaptic weight updates
        self.weights, self.last_plastic_updates = self.stdp.step_plasticity(
            weights=self.weights,
            adjacency_mask=self.adjacency_mask,
            spikes=spikes,
            step=self.current_step,
            neuromodulator=neuromodulator,
            dt=self.dt_ms,
        )

        # 7. Update smoothed continuous activation traces
        self.activations = self.trace_decay * self.activations + (1.0 - self.trace_decay) * spikes.astype(np.float32)

        # 8. Compute readout output: y = W_out · a + b_out
        y = np.dot(self.activations, self.W_out) + self.b_out
        self.last_output = y
        self.current_step += 1

        return y

    def observable_state(self) -> Dict[str, Any]:
        """Return standardized observability snapshot for telemetry frames and .brainlog."""
        pop_dict = self.population.get_state_dict()
        v_mem = self.population.v if hasattr(self.population, "v") else self.population.c
        return {
            "state": np.expand_dims(v_mem, axis=0),
            "activation": np.expand_dims(self.activations, axis=0),
            "spikes": self.last_spikes.tolist(),
            "population_summary": pop_dict,
            "in_transit_spikes": len(self.queue.active_spikes),
            "plastic_updates_last_step": len(self.last_plastic_updates),
        }


class SpikingLatentSpatialNetwork(BrainModel):
    """Coupled Spiking Spatial Recurrent Substrate with Persistent Latent Memory.

    Integrates:
    - Lower tier: SpikingSpatialNetwork (fast, discrete spikes, metric delays, STDP)
    - Upper tier: PersistentLatentModule (slow, multi-timescale latent retention, multi-step prediction)
    - Top-down subthreshold modulation feedback from latent state z_t to spiking substrate
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 40,
        latent_dim: int = 24,
        output_dim: int = 1,
        neuron_model: SpikingNeuronModel = SpikingNeuronModel.LIF,
        topology_type: TopologyFamily = TopologyFamily.MODULAR,
        target_edges: Optional[int] = None,
        propagation_speed: float = 5.0,
        plasticity_mode: PlasticityMode = PlasticityMode.STDP,
        persistence_min: float = 0.85,
        persistence_max: float = 0.98,
        feedback_strength: float = 0.5,
        dt_ms: float = 1.0,
        seed: int = 42,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        dt: Optional[float] = None,
    ):
        if input_dim is None and obs_dim is not None:
            input_dim = obs_dim
        if input_dim is None:
            input_dim = 8
        if dt is not None:
            dt_ms = dt
        self.input_dim = input_dim
        self.action_dim = action_dim if action_dim is not None else 4
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seed = seed

        # Lower tier: Spiking Spatial Substrate
        self.spiking_network = SpikingSpatialNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            neuron_model=neuron_model,
            topology_type=topology_type,
            target_edges=target_edges,
            propagation_speed=propagation_speed,
            plasticity_mode=plasticity_mode,
            dt_ms=dt_ms,
            seed=seed,
        )

        # Upper tier: Persistent Latent Memory
        self.latent_module = PersistentLatentModule(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            obs_dim=input_dim,
            action_dim=4,
            persistence_min=persistence_min,
            persistence_max=persistence_max,
            feedback_strength=feedback_strength,
            seed=seed,
        )

        # Top-down feedback current register
        self.current_i_latent = np.zeros(hidden_dim, dtype=np.float32)
        self.last_predictions: Dict[int, np.ndarray] = {}

    @property
    def queue(self) -> SpikePropagationQueue:
        return self.spiking_network.queue

    @property
    def weights(self) -> np.ndarray:
        return self.spiking_network.weights

    @weights.setter
    def weights(self, val: np.ndarray) -> None:
        self.spiking_network.weights = val.astype(np.float32)

    @property
    def W_rec(self) -> np.ndarray:
        return self.spiking_network.weights

    @W_rec.setter
    def W_rec(self, val: np.ndarray) -> None:
        self.spiking_network.weights = val.astype(np.float32)

    @property
    def neurons(self) -> Any:
        return self.spiking_network.population

    @property
    def spatial_geometry(self) -> Any:
        return self.spiking_network.spatial_geometry

    @property
    def positions(self) -> np.ndarray:
        return self.spiking_network.positions

    @property
    def module_ids(self) -> np.ndarray:
        return self.spiking_network.module_ids

    def reset(self, batch_size: int = 1) -> None:
        self.spiking_network.reset(batch_size=batch_size)
        self.latent_module.reset()
        self.current_i_latent.fill(0.0)
        self.last_predictions.clear()

    def step(
        self,
        x: np.ndarray,
        action: int = 0,
        neuromodulator: float = 1.0,
    ) -> np.ndarray:
        """Step both the spiking substrate and the persistent latent memory module."""
        # 1. Step spiking substrate using previous step's top-down feedback
        y = self.spiking_network.step(
            x=x,
            i_latent=self.current_i_latent,
            neuromodulator=neuromodulator,
        )

        # 2. Update persistent latent state from smoothed spiking activations
        z, i_latent_next = self.latent_module.update_latent_state(
            recurrent_act=self.spiking_network.activations,
            action=action,
        )
        self.current_i_latent = i_latent_next

        # 3. Compute multi-step forward predictions
        self.last_predictions = self.latent_module.predict_multi_step(action=action)

        return y


    def forward_sequence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size, seq_len, _ = X.shape
        Y = np.zeros((batch_size, seq_len, self.output_dim), dtype=np.float32)
        H = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)
        A = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)

        for b in range(batch_size):
            self.reset(batch_size=1)
            for t in range(seq_len):
                y_t = self.step(X[b, t, :])
                Y[b, t, :] = y_t
                H[b, t, :] = self.spiking_network.population.v if hasattr(self.spiking_network.population, "v") else np.zeros(self.hidden_dim)
                A[b, t, :] = self.spiking_network.activations

        return Y, H, A

    def backward_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        cache: dict,
    ) -> Dict[str, np.ndarray]:
        return {}

    def parameter_count(self) -> int:
        return self.spiking_network.parameter_count() + self.latent_module.W_hz.size + self.latent_module.W_zz.size

    def get_weights(self) -> Dict[str, np.ndarray]:
        weights = self.spiking_network.get_weights()
        weights["W_hz"] = self.latent_module.W_hz.copy()
        weights["W_zz"] = self.latent_module.W_zz.copy()
        weights["W_zh"] = self.latent_module.W_zh.copy()
        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.spiking_network.set_weights(weights)
        if "W_hz" in weights:
            self.latent_module.W_hz = weights["W_hz"].copy()
        if "W_zz" in weights:
            self.latent_module.W_zz = weights["W_zz"].copy()
        if "W_zh" in weights:
            self.latent_module.W_zh = weights["W_zh"].copy()

    def observable_state(self) -> Dict[str, Any]:
        obs = self.spiking_network.observable_state()
        obs["latent_z"] = self.latent_module.z.tolist()
        obs["latent_norm"] = float(np.linalg.norm(self.latent_module.z))
        obs["multistep_predictions"] = {k: v.tolist() for k, v in self.last_predictions.items()}
        return obs
