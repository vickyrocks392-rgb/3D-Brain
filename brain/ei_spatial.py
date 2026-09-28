"""Balanced Excitation/Inhibition (E/I) Spatially Embedded Substrates and Latent Memory Networks.

Implements the formal Phase 8 unified architecture:
- EIBalancedSpatialNetwork: 3D spatial geometry, Dale-constrained E/I connectivity,
  fast inhibitory interneurons, dual-current propagation queue, homeostatic scaling, and STDP.
- EIBalancedLatentSpatialNetwork: Two-tier architecture uniting balanced E/I spiking dynamics
  with persistent latent memory for long-delay cue retention and multi-step prediction.
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
    SpikeEvent,
    SpikingNeuronModel,
)
from brain.stdp import PlasticityMode, STDPEngine, STDPUpdateRecord
from brain.ei_dynamics import (
    NeuronType,
    EILIFPopulation,
    EIIzhikevichPopulation,
    EISpikePropagationQueue,
    build_dale_connectivity_matrix,
)
from brain.homeostasis import HomeostaticSynapticScaling


class EIBalancedSpatialNetwork(BrainModel):
    """Spatially embedded Dale-constrained E/I spiking neural substrate."""

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 40,
        output_dim: int = 1,
        neuron_model: SpikingNeuronModel = SpikingNeuronModel.LIF,
        topology_type: TopologyFamily = TopologyFamily.MODULAR,
        inhibitory_fraction: float = 0.20,
        density_ee: float = 0.18,
        density_ei: float = 0.28,
        density_ie: float = 0.38,
        density_ii: float = 0.20,
        weight_scale_e: float = 1.0,
        weight_scale_i: float = 2.0,
        propagation_speed: float = 5.0,  # mm / ms
        plasticity_mode: PlasticityMode = PlasticityMode.STDP,
        homeostasis_enabled: bool = True,
        homeostasis_mode: str = "local",
        target_rate_hz: float = 10.0,
        tau_homeo_ms: float = 200.0,
        eta_homeo: float = 0.002,
        dt_ms: float = 1.0,
        seed: int = 42,
        break_dale: bool = False,
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
        self.inhibitory_fraction = float(np.clip(inhibitory_fraction, 0.0, 1.0))
        self.propagation_speed = max(0.1, float(propagation_speed))
        self.plasticity_mode = plasticity_mode
        self.dt_ms = float(dt_ms)
        self.seed = seed
        self.break_dale = break_dale

        self.rng = np.random.default_rng(seed)

        # 1. 3D Spatial Geometry
        if topology_type in (TopologyFamily.MODULAR, TopologyFamily.LOCAL):
            self.positions, self.module_ids = SpatialGeometry.clustered_gaussian(
                n=hidden_dim, num_clusters=4, seed=seed
            )
        else:
            self.positions, self.module_ids = SpatialGeometry.uniform_sphere(
                n=hidden_dim, radius=10.0, seed=seed
            )

        # 2. Assign Neuron Types & Populations
        if neuron_model == SpikingNeuronModel.IZHIKEVICH:
            self.population = EIIzhikevichPopulation(
                num_neurons=hidden_dim,
                inhibitory_fraction=self.inhibitory_fraction,
                seed=seed,
            )
        else:
            self.population = EILIFPopulation(
                num_neurons=hidden_dim,
                inhibitory_fraction=self.inhibitory_fraction,
                seed=seed,
            )

        self.is_inhibitory = self.population.is_inhibitory
        self.is_excitatory = self.population.is_excitatory
        self.neuron_types = self.population.neuron_types
        self.num_inhibitory = int(np.sum(self.is_inhibitory))
        self.num_excitatory = int(np.sum(self.is_excitatory))

        # 3. Dale-compliant Recurrent Synaptic Weights W_rec [N, N]
        self.W_rec, _ = build_dale_connectivity_matrix(
            num_neurons=hidden_dim,
            inhibitory_fraction=self.inhibitory_fraction,
            density_ee=density_ee,
            density_ei=density_ei,
            density_ie=density_ie,
            density_ii=density_ii,
            weight_scale_e=weight_scale_e,
            weight_scale_i=weight_scale_i,
            seed=seed,
            break_dale=break_dale,
        )
        self.initial_W_rec = self.W_rec.copy()
        self.adjacency_mask = (np.abs(self.W_rec) > 1e-6)

        # 4. Sensory Input Projections W_in [N, input_dim] & Readout W_out [output_dim, N]
        scale_in = 6.0 / np.sqrt(max(1, input_dim))
        self.W_in = (self.rng.normal(0.0, scale_in, size=(hidden_dim, input_dim))).astype(np.float32)
        # Feedforward input is excitatory
        self.W_in = np.abs(self.W_in)
        self.b_soma = self.rng.uniform(19.0, 21.0, hidden_dim).astype(np.float32)
        self.W_out = (self.rng.normal(0.0, 0.2, size=(output_dim, hidden_dim))).astype(np.float32)

        # 5. Axonal Conduction Delays based on 3D Euclidean Distances
        diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        self.distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)
        # Delay in steps: latency = max(1, round(distance / (speed * dt)))
        self.delay_matrix = np.maximum(
            1, np.round(self.distance_matrix / (self.propagation_speed * self.dt_ms))
        ).astype(np.int32)

        # 6. Propagation Queue with separate I_E and I_I accounting
        max_del = int(np.max(self.delay_matrix)) + 2
        self.queue = EISpikePropagationQueue(num_neurons=hidden_dim, max_delay=max_del)

        # 7. Homeostatic Synaptic Scaling Engine
        self.homeostasis = HomeostaticSynapticScaling(
            num_neurons=hidden_dim,
            target_rate_hz=target_rate_hz,
            tau_homeo_ms=tau_homeo_ms,
            eta_homeo=eta_homeo,
            w_max_e=weight_scale_e * 3.0,
            w_max_i=weight_scale_i * 3.0,
            dt_ms=dt_ms,
            is_inhibitory=self.is_inhibitory,
            mode=homeostasis_mode,
            enabled=homeostasis_enabled,
            seed=seed,
        )

        # 8. STDP Engine with Dale Sign Constraint
        self.stdp_engine = STDPEngine(
            num_neurons=hidden_dim,
            mode=plasticity_mode,
            w_min=0.0,
            w_max=weight_scale_e * 3.0,
            w_max_i=weight_scale_i * 3.0,
            is_inhibitory=self.is_inhibitory,
        )

        # State tracking
        self.current_step = 0
        self.last_spikes = np.zeros(hidden_dim, dtype=bool)
        self.last_plastic_updates: List[STDPUpdateRecord] = []
        self.last_homeostatic_telemetry: Dict[str, Any] = {}

        # Current accounting buffers
        self.last_i_e = np.zeros(hidden_dim, dtype=np.float32)
        self.last_i_i = np.zeros(hidden_dim, dtype=np.float32)
        self.last_i_net = np.zeros(hidden_dim, dtype=np.float32)

        # Historical current tracking for balance analysis
        self.history_i_e: List[np.ndarray] = []
        self.history_i_i: List[np.ndarray] = []
        self.history_spikes: List[np.ndarray] = []

        # Activations (leaky smoothed firing rates) for readout compatibility
        self.activations = np.zeros(hidden_dim, dtype=np.float32)
        self.act_decay = float(np.exp(-self.dt_ms / 20.0))

    def reset(self) -> None:
        self.population.reset()
        self.queue.reset()
        self.homeostasis.reset()
        self.stdp_engine.reset()
        self.W_rec = self.initial_W_rec.copy()
        self.current_step = 0
        self.last_spikes.fill(False)
        self.last_plastic_updates.clear()
        self.last_i_e.fill(0.0)
        self.last_i_i.fill(0.0)
        self.last_i_net.fill(0.0)
        self.history_i_e.clear()
        self.history_i_i.clear()
        self.history_spikes.clear()
        self.activations.fill(0.0)

    def step(self, x: np.ndarray, action: int = 0) -> np.ndarray:
        """Advance the E/I balanced substrate by one simulation step dt."""
        x = np.asarray(x, dtype=np.float32)
        if len(x) < self.input_dim:
            padded_x = np.zeros(self.input_dim, dtype=np.float32)
            padded_x[: len(x)] = x
            x = padded_x
        elif len(x) > self.input_dim:
            x = x[: self.input_dim]

        # 1. Advance queue: receive arrived excitatory and inhibitory currents
        i_e_arrived, i_i_arrived, i_net_recurrent = self.queue.advance_step()

        # 2. External feedforward input current + tonic somatic drive
        i_ext = np.dot(self.W_in, x) + self.b_soma  # [N]
        i_e_total = i_e_arrived + np.maximum(0.0, i_ext)
        i_i_total = i_i_arrived  # Negative (<= 0)
        i_total = i_e_total + i_i_total

        self.last_i_e = i_e_total.astype(np.float32)
        self.last_i_i = i_i_total.astype(np.float32)
        self.last_i_net = i_total.astype(np.float32)

        self.history_i_e.append(self.last_i_e.copy())
        self.history_i_i.append(self.last_i_i.copy())

        # 3. Step heterogeneous neuron population membrane potentials & detect action potentials
        spikes = self.population.step(i_total, dt=self.dt_ms, current_step=self.current_step)
        self.last_spikes = spikes
        self.history_spikes.append(spikes.copy())

        # 4. Enqueue spikes into propagation queue with axonal conduction delays
        fired_indices = np.where(spikes)[0]
        if len(fired_indices) > 0:
            src_list, dst_list, w_list, del_list, dist_list, is_inh_list = [], [], [], [], [], []
            for src in fired_indices:
                post_targets = np.where(self.adjacency_mask[:, src])[0]
                if len(post_targets) > 0:
                    weights = self.W_rec[post_targets, src]
                    delays = self.delay_matrix[src, post_targets]
                    distances = self.distance_matrix[src, post_targets]
                    src_inh = self.is_inhibitory[src]

                    src_list.extend([src] * len(post_targets))
                    dst_list.extend(post_targets.tolist())
                    w_list.extend(weights.tolist())
                    del_list.extend(delays.tolist())
                    dist_list.extend(distances.tolist())
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

        # 5. Apply Homeostatic Synaptic Scaling
        self.W_rec, homeo_telemetry = self.homeostasis.update(spikes, self.W_rec, step=self.current_step)
        self.last_homeostatic_telemetry = homeo_telemetry

        # 6. Apply STDP (transposing to match STDPEngine [pre, post] convention)
        if self.plasticity_mode != PlasticityMode.FROZEN:
            W_stdp_in = self.W_rec.T  # [pre, post]
            adj_stdp_in = self.adjacency_mask.T
            W_stdp_out, records = self.stdp_engine.step_plasticity(
                weights=W_stdp_in,
                adjacency_mask=adj_stdp_in,
                spikes=spikes,
                step=self.current_step,
                dt=self.dt_ms,
            )
            self.W_rec = W_stdp_out.T  # Back to [dst, src]
            self.last_plastic_updates = records

        # 7. Update smoothed activations for linear readout
        self.activations = self.act_decay * self.activations + (1.0 - self.act_decay) * spikes.astype(np.float32)
        y = np.dot(self.W_out, self.activations)

        self.current_step += 1
        return y

    def observable_state(self) -> Dict[str, Any]:
        """Verify and expose full computational state without side effects."""
        v_mem = self.population.v if hasattr(self.population, "v") else np.zeros(self.hidden_dim)
        return {
            "state": np.expand_dims(v_mem, 0),
            "activation": np.expand_dims(self.activations, 0),
            "spikes": self.last_spikes.tolist(),
            "neuron_types": self.neuron_types.tolist(),
            "i_e": self.last_i_e.tolist(),
            "i_i": self.last_i_i.tolist(),
            "i_net": self.last_i_net.tolist(),
            "mean_i_e": float(np.mean(self.last_i_e)),
            "mean_i_i": float(np.mean(self.last_i_i)),
            "mean_i_net": float(np.mean(self.last_i_net)),
            "homeostatic_scale_mean": self.last_homeostatic_telemetry.get("mean_scale_factor", 1.0),
        }

    def parameter_count(self) -> int:
        return int(np.sum(self.adjacency_mask) + self.W_in.size + self.W_out.size)

    def forward_sequence(
        self,
        X: np.ndarray,
        hidden_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return {
            "W_in": np.zeros_like(self.W_in),
            "W_rec": np.zeros_like(self.W_rec),
            "W_out": np.zeros_like(self.W_out),
        }

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            "W_in": self.W_in.copy(),
            "W_rec": self.W_rec.copy(),
            "W_hh": self.W_rec.copy(),
            "W_out": self.W_out.copy(),
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        if "W_in" in weights:
            self.W_in = weights["W_in"].copy()
        if "W_rec" in weights:
            self.W_rec = weights["W_rec"].copy() * self.adjacency_mask
        if "W_out" in weights:
            self.W_out = weights["W_out"].copy()

    @property
    def spatial_geometry(self) -> Any:
        return type("SpatialGeom", (), {"coordinates": self.positions, "module_ids": self.module_ids})()


class EIBalancedLatentSpatialNetwork(BrainModel):
    """Two-tier architecture uniting fast Dale E/I spiking dynamics with slow persistent latent memory."""

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 40,
        latent_dim: int = 24,
        output_dim: int = 1,
        neuron_model: SpikingNeuronModel = SpikingNeuronModel.LIF,
        topology_type: TopologyFamily = TopologyFamily.MODULAR,
        inhibitory_fraction: float = 0.20,
        homeostasis_enabled: bool = True,
        plasticity_mode: PlasticityMode = PlasticityMode.STDP,
        dt_ms: float = 1.0,
        seed: int = 42,
        break_dale: bool = False,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ):
        if input_dim is None and obs_dim is not None:
            input_dim = obs_dim
        if input_dim is None:
            input_dim = 8

        self.input_dim = input_dim
        self.action_dim = action_dim if action_dim is not None else 4
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Fast Tier: Balanced E/I Spiking Spatial Network
        self.ei_network = EIBalancedSpatialNetwork(
            input_dim=input_dim + latent_dim,  # Accepts sensory input + persistent latent feedback
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            neuron_model=neuron_model,
            topology_type=topology_type,
            inhibitory_fraction=inhibitory_fraction,
            plasticity_mode=plasticity_mode,
            homeostasis_enabled=homeostasis_enabled,
            dt_ms=dt_ms,
            seed=seed,
            break_dale=break_dale,
        )

        # Slow Tier: Persistent Latent Memory Module from Phase 6/7
        self.latent_module = PersistentLatentModule(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            obs_dim=input_dim,
            action_dim=self.action_dim,
            seed=seed,
        )

        self.rng = np.random.default_rng(seed)
        self.W_out = (self.rng.normal(0.0, 0.2, size=(output_dim, hidden_dim + latent_dim))).astype(np.float32)

    def reset(self) -> None:
        self.ei_network.reset()
        self.latent_module.reset()

    def step(self, x: np.ndarray, action: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        # Latent feedback into spiking layer
        combined_input = np.concatenate([x, self.latent_module.z])
        # Step balanced spiking tier
        spiking_h = self.ei_network.step(combined_input)
        # Step persistent latent module
        z, _ = self.latent_module.update_latent_state(self.ei_network.activations, action=action)

        combined_state = np.concatenate([self.ei_network.activations, z])
        y = np.dot(self.W_out, combined_state)
        return y

    def forward_sequence(
        self,
        X: np.ndarray,
        hidden_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size, seq_len, _ = X.shape
        Y = np.zeros((batch_size, seq_len, self.output_dim), dtype=np.float32)
        H = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)
        A = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)

        for b in range(batch_size):
            self.reset()
            for t in range(seq_len):
                y_t = self.step(X[b, t, :])
                Y[b, t, :] = y_t
                H[b, t, :] = self.ei_network.population.v if hasattr(self.ei_network.population, "v") else np.zeros(self.hidden_dim)
                A[b, t, :] = self.ei_network.activations

        return Y, H, A

    def backward_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        cache: dict,
    ) -> Dict[str, np.ndarray]:
        return {
            "W_out": np.zeros_like(self.W_out),
        }

    def get_weights(self) -> Dict[str, np.ndarray]:
        w = self.ei_network.get_weights()
        w["W_out"] = self.W_out.copy()
        return w

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.ei_network.set_weights(weights)
        if "W_out" in weights:
            self.W_out = weights["W_out"].copy()

    def observable_state(self) -> Dict[str, Any]:
        obs = self.ei_network.observable_state()
        obs["latent_z"] = self.latent_module.z.tolist()
        obs["latent_norm"] = float(np.linalg.norm(self.latent_module.z))
        return obs

    @property
    def spatial_geometry(self) -> Any:
        return self.ei_network.spatial_geometry

    def parameter_count(self) -> int:
        return self.ei_network.parameter_count() + self.W_out.size
