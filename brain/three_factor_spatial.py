"""Three-Factor Spatially Embedded Spiking Network with Dale's Principle and Eligibility Traces.

Implements the formal Phase 9 unified computational substrate:
- 3D Euclidean spatial geometry with distance-dependent conduction delays
- Dale's Principle: ~80% regular-spiking excitatory, ~20% fast-spiking inhibitory interneurons
- Dual-buffer spike propagation queue with separate I_E and I_I current accounting
- Multiplicative homeostatic synaptic scaling
- Three-Factor Plasticity Engine: Delta w_ij = eta * M(t) * e_ij(t)
- Full support for causal credit assignment ablations (reward delay, eligibility zeroing,
  sign flipping, spatial shuffling, homeostasis ablations, Dale violations)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.base import BrainModel
from brain.spatial import (
    SpatialConnectivityBuilder,
    SpatialGeometry,
    TopologyFamily,
)
from brain.spiking import SpikingNeuronModel
from brain.ei_dynamics import (
    NeuronType,
    EILIFPopulation,
    EIIzhikevichPopulation,
    EISpikePropagationQueue,
    build_dale_connectivity_matrix,
)
from brain.homeostasis import HomeostaticSynapticScaling
from brain.three_factor import (
    EligibilityTraceMode,
    NeuromodulatorMode,
    ThreeFactorPlasticityEngine,
    ThreeFactorUpdateRecord,
)


class ThreeFactorSpatialNetwork(BrainModel):
    """Spatially embedded Dale-constrained E/I spiking network with Three-Factor credit assignment."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 50,
        output_dim: int = 1,
        neuron_model: SpikingNeuronModel = SpikingNeuronModel.LIF,
        topology_type: TopologyFamily = TopologyFamily.MODULAR,
        inhibitory_fraction: float = 0.20,
        density_ee: float = 0.18,
        density_ei: float = 0.28,
        density_ie: float = 0.38,
        density_ii: float = 0.20,
        weight_scale_e: float = 0.5,
        weight_scale_i: float = 1.0,
        propagation_speed: float = 8.0,  # mm / ms
        # Three-factor parameters
        learning_rate: float = 0.005,
        tau_eligibility_ms: float = 50.0,
        eligibility_mode: EligibilityTraceMode = EligibilityTraceMode.COINCIDENCE,
        neuromodulator_mode: NeuromodulatorMode = NeuromodulatorMode.REWARD_PREDICTION_ERROR,
        rpe_baseline_alpha: float = 0.05,
        # Homeostasis
        homeostasis_enabled: bool = True,
        homeostasis_mode: str = "local",
        target_rate_hz: float = 10.0,
        tau_homeo_ms: float = 200.0,
        eta_homeo: float = 0.001,
        # Ablations / Controls
        break_dale: bool = False,
        freeze_excitatory: bool = False,
        freeze_inhibitory: bool = False,
        freeze_recurrent: bool = False,
        dt_ms: float = 1.0,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.neuron_model_type = neuron_model
        self.topology_type = topology_type
        self.inhibitory_fraction = float(np.clip(inhibitory_fraction, 0.0, 1.0))
        self.propagation_speed = max(0.1, float(propagation_speed))
        self.dt_ms = float(dt_ms)
        self.seed = seed
        self.break_dale = break_dale
        self.freeze_excitatory = freeze_excitatory
        self.freeze_inhibitory = freeze_inhibitory
        self.freeze_recurrent = freeze_recurrent

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

        # 2. Heterogeneous E/I Population
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

        # 3. Dale-constrained Recurrent Matrix W_rec [N_post, N_pre]
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

        # 4. Sensory Inputs & Soma Drive
        scale_in = 4.0 / np.sqrt(max(1, input_dim))
        self.W_in = np.abs(self.rng.normal(0.0, scale_in, size=(hidden_dim, input_dim))).astype(np.float32)
        self.b_soma = self.rng.uniform(19.0, 21.0, hidden_dim).astype(np.float32)

        # 5. Readout Matrix W_out [output_dim, hidden_dim] and Eligibility
        self.W_out = self.rng.normal(0.0, 0.15, size=(output_dim, hidden_dim)).astype(np.float32)
        self.initial_W_out = self.W_out.copy()
        self.readout_eligibility = np.zeros_like(self.W_out)
        self.readout_eta = float(learning_rate * 2.0)

        # 6. Conduction Delays
        diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        self.distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)
        self.delay_matrix = np.maximum(
            1, np.round(self.distance_matrix / (self.propagation_speed * self.dt_ms))
        ).astype(np.int32)

        max_del = int(np.max(self.delay_matrix)) + 2
        self.queue = EISpikePropagationQueue(num_neurons=hidden_dim, max_delay=max_del)

        # 7. Homeostasis
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

        # 8. Three-Factor Plasticity Engine
        self.plasticity_engine = ThreeFactorPlasticityEngine(
            num_neurons=hidden_dim,
            learning_rate=learning_rate,
            tau_eligibility_ms=tau_eligibility_ms,
            eligibility_mode=eligibility_mode,
            neuromodulator_mode=neuromodulator_mode,
            rpe_baseline_alpha=rpe_baseline_alpha,
            w_max_e=weight_scale_e * 3.0,
            w_max_i=weight_scale_i * 3.0,
            is_inhibitory=self.is_inhibitory,
            module_ids=self.module_ids,
            dt_ms=dt_ms,
            seed=seed,
        )

        # Dynamic State Tracking
        self.current_step = 0
        self.last_spikes = np.zeros(hidden_dim, dtype=bool)
        self.last_plastic_updates: List[ThreeFactorUpdateRecord] = []
        self.last_homeostatic_telemetry: Dict[str, Any] = {}
        self.last_telemetry: Dict[str, Any] = {}

        self.last_i_e = np.zeros(hidden_dim, dtype=np.float32)
        self.last_i_i = np.zeros(hidden_dim, dtype=np.float32)
        self.last_i_net = np.zeros(hidden_dim, dtype=np.float32)

        self.history_i_e: List[np.ndarray] = []
        self.history_i_i: List[np.ndarray] = []
        self.history_spikes: List[np.ndarray] = []

        self.activations = np.zeros(hidden_dim, dtype=np.float32)
        self.act_decay = float(np.exp(-self.dt_ms / 20.0))

    def reset(self) -> None:
        """Reset internal biophysical states to rest while preserving learned weights."""
        self.population.reset()
        self.queue.reset()
        self.homeostasis.reset()
        self.plasticity_engine.reset()
        self.readout_eligibility.fill(0.0)
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
        self.last_y = np.zeros(self.output_dim, dtype=np.float32)

    def full_reset_weights(self) -> None:
        """Revert weights to initial random initialization."""
        self.reset()
        self.W_rec = self.initial_W_rec.copy()
        self.W_out = self.initial_W_out.copy()

    def step(
        self,
        x: np.ndarray,
        reward: float = 0.0,
        prediction_error: float = 0.0,
        community_rewards: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Advance the three-factor spiking network by one simulation step.

        Args:
            x: Input vector [input_dim]
            reward: Scalar teaching reward signal R(t)
            prediction_error: Prediction error teaching signal
            community_rewards: Optional community-specific rewards
        Returns:
            y: Linear readout vector [output_dim]
        """
        x = np.asarray(x, dtype=np.float32)
        if len(x) < self.input_dim:
            padded_x = np.zeros(self.input_dim, dtype=np.float32)
            padded_x[: len(x)] = x
            x = padded_x
        elif len(x) > self.input_dim:
            x = x[: self.input_dim]

        # 1. Advance spike queue: receive arriving currents
        i_e_arrived, i_i_arrived, _ = self.queue.advance_step()

        # 2. External feedforward drive + tonic somatic drive
        i_ext = np.dot(self.W_in, x) + self.b_soma
        i_e_total = i_e_arrived + np.maximum(0.0, i_ext)
        i_i_total = i_i_arrived  # Negative (<= 0)
        i_total = i_e_total + i_i_total

        self.last_i_e = i_e_total.astype(np.float32)
        self.last_i_i = i_i_total.astype(np.float32)
        self.last_i_net = i_total.astype(np.float32)

        self.history_i_e.append(self.last_i_e.copy())
        self.history_i_i.append(self.last_i_i.copy())

        # 3. Integrate membrane potential and detect spikes
        spikes = self.population.step(i_total, dt=self.dt_ms, current_step=self.current_step)
        self.last_spikes = spikes
        self.history_spikes.append(spikes.copy())

        # 4. Enqueue action potentials with axonal conduction delays
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

        # 5. Homeostatic Synaptic Scaling
        if not self.freeze_recurrent and self.homeostasis.enabled:
            self.W_rec, homeo_telemetry = self.homeostasis.update(spikes, self.W_rec, step=self.current_step)
            self.last_homeostatic_telemetry = homeo_telemetry
        else:
            self.last_homeostatic_telemetry = {}

        # 6. Three-Factor Plasticity on Recurrent Weights
        if not self.freeze_recurrent:
            # W_rec is [post, pre], transpose to [pre, post] for plasticity engine
            W_pre_post = self.W_rec.T
            adj_pre_post = self.adjacency_mask.T

            # If freezing excitatory or inhibitory subsets:
            adj_active = adj_pre_post.copy()
            if self.freeze_excitatory:
                adj_active[self.is_excitatory, :] = False
            if self.freeze_inhibitory:
                adj_active[self.is_inhibitory, :] = False

            new_W_pre_post, records, plast_telemetry = self.plasticity_engine.step(
                weights=W_pre_post,
                adjacency_mask=adj_active,
                spikes=spikes,
                reward=reward,
                prediction_error=prediction_error,
                community_rewards=community_rewards,
                step=self.current_step,
            )
            self.W_rec = new_W_pre_post.T  # Transpose back to [post, pre]
            self.last_plastic_updates = records
            self.last_telemetry = plast_telemetry
        else:
            # When recurrent weights are frozen, update eligibility traces for tracking only
            self.plasticity_engine.update_eligibility(spikes, self.adjacency_mask.T)
            self.last_plastic_updates = []
            self.last_telemetry = {
                "step": self.current_step,
                "mean_eligibility": self.plasticity_engine.last_mean_eligibility,
                "neuromodulator": 0.0,
                "total_weight_change": 0.0,
            }

        # 7. Update smoothed activations
        self.activations = self.act_decay * self.activations + (1.0 - self.act_decay) * spikes.astype(np.float32)

        # 8. Three-Factor Plasticity on Readout Weights
        # Directional readout eligibility: e_out = lambda_e * e_out + outer(tanh(y_prev), activations)
        lambda_e = self.plasticity_engine.lambda_e
        y_prev = getattr(self, "last_y", np.zeros(self.output_dim, dtype=np.float32))
        y_dir = np.tanh(y_prev)
        self.readout_eligibility = lambda_e * self.readout_eligibility + np.outer(
            y_dir, self.activations
        )
        if abs(reward) > 1e-6 and not self.freeze_recurrent:
            # Update readout: Delta W_out = readout_eta * RPE * readout_eligibility
            rpe = self.plasticity_engine.last_rpe if self.plasticity_engine.neuromodulator_mode == NeuromodulatorMode.REWARD_PREDICTION_ERROR else reward
            self.W_out += self.readout_eta * float(rpe) * self.readout_eligibility
            np.clip(self.W_out, -2.0, 2.0, out=self.W_out)

        y = np.dot(self.W_out, self.activations)
        self.last_y = y.copy()
        self.current_step += 1
        return y

    def observable_state(self) -> Dict[str, Any]:
        """Expose full computational state for passive telemetry capture."""
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
            "mean_eligibility": float(self.plasticity_engine.last_mean_eligibility),
            "max_eligibility": float(self.plasticity_engine.last_max_eligibility),
            "neuromodulator": float(self.plasticity_engine.last_raw_reward),
            "rpe": float(self.plasticity_engine.last_rpe),
            "homeostatic_scale_mean": self.last_homeostatic_telemetry.get("mean_scale_factor", 1.0),
        }

    def parameter_count(self) -> int:
        return int(np.count_nonzero(self.W_rec) + self.W_in.size + self.W_out.size + self.b_soma.size)

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
        elif "W_hh" in weights:
            self.W_rec = weights["W_hh"].copy() * self.adjacency_mask
        if "W_out" in weights:
            self.W_out = weights["W_out"].copy()

