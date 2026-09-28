"""Spatial 3D Neural Architecture, Metric Geometry, Propagation Queue, and Spatial Substrates.

Implements the formal spatial computing model for Phase 4:
- 3D metric positioning and distance computation
- Local, Modular, Small-World, and Random topology families with controlled parameter budgets
- Finite propagation speed, latency discretization, and signal propagation queue
- Recurrent spatial substrate with online synaptic plasticity and observable dynamics
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.base import BrainModel
from brain.plasticity import FrozenPlasticity, PlasticityRule


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


@dataclass
class SpatialNeuron:
    """Represents a single neuron situated in 3D metric space."""
    id: int
    pos: np.ndarray  # [x, y, z] in metric units
    module_id: int = 0
    state: float = 0.0
    activation: float = 0.0
    unit_type: str = "RECURRENT"  # "INPUT", "RECURRENT", "OUTPUT"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pos": [round(float(p), 3) for p in self.pos],
            "module_id": int(self.module_id),
            "state": round(float(self.state), 4),
            "activation": round(float(self.activation), 4),
            "unit_type": self.unit_type,
            "metadata": self.metadata,
        }


@dataclass
class SpatialSynapse:
    """Represents a directed synaptic connection between two spatial neurons."""
    source: int
    target: int
    weight: float
    distance: float
    propagation_delay: int  # in simulation time steps (>= 1)
    active: bool = True
    initial_weight: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(float(self.weight), 4),
            "distance": round(float(self.distance), 3),
            "propagation_delay": int(self.propagation_delay),
            "active": self.active,
            "initial_weight": round(float(self.initial_weight), 4),
        }


@dataclass
class InTransitSignal:
    """Represents a signal actively propagating along a synaptic connection."""
    source: int
    target: int
    emission_step: int
    arrival_step: int
    distance: float
    delay: int
    magnitude: float

    def progress(self, current_step: int) -> float:
        """Returns travel progress in [0.0, 1.0]."""
        if self.delay <= 0:
            return 1.0
        elapsed = current_step - self.emission_step
        return float(np.clip(elapsed / self.delay, 0.0, 1.0))

    def to_dict(self, current_step: int) -> dict:
        return {
            "src": self.source,
            "dst": self.target,
            "emission_step": self.emission_step,
            "arrival_step": self.arrival_step,
            "distance": round(float(self.distance), 2),
            "delay": int(self.delay),
            "mag": round(float(self.magnitude), 4),
            "progress": round(self.progress(current_step), 2),
        }


class TopologyFamily(str, Enum):
    RANDOM = "RANDOM"
    LOCAL = "LOCAL"
    MODULAR = "MODULAR"
    SMALL_WORLD = "SMALL_WORLD"


class SpatialGeometry:
    """Deterministic 3D geometric placement generators for neural substrates."""

    @staticmethod
    def uniform_sphere(n: int, radius: float = 10.0, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Distribute n neurons uniformly inside a sphere of given radius."""
        rng = np.random.default_rng(seed)
        # Uniform sampling inside sphere
        phi = rng.uniform(0, 2 * np.pi, n)
        costheta = rng.uniform(-1, 1, n)
        u = rng.uniform(0, 1, n)
        theta = np.arccos(costheta)
        r = radius * (u ** (1.0 / 3.0))

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        positions = np.stack([x, y, z], axis=1).astype(np.float32)
        module_ids = np.zeros(n, dtype=np.int32)
        return positions, module_ids

    @staticmethod
    def clustered_gaussian(
        n: int,
        num_clusters: int = 4,
        cluster_radius: float = 12.0,
        cluster_std: float = 2.0,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Distribute neurons into distinct Gaussian clusters in 3D space."""
        rng = np.random.default_rng(seed)
        # Place cluster centers symmetrically
        cluster_centers = []
        for c in range(num_clusters):
            angle = 2.0 * np.pi * c / num_clusters
            cx = cluster_radius * np.cos(angle)
            cy = cluster_radius * np.sin(angle)
            cz = (c % 2 * 2 - 1) * (cluster_radius * 0.3)
            cluster_centers.append([cx, cy, cz])
        cluster_centers = np.array(cluster_centers, dtype=np.float32)

        positions = np.zeros((n, 3), dtype=np.float32)
        module_ids = np.zeros(n, dtype=np.int32)

        units_per_cluster = n // num_clusters
        for i in range(n):
            c_idx = min(i // units_per_cluster, num_clusters - 1)
            module_ids[i] = c_idx
            noise = rng.normal(0.0, cluster_std, 3).astype(np.float32)
            positions[i] = cluster_centers[c_idx] + noise

        return positions, module_ids

    @staticmethod
    def modular_spatial(
        n: int,
        num_modules: int = 4,
        module_separation: float = 14.0,
        module_std: float = 2.0,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Distribute neurons into well-separated spatial modules."""
        return SpatialGeometry.clustered_gaussian(
            n=n,
            num_clusters=num_modules,
            cluster_radius=module_separation,
            cluster_std=module_std,
            seed=seed,
        )


class SpatialConnectivityBuilder:
    """Constructs controlled connectivity topologies matching target connection budgets."""

    @staticmethod
    def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
        """Compute all-pairs Euclidean distance matrix d_ij = ||x_i - x_j||."""
        # positions: [N, 3]
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)
        return dist

    @staticmethod
    def build_topology(
        positions: np.ndarray,
        module_ids: np.ndarray,
        topology_type: TopologyFamily,
        target_edges: int,
        propagation_speed: float = 5.0,  # distance units per simulation time step
        seed: int = 42,
        local_sigma: float = 4.0,
        rewiring_prob: float = 0.10,
        inter_module_ratio: float = 0.12,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct directed connectivity graph matching exact target edge count.

        Returns:
            adjacency_mask: [N, N] boolean matrix (True where directed edge i -> j exists)
            distances: [N, N] float32 matrix with pairwise Euclidean distances
            delays: [N, N] int32 matrix with propagation delays in discrete time steps (>= 1)
        """
        N = len(positions)
        rng = np.random.default_rng(seed)
        distances = SpatialConnectivityBuilder.compute_distance_matrix(positions)

        # Discretize delay: delay_ij = max(1, round(distance_ij / propagation_speed))
        if propagation_speed <= 0.0 or np.isinf(propagation_speed):
            delays = np.ones((N, N), dtype=np.int32)
        else:
            delays = np.maximum(1, np.round(distances / propagation_speed).astype(np.int32))

        # We exclude self-loops: i != j
        candidate_pairs = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    candidate_pairs.append((i, j))

        total_possible = len(candidate_pairs)
        target_edges = min(target_edges, total_possible)

        adjacency_mask = np.zeros((N, N), dtype=bool)

        if topology_type == TopologyFamily.RANDOM:
            # Connectivity is independent of spatial metric distance
            chosen_indices = rng.choice(total_possible, size=target_edges, replace=False)
            for idx in chosen_indices:
                i, j = candidate_pairs[idx]
                adjacency_mask[i, j] = True

        elif topology_type == TopologyFamily.LOCAL:
            # Distance-dependent exponential kernel: P(i -> j) ~ exp(-d^2 / 2sigma^2)
            dists = np.array([distances[i, j] for i, j in candidate_pairs], dtype=np.float32)
            probs = np.exp(-(dists ** 2) / (2.0 * (local_sigma ** 2)))
            probs /= np.sum(probs)
            chosen_indices = rng.choice(total_possible, size=target_edges, replace=False, p=probs)
            for idx in chosen_indices:
                i, j = candidate_pairs[idx]
                adjacency_mask[i, j] = True

        elif topology_type == TopologyFamily.MODULAR:
            # Higher probability within same module_id, sparse cross-module links
            same_module_pairs = []
            diff_module_pairs = []
            for i, j in candidate_pairs:
                if module_ids[i] == module_ids[j]:
                    same_module_pairs.append((i, j))
                else:
                    diff_module_pairs.append((i, j))

            num_inter = int(round(target_edges * inter_module_ratio))
            num_inter = min(num_inter, len(diff_module_pairs))
            num_intra = target_edges - num_inter
            num_intra = min(num_intra, len(same_module_pairs))

            # If intra not enough, fill with inter
            if num_intra < target_edges - num_inter:
                num_inter = min(target_edges - num_intra, len(diff_module_pairs))

            # Intra-module connections chosen with local distance bias
            if same_module_pairs and num_intra > 0:
                intra_dists = np.array([distances[i, j] for i, j in same_module_pairs], dtype=np.float32)
                intra_probs = np.exp(-(intra_dists ** 2) / (2.0 * (local_sigma ** 2)))
                intra_probs /= np.sum(intra_probs)
                chosen_intra = rng.choice(len(same_module_pairs), size=num_intra, replace=False, p=intra_probs)
                for idx in chosen_intra:
                    i, j = same_module_pairs[idx]
                    adjacency_mask[i, j] = True

            # Inter-module connections chosen uniformly or sparse
            if diff_module_pairs and num_inter > 0:
                chosen_inter = rng.choice(len(diff_module_pairs), size=num_inter, replace=False)
                for idx in chosen_inter:
                    i, j = diff_module_pairs[idx]
                    adjacency_mask[i, j] = True

        elif topology_type == TopologyFamily.SMALL_WORLD:
            # Watts-Strogatz inspired spatial small-world:
            # (1 - beta) local connections + beta long-range shortcuts
            num_shortcuts = int(round(target_edges * rewiring_prob))
            num_local = target_edges - num_shortcuts

            dists = np.array([distances[i, j] for i, j in candidate_pairs], dtype=np.float32)
            local_probs = np.exp(-(dists ** 2) / (2.0 * (local_sigma ** 2)))
            local_probs /= np.sum(local_probs)

            chosen_local = rng.choice(total_possible, size=num_local, replace=False, p=local_probs)
            for idx in chosen_local:
                i, j = candidate_pairs[idx]
                adjacency_mask[i, j] = True

            # Shortcuts: select from remaining unselected pairs preferentially with larger distance or uniform
            unselected_mask = np.ones(total_possible, dtype=bool)
            unselected_mask[chosen_local] = False
            unselected_indices = np.where(unselected_mask)[0]

            if len(unselected_indices) >= num_shortcuts:
                chosen_shortcuts = rng.choice(unselected_indices, size=num_shortcuts, replace=False)
                for idx in chosen_shortcuts:
                    i, j = candidate_pairs[idx]
                    adjacency_mask[i, j] = True

        return adjacency_mask, distances, delays


class SignalPropagationQueue:
    """High-performance circular ring-buffer propagation queue with finite transmission latency."""

    def __init__(self, num_neurons: int, max_delay: int = 50, track_in_transit: bool = True):
        self.num_neurons = num_neurons
        self.max_delay = max(1, max_delay)
        self.buffer_size = self.max_delay + 2
        self.track_in_transit = track_in_transit

        # Circular buffer accumulating incoming weighted activations arriving at future steps
        # Shape: [buffer_size, num_neurons]
        self.ring_buffer = np.zeros((self.buffer_size, num_neurons), dtype=np.float32)
        self.current_step = 0

        # Detailed in-transit signal records for observability/telemetry
        self.active_signals: List[InTransitSignal] = []

    def reset(self) -> None:
        self.ring_buffer.fill(0.0)
        self.current_step = 0
        self.active_signals.clear()

    def advance_step(self) -> np.ndarray:
        """Advance time by one simulation step and return arrived recurrent inputs."""
        slot = self.current_step % self.buffer_size
        arrived_inputs = self.ring_buffer[slot].copy()
        # Clear consumed slot for future circular cycles
        self.ring_buffer[slot].fill(0.0)
        self.current_step += 1

        # Prune delivered in-transit signals
        if self.track_in_transit:
            self.active_signals = [s for s in self.active_signals if s.arrival_step >= self.current_step]

        return arrived_inputs

    def enqueue_signals(
        self,
        src_indices: np.ndarray,
        dst_indices: np.ndarray,
        weights: np.ndarray,
        activations: np.ndarray,
        delays: np.ndarray,
        distances: Optional[np.ndarray] = None,
    ) -> None:
        """Enqueue outgoing synaptic signals into the propagation queue.

        Args:
            src_indices: Pre-synaptic unit indices [E]
            dst_indices: Post-synaptic unit indices [E]
            weights: Synaptic connection weights [E]
            activations: Pre-synaptic activations a_i [E]
            delays: Propagation delays in steps tau_ij [E]
            distances: Connection physical distances d_ij [E]
        """
        if len(src_indices) == 0:
            return

        signal_magnitudes = activations * weights  # [E]
        arrival_steps = self.current_step + delays  # [E]
        slots = arrival_steps % self.buffer_size

        # Vectorized accumulation into circular ring buffer
        np.add.at(self.ring_buffer, (slots, dst_indices), signal_magnitudes)

        # If telemetry tracking is active, record significant signals in transit
        if self.track_in_transit:
            for k in range(len(src_indices)):
                mag = float(signal_magnitudes[k])
                if abs(mag) >= 0.04:  # Threshold to avoid logging imperceptible noise
                    src = int(src_indices[k])
                    dst = int(dst_indices[k])
                    d = float(distances[k]) if distances is not None else 1.0
                    delay = int(delays[k])
                    self.active_signals.append(
                        InTransitSignal(
                            source=src,
                            target=dst,
                            emission_step=self.current_step,
                            arrival_step=int(arrival_steps[k]),
                            distance=d,
                            delay=delay,
                            magnitude=mag,
                        )
                    )

    def get_in_transit_signals(self, max_count: int = 100) -> List[InTransitSignal]:
        """Return subset of currently propagating signals for inspection and telemetry."""
        return self.active_signals[:max_count]


class SpatialPlasticRNN(BrainModel):
    """Spatially embedded recurrent neural substrate with metric geometry,

    propagation queue, and online synaptic plasticity.

    Preserves the validated DISC-LRNN leaky dynamical equation while replacing
    unconstrained matrix multiplication with distance-dependent propagation:
        raw_cand(t) = W_xh · x(t) + I_recurrent_arrived(t) + b_h
        cand_h(t) = tanh(raw_cand(t))
        h(t) = (1 - alpha) * h(t-1) + alpha * cand_h(t)
        a(t) = tanh(h(t))
        y(t) = W_hy · a(t) + b_y
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        topology_type: TopologyFamily = TopologyFamily.LOCAL,
        target_edges: Optional[int] = None,
        propagation_speed: float = 5.0,
        positions: Optional[np.ndarray] = None,
        module_ids: Optional[np.ndarray] = None,
        plasticity_rule: Optional[PlasticityRule] = None,
        seed: int = 42,
    ):
        super().__init__(input_dim, hidden_dim, output_dim, seed)
        self.topology_type = topology_type
        self.propagation_speed = float(propagation_speed)
        self.plasticity_rule = plasticity_rule or FrozenPlasticity()

        # 1. 3D Spatial Geometry
        if positions is not None:
            self.positions = positions.astype(np.float32)
            self.module_ids = module_ids if module_ids is not None else np.zeros(hidden_dim, dtype=np.int32)
        else:
            if topology_type == TopologyFamily.MODULAR:
                self.positions, self.module_ids = SpatialGeometry.modular_spatial(hidden_dim, num_modules=4, seed=seed)
            elif topology_type == TopologyFamily.SMALL_WORLD or topology_type == TopologyFamily.LOCAL:
                self.positions, self.module_ids = SpatialGeometry.clustered_gaussian(hidden_dim, num_clusters=4, seed=seed)
            else:
                self.positions, self.module_ids = SpatialGeometry.uniform_sphere(hidden_dim, radius=10.0, seed=seed)

        # 2. Controlled Connectivity Budget
        # Default target connection budget: average degree ~ 20 (or min(H-1, 20))
        default_k = min(20, hidden_dim - 1) if hidden_dim > 1 else 1
        self.target_edges = target_edges if target_edges is not None else hidden_dim * default_k

        # 3. Construct Adjacency, Distances, and Delays
        adj_mask, dist_matrix, delay_matrix = SpatialConnectivityBuilder.build_topology(
            positions=self.positions,
            module_ids=self.module_ids,
            topology_type=topology_type,
            target_edges=self.target_edges,
            propagation_speed=self.propagation_speed,
            seed=seed,
        )
        self.adjacency_mask = adj_mask
        self.distance_matrix = dist_matrix
        self.delay_matrix = delay_matrix

        # Extract active edge indices for fast vectorized indexing
        srcs, dsts = np.where(self.adjacency_mask)
        self.edge_srcs = srcs.astype(np.int32)
        self.edge_dsts = dsts.astype(np.int32)
        self.edge_distances = dist_matrix[srcs, dsts].astype(np.float32)
        self.edge_delays = delay_matrix[srcs, dsts].astype(np.int32)
        self.num_edges = len(self.edge_srcs)
        self.max_delay = int(np.max(self.edge_delays)) if self.num_edges > 0 else 1

        # 4. Synaptic Weights Initialization
        # Xavier/Glorot scaling based on degree
        scale_xh = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hh = np.sqrt(2.0 / (max(1, default_k) + hidden_dim))
        scale_hy = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W_xh = self.rng.normal(0.0, scale_xh, (input_dim, hidden_dim)).astype(np.float32)
        self.W_hh = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
        if self.num_edges > 0:
            init_weights = self.rng.normal(0.0, scale_hh, self.num_edges).astype(np.float32)
            self.W_hh[self.edge_srcs, self.edge_dsts] = init_weights

        self.b_h = np.zeros(hidden_dim, dtype=np.float32)
        self.rho = self.rng.normal(-1.0, 0.2, hidden_dim).astype(np.float32)
        self.W_hy = self.rng.normal(0.0, scale_hy, (hidden_dim, output_dim)).astype(np.float32)
        self.b_y = np.zeros(output_dim, dtype=np.float32)

        self.initial_W_hh = self.W_hh.copy()

        # 5. Propagation Queue
        self.queue = SignalPropagationQueue(hidden_dim, max_delay=self.max_delay, track_in_transit=True)

        # 6. Dynamic Inspection State
        self.prev_h = np.zeros((1, hidden_dim), dtype=np.float32)
        self.cand_h = np.zeros((1, hidden_dim), dtype=np.float32)
        self.arrived_recurrent = np.zeros((1, hidden_dim), dtype=np.float32)
        self.last_dw = np.zeros_like(self.W_hh)
        self.cumulative_abs_dw = np.zeros_like(self.W_hh)
        self.plastic_updates_count = 0
        self.online_plasticity_enabled = True

        self.reset()

    @property
    def alpha(self) -> np.ndarray:
        return sigmoid(self.rho)

    def reset(self, batch_size: int = 1) -> None:
        self.h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.a = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.prev_h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.cand_h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.arrived_recurrent = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.queue.reset()
        self.plasticity_rule.reset()

    def step(self, x: np.ndarray, modulation: Optional[float] = None) -> np.ndarray:
        """Step one simulation time increment.

        1. Advance propagation queue to receive signals that arrive at current step.
        2. Compute neural recurrent update.
        3. Enqueue newly emitted signals into propagation queue with their respective delays.
        4. Apply online local plasticity to active connections if enabled.
        """
        # 1. Fetch arrived signals from propagation queue
        arrived = self.queue.advance_step()  # [hidden_dim]
        self.arrived_recurrent[0] = arrived

        self.prev_h = self.h.copy()
        alpha = self.alpha

        # 2. Recurrent forward update
        # raw_cand = x · W_xh + I_arrived + b_h
        input_drive = np.dot(x, self.W_xh)
        raw_cand = input_drive + arrived + self.b_h
        self.cand_h = np.tanh(raw_cand)
        self.h = (1.0 - alpha) * self.prev_h + alpha * self.cand_h
        self.a = np.tanh(self.h)

        # Readout output
        y = np.dot(self.a, self.W_hy) + self.b_y

        # 3. Enqueue outgoing signals across all active synaptic connections
        if self.num_edges > 0:
            current_act = self.a[0]
            edge_acts = current_act[self.edge_srcs]
            edge_weights = self.W_hh[self.edge_srcs, self.edge_dsts]

            self.queue.enqueue_signals(
                src_indices=self.edge_srcs,
                dst_indices=self.edge_dsts,
                weights=edge_weights,
                activations=edge_acts,
                delays=self.edge_delays,
                distances=self.edge_distances,
            )

        # 4. Online local synaptic plasticity update (constrained to active structural edges)
        if self.online_plasticity_enabled and self.num_edges > 0:
            pre_vec = self.prev_h[0]
            post_vec = self.a[0]
            dw, _ = self.plasticity_rule.compute_delta(pre_vec, post_vec, self.W_hh, modulation)

            # Strictly mask dw by structural adjacency: zero out non-existent edges
            dw = dw * self.adjacency_mask

            self.W_hh += dw
            # Bound weights to prevent instability
            self.W_hh = np.clip(self.W_hh, -5.0, 5.0)

            self.last_dw = dw
            self.cumulative_abs_dw += np.abs(dw)
            self.plastic_updates_count += 1

        return y

    def forward_sequence(
        self, X: np.ndarray, modulations: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process sequence X of shape [batch_size, seq_len, input_dim]."""
        batch_size, seq_len, _ = X.shape
        Y = np.zeros((batch_size, seq_len, self.output_dim), dtype=np.float32)
        H = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)
        A = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)

        for t in range(seq_len):
            x_t = X[:, t, :]
            mod_t = float(modulations[t]) if modulations is not None else None
            y_t = self.step(x_t, modulation=mod_t)
            Y[:, t, :] = y_t
            H[:, t, :] = self.h
            A[:, t, :] = self.a

        return Y, H, A

    def backward_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        cache: dict,
    ) -> Dict[str, np.ndarray]:
        """Offline gradient computation for initial pretraining.

        Note: Recurrent gradient is projected onto active structural connections.
        """
        batch_size, seq_len, _ = X.shape
        H = cache["H"]
        cand_H = cache["cand_H"]
        A = cache["A"]
        alpha = self.alpha

        dY = (Y - targets) * mask
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.sum(dY, axis=(0, 1))
        for t in range(seq_len):
            dW_hy += np.dot(A[:, t, :].T, dY[:, t, :])

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        drho = np.zeros_like(self.rho)

        dh_next = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

        for t in reversed(range(seq_len)):
            da_t = np.dot(dY[:, t, :], self.W_hy.T)
            dtanh_a = 1.0 - A[:, t, :] ** 2
            dh_t = da_t * dtanh_a + dh_next

            h_prev = H[:, t - 1, :] if t > 0 else np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
            c_t = cand_H[:, t, :]

            d_alpha_term = dh_t * (c_t - h_prev)
            drho += np.sum(d_alpha_term * alpha * (1.0 - alpha), axis=0)

            dc_t = dh_t * alpha
            d_raw = dc_t * (1.0 - c_t ** 2)

            db_h += np.sum(d_raw, axis=0)
            dW_xh += np.dot(X[:, t, :].T, d_raw)

            # Recurrent projection
            if t > 0:
                dW_hh += np.dot(A[:, t - 1, :].T, d_raw)
                dh_next = np.dot(d_raw, self.W_hh.T) * (1.0 - A[:, t - 1, :] ** 2) + dh_t * (1.0 - alpha)
            else:
                dh_next = dh_t * (1.0 - alpha)

        # Strictly enforce structural topology on gradients
        dW_hh = dW_hh * self.adjacency_mask

        return {
            "W_xh": dW_xh,
            "W_hh": dW_hh,
            "b_h": db_h,
            "rho": drho,
            "W_hy": dW_hy,
            "b_y": db_y,
        }

    def parameter_count(self) -> int:
        """Total trainable parameters (sparse recurrent connections + I/O weights + biases)."""
        recurrent_params = self.num_edges
        io_params = self.input_dim * self.hidden_dim + self.hidden_dim + self.hidden_dim  # W_xh, b_h, rho
        out_params = self.hidden_dim * self.output_dim + self.output_dim
        return recurrent_params + io_params + out_params

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            "W_xh": self.W_xh.copy(),
            "W_hh": self.W_hh.copy(),
            "b_h": self.b_h.copy(),
            "rho": self.rho.copy(),
            "W_hy": self.W_hy.copy(),
            "b_y": self.b_y.copy(),
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.W_xh = weights["W_xh"].copy()
        self.W_hh = (weights["W_hh"] * self.adjacency_mask).copy()
        self.b_h = weights["b_h"].copy()
        self.rho = weights["rho"].copy()
        self.W_hy = weights["W_hy"].copy()
        self.b_y = weights["b_y"].copy()

    def observable_state(self) -> Dict[str, np.ndarray]:
        in_transit = self.queue.get_in_transit_signals(max_count=40)
        return {
            "state": self.get_state(),
            "activation": self.get_activation(),
            "arrived_recurrent": self.arrived_recurrent.copy(),
            "alpha": self.alpha.copy(),
            "tau": (1.0 / (self.alpha + 1e-7)).copy(),
            "cand_h": self.cand_h.copy(),
            "prev_h": self.prev_h.copy(),
            "mean_abs_dw": np.array([np.mean(np.abs(self.last_dw))], dtype=np.float32),
            "max_dw": np.array([np.max(np.abs(self.last_dw))], dtype=np.float32),
            "weight_norm": np.array([np.linalg.norm(self.W_hh)], dtype=np.float32),
            "in_transit_count": np.array([len(self.queue.active_signals)], dtype=np.float32),
        }
