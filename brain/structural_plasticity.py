"""Dynamic Structural Plasticity and Brain Growth for 3D-Brain Phase 5.

Implements controlled synaptogenesis, activity-dependent pruning, and neuron recruitment:
- Synapse Formation: distance-dependent, co-activity driven candidate creation
- Synapse Pruning: weight-decay and inactivity driven elimination
- Resource Constraints: strict bounds on degree (k_min, k_max) and edge count (E_max)
- Dynamic Neuron Growth (Dynamic-N): controlled recruitment under persistent prediction error
- Deterministic execution with fixed seeds
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class StructuralEvent:
    """Record of a discrete structural topology modification."""
    event_type: str  # "SYNAPSE_FORMED", "SYNAPSE_PRUNED", "NEURON_BORN", "NEURON_PRUNED"
    step: int
    src: Optional[int] = None
    dst: Optional[int] = None
    weight: Optional[float] = None
    distance: Optional[float] = None
    delay: Optional[int] = None
    neuron_id: Optional[int] = None
    pos: Optional[List[float]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"type": self.event_type, "step": self.step}
        if self.src is not None:
            d["src"] = self.src
        if self.dst is not None:
            d["dst"] = self.dst
        if self.weight is not None:
            d["weight"] = round(float(self.weight), 4)
        if self.distance is not None:
            d["distance"] = round(float(self.distance), 3)
        if self.delay is not None:
            d["delay"] = int(self.delay)
        if self.neuron_id is not None:
            d["neuron_id"] = int(self.neuron_id)
        if self.pos is not None:
            d["pos"] = [round(float(p), 3) for p in self.pos]
        if self.details:
            d["details"] = self.details
        return d


class StructuralPlasticity:
    """Manager for dynamic synaptic rewiring and neuron population growth."""

    def __init__(
        self,
        update_interval: int = 15,
        prune_threshold: float = 0.05,
        prune_fraction: float = 0.05,
        form_fraction: float = 0.05,
        min_degree: int = 2,
        max_degree: int = 25,
        max_edges: Optional[int] = None,
        sigma_dist: float = 6.0,
        enable_growth: bool = False,
        growth_threshold: float = 0.35,
        growth_cooldown: int = 40,
        max_neurons: Optional[int] = None,
        propagation_speed: float = 5.0,
        seed: int = 42,
    ):
        self.update_interval = int(update_interval)
        self.prune_threshold = float(prune_threshold)
        self.prune_fraction = float(prune_fraction)
        self.form_fraction = float(form_fraction)
        self.min_degree = int(min_degree)
        self.max_degree = int(max_degree)
        self.max_edges = max_edges
        self.sigma_dist = float(sigma_dist)

        # Dynamic Growth Parameters
        self.enable_growth = bool(enable_growth)
        self.growth_threshold = float(growth_threshold)
        self.growth_cooldown = int(growth_cooldown)
        self.max_neurons = max_neurons
        self.propagation_speed = float(propagation_speed)
        self.last_growth_step = -growth_cooldown

        self.rng = np.random.default_rng(seed)

        # Diagnostics & event tracking
        self.events_history: List[StructuralEvent] = []
        self.recent_events: List[StructuralEvent] = []
        self.activity_corr_accumulator: Optional[np.ndarray] = None
        self.corr_sample_count = 0

    def reset(self, hidden_dim: int) -> None:
        self.events_history.clear()
        self.recent_events.clear()
        self.activity_corr_accumulator = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
        self.corr_sample_count = 0
        self.last_growth_step = -self.growth_cooldown

    def record_activity(self, activations: np.ndarray) -> None:
        """Accumulate co-activity correlations across units for informed synaptogenesis."""
        if self.activity_corr_accumulator is None or self.activity_corr_accumulator.shape[0] != len(activations):
            self.activity_corr_accumulator = np.zeros((len(activations), len(activations)), dtype=np.float32)
            self.corr_sample_count = 0

        # Outer product of activations
        coact = np.outer(activations, activations)
        self.activity_corr_accumulator += coact
        self.corr_sample_count += 1

    def step(
        self,
        step: int,
        W_hh: np.ndarray,
        adjacency_mask: np.ndarray,
        positions: np.ndarray,
        dist_matrix: np.ndarray,
        delay_matrix: np.ndarray,
        mean_recent_error: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[StructuralEvent]]:
        """Perform periodic structural adaptation step.

        Returns:
            updated W_hh, adjacency_mask, positions, dist_matrix, delay_matrix, structural_events
        """
        self.recent_events.clear()
        if step % self.update_interval != 0 or step == 0:
            return W_hh, adjacency_mask, positions, dist_matrix, delay_matrix, []

        H = W_hh.shape[0]
        max_edges_budget = self.max_edges if self.max_edges is not None else int(H * 15)

        # 1. SYNAPSE PRUNING
        # Candidate edges to prune: low magnitude weights |w_{ij}| < prune_threshold
        # subject to minimum degree constraint per node
        in_degrees = np.sum(adjacency_mask, axis=0)  # Post-synaptic degree
        out_degrees = np.sum(adjacency_mask, axis=1)  # Pre-synaptic degree

        active_srcs, active_dsts = np.where(adjacency_mask)
        num_active = len(active_srcs)

        if num_active > 0:
            active_weights = np.abs(W_hh[active_srcs, active_dsts])
            # Sort edges by weight ascending
            sorted_indices = np.argsort(active_weights)
            max_prune_count = max(1, int(num_active * self.prune_fraction))
            prune_count = 0

            for idx in sorted_indices:
                if prune_count >= max_prune_count:
                    break
                s = active_srcs[idx]
                d = active_dsts[idx]
                w_val = active_weights[idx]

                # Only prune if weight is below threshold or we are over total edge budget
                if w_val < self.prune_threshold or num_active - prune_count > max_edges_budget:
                    # Check degree constraints
                    if out_degrees[s] > self.min_degree and in_degrees[d] > self.min_degree:
                        adjacency_mask[s, d] = False
                        W_hh[s, d] = 0.0
                        out_degrees[s] -= 1
                        in_degrees[d] -= 1
                        prune_count += 1

                        evt = StructuralEvent(
                            event_type="SYNAPSE_PRUNED",
                            step=step,
                            src=int(s),
                            dst=int(d),
                            weight=float(w_val),
                            distance=float(dist_matrix[s, d]),
                            delay=int(delay_matrix[s, d]),
                            details={"reason": "low_weight" if w_val < self.prune_threshold else "budget_cap"},
                        )
                        self.recent_events.append(evt)
                        self.events_history.append(evt)

        # 2. SYNAPSE FORMATION (SYNAPTOGENESIS)
        # Compute formation affinities based on spatial proximity and co-activity
        if self.corr_sample_count > 0:
            avg_coact = self.activity_corr_accumulator / max(1, self.corr_sample_count)
        else:
            avg_coact = np.zeros((H, H), dtype=np.float32)

        # Distance penalty: exp(-d^2 / (2 sigma^2))
        spatial_affinity = np.exp(-(dist_matrix ** 2) / (2.0 * self.sigma_dist ** 2))
        # Mask out self-connections and already existent connections
        candidate_mask = (~adjacency_mask) & (~np.eye(H, dtype=bool))
        # Mask out nodes exceeding maximum degree
        node_avail_src = out_degrees < self.max_degree
        node_avail_dst = in_degrees < self.max_degree
        valid_candidates = candidate_mask & np.outer(node_avail_src, node_avail_dst)

        cand_srcs, cand_dsts = np.where(valid_candidates)
        if len(cand_srcs) > 0:
            # Formation score = spatial_affinity * (0.5 + 0.5 * abs(coact))
            scores = spatial_affinity[cand_srcs, cand_dsts] * (0.5 + 0.5 * np.abs(avg_coact[cand_srcs, cand_dsts]))
            # Add stochastic exploration
            noise = self.rng.uniform(0.8, 1.2, size=len(scores))
            scores = scores * noise

            max_form_count = max(1, int(H * self.form_fraction))
            top_cand_indices = np.argsort(scores)[::-1][:max_form_count]

            scale_init = np.sqrt(2.0 / H)
            for c_idx in top_cand_indices:
                current_total_edges = np.sum(adjacency_mask)
                if current_total_edges >= max_edges_budget:
                    break

                s = cand_srcs[c_idx]
                d = cand_dsts[c_idx]
                if out_degrees[s] < self.max_degree and in_degrees[d] < self.max_degree:
                    # Form new synapse with small exploratory weight
                    init_w = float(self.rng.normal(0.0, scale_init * 0.5))
                    adjacency_mask[s, d] = True
                    W_hh[s, d] = init_w
                    out_degrees[s] += 1
                    in_degrees[d] += 1

                    dist = float(dist_matrix[s, d])
                    tau = int(delay_matrix[s, d])

                    evt = StructuralEvent(
                        event_type="SYNAPSE_FORMED",
                        step=step,
                        src=int(s),
                        dst=int(d),
                        weight=init_w,
                        distance=dist,
                        delay=tau,
                        details={"score": float(scores[c_idx])},
                    )
                    self.recent_events.append(evt)
                    self.events_history.append(evt)

        # Reset co-activity accumulator for next interval
        self.activity_corr_accumulator.fill(0.0)
        self.corr_sample_count = 0

        # 3. DYNAMIC NEURON GROWTH (DYNAMIC-N)
        # Evaluated when enabled, error is high, and cooldown has elapsed
        if (
            self.enable_growth
            and (self.max_neurons is None or H < self.max_neurons)
            and mean_recent_error >= self.growth_threshold
            and (step - self.last_growth_step) >= self.growth_cooldown
        ):
            # Recruit new neuron
            new_id = H
            self.last_growth_step = step

            # Place neuron in 3D metric volume near center with small random dispersion
            new_pos = self.rng.uniform(-6.0, 6.0, size=3).astype(np.float32)
            positions = np.vstack([positions, new_pos[None, :]])

            # Expand matrices: dist_matrix, delay_matrix, adjacency_mask, W_hh
            new_distances = np.linalg.norm(positions[:-1] - new_pos, axis=1).astype(np.float32)
            new_delays = np.maximum(1, np.round(new_distances / self.propagation_speed)).astype(np.int32)

            # Update dist_matrix
            new_dist_matrix = np.zeros((H + 1, H + 1), dtype=np.float32)
            new_dist_matrix[:H, :H] = dist_matrix
            new_dist_matrix[:H, H] = new_distances
            new_dist_matrix[H, :H] = new_distances
            dist_matrix = new_dist_matrix

            # Update delay_matrix
            new_delay_matrix = np.ones((H + 1, H + 1), dtype=np.int32)
            new_delay_matrix[:H, :H] = delay_matrix
            new_delay_matrix[:H, H] = new_delays
            new_delay_matrix[H, :H] = new_delays
            delay_matrix = new_delay_matrix

            # Update adjacency_mask and W_hh
            new_adj = np.zeros((H + 1, H + 1), dtype=bool)
            new_adj[:H, :H] = adjacency_mask
            adjacency_mask = new_adj

            new_W = np.zeros((H + 1, H + 1), dtype=np.float32)
            new_W[:H, :H] = W_hh
            W_hh = new_W

            # Form initial connections to k_min nearest neighbors
            nearest_k = np.argsort(new_distances)[: min(self.min_degree + 1, H)]
            scale_init = np.sqrt(2.0 / (H + 1))
            for n_idx in nearest_k:
                # Add bidirectional or directed connections
                w_in = float(self.rng.normal(0.0, scale_init))
                w_out = float(self.rng.normal(0.0, scale_init))
                adjacency_mask[n_idx, new_id] = True
                W_hh[n_idx, new_id] = w_in
                adjacency_mask[new_id, n_idx] = True
                W_hh[new_id, n_idx] = w_out

            evt_birth = StructuralEvent(
                event_type="NEURON_BORN",
                step=step,
                neuron_id=new_id,
                pos=[float(p) for p in new_pos],
                details={"mean_error": float(mean_recent_error), "new_dim": H + 1},
            )
            self.recent_events.append(evt_birth)
            self.events_history.append(evt_birth)

        return W_hh, adjacency_mask, positions, dist_matrix, delay_matrix, self.recent_events
