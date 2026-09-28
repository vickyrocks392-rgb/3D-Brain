r"""Spatially Embedded Predictive Recurrent Neural Substrate with Dynamic Structural Plasticity.

Implements the formal Phase 5 computational model:
- 3D metric geometry and finite transmission delays
- Ring-buffer signal propagation queue
- Closed sensorimotor loop: (s_t, h_{t-1}) -> a_t -> \hat{s}_{t+1} -> env.step(a_t) -> s_{t+1} -> e_t
- Pluggable PredictiveLearningRule (Frozen, Local, Neuromodulated, Gradient)
- Dynamic StructuralPlasticity (synaptogenesis, pruning, optional neuron recruitment)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.base import BrainModel
from brain.spatial import (
    SignalPropagationQueue,
    SpatialConnectivityBuilder,
    SpatialGeometry,
    TopologyFamily,
    sigmoid,
)
from brain.predictive_plasticity import FrozenPredictive, PredictiveLearningRule
from brain.structural_plasticity import StructuralEvent, StructuralPlasticity


class PredictiveSpatialRNN(BrainModel):
    """Spatially embedded recurrent brain model for self-supervised predictive world modeling."""

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 4,
        hidden_dim: int = 40,
        topology_type: TopologyFamily = TopologyFamily.LOCAL,
        target_edges: Optional[int] = None,
        propagation_speed: float = 5.0,
        positions: Optional[np.ndarray] = None,
        module_ids: Optional[np.ndarray] = None,
        predictive_rule: Optional[PredictiveLearningRule] = None,
        structural_plasticity: Optional[StructuralPlasticity] = None,
        seed: int = 42,
    ):
        super().__init__(input_dim=obs_dim, hidden_dim=hidden_dim, output_dim=obs_dim, seed=seed)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.topology_type = topology_type
        self.propagation_speed = float(propagation_speed)
        self.predictive_rule = predictive_rule or FrozenPredictive()
        self.structural_plasticity = structural_plasticity

        # 1. 3D Spatial Geometry
        if positions is not None:
            self.positions = positions.astype(np.float32)
            self.module_ids = module_ids if module_ids is not None else np.zeros(hidden_dim, dtype=np.int32)
        else:
            if topology_type == TopologyFamily.MODULAR:
                self.positions, self.module_ids = SpatialGeometry.modular_spatial(hidden_dim, num_modules=4, seed=seed)
            elif topology_type in [TopologyFamily.SMALL_WORLD, TopologyFamily.LOCAL]:
                self.positions, self.module_ids = SpatialGeometry.clustered_gaussian(hidden_dim, num_clusters=4, seed=seed)
            else:
                self.positions, self.module_ids = SpatialGeometry.uniform_sphere(hidden_dim, radius=10.0, seed=seed)

        # 2. Connectivity Budget
        default_k = min(15, hidden_dim - 1) if hidden_dim > 1 else 1
        self.target_edges = target_edges if target_edges is not None else hidden_dim * default_k

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

        # Synaptic weights
        scale_xh = np.sqrt(2.0 / (obs_dim + hidden_dim))
        scale_hh = np.sqrt(2.0 / (max(1, default_k) + hidden_dim))

        self.W_xh = self.rng.normal(0.0, scale_xh, (obs_dim, hidden_dim)).astype(np.float32)
        self.W_hh = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)

        srcs, dsts = np.where(self.adjacency_mask)
        if len(srcs) > 0:
            init_weights = self.rng.normal(0.0, scale_hh, len(srcs)).astype(np.float32)
            self.W_hh[srcs, dsts] = init_weights

        self.b_h = np.zeros(hidden_dim, dtype=np.float32)
        self.rho = self.rng.normal(-1.0, 0.2, hidden_dim).astype(np.float32)

        # Action policy weights: action logits from neural activations
        scale_act = np.sqrt(2.0 / (hidden_dim + action_dim))
        self.W_act = self.rng.normal(0.0, scale_act, (hidden_dim, action_dim)).astype(np.float32)
        self.b_act = np.zeros(action_dim, dtype=np.float32)

        # Predictive Head: predicts s_{t+1} from [a_t; u_t]
        scale_pred = np.sqrt(2.0 / (hidden_dim + action_dim + obs_dim))
        self.W_pred = self.rng.normal(0.0, scale_pred, (hidden_dim + action_dim, obs_dim)).astype(np.float32)
        self.b_pred = np.zeros(obs_dim, dtype=np.float32)

        # Ring-buffer signal propagation queue
        self._refresh_edge_cache()
        self.max_delay = int(np.max(self.edge_delays)) if len(self.edge_delays) > 0 else 1
        self.queue = SignalPropagationQueue(hidden_dim, max_delay=max(5, self.max_delay), track_in_transit=True)

        # Dynamic State Variables
        self.prev_h = np.zeros((1, hidden_dim), dtype=np.float32)
        self.cand_h = np.zeros((1, hidden_dim), dtype=np.float32)
        self.arrived_recurrent = np.zeros((1, hidden_dim), dtype=np.float32)
        self.last_pred_s = np.zeros(obs_dim, dtype=np.float32)
        self.last_error_vec = np.zeros(obs_dim, dtype=np.float32)
        self.last_pred_mse = 0.0
        self.recent_error_window: List[float] = []
        self.last_structural_events: List[StructuralEvent] = []

        self.online_learning_enabled = True
        self.structural_plasticity_enabled = True
        self.current_step = 0

        self.reset()

    def _refresh_edge_cache(self) -> None:
        """Cache edge indices for fast vectorized indexing and queue dispatch."""
        srcs, dsts = np.where(self.adjacency_mask)
        self.edge_srcs = srcs.astype(np.int32)
        self.edge_dsts = dsts.astype(np.int32)
        self.edge_distances = self.distance_matrix[srcs, dsts].astype(np.float32)
        self.edge_delays = self.delay_matrix[srcs, dsts].astype(np.int32)
        self.num_edges = len(self.edge_srcs)

    @property
    def alpha(self) -> np.ndarray:
        return sigmoid(self.rho)

    @property
    def W_rec(self) -> np.ndarray:
        return self.W_hh

    @W_rec.setter
    def W_rec(self, val: np.ndarray) -> None:
        self.W_hh = val.astype(np.float32)

    def reset(self, batch_size: int = 1) -> None:
        self.h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.a = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.prev_h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.cand_h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.arrived_recurrent = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.queue.reset()
        self.predictive_rule.reset()
        if self.structural_plasticity:
            self.structural_plasticity.reset(self.hidden_dim)
        self.recent_error_window.clear()
        self.current_step = 0

    def select_action(self, epsilon: float = 0.15) -> int:
        """Select action using internal neural activations with epsilon-greedy exploration."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.action_dim))
        logits = np.dot(self.a[0], self.W_act) + self.b_act
        return int(np.argmax(logits))

    def predict_next_state(self, action: int) -> np.ndarray:
        r"""Generate forward sensory prediction \hat{s}_{t+1} from internal state and action."""
        action_onehot = np.zeros(self.action_dim, dtype=np.float32)
        action_onehot[action % self.action_dim] = 1.0

        phi = np.concatenate([self.a[0], action_onehot], axis=0)
        raw_pred = np.dot(phi, self.W_pred) + self.b_pred
        # Sensory observations are bounded in [0, 1]
        pred_s = np.clip(raw_pred, 0.0, 1.0).astype(np.float32)
        self.last_pred_s = pred_s
        return pred_s

    def step(
        self,
        obs_t: np.ndarray,
        action_t: int,
        obs_next: np.ndarray,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        r"""Advance one full embodied step in the closed sensorimotor loop.

        Args:
            obs_t: Sensory observation at time t [obs_dim]
            action_t: Action taken at time t
            obs_next: Actual sensory observation returned by environment at time t+1 [obs_dim]

        Returns:
            pred_next: Predicted sensory observation \hat{s}_{t+1}
            error_mse: Prediction error MSE
            step_metrics: Detailed computational metrics dictionary
        """
        # 1. Fetch arrived recurrent signals from propagation queue
        arrived = self.queue.advance_step()
        self.arrived_recurrent[0] = arrived

        self.prev_h = self.h.copy()
        pre_act = self.a[0].copy()

        # 2. Recurrent forward update:
        # raw_cand = obs · W_xh + I_arrived + b_h
        input_drive = np.dot(obs_t.reshape(1, -1), self.W_xh)[0]
        raw_cand = input_drive + arrived + self.b_h
        self.cand_h[0] = np.tanh(raw_cand)
        alpha = self.alpha
        self.h[0] = (1.0 - alpha) * self.prev_h[0] + alpha * self.cand_h[0]
        self.a[0] = np.tanh(self.h[0])

        # Record activity for structural correlation tracking
        if self.structural_plasticity:
            self.structural_plasticity.record_activity(self.a[0])

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

        # 4. Predict next sensory state
        pred_next = self.predict_next_state(action_t)

        # 5. Compute real computational prediction error
        error_vec = obs_next - pred_next
        error_mse = float(np.mean(error_vec ** 2))
        self.last_error_vec = error_vec
        self.last_pred_mse = error_mse
        self.recent_error_window.append(error_mse)
        if len(self.recent_error_window) > 30:
            self.recent_error_window.pop(0)

        # 6. Apply predictive learning rule
        learning_metrics = {"dw_norm": 0.0, "modulation": 0.0}
        if self.online_learning_enabled:
            action_onehot = np.zeros(self.action_dim, dtype=np.float32)
            action_onehot[action_t % self.action_dim] = 1.0

            dW_hh, dW_pred, db_pred, l_metrics = self.predictive_rule.update(
                pre_act=pre_act,
                post_act=self.a[0],
                post_cand=self.cand_h[0],
                W_hh=self.W_hh,
                W_pred=self.W_pred,
                b_pred=self.b_pred,
                action_onehot=action_onehot,
                target_s_next=obs_next,
                pred_s_next=pred_next,
                error_vec=error_vec,
                adjacency_mask=self.adjacency_mask,
            )
            learning_metrics.update(l_metrics)

            self.W_hh += dW_hh
            self.W_hh = np.clip(self.W_hh, -5.0, 5.0)
            self.W_pred += dW_pred
            self.W_pred = np.clip(self.W_pred, -5.0, 5.0)
            self.b_pred += db_pred
            self.b_pred = np.clip(self.b_pred, -5.0, 5.0)

        # 7. Apply structural plasticity (pruning, formation, dynamic neuron growth)
        self.last_structural_events = []
        if self.structural_plasticity_enabled and self.structural_plasticity is not None:
            mean_recent_err = float(np.mean(self.recent_error_window)) if self.recent_error_window else error_mse
            old_H = self.hidden_dim

            W_hh_new, adj_new, pos_new, dist_new, delay_new, struct_events = self.structural_plasticity.step(
                step=self.current_step,
                W_hh=self.W_hh,
                adjacency_mask=self.adjacency_mask,
                positions=self.positions,
                dist_matrix=self.distance_matrix,
                delay_matrix=self.delay_matrix,
                mean_recent_error=mean_recent_err,
            )
            self.last_structural_events = struct_events

            # If dynamic neuron growth occurred, adapt model parameters
            new_H = W_hh_new.shape[0]
            if new_H > old_H:
                # Expand hidden dimension
                added = new_H - old_H
                self.hidden_dim = new_H
                self.W_hh = W_hh_new
                self.adjacency_mask = adj_new
                self.positions = pos_new
                self.distance_matrix = dist_new
                self.delay_matrix = delay_new

                # Expand W_xh, b_h, rho, W_act, W_pred
                scale_xh = np.sqrt(2.0 / (self.obs_dim + new_H))
                new_W_xh = np.zeros((self.obs_dim, new_H), dtype=np.float32)
                new_W_xh[:, :old_H] = self.W_xh
                new_W_xh[:, old_H:] = self.rng.normal(0.0, scale_xh, (self.obs_dim, added))
                self.W_xh = new_W_xh

                self.b_h = np.pad(self.b_h, (0, added), "constant")
                new_rho = self.rng.normal(-1.0, 0.2, added).astype(np.float32)
                self.rho = np.concatenate([self.rho, new_rho])

                new_W_act = np.zeros((new_H, self.action_dim), dtype=np.float32)
                new_W_act[:old_H, :] = self.W_act
                new_W_act[old_H:, :] = self.rng.normal(0.0, 0.1, (added, self.action_dim))
                self.W_act = new_W_act

                # Expand W_pred: shape is [(H + A), S]
                old_pred_h = self.W_pred[:old_H, :]
                old_pred_act = self.W_pred[old_H:, :]
                new_pred_added = self.rng.normal(0.0, 0.1, (added, self.obs_dim)).astype(np.float32)
                self.W_pred = np.vstack([old_pred_h, new_pred_added, old_pred_act])

                # Expand state vectors
                self.h = np.pad(self.h, ((0, 0), (0, added)), "constant")
                self.a = np.pad(self.a, ((0, 0), (0, added)), "constant")
                self.prev_h = np.pad(self.prev_h, ((0, 0), (0, added)), "constant")
                self.cand_h = np.pad(self.cand_h, ((0, 0), (0, added)), "constant")
                self.arrived_recurrent = np.pad(self.arrived_recurrent, ((0, 0), (0, added)), "constant")

                # Expand queue
                new_max_delay = int(np.max(self.delay_matrix))
                self.queue = SignalPropagationQueue(new_H, max_delay=max(5, new_max_delay), track_in_transit=True)

                # Adapt predictive learning rule buffers
                self.predictive_rule.adapt_dimensions(new_H)
            else:
                self.W_hh = W_hh_new
                self.adjacency_mask = adj_new

            self._refresh_edge_cache()

        self.current_step += 1

        step_metrics = {
            "step": self.current_step,
            "pred_mse": error_mse,
            "error_norm": float(np.linalg.norm(error_vec)),
            "num_edges": self.num_edges,
            "num_neurons": self.hidden_dim,
            "dw_norm": learning_metrics.get("dw_norm", 0.0),
            "modulation": learning_metrics.get("modulation", 0.0),
            "structural_events_count": len(self.last_structural_events),
        }
        return pred_next, error_mse, step_metrics

    def forward_sequence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process sequence X of shape [batch_size, seq_len, obs_dim]."""
        batch_size, seq_len, _ = X.shape
        Y = np.zeros((batch_size, seq_len, self.output_dim), dtype=np.float32)
        H = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)
        A = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)

        for t in range(seq_len):
            x_t = X[0, t, :]
            act = self.select_action()
            pred, _, _ = self.step(x_t, act, x_t)
            Y[0, t, :] = pred
            H[0, t, :] = self.h[0]
            A[0, t, :] = self.a[0]

        return Y, H, A

    def backward_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        cache: dict,
    ) -> Dict[str, np.ndarray]:
        """Offline gradient computation (control baseline)."""
        return {"W_hh": np.zeros_like(self.W_hh), "W_xh": np.zeros_like(self.W_xh)}

    def parameter_count(self) -> int:
        """Total parameters including sparse recurrent edges, predictive head, and policy weights."""
        recurrent_params = self.num_edges
        io_params = self.obs_dim * self.hidden_dim + self.hidden_dim + self.hidden_dim  # W_xh, b_h, rho
        pred_params = (self.hidden_dim + self.action_dim) * self.obs_dim + self.obs_dim
        act_params = self.hidden_dim * self.action_dim + self.action_dim
        return recurrent_params + io_params + pred_params + act_params

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            "W_xh": self.W_xh.copy(),
            "W_hh": self.W_hh.copy(),
            "b_h": self.b_h.copy(),
            "rho": self.rho.copy(),
            "W_act": self.W_act.copy(),
            "b_act": self.b_act.copy(),
            "W_pred": self.W_pred.copy(),
            "b_pred": self.b_pred.copy(),
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.W_xh = weights["W_xh"].copy()
        self.W_hh = (weights["W_hh"] * self.adjacency_mask).copy()
        self.b_h = weights["b_h"].copy()
        self.rho = weights["rho"].copy()
        if "W_act" in weights:
            self.W_act = weights["W_act"].copy()
        if "b_act" in weights:
            self.b_act = weights["b_act"].copy()
        if "W_pred" in weights:
            self.W_pred = weights["W_pred"].copy()
        if "b_pred" in weights:
            self.b_pred = weights["b_pred"].copy()

    def observable_state(self) -> Dict[str, Any]:
        """Return non-invasive state snapshot for observability engine."""
        return {
            "state": self.h,
            "activation": self.a,
            "previous_state": self.prev_h,
            "cand_h": self.cand_h,
            "arrived_recurrent": self.arrived_recurrent,
            "alpha": self.alpha,
            "pred_s": self.last_pred_s,
            "error_vec": self.last_error_vec,
            "pred_mse": self.last_pred_mse,
            "num_edges": self.num_edges,
            "num_neurons": self.hidden_dim,
        }

