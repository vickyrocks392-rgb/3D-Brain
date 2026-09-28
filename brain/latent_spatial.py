r"""Spatially Embedded Predictive Recurrent Neural Substrate with Persistent Latent Memory.

Extends PredictiveSpatialRNN (Phase 5) to integrate:
- Multi-timescale persistent latent state z_t
- Top-down feedback I_latent = z_{t-1} W_{zh} into spatial recurrent units
- Multi-step predictive heads for horizons k in {1, 2, 5, 10, 25}
- Causal ablations for persistence, step resets, frozen weights, and gradient controls
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.predictive_spatial import PredictiveSpatialRNN
from brain.spatial import TopologyFamily
from brain.predictive_plasticity import PredictiveLearningRule
from brain.structural_plasticity import StructuralPlasticity
from brain.latent_memory import PersistentLatentModule


class PersistentLatentSpatialRNN(PredictiveSpatialRNN):
    """Spatially embedded recurrent brain model with persistent latent memory and multi-step predictive heads."""

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 4,
        hidden_dim: int = 40,
        latent_dim: int = 24,
        topology_type: TopologyFamily = TopologyFamily.LOCAL,
        target_edges: Optional[int] = None,
        propagation_speed: float = 5.0,
        positions: Optional[np.ndarray] = None,
        module_ids: Optional[np.ndarray] = None,
        predictive_rule: Optional[PredictiveLearningRule] = None,
        structural_plasticity: Optional[StructuralPlasticity] = None,
        prediction_horizons: Optional[List[int]] = None,
        persistence_min: float = 0.85,
        persistence_max: float = 0.98,
        feedback_strength: float = 0.5,
        latent_lr: float = 0.01,
        seed: int = 42,
    ):
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            topology_type=topology_type,
            target_edges=target_edges,
            propagation_speed=propagation_speed,
            positions=positions,
            module_ids=module_ids,
            predictive_rule=predictive_rule,
            structural_plasticity=structural_plasticity,
            seed=seed,
        )
        self.latent_dim = latent_dim
        self.prediction_horizons = prediction_horizons or [1, 2, 5, 10, 25]

        # Explicit Persistent Latent Memory Module
        self.latent = PersistentLatentModule(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            prediction_horizons=self.prediction_horizons,
            persistence_min=persistence_min,
            persistence_max=persistence_max,
            feedback_strength=feedback_strength,
            learning_rate=latent_lr,
            seed=seed,
        )

        # Buffer for multi-step history and future observation lookup
        self.observation_buffer: List[np.ndarray] = []
        self.action_buffer: List[int] = []
        self.latent_feedback = np.zeros(hidden_dim, dtype=np.float32)

        # Latent enabled flag (for Baseline 1 / Ablation G)
        self.latent_enabled = True

    def reset(self, batch_size: int = 1) -> None:
        """Reset spatial recurrent state and persistent latent memory."""
        super().reset(batch_size=batch_size)
        if hasattr(self, "latent"):
            self.latent.reset()
        if hasattr(self, "observation_buffer"):
            self.observation_buffer.clear()
        if hasattr(self, "action_buffer"):
            self.action_buffer.clear()
        if hasattr(self, "latent_feedback"):
            self.latent_feedback = np.zeros(self.hidden_dim, dtype=np.float32)

    def select_action(self, epsilon: float = 0.15) -> int:
        """Select action combining recurrent activation and persistent latent state."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.action_dim))
        
        # Policy reads both recurrent activation and persistent latent memory
        if self.latent_enabled:
            policy_feature = np.concatenate([self.a[0], self.latent.z])
            if not hasattr(self, "W_act_latent"):
                scale = np.sqrt(2.0 / (self.hidden_dim + self.latent_dim + self.action_dim))
                self.W_act_latent = self.rng.normal(0.0, scale, (self.hidden_dim + self.latent_dim, self.action_dim)).astype(np.float32)
            logits = np.dot(policy_feature, self.W_act_latent) + self.b_act
        else:
            logits = np.dot(self.a[0], self.W_act) + self.b_act

        return int(np.argmax(logits))

    def step(
        self,
        obs_t: np.ndarray,
        action_t: int,
        obs_next: np.ndarray,
        future_observations: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        r"""Advance one full embodied step in the closed sensorimotor loop with persistent latent dynamics.

        Args:
            obs_t: Sensory observation at time t [obs_dim]
            action_t: Action taken at time t
            obs_next: Actual sensory observation returned by environment at time t+1 [obs_dim]
            future_observations: Optional dict mapping horizon k -> s_{t+k}

        Returns:
            pred_next: Predicted sensory observation \hat{s}_{t+1}
            error_mse: One-step prediction error MSE
            step_metrics: Detailed computational metrics including latent state and multi-step predictions
        """
        # Store in circular observation buffer for rolling evaluation
        self.observation_buffer.append(obs_t.copy())
        self.action_buffer.append(action_t)
        if len(self.observation_buffer) > 100:
            self.observation_buffer.pop(0)
            self.action_buffer.pop(0)

        # 1. Fetch arrived recurrent signals from propagation queue
        arrived = self.queue.advance_step()
        self.arrived_recurrent[0] = arrived

        self.prev_h = self.h.copy()
        pre_act = self.a[0].copy()

        # 2. Recurrent forward update:
        # drive = W_xh · obs + arrived + I_latent + b_h
        input_drive = np.dot(obs_t.reshape(1, -1), self.W_xh)[0]
        I_latent = self.latent_feedback if self.latent_enabled else np.zeros(self.hidden_dim, dtype=np.float32)
        raw_cand = input_drive + arrived + I_latent + self.b_h
        self.cand_h[0] = np.tanh(raw_cand)
        alpha = self.alpha
        self.h[0] = (1.0 - alpha) * self.prev_h[0] + alpha * self.cand_h[0]
        self.a[0] = np.tanh(self.h[0])

        # Record activity for structural correlation tracking
        if self.structural_plasticity:
            self.structural_plasticity.record_activity(self.a[0])

        # 3. Update persistent latent state z_t and compute next feedback
        if self.latent_enabled:
            z_t, next_feedback = self.latent.update_latent_state(self.a[0], action_t)
            self.latent_feedback = next_feedback
            multistep_preds = self.latent.predict_multi_step(action_t)
        else:
            z_t = np.zeros(self.latent_dim, dtype=np.float32)
            self.latent_feedback.fill(0.0)
            multistep_preds = {}

        # 4. Enqueue outgoing signals across all active synaptic connections
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

        # 5. Predict next sensory state
        pred_next = self.predict_next_state(action_t)

        # 6. Compute real computational prediction error (one-step)
        error_vec = obs_next - pred_next
        error_mse = float(np.mean(error_vec ** 2))
        self.last_error_vec = error_vec
        self.last_pred_mse = error_mse
        self.recent_error_window.append(error_mse)
        if len(self.recent_error_window) > 30:
            self.recent_error_window.pop(0)

        # 7. Apply recurrent predictive learning rule
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

        # 8. Apply latent memory learning across multi-step horizons
        latent_metrics = {"mean_multistep_mse": 0.0, "dw_norm": 0.0}
        if self.latent_enabled and self.online_learning_enabled:
            targets = future_observations if future_observations is not None else {1: obs_next}
            latent_metrics = self.latent.learn_step(self.a[0], action_t, targets)

        # 9. Apply structural plasticity (synaptogenesis, pruning, dynamic growth)
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

            new_H = W_hh_new.shape[0]
            if new_H > old_H:
                added = new_H - old_H
                self.hidden_dim = new_H
                self.W_hh = W_hh_new
                self.adjacency_mask = adj_new
                self.positions = pos_new
                self.distance_matrix = dist_new
                self.delay_matrix = delay_new

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

                old_pred_h = self.W_pred[:old_H, :]
                old_pred_act = self.W_pred[old_H:, :]
                new_pred_added = self.rng.normal(0.0, 0.1, (added, self.obs_dim)).astype(np.float32)
                self.W_pred = np.vstack([old_pred_h, new_pred_added, old_pred_act])

                self.h = np.pad(self.h, ((0, 0), (0, added)), "constant")
                self.a = np.pad(self.a, ((0, 0), (0, added)), "constant")
                self.prev_h = np.pad(self.prev_h, ((0, 0), (0, added)), "constant")
                self.cand_h = np.pad(self.cand_h, ((0, 0), (0, added)), "constant")
                self.arrived_recurrent = np.pad(self.arrived_recurrent, ((0, 0), (0, added)), "constant")
                self.latent_feedback = np.pad(self.latent_feedback, (0, added), "constant")

                new_max_delay = int(np.max(self.delay_matrix))
                self.queue = SignalPropagationQueue(new_H, max_delay=max(5, new_max_delay), track_in_transit=True)

                self.predictive_rule.adapt_dimensions(new_H)
                if self.latent_enabled:
                    self.latent.adapt_dimensions(new_H)
                    if hasattr(self, "W_act_latent"):
                        scale_act_lat = np.sqrt(2.0 / (new_H + self.latent_dim + self.action_dim))
                        self.W_act_latent = self.rng.normal(0.0, scale_act_lat, (new_H + self.latent_dim, self.action_dim)).astype(np.float32)
            else:
                self.W_hh = W_hh_new
                self.adjacency_mask = adj_new

            self._refresh_edge_cache()

        self.current_step += 1

        multistep_mse_dict = {}
        if future_observations:
            for k, target in future_observations.items():
                if k in multistep_preds:
                    multistep_mse_dict[k] = float(np.mean((target - multistep_preds[k]) ** 2))

        step_metrics = {
            "step": self.current_step,
            "pred_mse": error_mse,
            "error_norm": float(np.linalg.norm(error_vec)),
            "num_edges": self.num_edges,
            "num_neurons": self.hidden_dim,
            "dw_norm": learning_metrics.get("dw_norm", 0.0),
            "latent_dw_norm": latent_metrics.get("dw_norm", 0.0),
            "latent_norm": float(np.linalg.norm(z_t)),
            "latent_z": z_t.copy(),
            "multistep_preds": {k: v.copy() for k, v in multistep_preds.items()},
            "multistep_mse": multistep_mse_dict,
            "structural_events_count": len(self.last_structural_events),
        }
        return pred_next, error_mse, step_metrics
