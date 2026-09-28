r"""Predictive Learning Rules and Self-Supervised Objectives for 3D-Brain Phase 5.

Implements the formal predictive learning abstractions:
- Sensory prediction readout: (h_t, action_t) -> \hat{s}_{t+1}
- Real computational prediction error: e_t = s_{t+1} - \hat{s}_{t+1}
- Four learning regimes:
  1. FrozenPredictive (Rule A: Frozen baseline control)
  2. LocalPredictivePlasticity (Rule B: Local predictive coding with feedback alignment)
  3. NeuromodulatedPredictivePlasticity (Rule C: Global surprise / error modulation of local eligibility)
  4. GradientPredictiveBaseline (Rule D: Exact gradient descent control condition)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class PredictiveLearningRule(ABC):
    """Abstract base class for predictive self-supervised learning rules."""

    def __init__(self, name: str, lr: float = 1e-3, lr_pred: float = 5e-3):
        self.name = name
        self.lr = float(lr)
        self.lr_pred = float(lr_pred)

    @abstractmethod
    def reset(self) -> None:
        """Reset internal rule state."""
        pass

    @abstractmethod
    def update(
        self,
        pre_act: np.ndarray,
        post_act: np.ndarray,
        post_cand: np.ndarray,
        W_hh: np.ndarray,
        W_pred: np.ndarray,
        b_pred: np.ndarray,
        action_onehot: np.ndarray,
        target_s_next: np.ndarray,
        pred_s_next: np.ndarray,
        error_vec: np.ndarray,
        adjacency_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        r"""Compute updates for recurrent weights W_hh and prediction head W_pred, b_pred.

        Args:
            pre_act: Pre-synaptic activations a(t-1) of shape [H]
            post_act: Post-synaptic activations a(t) of shape [H]
            post_cand: Post-synaptic candidate states cand_h(t) of shape [H]
            W_hh: Recurrent synaptic weights [H, H]
            W_pred: Predictive projection weights [H + A, S]
            b_pred: Predictive bias [S]
            action_onehot: One-hot encoded action vector [A]
            target_s_next: Ground-truth actual future observation s(t+1) of shape [S]
            pred_s_next: Predicted future observation \hat{s}(t+1) of shape [S]
            error_vec: Residual prediction error s(t+1) - \hat{s}(t+1) of shape [S]
            adjacency_mask: Structural adjacency mask [H, H] (1 for existent edge, 0 for none)

        Returns:
            delta_W_hh: Weight update matrix [H, H] (masked by adjacency_mask)
            delta_W_pred: Prediction head weight update [(H + A), S]
            delta_b_pred: Prediction head bias update [S]
            metrics: Diagnostic metrics dictionary
        """
        pass

    def adapt_dimensions(self, new_H: int) -> None:
        """Adapt internal parameters/buffers if dynamic neuron growth occurs."""
        pass


class FrozenPredictive(PredictiveLearningRule):
    """Rule A: Frozen baseline control (no weight adjustments)."""

    def __init__(self):
        super().__init__(name="FrozenPredictive", lr=0.0, lr_pred=0.0)

    def reset(self) -> None:
        pass

    def update(
        self,
        pre_act: np.ndarray,
        post_act: np.ndarray,
        post_cand: np.ndarray,
        W_hh: np.ndarray,
        W_pred: np.ndarray,
        b_pred: np.ndarray,
        action_onehot: np.ndarray,
        target_s_next: np.ndarray,
        pred_s_next: np.ndarray,
        error_vec: np.ndarray,
        adjacency_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        mse = float(np.mean(error_vec ** 2))
        return (
            np.zeros_like(W_hh),
            np.zeros_like(W_pred),
            np.zeros_like(b_pred),
            {"pred_mse": mse, "dw_norm": 0.0, "modulation": 0.0},
        )


class LocalPredictivePlasticity(PredictiveLearningRule):
    r"""Rule B: Local predictive plasticity using local predictive feedback.

    The predictive readout is updated via Widrow-Hoff delta rule.
    Recurrent weights are updated using local feedback alignment:
    unit j receives local error signal e_j = \sum_k e_{pred, k} B_{jk},
    and updates connections locally:
        \Delta w_{ij} = \eta \cdot a_i(t-1) \cdot e_j(t) \cdot (1 - a_j(t)^2) - \lambda w_{ij}
    """

    def __init__(
        self,
        hidden_dim: int,
        obs_dim: int,
        lr: float = 2e-3,
        lr_pred: float = 8e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ):
        super().__init__(name="LocalPredictive", lr=lr, lr_pred=lr_pred)
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.weight_decay = float(weight_decay)
        # Random fixed feedback alignment matrix B [hidden_dim, obs_dim]
        rng = np.random.default_rng(seed)
        self.B_feedback = rng.normal(0.0, 1.0 / np.sqrt(obs_dim), (hidden_dim, obs_dim)).astype(np.float32)

    def reset(self) -> None:
        pass

    def update(
        self,
        pre_act: np.ndarray,
        post_act: np.ndarray,
        post_cand: np.ndarray,
        W_hh: np.ndarray,
        W_pred: np.ndarray,
        b_pred: np.ndarray,
        action_onehot: np.ndarray,
        target_s_next: np.ndarray,
        pred_s_next: np.ndarray,
        error_vec: np.ndarray,
        adjacency_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        mse = float(np.mean(error_vec ** 2))

        # 1. Update predictive readout head:
        # feature vector phi = [post_act, action_onehot] of shape [H + A]
        phi = np.concatenate([post_act, action_onehot], axis=0)  # [H + A]
        # delta rule: \Delta W_pred = lr_pred * phi^T * error_vec
        delta_W_pred = self.lr_pred * np.outer(phi, error_vec)
        delta_b_pred = self.lr_pred * error_vec

        # 2. Local feedback alignment error to hidden units:
        # local_error_h = B_feedback · error_vec of shape [H]
        local_error_h = np.dot(self.B_feedback, error_vec)  # [H]

        # Post-synaptic derivative of tanh: (1 - post_act^2)
        post_deriv = 1.0 - (post_act ** 2)
        effective_post_err = local_error_h * post_deriv  # [H]

        # 3. Local recurrent weight update:
        # pre_act: [H] (row), effective_post_err: [H] (col)
        # \Delta w_{ij} = lr * pre_i * post_err_j - weight_decay * w_{ij}
        delta_W_hh = self.lr * np.outer(pre_act, effective_post_err) - self.weight_decay * W_hh
        delta_W_hh = delta_W_hh * adjacency_mask

        dw_norm = float(np.linalg.norm(delta_W_hh))
        return (
            delta_W_hh,
            delta_W_pred,
            delta_b_pred,
            {"pred_mse": mse, "dw_norm": dw_norm, "modulation": 0.0},
        )

    def adapt_dimensions(self, new_H: int) -> None:
        if new_H > self.hidden_dim:
            added = new_H - self.hidden_dim
            rng = np.random.default_rng()
            new_B = rng.normal(0.0, 1.0 / np.sqrt(self.obs_dim), (added, self.obs_dim)).astype(np.float32)
            self.B_feedback = np.vstack([self.B_feedback, new_B])
            self.hidden_dim = new_H


class NeuromodulatedPredictivePlasticity(PredictiveLearningRule):
    """Rule C: Neuromodulated predictive learning.

    Local synaptic eligibility traces are gated by a global neuromodulatory
    scalar M(t) = tanh(beta * prediction_error_norm) representing surprise.
    When prediction error is high, the neuromodulator allows plastic reorganization;
    when error is low (familiar environment), stability is preserved.
    """

    def __init__(
        self,
        hidden_dim: int,
        lr: float = 3e-3,
        lr_pred: float = 8e-3,
        beta: float = 2.5,
        trace_decay: float = 0.85,
        weight_decay: float = 1e-4,
    ):
        super().__init__(name="NeuromodulatedPredictive", lr=lr, lr_pred=lr_pred)
        self.hidden_dim = hidden_dim
        self.beta = float(beta)
        self.trace_decay = float(trace_decay)
        self.weight_decay = float(weight_decay)
        self.eligibility = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)

    def reset(self) -> None:
        self.eligibility.fill(0.0)

    def update(
        self,
        pre_act: np.ndarray,
        post_act: np.ndarray,
        post_cand: np.ndarray,
        W_hh: np.ndarray,
        W_pred: np.ndarray,
        b_pred: np.ndarray,
        action_onehot: np.ndarray,
        target_s_next: np.ndarray,
        pred_s_next: np.ndarray,
        error_vec: np.ndarray,
        adjacency_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        mse = float(np.mean(error_vec ** 2))
        err_norm = float(np.linalg.norm(error_vec))

        # Global surprise / neuromodulatory burst:
        modulation = float(np.tanh(self.beta * err_norm))

        # 1. Update predictive readout head:
        phi = np.concatenate([post_act, action_onehot], axis=0)
        delta_W_pred = self.lr_pred * np.outer(phi, error_vec)
        delta_b_pred = self.lr_pred * error_vec

        # 2. Update local eligibility trace:
        # Hebbian co-activation: pre_act outer post_act
        instantaneous_coact = np.outer(pre_act, post_act)
        self.eligibility = self.trace_decay * self.eligibility + (1.0 - self.trace_decay) * instantaneous_coact

        # 3. Neuromodulated weight update:
        delta_W_hh = (self.lr * modulation) * self.eligibility - self.weight_decay * W_hh
        delta_W_hh = delta_W_hh * adjacency_mask

        dw_norm = float(np.linalg.norm(delta_W_hh))
        return (
            delta_W_hh,
            delta_W_pred,
            delta_b_pred,
            {"pred_mse": mse, "dw_norm": dw_norm, "modulation": modulation},
        )

    def adapt_dimensions(self, new_H: int) -> None:
        if new_H > self.hidden_dim:
            new_elig = np.zeros((new_H, new_H), dtype=np.float32)
            new_elig[:self.hidden_dim, :self.hidden_dim] = self.eligibility
            self.eligibility = new_elig
            self.hidden_dim = new_H


class GradientPredictiveBaseline(PredictiveLearningRule):
    """Rule D: Online gradient-based baseline (BPTT / SGD control)."""

    def __init__(
        self,
        lr: float = 3e-3,
        lr_pred: float = 8e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__(name="GradientPredictive", lr=lr, lr_pred=lr_pred)
        self.weight_decay = float(weight_decay)

    def reset(self) -> None:
        pass

    def update(
        self,
        pre_act: np.ndarray,
        post_act: np.ndarray,
        post_cand: np.ndarray,
        W_hh: np.ndarray,
        W_pred: np.ndarray,
        b_pred: np.ndarray,
        action_onehot: np.ndarray,
        target_s_next: np.ndarray,
        pred_s_next: np.ndarray,
        error_vec: np.ndarray,
        adjacency_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        mse = float(np.mean(error_vec ** 2))
        H = len(post_act)

        # Gradient on prediction head: loss = 0.5 * ||pred - target||^2
        # dL / d(pred) = -error_vec
        phi = np.concatenate([post_act, action_onehot], axis=0)
        delta_W_pred = self.lr_pred * np.outer(phi, error_vec)
        delta_b_pred = self.lr_pred * error_vec

        # Backpropagation into post_act:
        # dL / d(post_act) = - W_pred[:H] · error_vec
        W_pred_h = W_pred[:H, :]  # [H, S]
        grad_post_act = np.dot(W_pred_h, error_vec)  # [H]

        # Derivative through post_act = tanh(h):
        post_deriv = 1.0 - (post_act ** 2)
        grad_post = grad_post_act * post_deriv  # [H]

        # dL / dW_hh = pre_act^T · grad_post
        delta_W_hh = self.lr * np.outer(pre_act, grad_post) - self.weight_decay * W_hh
        delta_W_hh = delta_W_hh * adjacency_mask

        dw_norm = float(np.linalg.norm(delta_W_hh))
        return (
            delta_W_hh,
            delta_W_pred,
            delta_b_pred,
            {"pred_mse": mse, "dw_norm": dw_norm, "modulation": 0.0},
        )
