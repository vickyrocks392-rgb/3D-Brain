"""Online synaptic plasticity rules and plastic recurrent models for 3D-Brain.

Implements the formal plasticity abstractions specified in:
- research/ONLINE_PLASTICITY_REVIEW.md
- experiments/EXPERIMENT_003_SPECIFICATION.md
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import numpy as np

from brain.base import BrainModel


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


class PlasticityRule(ABC):
    """Abstract base class for strictly local synaptic plasticity rules."""

    def __init__(self, name: str, lr: float = 1e-3):
        self.name = name
        self.lr = float(lr)

    @abstractmethod
    def reset(self) -> None:
        """Reset internal rule state."""
        pass

    @abstractmethod
    def compute_delta(
        self,
        pre_act: np.ndarray,
        post_act: np.ndarray,
        weights: np.ndarray,
        modulation: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute weight update delta_W.
        
        Args:
            pre_act: Pre-synaptic activations of shape [hidden_dim]
            post_act: Post-synaptic activations of shape [hidden_dim]
            weights: Current weight matrix W of shape [hidden_dim, hidden_dim]
                     where W[i, j] is connection from pre-synaptic i to post-synaptic j.
            modulation: Optional scalar neuromodulatory/reward/error signal M(t).

        Returns:
            delta_W: Weight update matrix of shape [hidden_dim, hidden_dim].
            metrics: Local diagnostic dictionary.
        """
        pass


class FrozenPlasticity(PlasticityRule):
    """Rule A: Frozen baseline control (zero online modification)."""

    def __init__(self):
        super().__init__(name="Frozen", lr=0.0)

    def reset(self) -> None:
        pass

    def compute_delta(
        self,
        pre_act: np.ndarray,
        post_act: np.ndarray,
        weights: np.ndarray,
        modulation: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return np.zeros_like(weights), {"modulation": 0.0, "dw_norm": 0.0}


class HebbianPlasticity(PlasticityRule):
    """Rule B: Plain correlation-based Hebbian plasticity with bounding."""

    def __init__(self, lr: float = 1e-3, w_min: float = -3.0, w_max: float = 3.0, weight_decay: float = 1e-4):
        super().__init__(name="Hebbian", lr=lr)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.weight_decay = float(weight_decay)

    def reset(self) -> None:
        pass

    def compute_delta(
        self,
        pre_act: np.ndarray,
        post_act: np.ndarray,
        weights: np.ndarray,
        modulation: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # dW_ij = lr * a_i * a_j - decay * W_ij
        outer = np.outer(pre_act, post_act).astype(np.float32)
        dw = (self.lr * outer - self.weight_decay * weights).astype(np.float32)
        return dw, {"dw_norm": float(np.linalg.norm(dw))}


class OjaPlasticity(PlasticityRule):
    """Rule C: Oja's self-stabilizing normalized Hebbian plasticity."""

    def __init__(self, lr: float = 1e-3):
        super().__init__(name="Oja", lr=lr)

    def reset(self) -> None:
        pass

    def compute_delta(
        self,
        pre_act: np.ndarray,
        post_act: np.ndarray,
        weights: np.ndarray,
        modulation: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # dW_ij = lr * a_j * (a_i - a_j * W_ij)
        # = lr * [ (a_pre (x) a_post) - W * diag(a_post^2) ]
        outer = np.outer(pre_act, post_act).astype(np.float32)
        post_sq = (post_act ** 2).reshape(1, -1).astype(np.float32)
        dw = (self.lr * (outer - weights * post_sq)).astype(np.float32)
        return dw, {"dw_norm": float(np.linalg.norm(dw))}


class ModulatedPlasticity(PlasticityRule):
    """Rule D: Three-factor neuromodulated / prediction-error gated plasticity."""

    def __init__(self, lr: float = 2e-3, baseline_leak: float = 0.0):
        super().__init__(name="Modulated", lr=lr)
        self.baseline_leak = float(baseline_leak)

    def reset(self) -> None:
        pass

    def compute_delta(
        self,
        pre_act: np.ndarray,
        post_act: np.ndarray,
        weights: np.ndarray,
        modulation: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        mod = 1.0 if modulation is None else float(modulation)
        outer = np.outer(pre_act, post_act).astype(np.float32)
        post_sq = (post_act ** 2).reshape(1, -1).astype(np.float32)
        base_dw = (outer - weights * post_sq).astype(np.float32)
        dw = (self.lr * mod * base_dw - self.baseline_leak * weights).astype(np.float32)
        return dw, {"modulation": mod, "dw_norm": float(np.linalg.norm(dw))}


class HomeostaticPlasticity(PlasticityRule):
    """Rule E: Bounded homeostatic plasticity with slow synaptic scaling."""

    def __init__(
        self,
        lr: float = 1e-3,
        target_activity: float = 0.2,
        gamma: float = 1e-4,
        beta: float = 0.05,
    ):
        super().__init__(name="Homeostatic", lr=lr)
        self.target_activity = float(target_activity)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.running_activity: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.running_activity = None

    def compute_delta(
        self,
        pre_act: np.ndarray,
        post_act: np.ndarray,
        weights: np.ndarray,
        modulation: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        dim = post_act.shape[0]
        if self.running_activity is None or self.running_activity.shape[0] != dim:
            self.running_activity = np.abs(post_act).copy()
        else:
            self.running_activity = (
                (1.0 - self.beta) * self.running_activity + self.beta * np.abs(post_act)
            )

        # Oja fast term
        outer = np.outer(pre_act, post_act).astype(np.float32)
        post_sq = (post_act ** 2).reshape(1, -1).astype(np.float32)
        oja_term = outer - weights * post_sq

        # Homeostatic slow scaling: gamma * (a^* - running_a_j) * W_ij
        act_diff = (self.target_activity - self.running_activity).reshape(1, -1).astype(np.float32)
        homeo_term = act_diff * weights

        dw = (self.lr * oja_term + self.gamma * homeo_term).astype(np.float32)
        return dw, {
            "dw_norm": float(np.linalg.norm(dw)),
            "mean_running_act": float(np.mean(self.running_activity)),
        }


class PlasticRNN(BrainModel):
    """Discrete Leaky RNN equipped with online local synaptic plasticity.

    Architecture matches DISC-LRNN (the best-performing substrate from Phase 2),
    with recurrent weight matrix W_hh subject to online local modification:
        W_hh(t + 1) = W_hh(t) + delta_W_hh(t)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        plasticity_rule: Optional[PlasticityRule] = None,
        seed: int = 42,
    ):
        super().__init__(input_dim, hidden_dim, output_dim, seed)
        self.plasticity_rule = plasticity_rule or FrozenPlasticity()

        scale_xh = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hh = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        scale_hy = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W_xh = self.rng.normal(0.0, scale_xh, (input_dim, hidden_dim)).astype(np.float32)
        self.W_hh = self.rng.normal(0.0, scale_hh, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)
        self.rho = self.rng.normal(-1.0, 0.2, hidden_dim).astype(np.float32)
        self.W_hy = self.rng.normal(0.0, scale_hy, (hidden_dim, output_dim)).astype(np.float32)
        self.b_y = np.zeros(output_dim, dtype=np.float32)

        # Baseline snapshot of initial recurrent weights for cumulative delta tracking
        self.initial_W_hh = self.W_hh.copy()

        # Telemetry & dynamic inspectable registers
        self.prev_h = np.zeros((1, hidden_dim), dtype=np.float32)
        self.cand_h = np.zeros((1, hidden_dim), dtype=np.float32)
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
        self.plasticity_rule.reset()

    def reset_weights_to_initial(self) -> None:
        """Reset recurrent weights back to baseline snapshot."""
        self.W_hh = self.initial_W_hh.copy()
        self.last_dw = np.zeros_like(self.W_hh)
        self.cumulative_abs_dw = np.zeros_like(self.W_hh)
        self.plastic_updates_count = 0
        self.plasticity_rule.reset()

    def step(self, x: np.ndarray, modulation: Optional[float] = None) -> np.ndarray:
        """Step one timestep with optional online plasticity update."""
        alpha = self.alpha
        self.prev_h = self.h.copy()

        # Recurrent forward update
        raw_cand = np.dot(x, self.W_xh) + np.dot(self.h, self.W_hh) + self.b_h
        self.cand_h = np.tanh(raw_cand)
        self.h = (1.0 - alpha) * self.prev_h + alpha * self.cand_h
        self.a = np.tanh(self.h)

        # Output
        y = np.dot(self.a, self.W_hy) + self.b_y

        # Online local plasticity update on W_hh (using pre=a_(t-1), post=a_t)
        if self.online_plasticity_enabled:
            pre_vec = self.prev_h[0]  # [hidden_dim]
            post_vec = self.a[0]      # [hidden_dim]
            dw, _ = self.plasticity_rule.compute_delta(pre_vec, post_vec, self.W_hh, modulation)
            
            # Apply bounds to prevent explosive divergence
            self.W_hh += dw
            self.W_hh = np.clip(self.W_hh, -5.0, 5.0)
            
            self.last_dw = dw
            self.cumulative_abs_dw += np.abs(dw)
            self.plastic_updates_count += 1

        return y

    def forward_sequence(
        self, X: np.ndarray, modulations: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process sequence X [batch_size, seq_len, input_dim]."""
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
        """Offline BPTT backward pass for initial pretraining (with plasticity disabled)."""
        # Matches DiscreteLeakyRNN exact analytical gradients
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

            dW_xh += np.dot(X[:, t, :].T, d_raw)
            dW_hh += np.dot(h_prev.T, d_raw)
            db_h += np.sum(d_raw, axis=0)

            dh_next = dh_t * (1.0 - alpha) + np.dot(d_raw, self.W_hh.T)

        return {
            "W_xh": dW_xh,
            "W_hh": dW_hh,
            "b_h": db_h,
            "rho": drho,
            "W_hy": dW_hy,
            "b_y": db_y,
        }

    def parameter_count(self) -> int:
        return (
            self.W_xh.size
            + self.W_hh.size
            + self.b_h.size
            + self.rho.size
            + self.W_hy.size
            + self.b_y.size
        )

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
        self.W_hh = weights["W_hh"].copy()
        self.b_h = weights["b_h"].copy()
        self.rho = weights["rho"].copy()
        self.W_hy = weights["W_hy"].copy()
        self.b_y = weights["b_y"].copy()
        self.initial_W_hh = self.W_hh.copy()

    def observable_state(self) -> Dict[str, np.ndarray]:
        d = super().observable_state()
        d["alpha"] = self.alpha
        d["tau"] = 1.0 / np.maximum(self.alpha, 1e-4)
        d["cand_h"] = self.cand_h.copy()
        d["prev_h"] = self.prev_h.copy()
        d["mean_abs_dw"] = np.array([float(np.mean(np.abs(self.last_dw)))], dtype=np.float32)
        d["max_dw"] = np.array([float(np.max(np.abs(self.last_dw)))], dtype=np.float32)
        d["weight_norm"] = np.array([float(np.linalg.norm(self.W_hh))], dtype=np.float32)
        d["cumulative_abs_dw"] = self.cumulative_abs_dw.copy()
        return d
