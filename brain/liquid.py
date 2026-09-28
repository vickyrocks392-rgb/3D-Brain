"""CAND-LI: Continuous-time Leaky-Integrator / Liquid neural substrate.

Implements continuous dynamical state updates with learnable time constants
as defined in Phase 0 specifications.
"""

from typing import Dict, Tuple
import numpy as np
from brain.base import BrainModel


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


class CandLiquid(BrainModel):
    """Continuous-Time Leaky-Integrator (Liquid) network with learnable time-constants."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42
    ):
        super().__init__(input_dim, hidden_dim, output_dim, seed)

        scale_xh = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hh = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        scale_hy = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W_xh = self.rng.normal(0.0, scale_xh, (input_dim, hidden_dim)).astype(np.float32)
        self.W_hh = self.rng.normal(0.0, scale_hh, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)

        # Learnable log-rate parameter rho such that alpha = sigmoid(rho) in (0, 1)
        # Initialized to rho = -1.0 so alpha ~= 0.27 (decay timescale tau ~= 3.7 steps)
        self.rho = self.rng.normal(-1.0, 0.2, hidden_dim).astype(np.float32)

        self.W_hy = self.rng.normal(0.0, scale_hy, (hidden_dim, output_dim)).astype(np.float32)
        self.b_y = np.zeros(output_dim, dtype=np.float32)

        self.reset()

    @property
    def alpha(self) -> np.ndarray:
        """Effective discretization rate alpha = dt / tau in (0, 1)."""
        return sigmoid(self.rho)

    @property
    def tau(self) -> np.ndarray:
        """Effective time constants tau = 1.0 / alpha."""
        return 1.0 / (self.alpha + 1e-7)

    def reset(self, batch_size: int = 1) -> None:
        self.h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.a = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        alpha = self.alpha
        u = np.tanh(x @ self.W_xh + self.a @ self.W_hh + self.b_h)
        self.h = (1.0 - alpha) * self.h + alpha * u
        self.a = np.tanh(self.h)
        y = self.a @ self.W_hy + self.b_y
        return y

    def forward_sequence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        B, T, _ = X.shape
        self.reset(batch_size=B)

        H = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        A = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        U = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        Y = np.zeros((B, T, self.output_dim), dtype=np.float32)

        H_prev = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        A_prev = np.zeros((B, T, self.hidden_dim), dtype=np.float32)

        h_prev = np.zeros((B, self.hidden_dim), dtype=np.float32)
        a_prev = np.zeros((B, self.hidden_dim), dtype=np.float32)
        alpha = self.alpha  # [hidden_dim]

        for t in range(T):
            x_t = X[:, t, :]
            H_prev[:, t, :] = h_prev
            A_prev[:, t, :] = a_prev

            u_t = np.tanh(x_t @ self.W_xh + a_prev @ self.W_hh + self.b_h)
            h_t = (1.0 - alpha) * h_prev + alpha * u_t
            a_t = np.tanh(h_t)
            y_t = a_t @ self.W_hy + self.b_y

            U[:, t, :] = u_t
            H[:, t, :] = h_t
            A[:, t, :] = a_t
            Y[:, t, :] = y_t

            h_prev = h_t
            a_prev = a_t

        self.h = h_prev
        self.a = a_prev

        cache = {
            "X": X,
            "H": H,
            "A": A,
            "U": U,
            "H_prev": H_prev,
            "A_prev": A_prev,
            "alpha": alpha,
        }
        return Y, H, A, cache

    def backward_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        cache: dict,
    ) -> Dict[str, np.ndarray]:
        B, T, _ = X.shape
        H = cache["H"]
        A = cache["A"]
        U = cache["U"]
        H_prev = cache["H_prev"]
        A_prev = cache["A_prev"]
        alpha = cache["alpha"]

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        drho = np.zeros_like(self.rho)

        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        mask_count = float(np.sum(mask)) + 1e-8
        dY = (2.0 / mask_count) * (Y - targets) * mask

        # Readout grads
        for t in range(T):
            dW_hy += A[:, t, :].T @ dY[:, t, :]
            db_y += np.sum(dY[:, t, :], axis=0)

        dh_next = np.zeros((B, self.hidden_dim), dtype=np.float32)
        da_next = np.zeros((B, self.hidden_dim), dtype=np.float32)

        for t in reversed(range(T)):
            # Total gradient arriving at activation a_t
            da_t = dY[:, t, :] @ self.W_hy.T + da_next
            # Through a_t = tanh(h_t)
            dh_t = da_t * (1.0 - A[:, t, :] ** 2) + dh_next

            # Through h_t = (1 - alpha)*h_prev + alpha*u_t
            du_t = dh_t * alpha
            dalpha_t = np.sum(dh_t * (U[:, t, :] - H_prev[:, t, :]), axis=0)
            drho += dalpha_t * alpha * (1.0 - alpha)

            # Through u_t = tanh(x*W_xh + a_prev*W_hh + b_h)
            dz_t = du_t * (1.0 - U[:, t, :] ** 2)
            dW_xh += X[:, t, :].T @ dz_t
            dW_hh += A_prev[:, t, :].T @ dz_t
            db_h += np.sum(dz_t, axis=0)

            # Gradients propagating back to step t-1
            da_next = dz_t @ self.W_hh.T
            dh_next = dh_t * (1.0 - alpha)

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
        for k, v in weights.items():
            setattr(self, k, v.astype(np.float32))
