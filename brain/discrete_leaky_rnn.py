"""DISC-LRNN: Explicit discrete-time leaky recurrent neural network substrate.

Implements discrete leaky recurrence with independently inspectable autoregressive
persistence factors as required by Experiment 002.
"""

from typing import Dict, Tuple
import numpy as np
from brain.base import BrainModel


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


class DiscreteLeakyRNN(BrainModel):
    """Discrete Leaky Recurrent Neural Network:

    h_t = (1 - alpha) * h_(t-1) + alpha * tanh(x_t * W_xh + h_(t-1) * W_hh + b_h)
    a_t = tanh(h_t)
    y_t = a_t * W_hy + b_y
    """

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

        # Learnable logit of leak rate alpha = sigmoid(rho)
        # Initialized to -1.0 -> alpha ~= 0.27 (retention ~= 0.73)
        self.rho = self.rng.normal(-1.0, 0.2, hidden_dim).astype(np.float32)

        self.W_hy = self.rng.normal(0.0, scale_hy, (hidden_dim, output_dim)).astype(np.float32)
        self.b_y = np.zeros(output_dim, dtype=np.float32)

        # Inspectable dynamic state registers
        self.prev_h = np.zeros((1, hidden_dim), dtype=np.float32)
        self.cand_h = np.zeros((1, hidden_dim), dtype=np.float32)

        self.reset()

    @property
    def alpha(self) -> np.ndarray:
        """Effective leak/update rate alpha in (0, 1)."""
        return sigmoid(self.rho)

    def reset(self, batch_size: int = 1) -> None:
        self.h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.a = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.prev_h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.cand_h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        alpha = self.alpha
        self.prev_h = self.h.copy()
        z = x @ self.W_xh + self.prev_h @ self.W_hh + self.b_h
        self.cand_h = np.tanh(z)
        self.h = (1.0 - alpha) * self.prev_h + alpha * self.cand_h
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
        C = np.zeros((B, T, self.hidden_dim), dtype=np.float32)  # Candidate states
        Y = np.zeros((B, T, self.output_dim), dtype=np.float32)

        H_prev = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        h_prev = np.zeros((B, self.hidden_dim), dtype=np.float32)
        alpha = self.alpha

        for t in range(T):
            x_t = X[:, t, :]
            H_prev[:, t, :] = h_prev

            z_t = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h
            c_t = np.tanh(z_t)
            h_t = (1.0 - alpha) * h_prev + alpha * c_t
            a_t = np.tanh(h_t)
            y_t = a_t @ self.W_hy + self.b_y

            C[:, t, :] = c_t
            H[:, t, :] = h_t
            A[:, t, :] = a_t
            Y[:, t, :] = y_t

            h_prev = h_t

        self.h = h_prev
        self.a = np.tanh(h_prev)
        self.prev_h = H_prev[:, -1, :].copy()
        self.cand_h = C[:, -1, :].copy()

        cache = {
            "X": X,
            "H": H,
            "A": A,
            "C": C,
            "H_prev": H_prev,
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
        C = cache["C"]
        H_prev = cache["H_prev"]
        alpha = cache["alpha"]

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        drho = np.zeros_like(self.rho)

        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        mask_count = float(np.sum(mask)) + 1e-8
        dY = (2.0 / mask_count) * (Y - targets) * mask

        # Readout gradients
        for t in range(T):
            dW_hy += A[:, t, :].T @ dY[:, t, :]
            db_y += np.sum(dY[:, t, :], axis=0)

        dh_next = np.zeros((B, self.hidden_dim), dtype=np.float32)

        for t in reversed(range(T)):
            da_t = dY[:, t, :] @ self.W_hy.T
            # Through a_t = tanh(h_t)
            dh_t = da_t * (1.0 - A[:, t, :] ** 2) + dh_next

            # Through h_t = (1 - alpha)*h_prev + alpha*c_t
            dc_t = dh_t * alpha
            dalpha_t = np.sum(dh_t * (C[:, t, :] - H_prev[:, t, :]), axis=0)
            drho += dalpha_t * alpha * (1.0 - alpha)

            # Through c_t = tanh(z_t)
            dz_t = dc_t * (1.0 - C[:, t, :] ** 2)
            dW_xh += X[:, t, :].T @ dz_t
            dW_hh += H_prev[:, t, :].T @ dz_t
            db_h += np.sum(dz_t, axis=0)

            # Gradient back to h_prev: direct leaky branch + recurrent branch through z_t
            dh_prev_leak = dh_t * (1.0 - alpha)
            dh_prev_rec = dz_t @ self.W_hh.T
            dh_next = dh_prev_leak + dh_prev_rec

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

    def observable_state(self) -> Dict[str, np.ndarray]:
        """Return actual internal computational variables for observability."""
        return {
            "state": self.get_state(),
            "activation": self.get_activation(),
            "previous_state": self.prev_h.copy(),
            "candidate_state": self.cand_h.copy(),
            "alpha": self.alpha.copy(),
        }
