"""BASE-RNN: Standard Elman recurrent neural network substrate.

Serves as the foundational baseline control for Experiment 001.
"""

from typing import Dict, Tuple
import numpy as np
from brain.base import BrainModel


class BaseRNN(BrainModel):
    """Standard Elman Recurrent Neural Network with tanh activation."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42
    ):
        super().__init__(input_dim, hidden_dim, output_dim, seed)

        # Xavier/Glorot initialization
        scale_xh = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hh = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        scale_hy = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W_xh = (self.rng.normal(0.0, scale_xh, (input_dim, hidden_dim))).astype(
            np.float32
        )
        self.W_hh = (self.rng.normal(0.0, scale_hh, (hidden_dim, hidden_dim))).astype(
            np.float32
        )
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)

        self.W_hy = (self.rng.normal(0.0, scale_hy, (hidden_dim, output_dim))).astype(
            np.float32
        )
        self.b_y = np.zeros(output_dim, dtype=np.float32)

        self.reset()

    def reset(self, batch_size: int = 1) -> None:
        self.h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.a = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        # x: [batch_size, input_dim]
        z = x @ self.W_xh + self.h @ self.W_hh + self.b_h
        self.h = np.tanh(z)
        self.a = self.h
        y = self.a @ self.W_hy + self.b_y
        return y

    def forward_sequence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        # X: [batch_size, seq_len, input_dim]
        B, T, _ = X.shape
        self.reset(batch_size=B)

        H = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        A = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        Y = np.zeros((B, T, self.output_dim), dtype=np.float32)

        h_prev = np.zeros((B, self.hidden_dim), dtype=np.float32)
        H_prev_all = np.zeros((B, T, self.hidden_dim), dtype=np.float32)

        for t in range(T):
            x_t = X[:, t, :]
            H_prev_all[:, t, :] = h_prev
            z_t = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h
            h_t = np.tanh(z_t)
            a_t = h_t
            y_t = a_t @ self.W_hy + self.b_y

            H[:, t, :] = h_t
            A[:, t, :] = a_t
            Y[:, t, :] = y_t
            h_prev = h_t

        self.h = h_prev
        self.a = h_prev

        cache = {
            "X": X,
            "H": H,
            "A": A,
            "H_prev": H_prev_all,
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
        H_prev = cache["H_prev"]

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        # Output error: dL/dY = 2*(Y - target)*mask / total_valid_steps
        mask_count = float(np.sum(mask)) + 1e-8
        dY = (2.0 / mask_count) * (Y - targets) * mask  # [B, T, output_dim]

        # Readout gradients
        for t in range(T):
            dW_hy += H[:, t, :].T @ dY[:, t, :]
            db_y += np.sum(dY[:, t, :], axis=0)

        # Backpropagation Through Time (BPTT)
        dh_next = np.zeros((B, self.hidden_dim), dtype=np.float32)

        for t in reversed(range(T)):
            # Direct gradient from output plus recurrent gradient from t+1
            dh_t = dY[:, t, :] @ self.W_hy.T + dh_next
            # Derivative of tanh: 1 - h^2
            dz_t = dh_t * (1.0 - H[:, t, :] ** 2)

            dW_xh += X[:, t, :].T @ dz_t
            dW_hh += H_prev[:, t, :].T @ dz_t
            db_h += np.sum(dz_t, axis=0)

            # Recurrent gradient for previous step
            dh_next = dz_t @ self.W_hh.T

        return {
            "W_xh": dW_xh,
            "W_hh": dW_hh,
            "b_h": db_h,
            "W_hy": dW_hy,
            "b_y": db_y,
        }

    def parameter_count(self) -> int:
        return (
            self.W_xh.size
            + self.W_hh.size
            + self.b_h.size
            + self.W_hy.size
            + self.b_y.size
        )

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            "W_xh": self.W_xh.copy(),
            "W_hh": self.W_hh.copy(),
            "b_h": self.b_h.copy(),
            "W_hy": self.W_hy.copy(),
            "b_y": self.b_y.copy(),
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.W_xh = weights["W_xh"].astype(np.float32)
        self.W_hh = weights["W_hh"].astype(np.float32)
        self.b_h = weights["b_h"].astype(np.float32)
        self.W_hy = weights["W_hy"].astype(np.float32)
        self.b_y = weights["b_y"].astype(np.float32)
