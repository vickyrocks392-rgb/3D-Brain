"""BASE-GRU: Standard Gated Recurrent Unit neural substrate.

Serves as the high-capacity discrete gated control baseline for Experiment 001.
"""

from typing import Dict, Tuple
import numpy as np
from brain.base import BrainModel


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


class BaseGRU(BrainModel):
    """Gated Recurrent Unit (Cho et al., 2014) with exact analytical gradients."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42
    ):
        super().__init__(input_dim, hidden_dim, output_dim, seed)

        scale_x = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_h = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        scale_y = np.sqrt(2.0 / (hidden_dim + output_dim))

        # Reset gate
        self.W_xr = self.rng.normal(0.0, scale_x, (input_dim, hidden_dim)).astype(np.float32)
        self.W_hr = self.rng.normal(0.0, scale_h, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_r = np.zeros(hidden_dim, dtype=np.float32)

        # Update gate (bias initialized to 1.0 to encourage initial memory retention)
        self.W_xz = self.rng.normal(0.0, scale_x, (input_dim, hidden_dim)).astype(np.float32)
        self.W_hz = self.rng.normal(0.0, scale_h, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_z = np.ones(hidden_dim, dtype=np.float32)

        # Candidate state
        self.W_xh = self.rng.normal(0.0, scale_x, (input_dim, hidden_dim)).astype(np.float32)
        self.W_hh = self.rng.normal(0.0, scale_h, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)

        # Readout
        self.W_hy = self.rng.normal(0.0, scale_y, (hidden_dim, output_dim)).astype(np.float32)
        self.b_y = np.zeros(output_dim, dtype=np.float32)

        self.reset()

    def reset(self, batch_size: int = 1) -> None:
        self.h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.a = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        r = sigmoid(x @ self.W_xr + self.h @ self.W_hr + self.b_r)
        z = sigmoid(x @ self.W_xz + self.h @ self.W_hz + self.b_z)
        h_cand = np.tanh(x @ self.W_xh + (r * self.h) @ self.W_hh + self.b_h)
        self.h = (1.0 - z) * self.h + z * h_cand
        self.a = self.h
        y = self.a @ self.W_hy + self.b_y
        return y

    def forward_sequence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        B, T, _ = X.shape
        self.reset(batch_size=B)

        H = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        A = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        Y = np.zeros((B, T, self.output_dim), dtype=np.float32)

        R = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        Z = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        H_cand = np.zeros((B, T, self.hidden_dim), dtype=np.float32)
        H_prev = np.zeros((B, T, self.hidden_dim), dtype=np.float32)

        h_prev = np.zeros((B, self.hidden_dim), dtype=np.float32)

        for t in range(T):
            x_t = X[:, t, :]
            H_prev[:, t, :] = h_prev

            r_t = sigmoid(x_t @ self.W_xr + h_prev @ self.W_hr + self.b_r)
            z_t = sigmoid(x_t @ self.W_xz + h_prev @ self.W_hz + self.b_z)
            h_cand_t = np.tanh(x_t @ self.W_xh + (r_t * h_prev) @ self.W_hh + self.b_h)
            h_t = (1.0 - z_t) * h_prev + z_t * h_cand_t

            R[:, t, :] = r_t
            Z[:, t, :] = z_t
            H_cand[:, t, :] = h_cand_t
            H[:, t, :] = h_t
            A[:, t, :] = h_t
            Y[:, t, :] = h_t @ self.W_hy + self.b_y
            h_prev = h_t

        self.h = h_prev
        self.a = h_prev

        cache = {
            "X": X,
            "H": H,
            "A": A,
            "R": R,
            "Z": Z,
            "H_cand": H_cand,
            "H_prev": H_prev,
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
        R = cache["R"]
        Z = cache["Z"]
        H_cand = cache["H_cand"]
        H_prev = cache["H_prev"]

        dW_xr = np.zeros_like(self.W_xr)
        dW_hr = np.zeros_like(self.W_hr)
        db_r = np.zeros_like(self.b_r)

        dW_xz = np.zeros_like(self.W_xz)
        dW_hz = np.zeros_like(self.W_hz)
        db_z = np.zeros_like(self.b_z)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)

        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        mask_count = float(np.sum(mask)) + 1e-8
        dY = (2.0 / mask_count) * (Y - targets) * mask

        # Readout grads
        for t in range(T):
            dW_hy += H[:, t, :].T @ dY[:, t, :]
            db_y += np.sum(dY[:, t, :], axis=0)

        dh_next = np.zeros((B, self.hidden_dim), dtype=np.float32)

        for t in reversed(range(T)):
            dh_t = dY[:, t, :] @ self.W_hy.T + dh_next
            r_t = R[:, t, :]
            z_t = Z[:, t, :]
            h_cand_t = H_cand[:, t, :]
            h_prev_t = H_prev[:, t, :]

            # Gradients through h_t = (1 - z_t)*h_prev + z_t*h_cand
            dz_t = dh_t * (h_cand_t - h_prev_t)
            dh_cand_t = dh_t * z_t
            dh_prev_direct = dh_t * (1.0 - z_t)

            # Through h_cand = tanh(x*W_xh + (r*h_prev)*W_hh + b_h)
            dz_cand = dh_cand_t * (1.0 - h_cand_t**2)
            dW_xh += X[:, t, :].T @ dz_cand
            dW_hh += (r_t * h_prev_t).T @ dz_cand
            db_h += np.sum(dz_cand, axis=0)

            dr_h_prev = dz_cand @ self.W_hh.T
            dr_t = dr_h_prev * h_prev_t
            dh_prev_from_cand = dr_h_prev * r_t

            # Through z_t = sigmoid(x*W_xz + h_prev*W_hz + b_z)
            dz_act = dz_t * z_t * (1.0 - z_t)
            dW_xz += X[:, t, :].T @ dz_act
            dW_hz += h_prev_t.T @ dz_act
            db_z += np.sum(dz_act, axis=0)
            dh_prev_from_z = dz_act @ self.W_hz.T

            # Through r_t = sigmoid(x*W_xr + h_prev*W_hr + b_r)
            dr_act = dr_t * r_t * (1.0 - r_t)
            dW_xr += X[:, t, :].T @ dr_act
            dW_hr += h_prev_t.T @ dr_act
            db_r += np.sum(dr_act, axis=0)
            dh_prev_from_r = dr_act @ self.W_hr.T

            # Accumulated gradient to pass to step t-1
            dh_next = dh_prev_direct + dh_prev_from_cand + dh_prev_from_z + dh_prev_from_r

        return {
            "W_xr": dW_xr,
            "W_hr": dW_hr,
            "b_r": db_r,
            "W_xz": dW_xz,
            "W_hz": dW_hz,
            "b_z": db_z,
            "W_xh": dW_xh,
            "W_hh": dW_hh,
            "b_h": db_h,
            "W_hy": dW_hy,
            "b_y": db_y,
        }

    def parameter_count(self) -> int:
        return (
            self.W_xr.size
            + self.W_hr.size
            + self.b_r.size
            + self.W_xz.size
            + self.W_hz.size
            + self.b_z.size
            + self.W_xh.size
            + self.W_hh.size
            + self.b_h.size
            + self.W_hy.size
            + self.b_y.size
        )

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            "W_xr": self.W_xr.copy(),
            "W_hr": self.W_hr.copy(),
            "b_r": self.b_r.copy(),
            "W_xz": self.W_xz.copy(),
            "W_hz": self.W_hz.copy(),
            "b_z": self.b_z.copy(),
            "W_xh": self.W_xh.copy(),
            "W_hh": self.W_hh.copy(),
            "b_h": self.b_h.copy(),
            "W_hy": self.W_hy.copy(),
            "b_y": self.b_y.copy(),
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        for k, v in weights.items():
            setattr(self, k, v.astype(np.float32))
