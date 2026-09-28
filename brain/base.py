"""Abstract base class and optimization utilities for 3D-Brain computational substrates."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np


class BrainModel(ABC):
    """Common experimental interface for all candidate substrates."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.h = np.zeros((1, hidden_dim), dtype=np.float32)
        self.a = np.zeros((1, hidden_dim), dtype=np.float32)

    @abstractmethod
    def reset(self, batch_size: int = 1) -> None:
        """Reset internal hidden state to resting values."""
        pass

    @abstractmethod
    def step(self, x: np.ndarray) -> np.ndarray:
        """Step one time increment for input x [batch_size, input_dim].
        
        Returns output y [batch_size, output_dim].
        """
        pass

    @abstractmethod
    def forward_sequence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process sequence X of shape [batch_size, seq_len, input_dim].
        
        Returns:
            Y: [batch_size, seq_len, output_dim]
            H: [batch_size, seq_len, hidden_dim] (internal state)
            A: [batch_size, seq_len, hidden_dim] (output activations)
        """
        pass

    @abstractmethod
    def backward_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        cache: dict,
    ) -> Dict[str, np.ndarray]:
        """Compute gradients via Backpropagation Through Time (BPTT).
        
        Returns:
            Dictionary mapping parameter names to gradient arrays.
        """
        pass

    @abstractmethod
    def parameter_count(self) -> int:
        """Return exact total count of trainable parameters."""
        pass

    @abstractmethod
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return dict of weight matrices and biases."""
        pass

    @abstractmethod
    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set model weights from dictionary."""
        pass

    def get_state(self) -> np.ndarray:
        """Return current hidden state vector."""
        return self.h.copy()

    def get_activation(self) -> np.ndarray:
        """Return current activation vector."""
        return self.a.copy()


class AdamWOptimizer:
    """Deterministic, self-contained AdamW optimizer with gradient clipping."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        clip_norm: float = 1.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t = 0

    def step(
        self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]
    ) -> None:
        """Perform in-place AdamW parameter update with global gradient clipping."""
        self.t += 1

        # Global gradient norm clipping
        total_norm_sq = 0.0
        for g in grads.values():
            total_norm_sq += float(np.sum(g**2))
        total_norm = np.sqrt(total_norm_sq)

        scale = 1.0
        if total_norm > self.clip_norm:
            scale = self.clip_norm / (total_norm + 1e-9)

        for name, p in params.items():
            if name not in grads:
                continue
            g = grads[name] * scale

            if name not in self.m:
                self.m[name] = np.zeros_like(p)
                self.v[name] = np.zeros_like(p)

            # Update biased 1st and 2nd moment estimates
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (g**2)

            # Bias correction
            m_hat = self.m[name] / (1.0 - self.beta1**self.t)
            v_hat = self.v[name] / (1.0 - self.beta2**self.t)

            # Weight decay (excluding 1D biases or time-constants)
            if p.ndim > 1 and self.weight_decay > 0:
                p -= self.lr * self.weight_decay * p

            # Parameter update
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
