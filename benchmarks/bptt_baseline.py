"""Compact BPTT / Gradient Descent Reference Baseline for Phase 10 Experiment 010.

NOTE (Per Rule 10, 11, 12, 16):
This gradient baseline exists ONLY to establish an upper / reference bound on the
delayed contextual credit assignment benchmark. It is strictly isolated and clearly
identified as a non-biological gradient control, not as the primary substrate.
"""

from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np

from brain.base import BrainModel, AdamWOptimizer
from brain.discrete_leaky_rnn import DiscreteLeakyRNN


class BPTTGradientBaseline(BrainModel):
    """Compact Leaky RNN trained via exact analytical Backpropagation Through Time (BPTT)."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 50,
        output_dim: int = 1,
        learning_rate: float = 0.01,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.seed = seed

        self.model = DiscreteLeakyRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seed=seed,
        )
        self.optimizer = AdamWOptimizer(lr=learning_rate)

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """Run forward sequence on input trajectory, returning output trajectory."""
        if x_seq.ndim == 2:
            X = np.expand_dims(x_seq, axis=0)
            Y, H, A, cache = self.model.forward_sequence(X)
            self._last_cache = cache
            self._last_X = X
            self._last_Y = Y
            return Y[0]
        else:
            Y, H, A, cache = self.model.forward_sequence(x_seq)
            self._last_cache = cache
            self._last_X = x_seq
            self._last_Y = Y
            return Y

    def train_trial(
        self,
        x_seq: np.ndarray,
        target_action: float,
        choice_step_idx: int,
    ) -> float:
        """Execute one analytical BPTT training step on a complete trial sequence."""
        T = len(x_seq)
        targets = np.zeros((1, T, self.output_dim), dtype=np.float32)
        mask = np.zeros((1, T, self.output_dim), dtype=np.float32)

        targets[0, choice_step_idx, 0] = target_action
        mask[0, choice_step_idx, 0] = 1.0

        outputs = self.forward(x_seq)
        loss = 0.5 * float((outputs[choice_step_idx, 0] - target_action) ** 2)

        grads = self.model.backward_sequence(
            self._last_X,
            self._last_Y,
            targets,
            mask,
            self._last_cache,
        )
        weights = self.model.get_weights()
        self.optimizer.step(weights, grads)
        self.model.set_weights(weights)
        return loss

    def step(self, x: np.ndarray) -> np.ndarray:
        return self.model.step(x)

    def reset(self) -> None:
        self.model.reset()

    def observable_state(self) -> Dict[str, Any]:
        return self.model.observable_state()

    def get_weights(self) -> Dict[str, np.ndarray]:
        return self.model.get_weights()

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.model.set_weights(weights)

    def forward_sequence(
        self,
        X: np.ndarray,
        hidden_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.model.forward_sequence(X, hidden_init)

    def backward_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        cache: dict,
    ) -> Dict[str, np.ndarray]:
        return self.model.backward_sequence(X, Y, targets, mask, cache)

    def parameter_count(self) -> int:
        return (
            self.input_dim * self.hidden_dim
            + self.hidden_dim * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.output_dim
            + self.output_dim
        )
