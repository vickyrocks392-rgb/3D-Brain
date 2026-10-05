"""Typed Probabilistic Decision Baseline (System One-Inspired Reference Model).

SCIENTIFIC INTEGRITY & NON-PROPRIETARY DISCLAIMER:
--------------------------------------------------
This module implements an independent, non-proprietary typed probabilistic decision
baseline inspired strictly by the publicly describable interface characteristics of
"System One" typed probabilistic decision systems (discrete typed feature inputs,
calibrated posterior probability distributions over typed outcomes, and epistemic
uncertainty quantification).

It is NOT the proprietary JEV decision model developed by Typesafe AI, does not contain
any proprietary source code, internal weights, or trade secrets, and makes no claim
of proprietary equivalence. It serves strictly as a fair, reproducible typed probabilistic
decision control for sample-efficiency benchmarking.

Key Architectural Characteristics:
1. Input: Dense typed sensory feature vector x in R^D
2. Output: Strictly typed probability distribution P(Y = c | x) over discrete action classes
3. Uncertainty Quantification: Epistemic entropy H(P) and confidence margin
4. Decision Latency: Deliberation latency scaled inversely by posterior confidence
5. Resource Accounting: Exact parameter count, dynamic state footprint, and analytical FLOPs.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain.base import BrainModel, AdamWOptimizer


class TypedProbabilisticDecisionModel(BrainModel):
    """Calibrated Typed Probabilistic Decision Model (System One-style Interface)."""

    def __init__(
        self,
        input_dim: int = 40,
        output_dim: int = 2,
        learning_rate: float = 0.05,
        temperature: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(input_dim=input_dim, hidden_dim=output_dim, output_dim=output_dim, seed=seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.temperature = max(1e-4, float(temperature))
        self.seed = seed

        # Weight initialization (Xavier / Glorot)
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.W = self.rng.normal(0.0, scale, (input_dim, output_dim)).astype(np.float32)
        self.b = np.zeros(output_dim, dtype=np.float32)

        self.optimizer = AdamWOptimizer(lr=learning_rate, weight_decay=1e-4)

        # Dynamic state (observable)
        self.last_input: np.ndarray = np.zeros(input_dim, dtype=np.float32)
        self.last_logits: np.ndarray = np.zeros(output_dim, dtype=np.float32)
        self.last_probs: np.ndarray = np.ones(output_dim, dtype=np.float32) / output_dim
        self.last_selected_action: int = 0
        self.last_entropy: float = float(np.log(output_dim))
        self.last_confidence: float = 1.0 / output_dim
        self.last_latency_ms: float = 10.0

        # Memory & learning controls
        self.frozen: bool = False
        self.total_updates: int = 0

    def reset(self, batch_size: int = 1) -> None:
        """Reset transient dynamic state without resetting learned weights."""
        self.last_input = np.zeros(self.input_dim, dtype=np.float32)
        self.last_logits = np.zeros(self.output_dim, dtype=np.float32)
        self.last_probs = np.ones(self.output_dim, dtype=np.float32) / self.output_dim
        self.last_selected_action = 0
        self.last_entropy = float(np.log(self.output_dim))
        self.last_confidence = 1.0 / self.output_dim
        self.last_latency_ms = 10.0
        self.h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        self.a = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

    def forward_sequence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process sequence X of shape [batch_size, seq_len, input_dim]."""
        B, T, D = X.shape
        Y = np.zeros((B, T, self.output_dim), dtype=np.float32)
        H = np.zeros((B, T, self.output_dim), dtype=np.float32)
        A = np.zeros((B, T, self.output_dim), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                p = self.forward_probs(X[b, t])
                Y[b, t] = p
                H[b, t] = p
                A[b, t] = p
        return Y, H, A

    def backward_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        cache: dict = None,
    ) -> Dict[str, np.ndarray]:
        """Compute gradients matching BrainModel interface."""
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        B, T, _ = targets.shape
        for b in range(B):
            for t in range(T):
                if mask[b, t, 0] > 0:
                    dy = Y[b, t] - targets[b, t]
                    dW += np.outer(X[b, t], dy)
                    db += dy
        return {"W": dW, "b": db}

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights dictionary."""
        return {"W": self.W, "b": self.b}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set model weights dictionary."""
        self.W = weights["W"].copy()
        self.b = weights["b"].copy()

    def forward_probs(self, x: np.ndarray) -> np.ndarray:
        """Compute typed posterior probability distribution over categorical choices."""
        x_flat = np.asarray(x, dtype=np.float32).ravel()
        if len(x_flat) < self.input_dim:
            padded = np.zeros(self.input_dim, dtype=np.float32)
            padded[: len(x_flat)] = x_flat
            x_flat = padded
        elif len(x_flat) > self.input_dim:
            x_flat = x_flat[: self.input_dim]

        self.last_input = x_flat
        logits = (x_flat @ self.W + self.b) / self.temperature
        self.last_logits = logits

        # Numerically stable softmax
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)
        self.last_probs = probs

        # Epistemic entropy & confidence
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        confidence = float(np.max(probs))
        self.last_entropy = entropy
        self.last_confidence = confidence

        # Decision latency: calibrated deliberative time (ms)
        # Higher uncertainty/entropy increases deliberation latency
        self.last_latency_ms = float(5.0 + 15.0 * (entropy / np.log(max(2, self.output_dim))))

        self.last_selected_action = int(np.argmax(probs))
        return probs

    def step(self, x: np.ndarray) -> np.ndarray:
        """Single-step inference returning the predicted action probabilities."""
        return self.forward_probs(x)

    def train_step(self, x: np.ndarray, target_action: int) -> float:
        """Perform one supervised update via cross-entropy loss with analytical gradients."""
        if self.frozen:
            return 0.0

        probs = self.forward_probs(x)
        target_idx = int(target_action) % self.output_dim

        # Cross-entropy loss
        loss = -float(np.log(probs[target_idx] + 1e-12))

        # Softmax cross-entropy gradient: dL/d(logits) = probs - one_hot
        grad_logits = probs.copy()
        grad_logits[target_idx] -= 1.0

        # Backpropagation to W and b
        dW = np.outer(self.last_input, grad_logits)
        db = grad_logits

        weights = {"W": self.W, "b": self.b}
        grads = {"W": dW, "b": db}
        self.optimizer.step(weights, grads)

        self.W = weights["W"]
        self.b = weights["b"]
        self.total_updates += 1
        return loss

    def freeze(self) -> None:
        """Freeze model parameters for strictly held-out frozen evaluation."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze model parameters."""
        self.frozen = False

    def parameter_count(self) -> int:
        """Total neural parameter count."""
        return self.W.size + self.b.size

    def trainable_parameter_count(self) -> int:
        """Trainable parameter count."""
        return 0 if self.frozen else (self.W.size + self.b.size)

    def dynamic_state_size_floats(self) -> int:
        """Number of floating point state variables in dynamic memory."""
        # last_input (input_dim) + last_logits (output_dim) + last_probs (output_dim) + scalars (4)
        return self.input_dim + 2 * self.output_dim + 4

    def external_memory_bytes(self) -> int:
        """External memory footprint (strictly 0 for this baseline)."""
        return 0

    def analytical_flops_per_step(self) -> Dict[str, int]:
        """Exact analytical FLOPs per inference step."""
        # Matmul: 2 * input_dim * output_dim
        # Bias addition: output_dim
        # Softmax: 3 * output_dim (sub max, exp, sum, div)
        # Entropy & latency: 3 * output_dim + 5
        projection = 2 * self.input_dim * self.output_dim + self.output_dim
        softmax = 3 * self.output_dim
        uncertainty = 3 * self.output_dim + 5
        total = projection + softmax + uncertainty
        return {
            "projection": projection,
            "softmax": softmax,
            "uncertainty": uncertainty,
            "total": total,
        }

    def observable_state(self) -> Dict[str, Any]:
        """Expose actual computational state for telemetry inspection."""
        return {
            "is_proprietary_jev": False,
            "system_type": "TYPED_PROBABILISTIC_DECISION",
            "selected_action": self.last_selected_action,
            "action_probs": [round(float(p), 4) for p in self.last_probs],
            "confidence": round(float(self.last_confidence), 4),
            "entropy": round(float(self.last_entropy), 4),
            "decision_latency_ms": round(float(self.last_latency_ms), 2),
            "total_updates": self.total_updates,
            "frozen": self.frozen,
        }
