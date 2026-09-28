"""Small Causal Transformer Reference Baseline for 3D-Brain Benchmarks.

Implemented in pure NumPy with full causal self-attention, layer normalization,
feedforward MLP, and exact analytical backpropagation. No external deep learning
framework dependencies required.

Enables fair, rigorous empirical comparison between 3D-Brain's recurrent plastic
substrate and conventional attention-based Transformer architectures at comparable
parameter and compute budgets.
"""

from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np


class CausalSelfAttention:
    """Multi-Head Causal Self-Attention in pure NumPy."""

    def __init__(self, d_model: int = 32, num_heads: int = 2, seed: int = 42):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rng = np.random.default_rng(seed)

        scale = 1.0 / np.sqrt(self.d_model)
        self.W_q = self.rng.normal(0.0, scale, (d_model, d_model)).astype(np.float32)
        self.W_k = self.rng.normal(0.0, scale, (d_model, d_model)).astype(np.float32)
        self.W_v = self.rng.normal(0.0, scale, (d_model, d_model)).astype(np.float32)
        self.W_o = self.rng.normal(0.0, scale, (d_model, d_model)).astype(np.float32)

        # Gradients
        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)

        # Cached tensors for backward pass
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward causal self-attention.
        
        Args:
            x: Input sequence [T, d_model]
        Returns:
            out: Contextual representation [T, d_model]
        """
        T, D = x.shape
        Q = np.dot(x, self.W_q)  # [T, D]
        K = np.dot(x, self.W_k)  # [T, D]
        V = np.dot(x, self.W_v)  # [T, D]

        # Reshape into heads: [num_heads, T, d_k]
        Q_h = Q.reshape(T, self.num_heads, self.d_k).transpose(1, 0, 2)
        K_h = K.reshape(T, self.num_heads, self.d_k).transpose(1, 0, 2)
        V_h = V.reshape(T, self.num_heads, self.d_k).transpose(1, 0, 2)

        # Scaled dot-product attention
        scores = np.matmul(Q_h, K_h.transpose(0, 2, 1)) / np.sqrt(self.d_k)  # [H, T, T]

        # Apply causal mask: mask future steps where t_k > t_q
        causal_mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
        scores += causal_mask[np.newaxis, :, :]

        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)

        # Context
        context_h = np.matmul(weights, V_h)  # [H, T, d_k]
        context = context_h.transpose(1, 0, 2).reshape(T, D)  # [T, D]
        out = np.dot(context, self.W_o)  # [T, D]

        self.cache = {
            "x": x,
            "Q_h": Q_h,
            "K_h": K_h,
            "V_h": V_h,
            "weights": weights,
            "context": context,
        }
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Analytical backward pass for causal self-attention."""
        x = self.cache["x"]
        weights = self.cache["weights"]
        V_h = self.cache["V_h"]
        Q_h = self.cache["Q_h"]
        K_h = self.cache["K_h"]
        context = self.cache["context"]
        T, D = x.shape

        self.dW_o = np.dot(context.T, d_out)
        d_context = np.dot(d_out, self.W_o.T)

        d_context_h = d_context.reshape(T, self.num_heads, self.d_k).transpose(1, 0, 2)

        # Gradient w.r.t V
        d_V_h = np.matmul(weights.transpose(0, 2, 1), d_context_h)  # [H, T, d_k]
        d_weights = np.matmul(d_context_h, V_h.transpose(0, 2, 1))  # [H, T, T]

        # Softmax backward
        sum_w_dw = np.sum(d_weights * weights, axis=-1, keepdims=True)
        d_scores = weights * (d_weights - sum_w_dw)
        d_scores = d_scores / np.sqrt(self.d_k)

        # Mask future steps
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        d_scores[:, causal_mask] = 0.0

        # Gradients w.r.t Q and K
        d_Q_h = np.matmul(d_scores, K_h)  # [H, T, d_k]
        d_K_h = np.matmul(d_scores.transpose(0, 2, 1), Q_h)  # [H, T, d_k]

        d_Q = d_Q_h.transpose(1, 0, 2).reshape(T, D)
        d_K = d_K_h.transpose(1, 0, 2).reshape(T, D)
        d_V = d_V_h.transpose(1, 0, 2).reshape(T, D)

        self.dW_q = np.dot(x.T, d_Q)
        self.dW_k = np.dot(x.T, d_K)
        self.dW_v = np.dot(x.T, d_V)

        dx = np.dot(d_Q, self.W_q.T) + np.dot(d_K, self.W_k.T) + np.dot(d_V, self.W_v.T)
        return dx


class LayerNorm:
    """Layer Normalization in pure NumPy."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.eps = eps
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        out = self.gamma * x_norm + self.beta
        self.cache = {"x": x, "mean": mean, "var": var, "std": std, "x_norm": x_norm}
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        x_norm = self.cache["x_norm"]
        std = self.cache["std"]
        N = x_norm.shape[-1]
        self.dgamma = np.sum(d_out * x_norm, axis=0)
        self.dbeta = np.sum(d_out, axis=0)

        dx_norm = d_out * self.gamma
        dvar = np.sum(dx_norm * (self.cache["x"] - self.cache["mean"]) * -0.5 * (std ** -3), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -1.0 / std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (self.cache["x"] - self.cache["mean"]), axis=-1, keepdims=True)
        dx = dx_norm / std + dvar * 2.0 * (self.cache["x"] - self.cache["mean"]) / N + dmean / N
        return dx


class SmallCausalTransformer:
    """Compact Causal Transformer baseline for benchmark comparison against 3D-Brain."""

    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 32,
        num_heads: int = 2,
        d_ff: int = 64,
        output_dim: int = 1,
        max_seq_len: int = 64,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.rng = np.random.default_rng(seed)

        # 1. Input embedding and positional encoding
        self.W_in = self.rng.normal(0.0, 1.0 / np.sqrt(input_dim), (input_dim, d_model)).astype(np.float32)
        self.pos_emb = self.rng.normal(0.0, 0.02, (max_seq_len, d_model)).astype(np.float32)

        # 2. Causal Self-Attention Block
        self.attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads, seed=seed)
        self.ln1 = LayerNorm(d_model)

        # 3. Feedforward MLP
        self.W_ff1 = self.rng.normal(0.0, 1.0 / np.sqrt(d_model), (d_model, d_ff)).astype(np.float32)
        self.b_ff1 = np.zeros(d_ff, dtype=np.float32)
        self.W_ff2 = self.rng.normal(0.0, 1.0 / np.sqrt(d_ff), (d_ff, d_model)).astype(np.float32)
        self.b_ff2 = np.zeros(d_model, dtype=np.float32)
        self.ln2 = LayerNorm(d_model)

        # 4. Readout Head
        self.W_out = self.rng.normal(0.0, 1.0 / np.sqrt(d_model), (d_model, output_dim)).astype(np.float32)
        self.b_out = np.zeros(output_dim, dtype=np.float32)

        # Gradients
        self.dW_in = np.zeros_like(self.W_in)
        self.dpos_emb = np.zeros_like(self.pos_emb)
        self.dW_ff1 = np.zeros_like(self.W_ff1)
        self.db_ff1 = np.zeros_like(self.b_ff1)
        self.dW_ff2 = np.zeros_like(self.W_ff2)
        self.db_ff2 = np.zeros_like(self.b_ff2)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)

        self.cache: Dict[str, Any] = {}

    def parameter_count(self) -> int:
        params = (
            self.W_in.size
            + self.pos_emb.size
            + self.attn.W_q.size * 4
            + self.ln1.gamma.size * 2
            + self.W_ff1.size
            + self.b_ff1.size
            + self.W_ff2.size
            + self.b_ff2.size
            + self.ln2.gamma.size * 2
            + self.W_out.size
            + self.b_out.size
        )
        return int(params)

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """Forward pass through causal Transformer.

        Args:
            x_seq: [T, input_dim] sequence of sensory inputs
        Returns:
            y_seq: [T, output_dim] output predictions
        """
        T = len(x_seq)
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        # 1. Project to d_model + positional encoding
        h0 = np.dot(x_seq, self.W_in) + self.pos_emb[:T]

        # 2. Attention Block + Residual + Norm
        attn_out = self.attn.forward(h0)
        h1 = self.ln1.forward(h0 + attn_out)

        # 3. Feedforward MLP + ReLU + Residual + Norm
        ff1 = np.maximum(0.0, np.dot(h1, self.W_ff1) + self.b_ff1)  # ReLU
        ff2 = np.dot(ff1, self.W_ff2) + self.b_ff2
        h2 = self.ln2.forward(h1 + ff2)

        # 4. Readout
        y = np.dot(h2, self.W_out) + self.b_out

        self.cache = {
            "x_seq": x_seq,
            "h0": h0,
            "attn_out": attn_out,
            "h1": h1,
            "ff1": ff1,
            "h2": h2,
        }
        return y

    def train_step(
        self,
        x_seq: np.ndarray,
        target_action: float,
        action_step_idx: int,
        lr: float = 0.01,
    ) -> float:
        """Execute single backpropagation step matching the delayed contextual choice task."""
        y_seq = self.forward(x_seq)
        pred_action = float(y_seq[action_step_idx, 0])
        loss = 0.5 * (pred_action - target_action) ** 2

        # Backward pass from action_step_idx
        T = len(x_seq)
        dy = np.zeros_like(y_seq)
        dy[action_step_idx, 0] = pred_action - target_action

        # Readout gradients
        h2 = self.cache["h2"]
        self.dW_out = np.dot(h2.T, dy)
        self.db_out = np.sum(dy, axis=0)
        dh2 = np.dot(dy, self.W_out.T)

        # LN2 backward
        dh1_plus_ff2 = self.ln2.backward(dh2)

        # MLP backward
        ff1 = self.cache["ff1"]
        h1 = self.cache["h1"]
        self.dW_ff2 = np.dot(ff1.T, dh1_plus_ff2)
        self.db_ff2 = np.sum(dh1_plus_ff2, axis=0)

        dff1 = np.dot(dh1_plus_ff2, self.W_ff2.T)
        dff1_pre = dff1 * (ff1 > 0).astype(np.float32)  # ReLU grad
        self.dW_ff1 = np.dot(h1.T, dff1_pre)
        self.db_ff1 = np.sum(dff1_pre, axis=0)

        dh1 = dh1_plus_ff2 + np.dot(dff1_pre, self.W_ff1.T)

        # LN1 backward
        dh0_plus_attn = self.ln1.backward(dh1)
        dattn = self.attn.backward(dh0_plus_attn)
        dh0 = dh0_plus_attn + dattn

        # Embedding gradients
        x_seq_cached = self.cache["x_seq"]
        self.dW_in = np.dot(x_seq_cached.T, dh0)
        self.dpos_emb[:T] = dh0

        # Gradient Descent update
        for p, dp in [
            (self.W_in, self.dW_in),
            (self.attn.W_q, self.attn.dW_q),
            (self.attn.W_k, self.attn.dW_k),
            (self.attn.W_v, self.attn.dW_v),
            (self.attn.W_o, self.attn.dW_o),
            (self.W_ff1, self.dW_ff1),
            (self.b_ff1, self.db_ff1),
            (self.W_ff2, self.dW_ff2),
            (self.b_ff2, self.db_ff2),
            (self.W_out, self.dW_out),
            (self.b_out, self.db_out),
        ]:
            p -= lr * np.clip(dp, -1.0, 1.0)

        return float(loss)


def benchmark_transformer_on_task(
    num_train_trials: int = 100,
    num_test_trials: int = 40,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate the reference Transformer on the Delayed Contextual Choice benchmark."""
    from benchmarks.delayed_contextual_choice import DelayedContextualChoiceTask

    task = DelayedContextualChoiceTask(obs_dim=8, seed=seed)
    model = SmallCausalTransformer(input_dim=8, d_model=32, num_heads=2, d_ff=64, seed=seed)

    # Training
    t0 = time.perf_counter()
    train_losses = []
    for tr in range(num_train_trials):
        ctx_id = 1 if (tr % 2 == 0) else 2
        trial_steps = task.generate_trial(context_id=ctx_id, reward_delay=0, trial_seed=seed + tr)
        x_seq = np.array([s.sensory_input for s in trial_steps], dtype=np.float32)
        action_step = [idx for idx, s in enumerate(trial_steps) if s.is_action_step][0]
        target = trial_steps[action_step].target_action

        loss = model.train_step(x_seq, target_action=target, action_step_idx=action_step, lr=0.02)
        train_losses.append(loss)
    train_time = time.perf_counter() - t0

    # Evaluation
    t_eval0 = time.perf_counter()
    correct = 0
    test_losses = []
    for tr in range(num_test_trials):
        ctx_id = 1 if (tr % 2 == 0) else 2
        trial_steps = task.generate_trial(context_id=ctx_id, reward_delay=0, trial_seed=seed + 1000 + tr)
        x_seq = np.array([s.sensory_input for s in trial_steps], dtype=np.float32)
        action_step = [idx for idx, s in enumerate(trial_steps) if s.is_action_step][0]
        target = trial_steps[action_step].target_action

        y_seq = model.forward(x_seq)
        pred_action = float(y_seq[action_step, 0])
        pred_sign = 1.0 if pred_action >= 0.0 else -1.0
        if pred_sign == target:
            correct += 1
        test_losses.append(0.5 * (pred_action - target) ** 2)
    eval_time = time.perf_counter() - t_eval0

    accuracy = float(correct) / float(num_test_trials)
    latency_per_step_ms = (eval_time / (num_test_trials * len(trial_steps))) * 1000.0

    return {
        "architecture": "Causal_Transformer_Baseline",
        "parameters": model.parameter_count(),
        "train_trials": num_train_trials,
        "test_trials": num_test_trials,
        "accuracy": accuracy,
        "mean_test_loss": float(np.mean(test_losses)),
        "train_time_sec": round(train_time, 4),
        "inference_latency_per_step_ms": round(latency_per_step_ms, 3),
        "memory_mb": round((model.parameter_count() * 4) / (1024 * 1024), 4),
    }
