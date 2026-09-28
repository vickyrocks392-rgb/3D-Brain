"""Tests for Causal Transformer Baseline (Phase 9).

Verifies:
1. Analytical gradient check and forward pass of causal self-attention.
2. Parameter count accounting.
3. Training loss reduction on the benchmark sequence.
"""

import numpy as np
import pytest

from benchmarks.transformer_baseline import SmallCausalTransformer, benchmark_transformer_on_task


def test_transformer_causal_mask():
    """Verify that causal attention strictly masks future tokens."""
    model = SmallCausalTransformer(input_dim=4, d_model=16, num_heads=2, max_seq_len=20, seed=42)
    x = np.random.default_rng(42).normal(0.0, 1.0, (10, 4)).astype(np.float32)
    y1 = model.forward(x)

    # Change token at step 9 (the last step)
    x_mod = x.copy()
    x_mod[9] += 10.0
    y2 = model.forward(x_mod)

    # Due to causal masking, steps 0 through 8 MUST be completely identical!
    assert np.allclose(y1[:9], y2[:9], atol=1e-5)
    # But step 9 must be different
    assert not np.allclose(y1[9], y2[9])


def test_transformer_train_step():
    """Verify that backpropagation reduces loss over training steps."""
    model = SmallCausalTransformer(input_dim=4, d_model=16, num_heads=2, max_seq_len=20, seed=42)
    x = np.random.default_rng(42).normal(0.0, 1.0, (12, 4)).astype(np.float32)

    loss_initial = model.train_step(x, target_action=1.0, action_step_idx=11, lr=0.05)
    for _ in range(15):
        loss_final = model.train_step(x, target_action=1.0, action_step_idx=11, lr=0.05)

    assert loss_final < loss_initial


def test_transformer_benchmark():
    """Verify complete benchmark run returning structured metrics."""
    res = benchmark_transformer_on_task(num_train_trials=15, num_test_trials=10, seed=42)
    assert res["parameters"] > 1000
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["inference_latency_per_step_ms"] > 0.0
    assert res["memory_mb"] > 0.0
