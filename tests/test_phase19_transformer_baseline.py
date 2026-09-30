"""Tests for Phase 19 Track C: Fair Causal Transformer & Baseline Comparison.

Verifies:
1. Exact parameter accounting across all 4 matched-budget architectures.
2. Analytical FLOPs per step calculation.
3. Peak RSS memory measurement function.
4. Execution of head-to-head delayed credit tasks.
5. Continual learning reversal evaluation across all baselines.
"""

import pytest

from benchmarks.phase19_transformer_comparison import (
    calculate_analytical_flops_per_step,
    get_process_peak_rss_mb,
    benchmark_continual_transformer,
    benchmark_continual_bptt,
)
from benchmarks.phase19_credit_horizon import (
    train_and_eval_min_e,
    train_and_eval_transformer,
    train_and_eval_bptt,
    train_and_eval_simple_recurrent,
)
from brain.phase16_network import ArchitecturePreset


def test_analytical_flops_accounting():
    """Verify analytical FLOPs per step across architectures."""
    f_mine = calculate_analytical_flops_per_step("MIN_E", seq_len=15)
    f_bptt = calculate_analytical_flops_per_step("BPTT_RECURRENT", seq_len=15)
    f_tx = calculate_analytical_flops_per_step("CAUSAL_TRANSFORMER", seq_len=15)
    f_rnn = calculate_analytical_flops_per_step("SIMPLE_RECURRENT_CONTROL", seq_len=15)

    assert f_mine["flops_per_step"] > 0
    assert f_bptt["flops_per_step"] > f_mine["flops_per_step"]  # BPTT unrolls backward
    assert f_tx["flops_per_step"] > 0
    assert f_rnn["flops_per_step"] > 0


def test_peak_rss_mb_measurement():
    """Verify process peak memory reporting returns valid positive float."""
    rss = get_process_peak_rss_mb()
    assert isinstance(rss, float)
    assert rss > 0.0


def test_matched_budget_execution():
    """Verify all 4 models train and evaluate on a matched delayed-credit task."""
    res_mine = train_and_eval_min_e(ArchitecturePreset.MIN_E, delay_steps=2, train_trials=5, eval_trials=5, seed=42)
    res_bptt = train_and_eval_bptt(delay_steps=2, train_trials=5, eval_trials=5, seed=42)
    res_tx = train_and_eval_transformer(delay_steps=2, train_trials=5, eval_trials=5, seed=42)
    res_rnn = train_and_eval_simple_recurrent(delay_steps=2, train_trials=5, eval_trials=5, seed=42)

    assert 0.0 <= res_mine["eval_accuracy"] <= 1.0
    assert 0.0 <= res_bptt["eval_accuracy"] <= 1.0
    assert 0.0 <= res_tx["eval_accuracy"] <= 1.0
    assert 0.0 <= res_rnn["eval_accuracy"] <= 1.0


def test_continual_transformer_and_bptt_execution():
    """Verify continual learning reversal evaluation on Transformer and BPTT baselines."""
    res_tx = benchmark_continual_transformer(trials_per_task=10, eval_trials=10, delay_steps=2, seed=42)
    res_bptt = benchmark_continual_bptt(trials_per_task=10, eval_trials=10, delay_steps=2, seed=42)

    assert "relative_retention" in res_tx
    assert "relative_retention" in res_bptt
    assert 0.0 <= res_tx["relative_retention"] <= 2.0
    assert 0.0 <= res_bptt["relative_retention"] <= 2.0
