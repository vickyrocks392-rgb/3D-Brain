"""Unit tests for Asynchronous-Irregular (AI) regime diagnostics and Multi-Scale Synchrony."""

import numpy as np
import pytest

from brain.ai_regime_analysis import (
    compute_cv_isi,
    compute_fano_factor,
    compute_pairwise_spike_correlation,
    compute_autocorrelation_decay,
    estimate_branching_ratio,
    evaluate_ai_regime_diagnostic,
)
from brain.synchronization import (
    extract_instantaneous_phases,
    compute_group_kuramoto_and_plv,
    analyze_multiscale_synchronization,
)
from brain.ei_analysis import compute_ei_balance_metrics


def test_cv_isi_and_fano_factor():
    """Verify CV_ISI and Fano factor computation on regular vs Poisson-like spike trains."""
    T, N = 1000, 10
    dt_ms = 1.0

    # 1. Perfectly regular clock-like spikes every 50ms
    regular_spikes = np.zeros((T, N), dtype=float)
    regular_spikes[::50, :] = 1.0
    cv_mean, cv_std, _ = compute_cv_isi(regular_spikes, dt_ms=dt_ms)
    assert cv_mean == pytest.approx(0.0, abs=1e-3)

    # 2. Random Poisson-like spikes (independent Bernoulli)
    rng = np.random.default_rng(42)
    poisson_spikes = (rng.random((T, N)) < 0.05).astype(float)
    cv_p, _, _ = compute_cv_isi(poisson_spikes, dt_ms=dt_ms)
    assert cv_p > 0.70  # Poisson process has CV_ISI ~ 1.0

    fano = compute_fano_factor(poisson_spikes, dt_ms=dt_ms, window_ms=50.0)
    assert 0.70 <= fano <= 1.30


def test_pairwise_spike_correlation():
    """Verify pairwise spike count correlation computation."""
    T, N = 500, 6
    # Perfectly synchronous spikes across all neurons
    sync_spikes = np.zeros((T, N), dtype=float)
    sync_spikes[::20, :] = 1.0
    r_mean, _ = compute_pairwise_spike_correlation(sync_spikes, bin_size_steps=10)
    assert r_mean > 0.90

    # Independent uncorrelated spikes
    rng = np.random.default_rng(42)
    indep_spikes = (rng.random((T, N)) < 0.08).astype(float)
    r_indep, _ = compute_pairwise_spike_correlation(indep_spikes, bin_size_steps=10)
    assert abs(r_indep) < 0.15


def test_ai_regime_diagnostic_evaluation():
    """Verify that operational AI diagnostic classifies synthetic regular vs irregular regimes."""
    T, N = 500, 20
    rng = np.random.default_rng(42)

    # Synthetic AI regime: active irregular Poisson spikes, low global synchrony
    ai_spikes = (rng.random((T, N)) < 0.03).astype(float)
    diag = evaluate_ai_regime_diagnostic(ai_spikes, kuramoto_r_global=0.15, dt_ms=1.0)

    assert "ai_score" in diag
    assert "is_ai_regime" in diag
    assert diag["ai_score"] >= 0.60
    assert diag["criteria"]["low_correlation"] is True
    assert diag["criteria"]["global_asynchrony"] is True


def test_multiscale_synchronization():
    """Verify multi-scale spatial synchronization analysis."""
    T, N = 300, 16
    rng = np.random.default_rng(42)
    spikes = (rng.random((T, N)) < 0.04).astype(float)
    positions = rng.uniform(-3.0, 3.0, size=(N, 3))

    sync_res = analyze_multiscale_synchronization(
        spike_matrix=spikes,
        positions_3d=positions,
        k_neighbors=3,
        dt_ms=1.0,
    )

    for key in ["r_global", "r_local", "r_community", "plv_global", "plv_local", "synchrony_gradient"]:
        assert key in sync_res
        assert np.isfinite(sync_res[key])
