"""Tests for Phase 14 Population Resonance & Spatial Synchrony Engine."""

import numpy as np
import pytest

from brain.population_resonance import (
    PopulationResonanceEngine,
    ResonanceRegime,
    SpatialResonanceMode,
    PopulationCodingMode,
)


def test_resonance_engine_initialization():
    engine = PopulationResonanceEngine(
        num_nodes=20,
        regime=ResonanceRegime.R4_EI_BALANCED,
        spatial_mode=SpatialResonanceMode.L1_LOCAL_COMMUNITY,
        seed=42,
    )
    assert engine.num_nodes == 20
    assert engine.phases.shape == (20,)
    assert engine.K.shape == (20, 20)


def test_resonance_step_and_metrics():
    engine = PopulationResonanceEngine(
        num_nodes=24,
        regime=ResonanceRegime.R2_MODERATE,
        spatial_mode=SpatialResonanceMode.L4_DISTANCE_GATED,
        seed=42,
    )

    for _ in range(15):
        out = engine.step(node_activities=np.ones(24, dtype=np.float32) * 0.5)
        assert 0.0 <= out["r_global"] <= 1.0
        assert 0.0 <= out["r_local"] <= 1.0
        assert -np.pi <= out["global_phase"] <= np.pi
        assert "plv" in out


def test_participation_ratio_and_modularity():
    rng = np.random.default_rng(42)
    # High-dimensional random activity
    X_high = rng.normal(0, 1, (50, 20))
    pr_high = PopulationResonanceEngine.compute_participation_ratio(X_high)
    assert pr_high > 1.0

    # Low-dimensional rank-1 synchronized activity
    v = rng.normal(0, 1, (20, 1))
    X_low = rng.normal(0, 1, (50, 1)) @ v.T
    pr_low = PopulationResonanceEngine.compute_participation_ratio(X_low)
    # Synchronization collapses dimensionality: PR should drop
    assert pr_low < pr_high

    # Functional modularity
    comms = np.repeat([0, 1, 2, 3], 5)
    q_func, q_struct = PopulationResonanceEngine.compute_functional_modularity(X_high, comms)
    assert isinstance(q_func, float)
    assert isinstance(q_struct, float)
