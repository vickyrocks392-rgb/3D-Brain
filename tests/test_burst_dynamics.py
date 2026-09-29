"""Tests for Phase 14 Burst Dynamics Engine (B0–B7)."""

import numpy as np
import pytest

from brain.burst_dynamics import BurstDynamicsEngine, BurstStrategy, BurstRecord


def test_burst_engine_initialization():
    engine = BurstDynamicsEngine(
        num_actions=2,
        neurons_per_action=8,
        strategy=BurstStrategy.B7_FULL_BURST_TEMPORAL,
    )
    assert engine.num_actions == 2
    assert engine.neurons_per_action == 8
    assert engine.calcium.shape == (2, 8)
    assert engine.v_soma.shape == (2, 8)


def test_burst_strategies_execution():
    strategies = [
        BurstStrategy.B0_POINT_PROCESS,
        BurstStrategy.B1_SIMPLE_BURST,
        BurstStrategy.B2_CALCIUM_DEPENDENT,
        BurstStrategy.B3_DENDRITIC_PLATEAU,
        BurstStrategy.B4_NMDA_CALCIUM_COINCIDENCE,
        BurstStrategy.B5_ADAPTIVE_BURST_THRESHOLD,
        BurstStrategy.B6_BURST_POPULATION,
        BurstStrategy.B7_FULL_BURST_TEMPORAL,
    ]

    for strat in strategies:
        engine = BurstDynamicsEngine(
            num_actions=2,
            neurons_per_action=4,
            strategy=strat,
            seed=42,
        )
        engine.reset()
        for t in range(15):
            out = engine.step(
                input_drives=np.array([0.8, 0.2], dtype=np.float32),
                v_dend=np.array([-40.0, -60.0], dtype=np.float32),
                g_nmda=np.array([0.4, 0.0], dtype=np.float32),
            )
            assert out["spikes"].shape == (2, 4)
            assert out["selected_action"] in (0, 1)
            assert "pool_evidence" in out


def test_burst_records_and_statistics():
    engine = BurstDynamicsEngine(
        num_actions=2,
        neurons_per_action=8,
        strategy=BurstStrategy.B7_FULL_BURST_TEMPORAL,
        seed=42,
    )
    # Deliver strong drive to trigger bursts
    for t in range(25):
        engine.step(
            input_drives=np.array([0.9, 0.1], dtype=np.float32),
            v_dend=np.array([-38.0, -65.0], dtype=np.float32),
            g_nmda=np.array([0.5, 0.0], dtype=np.float32),
        )

    stats = engine.get_recent_burst_statistics()
    assert isinstance(stats, dict)
    assert "burst_count" in stats
    assert "mean_spikes_per_burst" in stats
    assert "burst_frequency_hz" in stats
