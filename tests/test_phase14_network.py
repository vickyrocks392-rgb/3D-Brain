"""Tests for Phase 14 Integrated Network."""

import numpy as np
import pytest

from brain.phase14_network import Phase14Network
from brain.burst_dynamics import BurstStrategy
from brain.temporal_coding import TemporalCodingScheme
from brain.population_resonance import ResonanceRegime, SpatialResonanceMode


def test_phase14_network_init():
    net = Phase14Network(
        input_dim=8,
        hidden_dim=40,
        output_dim=2,
        burst_strategy=BurstStrategy.B7_FULL_BURST_TEMPORAL,
        temporal_scheme=TemporalCodingScheme.T8_COMBINED,
        resonance_regime=ResonanceRegime.R4_EI_BALANCED,
        spatial_resonance_mode=SpatialResonanceMode.L1_LOCAL_COMMUNITY,
        seed=42,
    )
    assert net.hidden_dim == 40
    assert net.output_dim == 2
    assert net.burst_engine is not None
    assert net.coincidence_detector is not None
    assert net.resonance_engine is not None


def test_phase14_network_step():
    net = Phase14Network(
        input_dim=8,
        hidden_dim=30,
        output_dim=2,
        seed=42,
    )
    net.reset()
    probs, act = net.step(
        x=np.zeros(8, dtype=np.float32),
        target_action=0,
        context_id=0,
        is_choice_step=True,
    )
    assert len(probs) == 2
    assert act in (0, 1)
    assert len(net.last_11stage_audit) > 0
    assert "burst_representation" in net.last_11stage_audit
    assert "population_temporal" in net.last_11stage_audit


def test_phase14_causal_interventions():
    net = Phase14Network(
        input_dim=8,
        hidden_dim=30,
        output_dim=2,
        seed=42,
    )

    interventions = [
        "C1_REMOVE_BURSTS",
        "C2_REMOVE_COINCIDENCE",
        "C3_REMOVE_NMDA",
        "C4_REMOVE_TEMPORAL",
        "C5_SHUFFLE_TIMING",
        "C6_SHUFFLE_BURSTS",
        "C7_SHUFFLE_COUNTS",
        "C8_DESTROY_PHASE",
        "C11_RANDOMIZE_RESONANCE",
        "C12_FREEZE_PLASTICITY",
    ]

    for c in interventions:
        net.apply_causal_intervention(c, True)
        net.step(np.zeros(8, dtype=np.float32), context_id=0)
        net.apply_causal_intervention(c, False)


def test_phase14_observable_state():
    net = Phase14Network(
        input_dim=8,
        hidden_dim=30,
        output_dim=2,
        seed=42,
    )
    obs = net.observable_state()
    assert "temporal_coding" in obs
    assert "total_bursts" in obs["temporal_coding"]
    assert "r_global" in obs["temporal_coding"]
