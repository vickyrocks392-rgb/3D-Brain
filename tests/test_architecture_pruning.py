"""Tests for Phase 16 Architecture Pruning & Minimality Verification."""

import numpy as np
import pytest

from brain.phase16_network import (
    Phase16Network,
    Phase16Mechanism,
    ArchitecturePreset,
)
from benchmarks.delayed_credit_assignment import LongDelayContextualCreditTask


def test_pairwise_ablation():
    net = Phase16Network(
        hidden_dim=30,
        output_dim=2,
        input_dim=8,
        preset=ArchitecturePreset.FULL_P15,
        seed=42,
    )
    # Ablate STDP + Retrograde
    net.ablate_mechanism(Phase16Mechanism.D_DENDRITIC_STDP)
    net.ablate_mechanism(Phase16Mechanism.E_RETROGRADE_SIGNALING)
    assert not net.is_mechanism_active(Phase16Mechanism.D_DENDRITIC_STDP)
    assert not net.is_mechanism_active(Phase16Mechanism.E_RETROGRADE_SIGNALING)
    assert len(net.get_removed_mechanisms()) == 2

    x = np.ones(8)
    probs, sel = net.step(x=x, target_action=0, is_choice_step=True)
    assert len(probs) == 2


def test_direct_readout_mode():
    net = Phase16Network(
        hidden_dim=30,
        output_dim=2,
        input_dim=8,
        preset=ArchitecturePreset.MIN_A,
        seed=42,
    )
    assert net.direct_readout is True
    x = np.ones(8)
    probs, sel = net.step(x=x, target_action=0, is_choice_step=True)
    assert np.isclose(np.sum(probs), 1.0)
    assert net.last_phase15_telemetry["total_switches"] == 0
