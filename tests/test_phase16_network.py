"""Unit tests for Phase16Network: Minimal Sufficient Brain & Causal Architecture Pruning."""

import numpy as np
import pytest

from brain.phase16_network import (
    Phase16Network,
    Phase16Mechanism,
    ArchitecturePreset,
)
from brain.action_readout import (
    TeachingSignalType,
    RetrogradeSignalType,
    DendriticSTDPType,
)
from brain.multi_timescale_assembly import AssemblyTimescaleMode


def test_phase16_initialization():
    net = Phase16Network(
        hidden_dim=30,
        output_dim=2,
        input_dim=8,
        preset=ArchitecturePreset.FULL_P15,
        seed=42,
    )
    assert net.num_neurons == 30
    assert net.output_dim == 2
    assert len(net.get_active_mechanisms()) == 12
    assert len(net.get_removed_mechanisms()) == 0


def test_phase16_presets():
    net = Phase16Network(
        hidden_dim=30,
        output_dim=2,
        input_dim=8,
        preset=ArchitecturePreset.FULL_P15,
        seed=42,
    )
    assert len(net.get_active_mechanisms()) == 12

    # Switch to MIN_A: direct readout
    net.apply_preset(ArchitecturePreset.MIN_A)
    assert net.direct_readout is True
    assert Phase16Mechanism.D_DENDRITIC_STDP.value in net.get_removed_mechanisms()
    assert Phase16Mechanism.E_RETROGRADE_SIGNALING.value in net.get_removed_mechanisms()

    # Switch to MIN_B: single slow assembly
    net.apply_preset(ArchitecturePreset.MIN_B)
    assert net.direct_readout is False
    assert Phase16Mechanism.H_SLOW_ACTION_ASSEMBLY.value in net.get_active_mechanisms()
    assert Phase16Mechanism.F_FAST_ACTION_ASSEMBLY.value in net.get_removed_mechanisms()

    # Switch to MIN_E: minimal candidate
    net.apply_preset(ArchitecturePreset.MIN_E)
    assert Phase16Mechanism.K_PERSISTENT_LATENT_NMDA.value in net.get_active_mechanisms()
    assert Phase16Mechanism.J_TEMPORAL_POPULATION.value in net.get_active_mechanisms()
    assert Phase16Mechanism.A_TARGET_SPECIFIC_TEACHING.value in net.get_active_mechanisms()
    assert Phase16Mechanism.D_DENDRITIC_STDP.value in net.get_removed_mechanisms()
    assert Phase16Mechanism.E_RETROGRADE_SIGNALING.value in net.get_removed_mechanisms()


def test_phase16_ablate_restore():
    net = Phase16Network(
        hidden_dim=30,
        output_dim=2,
        input_dim=8,
        preset=ArchitecturePreset.FULL_P15,
        seed=42,
    )
    assert net.is_mechanism_active(Phase16Mechanism.E_RETROGRADE_SIGNALING) is True

    # Ablate retrograde
    net.ablate_mechanism(Phase16Mechanism.E_RETROGRADE_SIGNALING)
    assert net.is_mechanism_active(Phase16Mechanism.E_RETROGRADE_SIGNALING) is False
    assert net.action_readout.retrograde_type == RetrogradeSignalType.R0_NONE

    # Restore retrograde
    net.restore_mechanism(Phase16Mechanism.E_RETROGRADE_SIGNALING)
    assert net.is_mechanism_active(Phase16Mechanism.E_RETROGRADE_SIGNALING) is True


def test_phase16_complexity_stats():
    net = Phase16Network(
        hidden_dim=40,
        output_dim=2,
        input_dim=8,
        preset=ArchitecturePreset.FULL_P15,
        seed=42,
    )
    stats_full = net.get_complexity_stats()
    assert stats_full["mechanism_count"] == 12
    assert stats_full["readout_synapses"] == 80
    assert stats_full["trainable_parameters"] == 80

    net.apply_preset(ArchitecturePreset.MIN_E)
    stats_min = net.get_complexity_stats()
    assert stats_min["mechanism_count"] < stats_full["mechanism_count"]
    assert stats_min["state_variables"] < stats_full["state_variables"]


def test_phase16_step():
    net = Phase16Network(
        hidden_dim=30,
        output_dim=2,
        input_dim=8,
        preset=ArchitecturePreset.MIN_E,
        seed=42,
    )
    x = np.ones(8)
    probs, sel = net.step(
        x=x,
        target_action=0,
        context_id=0,
        reward=1.0,
        is_choice_step=True,
    )
    assert len(probs) == 2
    assert sel in [0, 1]
    assert np.isclose(np.sum(probs), 1.0)
    assert "credit_specificity_index" in net.last_phase15_telemetry
    assert "active_mechanisms_count" in net.last_phase15_telemetry
    assert net.last_phase15_telemetry["active_mechanisms_count"] == 8
