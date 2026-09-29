"""Tests for Phase15Network integrated architecture and causal action interventions."""

import numpy as np
import pytest

from brain.phase15_network import Phase15Network
from brain.action_readout import (
    InputRepresentationType,
    LearningMechanism,
    TeachingSignalType,
    RetrogradeSignalType,
    DendriticSTDPType,
)
from brain.multi_timescale_assembly import AssemblyTimescaleMode


def test_phase15_network_initialization():
    net = Phase15Network(
        hidden_dim=30,
        output_dim=2,
        input_dim=8,
        seed=42,
    )
    assert net.num_neurons == 30
    assert net.output_dim == 2
    assert net.action_readout is not None
    assert net.multi_timescale_assembly is not None
    assert len(net.action_causal_interventions) == 12


def test_phase15_network_step():
    net = Phase15Network(
        hidden_dim=30,
        output_dim=2,
        input_dim=8,
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
    assert "decision_confidence" in net.last_phase15_telemetry


def test_phase15_causal_interventions():
    net = Phase15Network(
        hidden_dim=30,
        output_dim=2,
        input_dim=8,
        seed=42,
    )

    interventions = [
        "C1_CONTEXT_SCRAMBLE",
        "C2_CONTEXT_SWAP",
        "C3_READOUT_LESION",
        "C4_IRRELEVANT_CONTEXT",
        "C5_LABEL_PERMUTATION",
        "C6_FREEZE_READOUT",
        "C7_ACTION_POP_SWAP",
        "C8_TARGET_SIGNAL_SHUFFLE",
        "C9_TARGET_SIGNAL_SIGN_FLIP",
        "C10_RETROGRADE_REMOVAL",
        "C11_ELIGIBILITY_REMOVAL",
        "C12_TEMPORAL_REP_REMOVAL",
    ]

    x = np.ones(8)
    for c in interventions:
        net.set_action_causal_intervention(c, True)
        assert net.action_causal_interventions[c] is True
        probs, sel = net.step(x=x, target_action=0, context_id=0, is_choice_step=True)
        assert len(probs) == 2
        net.set_action_causal_intervention(c, False)
