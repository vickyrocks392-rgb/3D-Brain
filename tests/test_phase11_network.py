"""Unit tests for Phase11SpatialNetwork integrated substrate."""

import numpy as np
import pytest

from brain.phase11_network import Phase11SpatialNetwork
from brain.predictive_gating import GateMode
from brain.attractor_network import AttractorCouplingMode
from brain.targeted_feedback import FeedbackMode


def test_phase11_network_initialization():
    net = Phase11SpatialNetwork(
        input_dim=8,
        hidden_dim=30,
        output_dim=2,
        seed=42,
    )
    assert net.input_dim == 8
    assert net.hidden_dim == 30
    assert net.output_dim == 2
    assert net.W_rec.shape == (30, 30)
    assert net.positions.shape == (30, 3)
    assert net.parameter_count() > 0


def test_phase11_network_step():
    net = Phase11SpatialNetwork(input_dim=8, hidden_dim=20, output_dim=2, seed=42)
    x = np.ones(8, dtype=np.float32)

    v_act, sel_act = net.step(x, reward=0.0, target_action=0, context_id=0)
    assert v_act.shape == (2,)
    assert sel_act in (0, 1)
    assert net.current_step == 1
    assert "active_attractor" in net.last_telemetry
    assert "gate_selectivity" in net.last_telemetry


def test_phase11_network_forward_sequence():
    net = Phase11SpatialNetwork(input_dim=4, hidden_dim=15, output_dim=2, seed=42)
    X = np.ones((2, 5, 4), dtype=np.float32)

    Y, H, A, cache = net.forward_sequence(X)
    assert Y.shape == (2, 5, 2)
    assert H.shape == (2, 5, 15)
    assert A.shape == (2, 5, 15)


def test_phase11_network_weights_and_reset():
    net = Phase11SpatialNetwork(input_dim=4, hidden_dim=10, output_dim=2, seed=42)
    weights = net.get_weights()
    assert "W_in" in weights
    assert "W_rec" in weights
    assert "W_route" in weights
    assert "W_pred" in weights

    net.reset()
    assert net.current_step == 0
