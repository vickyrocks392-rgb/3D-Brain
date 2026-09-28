"""Unit tests for TwoCompartmentSpatialNetwork (Phase 10)."""

import numpy as np
import pytest

from brain.spatial import TopologyFamily
from brain.two_compartment_spatial import TwoCompartmentSpatialNetwork


def test_two_compartment_spatial_network_initialization():
    net = TwoCompartmentSpatialNetwork(
        input_dim=8,
        hidden_dim=30,
        output_dim=1,
        topology_type=TopologyFamily.MODULAR,
        inhibitory_fraction=0.20,
        seed=42,
    )
    assert net.hidden_dim == 30
    assert net.positions.shape == (30, 3)
    assert net.delay_matrix.shape == (30, 30)
    assert net.W_rec.shape == (30, 30)
    assert net.W_in.shape == (30, 8)
    assert net.W_out.shape == (1, 30)


def test_two_compartment_network_step():
    net = TwoCompartmentSpatialNetwork(
        input_dim=8,
        hidden_dim=20,
        output_dim=1,
        seed=42,
    )
    x = np.ones(8, dtype=np.float32)
    y = net.step(x, reward=0.0)
    assert y.shape == (1,)
    assert net.current_step == 1

    obs = net.observable_state()
    assert "v_soma" in obs
    assert "v_dend" in obs
    assert "g_nmda" in obs
    assert "i_coupling" in obs
    assert len(obs["v_soma"]) == 20
    assert len(obs["v_dend"]) == 20


def test_two_compartment_plasticity_application():
    net = TwoCompartmentSpatialNetwork(
        input_dim=4,
        hidden_dim=15,
        output_dim=1,
        seed=42,
    )
    x = np.ones(4, dtype=np.float32)
    initial_w = net.W_rec.copy()

    # Step without reward
    for _ in range(5):
        net.step(x, reward=0.0)

    # Step with reward
    net.step(x, reward=1.0, target_action=1.0)
    assert net.current_step == 6


def test_two_compartment_weight_reset():
    net = TwoCompartmentSpatialNetwork(input_dim=4, hidden_dim=10, output_dim=1, seed=42)
    initial_w = net.W_rec.copy()

    # Mutate weights
    net.W_rec += 0.5
    assert not np.allclose(net.W_rec, initial_w)

    net.full_reset_weights()
    assert np.allclose(net.W_rec, initial_w)
    assert net.current_step == 0
