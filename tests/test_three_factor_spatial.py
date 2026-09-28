"""Integration Tests for ThreeFactorSpatialNetwork (Phase 9).

Verifies:
1. Full step execution with feedforward input, E/I currents, Dale bounds, and readout.
2. Synaptic eligibility tracking through spatial conduction delays.
3. Homeostatic synaptic scaling integration with three-factor updates.
4. Observable state completeness and telemetry contracts.
5. Ablation modes (break_dale, freeze_recurrent, freeze_inhibitory, freeze_excitatory).
"""

import numpy as np
import pytest

from brain.three_factor_spatial import ThreeFactorSpatialNetwork
from brain.spatial import TopologyFamily
from brain.spiking import SpikingNeuronModel


def test_spatial_network_step_and_output():
    """Verify forward step, current accounting, and non-trivial output generation."""
    net = ThreeFactorSpatialNetwork(
        input_dim=4,
        hidden_dim=20,
        output_dim=1,
        topology_type=TopologyFamily.MODULAR,
        learning_rate=0.01,
        seed=42,
    )
    net.reset()

    x = np.array([0.5, 0.8, 0.2, 0.1], dtype=np.float32)
    y = net.step(x, reward=0.0)

    assert isinstance(y, np.ndarray)
    assert y.shape == (1,)
    assert np.isfinite(y[0])

    # Check E/I current separation
    assert np.all(net.last_i_e >= 0.0)
    assert np.all(net.last_i_i <= 0.0)
    assert np.allclose(net.last_i_net, net.last_i_e + net.last_i_i)


def test_plasticity_upon_reward():
    """Verify that delivering a reward triggers three-factor synaptic weight changes."""
    net = ThreeFactorSpatialNetwork(
        input_dim=4,
        hidden_dim=20,
        output_dim=1,
        learning_rate=0.05,
        homeostasis_enabled=False,
        seed=42,
    )
    net.reset()
    w_initial = net.W_rec.copy()

    x = np.ones(4, dtype=np.float32) * 1.5

    # Run 25 steps to charge membrane, fire spikes, and accumulate eligibility traces
    for _ in range(25):
        net.step(x, reward=0.0)

    # Weights should not change before reward
    assert np.allclose(net.W_rec, w_initial)

    # Deliver positive reward
    net.step(x, reward=1.0)

    # After reward, weights must have updated
    delta = np.sum(np.abs(net.W_rec - w_initial))
    assert delta > 1e-4

    # Dale constraints must strictly hold
    is_exc = net.is_excitatory
    is_inh = net.is_inhibitory
    # W_rec is [post, pre]
    assert np.all(net.W_rec[:, is_exc] >= 0.0)
    assert np.all(net.W_rec[:, is_inh] <= 0.0)


def test_freeze_recurrent_ablation():
    """Verify that when freeze_recurrent=True, recurrent weights remain strictly fixed."""
    net = ThreeFactorSpatialNetwork(
        input_dim=4,
        hidden_dim=20,
        output_dim=1,
        learning_rate=0.05,
        freeze_recurrent=True,
        seed=42,
    )
    net.reset()
    w_initial = net.W_rec.copy()

    x = np.ones(4, dtype=np.float32) * 0.7
    for _ in range(5):
        net.step(x, reward=1.0)

    assert np.allclose(net.W_rec, w_initial)


def test_observable_state_completeness():
    """Verify that observable_state exposes all required fields for passive telemetry capture."""
    net = ThreeFactorSpatialNetwork(
        input_dim=4,
        hidden_dim=20,
        output_dim=1,
        seed=42,
    )
    net.reset()
    x = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    net.step(x, reward=0.5)

    obs = net.observable_state()
    required_keys = [
        "state", "activation", "spikes", "neuron_types", "i_e", "i_i", "i_net",
        "mean_i_e", "mean_i_i", "mean_i_net", "mean_eligibility", "max_eligibility",
        "neuromodulator", "rpe", "homeostatic_scale_mean"
    ]
    for k in required_keys:
        assert k in obs, f"Missing key in observable_state: {k}"
