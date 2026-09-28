"""Unit tests for Dale's Principle, E/I Dynamics, Homeostatic Scaling, and E/I Substrates."""

import numpy as np
import pytest

from brain.ei_dynamics import (
    NeuronType,
    EILIFPopulation,
    EIIzhikevichPopulation,
    EISpikePropagationQueue,
    build_dale_connectivity_matrix,
)
from brain.homeostasis import HomeostaticSynapticScaling
from brain.ei_analysis import compute_ei_balance_metrics
from brain.ei_spatial import EIBalancedSpatialNetwork, EIBalancedLatentSpatialNetwork
from brain.spiking import SpikingNeuronModel
from brain.stdp import PlasticityMode
from brain.spatial import TopologyFamily
from observability.brainlog import BrainLogSerializer, BrainLogValidator
from observability.capture import StateCapture
from observability.topology import TopologyExtractor
from observability.replay import ReplayEngine


def test_dale_connectivity_sign_constraints():
    """Verify that Dale's principle strictly constrains excitatory and inhibitory columns."""
    N = 50
    W, types = build_dale_connectivity_matrix(
        num_neurons=N,
        inhibitory_fraction=0.20,
        density_ee=0.2,
        density_ei=0.3,
        density_ie=0.3,
        density_ii=0.2,
        seed=42,
    )
    is_inh = (types == NeuronType.INHIBITORY.value)
    is_exc = ~is_inh

    assert np.sum(is_inh) == 10
    assert np.sum(is_exc) == 40

    # Excitatory columns must be >= 0
    assert np.all(W[:, is_exc] >= 0.0)
    # Inhibitory columns must be <= 0
    assert np.all(W[:, is_inh] <= 0.0)
    # Diagonal must be 0 (no self-loops)
    assert np.all(np.diag(W) == 0.0)


def test_heterogeneous_ei_populations():
    """Verify distinct biophysical parameters for E and I populations."""
    N = 30
    lif_pop = EILIFPopulation(num_neurons=N, inhibitory_fraction=0.20, e_tau_m=20.0, i_tau_m=10.0)
    assert np.sum(lif_pop.is_inhibitory) == 6
    assert np.sum(lif_pop.is_excitatory) == 24
    assert np.all(lif_pop.tau_m[lif_pop.is_inhibitory] == 10.0)
    assert np.all(lif_pop.tau_m[lif_pop.is_excitatory] == 20.0)

    # Fast inhibitory interneurons refractory period is 1ms vs excitatory 3ms
    assert np.all(lif_pop.refractory_period[lif_pop.is_inhibitory] == 1)
    assert np.all(lif_pop.refractory_period[lif_pop.is_excitatory] == 3)

    izh_pop = EIIzhikevichPopulation(num_neurons=N, inhibitory_fraction=0.20)
    # Regular spiking (a=0.02, d=8.0) vs Fast spiking (a=0.10, d=2.0)
    assert np.all(izh_pop.a[izh_pop.is_inhibitory] == 0.10)
    assert np.all(izh_pop.a[izh_pop.is_excitatory] == 0.02)
    assert np.all(izh_pop.d[izh_pop.is_inhibitory] == 2.0)
    assert np.all(izh_pop.d[izh_pop.is_excitatory] == 8.0)


def test_ei_spike_propagation_queue_current_separation():
    """Verify that EISpikePropagationQueue delivers I_E >= 0 and I_I <= 0 accurately."""
    N = 4
    queue = EISpikePropagationQueue(num_neurons=N, max_delay=10)

    # Enqueue 1 excitatory spike to dst 1 and 1 inhibitory spike to dst 2
    src = np.array([0, 3], dtype=np.int32)
    dst = np.array([1, 2], dtype=np.int32)
    weights = np.array([1.5, 2.0], dtype=np.float32)
    delays = np.array([2, 2], dtype=np.int32)
    is_inh = np.array([False, True], dtype=bool)

    queue.enqueue_spikes(src, dst, weights, delays, is_inh)

    # Step 0
    ie, ii, inet = queue.advance_step()
    assert np.all(ie == 0) and np.all(ii == 0)

    # Step 1
    ie, ii, inet = queue.advance_step()
    assert np.all(ie == 0) and np.all(ii == 0)

    # Step 2: Spikes arrive!
    ie, ii, inet = queue.advance_step()
    assert ie[1] == pytest.approx(1.5)
    assert ii[2] == pytest.approx(-2.0)
    assert inet[1] == pytest.approx(1.5)
    assert inet[2] == pytest.approx(-2.0)


def test_homeostatic_synaptic_scaling():
    """Verify that homeostatic scaling maintains bounded weights and scales appropriately."""
    N = 10
    W = np.ones((N, N), dtype=np.float32) * 0.5
    np.fill_diagonal(W, 0.0)
    is_inh = np.zeros(N, dtype=bool)
    is_inh[-2:] = True
    W[:, is_inh] = -0.5

    scaling = HomeostaticSynapticScaling(
        num_neurons=N,
        target_rate_hz=10.0,
        tau_homeo_ms=50.0,
        eta_homeo=0.01,
        w_max_e=2.0,
        w_max_i=4.0,
        is_inhibitory=is_inh,
    )

    # Under-active neurons (0 spikes) should receive scaling up (mult > 1.0)
    spikes = np.zeros(N, dtype=bool)
    for s in range(25):
        W_scaled, telem = scaling.update(spikes, W, step=s)
    assert telem["mean_scale_factor"] > 1.0
    # Dale signs preserved
    assert np.all(W_scaled[:, ~is_inh] >= 0.0)
    assert np.all(W_scaled[:, is_inh] <= 0.0)


def test_ei_balanced_spatial_network_execution():
    """Verify end-to-end execution of EIBalancedSpatialNetwork."""
    net = EIBalancedSpatialNetwork(
        input_dim=4,
        hidden_dim=20,
        output_dim=2,
        neuron_model=SpikingNeuronModel.LIF,
        inhibitory_fraction=0.20,
        seed=42,
    )
    net.reset()
    assert net.num_inhibitory == 4
    assert net.num_excitatory == 16

    x = np.array([0.5, -0.2, 0.8, 0.0], dtype=np.float32)
    for t in range(20):
        y = net.step(x)
        assert y.shape == (2,)

    obs = net.observable_state()
    assert "i_e" in obs
    assert "i_i" in obs
    assert "i_net" in obs
    assert len(obs["neuron_types"]) == 20


def test_brainlog_v26_validation_and_replay():
    """Verify that BrainLog v2.6 traces validate and replay correctly."""
    net = EIBalancedSpatialNetwork(
        input_dim=4,
        hidden_dim=15,
        output_dim=2,
        seed=42,
    )
    net.reset()

    extractor = TopologyExtractor()
    topology = extractor.extract(net, "EI_Test_Network")
    capture = StateCapture(topology, dt_ms=1.0)

    for step in range(5):
        x = np.ones(4) * 0.5
        y = net.step(x)
        capture.capture_ei_step(
            step=step,
            input_vec=x,
            model=net,
            y_val=float(y[0]),
            target_val=0.0,
            mask_val=1.0,
            spikes=net.last_spikes,
            i_e=net.last_i_e,
            i_i=net.last_i_i,
        )

    blog_dict = {
        "brainlog_version": "2.6",
        "metadata": {"test": "v2.6"},
        "topology": topology.to_dict(),
        "frames": [f.to_dict() for f in capture.frames],
        "events": [e.to_dict() for e in capture.all_events],
        "snapshots": capture.snapshots,
    }

    valid, msg = BrainLogValidator.validate(blog_dict)
    assert valid, f"Validation failed: {msg}"

    replay = ReplayEngine(blog_dict)
    assert replay.total_steps == 5
    node_insp = replay.inspect_node(5)
    assert node_insp is not None
    assert hasattr(node_insp, "neuron_type")
