"""Unit and integration tests for Phase 7 Spiking Neural Dynamics and Delayed Queues."""

import pytest
import numpy as np

from brain.spiking import (
    SpikeEvent,
    SpikingNeuronModel,
    LIFNeuronPopulation,
    IzhikevichNeuronPopulation,
    SpikePropagationQueue,
)
from brain.spiking_spatial import SpikingSpatialNetwork, SpikingLatentSpatialNetwork
from brain.spatial import TopologyFamily
from brain.stdp import PlasticityMode


class TestLIFNeuronPopulation:
    def test_subthreshold_integration_and_leak(self):
        pop = LIFNeuronPopulation(num_neurons=5, v_rest=-70.0, v_thresh=-50.0, tau_m=20.0, noise_std=0.0)
        # Moderate input current below threshold
        i_sub = np.full(5, 5.0, dtype=np.float32)
        spikes = pop.step(i_sub, dt=1.0, current_step=0)

        assert not np.any(spikes), "Subthreshold current should not trigger action potentials"
        assert np.all(pop.v > -70.0), "Membrane potential should depolarize from rest"
        assert np.all(pop.v < -50.0), "Membrane potential should remain subthreshold"

    def test_threshold_crossing_and_reset(self):
        pop = LIFNeuronPopulation(num_neurons=3, v_rest=-70.0, v_reset=-75.0, v_thresh=-50.0, tau_m=10.0, noise_std=0.0)
        # Strong suprathreshold current to trigger immediate spike
        i_strong = np.full(3, 500.0, dtype=np.float32)
        spikes = pop.step(i_strong, dt=1.0, current_step=0)

        assert np.all(spikes), "Suprathreshold current must trigger action potentials"
        assert np.all(pop.v == -75.0), "Membrane potential must reset to v_reset after spike"
        assert np.all(pop.refractory_remaining == 3), "Neuron must enter refractory state"
        assert np.all(pop.spike_counts == 1), "Spike counter must increment"

    def test_refractory_clamping(self):
        pop = LIFNeuronPopulation(num_neurons=2, v_rest=-70.0, v_reset=-75.0, v_thresh=-50.0, refractory_period=2, noise_std=0.0)
        i_strong = np.full(2, 500.0, dtype=np.float32)
        # Step 0: Spike emitted
        s0 = pop.step(i_strong, dt=1.0, current_step=0)
        assert np.all(s0)

        # Step 1: In refractory period -> no spike despite high input
        s1 = pop.step(i_strong, dt=1.0, current_step=1)
        assert not np.any(s1), "Cannot fire during absolute refractory period"
        assert np.all(pop.v == -75.0), "Voltage must remain clamped at reset during refractory"


class TestIzhikevichNeuronPopulation:
    def test_regular_spiking_dynamics(self):
        # Regular Spiking parameters: a=0.02, b=0.2, c=-65, d=8
        pop = IzhikevichNeuronPopulation(num_neurons=4, a=0.02, b=0.2, c=-65.0, d=8.0)
        pop.reset()

        # Step with tonic current I = 10.0
        i_tonic = np.full(4, 10.0, dtype=np.float32)
        total_spikes = 0
        for step in range(100):
            spikes = pop.step(i_tonic, dt=1.0, current_step=step)
            total_spikes += int(np.sum(spikes))

        assert total_spikes > 0, "Izhikevich RS population should fire periodic spikes under tonic drive"
        state = pop.get_state_dict()
        assert state["model"] == "IZHIKEVICH"
        assert "u_mean" in state


class TestSpikePropagationQueue:
    def test_delayed_arrival_and_circular_buffer(self):
        queue = SpikePropagationQueue(num_neurons=10, max_delay=10)
        # Neuron 1 connects to Neuron 5 with delay 3 and weight 0.8
        src = np.array([1], dtype=np.int32)
        dst = np.array([5], dtype=np.int32)
        w = np.array([0.8], dtype=np.float32)
        delays = np.array([3], dtype=np.int32)

        events = queue.enqueue_spikes(src, dst, w, delays)
        assert len(events) == 1
        assert events[0].arrival_step == 3
        assert events[0].delay == 3

        # Step 0
        arrived_0 = queue.advance_step()
        assert arrived_0[5] == 0.0

        # Step 1
        arrived_1 = queue.advance_step()
        assert arrived_1[5] == 0.0

        # Step 2
        arrived_2 = queue.advance_step()
        assert arrived_2[5] == 0.0

        # Step 3: Arrival time!
        arrived_3 = queue.advance_step()
        assert np.isclose(arrived_3[5], 0.8), f"Expected weight 0.8 delivered at arrival step, got {arrived_3[5]}"


class TestSpikingSpatialNetworks:
    def test_spiking_spatial_network_execution(self):
        net = SpikingSpatialNetwork(
            input_dim=4,
            hidden_dim=20,
            output_dim=2,
            neuron_model=SpikingNeuronModel.LIF,
            topology_type=TopologyFamily.MODULAR,
            plasticity_mode=PlasticityMode.STDP,
            seed=42,
        )
        net.reset()
        x = np.array([0.5, 0.2, 0.9, 0.1], dtype=np.float32)
        y = net.step(x)

        assert y.shape == (2,)
        obs = net.observable_state()
        assert "spikes" in obs
        assert "population_summary" in obs
        assert obs["population_summary"]["model"] == "LIF"

    def test_spiking_latent_spatial_network_coupling(self):
        hybrid = SpikingLatentSpatialNetwork(
            input_dim=4,
            hidden_dim=24,
            latent_dim=12,
            output_dim=1,
            seed=42,
        )
        hybrid.reset()
        x = np.array([0.8, 0.1, 0.3, 0.6], dtype=np.float32)
        y = hybrid.step(x, action=1)

        assert y.shape == (1,)
        obs = hybrid.observable_state()
        assert "latent_z" in obs
        assert len(obs["latent_z"]) == 12
        assert "multistep_predictions" in obs

    def test_spiking_brainlog_v2_5_and_replay(self, tmp_path):
        from observability.topology import TopologyExtractor
        from observability.capture import StateCapture
        from observability.brainlog import BrainLogSerializer, BrainLogValidator
        from observability.replay import ReplayEngine

        net = SpikingSpatialNetwork(
            input_dim=2,
            hidden_dim=10,
            output_dim=1,
            seed=42,
        )
        topo = TopologyExtractor().extract(net, "SpikingSpatialNetwork")
        capture = StateCapture(topology=topo, dt_ms=1.0)

        for step in range(15):
            x = np.array([0.5, 0.8], dtype=np.float32)
            y = net.step(x)
            capture.capture_spiking_step(
                step=step,
                input_vec=x,
                model=net,
                y_val=float(y[0]),
                target_val=1.0,
                mask_val=1.0,
                kuramoto_r=0.6 if step > 5 else 0.2,
            )

        log_path = tmp_path / "spiking_test.brainlog"
        BrainLogSerializer.save(
            filepath=log_path,
            metadata={"phase": "Phase 7", "substrate": "SpikingSpatialNetwork"},
            topology=topo,
            frames=capture.frames,
            events=capture.all_events,
            snapshots=capture.snapshots,
            version="2.5",
        )

        valid, msg = BrainLogValidator.validate(log_path)
        assert valid, f"Validation failed: {msg}"

        replay = ReplayEngine()
        replay.load(log_path)
        assert replay.total_steps == 15

        spk_events = replay.get_spike_events(0)
        mem = replay.get_membrane_states(0)
        assert "v_mem_mean" in mem
        phase = replay.get_phase_metrics(10)
        assert phase["kuramoto_r"] == 0.6

