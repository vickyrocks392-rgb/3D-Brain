"""Unit tests for Spike-Timing-Dependent Plasticity (STDP) and Homeostasis."""

import pytest
import numpy as np

from brain.stdp import PlasticityMode, STDPEngine


class TestSTDPEngine:
    def test_pre_before_post_potentiation(self):
        engine = STDPEngine(num_neurons=2, mode=PlasticityMode.STDP, a_plus=0.1, a_minus=0.1, w_max=2.0)
        adj = np.array([[False, True], [False, False]])
        weights = np.array([[0.0, 0.5], [0.0, 0.0]], dtype=np.float32)

        # Step 0: Pre-synaptic neuron 0 fires
        spikes_0 = np.array([True, False])
        w_after_0, recs_0 = engine.step_plasticity(weights, adj, spikes_0, step=0)
        # Pre trace x_pre[0] incremented to 1.0

        # Step 1: Post-synaptic neuron 1 fires (pre before post -> LTP!)
        spikes_1 = np.array([False, True])
        w_after_1, recs_1 = engine.step_plasticity(w_after_0, adj, spikes_1, step=1)

        assert w_after_1[0, 1] > weights[0, 1], f"Expected LTP potentiation: {w_after_1[0, 1]} vs {weights[0, 1]}"
        assert any(r.rule == "LTP" for r in recs_1), "Must log LTP update record"

    def test_post_before_pre_depression(self):
        engine = STDPEngine(num_neurons=2, mode=PlasticityMode.STDP, a_plus=0.1, a_minus=0.1, w_min=0.0)
        adj = np.array([[False, True], [False, False]])
        weights = np.array([[0.0, 0.5], [0.0, 0.0]], dtype=np.float32)

        # Step 0: Post-synaptic neuron 1 fires
        spikes_0 = np.array([False, True])
        w_after_0, recs_0 = engine.step_plasticity(weights, adj, spikes_0, step=0)
        # Post trace y_post[1] incremented to 1.0

        # Step 1: Pre-synaptic neuron 0 fires (post before pre -> LTD!)
        spikes_1 = np.array([True, False])
        w_after_1, recs_1 = engine.step_plasticity(w_after_0, adj, spikes_1, step=1)

        assert w_after_1[0, 1] < weights[0, 1], f"Expected LTD depression: {w_after_1[0, 1]} vs {weights[0, 1]}"
        assert any(r.rule == "LTD" for r in recs_1), "Must log LTD update record"

    def test_frozen_plasticity_mode(self):
        engine = STDPEngine(num_neurons=2, mode=PlasticityMode.FROZEN)
        adj = np.array([[False, True], [False, False]])
        weights = np.array([[0.0, 0.5], [0.0, 0.0]], dtype=np.float32)

        spikes = np.array([True, True])
        w_new, recs = engine.step_plasticity(weights, adj, spikes, step=0)
        assert np.array_equal(w_new, weights), "Frozen mode must preserve exact weights"
        assert len(recs) == 0

    def test_homeostatic_scaling(self):
        # Target rate 0.1, low firing -> homeostatic drive scales weights up
        engine = STDPEngine(
            num_neurons=2,
            mode=PlasticityMode.STDP_HOMEOSTASIS,
            target_rate=0.5,
            eta_homeo=0.05,
            tau_homeo=10.0,
        )
        adj = np.array([[False, True], [False, False]])
        weights = np.array([[0.0, 0.4], [0.0, 0.0]], dtype=np.float32)

        # Inactive steps
        spikes_silent = np.array([False, False])
        w_curr = weights.copy()
        for s in range(5):
            w_curr, _ = engine.step_plasticity(w_curr, adj, spikes_silent, step=s)

        assert w_curr[0, 1] > weights[0, 1], "Homeostasis must scale up weights when firing below target"
