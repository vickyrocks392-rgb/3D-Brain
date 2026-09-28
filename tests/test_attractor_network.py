"""Unit tests for RecurrentAttractorCircuit and Attractor Diagnostics (Phase 11)."""

import numpy as np
import pytest

from brain.attractor_network import AttractorCouplingMode, RecurrentAttractorCircuit
from brain.ei_dynamics import NeuronType


def test_attractor_circuit_dale_conformance():
    circuit = RecurrentAttractorCircuit(num_neurons=50, inhibitory_fraction=0.20, seed=42)
    assert circuit.num_neurons == 50
    assert circuit.num_inhibitory == 10
    assert circuit.num_excitatory == 40

    # Test Dale's sign constraint on recurrent weights (column pre contains outgoing weights)
    for i in range(50):
        if circuit.neuron_types[i] == NeuronType.EXCITATORY.value:
            # Excitatory pre-synaptic neurons have weights >= 0
            assert np.all(circuit.W_rec[:, i] >= 0.0), f"Excitatory neuron {i} has negative outgoing weights"
        else:
            # Inhibitory pre-synaptic neurons have weights <= 0
            assert np.all(circuit.W_rec[:, i] <= 0.0), f"Inhibitory neuron {i} has positive outgoing weights"

    # Diagonal must be 0 (no autapses)
    assert np.all(np.diag(circuit.W_rec) == 0.0)


def test_attractor_recurrent_currents():
    circuit = RecurrentAttractorCircuit(num_neurons=20, inhibitory_fraction=0.20, seed=42)
    spikes = np.ones(20, dtype=bool)

    i_e, i_i = circuit.compute_recurrent_currents(spikes)
    assert i_e.shape == (20,)
    assert i_i.shape == (20,)
    assert np.all(i_e >= 0.0)
    assert np.all(i_i <= 0.0)
    assert np.any(i_e > 0.0)
    assert np.any(i_i < 0.0)


def test_attractor_coupling_mode_ablations():
    # 1. NO_EXCITATION: excitatory recurrent weights clamped to 0
    c_no_e = RecurrentAttractorCircuit(num_neurons=20, coupling_mode=AttractorCouplingMode.NO_EXCITATION, seed=42)
    exc_idx = np.where(c_no_e.excitatory_mask)[0]
    assert np.all(c_no_e.W_rec[:, exc_idx] == 0.0)

    # 2. NO_INHIBITION: inhibitory cross-pool weights clamped to 0
    c_no_i = RecurrentAttractorCircuit(num_neurons=20, coupling_mode=AttractorCouplingMode.NO_INHIBITION, seed=42)
    inh_idx = np.where(c_no_i.inhibitory_mask)[0]
    assert np.all(c_no_i.W_rec[:, inh_idx] == 0.0)


def test_attractor_diagnostics_suite():
    circuit = RecurrentAttractorCircuit(num_neurons=20, seed=42)
    # Synthetic trajectory A (Pool 0 active) and trajectory B (Pool 1 active)
    T = 20
    traj_a = np.zeros((T, 20), dtype=np.float32)
    traj_b = np.zeros((T, 20), dtype=np.float32)

    pool_0_idx = np.where(circuit.pool_ids == 0)[0]
    pool_1_idx = np.where(circuit.pool_ids == 1)[0]

    traj_a[:, pool_0_idx] = 1.0 + 0.05 * np.random.randn(T, len(pool_0_idx))
    traj_b[:, pool_1_idx] = 1.0 + 0.05 * np.random.randn(T, len(pool_1_idx))

    diag = circuit.evaluate_attractor_diagnostics(traj_a, traj_b)
    assert "mean_state_drift" in diag
    assert "attractor_separation_ratio" in diag
    assert "spontaneous_switches" in diag
    assert "effective_dimensionality" in diag
    assert diag["attractor_separation_ratio"] > 1.0
    assert diag["spontaneous_switches"] == 0
