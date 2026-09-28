"""Unit tests for TwoCompartmentPopulation (Phase 10)."""

import numpy as np
import pytest

from brain.two_compartment import TwoCompartmentPopulation
from brain.ei_dynamics import NeuronType


def test_two_compartment_initialization():
    pop = TwoCompartmentPopulation(num_neurons=50, inhibitory_fraction=0.20, seed=42)
    assert pop.num_neurons == 50
    assert pop.num_inhibitory == 10
    assert pop.num_excitatory == 40
    assert pop.v_soma.shape == (50,)
    assert pop.v_dend.shape == (50,)
    assert pop.g_nmda.shape == (50,)
    assert np.all(pop.v_soma == -70.0)
    assert np.all(pop.v_dend == -70.0)
    assert np.all(pop.g_nmda == 0.0)
    assert np.all(pop.spikes == False)


def test_slow_nmda_conductance_integration_and_decay():
    pop = TwoCompartmentPopulation(num_neurons=10, tau_nmda=50.0, alpha_nmda=0.1, seed=42)
    i_e = np.full(10, 20.0, dtype=np.float32)
    i_i = np.zeros(10, dtype=np.float32)

    # 1. Drive with excitatory current
    pop.step(i_e, i_i, dt=1.0)
    assert np.all(pop.g_nmda > 0.0)
    initial_g = pop.g_nmda.copy()

    # 2. Remove input: conductance must decay with tau_NMDA
    zero_e = np.zeros(10, dtype=np.float32)
    pop.step(zero_e, i_i, dt=1.0)
    expected_decay = float(np.exp(-1.0 / 50.0))
    assert np.allclose(pop.g_nmda, initial_g * expected_decay, atol=1e-4)


def test_inter_compartment_coupling():
    pop = TwoCompartmentPopulation(num_neurons=5, g_coupling=0.8, seed=42)
    # Set dendrite depolarized relative to soma
    pop.v_dend.fill(-55.0)
    pop.v_soma.fill(-70.0)

    i_e = np.zeros(5, dtype=np.float32)
    i_i = np.zeros(5, dtype=np.float32)
    pop.step(i_e, i_i, dt=1.0)

    # Coupling current I_c = 0.8 * (-55 - (-70)) = 0.8 * 15 = 12 nA
    assert np.all(pop.last_i_coupling > 0.0)
    # Soma must have depolarized due to positive coupling
    assert np.all(pop.v_soma > -70.0)


def test_nonlinear_dendritic_plateau_trigger():
    pop = TwoCompartmentPopulation(
        num_neurons=10,
        nonlinear_dendrite=True,
        theta_dend=-45.0,
        g_nl=0.5,
        plateau_duration_ms=4.0,
        seed=42,
    )
    # Set dendrite above plateau threshold
    pop.v_dend.fill(-40.0)
    i_e = np.zeros(10, dtype=np.float32)
    i_i = np.zeros(10, dtype=np.float32)

    pop.step(i_e, i_i, dt=1.0)
    # Plateau should be active on excitatory neurons
    assert np.any(pop.plateau_remaining > 0)
    assert np.any(pop.last_i_plateau > 0.0)


def test_somatic_spike_threshold_reset_and_refractory():
    pop = TwoCompartmentPopulation(num_neurons=5, e_v_thresh=-50.0, v_reset=-75.0, seed=42)
    # Strongly drive soma to threshold with somatic external current
    i_e = np.zeros(5, dtype=np.float32)
    i_i = np.zeros(5, dtype=np.float32)
    i_soma = np.full(5, 50.0, dtype=np.float32)

    spikes = pop.step(i_e, i_i, i_soma_ext=i_soma, dt=1.0)
    assert np.all(spikes == True)
    assert np.all(pop.v_soma == -75.0)
    assert np.all(pop.refractory_remaining > 0)

    # Next step: neuron must remain in refractory reset even with drive
    spikes2 = pop.step(i_e, i_i, i_soma_ext=i_soma, dt=1.0)
    assert np.all(spikes2 == False)
    assert np.all(pop.v_soma == -75.0)


def test_backpropagating_action_potential_bap():
    pop = TwoCompartmentPopulation(num_neurons=5, bap_amplitude=15.0, seed=42)
    initial_vdend = pop.v_dend.copy()

    # Trigger somatic spike via somatic external drive
    i_e = np.zeros(5, dtype=np.float32)
    i_i = np.zeros(5, dtype=np.float32)
    i_soma = np.full(5, 50.0, dtype=np.float32)

    spikes = pop.step(i_e, i_i, i_soma_ext=i_soma, dt=1.0)
    assert np.all(spikes == True)
    # Dendrite must receive bAP depolarizing boost
    assert np.all(pop.v_dend > initial_vdend)


def test_two_compartment_reset():
    pop = TwoCompartmentPopulation(num_neurons=10, seed=42)
    pop.v_soma.fill(-45.0)
    pop.v_dend.fill(-30.0)
    pop.g_nmda.fill(2.5)
    pop.plateau_remaining.fill(3)

    pop.reset()
    assert np.all(pop.v_soma == -70.0)
    assert np.all(pop.v_dend == -70.0)
    assert np.all(pop.g_nmda == 0.0)
    assert np.all(pop.plateau_remaining == 0)
    assert np.all(pop.spikes == False)
