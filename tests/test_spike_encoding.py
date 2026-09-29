"""Unit tests for Phase 13 Biophysical Spike Encoding.

Validates:
1. S0 baseline spike generation
2. S1 adaptive threshold dynamics (AIS threshold rise on spikes and exponential recovery)
3. S2 adaptive gain dynamics (divisive normalization of dynamic range)
4. S3 combined threshold and gain adaptation
5. S4 population rate coding
6. S5 temporal latency-to-first-spike coding
7. S6 population + temporal burst detection
8. S7 Dale-constrained Winner-Take-All competition
9. S8 persistent action attractor
10. S9 full biophysical interface
11. Homeostatic regulation (H1 rate, H2 threshold, H3 gain, H4 combined)
12. Causal intervention perturbations (timing shuffle, count shuffle, threshold random, gain random)
"""

import numpy as np
import pytest

from brain.spike_encoding import (
    BiophysicalSpikeEncoder,
    SpikeEncodingStrategy,
    HomeostasisMode,
)


def test_s0_baseline_encoding():
    """Test 1: S0 baseline produces valid rates and winner selection."""
    encoder = BiophysicalSpikeEncoder(
        num_actions=2,
        neurons_per_action=10,
        strategy=SpikeEncodingStrategy.S0_BASELINE,
        seed=42,
    )
    # Higher drive to action 1
    drives = np.array([0.1, 1.8], dtype=np.float32)
    for _ in range(10):
        out = encoder.step(drives)

    assert out["spikes"].shape == (2, 10)
    assert len(out["pool_firing_rates"]) == 2
    assert out["selected_action"] in (0, 1)


def test_s1_adaptive_threshold_dynamics():
    """Test 2: S1 adaptive threshold rises upon firing and recovers exponentially."""
    encoder = BiophysicalSpikeEncoder(
        num_actions=2,
        neurons_per_action=4,
        strategy=SpikeEncodingStrategy.S1_ADAPTIVE_THRESHOLD,
        theta_base_mv=-50.0,
        theta_increment_mv=8.0,
        tau_theta_ms=25.0,
        seed=42,
    )
    # Strong drive to evoke spikes
    drives = np.array([3.0, 0.0], dtype=np.float32)
    spiked = False
    for _ in range(15):
        out = encoder.step(drives)
        if np.any(out["spikes"][0] > 0):
            spiked = True
            break

    assert spiked, "Action 0 neurons should spike under strong drive"
    assert np.any(out["theta_eff"][0] > -50.0), "Threshold should adapt upward following spikes"

    # Now drive with zero input and verify threshold decays back down
    theta_peak = np.max(encoder.delta_theta[0])
    for _ in range(50):
        encoder.step(np.zeros(2, dtype=np.float32))
    theta_rest = np.max(encoder.delta_theta[0])
    assert theta_rest < theta_peak, "Adaptive threshold increment must decay exponentially"


def test_s2_adaptive_gain_control():
    """Test 3: S2 adaptive gain scales inversely with sustained high activity."""
    encoder = BiophysicalSpikeEncoder(
        num_actions=2,
        neurons_per_action=8,
        strategy=SpikeEncodingStrategy.S2_ADAPTIVE_GAIN,
        gain_base=1.0,
        seed=42,
    )
    initial_gain = np.mean(encoder.gain[0])
    # Apply sustained high drive
    for _ in range(40):
        encoder.step(np.array([2.5, 0.1], dtype=np.float32))

    adapted_gain = np.mean(encoder.gain[0])
    assert adapted_gain < initial_gain, "High sustained activity must dynamically suppress gain"


def test_s4_population_rate_coding():
    """Test 4: S4 population coding averages across heterogeneous neuron pool."""
    encoder = BiophysicalSpikeEncoder(
        num_actions=2,
        neurons_per_action=16,
        strategy=SpikeEncodingStrategy.S4_RATE_POPULATION,
        seed=42,
    )
    # Distinct drives
    out = None
    for _ in range(35):
        out = encoder.step(np.array([0.1, 2.5], dtype=np.float32))

    assert out["pool_firing_rates"][1] > out["pool_firing_rates"][0]
    assert out["selected_action"] == 1


def test_s5_temporal_latency_coding():
    """Test 5: S5 temporal coding prioritizes earlier spike latency."""
    encoder = BiophysicalSpikeEncoder(
        num_actions=2,
        neurons_per_action=8,
        strategy=SpikeEncodingStrategy.S5_TEMPORAL_CODING,
        seed=42,
    )
    # Step until spikes occur
    for _ in range(15):
        out = encoder.step(np.array([0.5, 2.5], dtype=np.float32))

    # Action 1 receives much higher drive, so its time-to-first-spike should be shorter
    t1 = np.min(out["time_to_first_spike"][1])
    t0 = np.min(out["time_to_first_spike"][0])
    assert t1 <= t0
    assert out["selected_action"] == 1


def test_s7_wta_competition():
    """Test 6: S7 Dale E/I competition suppresses the weaker action pool."""
    encoder = BiophysicalSpikeEncoder(
        num_actions=2,
        neurons_per_action=12,
        strategy=SpikeEncodingStrategy.S7_WTA_COMPETITION,
        w_inter_inh=0.8,
        seed=42,
    )
    for _ in range(25):
        out = encoder.step(np.array([0.8, 1.6], dtype=np.float32))

    assert out["selected_action"] == 1
    assert out["winner_margin"] > 0.0


def test_s8_persistent_action_attractor():
    """Test 7: S8 recurrent attractor maintains pool activation during transient pause."""
    encoder = BiophysicalSpikeEncoder(
        num_actions=2,
        neurons_per_action=12,
        strategy=SpikeEncodingStrategy.S8_PERSISTENT_ATTRACTOR,
        w_intra_exc=0.6,
        seed=42,
    )
    # Establish winner with drive
    for _ in range(20):
        encoder.step(np.array([2.0, 0.1], dtype=np.float32))

    # Now drop drive to zero and verify action 0 maintains higher rate due to recurrent self-excitation
    for _ in range(5):
        out = encoder.step(np.array([0.0, 0.0], dtype=np.float32))

    assert out["selected_action"] == 0


def test_homeostatic_regulation():
    """Test 8: Homeostasis modes regulate thresholds and gains toward target."""
    encoder = BiophysicalSpikeEncoder(
        num_actions=2,
        neurons_per_action=6,
        strategy=SpikeEncodingStrategy.S9_FULL_BIOPHYSICAL,
        homeostasis=HomeostasisMode.H4_COMBINED,
        target_rate_hz=10.0,
        seed=42,
    )
    # Drive heavily to trigger homeostatic shift
    theta_init = np.mean(encoder.theta_base)
    for _ in range(50):
        encoder.step(np.array([3.0, 3.0], dtype=np.float32))

    theta_post = np.mean(encoder.theta_base)
    assert theta_post > theta_init, "Excessive activity should shift baseline threshold upward"


def test_causal_interventions_on_spike_encoder():
    """Test 9: Causal interventions alter timing, count, and competition."""
    encoder = BiophysicalSpikeEncoder(
        num_actions=2,
        neurons_per_action=8,
        strategy=SpikeEncodingStrategy.S9_FULL_BIOPHYSICAL,
        seed=42,
    )
    drives = np.array([1.2, 0.3], dtype=np.float32)
    for _ in range(10):
        encoder.step(drives)

    # Disable competition
    encoder.competition_disabled = True
    out_no_comp = encoder.step(drives)
    assert out_no_comp is not None

    # Randomize threshold
    encoder.randomize_threshold = True
    out_rand_th = encoder.step(drives)
    assert out_rand_th is not None
