"""Unit tests for Phase 7 Spiking Analysis, Modularity, and Synchronization Metrics."""

import pytest
import numpy as np

from brain.spiking_analysis import (
    compute_spiking_statistics,
    compute_spike_correlation_matrix,
    detect_functional_communities_louvain,
    compute_functional_modularity_from_spikes,
    compute_oscillatory_spectrum,
    compute_phase_locking_and_kuramoto,
    compute_spatial_participation_and_waves,
)


class TestSpikingAnalysis:
    def test_spiking_statistics(self):
        # 100 timesteps, 10 neurons, periodic firing
        spikes = np.zeros((100, 10), dtype=bool)
        spikes[::10, :] = True  # Every 10 steps

        stats = compute_spiking_statistics(spikes, dt_ms=1.0)
        assert stats["total_spikes"] == 100
        assert stats["active_neuron_fraction"] == 1.0
        assert stats["mean_firing_rate_hz"] > 0.0

    def test_functional_modularity_on_correlated_clusters(self):
        # Create 2 synthetic disjoint assemblies that fire synchronously within assembly
        # Group 1: units 0-4; Group 2: units 5-9
        spikes = np.zeros((200, 10), dtype=bool)
        for t in range(0, 200, 8):
            spikes[t, 0:5] = True
        for t in range(4, 200, 8):
            spikes[t, 5:10] = True

        struct_modules = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        res = compute_functional_modularity_from_spikes(spikes, structural_modules=struct_modules, bin_size_steps=4)

        assert res["q_func"] > 0.30, f"Expected high Q_func on clearly segregated clusters, got {res['q_func']}"
        assert res["num_functional_modules"] == 2
        # Ensure within-cluster correlation exceeds cross-cluster
        assert res["mean_within_func_corr"] > res["mean_cross_func_corr"]

    def test_oscillatory_spectrum(self):
        # Periodic 50 Hz population oscillation (period = 20 ms)
        spikes = np.zeros((500, 20), dtype=bool)
        for t in range(0, 500, 20):
            spikes[t, :] = True

        spec = compute_oscillatory_spectrum(spikes, dt_ms=1.0)
        assert abs(spec["dominant_frequency_hz"] - 50.0) <= 5.0, (
            f"Expected ~50 Hz dominant peak, got {spec['dominant_frequency_hz']}"
        )

    def test_kuramoto_phase_synchronization(self):
        # Synchronized firing -> high Kuramoto R
        spikes_sync = np.zeros((100, 10), dtype=bool)
        for t in range(0, 100, 10):
            spikes_sync[t, :] = True

        mean_r, r_t, plv = compute_phase_locking_and_kuramoto(spikes_sync, dt_ms=1.0)
        assert mean_r > 0.5, f"Expected high Kuramoto R for synchronized bursts, got {mean_r}"
        assert plv > 0.5, f"Expected high PLV for synchronized bursts, got {plv}"
