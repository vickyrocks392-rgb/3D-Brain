"""Tests for Phase 14 Temporal Coding Engine (T0–T8)."""

import numpy as np
import pytest

from brain.temporal_coding import TemporalSpikeDecoder, TemporalCodingScheme


def test_temporal_decoder_schemes():
    decoder = TemporalSpikeDecoder(num_neurons=16, default_window_steps=10, seed=42)
    # Generate synthetic raster [20 steps, 16 neurons]
    rng = np.random.default_rng(42)
    raster = (rng.random((20, 16)) < 0.2).astype(np.float32)

    schemes = [
        TemporalCodingScheme.T0_INSTANTANEOUS,
        TemporalCodingScheme.T1_SPIKE_COUNT,
        TemporalCodingScheme.T2_LATENCY,
        TemporalCodingScheme.T3_ISI,
        TemporalCodingScheme.T4_BURST_STRUCTURE,
        TemporalCodingScheme.T5_RELATIVE_TIMING,
        TemporalCodingScheme.T6_PHASE_CODE,
        TemporalCodingScheme.T7_POPULATION_SEQUENCE,
        TemporalCodingScheme.T8_COMBINED,
    ]

    for sch in schemes:
        feats = decoder.extract_features(
            spike_raster=raster,
            scheme=sch,
            window_steps=10,
            oscillation_phase=1.57,
        )
        assert isinstance(feats, np.ndarray)
        assert feats.ndim == 1
        assert len(feats) > 0


def test_jitter_and_dropout():
    decoder = TemporalSpikeDecoder(num_neurons=10, seed=42)
    raster = np.zeros((20, 10), dtype=np.float32)
    raster[10, :] = 1.0  # single volley at t=10

    # Jitter
    jittered = decoder.apply_jitter(raster, jitter_fraction=0.10)
    assert jittered.shape == raster.shape
    assert np.sum(jittered) == np.sum(raster)  # all spikes retained, shifted

    # Dropout
    dropped = decoder.apply_dropout(raster, dropout_fraction=0.50)
    assert dropped.shape == raster.shape
    assert np.sum(dropped) <= np.sum(raster)
