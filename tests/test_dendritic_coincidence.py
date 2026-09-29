"""Tests for Phase 14 Dendritic Calcium Coincidence Detector."""

import numpy as np
import pytest

from brain.dendritic_coincidence import DendriticCoincidenceDetector, CoincidenceTriggerType


def test_coincidence_detector_initialization():
    detector = DendriticCoincidenceDetector(num_compartments=20, delta_t_ms=10.0)
    assert detector.num_compartments == 20
    assert detector.delta_t_ms == 10.0
    assert len(detector.input_history) == 20


def test_coincidence_detector_temporal_sweep():
    delta_ts = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    for dt in delta_ts:
        detector = DendriticCoincidenceDetector(num_compartments=10, delta_t_ms=dt, seed=42)
        detector.reset()

        # Step with active inputs
        out = detector.step(
            synaptic_inputs=np.full(10, 0.5, dtype=np.float32),
            v_dend=np.full(10, -45.0, dtype=np.float32),
            v_soma=np.full(10, -50.0, dtype=np.float32),
            g_nmda=np.full(10, 0.3, dtype=np.float32),
            calcium=np.full(10, 0.4, dtype=np.float32),
        )
        assert out["effective_drive"].shape == (10,)
        assert "coincidence_detected" in out
        assert "plateau_active" in out


def test_larkum_bap_coincidence():
    detector = DendriticCoincidenceDetector(num_compartments=10, delta_t_ms=10.0, seed=42)
    # Step 1: Soma emits bAP
    detector.step(
        synaptic_inputs=np.zeros(10, dtype=np.float32),
        v_dend=np.full(10, -60.0, dtype=np.float32),
        v_soma=np.full(10, 20.0, dtype=np.float32),
        g_nmda=np.zeros(10, dtype=np.float32),
        calcium=np.zeros(10, dtype=np.float32),
        somatic_spikes=np.ones(10, dtype=np.float32),
    )
    assert len(detector.bap_history) > 0

    # Step 2: Coincident apical input within delta_t
    out = detector.step(
        synaptic_inputs=np.full(10, 0.6, dtype=np.float32),
        v_dend=np.full(10, -42.0, dtype=np.float32),
        v_soma=np.full(10, -60.0, dtype=np.float32),
        g_nmda=np.full(10, 0.35, dtype=np.float32),
        calcium=np.full(10, 0.45, dtype=np.float32),
    )
    assert np.any(out["coincidence_detected"])
    assert np.any(out["plateau_active"])
