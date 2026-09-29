"""Tests for InformationMeasurementAuditor and common methodology comparisons."""

import numpy as np
import pytest

from brain.information_measurement_audit import (
    InformationMeasurementAuditor,
    CommonAuditRecord,
)


def test_information_measurement_auditor_shannon_mi():
    auditor = InformationMeasurementAuditor(seed=42)
    rng = np.random.default_rng(42)

    # Perfectly separable classes
    X_sep = np.concatenate([
        rng.normal(loc=-3.0, scale=0.5, size=(20, 4)),
        rng.normal(loc=3.0, scale=0.5, size=(20, 4)),
    ])
    y_sep = np.array([0] * 20 + [1] * 20)

    mi_sep = auditor.compute_gaussian_shannon_mi(X_sep, y_sep)
    assert mi_sep > 0.8  # Should be close to 1.0 bit

    # Completely random / independent
    X_rnd = rng.normal(loc=0.0, scale=1.0, size=(40, 4))
    y_rnd = np.array([0] * 20 + [1] * 20)
    mi_rnd = auditor.compute_gaussian_shannon_mi(X_rnd, y_rnd)
    assert mi_rnd < 0.2


def test_information_measurement_auditor_evaluate_stage():
    auditor = InformationMeasurementAuditor(seed=42)
    rng = np.random.default_rng(42)

    X = rng.normal(0.0, 1.0, size=(30, 8))
    ctx = np.array([0] * 15 + [1] * 15)
    act = np.array([0] * 15 + [1] * 15)

    rec = auditor.evaluate_stage(X, ctx, act, stage_idx=1, stage_name="1_CONTEXT_VECTOR")
    assert isinstance(rec, CommonAuditRecord)
    d = rec.to_dict()
    assert "legacy_mi_bsc_proxy_bits" in d
    assert "context_mi_shannon_bits" in d
    assert "action_relevant_mi_bits" in d
    assert "causal_action_shift" in d


def test_audit_full_pathway():
    auditor = InformationMeasurementAuditor(seed=42)
    trial_records = []
    rng = np.random.default_rng(42)

    for i in range(20):
        ctx = i % 2
        trial_records.append({
            "true_context_label": ctx,
            "target_action": ctx,
            "context_vector": np.full(10, float(ctx)),
            "v_dend": rng.normal(0.0, 1.0, size=10),
            "g_nmda": np.full(10, float(ctx) * 0.5),
            "calcium": rng.normal(0.0, 1.0, size=8),
            "v_soma": np.full(10, float(ctx) * 2.0),
            "threshold_gain": np.ones(10),
            "spikes": rng.integers(0, 2, size=8),
            "burst_representation": np.array([1.0, 0.0]) if ctx == 0 else np.array([0.0, 1.0]),
            "population_temporal": rng.normal(0.0, 1.0, size=16),
            "action_population": np.array([1.0, 0.0]) if ctx == 0 else np.array([0.0, 1.0]),
            "persistent_action": np.array([2.0, -1.0]) if ctx == 0 else np.array([-1.0, 2.0]),
        })

    summary = auditor.audit_full_pathway(trial_records)
    assert "_comparison" in summary
    assert "1_CONTEXT_VECTOR" in summary
    assert "11_PERSISTENT_ACTION" in summary
