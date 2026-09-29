"""Tests for Phase 14 11-Stage Information Bottleneck Auditor."""

import numpy as np
import pytest

from brain.information_bottleneck import InformationBottleneckAuditor, StageAuditRecord


def test_stage_audit_analysis():
    auditor = InformationBottleneckAuditor(seed=42)
    rng = np.random.default_rng(42)

    # Context 0 vs Context 1 features
    features_c0 = rng.normal(-1.5, 0.5, (25, 10))
    features_c1 = rng.normal(1.5, 0.5, (25, 10))
    features = np.vstack([features_c0, features_c1])
    labels = np.array([0] * 25 + [1] * 25)

    rec = auditor.analyze_stage(features, labels, stage_idx=1, stage_name="1_CONTEXT_VECTOR")
    assert rec.linear_probe_acc >= 0.90
    assert rec.mutual_info_bits > 0.50
    assert rec.separation_ratio > 1.0


def test_full_11stage_pathway_audit():
    auditor = InformationBottleneckAuditor(seed=42)
    rng = np.random.default_rng(42)

    trial_records = []
    for _ in range(30):
        ctx = rng.choice([0, 1])
        c_val = -1.0 if ctx == 0 else 1.0
        rec = {
            "true_context_label": ctx,
            "context_vector": np.full(8, c_val),
            "v_dend": np.full(10, c_val * 10.0),
            "g_nmda": np.full(10, 0.2 + (0.3 if ctx == 1 else 0.0)),
            "calcium": np.full(10, 0.3 + (0.4 if ctx == 1 else 0.0)),
            "v_soma": np.full(10, -50.0 + c_val * 5.0),
            "threshold_gain": np.full(10, 1.0),
            "spikes": rng.binomial(1, 0.3, 16),
            "burst_representation": np.array([3 if ctx == 0 else 1, 1 if ctx == 0 else 3]),
            "population_temporal": np.full(16, c_val),
            "action_population": np.array([1.0, 0.0] if ctx == 0 else [0.0, 1.0]),
            "persistent_action": np.array([0.9, 0.1] if ctx == 0 else [0.1, 0.9]),
        }
        trial_records.append(rec)

    results = auditor.audit_pathway(trial_records)
    assert "_summary" in results
    assert "1_CONTEXT_VECTOR" in results
    assert "8_BURST_REPRESENTATION" in results
    assert "11_PERSISTENT_ACTION" in results
    assert results["_summary"]["continuous_context_mi_bits"] > 0.5
