"""Unit and integration tests for Phase 13 Network and Biophysical Pathway Audit.

Validates:
1. Network initialization with frozen substrate, biophysical spike encoder, and persistence.
2. Complete forward simulation step and observable state schema.
3. Causal interventions suite C1 through C12.
4. Zero recurrent credit leakage contract.
5. Multi-stage biophysical pathway audit and information preservation pipeline.
"""

import numpy as np
import pytest

from brain.phase13_network import Phase13Network
from brain.spike_encoding import SpikeEncodingStrategy
from brain.biophysical_audit import BiophysicalPathwayAuditor


def test_phase13_network_initialization_and_forward():
    """Test 1: Phase13Network initializes and steps through complete biophysical pathway."""
    net = Phase13Network(
        input_dim=4,
        hidden_dim=20,
        output_dim=2,
        neurons_per_action=8,
        spike_strategy=SpikeEncodingStrategy.S9_FULL_BIOPHYSICAL,
        seed=42,
    )
    assert net.parameter_count() > 0
    assert net.spike_encoder.total_action_neurons == 16

    x = np.ones(4, dtype=np.float32)
    probs, chosen = net.step(x, target_action=0, is_choice_step=True)

    assert len(probs) == 2
    assert chosen in (0, 1)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    obs = net.observable_state()
    assert "spike_encoder" in obs
    assert "persistence" in obs
    assert obs["interface"]["recurrent_leakage"] == 0.0


def test_phase13_causal_interventions_c1_through_c12():
    """Test 2: All causal interventions C1-C12 can be toggled without error."""
    net = Phase13Network(input_dim=4, hidden_dim=16, output_dim=2, seed=42)
    interventions = [
        "C1_CONTEXT_SCRAMBLE",
        "C2_CONTEXT_SWAP",
        "C3_LESION_25",
        "C3_LESION_50",
        "C3_LESION_100",
        "C4_IRRELEVANT_CONTEXT",
        "C5_LABEL_PERMUTATION",
        "C6_FREEZE_PLASTICITY",
        "C7_SPIKE_TIMING_SHUFFLE",
        "C8_SPIKE_COUNT_SHUFFLE",
        "C9_THRESHOLD_RANDOMIZATION",
        "C10_GAIN_RANDOMIZATION",
        "C11_COMPETITION_DISABLED",
        "C12_PERSISTENCE_DISABLED",
    ]
    x = np.ones(4, dtype=np.float32)
    for c_id in interventions:
        net.apply_causal_intervention(c_id, True)
        probs, chosen = net.step(x)
        assert len(probs) == 2
        net.apply_causal_intervention(c_id, False)


def test_biophysical_pathway_audit():
    """Test 3: BiophysicalPathwayAuditor computes multi-stage metrics and information loss."""
    auditor = BiophysicalPathwayAuditor(seed=42)
    trial_records = []

    # Generate synthetic trial records with 2 distinct contexts
    for i in range(20):
        ctx_label = i % 2
        noise = np.random.normal(0, 0.1, 10).astype(np.float32)
        mean_val = 1.0 if ctx_label == 1 else -1.0
        vec = np.full(10, mean_val, dtype=np.float32) + noise

        rec = {
            "true_context_label": ctx_label,
            "context_vector": vec,
            "v_dend": vec * 1.5,
            "g_nmda": np.abs(vec) * 0.2,
            "v_soma": vec * 1.2,
            "theta_eff": np.full(10, -50.0) + (0.5 if ctx_label == 1 else -0.5),
            "action_spikes": np.where(vec > 0, 1.0, 0.0),
            "action_pool_rates": np.array([20.0 if ctx_label == 0 else 5.0, 5.0 if ctx_label == 0 else 20.0]),
            "persistent_action_state": np.array([1.0 if ctx_label == 0 else 0.1, 0.1 if ctx_label == 0 else 1.0]),
        }
        trial_records.append(rec)

    results = auditor.run_pathway_audit(trial_records)
    assert "_summary" in results
    assert "1_CONTEXT_VECTOR" in results
    assert "8_PERSISTENT_ACTION" in results
    assert results["1_CONTEXT_VECTOR"]["linear_probe_acc"] >= 0.8
    assert "retention_percentage" in results["_summary"]
