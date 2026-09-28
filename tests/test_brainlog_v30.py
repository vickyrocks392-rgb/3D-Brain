"""Unit tests for BrainLog v3.0 serialization, validation, and Phase 12 interface events."""

from pathlib import Path
import numpy as np
import pytest

from brain.phase11_network import Phase11SpatialNetwork
from observability.brainlog import BrainLog, BrainLogValidator, BrainLogSerializer
from observability.capture import StateCapture
from observability.events import ComputationalEvent, EventType
from observability.topology import TopologyExtractor


def test_brainlog_v30_event_types():
    assert EventType.CONTEXT_READOUT.value == "CONTEXT_READOUT"
    assert EventType.ACTION_EVIDENCE_ACCUMULATED.value == "ACTION_EVIDENCE_ACCUMULATED"
    assert EventType.ACTION_ERROR_COMPUTED.value == "ACTION_ERROR_COMPUTED"
    assert EventType.ACTION_ELIGIBILITY_UPDATED.value == "ACTION_ELIGIBILITY_UPDATED"
    assert EventType.ACTION_CREDIT_ASSIGNED.value == "ACTION_CREDIT_ASSIGNED"
    assert EventType.ACTION_SYNAPSE_UPDATED.value == "ACTION_SYNAPSE_UPDATED"
    assert EventType.ACTION_READOUT_LESION.value == "ACTION_READOUT_LESION"
    assert EventType.CONTEXT_INTERVENTION.value == "CONTEXT_INTERVENTION"
    assert EventType.ACTION_INTERFACE_STATE.value == "ACTION_INTERFACE_STATE"


def test_brainlog_v30_serialization_and_validation(tmp_path: Path):
    model = Phase11SpatialNetwork(input_dim=4, hidden_dim=10, output_dim=2, seed=42)
    topology = TopologyExtractor.from_network(model, "test_v30_model")
    capture = StateCapture(topology=topology, dt_ms=1.0)

    # Capture 5 steps of Phase 12
    x = np.ones(4, dtype=np.float32)
    for s in range(5):
        v_act, sel_act = model.step(x, reward=0.0, target_action=0, context_id=0)
        capture.capture_phase12_step(
            step=s,
            input_vec=x,
            model=model,
            y_val=v_act,
            target_val=0.0,
            mask_val=1.0 if s == 4 else 0.0,
            reward=1.0 if s == 4 else 0.0,
            context_id=0,
            interface_telemetry={
                "context_source": "ATTRACTOR",
                "context_norm": 1.25,
                "context_dim": 10,
                "action_evidence": [0.8, -0.4],
                "selected_action": 0,
                "winner_margin": 1.2,
                "action_entropy": 0.35,
                "action_probs": [0.77, 0.23],
                "learning_rule": "R3_THREE_FACTOR",
                "eligibility_updated": True,
                "eligibility_norm": 0.45,
                "credit_assigned": True,
                "dw_norm": 0.005,
                "csi": 0.85,
                "precision": 0.92,
                "leakage": 0.08,
                "recurrent_leakage": 0.0,
                "interface_credit": 0.005,
                "collateral_credit": 0.0004,
            },
            phase_label="PHASE12_TEST",
        )

    capture.complete_trial(final_acc=1.0)

    log_path = tmp_path / "test_v30.brainlog"
    saved = BrainLogSerializer.save(
        filepath=log_path,
        metadata={"experiment_id": "EXP012_TEST", "phase": "12"},
        topology=topology,
        frames=capture.frames,
        events=capture.all_events,
        snapshots=capture.snapshots,
        version="3.0",
    )

    valid, reason = BrainLogValidator.validate(saved)
    assert valid, f"BrainLog v3.0 validation failed: {reason}"
