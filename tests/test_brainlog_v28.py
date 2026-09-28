"""Unit tests for BrainLog v2.8 schema and Phase 10 events."""

from pathlib import Path
import numpy as np
import pytest

from observability.brainlog import BrainLogSerializer, BrainLogValidator
from observability.capture import StateCapture
from observability.events import ComputationalEvent, EventType
from observability.topology import TopologyExtractor
from brain.two_compartment_spatial import TwoCompartmentSpatialNetwork


def test_brainlog_v28_event_types():
    assert EventType.DENDRITIC_STATE_UPDATED.value == "DENDRITIC_STATE_UPDATED"
    assert EventType.SOMATIC_STATE_UPDATED.value == "SOMATIC_STATE_UPDATED"
    assert EventType.SLOW_CONDUCTANCE_UPDATED.value == "SLOW_CONDUCTANCE_UPDATED"
    assert EventType.DENDRITIC_PLATEAU.value == "DENDRITIC_PLATEAU"
    assert EventType.COMPARTMENT_COUPLED.value == "COMPARTMENT_COUPLED"
    assert EventType.ELIGIBILITY_DISTRIBUTED.value == "ELIGIBILITY_DISTRIBUTED"
    assert EventType.TARGETED_FEEDBACK_RECEIVED.value == "TARGETED_FEEDBACK_RECEIVED"
    assert EventType.CREDIT_LEAKAGE_DETECTED.value == "CREDIT_LEAKAGE_DETECTED"


def test_brainlog_v28_serialization_and_validation(tmp_path: Path):
    model = TwoCompartmentSpatialNetwork(input_dim=4, hidden_dim=10, output_dim=1, seed=42)
    topology = TopologyExtractor.from_network(model, "test_v28_model")
    capture = StateCapture(topology=topology, dt_ms=1.0)

    # Capture steps using capture_two_compartment_step
    x = np.ones(4, dtype=np.float32)
    for step in range(3):
        y = model.step(x, reward=1.0)
        capture.capture_two_compartment_step(
            step=step,
            input_vec=x,
            model=model,
            y_val=float(y[0]),
            target_val=1.0,
            mask_val=1.0,
            reward=1.0,
            metrics={"csi": 0.5, "precision": 0.8, "leakage": 0.2},
        )

    out_file = tmp_path / "test_v28.brainlog"
    metadata = {
        "title": "Test Trace v2.8",
        "phase": 10,
    }

    BrainLogSerializer.save(
        filepath=out_file,
        metadata=metadata,
        topology=topology,
        frames=capture.frames,
        events=capture.all_events,
        snapshots=capture.snapshots,
        version="2.8",
    )

    valid, msg = BrainLogValidator.validate(out_file)
    assert valid, f"Validation failed: {msg}"
