"""Unit tests for BrainLog v2.9 serialization, validation, and Phase 11 events."""

from pathlib import Path
import numpy as np
import pytest

from brain.phase11_network import Phase11SpatialNetwork
from observability.brainlog import BrainLog, BrainLogValidator, BrainLogSerializer
from observability.capture import StateCapture
from observability.events import ComputationalEvent, EventType
from observability.topology import TopologyExtractor


def test_brainlog_v29_event_types():
    assert EventType.PREDICTIVE_GATE_UPDATED.value == "PREDICTIVE_GATE_UPDATED"
    assert EventType.DENDRITIC_DISINHIBITION.value == "DENDRITIC_DISINHIBITION"
    assert EventType.ATTRACTOR_MAINTAINED.value == "ATTRACTOR_MAINTAINED"
    assert EventType.ACTION_POPULATION_ACTIVATED.value == "ACTION_POPULATION_ACTIVATED"
    assert EventType.CREDIT_PATH_REINFORCED.value == "CREDIT_PATH_REINFORCED"


def test_brainlog_v29_serialization_and_validation(tmp_path: Path):
    model = Phase11SpatialNetwork(input_dim=4, hidden_dim=10, output_dim=2, seed=42)
    topology = TopologyExtractor.from_network(model, "test_v29_model")
    capture = StateCapture(topology=topology, dt_ms=1.0)

    # Capture 5 steps of Phase 11
    x = np.ones(4, dtype=np.float32)
    for s in range(5):
        v_act, sel_act = model.step(x, reward=0.0, target_action=0, context_id=0)
        capture.capture_phase11_step(
            step=s,
            input_vec=x,
            model=model,
            y_val=v_act,
            target_val=0.0,
            mask_val=1.0 if s == 4 else 0.0,
            reward=1.0 if s == 4 else 0.0,
            metrics=model.last_telemetry,
            context_id=0,
            phase_label="PHASE11_TEST",
        )

    capture.complete_trial(final_acc=1.0)

    log_path = tmp_path / "test_v29.brainlog"
    saved = BrainLogSerializer.save(
        filepath=log_path,
        metadata={"experiment_id": "EXP011_TEST", "phase": "11"},
        topology=topology,
        frames=capture.frames,
        events=capture.all_events,
        snapshots=capture.snapshots,
        version="2.9",
    )

    valid, reason = BrainLogValidator.validate(saved)
    assert valid, f"BrainLog v2.9 validation failed: {reason}"
