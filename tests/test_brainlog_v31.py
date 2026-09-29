"""Unit tests for BrainLog v3.1 schema and Phase 13 event types.

Validates:
1. BrainLogValidator accepts v3.1 traces.
2. BrainLogSerializer serializes and validates v3.1.
3. ComputationalEvent accepts new Phase 13 event types (THRESHOLD_UPDATED, SPIKE_ENCODED, etc.).
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from observability.brainlog import BrainLog, BrainLogSerializer, BrainLogValidator
from observability.events import ComputationalEvent, EventType
from observability.capture import TelemetryFrame
from observability.topology import BrainTopology, TopologyNode, TopologyEdge


def test_brainlog_v31_validation():
    """Test 1: BrainLogValidator validates v3.1 trace."""
    nodes = [TopologyNode(id=0, pos=(0, 0, 0), type="excitatory", label="N0", unit_index=0, module_id=0)]
    edges = []
    topo = BrainTopology(topology_id="topo_test", model_id="model_test", nodes=nodes, edges=edges)

    frame = TelemetryFrame(
        step=0,
        time_ms=0.0,
        inputs=[0.0],
        activations=[0.0],
        states=[0.0],
        output=0.0,
        target=0.0,
        error=0.0,
        mask=1.0,
    )

    ev = ComputationalEvent(
        event_type=EventType.SPIKE_ENCODED,
        step=0,
        time_ms=0.0,
        node_id=0,
        magnitude=1.0,
        payload={"strategy": "S9_FULL_BIOPHYSICAL", "gain": 1.0},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_v31.brainlog"
        BrainLogSerializer.save(
            filepath=log_path,
            metadata={"experiment": "EXP013_TEST", "phase": 13},
            topology=topo,
            frames=[frame],
            events=[ev],
            snapshots={},
            version="3.1",
        )

        valid, msg = BrainLogValidator.validate(log_path)
        assert valid, f"Validation failed: {msg}"

        loaded = BrainLogSerializer.load(log_path)
        assert loaded["brainlog_version"] == "3.1"
        assert len(loaded["events"]) == 1
        assert loaded["events"][0]["type"] == EventType.SPIKE_ENCODED.value
