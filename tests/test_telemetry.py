"""Tests for telemetry schema and .brainlog serialization/validation."""

import json
from pathlib import Path
import numpy as np
import pytest
from brain.liquid import CandLiquid
from brain.rnn import BaseRNN
from visualization.telemetry import (
    build_frame,
    build_topology,
    export_brainlog,
    validate_brainlog,
)


def test_build_topology_structure():
    """Verify topology contains valid 3D coordinates, node counts, and edge weights."""
    model = BaseRNN(input_dim=2, hidden_dim=10, output_dim=1, seed=42)
    topology = build_topology(model, "BASE-RNN")

    # Nodes: 2 inputs + 10 recurrent + 1 output = 13 nodes
    assert topology["node_count"] == 13
    assert len(topology["nodes"]) == 13
    assert len(topology["edges"]) > 0

    # Check node 0 position (input at Z = -8.0)
    assert topology["nodes"][0]["pos"][2] == -8.0
    # Check output position (output at Z = +8.0)
    assert topology["nodes"][-1]["pos"][2] == 8.0

    for node in topology["nodes"]:
        assert len(node["pos"]) == 3
        assert "type" in node
        assert "label" in node


def test_build_frame_conformance():
    """Verify that build_frame outputs compliant activation and state arrays."""
    inp = np.array([1.0, 2.0], dtype=np.float32)
    h = np.zeros(6, dtype=np.float32)
    a = np.ones(6, dtype=np.float32) * 0.5
    edges = [{"src": 0, "dst": 2, "weight": 0.5, "latency_steps": 1}]

    frame = build_frame(
        step=1,
        time_ms=40.0,
        input_vec=inp,
        h_vec=h,
        a_vec=a,
        y_val=0.8,
        target_val=1.0,
        mask_val=1.0,
        edges=edges,
    )

    # 2 inputs + 6 hidden + 1 output = 9 values
    assert len(frame["activations"]) == 9
    assert len(frame["states"]) == 9
    assert frame["step"] == 1
    assert frame["time_ms"] == 40.0
    assert frame["error"] == pytest.approx(0.2, abs=1e-4)


def test_export_and_validate_brainlog_roundtrip(tmp_path):
    """Verify export_brainlog produces a valid file that passes strict validation."""
    model = CandLiquid(input_dim=2, hidden_dim=8, output_dim=1, seed=42)
    topology = build_topology(model, "CAND-LI")

    frames = []
    for step in range(5):
        frame = build_frame(
            step=step,
            time_ms=step * 40.0,
            input_vec=np.array([1.0, 0.0]),
            h_vec=np.zeros(8),
            a_vec=np.zeros(8),
            y_val=0.0,
            target_val=0.0,
            mask_val=0.0,
            edges=topology["edges"],
        )
        frames.append(frame)

    metadata = {
        "experiment_id": "TEST_EXP",
        "model_id": "CAND-LI",
        "budget": 300,
        "seed": 42,
        "delay": 10,
        "total_steps": 5,
        "dt_ms": 40.0,
    }

    out_file = tmp_path / "test_trace.brainlog"
    export_brainlog(out_file, metadata, topology, frames)

    assert out_file.exists()
    valid, msg = validate_brainlog(out_file)
    assert valid, f"Validation failed: {msg}"


def test_validate_rejects_corrupted_brainlog():
    """Verify validator catches schema corruptions."""
    corrupted_data = {"brainlog_version": "0.9"}
    valid, msg = validate_brainlog(corrupted_data)
    assert not valid
    assert "incompatible" in msg
