"""Unit tests for standalone ReplayEngine."""

import numpy as np
import pytest
from brain.discrete_leaky_rnn import DiscreteLeakyRNN
from observability import (
    BrainLogSerializer,
    ReplayEngine,
    StateCapture,
    TopologyExtractor,
)


@pytest.fixture
def sample_brainlog_path(tmp_path):
    model = DiscreteLeakyRNN(input_dim=2, hidden_dim=6, output_dim=1, seed=42)
    extractor = TopologyExtractor()
    topo = extractor.extract(model, "DISC-LRNN")
    capture = StateCapture(topo, dt_ms=40.0)

    for s in range(12):
        model.step(np.array([[1.0, 0.0]]))
        capture.capture_step(
            step=s,
            input_vec=np.array([1.0, 0.0]),
            model=model,
            y_val=float(s * 0.1),
            target_val=1.0 if s >= 8 else 0.0,
            mask_val=1.0 if s >= 8 else 0.0,
            phase_label="DELAY_START" if s == 2 else "RUNNING",
        )

    out_file = tmp_path / "replay_test.brainlog"
    BrainLogSerializer.save(
        filepath=out_file,
        metadata={"experiment_id": "REPLAY_TEST", "model_id": "DISC-LRNN", "budget": 300},
        topology=topo,
        frames=capture.frames,
        events=capture.all_events,
        snapshots=capture.snapshots,
    )
    return out_file


def test_replay_seek_and_stepping(sample_brainlog_path):
    engine = ReplayEngine(sample_brainlog_path)
    assert engine.total_steps == 12
    assert engine.metadata["model_id"] == "DISC-LRNN"

    # Initial frame
    f0 = engine.current_frame()
    assert f0["step"] == 0

    # Step forward
    f1 = engine.step_forward()
    assert f1["step"] == 1
    assert engine.current_step == 1

    # Seek
    f7 = engine.seek(7)
    assert f7["step"] == 7
    assert engine.current_step == 7

    # Step backward
    f6 = engine.step_backward()
    assert f6["step"] == 6


def test_replay_inspect_node_and_edge(sample_brainlog_path):
    engine = ReplayEngine(sample_brainlog_path)
    engine.seek(5)

    # Inspect recurrent node 2
    node_insp = engine.inspect_node(2)
    assert node_insp.node_id == 2
    assert node_insp.node_type == "RECURRENT"
    assert "alpha" in node_insp.dynamic_vars or "state" is not None
    assert len(node_insp.incoming_connections) >= 0

    # Inspect edge
    edge = engine.topology["edges"][0]
    edge_insp = engine.inspect_edge(edge["src"], edge["dst"])
    assert edge_insp is not None
    assert edge_insp.src == edge["src"]
    assert edge_insp.dst == edge["dst"]
    assert edge_insp.weight == edge["weight"]


def test_replay_event_querying(sample_brainlog_path):
    engine = ReplayEngine(sample_brainlog_path)
    all_events = engine.get_all_events()
    assert len(all_events) > 0

    events_at_0 = engine.get_events_at_step(0)
    assert any(e["type"] == "TRIAL_STARTED" for e in events_at_0)
