"""Unit tests for Observability Engine: topology extraction, state capture, events, and brainlog v2.0."""

import numpy as np
import pytest
from brain.liquid import CandLiquid
from brain.discrete_leaky_rnn import DiscreteLeakyRNN
from observability import (
    BrainLogSerializer,
    BrainLogValidator,
    EventType,
    StateCapture,
    TopologyExtractor,
)


def test_topology_extractor_nodes_and_edges():
    model = DiscreteLeakyRNN(input_dim=2, hidden_dim=15, output_dim=1, seed=42)
    extractor = TopologyExtractor(edge_weight_threshold=0.04)
    topo = extractor.extract(model, "DISC-LRNN")

    # 2 input nodes + 15 recurrent + 1 output = 18 nodes
    assert topo.node_count == 18
    assert len(topo.nodes) == 18
    assert topo.edge_count > 0

    # Verify input nodes positions
    assert topo.nodes[0].type == "INPUT"
    assert topo.nodes[0].pos[2] == -8.0
    assert topo.nodes[1].type == "INPUT"
    assert topo.nodes[1].pos[2] == -8.0

    # Verify output node position
    assert topo.nodes[-1].type == "OUTPUT"
    assert topo.nodes[-1].pos[2] == 8.0

    # Verify recurrent node positions are in [-2.5, 2.5]
    for n in topo.nodes[2:-1]:
        assert n.type == "RECURRENT"
        assert -3.0 <= n.pos[2] <= 3.0


def test_state_capture_events_and_snapshots():
    model = CandLiquid(input_dim=2, hidden_dim=8, output_dim=1, seed=42)
    extractor = TopologyExtractor()
    topo = extractor.extract(model, "CAND-LI")

    capture = StateCapture(topo, dt_ms=40.0)

    # Step 0 (Cue phase)
    f0 = capture.capture_step(
        step=0,
        input_vec=np.array([1.0, 1.0]),
        model=model,
        y_val=0.01,
        target_val=0.0,
        mask_val=0.0,
        phase_label="CUE",
    )
    assert f0.step == 0
    assert any(e.event_type == EventType.TRIAL_STARTED for e in f0.events)
    assert any(e.event_type == EventType.INPUT_PRESENTED for e in f0.events)

    # Step 5 (Delay phase)
    f5 = capture.capture_step(
        step=5,
        input_vec=np.array([0.02, 0.0]),
        model=model,
        y_val=0.02,
        target_val=0.0,
        mask_val=0.0,
        phase_label="DELAY_START",
    )
    assert any(e.event_type == EventType.MEMORY_RETAINED for e in f5.events)

    # Step 30 (Probe phase with error)
    f30 = capture.capture_step(
        step=30,
        input_vec=np.array([1.0, 2.0]),
        model=model,
        y_val=-0.2,
        target_val=1.0,
        mask_val=1.0,
        phase_label="PROBE_START",
    )
    assert any(e.event_type == EventType.OUTPUT_COMPUTED for e in f30.events)
    assert any(e.event_type == EventType.PREDICTION_ERROR for e in f30.events)

    capture.complete_trial(final_acc=1.0)
    assert any(e.event_type == EventType.TRIAL_COMPLETED for e in capture.all_events)
    assert 0 in capture.snapshots


def test_brainlog_v2_serialization_and_validation(tmp_path):
    model = DiscreteLeakyRNN(input_dim=2, hidden_dim=8, output_dim=1, seed=42)
    extractor = TopologyExtractor()
    topo = extractor.extract(model, "DISC-LRNN")
    capture = StateCapture(topo)

    for s in range(5):
        capture.capture_step(
            step=s,
            input_vec=np.array([1.0, 0.0]),
            model=model,
            y_val=0.1,
            target_val=0.0,
            mask_val=0.0,
        )

    out_file = tmp_path / "test_v2.brainlog"
    BrainLogSerializer.save(
        filepath=out_file,
        metadata={"experiment_id": "EXP002", "model_id": "DISC-LRNN"},
        topology=topo,
        frames=capture.frames,
        events=capture.all_events,
        snapshots=capture.snapshots,
    )

    assert out_file.exists()
    valid, msg = BrainLogValidator.validate(out_file)
    assert valid, f"v2.0 validation failed: {msg}"
