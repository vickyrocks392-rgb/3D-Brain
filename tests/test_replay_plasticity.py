"""Unit tests for BrainLog v2.1 serialization, validation, and ReplayEngine plasticity inspection."""

import numpy as np
import pytest

from brain.plasticity import OjaPlasticity, PlasticRNN
from observability.brainlog import BrainLog, BrainLogValidator
from observability.capture import StateCapture
from observability.events import ComputationalEvent, EventType
from observability.replay import ReplayEngine
from observability.topology import TopologyExtractor


def test_brainlog_v2_1_validation_and_plasticity_replay(tmp_path):
    model = PlasticRNN(input_dim=2, hidden_dim=6, output_dim=1, plasticity_rule=OjaPlasticity(lr=0.01), seed=42)
    topology = TopologyExtractor().extract(model, "Plastic_6units")
    capture = StateCapture(topology, dt_ms=40.0)

    # Simulate 5 steps with online plasticity
    for step in range(5):
        x = np.array([[1.0, 0.5]], dtype=np.float32)
        y = model.step(x)
        capture.capture_step(
            step=step,
            input_vec=x[0],
            model=model,
            y_val=float(y[0, 0]),
            target_val=1.0,
            mask_val=1.0 if step >= 3 else 0.0,
        )

    # Save v2.1 brainlog
    blog = BrainLog(
        metadata={"experiment_id": "EXP003", "model_id": "PLASTIC-OJA"},
        topology=topology,
        frames=capture.frames,
        events=capture.all_events,
        snapshots={},
        version="2.1",
    )
    blog_path = tmp_path / "test_exp003.brainlog"
    blog.save(blog_path)

    # Validate
    is_valid, msg = BrainLogValidator.validate(blog_path)
    assert is_valid, f"Validation failed: {msg}"

    # Replay
    replay = ReplayEngine(blog.to_dict())
    assert replay.total_steps == 5

    # Check plasticity events
    p_events = replay.get_plasticity_events()
    assert len(p_events) > 0
    assert p_events[0]["type"] == EventType.PLASTICITY_UPDATE.value

    # Edge inspection
    edge_insp = replay.inspect_edge(src=2, dst=3)
    assert edge_insp is not None
    assert edge_insp.current_weight is not None
    assert edge_insp.recent_dw is not None
