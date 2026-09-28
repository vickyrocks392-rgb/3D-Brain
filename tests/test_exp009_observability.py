"""Observability, BrainLog v2.7, and Replay Tests for Phase 9.

Verifies:
1. BrainLog v2.7 schema validation with new computational events:
   ELIGIBILITY_UPDATED, NEUROMODULATOR_RELEASED, REWARD_RECEIVED, RPE_COMPUTED,
   PLASTICITY_APPLIED, CREDIT_ASSIGNED, COMMUNITY_REINFORCED, COMMUNITY_SUPPRESSED.
2. ReplayEngine reward timeline inspection at T-100, T-50, T-25, T-10, T-1, T, T+1.
"""

from pathlib import Path
import numpy as np
import pytest

from observability.brainlog import BrainLog, BrainLogValidator, BrainLogSerializer
from observability.capture import StateCapture
from observability.events import ComputationalEvent, EventType
from observability.replay import ReplayEngine
from observability.topology import BrainTopology, TopologyNode, TopologyEdge
from brain.three_factor_spatial import ThreeFactorSpatialNetwork
from benchmarks.delayed_contextual_choice import DelayedContextualChoiceTask


def test_brainlog_v27_schema_validation():
    """Verify that BrainLogValidator approves v2.7 traces containing Phase 9 events."""
    nodes = [
        TopologyNode(id=0, label="Input_0", type="INPUT", pos=[0.0, 0.0, -5.0], unit_index=-1),
        TopologyNode(id=1, label="Unit_0", type="RECURRENT", pos=[0.0, 0.0, 0.0], unit_index=0),
        TopologyNode(id=2, label="Output_0", type="OUTPUT", pos=[0.0, 0.0, 5.0], unit_index=-1),
    ]
    edges = [
        TopologyEdge(src=0, dst=1, weight=0.5),
        TopologyEdge(src=1, dst=2, weight=0.8),
    ]
    topo = BrainTopology(topology_id="topo_v27_test", model_id="spatial_three_factor", nodes=nodes, edges=edges)
    capture = StateCapture(topo, dt_ms=1.0)

    # Frame 0
    capture.frames.append(
        capture.capture_step(
            step=0,
            input_vec=np.array([0.5]),
            model=ThreeFactorSpatialNetwork(input_dim=1, hidden_dim=1, output_dim=1, seed=42),
            y_val=0.4,
            target_val=1.0,
            mask_val=1.0,
        )
    )

    # Add Phase 9 events
    events = [
        ComputationalEvent(event_type=EventType.ELIGIBILITY_UPDATED, step=0, time_ms=0.0, magnitude=0.45),
        ComputationalEvent(event_type=EventType.NEUROMODULATOR_RELEASED, step=0, time_ms=0.0, magnitude=1.0),
        ComputationalEvent(event_type=EventType.REWARD_RECEIVED, step=0, time_ms=0.0, magnitude=1.0),
        ComputationalEvent(event_type=EventType.RPE_COMPUTED, step=0, time_ms=0.0, magnitude=0.85),
        ComputationalEvent(event_type=EventType.PLASTICITY_APPLIED, step=0, time_ms=0.0, magnitude=0.02),
        ComputationalEvent(event_type=EventType.CREDIT_ASSIGNED, step=0, time_ms=0.0, magnitude=0.02),
        ComputationalEvent(event_type=EventType.COMMUNITY_REINFORCED, step=0, time_ms=0.0, magnitude=0.05),
        ComputationalEvent(event_type=EventType.COMMUNITY_SUPPRESSED, step=0, time_ms=0.0, magnitude=0.03),
    ]

    blog_dict = {
        "brainlog_version": "2.7",
        "metadata": {"test": "phase9"},
        "topology": topo.to_dict(),
        "frames": [f.to_dict() for f in capture.frames],
        "events": [e.to_dict() for e in events],
        "snapshots": {},
    }

    valid, msg = BrainLogValidator.validate(blog_dict)
    assert valid, f"Validation failed: {msg}"


def test_replay_reward_timeline_inspection():
    """Verify ReplayEngine inspect_reward_timeline at T-100, T-50, T-25, T-10, T-1, T, T+1."""
    nodes = [
        TopologyNode(id=0, label="Input_0", type="INPUT", pos=[0.0, 0.0, -5.0], unit_index=-1),
        TopologyNode(id=1, label="Unit_0", type="RECURRENT", pos=[0.0, 0.0, 0.0], unit_index=0),
        TopologyNode(id=2, label="Output_0", type="OUTPUT", pos=[0.0, 0.0, 5.0], unit_index=-1),
    ]
    edges = [TopologyEdge(src=0, dst=1, weight=0.5)]
    topo = BrainTopology(topology_id="topo_timeline_test", model_id="spatial_three_factor", nodes=nodes, edges=edges)

    frames = []
    # Create 200 synthetic frames
    for s in range(200):
        is_reward_step = (s == 120)
        frames.append({
            "step": s,
            "time_ms": float(s),
            "inputs": [0.5],
            "activations": [0.5, 0.7, 0.2],
            "states": [0.5, -60.0, 0.2],
            "output": 0.2,
            "target": 1.0,
            "error": 0.8,
            "mask": 1.0,
            "signals": [],
            "events": [
                {"type": "REWARD_RECEIVED", "step": s, "time_ms": float(s), "magnitude": 1.0}
            ] if is_reward_step else [],
            "dynamic_details": {
                "mean_eligibility": 0.15,
                "reward": 1.0 if is_reward_step else 0.0,
                "rpe": 0.8 if is_reward_step else 0.0,
            }
        })

    blog_dict = {
        "brainlog_version": "2.7",
        "metadata": {"test": "timeline"},
        "topology": topo.to_dict(),
        "frames": frames,
        "events": [],
        "snapshots": {},
    }

    engine = ReplayEngine(blog_dict)
    timeline = engine.inspect_reward_timeline(reward_step=120, offsets=[-100, -50, -25, -10, -1, 0, 1])

    # Must inspect each of the requested offsets
    assert 120 - 100 in timeline
    assert 120 - 50 in timeline
    assert 120 - 25 in timeline
    assert 120 - 10 in timeline
    assert 120 - 1 in timeline
    assert 120 in timeline
    assert 120 + 1 in timeline

    # At step 120 (T), reward and RPE are active
    t_step = timeline[120]
    assert t_step["reward"] == 1.0
    assert t_step["rpe"] == 0.8
    assert t_step["mean_eligibility"] == 0.15
