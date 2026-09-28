"""Tests for PredictiveSpatialRNN, brainlog v2.3, and ReplayEngine in Phase 5."""

import json
from pathlib import Path
import numpy as np
import pytest

from benchmarks.embodied_environment import MinimalEmbodiedEnvironment
from brain.predictive_plasticity import LocalPredictivePlasticity
from brain.predictive_spatial import PredictiveSpatialRNN
from brain.structural_plasticity import StructuralPlasticity
from observability.brainlog import BrainLogSerializer, BrainLogValidator
from observability.capture import StateCapture
from observability.replay import ReplayEngine
from observability.topology import TopologyExtractor


def test_closed_sensorimotor_loop():
    env = MinimalEmbodiedEnvironment(arena_size=10.0, seed=42)
    obs = env.reset()

    rule = LocalPredictivePlasticity(hidden_dim=15, obs_dim=8, lr=0.01, seed=42)
    struct = StructuralPlasticity(update_interval=5, min_degree=1, max_degree=8, seed=42)
    brain = PredictiveSpatialRNN(
        obs_dim=8,
        action_dim=4,
        hidden_dim=15,
        predictive_rule=rule,
        structural_plasticity=struct,
        seed=42,
    )

    errors = []
    for t in range(25):
        # 1. Brain selects action
        action = brain.select_action(epsilon=0.1)
        # 2. Environment executes action
        next_obs, info = env.step(action)
        # 3. Brain processes closed-loop step
        pred_next, err_mse, metrics = brain.step(obs, action, next_obs)
        errors.append(err_mse)
        obs = next_obs

    assert len(errors) == 25
    assert all(e >= 0.0 for e in errors)
    assert brain.current_step == 25


def test_brainlog_v23_serialization_and_replay(tmp_path: Path):
    env = MinimalEmbodiedEnvironment(seed=42)
    obs = env.reset()

    rule = LocalPredictivePlasticity(hidden_dim=12, obs_dim=8, seed=42)
    struct = StructuralPlasticity(update_interval=5, min_degree=1, max_degree=6, seed=42)
    brain = PredictiveSpatialRNN(
        obs_dim=8,
        action_dim=4,
        hidden_dim=12,
        predictive_rule=rule,
        structural_plasticity=struct,
        seed=42,
    )

    extractor = TopologyExtractor()
    topology = extractor.extract(brain, "PRED-SPATIAL")
    capture = StateCapture(topology, dt_ms=40.0)

    for step in range(20):
        action = brain.select_action()
        next_obs, info = env.step(action)
        pred_next, err, metrics = brain.step(obs, action, next_obs)

        capture.capture_embodied_step(
            step=step,
            obs_t=obs,
            action=action,
            obs_next=next_obs,
            pred_s=pred_next,
            pred_error=err,
            model=brain,
            env_info=info,
            structural_events=brain.last_structural_events,
        )
        obs = next_obs

    log_path = tmp_path / "test_embodied_v23.brainlog"
    metadata = {
        "architecture": "PRED-SPATIAL",
        "phase": "Phase 5",
        "env_id": "ENV_A",
        "total_steps": 20,
    }

    BrainLogSerializer.save(
        filepath=log_path,
        metadata=metadata,
        topology=topology,
        frames=capture.frames,
        events=capture.all_events,
        snapshots=capture.snapshots,
        version="2.3",
    )

    # 1. Schema Validation
    valid, msg = BrainLogValidator.validate(log_path)
    assert valid, f"Validation failure: {msg}"

    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["brainlog_version"] == "2.3"

    # 2. ReplayEngine Verification
    replay = ReplayEngine(data)
    assert replay.total_steps == 20

    # Test seeking and inspecting embodied state
    replay.seek(5)
    env_state = replay.get_environment_state()
    assert env_state is not None
    assert "agent_pos" in env_state
    assert "action" in env_state

    pred_info = replay.get_prediction()
    assert pred_info is not None
    assert len(pred_info["pred_s"]) == 8
    assert len(pred_info["obs_next"]) == 8

    # Test structural events
    struct_events = replay.get_structural_events()
    assert isinstance(struct_events, list)

    active_topo = replay.get_active_topology_at_step(10)
    assert "node_count" in active_topo
    assert "edge_count" in active_topo
