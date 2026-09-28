"""
Tests for Phase 5 Embodied Replay Engine methods.
Validates exact computational reconstruction of environment state, actions,
sensory predictions, prediction errors, structural events, and active topology.
"""

import json
from pathlib import Path
import pytest
from observability.replay import ReplayEngine


@pytest.fixture
def sample_v23_brainlog_path():
    path = Path("experiments/results/brainlogs/EXP005_LOCAL_PRED_DYNAMIC_SYNAPSES_ACTIVE_s42.brainlog")
    assert path.exists(), f"Brainlog file missing at {path}"
    return path


def test_embodied_replay_initialization(sample_v23_brainlog_path):
    engine = ReplayEngine(sample_v23_brainlog_path)
    assert engine.total_steps > 0
    assert engine.metadata["experiment"] == "EXP-005"
    assert engine.metadata["embodiment"] == "ACTIVE"


def test_embodied_replay_environment_and_action(sample_v23_brainlog_path):
    engine = ReplayEngine(sample_v23_brainlog_path)
    
    # Step 0
    engine.seek(0)
    env_state = engine.get_environment_state()
    assert env_state is not None
    assert "agent_pos" in env_state
    assert "env_id" in env_state
    assert env_state["env_id"] == "ENV_A"
    
    action = engine.get_action()
    assert action is not None
    assert isinstance(action, int)


def test_embodied_replay_prediction(sample_v23_brainlog_path):
    engine = ReplayEngine(sample_v23_brainlog_path)
    engine.seek(5)
    
    pred_data = engine.get_prediction()
    assert pred_data is not None
    assert "pred_s" in pred_data
    assert "obs_next" in pred_data
    assert "pred_mse" in pred_data
    assert len(pred_data["pred_s"]) == 8
    assert len(pred_data["obs_next"]) == 8
    assert isinstance(pred_data["pred_mse"], float)


def test_embodied_replay_structural_events(sample_v23_brainlog_path):
    engine = ReplayEngine(sample_v23_brainlog_path)
    events = engine.get_structural_events()
    assert isinstance(events, list)
    # The run should contain structural events like SYNAPSE_FORMED or SYNAPSE_PRUNED
    event_types = {e.get("type") for e in events}
    assert ("SYNAPSE_FORMED" in event_types) or ("SYNAPSE_PRUNED" in event_types)


def test_embodied_replay_active_topology_at_step(sample_v23_brainlog_path):
    engine = ReplayEngine(sample_v23_brainlog_path)
    
    # Check topology at start vs later: 8 sensors + 40 recurrent + 8 output = 56 nodes
    top_0 = engine.get_active_topology_at_step(0)
    assert "nodes" in top_0
    assert "edges" in top_0
    assert len(top_0["nodes"]) == 56
    
    top_end = engine.get_active_topology_at_step(engine.total_steps - 1)
    assert len(top_end["nodes"]) == 56
    assert len(top_end["edges"]) > 0
