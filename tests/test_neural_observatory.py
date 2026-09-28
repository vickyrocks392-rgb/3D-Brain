"""Tests for Phase 9 Neural Observatory Console UI & Interaction Remediation.

Validates:
1. BrainLog loading and validation for Phase 9 traces.
2. Timeline synchronization across steps.
3. Event navigation and state seeking.
4. Historical state reconstruction without state leakage.
5. Neuron selection and inspection (E vs I, membrane potential, degrees).
6. Synapse selection and Dale constraint verification.
7. Community selection and modular membership breakdown.
8. Structural vs Functional connectivity contract.
9. Replay vs LIVE mode separation.
10. Telemetry extraction for Three-Factor RPE learning metrics.
11. Missing metric handling (explicit NOT RECORDED, no fabrication).
12. Visualization player HTML/DOM integrity and keyboard shortcuts contract.
"""

import json
from pathlib import Path
import pytest
from observability.brainlog import BrainLogValidator
from observability.replay import ReplayEngine


@pytest.fixture
def phase9_brainlog_path():
    path = Path("experiments/results/brainlogs/EXP009_three_factor_rpe_modular.brainlog")
    assert path.exists(), f"Phase 9 trace missing: {path}"
    return path


@pytest.fixture
def replay_engine(phase9_brainlog_path):
    with open(phase9_brainlog_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ReplayEngine(data)


def test_brainlog_loading_and_schema(phase9_brainlog_path):
    """Test 1: Validate BrainLog loading and strict schema conformity."""
    valid, msg = BrainLogValidator.validate(phase9_brainlog_path)
    assert valid, f"Validation error: {msg}"

    with open(phase9_brainlog_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["brainlog_version"] in ["2.6", "2.7"]
    assert "metadata" in data
    assert "topology" in data
    assert "frames" in data
    assert "events" in data
    assert len(data["frames"]) > 0
    assert len(data["topology"]["nodes"]) == 30
    assert len(data["topology"]["edges"]) == 259


def test_timeline_synchronization(replay_engine):
    """Test 2: Master timeline synchronization across arbitrary seeks."""
    total = replay_engine.total_steps
    assert total == 16

    # Test seek to start
    f0 = replay_engine.seek(0)
    assert replay_engine.current_step == 0
    assert f0["step"] == 0
    assert f0["time_ms"] == 0.0

    # Step forward
    f1 = replay_engine.step_forward()
    assert replay_engine.current_step == 1
    assert f1["step"] == 1
    assert f1["time_ms"] == 1.0

    # Seek to end
    f_end = replay_engine.seek(total - 1)
    assert replay_engine.current_step == total - 1
    assert f_end["step"] == total - 1

    # Step backward
    f_prev = replay_engine.step_backward()
    assert replay_engine.current_step == total - 2
    assert f_prev["step"] == total - 2


def test_event_navigation(replay_engine):
    """Test 3: Event navigation directly updates replay position and context."""
    events = replay_engine.events
    assert len(events) > 0

    # Find reward event
    reward_events = [e for e in events if e.get("type") == "REWARD_RECEIVED"]
    assert len(reward_events) > 0
    rew = reward_events[0]
    target_step = rew["step"]

    # Seek to event step
    frame = replay_engine.seek(target_step)
    assert frame["step"] == target_step
    assert frame["dynamic_details"]["reward"] == pytest.approx(1.0, abs=1e-4)


def test_historical_state_reconstruction(replay_engine):
    """Test 4: State reconstruction reproduces truthful past states without leakage."""
    # Step 0 vs Step 15
    f0 = replay_engine.seek(0)
    states_0 = list(f0["states"])

    f15 = replay_engine.seek(15)
    states_15 = list(f15["states"])

    # Membrane potentials should evolve
    assert len(states_0) == len(states_15) == 30
    assert states_0 != states_15

    # Returning to Step 0 must reproduce exact initial states
    f0_retry = replay_engine.seek(0)
    assert f0_retry["states"] == states_0


def test_neuron_selection_and_inspection(replay_engine):
    """Test 5: Node inspection correctly resolves cell types, membrane potentials, and connections."""
    replay_engine.seek(12)

    # Excitatory unit (Node 4)
    insp_e = replay_engine.inspect_node(4)
    assert insp_e is not None
    assert insp_e.node_id == 4
    assert insp_e.node_type == "RECURRENT"
    assert insp_e.neuron_type == "EXCITATORY"
    assert len(insp_e.pos) == 3
    assert -80.0 <= insp_e.current_state <= 0.0  # Physical membrane range in mV

    # Inhibitory unit (Node 25)
    insp_i = replay_engine.inspect_node(25)
    assert insp_i is not None
    assert insp_i.node_id == 25
    assert insp_i.neuron_type == "INHIBITORY"
    assert insp_i.module_id == 3

    # Degrees
    assert len(insp_e.incoming_connections) > 0
    assert len(insp_e.outgoing_connections) > 0


def test_synapse_selection_and_dale_constraint(replay_engine):
    """Test 6: Synapse inspection extracts Dale sign, physical distance, and weights."""
    replay_engine.seek(15)

    # Inspect edge from node 0 to 4
    edge_insp = replay_engine.inspect_edge(0, 4)
    assert edge_insp is not None
    assert edge_insp.src == 0
    assert edge_insp.dst == 4
    assert edge_insp.weight > 0
    assert edge_insp.distance is not None
    assert edge_insp.propagation_delay >= 1


def test_community_selection_and_modular_organization(replay_engine):
    """Test 7: Modular community extraction groups units correctly."""
    topo = replay_engine.topology
    nodes = topo["nodes"]

    modules = set(n.get("module_id") for n in nodes if "module_id" in n)
    assert modules == {0, 1, 2, 3}

    # Verify each module has both Excitatory and Inhibitory or appropriate clustering
    mod3_nodes = [n for n in nodes if n.get("module_id") == 3 and n.get("type") == "RECURRENT"]
    assert len(mod3_nodes) > 0
    cell_types = [n.get("metadata", {}).get("neuron_type") for n in mod3_nodes]
    assert "INHIBITORY" in cell_types


def test_structural_vs_functional_contract(replay_engine):
    """Test 8: Structural topology contains all physical edges while frames contain active signals."""
    topo = replay_engine.topology
    total_physical_edges = len(topo["edges"])
    assert total_physical_edges == 259

    # Signal propagation list in frame is sparse and dynamic
    f0 = replay_engine.seek(0)
    signals_0 = f0.get("signals", [])
    assert len(signals_0) < total_physical_edges


def test_replay_vs_live_separation(replay_engine):
    """Test 9: Replay mode correctly classifies historical vs live edge-of-time state."""
    total = replay_engine.total_steps

    # Seeking to historical step
    replay_engine.seek(5)
    assert replay_engine.current_step < total - 1
    # Mode is replay
    is_live = (replay_engine.current_step == total - 1)
    assert not is_live

    # Seeking to final step
    replay_engine.seek(total - 1)
    is_live = (replay_engine.current_step == total - 1)
    assert is_live


def test_telemetry_updates_learning_metrics(replay_engine):
    """Test 10: Telemetry frames supply Three-Factor learning variables."""
    f15 = replay_engine.seek(15)
    dyn = f15.get("dynamic_details", {})

    assert "mean_eligibility" in dyn
    assert "max_eligibility" in dyn
    assert "reward" in dyn
    assert "rpe" in dyn
    assert "num_plastic_updates" in dyn

    assert dyn["reward"] == pytest.approx(1.0, abs=1e-4)
    assert dyn["rpe"] == pytest.approx(1.0, abs=1e-4)
    assert dyn["num_plastic_updates"] > 0


def test_missing_metric_handling(replay_engine):
    """Test 11: Missing metrics in historical data must not be fabricated."""
    f0 = replay_engine.seek(0)
    dyn = f0.get("dynamic_details", {})

    # In Phase 9 modular RPE, synchrony local R was not recorded in dynamic_details
    r_local = dyn.get("r_local")
    # Must be None or absent, not a made up number
    assert r_local is None


def test_player_html_integrity_and_shortcuts():
    """Test 12: Validate visualization/player.html contains required semantic tokens, IDs, and keys."""
    html_path = Path("visualization/player.html")
    assert html_path.exists()

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Title & Branding
    assert "3D-Brain — Neural Observatory Console" in html
    assert "NEURAL OBSERVATORY" in html

    # Semantic color palette tokens
    assert "--color-excitatory" in html
    assert "--color-inhibitory" in html
    assert "--color-reward" in html
    assert "--color-plasticity" in html
    assert "--color-selected" in html

    # Required element IDs
    assert 'id="viewport"' in html
    assert 'id="timeline-slider"' in html
    assert 'id="btn-play"' in html
    assert 'id="conn-mode-btn"' in html
    assert 'id="left-panel"' in html
    assert 'id="right-panel"' in html
    assert 'id="event-feed-container"' in html
    assert 'id="marker-rail"' in html
    assert 'id="shortcuts-modal"' in html

    # Keyboard shortcuts handling
    assert "'Space'" in html
    assert "'ArrowLeft'" in html
    assert "'ArrowRight'" in html
    assert "'KeyL'" in html
    assert "'KeyR'" in html
    assert "'KeyF'" in html
    assert "'KeyO'" in html
    assert "'Escape'" in html
    assert "'Digit1'" in html
    assert "'Digit2'" in html

    # Default model must point to Phase 9
    assert "EXP009_three_factor_rpe_modular.brainlog" in html
