"""Unit tests for Phase 20 Track C: Integrated Architecture, BrainLog v3.6, and Frozen Evaluation."""

import numpy as np
import pytest

from brain.phase16_network import Phase16Network, ArchitecturePreset
from brain.modular_routing import RoutingCondition
from brain.hippocampal_replay import HippocampalReplayBuffer, ReplayCondition
from brain.synaptic_consolidation import ConsolidationMode
from benchmarks.phase20_integrated import run_integrated_continual_experiment
from observability.events import ComputationalEvent, EventType
from observability.topology import BrainTopology, TopologyNode, TopologyEdge
from observability.capture import TelemetryFrame
from observability.brainlog import BrainLog, BrainLogValidator


def test_integrated_c4_components_active():
    """Verify that full integrated system C4 has all three subcomponents active."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=42)
    net.set_routing_mode(RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY, num_branches=3)
    net.set_consolidation_mode(ConsolidationMode.DUAL_TIMESCALE)
    buffer = HippocampalReplayBuffer(capacity=64, replay_budget=4, seed=42)
    net.set_replay_buffer(buffer, condition=ReplayCondition.B5_PRIORITIZED_CONSOLIDATION)

    assert net.action_readout.modular_readout is not None
    assert net.action_readout.routing_condition == RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY
    assert net.action_readout.consolidation_module.mode == ConsolidationMode.DUAL_TIMESCALE
    assert net.replay_buffer is not None
    assert net.replay_condition == ReplayCondition.B5_PRIORITIZED_CONSOLIDATION


def test_frozen_evaluation_zero_plasticity():
    """Verify that under frozen evaluation, no weights or consolidation states update."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=42)
    net.set_routing_mode(RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY, num_branches=3)
    net.set_consolidation_mode(ConsolidationMode.DUAL_TIMESCALE)
    net.set_frozen(True)

    w_before = [b.weights.copy() for b in net.action_readout.modular_readout.branches]
    x = np.ones(8, dtype=np.float32)

    # Step network with reward feedback
    for _ in range(10):
        net.step(x, target_action=0, reward=1.0, is_choice_step=True, is_reward_step=True)

    w_after = [b.weights.copy() for b in net.action_readout.modular_readout.branches]
    for b_bef, b_aft in zip(w_before, w_after):
        assert np.allclose(b_bef, b_aft), "Weights must remain strictly invariant during frozen evaluation"


def test_brainlog_v36_schema_validation(tmp_path):
    """Verify that a generated BrainLog v3.6 trace conforms to strict schema validation."""
    nodes = [
        TopologyNode(id=0, label="in_0", type="INPUT", pos=[0, 0, 0], unit_index=-1),
        TopologyNode(id=1, label="h_0", type="RECURRENT", pos=[10, 0, 0], unit_index=0),
        TopologyNode(id=2, label="act_0", type="OUTPUT", pos=[20, 0, 0], unit_index=-1),
    ]
    edges = [TopologyEdge(src=0, dst=1, weight=0.5), TopologyEdge(src=1, dst=2, weight=0.5)]
    topo = BrainTopology(topology_id="topo_v36", model_id="min_e_modular", nodes=nodes, edges=edges)

    ev = ComputationalEvent(
        event_type=EventType.ROUTING_GATE_COMPUTED,
        step=0,
        time_ms=0.0,
        payload={"distribution": [1.0, 0.0, 0.0]},
    )
    frame = TelemetryFrame(
        step=0,
        time_ms=0.0,
        inputs=[1.0],
        activations=[1.0, 0.5, 0.2],
        states=[1.0, 0.5, 0.2],
        output=0.0,
        target=0.0,
        error=0.0,
        mask=1.0,
        events=[ev],
        dynamic_details={"routing": {"active_branch": 0}},
    )

    blog = BrainLog(
        metadata={"phase": 20, "experiment": "EXP020_test"},
        topology=topo,
        frames=[frame],
        events=[ev],
        snapshots={},
        version="3.6",
    )

    bfile = tmp_path / "test_v36.brainlog"
    blog.save(bfile)

    valid, reason = BrainLogValidator.validate(bfile)
    assert valid, f"BrainLog v3.6 failed validation: {reason}"
