"""Unit tests for ActionSelectionCircuit and Causal Pathway Lesioning (Phase 11)."""

import numpy as np
import pytest

from brain.action_routing import ActionSelectionCircuit


def test_action_circuit_initialization():
    circuit = ActionSelectionCircuit(num_hidden=30, num_actions=2, seed=42)
    assert circuit.num_hidden == 30
    assert circuit.num_actions == 2
    assert circuit.W_route.shape == (30, 2)
    assert circuit.v_action.shape == (2,)


def test_action_step_and_mutual_inhibition():
    circuit = ActionSelectionCircuit(num_hidden=20, num_actions=2, mutual_inhibition=0.5, seed=42)
    # Drive Action 0 strongly
    s = np.zeros(20, dtype=np.float32)
    s[:10] = 5.0  # Pool A drives Action 0

    v_act, sel_act, margin = circuit.step(s, dt=1.0)
    assert sel_act == 0
    assert margin > 0.0
    assert v_act[0] > v_act[1]


def test_action_routing_metrics():
    circuit = ActionSelectionCircuit(num_hidden=20, num_actions=2, seed=42)
    # Drive Action 0
    s = np.zeros(20, dtype=np.float32)
    s[:10] = 5.0
    circuit.step(s, dt=1.0)

    metrics = circuit.compute_routing_metrics(target_action=0, context_id=0)
    assert "action_selectivity" in metrics
    assert "winner_margin" in metrics
    assert "target_action_prob" in metrics
    assert metrics["action_selectivity"] > 0.0


def test_causal_pathway_lesioning():
    circuit = ActionSelectionCircuit(num_hidden=20, num_actions=2, seed=42)
    relevant_mask = np.array([True] * 10 + [False] * 10, dtype=bool)

    orig_w = circuit.W_route.copy()

    # 1. Lesion Relevant Pathway
    n_lesioned = circuit.lesion_relevant_pathway(relevant_mask)
    assert n_lesioned == 10 * 2
    assert np.all(circuit.W_route[:10, :] == 0.0)
    assert np.any(circuit.W_route[10:, :] != 0.0)

    # 2. Restore Lesions
    circuit.restore_lesions()
    assert np.allclose(circuit.W_route, orig_w)

    # 3. Lesion Irrelevant Pathway
    n_irrel = circuit.lesion_irrelevant_pathway(relevant_mask)
    assert n_irrel == 10 * 2
    assert np.any(circuit.W_route[:10, :] != 0.0)
