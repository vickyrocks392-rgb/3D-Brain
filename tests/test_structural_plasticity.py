"""Tests for Dynamic Structural Plasticity and Brain Growth in Phase 5."""

import numpy as np
import pytest

from brain.structural_plasticity import StructuralPlasticity


def test_structural_plasticity_pruning():
    H = 15
    struct = StructuralPlasticity(
        update_interval=5,
        prune_threshold=0.1,
        prune_fraction=0.2,
        min_degree=1,
        max_degree=10,
        enable_growth=False,
        seed=42,
    )
    struct.reset(H)

    # Fully connected initial mask
    adj = np.ones((H, H), dtype=bool)
    np.fill_diagonal(adj, False)
    W_hh = np.ones((H, H), dtype=np.float32) * 0.5
    # Set a few specific edges to very low weight below threshold
    W_hh[0, 1] = 0.02
    W_hh[1, 2] = 0.01

    pos = np.zeros((H, 3), dtype=np.float32)
    dist = np.ones((H, H), dtype=np.float32) * 2.0
    delay = np.ones((H, H), dtype=np.int32)

    W_new, adj_new, _, _, _, events = struct.step(
        step=5,
        W_hh=W_hh,
        adjacency_mask=adj,
        positions=pos,
        dist_matrix=dist,
        delay_matrix=delay,
    )

    # Some edges should be pruned
    pruned_events = [e for e in events if e.event_type == "SYNAPSE_PRUNED"]
    assert len(pruned_events) > 0
    # Pruned edge weights should be zero in W_new
    for pe in pruned_events:
        assert adj_new[pe.src, pe.dst] == False
        assert W_new[pe.src, pe.dst] == 0.0


def test_structural_plasticity_formation():
    H = 12
    struct = StructuralPlasticity(
        update_interval=5,
        form_fraction=0.2,
        min_degree=1,
        max_degree=10,
        max_edges=100,
        enable_growth=False,
        seed=42,
    )
    struct.reset(H)

    # Sparse initial mask
    adj = np.eye(H, dtype=bool)
    W_hh = np.zeros((H, H), dtype=np.float32)
    pos = np.random.default_rng(42).uniform(-5.0, 5.0, (H, 3)).astype(np.float32)
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    delay = np.maximum(1, np.round(dist / 5.0)).astype(np.int32)

    # Record co-activation between units 2 and 3
    act = np.zeros(H, dtype=np.float32)
    act[2] = 0.9
    act[3] = 0.9
    struct.record_activity(act)

    W_new, adj_new, _, _, _, events = struct.step(
        step=5,
        W_hh=W_hh,
        adjacency_mask=adj,
        positions=pos,
        dist_matrix=dist,
        delay_matrix=delay,
    )

    formed_events = [e for e in events if e.event_type == "SYNAPSE_FORMED"]
    assert len(formed_events) > 0
    for fe in formed_events:
        assert adj_new[fe.src, fe.dst] == True
        assert W_new[fe.src, fe.dst] != 0.0


def test_degree_constraints():
    H = 10
    min_k = 2
    max_k = 4
    struct = StructuralPlasticity(
        update_interval=5,
        prune_threshold=1.0,  # Aggressive pruning
        prune_fraction=0.8,
        min_degree=min_k,
        max_degree=max_k,
        enable_growth=False,
        seed=42,
    )
    struct.reset(H)

    adj = np.ones((H, H), dtype=bool)
    np.fill_diagonal(adj, False)
    W_hh = np.ones((H, H), dtype=np.float32) * 0.01  # All below threshold
    pos = np.zeros((H, 3), dtype=np.float32)
    dist = np.ones((H, H), dtype=np.float32)
    delay = np.ones((H, H), dtype=np.int32)

    _, adj_new, _, _, _, _ = struct.step(
        step=5,
        W_hh=W_hh,
        adjacency_mask=adj,
        positions=pos,
        dist_matrix=dist,
        delay_matrix=delay,
    )

    # In-degree and out-degree should not drop below min_degree
    in_degrees = np.sum(adj_new, axis=0)
    out_degrees = np.sum(adj_new, axis=1)
    assert np.all(in_degrees >= min_k)
    assert np.all(out_degrees >= min_k)


def test_dynamic_neuron_growth():
    H = 8
    struct = StructuralPlasticity(
        update_interval=5,
        enable_growth=True,
        growth_threshold=0.20,
        growth_cooldown=5,
        max_neurons=12,
        seed=42,
    )
    struct.reset(H)

    adj = np.ones((H, H), dtype=bool)
    np.fill_diagonal(adj, False)
    W_hh = np.ones((H, H), dtype=np.float32) * 0.5
    pos = np.zeros((H, 3), dtype=np.float32)
    dist = np.ones((H, H), dtype=np.float32)
    delay = np.ones((H, H), dtype=np.int32)

    # Step with high error exceeding growth_threshold
    W_new, adj_new, pos_new, dist_new, delay_new, events = struct.step(
        step=5,
        W_hh=W_hh,
        adjacency_mask=adj,
        positions=pos,
        dist_matrix=dist,
        delay_matrix=delay,
        mean_recent_error=0.45,
    )

    birth_events = [e for e in events if e.event_type == "NEURON_BORN"]
    assert len(birth_events) == 1
    assert W_new.shape[0] == H + 1
    assert pos_new.shape[0] == H + 1
    assert dist_new.shape[0] == H + 1
