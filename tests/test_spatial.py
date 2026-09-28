"""Unit and integration tests for Phase 4: Spatialized 3D Topology, Propagation Delays,

and Parameter Scaling.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from brain.plasticity import FrozenPlasticity, ModulatedPlasticity
from brain.spatial import (
    InTransitSignal,
    SignalPropagationQueue,
    SpatialConnectivityBuilder,
    SpatialGeometry,
    SpatialPlasticRNN,
    TopologyFamily,
)
from brain.spatial_analysis import (
    compute_clustering_and_path_length,
    compute_functional_modularity,
    compute_participation_coefficient,
    compute_phase_synchronization,
    compute_structural_modularity,
    compute_within_and_cross_metrics,
)
from benchmarks.spatial_tasks import SpatialDistributedTemporalTask, SpatialTaskConfig
from observability.brainlog import BrainLog, BrainLogSerializer, BrainLogValidator
from observability.capture import StateCapture, TelemetryFrame, TelemetrySignal
from observability.events import ComputationalEvent, EventType
from observability.replay import ReplayEngine
from observability.topology import TopologyExtractor


# ==============================================================================
# 1. 3D Geometry and Distance Calculation Tests
# ==============================================================================

def test_deterministic_spatial_geometry_placement():
    """Verify that 3D geometric placement generators are strictly deterministic under seed."""
    pos1, mod1 = SpatialGeometry.modular_spatial(n=64, num_modules=4, seed=123)
    pos2, mod2 = SpatialGeometry.modular_spatial(n=64, num_modules=4, seed=123)
    pos3, mod3 = SpatialGeometry.modular_spatial(n=64, num_modules=4, seed=999)

    assert np.allclose(pos1, pos2), "Identical seed must produce identical positions"
    assert np.array_equal(mod1, mod2), "Identical seed must produce identical module assignments"
    assert not np.allclose(pos1, pos3), "Different seeds must produce different positions"
    assert pos1.shape == (64, 3)
    assert len(np.unique(mod1)) == 4


def test_distance_matrix_metric_properties():
    """Verify that distance matrix satisfies metric axioms (zero diagonal, symmetry, triangle ineq)."""
    positions, _ = SpatialGeometry.uniform_sphere(n=30, radius=10.0, seed=42)
    dist = SpatialConnectivityBuilder.compute_distance_matrix(positions)

    # Diagonal is zero
    assert np.allclose(np.diag(dist), 0.0)
    # Symmetry
    assert np.allclose(dist, dist.T)
    # Non-negativity
    assert np.all(dist >= 0.0)
    # Triangle inequality d(i, k) <= d(i, j) + d(j, k)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                assert dist[i, k] <= dist[i, j] + dist[j, k] + 1e-5


# ==============================================================================
# 2. Connectivity Builders and Control Topologies
# ==============================================================================

@pytest.mark.parametrize("topo_type", [
    TopologyFamily.RANDOM,
    TopologyFamily.LOCAL,
    TopologyFamily.MODULAR,
    TopologyFamily.SMALL_WORLD,
])
def test_topology_builder_exact_connection_budget(topo_type):
    """Verify that all topology families strictly conform to target edge count without confounds."""
    N = 40
    target_edges = 200
    positions, modules = SpatialGeometry.modular_spatial(N, num_modules=4, seed=42)
    adj, dist, delays = SpatialConnectivityBuilder.build_topology(
        positions=positions,
        module_ids=modules,
        topology_type=topo_type,
        target_edges=target_edges,
        propagation_speed=5.0,
        seed=42,
    )

    actual_edges = int(np.sum(adj))
    assert actual_edges == target_edges, f"{topo_type} generated {actual_edges} edges, expected {target_edges}"
    assert np.all(np.diag(adj) == False), "No self-loops permitted"
    assert np.all(delays[adj] >= 1), "All active connections must have propagation delay >= 1"


def test_modular_vs_random_structural_modularity():
    """Verify that Modular topology has significantly higher structural modularity Q than Random."""
    N = 48
    target_edges = 240
    positions, modules = SpatialGeometry.modular_spatial(N, num_modules=4, seed=42)

    adj_mod, _, _ = SpatialConnectivityBuilder.build_topology(
        positions, modules, TopologyFamily.MODULAR, target_edges, seed=42, inter_module_ratio=0.08
    )
    adj_rand, _, _ = SpatialConnectivityBuilder.build_topology(
        positions, modules, TopologyFamily.RANDOM, target_edges, seed=42
    )

    Q_mod = compute_structural_modularity(adj_mod, modules)
    Q_rand = compute_structural_modularity(adj_rand, modules)

    assert Q_mod > Q_rand, f"Modular Q ({Q_mod:.3f}) must exceed Random Q ({Q_rand:.3f})"
    assert Q_mod > 0.40, f"Modular network must have strong community structure Q > 0.40, got {Q_mod:.3f}"


def test_small_world_clustering_and_path_length():
    """Verify that Small-World topology maintains high clustering relative to random graphs."""
    N = 50
    target_edges = 300
    positions, modules = SpatialGeometry.uniform_sphere(N, radius=10.0, seed=42)

    adj_sw, _, _ = SpatialConnectivityBuilder.build_topology(
        positions, modules, TopologyFamily.SMALL_WORLD, target_edges, seed=42, rewiring_prob=0.10
    )
    adj_rand, _, _ = SpatialConnectivityBuilder.build_topology(
        positions, modules, TopologyFamily.RANDOM, target_edges, seed=42
    )

    metrics_sw = compute_clustering_and_path_length(adj_sw)
    metrics_rand = compute_clustering_and_path_length(adj_rand)

    assert metrics_sw["clustering_coefficient"] > metrics_rand["clustering_coefficient"], (
        f"Small-World clustering ({metrics_sw['clustering_coefficient']}) must exceed Random ({metrics_rand['clustering_coefficient']})"
    )


# ==============================================================================
# 3. Finite Signal Propagation Queue Correctness
# ==============================================================================

def test_propagation_queue_temporal_correctness():
    """Verify that SignalPropagationQueue delivers signals at exactly step + delay."""
    num_neurons = 5
    queue = SignalPropagationQueue(num_neurons=num_neurons, max_delay=10, track_in_transit=True)

    # At step 0: neuron 0 sends signal to neuron 2 with delay = 3, weight = 1.0, act = 2.0
    srcs = np.array([0], dtype=np.int32)
    dsts = np.array([2], dtype=np.int32)
    weights = np.array([1.0], dtype=np.float32)
    acts = np.array([2.0], dtype=np.float32)
    delays = np.array([3], dtype=np.int32)

    queue.enqueue_signals(srcs, dsts, weights, acts, delays)

    # Step 0 advance: delivered input should be 0.0
    arr0 = queue.advance_step()
    assert np.allclose(arr0, 0.0)

    # Step 1 advance
    arr1 = queue.advance_step()
    assert np.allclose(arr1, 0.0)

    # Step 2 advance
    arr2 = queue.advance_step()
    assert np.allclose(arr2, 0.0)

    # Step 3 advance: signal must arrive at unit 2 with magnitude 2.0
    arr3 = queue.advance_step()
    assert arr3[2] == 2.0
    assert arr3[0] == 0.0

    # Step 4 advance: slot cleared
    arr4 = queue.advance_step()
    assert np.allclose(arr4, 0.0)


def test_simultaneous_and_overlapping_arrivals():
    """Verify that multiple simultaneous signals arriving from different paths sum linearly."""
    queue = SignalPropagationQueue(num_neurons=4, max_delay=10, track_in_transit=True)

    # Two signals targeting unit 3 at delay 2:
    # 0 -> 3: 0.5 * 1.0 = 0.5
    # 1 -> 3: 1.5 * 1.0 = 1.5
    srcs = np.array([0, 1], dtype=np.int32)
    dsts = np.array([3, 3], dtype=np.int32)
    weights = np.array([1.0, 1.0], dtype=np.float32)
    acts = np.array([0.5, 1.5], dtype=np.float32)
    delays = np.array([2, 2], dtype=np.int32)

    queue.enqueue_signals(srcs, dsts, weights, acts, delays)

    queue.advance_step()  # step 0
    queue.advance_step()  # step 1
    arr2 = queue.advance_step()  # step 2
    assert np.isclose(arr2[3], 2.0)


# ==============================================================================
# 4. Spatial Substrate Forward Pass and Plasticity
# ==============================================================================

def test_spatial_plastic_rnn_forward_pass_and_shape_invariants():
    """Verify that SpatialPlasticRNN produces valid dimensions and state invariants."""
    model = SpatialPlasticRNN(
        input_dim=2,
        hidden_dim=20,
        output_dim=1,
        topology_type=TopologyFamily.LOCAL,
        target_edges=80,
        propagation_speed=4.0,
        seed=42,
    )

    X = np.random.default_rng(42).normal(0.0, 1.0, (1, 15, 2)).astype(np.float32)
    Y, H, A = model.forward_sequence(X)

    assert Y.shape == (1, 15, 1)
    assert H.shape == (1, 15, 20)
    assert A.shape == (1, 15, 20)
    assert np.all(np.abs(A) <= 1.0), "tanh activation must be in [-1, 1]"
    assert model.parameter_count() > 0


def test_spatial_plasticity_respects_sparse_topology():
    """Verify that online plasticity ONLY updates active structural connections and preserves zero edges."""
    rule = ModulatedPlasticity(lr=0.05)
    model = SpatialPlasticRNN(
        input_dim=2,
        hidden_dim=15,
        output_dim=1,
        topology_type=TopologyFamily.LOCAL,
        target_edges=40,
        plasticity_rule=rule,
        seed=42,
    )

    initial_W = model.W_hh.copy()
    zero_mask = (model.adjacency_mask == False)
    assert np.all(initial_W[zero_mask] == 0.0), "Non-edges must start at zero"

    # Step through 20 cycles with active modulation
    rng = np.random.default_rng(42)
    for t in range(20):
        x = rng.normal(0.0, 1.0, (1, 2)).astype(np.float32)
        model.step(x, modulation=1.0)

    updated_W = model.W_hh.copy()
    # Structural zeros MUST remain zero
    assert np.all(updated_W[zero_mask] == 0.0), "Online plasticity must never create edges outside structural topology"
    # Active edges should have changed
    assert not np.allclose(updated_W[model.adjacency_mask], initial_W[model.adjacency_mask])


# ==============================================================================
# 5. Analysis Metrics: Modularity, Synchrony, and Participation
# ==============================================================================

def test_functional_modularity_and_phase_synchronization():
    """Verify functional modularity Q_func and Kuramoto order parameter computation."""
    T, N = 50, 16
    modules = np.array([0] * 8 + [1] * 8, dtype=np.int32)

    # Create synthetic modular time series: units within module 0 correlated, module 1 correlated
    rng = np.random.default_rng(42)
    t_axis = np.linspace(0, 4 * np.pi, T)[:, np.newaxis]
    m0_signal = np.sin(t_axis) + rng.normal(0.0, 0.1, (T, 8))
    m1_signal = np.cos(t_axis) + rng.normal(0.0, 0.1, (T, 8))
    acts = np.hstack([m0_signal, m1_signal]).astype(np.float32)

    Q_func, corr = compute_functional_modularity(acts, modules)
    assert Q_func > 0.25, f"Expected positive modularity for synchronized modules, got {Q_func:.3f}"

    mean_R, R_t = compute_phase_synchronization(acts)
    assert 0.0 <= mean_R <= 1.0
    assert len(R_t) == T

    metrics = compute_within_and_cross_metrics(np.ones((N, N)), corr, modules)
    assert metrics["mean_within_correlation"] > metrics["mean_cross_correlation"]


# ==============================================================================
# 6. Observability Engine & Replay Extension (.brainlog v2.2)
# ==============================================================================

def test_brainlog_v2_2_roundtrip_and_spatial_replay():
    """Verify backward-compatible .brainlog v2.2 serialization and spatial ReplayEngine inspection."""
    model = SpatialPlasticRNN(
        input_dim=2,
        hidden_dim=10,
        output_dim=1,
        topology_type=TopologyFamily.MODULAR,
        target_edges=30,
        propagation_speed=3.0,
        seed=42,
    )

    extractor = TopologyExtractor()
    topology = extractor.extract(model, model_id="SPATIAL_TEST")
    capture = StateCapture(topology, dt_ms=40.0)

    # Capture 5 steps
    rng = np.random.default_rng(42)
    for s in range(5):
        x = rng.normal(0.0, 1.0, (2,)).astype(np.float32)
        y = float(model.step(x.reshape(1, 2))[0, 0])
        capture.capture_step(
            step=s,
            input_vec=x,
            model=model,
            y_val=y,
            target_val=1.0,
            mask_val=1.0 if s == 4 else 0.0,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_spatial.brainlog"
        BrainLogSerializer.save(
            filepath=log_path,
            metadata={"experiment": "EXP004_TEST", "rule": "Frozen"},
            topology=topology,
            frames=capture.frames,
            events=capture.all_events,
            snapshots=capture.snapshots,
            version="2.2",
        )

        # Validate file
        valid, msg = BrainLogValidator.validate(log_path)
        assert valid, f"Validation failed: {msg}"

        # Load in ReplayEngine
        replay = ReplayEngine()
        replay.load(log_path)
        assert replay.total_steps == 5

        # Test node inspection includes module_id
        node_insp = replay.inspect_node(node_id=2)
        assert hasattr(node_insp, "module_id")

        # Test edge inspection includes distance, delay, progress
        recurrent_edge = next(e for e in topology.edges if e.src >= 2 and e.dst >= 2)
        edge_insp = replay.inspect_edge(recurrent_edge.src, recurrent_edge.dst)
        assert edge_insp is not None
        assert edge_insp.distance is not None
        assert edge_insp.propagation_delay >= 1

        # Test signals in transit
        signals = replay.get_signals_in_transit(0)
        assert isinstance(signals, list)


# ==============================================================================
# 7. Task C: Spatially Distributed Temporal Task Determinism
# ==============================================================================

def test_spatial_distributed_task_determinism():
    """Verify Task C generation is deterministic and structurally valid."""
    cfg = SpatialTaskConfig(cue1_steps=4, inter_cue_steps=3, cue2_steps=4, delay_steps=6, probe_steps=4)
    task = SpatialDistributedTemporalTask(cfg)

    X1, Y1, M1, meta1 = task.generate_dataset(num_trials=5, seed=123)
    X2, Y2, M2, meta2 = task.generate_dataset(num_trials=5, seed=123)

    assert np.allclose(X1, X2)
    assert np.allclose(Y1, Y2)
    assert np.allclose(M1, M2)
    assert X1.shape == (5, cfg.total_steps, 3)
    assert Y1.shape == (5, cfg.total_steps, 1)

    # Check XOR / Equivalence logic: sign(cue1) == sign(cue2) => +1.0 else -1.0
    for i in range(5):
        c1 = meta1[i]["cue1"]
        c2 = meta1[i]["cue2"]
        target = meta1[i]["target"]
        expected = 1.0 if (c1 == c2) else -1.0
        assert target == expected
