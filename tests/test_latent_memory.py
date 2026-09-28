r"""Tests for Phase 6 Persistent Latent Memory, Multi-Step Prediction, and Representation Analysis."""

import numpy as np
import pytest
from pathlib import Path

from brain.latent_memory import PersistentLatentModule
from brain.latent_spatial import PersistentLatentSpatialRNN
from brain.representation_analysis import (
    LinearDecodingProbe,
    RepresentationalSimilarityAnalysis,
    StateDynamicsAnalyzer,
)
from benchmarks.temporal_memory import (
    DelayedCueRecallTask,
    SequenceReconstructionTask,
    TemporalContextDisambiguationTask,
    MultiStepPredictionEvaluator,
)
from observability.events import ComputationalEvent, EventType
from observability.brainlog import BrainLog, BrainLogSerializer, BrainLogValidator
from observability.capture import StateCapture, TelemetryFrame
from observability.replay import ReplayEngine
from observability.topology import BrainTopology, TopologyExtractor


def test_latent_module_initialization_and_dynamics():
    H = 20
    D_z = 16
    S = 8
    A = 4
    lat = PersistentLatentModule(
        hidden_dim=H, latent_dim=D_z, obs_dim=S, action_dim=A, seed=42
    )

    assert lat.z.shape == (D_z,)
    assert lat.W_hz.shape == (H, D_z)
    assert lat.W_zz.shape == (D_z, D_z)
    assert lat.W_zh.shape == (D_z, H)

    # Initial state is zeros
    assert np.allclose(lat.z, 0.0)

    # Forward update with recurrent activation
    act = np.ones(H, dtype=np.float32) * 0.5
    z1, feedback1 = lat.update_latent_state(act, action=1)
    assert z1.shape == (D_z,)
    assert feedback1.shape == (H,)
    assert np.linalg.norm(z1) > 0.0

    # Multi-step prediction
    preds = lat.predict_multi_step(action=1)
    for k in [1, 2, 5, 10, 25]:
        assert k in preds
        assert preds[k].shape == (S,)
        assert np.all(preds[k] >= 0.0) and np.all(preds[k] <= 1.0)


def test_persistence_vs_decay_and_reset():
    lat = PersistentLatentModule(
        hidden_dim=10, latent_dim=10, persistence_min=0.95, persistence_max=0.98, seed=42
    )

    # Inject stimulus across 5 steps to reach steady state
    act = np.ones(10, dtype=np.float32)
    for _ in range(5):
        lat.update_latent_state(act, action=0)
    norm_after_stim = float(np.linalg.norm(lat.z))
    assert norm_after_stim > 0.1

    # Remove stimulus (zero activation) for 10 steps
    zero_act = np.zeros(10, dtype=np.float32)
    for _ in range(10):
        lat.update_latent_state(zero_act, action=0)

    # Should retain high percentage of norm due to high persistence (0.95 - 0.98)
    norm_after_10_steps = float(np.linalg.norm(lat.z))
    assert norm_after_10_steps > 0.5 * norm_after_stim

    # Now test reset_per_step
    lat.reset_per_step = True
    lat.update_latent_state(zero_act, action=0)
    # With reset and zero input, drive is only bias and prior is wiped
    assert np.linalg.norm(lat.prev_z) == 0.0


def test_persistent_latent_spatial_rnn_step():
    model = PersistentLatentSpatialRNN(
        obs_dim=8,
        action_dim=4,
        hidden_dim=30,
        latent_dim=16,
        seed=42,
    )

    obs_t = np.random.uniform(0.1, 0.9, 8).astype(np.float32)
    obs_next = np.random.uniform(0.1, 0.9, 8).astype(np.float32)
    action = model.select_action(epsilon=0.0)

    future_obs = {
        1: obs_next,
        2: np.random.uniform(0.1, 0.9, 8).astype(np.float32),
        5: np.random.uniform(0.1, 0.9, 8).astype(np.float32),
    }

    pred_next, err, step_metrics = model.step(
        obs_t=obs_t,
        action_t=action,
        obs_next=obs_next,
        future_observations=future_obs,
    )

    assert pred_next.shape == (8,)
    assert err >= 0.0
    assert "latent_z" in step_metrics
    assert step_metrics["latent_z"].shape == (16,)
    assert "multistep_preds" in step_metrics
    assert 1 in step_metrics["multistep_preds"]
    assert 2 in step_metrics["multistep_preds"]
    assert 5 in step_metrics["multistep_preds"]


def test_temporal_context_disambiguation_task():
    task = TemporalContextDisambiguationTask(obs_dim=8, seed=42)

    in1, tgt1, meta1 = task.generate_context_trial(context_id=1)
    in2, tgt2, meta2 = task.generate_context_trial(context_id=2)

    # Crucial property: step 2 observation X is identical
    assert np.allclose(in1[2], in2[2])
    # But step 2 targets Y1 and Y2 differ
    assert not np.allclose(tgt1[2], tgt2[2])

    # Run model on both histories
    model = PersistentLatentSpatialRNN(
        obs_dim=8, action_dim=4, hidden_dim=25, latent_dim=16, seed=42
    )

    # History 1
    model.reset()
    for t in range(3):
        pred, err, m1 = model.step(in1[t], action_t=0, obs_next=tgt1[t])
    z1 = m1["latent_z"]
    act1 = model.a[0].copy()

    # History 2
    model.reset()
    for t in range(3):
        pred, err, m2 = model.step(in2[t], action_t=0, obs_next=tgt2[t])
    z2 = m2["latent_z"]
    act2 = model.a[0].copy()

    # At step 2, sensory input is identical, but latent states must separate history!
    latent_dist = float(np.linalg.norm(z1 - z2))
    assert latent_dist > 0.01


def test_delayed_cue_recall_task():
    task = DelayedCueRecallTask(obs_dim=8, num_cues=4, delays=[10, 25], seed=42)
    inputs, targets, mask = task.generate_trial(cue_idx=2, delay=10)

    assert len(inputs) == 5 + 10 + 1
    assert len(targets) == len(inputs)
    assert len(mask) == len(inputs)
    assert np.sum(mask) == 1.0  # Only 1 recall probe step
    assert mask[-1] == 1.0


def test_sequence_reconstruction_task():
    task = SequenceReconstructionTask(obs_dim=8, seed=42)
    inputs, targets, mask, meta = task.generate_sequence_trial(seq_idx=0, delay=5)

    assert len(inputs) == 4 + 5 + 2
    assert np.sum(mask) == 2.0  # 2 queries


def test_multi_step_prediction_evaluator():
    evaluator = MultiStepPredictionEvaluator(horizons=[1, 2, 5])
    model = PersistentLatentSpatialRNN(obs_dim=8, hidden_dim=20, latent_dim=12, seed=42)

    T = 20
    obs = np.random.uniform(0.1, 0.9, (T, 8)).astype(np.float32)
    acts = np.random.randint(0, 4, T)

    res = evaluator.evaluate_trajectory(model, obs, acts)
    assert "horizon_1_mse_mean" in res
    assert "horizon_2_mse_mean" in res
    assert "horizon_5_mse_mean" in res


def test_representation_analysis_probes():
    # 1. Linear Decoding Probe
    N = 50
    D = 10
    X = np.random.normal(0.0, 1.0, (N, D)).astype(np.float32)
    # Target linearly dependent on X
    W_true = np.random.normal(0.0, 1.0, (D, 2)).astype(np.float32)
    Y = X @ W_true + np.random.normal(0.0, 0.01, (N, 2)).astype(np.float32)

    res = LinearDecodingProbe.train_and_evaluate(X, Y)
    assert res["r2"] > 0.8
    assert res["corr"] > 0.8

    # 2. RSA
    rdm = RepresentationalSimilarityAnalysis.compute_rdm(X[:5])
    assert rdm.shape == (5, 5)
    assert np.allclose(np.diag(rdm), 0.0)

    # 3. Autocorrelation Decay
    traj = np.cumsum(np.random.normal(0, 0.1, (100, 5)), axis=0).astype(np.float32)
    curve, decay = StateDynamicsAnalyzer.autocorrelation_decay(traj, max_lag=20)
    assert len(curve) == 20
    assert curve[0] == pytest.approx(1.0, abs=1e-3)

    # 4. Fisher Separability
    c1 = np.random.normal(1.0, 0.2, (20, 5))
    c2 = np.random.normal(-1.0, 0.2, (20, 5))
    sep = StateDynamicsAnalyzer.state_separability(c1, c2)
    assert sep["fdr"] > 5.0

    # 5. Participation Ratio
    pr = StateDynamicsAnalyzer.participation_ratio(X)
    assert 1.0 <= pr <= D


def test_brainlog_v2_4_serialization_and_replay(tmp_path: Path):
    from observability.topology import TopologyNode, TopologyEdge
    nodes = [
        TopologyNode(id=0, type="INPUT", label="I0", pos=[0.0, 0.0, 0.0], unit_index=-1),
        TopologyNode(id=1, type="RECURRENT", label="R1", pos=[1.0, 0.0, 0.0], unit_index=0),
    ]
    edges = [TopologyEdge(src=0, dst=1, weight=0.5)]
    topo = BrainTopology(topology_id="topo_v24", model_id="test_model", nodes=nodes, edges=edges)

    f0 = TelemetryFrame(
        step=0,
        time_ms=0.0,
        inputs=[0.5],
        activations=[0.0, 0.2],
        states=[0.0, 0.2],
        output=0.2,
        target=0.3,
        error=0.01,
        mask=1.0,
        dynamic_details={
            "latent_z": [0.1, -0.2, 0.3],
            "latent_norm": 0.374,
            "multistep_preds": {"1": [0.2], "2": [0.25]},
            "context_id": "CTX_A",
        },
    )
    f1 = TelemetryFrame(
        step=1,
        time_ms=40.0,
        inputs=[0.5],
        activations=[0.1, 0.3],
        states=[0.1, 0.3],
        output=0.3,
        target=0.3,
        error=0.0,
        mask=1.0,
        dynamic_details={
            "latent_z": [0.4, 0.1, 0.2],
            "latent_norm": 0.458,
            "multistep_preds": {"1": [0.3], "2": [0.31]},
            "context_id": "CTX_B",
        },
    )

    ev = ComputationalEvent(
        event_type=EventType.LATENT_STATE_UPDATED,
        step=0,
        time_ms=0.0,
        magnitude=0.374,
        payload={"latent_norm": 0.374},
    )

    save_path = tmp_path / "test_v24.brainlog"
    BrainLogSerializer.save(
        filepath=save_path,
        metadata={"experiment_id": "EXP006_TEST", "phase": "6"},
        topology=topo,
        frames=[f0, f1],
        events=[ev],
        snapshots={},
        version="2.4",
    )

    # Validate
    valid, msg = BrainLogValidator.validate(save_path)
    assert valid, f"Validation error: {msg}"

    # Replay engine inspection
    replay = ReplayEngine()
    replay.load(save_path)
    assert replay.total_steps == 2

    # Test latent state inspection
    lat_info = replay.get_latent_state(0)
    assert lat_info is not None
    assert lat_info["latent_norm"] == pytest.approx(0.374, abs=1e-3)
    assert lat_info["context_id"] == "CTX_A"

    # Test multi-step predictions inspection
    ms_info = replay.get_multistep_predictions(0)
    assert ms_info is not None
    assert "1" in ms_info["multistep_preds"]

    # Test history comparison
    comp = replay.compareHistories(0, 1)
    assert comp["inputs_identical"] is True
    assert comp["latent_state_distance"] > 0.0
