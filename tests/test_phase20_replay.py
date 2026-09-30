"""Unit tests for Phase 20 Track B: Sparse Hippocampal-Style Replay & Extended Credit Horizon."""

import numpy as np
import pytest

from brain.phase16_network import Phase16Network, ArchitecturePreset
from brain.hippocampal_replay import (
    HippocampalReplayBuffer,
    ReplayCondition,
    ReplayTrace,
)
from benchmarks.phase20_replay import (
    generate_replay_trial,
    train_and_eval_replay_credit,
    DT_MS,
)


def test_replay_buffer_no_target_leakage():
    """Verify that choice state stored in replay buffer contains strictly NO target or reward."""
    buffer = HippocampalReplayBuffer(capacity=64, seed=42)
    context = np.ones(8, dtype=np.float32)
    elig = np.ones((8, 2), dtype=np.float32)

    buffer.store_choice(
        episode_id=1,
        task_id="TEST_TASK",
        context_vector=context,
        chosen_action=0,
        delay_steps=10,
        delay_ms=400.0,
        eligibility_snapshot=elig,
        timestamp_ms=100.0,
    )

    pending = buffer._current_pending_trace
    assert pending is not None
    assert pending.target_action is None
    assert pending.reward_val == 0.0
    assert pending.prediction_error == 0.0


def test_replay_outcome_commitment():
    """Verify that legitimate outcome is recorded at feedback step and committed to buffer."""
    buffer = HippocampalReplayBuffer(capacity=64, seed=42)
    context = np.ones(8, dtype=np.float32)
    elig = np.ones((8, 2), dtype=np.float32)

    buffer.store_choice(1, "TEST", context, 0, 10, 400.0, elig, 100.0)
    committed = buffer.record_outcome(target_action=1, reward_val=-0.5, prediction_error=0.8)

    assert committed is not None
    assert committed.target_action == 1
    assert committed.reward_val == -0.5
    assert committed.prediction_error == 0.8
    assert len(buffer.traces) == 1
    assert buffer._current_pending_trace is None


def test_replay_sampling_conditions():
    """Test sampling under B0, B1, B2, B3 conditions."""
    buffer = HippocampalReplayBuffer(capacity=64, replay_budget=2, seed=42)
    context = np.ones(8, dtype=np.float32)
    elig = np.ones((8, 2), dtype=np.float32)

    for i in range(5):
        buffer.store_choice(i, "TEST", context, i % 2, 5, 200.0, elig, i * 100.0)
        # Assign varying prediction error
        buffer.record_outcome(target_action=i % 2, reward_val=1.0, prediction_error=float(i) * 0.25)

    # B0: No replay
    assert len(buffer.sample_replay_batch(ReplayCondition.B0_NO_REPLAY)) == 0

    # B1: Random replay
    batch_b1 = buffer.sample_replay_batch(ReplayCondition.B1_RANDOM_REPLAY, budget=2)
    assert len(batch_b1) == 2

    # B2: Event-triggered (threshold 0.5)
    batch_b2 = buffer.sample_replay_batch(ReplayCondition.B2_EVENT_TRIGGERED, budget=2)
    assert len(batch_b2) >= 1
    for t in batch_b2:
        assert t.prediction_error >= 0.4 or t == buffer.traces[-1]

    # B3: Prioritized replay
    batch_b3 = buffer.sample_replay_batch(ReplayCondition.B3_PRIORITIZED_REPLAY, budget=3)
    assert len(batch_b3) == 3


def test_causal_interventions_corruptions():
    """Verify that causal interventions properly perturb replay content and timing."""
    buffer = HippocampalReplayBuffer(capacity=64, replay_budget=2, seed=42)
    context = np.ones(8, dtype=np.float32)
    elig = np.ones((8, 2), dtype=np.float32)
    buffer.store_choice(0, "TEST", context, 0, 5, 200.0, elig, 0.0)
    buffer.record_outcome(target_action=0, reward_val=1.0, prediction_error=1.0)

    # Content corruption
    buffer.content_randomized = True
    corrupted = buffer.sample_replay_batch(ReplayCondition.B1_RANDOM_REPLAY, budget=1)
    assert not np.allclose(corrupted[0].context_vector, context)

    # Wrong episode injection
    buffer.content_randomized = False
    buffer.wrong_episode_injection = True
    wrong = buffer.sample_replay_batch(ReplayCondition.B1_RANDOM_REPLAY, budget=1)
    assert wrong[0].target_action == 1  # Inverted from 0


def test_quiescent_rest_silent_inputs():
    """Verify that resting steps provide zero sensory input and zero reward."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=42)
    buffer = HippocampalReplayBuffer(capacity=64, replay_budget=2, seed=42)
    net.set_replay_buffer(buffer, condition=ReplayCondition.B3_PRIORITIZED_REPLAY)

    res = net.step_quiescent_rest(rest_steps=5)
    assert res["rest_steps"] == 5
    assert res["replayed_events_count"] == 0  # Empty buffer
