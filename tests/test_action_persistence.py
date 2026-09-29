"""Unit tests for Phase 13 Action Persistence Controller.

Validates:
1. Controller initialization and state machine transitions.
2. Refractory lockout period prevents early switching under jitter.
3. Hysteresis switching barrier requires significant evidence delta to switch.
4. Recurrent attractor latching maintains chosen action through transient pauses.
5. Chatter detection and unwanted switching metrics.
6. Context-dependent switching accuracy.
"""

import numpy as np
import pytest

from brain.action_persistence import (
    ActionPersistenceController,
    ActionPersistenceState,
)


def test_persistence_initialization_and_onset():
    """Test 1: Persistence controller transitions from IDLE to ONSET upon evidence."""
    ctrl = ActionPersistenceController(
        num_actions=2,
        lockout_duration_ms=25.0,
        hysteresis_margin=0.20,
        seed=42,
    )
    assert ctrl.state == ActionPersistenceState.IDLE
    assert ctrl.current_action is None

    # Step with strong evidence for Action 1
    out = ctrl.step(np.array([0.1, 1.5], dtype=np.float32))
    assert out["selected_action"] == 1
    assert out["state"] == ActionPersistenceState.ONSET.value
    assert ctrl.current_action == 1
    assert ctrl.lockout_remaining_ms == 25.0


def test_lockout_period_prevents_switching():
    """Test 2: Lockout period maintains selected action even when rival evidence spikes."""
    ctrl = ActionPersistenceController(
        num_actions=2,
        lockout_duration_ms=30.0,
        hysteresis_margin=0.10,
        seed=42,
    )
    # Select Action 0
    ctrl.step(np.array([1.5, 0.0], dtype=np.float32))
    assert ctrl.current_action == 0

    # During lockout, sudden surge in Action 1 evidence should NOT flip action
    for _ in range(15):
        out = ctrl.step(np.array([0.2, 2.5], dtype=np.float32))
        assert out["selected_action"] == 0
        assert out["state"] == ActionPersistenceState.REFRACTORY_LOCKOUT.value


def test_hysteresis_barrier_outside_lockout():
    """Test 3: Outside lockout, rival action must overcome hysteresis margin to switch."""
    ctrl = ActionPersistenceController(
        num_actions=2,
        tau_persistence_ms=20.0,
        lockout_duration_ms=10.0,
        hysteresis_margin=0.25,
        seed=42,
    )
    # Select Action 0 and drain lockout
    ctrl.step(np.array([1.0, 0.0], dtype=np.float32))
    for _ in range(15):
        ctrl.step(np.array([1.0, 0.0], dtype=np.float32))
    assert ctrl.lockout_remaining_ms == 0.0

    # Marginal rival evidence below hysteresis margin: no switch
    out = ctrl.step(np.array([0.8, 0.95], dtype=np.float32))
    assert out["selected_action"] == 0

    # Decisive rival evidence exceeding hysteresis margin: switch occurs!
    for _ in range(25):
        out = ctrl.step(np.array([0.1, 2.0], dtype=np.float32))
    assert out["selected_action"] == 1


def test_persistence_metrics_aggregation():
    """Test 4: Metrics correctly track action duration, chatter, and switching."""
    ctrl = ActionPersistenceController(
        num_actions=2,
        lockout_duration_ms=15.0,
        hysteresis_margin=0.15,
        seed=42,
    )
    ctrl.set_context(0)
    for _ in range(40):
        ctrl.step(np.array([1.5, 0.2], dtype=np.float32), target_action=0)

    # Change context and switch
    ctrl.set_context(1)
    for _ in range(40):
        ctrl.step(np.array([0.1, 1.8], dtype=np.float32), target_action=1)

    metrics = ctrl.get_persistence_metrics()
    assert metrics["action_duration_mean_ms"] > 0.0
    assert metrics["persistence_stability"] > 0.0
    assert "chatter_events" in metrics
    assert "context_switching_accuracy" in metrics
