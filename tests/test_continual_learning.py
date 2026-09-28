"""Unit tests for Continual Delayed Association benchmark generator."""

import numpy as np
import pytest

from benchmarks.continual_adaptation import (
    ContinualCondition,
    ContinualConfig,
    ContinualDelayedAssociationTask,
)


@pytest.mark.parametrize("condition,expected_trials", [
    (ContinualCondition.STATIONARY, 100),
    (ContinualCondition.ABRUPT_SHIFT, 100),
    (ContinualCondition.RECURRING, 120),
    (ContinualCondition.MULTIPLE, 150),
    (ContinualCondition.GRADUAL_DRIFT, 100),
])
def test_continual_conditions_trial_counts(condition, expected_trials):
    cfg = ContinualConfig(condition=condition, t_cue=3, t_delay=5, t_probe=3, seed=42)
    task = ContinualDelayedAssociationTask(cfg)
    X, Y, mask, meta = task.generate_sequence()

    assert X.shape[0] == expected_trials
    assert Y.shape[0] == expected_trials
    assert mask.shape[0] == expected_trials
    assert len(meta) == expected_trials
    assert X.shape[1] == 11  # 3 + 5 + 3


def test_continual_task_determinism():
    cfg1 = ContinualConfig(condition=ContinualCondition.RECURRING, seed=137)
    task1 = ContinualDelayedAssociationTask(cfg1)
    X1, Y1, _, _ = task1.generate_sequence(seed=137)

    cfg2 = ContinualConfig(condition=ContinualCondition.RECURRING, seed=137)
    task2 = ContinualDelayedAssociationTask(cfg2)
    X2, Y2, _, _ = task2.generate_sequence(seed=137)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(Y1, Y2)


def test_environment_switch_flags():
    cfg = ContinualConfig(condition=ContinualCondition.RECURRING, seed=42)
    task = ContinualDelayedAssociationTask(cfg)
    _, _, _, meta = task.generate_sequence()

    # In RECURRING: Env A (40 trials), Env B (40 trials), Env A (40 trials)
    # Switches should happen at trial 40 and trial 80
    assert meta[0]["env_switch"] == 0.0
    assert meta[40]["env_switch"] == 1.0
    assert meta[40]["env_name"] == "ENV_B"
    assert meta[80]["env_switch"] == 1.0
    assert meta[80]["env_name"] == "ENV_A"
