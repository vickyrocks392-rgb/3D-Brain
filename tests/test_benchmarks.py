"""Tests for Delayed Match-to-Sample benchmark generator."""

import numpy as np
import pytest
from benchmarks.delayed_match import DelayedMatchTask, TaskConfig


def test_deterministic_task_generation():
    """Verify that identical seeds produce bit-exact identical trial sequences."""
    c1 = TaskConfig(t_cue=5, t_delay=20, t_probe=5, seed=42)
    c2 = TaskConfig(t_cue=5, t_delay=20, t_probe=5, seed=42)

    t1 = DelayedMatchTask(c1)
    t2 = DelayedMatchTask(c2)

    X1, Y1, m1, _ = t1.generate_dataset(10, seed=123)
    X2, Y2, m2, _ = t2.generate_dataset(10, seed=123)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(Y1, Y2)
    np.testing.assert_array_equal(m1, m2)


def test_different_seeds_produce_different_trials():
    """Verify that different seeds produce statistically distinct sequences."""
    c = TaskConfig(t_cue=5, t_delay=20, t_probe=5, seed=42)
    task = DelayedMatchTask(c)

    X1, _, _, _ = task.generate_dataset(10, seed=100)
    X2, _, _, _ = task.generate_dataset(10, seed=200)

    assert not np.array_equal(X1, X2)


def test_mask_and_target_semantics():
    """Verify mask is 1.0 strictly during probe and 0.0 during cue and delay."""
    c = TaskConfig(t_cue=5, t_delay=15, t_probe=5, distractor_sigma=0.0)
    task = DelayedMatchTask(c)

    X, Y, mask, meta = task.generate_dataset(20, seed=42)

    # Total steps = 5 + 15 + 5 = 25
    assert X.shape[1] == 25
    assert Y.shape[1] == 25
    assert mask.shape[1] == 25

    # Mask must be 0 for t in [0, 20) and 1 for t in [20, 25)
    np.testing.assert_array_equal(mask[:, 0:20, :], 0.0)
    np.testing.assert_array_equal(mask[:, 20:25, :], 1.0)

    # Targets must be 0 during cue and delay
    np.testing.assert_array_equal(Y[:, 0:20, :], 0.0)

    # During probe, targets must match the metadata target (+1.0 for match, -1.0 for mismatch)
    for i in range(20):
        expected_target = meta[i]["target"]
        assert expected_target in [1.0, -1.0]
        np.testing.assert_array_equal(Y[i, 20:25, 0], expected_target)
