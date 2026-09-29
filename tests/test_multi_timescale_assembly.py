"""Tests for MultiTimescaleAssembly component across A0-A5 timescale modes."""

import numpy as np
import pytest

from brain.multi_timescale_assembly import (
    MultiTimescaleAssembly,
    AssemblyTimescaleMode,
)


def test_multi_timescale_assembly_initialization():
    assembly = MultiTimescaleAssembly(num_actions=2, seed=42)
    assert assembly.act_fast.shape == (2,)
    assert assembly.act_med.shape == (2,)
    assert assembly.act_slow.shape == (2,)
    assert len(assembly.action_probs) == 2


def test_multi_timescale_assembly_modes():
    drives = np.array([1.5, -0.5], dtype=np.float32)

    for mode in AssemblyTimescaleMode:
        assembly = MultiTimescaleAssembly(num_actions=2, mode=mode, seed=42)
        out = assembly.step(drives)
        assert out["selected_action"] in [0, 1]
        assert len(out["action_probs"]) == 2
        assert np.isclose(np.sum(out["action_probs"]), 1.0)
        assert 0.0 <= out["decision_confidence"] <= 1.0
        assert out["action_margin"] >= 0.0
        assert out["timescale_mode"] == mode.value


def test_timescale_persistence_and_switching():
    assembly = MultiTimescaleAssembly(
        num_actions=2,
        mode=AssemblyTimescaleMode.A5_FAST_MEDIUM_SLOW,
        seed=42,
    )

    # Step with strong evidence for Action 0
    for _ in range(20):
        out = assembly.step(np.array([2.0, -1.0]))
    assert out["selected_action"] == 0
    assert out["winner_duration_steps"] == 20

    # Flip evidence to Action 1
    for _ in range(50):
        out = assembly.step(np.array([-2.0, 2.0]))
    assert out["selected_action"] == 1
    assert out["total_switches"] >= 1
