"""Tests for MinimalEmbodiedEnvironment in Phase 5."""

import numpy as np
import pytest

from benchmarks.embodied_environment import Action, MinimalEmbodiedEnvironment


def test_environment_initialization():
    env = MinimalEmbodiedEnvironment(arena_size=10.0, seed=42)
    obs = env.reset()
    assert obs.shape == (8,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert len(env.landmarks) == 4


def test_deterministic_action_transitions():
    env = MinimalEmbodiedEnvironment(arena_size=10.0, step_size=0.5, seed=42)
    env.reset(initial_pos=np.array([5.0, 5.0]))

    # Move East: x should increase
    obs, info = env.step(Action.MOVE_EAST)
    assert info["agent_pos"][0] == pytest.approx(5.5, abs=1e-3)
    assert info["agent_pos"][1] == pytest.approx(5.0, abs=1e-3)

    # Move North: y should increase
    obs, info = env.step(Action.MOVE_NORTH)
    assert info["agent_pos"][0] == pytest.approx(5.5, abs=1e-3)
    assert info["agent_pos"][1] == pytest.approx(5.5, abs=1e-3)

    # Move West: x should decrease
    obs, info = env.step(Action.MOVE_WEST)
    assert info["agent_pos"][0] == pytest.approx(5.0, abs=1e-3)
    assert info["agent_pos"][1] == pytest.approx(5.5, abs=1e-3)

    # Move South: y should decrease
    obs, info = env.step(Action.MOVE_SOUTH)
    assert info["agent_pos"][0] == pytest.approx(5.0, abs=1e-3)
    assert info["agent_pos"][1] == pytest.approx(5.0, abs=1e-3)


def test_boundary_clamping():
    env = MinimalEmbodiedEnvironment(arena_size=10.0, step_size=1.0, seed=42)
    env.reset(initial_pos=np.array([9.5, 9.5]))

    # Move East twice beyond boundary (arena_size=10)
    env.step(Action.MOVE_EAST)
    obs, info = env.step(Action.MOVE_EAST)
    assert info["agent_pos"][0] <= 10.0
    assert info["agent_pos"][0] >= 9.9


def test_environment_reconfiguration():
    env = MinimalEmbodiedEnvironment(arena_size=10.0, seed=42)
    obs_a = env.reset(initial_pos=np.array([5.0, 5.0]))
    assert env.env_id == "ENV_A"

    env.set_environment("ENV_B")
    assert env.env_id == "ENV_B"
    obs_b = env.get_observation()

    # Landmark positions shifted between ENV_A and ENV_B, so observation must change
    assert not np.allclose(obs_a, obs_b, atol=1e-3)


def test_seed_reproducibility():
    env1 = MinimalEmbodiedEnvironment(seed=123)
    obs1 = env1.reset(seed=123)
    actions = [Action.MOVE_EAST, Action.MOVE_NORTH, Action.MOVE_WEST, Action.MOVE_SOUTH]

    traj1 = []
    for a in actions:
        o, inf = env1.step(a)
        traj1.append((o, inf["agent_pos"]))

    env2 = MinimalEmbodiedEnvironment(seed=123)
    obs2 = env2.reset(seed=123)
    assert np.allclose(obs1, obs2)

    traj2 = []
    for a in actions:
        o, inf = env2.step(a)
        traj2.append((o, inf["agent_pos"]))

    for (o1, p1), (o2, p2) in zip(traj1, traj2):
        assert np.allclose(o1, o2)
        assert p1 == p2
