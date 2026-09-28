"""Minimal Embodied Sensorimotor Environment for Phase 5.

Implements a deterministic, closed-loop 2D spatial arena containing:
- Agent metric position (x, y) bounded in [0, L] x [0, L]
- Multiple spatial landmarks with distinct signatures
- Latent environmental hidden state (e.g. cyclical phase)
- Local multi-channel sensory observations (boundary proximities, landmark fields)
- Discrete motor actions with deterministic spatial consequences
- Non-stationary reconfigurations (Environment A vs. Environment B)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Action(IntEnum):
    MOVE_NORTH = 0
    MOVE_EAST = 1
    MOVE_SOUTH = 2
    MOVE_WEST = 3


@dataclass
class Landmark:
    id: int
    name: str
    pos: np.ndarray  # [x, y]
    signature: float = 1.0


@dataclass
class EnvState:
    agent_pos: np.ndarray  # [x, y]
    agent_heading: float  # radians
    hidden_cycle: int  # latent environmental cycle (0..cycle_period-1)
    step_count: int
    env_id: str  # "ENV_A" or "ENV_B"


class MinimalEmbodiedEnvironment:
    """Deterministic, metric 2D arena for embodied predictive modeling."""

    def __init__(
        self,
        arena_size: float = 10.0,
        step_size: float = 0.5,
        cycle_period: int = 40,
        env_id: str = "ENV_A",
        seed: int = 42,
    ):
        self.arena_size = float(arena_size)
        self.step_size = float(step_size)
        self.cycle_period = int(cycle_period)
        self.env_id = env_id
        self.initial_seed = seed
        self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.agent_pos = np.array([self.arena_size / 2.0, self.arena_size / 2.0], dtype=np.float32)
        self.agent_heading = 0.0

        # Action vectors: [dx, dy]
        self.action_vectors = {
            Action.MOVE_NORTH: np.array([0.0, self.step_size], dtype=np.float32),
            Action.MOVE_EAST: np.array([self.step_size, 0.0], dtype=np.float32),
            Action.MOVE_SOUTH: np.array([0.0, -self.step_size], dtype=np.float32),
            Action.MOVE_WEST: np.array([-self.step_size, 0.0], dtype=np.float32),
        }

        # Sensory dimensionality: 4 wall distance sensors + 4 landmark proximity sensors = 8
        self.obs_dim = 8
        self.action_dim = 4

        self.landmarks: List[Landmark] = []
        self._init_landmarks()

    def _init_landmarks(self) -> None:
        """Initialize landmark positions based on current environment configuration."""
        self.landmarks.clear()
        L = self.arena_size
        if self.env_id == "ENV_A":
            # Corner layout
            coords = [
                (0.2 * L, 0.2 * L, "Beacon_SW", 1.0),
                (0.8 * L, 0.2 * L, "Beacon_SE", 0.75),
                (0.8 * L, 0.8 * L, "Beacon_NE", 0.5),
                (0.2 * L, 0.8 * L, "Beacon_NW", 0.25),
            ]
        elif self.env_id == "ENV_B":
            # Reconfigured cross/midpoint layout with altered signatures
            coords = [
                (0.5 * L, 0.2 * L, "Beacon_S", 0.5),
                (0.8 * L, 0.5 * L, "Beacon_E", 1.0),
                (0.5 * L, 0.8 * L, "Beacon_N", 0.25),
                (0.2 * L, 0.5 * L, "Beacon_W", 0.75),
            ]
        else:
            raise ValueError(f"Unknown env_id: {self.env_id}")

        for i, (x, y, name, sig) in enumerate(coords):
            self.landmarks.append(
                Landmark(id=i, name=name, pos=np.array([x, y], dtype=np.float32), signature=sig)
            )

    def set_environment(self, env_id: str) -> None:
        """Switch environment configuration (e.g. during non-stationary testing)."""
        if env_id != self.env_id:
            self.env_id = env_id
            self._init_landmarks()

    # Alias for reconfigure
    reconfigure = set_environment

    def reset(self, seed: Optional[int] = None, initial_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0

        if initial_pos is not None:
            self.agent_pos = np.clip(np.array(initial_pos, dtype=np.float32), 0.0, self.arena_size)
        else:
            # Deterministic central starting position with slight seed jitter
            jitter = self.rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
            self.agent_pos = np.clip(
                np.array([self.arena_size / 2.0, self.arena_size / 2.0], dtype=np.float32) + jitter,
                0.5,
                self.arena_size - 0.5,
            )

        self.agent_heading = 0.0
        return self.get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Advance one deterministic simulation time step."""
        action_enum = Action(int(action) % len(Action))
        disp = self.action_vectors[action_enum]

        # In ENV_B, slip or directional friction slightly shifts dynamics
        if self.env_id == "ENV_B":
            # Subtle orthogonal perturbation in ENV_B to test adaptive dynamics transfer
            ortho = np.array([-disp[1] * 0.15, disp[0] * 0.15], dtype=np.float32)
            actual_disp = disp + ortho
        else:
            actual_disp = disp

        # Apply position transition with rigid wall boundaries
        new_pos = self.agent_pos + actual_disp
        self.agent_pos = np.clip(new_pos, 0.0, self.arena_size)

        # Update heading
        if np.linalg.norm(actual_disp) > 1e-5:
            self.agent_heading = float(np.arctan2(actual_disp[1], actual_disp[0]))

        self.step_count += 1
        obs = self.get_observation()

        info = {
            "step": self.step_count,
            "agent_pos": [round(float(p), 4) for p in self.agent_pos],
            "agent_heading": round(float(self.agent_heading), 4),
            "env_id": self.env_id,
            "action": int(action_enum),
            "hidden_cycle": self.step_count % self.cycle_period,
        }
        return obs, info

    def get_observation(self) -> np.ndarray:
        """Compute local multi-channel continuous observation vector [obs_dim]."""
        L = self.arena_size
        x, y = self.agent_pos

        # Channels 0-3: Normalized distances to boundaries [0, 1]
        dist_north = (L - y) / L
        dist_east = (L - x) / L
        dist_south = y / L
        dist_west = x / L

        # Latent cycle modulation: beacon brightness subtly pulses with environmental hidden cycle
        cycle_phase = (self.step_count % self.cycle_period) / float(self.cycle_period)
        cycle_mod = 0.85 + 0.15 * np.cos(2.0 * np.pi * cycle_phase)

        # Channels 4-7: Landmark proximity field 1.0 / (1.0 + d_k) * signature * cycle_mod
        landmark_proximities = []
        for lm in self.landmarks:
            d = np.linalg.norm(self.agent_pos - lm.pos)
            prox = (1.0 / (1.0 + 0.5 * d)) * lm.signature * cycle_mod
            landmark_proximities.append(float(prox))

        obs = np.array(
            [dist_north, dist_east, dist_south, dist_west] + landmark_proximities,
            dtype=np.float32,
        )
        return obs

    def get_state(self) -> EnvState:
        """Return full environmental state."""
        return EnvState(
            agent_pos=self.agent_pos.copy(),
            agent_heading=self.agent_heading,
            hidden_cycle=self.step_count % self.cycle_period,
            step_count=self.step_count,
            env_id=self.env_id,
        )
