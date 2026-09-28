"""Continual Delayed Association benchmark task generator for Experiment 003.

Implements the multi-environment sequential tasks specified in:
experiments/EXPERIMENT_003_SPECIFICATION.md
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ContinualCondition(str, Enum):
    STATIONARY = "STATIONARY"
    ABRUPT_SHIFT = "ABRUPT_SHIFT"
    RECURRING = "RECURRING"
    MULTIPLE = "MULTIPLE"
    GRADUAL_DRIFT = "GRADUAL_DRIFT"


@dataclass(frozen=True)
class ContinualConfig:
    condition: ContinualCondition = ContinualCondition.RECURRING
    t_cue: int = 5
    t_delay: int = 15
    t_probe: int = 5
    distractor_sigma: float = 0.05
    seed: int = 42

    @property
    def total_steps(self) -> int:
        return self.t_cue + self.t_delay + self.t_probe


class ContinualDelayedAssociationTask:
    """Benchmark task generator for continual adaptation across non-stationary regimes."""

    def __init__(self, config: ContinualConfig):
        self.config = config

    def get_schedule(self) -> List[Tuple[str, int]]:
        """Return sequence of (env_name, num_trials) for the configured condition."""
        cond = self.config.condition
        if cond == ContinualCondition.STATIONARY:
            return [("ENV_A", 100)]
        elif cond == ContinualCondition.ABRUPT_SHIFT:
            return [("ENV_A", 50), ("ENV_B", 50)]
        elif cond == ContinualCondition.RECURRING:
            return [("ENV_A", 40), ("ENV_B", 40), ("ENV_A", 40)]
        elif cond == ContinualCondition.MULTIPLE:
            return [("ENV_A", 30), ("ENV_B", 30), ("ENV_C", 30), ("ENV_A", 30), ("ENV_B", 30)]
        elif cond == ContinualCondition.GRADUAL_DRIFT:
            return [("ENV_DRIFT", 100)]
        else:
            raise ValueError(f"Unknown condition: {cond}")

    def generate_trial(
        self,
        trial_idx: int,
        env_name: str,
        rng: np.random.Generator,
        drift_angle: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        """Generate a single trial under a specific environmental mapping rule."""
        c = self.config
        total = c.total_steps
        X = np.zeros((total, 2), dtype=np.float32)
        Y = np.zeros((total, 1), dtype=np.float32)
        mask = np.zeros((total, 1), dtype=np.float32)

        # 1. Cue selection (+1.0 or -1.0)
        s_cue = float(rng.choice([-1.0, 1.0]))

        # 2. Match condition (50% match, 50% mismatch)
        is_match = bool(rng.choice([True, False]))
        s_probe = s_cue if is_match else -s_cue

        # 3. Cue phase
        X[0 : c.t_cue, 0] = s_cue
        X[0 : c.t_cue, 1] = 1.0

        # 4. Delay phase
        delay_start = c.t_cue
        delay_end = c.t_cue + c.t_delay
        if c.distractor_sigma > 0:
            X[delay_start:delay_end, 0] = rng.normal(
                0.0, c.distractor_sigma, size=c.t_delay
            )
        X[delay_start:delay_end, 1] = 0.0

        # 5. Probe phase & target computation based on active environment rule
        match_sign = 1.0 if is_match else -1.0
        if env_name == "ENV_A":
            # Identity: Match -> +1.0, Mismatch -> -1.0
            probe_target = match_sign
        elif env_name == "ENV_B":
            # Inverted: Match -> -1.0, Mismatch -> +1.0
            probe_target = -match_sign
        elif env_name == "ENV_C":
            # Attenuated: Match -> +0.5, Mismatch -> -0.5
            probe_target = 0.5 * match_sign
        elif env_name == "ENV_DRIFT":
            # Continuously rotating target mapping
            probe_target = float(np.cos(drift_angle) * match_sign)
        else:
            probe_target = match_sign

        X[delay_end:total, 0] = s_probe
        X[delay_end:total, 1] = 2.0  # Probe flag

        Y[delay_end:total, 0] = probe_target
        mask[delay_end:total, 0] = 1.0

        metadata = {
            "trial_idx": float(trial_idx),
            "env_name": env_name,
            "s_cue": s_cue,
            "s_probe": s_probe,
            "is_match": 1.0 if is_match else 0.0,
            "target": probe_target,
            "t_delay": float(c.t_delay),
            "drift_angle": float(drift_angle),
        }
        return X, Y, mask, metadata

    def generate_sequence(
        self, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Generate the full sequential dataset across all phases deterministically."""
        actual_seed = seed if seed is not None else self.config.seed
        rng = np.random.default_rng(actual_seed)
        schedule = self.get_schedule()

        total_trials = sum(n for _, n in schedule)
        total_steps = self.config.total_steps

        X = np.zeros((total_trials, total_steps, 2), dtype=np.float32)
        Y = np.zeros((total_trials, total_steps, 1), dtype=np.float32)
        mask = np.zeros((total_trials, total_steps, 1), dtype=np.float32)
        meta: List[Dict[str, Any]] = []

        trial_counter = 0
        drift_angle = 0.0
        drift_step = float(np.pi / 50.0)

        for phase_idx, (env_name, num_trials) in enumerate(schedule):
            for t_in_phase in range(num_trials):
                is_switch = (t_in_phase == 0 and phase_idx > 0)
                x_i, y_i, m_i, meta_i = self.generate_trial(
                    trial_idx=trial_counter,
                    env_name=env_name,
                    rng=rng,
                    drift_angle=drift_angle,
                )
                meta_i["env_switch"] = 1.0 if is_switch else 0.0
                meta_i["phase_idx"] = float(phase_idx)

                X[trial_counter] = x_i
                Y[trial_counter] = y_i
                mask[trial_counter] = m_i
                meta.append(meta_i)

                trial_counter += 1
                if env_name == "ENV_DRIFT":
                    drift_angle += drift_step

        return X, Y, mask, meta
