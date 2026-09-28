"""Delayed Match-to-Sample benchmark task generator for Experiment 001.

Implements the continuous delayed comparison paradigm specified in
experiments/EXPERIMENT_001_SPECIFICATION.md.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    t_cue: int = 5
    t_delay: int = 25
    t_probe: int = 5
    distractor_sigma: float = 0.1
    seed: int = 42

    @property
    def total_steps(self) -> int:
        return self.t_cue + self.t_delay + self.t_probe


class DelayedMatchTask:
    """Deterministic, reproducible Delayed Match-to-Sample task generator."""

    def __init__(self, config: TaskConfig):
        self.config = config

    def generate_trial(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        """Generate a single trial.

        Returns:
            X: Array of shape [total_steps, 2] containing:
               - x[:, 0]: cue/probe stimulus or distractor noise
               - x[:, 1]: phase flag (1.0 = cue, 0.0 = delay, 2.0 = probe)
            Y: Array of shape [total_steps, 1] containing target output:
               - 0.0 during cue and delay
               - +1.0 (match) or -1.0 (mismatch) during probe
            mask: Array of shape [total_steps, 1] indicating loss evaluation steps
                  (1.0 during probe phase, 0.0 otherwise)
            metadata: Trial properties dictionary
        """
        c = self.config
        total = c.total_steps
        X = np.zeros((total, 2), dtype=np.float32)
        Y = np.zeros((total, 1), dtype=np.float32)
        mask = np.zeros((total, 1), dtype=np.float32)

        # 1. Cue selection: +1.0 or -1.0 with equal probability
        s_cue = float(rng.choice([-1.0, 1.0]))

        # 2. Match condition: 50% match, 50% mismatch
        is_match = bool(rng.choice([True, False]))
        s_probe = s_cue if is_match else -s_cue

        # 3. Cue phase: t in [0, t_cue)
        X[0 : c.t_cue, 0] = s_cue
        X[0 : c.t_cue, 1] = 1.0  # Cue flag

        # 4. Delay phase: t in [t_cue, t_cue + t_delay)
        delay_start = c.t_cue
        delay_end = c.t_cue + c.t_delay
        if c.distractor_sigma > 0:
            X[delay_start:delay_end, 0] = rng.normal(
                0.0, c.distractor_sigma, size=c.t_delay
            )
        X[delay_start:delay_end, 1] = 0.0  # Delay flag

        # 5. Probe phase: t in [delay_end, total)
        probe_target = 1.0 if is_match else -1.0
        X[delay_end:total, 0] = s_probe
        X[delay_end:total, 1] = 2.0  # Probe flag

        Y[delay_end:total, 0] = probe_target
        mask[delay_end:total, 0] = 1.0

        metadata = {
            "s_cue": s_cue,
            "s_probe": s_probe,
            "is_match": 1.0 if is_match else 0.0,
            "target": probe_target,
            "t_delay": float(c.t_delay),
        }
        return X, Y, mask, metadata

    def generate_dataset(
        self, num_samples: int, seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Generate a batch dataset deterministically.

        Returns:
            X: [num_samples, total_steps, 2]
            Y: [num_samples, total_steps, 1]
            mask: [num_samples, total_steps, 1]
            metadata_list: list of trial metadata dicts
        """
        rng = np.random.default_rng(seed)
        total = self.config.total_steps

        X = np.zeros((num_samples, total, 2), dtype=np.float32)
        Y = np.zeros((num_samples, total, 1), dtype=np.float32)
        mask = np.zeros((num_samples, total, 1), dtype=np.float32)
        meta = []

        for i in range(num_samples):
            x_i, y_i, m_i, meta_i = self.generate_trial(rng)
            X[i] = x_i
            Y[i] = y_i
            mask[i] = m_i
            meta.append(meta_i)

        return X, Y, mask, meta
