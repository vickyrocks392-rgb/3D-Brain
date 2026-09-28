"""Task C: Spatially Distributed Temporal Task benchmark engine.

Specifies and generates trials where sensory information enters at spatially separated
regions and must be propagated, integrated, and evaluated over metric space and time.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SpatialTaskConfig:
    cue1_steps: int = 5
    inter_cue_steps: int = 4
    cue2_steps: int = 5
    delay_steps: int = 8
    probe_steps: int = 5
    noise_std: float = 0.05

    @property
    def total_steps(self) -> int:
        return self.cue1_steps + self.inter_cue_steps + self.cue2_steps + self.delay_steps + self.probe_steps


class SpatialDistributedTemporalTask:
    """Task C: Spatially Distributed Temporal Association.

    Input channels:
        Channel 0: Cue 1 (presented at Left Spatial Pole / Module 0)
        Channel 1: Cue 2 (presented at Right Spatial Pole / Module 1)
        Channel 2: Decision Trigger / Probe Flag (presented at both or trigger channel)

    Target:
        +1.0 if sign(Cue 1) == sign(Cue 2) else -1.0 during probe phase.
    """

    def __init__(self, config: Optional[SpatialTaskConfig] = None):
        self.config = config or SpatialTaskConfig()

    def generate_trial(
        self, trial_idx: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, any]]:
        """Generate a single trial sequence of shape:
        X: [total_steps, 3]
        Y: [total_steps, 1]
        mask: [total_steps, 1]
        """
        cfg = self.config
        total = cfg.total_steps
        X = np.zeros((total, 3), dtype=np.float32)
        Y = np.zeros((total, 1), dtype=np.float32)
        mask = np.zeros((total, 1), dtype=np.float32)

        # Discrete binary cues: +1.0 or -1.0
        val1 = float(rng.choice([-1.0, 1.0]))
        val2 = float(rng.choice([-1.0, 1.0]))
        target_val = 1.0 if (val1 == val2) else -1.0

        # Timeline phase offsets
        t0 = 0
        t1 = t0 + cfg.cue1_steps
        t2 = t1 + cfg.inter_cue_steps
        t3 = t2 + cfg.cue2_steps
        t4 = t3 + cfg.delay_steps
        t5 = t4 + cfg.probe_steps

        # Channel 0: Cue 1 (left pole)
        X[t0:t1, 0] = val1 + rng.normal(0.0, cfg.noise_std, cfg.cue1_steps)

        # Channel 1: Cue 2 (right pole)
        X[t2:t3, 1] = val2 + rng.normal(0.0, cfg.noise_std, cfg.cue2_steps)

        # Channel 2: Probe Flag
        X[t4:t5, 2] = 1.0

        # Background noise
        noise = rng.normal(0.0, cfg.noise_std, (total, 3)).astype(np.float32)
        X += noise

        # Target and loss mask during probe phase
        Y[t4:t5, 0] = target_val
        mask[t4:t5, 0] = 1.0

        meta = {
            "trial_idx": trial_idx,
            "cue1": val1,
            "cue2": val2,
            "target": target_val,
            "cue1_window": [t0, t1],
            "cue2_window": [t2, t3],
            "delay_window": [t3, t4],
            "probe_window": [t4, t5],
        }
        return X, Y, mask, meta

    def generate_dataset(
        self, num_trials: int, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        rng = np.random.default_rng(seed)
        total = self.config.total_steps
        X_all = np.zeros((num_trials, total, 3), dtype=np.float32)
        Y_all = np.zeros((num_trials, total, 1), dtype=np.float32)
        mask_all = np.zeros((num_trials, total, 1), dtype=np.float32)
        meta_all = []

        for i in range(num_trials):
            x, y, m, meta = self.generate_trial(i, rng)
            X_all[i] = x
            Y_all[i] = y
            mask_all[i] = m
            meta_all.append(meta)

        return X_all, Y_all, mask_all, meta_all
