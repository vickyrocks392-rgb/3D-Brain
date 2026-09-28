"""Experiment 010: Long-Delay Contextual Credit Assignment Benchmark (Phase 10).

Formal benchmark for temporal credit assignment and working memory retention:
- Contextual cue A or B presented during Context Phase (T_ctx steps)
- Sustained delay interval (T_delay steps) with distractor noise
- Ambiguous Probe / Choice opportunity (T_choice steps) where instantaneous sensory
  input is IDENTICAL across both contexts, requiring persistent working memory
- Reward delivered at variable delay delta_t in [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000]
- Ground-truth causal synapse relevance masks for computing Credit Precision, Credit Leakage,
  and Credit Specificity Index (CSI).
- Continual learning regimes: stationary, abrupt reversal, gradual drift, recurring.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ContinualRegime(str, Enum):
    STATIONARY = "STATIONARY"
    ABRUPT_REVERSAL = "ABRUPT_REVERSAL"
    GRADUAL_DRIFT = "GRADUAL_DRIFT"
    RECURRING = "RECURRING"


@dataclass
class StepRecord:
    step: int
    sensory_input: np.ndarray
    target_action: float
    is_choice_step: bool
    is_reward_step: bool
    reward_value: float
    context_id: int
    phase_name: str


class LongDelayContextualCreditTask:
    """Rigorous benchmark task for evaluating long-delay credit assignment and memory retention."""

    def __init__(
        self,
        obs_dim: int = 8,
        t_ctx: int = 5,
        t_delay: int = 25,
        t_choice: int = 2,
        distractor_std: float = 0.02,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.t_ctx = t_ctx
        self.t_delay = t_delay
        self.t_choice = t_choice
        self.distractor_std = distractor_std
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # High-dimensional orthogonal prototypes for Context A and B
        self.ctx_a = np.zeros(obs_dim, dtype=np.float32)
        self.ctx_a[0] = 1.0   # Context A channel
        self.ctx_a[2:5] = 0.6 # Unique subpattern

        self.ctx_b = np.zeros(obs_dim, dtype=np.float32)
        self.ctx_b[1] = 1.0   # Context B channel
        self.ctx_b[5:8] = 0.6 # Unique subpattern

        # Ambiguous probe: IDENTICAL across both contexts, decision trigger channel active
        self.probe_vec = np.zeros(obs_dim, dtype=np.float32)
        self.probe_vec[-1] = 1.0  # Go/Decision trigger

    def generate_trial(
        self,
        context_id: int,          # 1 for Context A, 2 for Context B
        delay_steps: int = 25,    # Inter-stimulus delay between context and choice
        reward_delay: int = 0,    # Delay between choice and reward delivery
        regime: ContinualRegime = ContinualRegime.STATIONARY,
        trial_index: int = 0,
        trial_seed: Optional[int] = None,
    ) -> List[StepRecord]:
        """Generate a complete trial sequence."""
        rng = np.random.default_rng(trial_seed) if trial_seed is not None else self.rng
        steps: List[StepRecord] = []
        step_idx = 0

        # Contingency mapping under continual regimes:
        # Default: Context 1 (A) -> +1.0, Context 2 (B) -> -1.0
        if regime == ContinualRegime.ABRUPT_REVERSAL and trial_index >= 30:
            target_action = -1.0 if context_id == 1 else 1.0
        elif regime == ContinualRegime.RECURRING:
            phase = (trial_index // 20) % 2
            if phase == 1:
                target_action = -1.0 if context_id == 1 else 1.0
            else:
                target_action = 1.0 if context_id == 1 else -1.0
        else:
            target_action = 1.0 if context_id == 1 else -1.0

        ctx_proto = self.ctx_a if context_id == 1 else self.ctx_b
        if regime == ContinualRegime.GRADUAL_DRIFT and trial_index > 0:
            drift = rng.normal(0.0, 0.005 * min(20, trial_index), self.obs_dim).astype(np.float32)
            ctx_proto = np.clip(ctx_proto + drift, 0.0, 1.0)

        # 1. Context Presentation Phase
        for _ in range(self.t_ctx):
            noise = rng.normal(0.0, self.distractor_std, self.obs_dim).astype(np.float32)
            steps.append(
                StepRecord(
                    step=step_idx,
                    sensory_input=np.clip(ctx_proto + noise, 0.0, 1.0),
                    target_action=target_action,
                    is_choice_step=False,
                    is_reward_step=False,
                    reward_value=0.0,
                    context_id=context_id,
                    phase_name="CONTEXT",
                )
            )
            step_idx += 1

        # 2. Inter-stimulus Delay Phase (Distractor maintenance)
        for _ in range(delay_steps):
            noise = rng.normal(0.2, self.distractor_std, self.obs_dim).astype(np.float32)
            noise[-1] = 0.0
            steps.append(
                StepRecord(
                    step=step_idx,
                    sensory_input=np.clip(noise, 0.0, 1.0),
                    target_action=target_action,
                    is_choice_step=False,
                    is_reward_step=False,
                    reward_value=0.0,
                    context_id=context_id,
                    phase_name="DELAY",
                )
            )
            step_idx += 1

        # 3. Ambiguous Probe / Choice Phase
        for c in range(self.t_choice):
            is_final_choice = (c == self.t_choice - 1)
            noise = rng.normal(0.0, self.distractor_std, self.obs_dim).astype(np.float32)
            steps.append(
                StepRecord(
                    step=step_idx,
                    sensory_input=np.clip(self.probe_vec + noise, 0.0, 1.0),
                    target_action=target_action,
                    is_choice_step=is_final_choice,
                    is_reward_step=(is_final_choice and reward_delay == 0),
                    reward_value=0.0,
                    context_id=context_id,
                    phase_name="CHOICE",
                )
            )
            step_idx += 1

        # 4. Post-Choice Reward Delay Interval
        for d in range(reward_delay):
            is_final_reward = (d == reward_delay - 1)
            noise = rng.normal(0.2, self.distractor_std, self.obs_dim).astype(np.float32)
            noise[-1] = 0.0
            steps.append(
                StepRecord(
                    step=step_idx,
                    sensory_input=np.clip(noise, 0.0, 1.0),
                    target_action=target_action,
                    is_choice_step=False,
                    is_reward_step=is_final_reward,
                    reward_value=0.0,
                    context_id=context_id,
                    phase_name="REWARD_DELAY" if not is_final_reward else "REWARD_DELIVERY",
                )
            )
            step_idx += 1

        return steps

    def compute_causal_relevance_mask(
        self,
        hidden_dim: int,
        context_id: int,
        module_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Construct the ground-truth boolean mask of causally relevant recurrent synapses.

        In 3D-Brain modular topologies:
        - Context A activates Community 0 and 1
        - Context B activates Community 2 and 3
        Synapses connecting within the active context's subnetwork are causally relevant,
        while synapses connecting to the alternate context's subnetwork are irrelevant.
        """
        mask = np.zeros((hidden_dim, hidden_dim), dtype=bool)
        if module_ids is not None:
            active_comm = {0, 1} if context_id == 1 else {2, 3}
            for i in range(hidden_dim):
                for j in range(hidden_dim):
                    if module_ids[i] in active_comm and module_ids[j] in active_comm:
                        mask[i, j] = True
        else:
            # First half of neurons encode Context A, second half Context B
            half = hidden_dim // 2
            if context_id == 1:
                mask[:half, :half] = True
            else:
                mask[half:, half:] = True
        return mask
