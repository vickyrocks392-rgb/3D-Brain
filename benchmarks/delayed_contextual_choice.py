"""Delayed Contextual Choice Benchmark Task for 3D-Brain Phase 9.

Implements a temporal credit assignment task where correct synaptic credit cannot be
assigned from instantaneous sensory observations alone:
    Context A + Cue X -> Action 1 (Target: +1.0) -> Reward
    Context B + Cue X -> Action 2 (Target: -1.0) -> Reward

Structure of each trial:
1. Context Phase (T_ctx steps): Context A or B sensory cue presented.
2. Inter-Stimulus Delay (T_delay1 steps): Distractor noise, cue absent.
3. Ambiguous Cue Phase (T_cue steps): Identical sensory vector Cue X presented in BOTH contexts.
4. Action Phase (1 step): Network generates action/response.
5. Post-Action Reward Delay (T_reward_delay steps): Reward delayed by delta_t in {0, 1, 5, 10, 25, 50, 100, 250}.
6. Reward Delivery: Feedback signal delivered to test temporal credit assignment across eligibility horizons.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrialStepData:
    step: int
    sensory_input: np.ndarray
    target_action: float
    is_action_step: bool
    is_reward_step: bool
    reward_value: float
    context_id: int
    phase_name: str


class DelayedContextualChoiceTask:
    """Benchmark task for temporal credit assignment under delayed feedback."""

    def __init__(
        self,
        obs_dim: int = 8,
        t_ctx: int = 4,
        t_delay1: int = 8,
        t_cue: int = 3,
        distractor_std: float = 0.02,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.t_ctx = t_ctx
        self.t_delay1 = t_delay1
        self.t_cue = t_cue
        self.distractor_std = distractor_std
        self.rng = np.random.default_rng(seed)

        # 1. Base sensory prototypes
        # Context A and B vectors (distinct high-dimensional patterns)
        self.ctx_a_vec = self.rng.uniform(0.2, 0.8, obs_dim).astype(np.float32)
        self.ctx_a_vec[0] = 0.8  # Context indicator channel
        self.ctx_a_vec[-1] = 0.0

        self.ctx_b_vec = self.rng.uniform(0.2, 0.8, obs_dim).astype(np.float32)
        self.ctx_b_vec[0] = 0.2  # Context indicator channel
        self.ctx_b_vec[-1] = 0.0

        # Ambiguous Cue X: IDENTICAL across both Context A and B
        self.cue_x_vec = self.rng.uniform(0.3, 0.7, obs_dim).astype(np.float32)
        self.cue_x_vec[0] = 0.5  # Neutral context channel
        self.cue_x_vec[1] = 0.9  # Cue X feature flag
        self.cue_x_vec[-1] = 0.0

        # Probe / Decision trigger vector
        self.probe_vec = np.zeros(obs_dim, dtype=np.float32)
        self.probe_vec[-1] = 1.0  # Decision trigger channel active

    def generate_trial(
        self,
        context_id: int,         # 1 for Context A, 2 for Context B
        reward_delay: int = 0,   # Steps between action and reward delivery
        trial_seed: Optional[int] = None,
    ) -> List[TrialStepData]:
        """Generate a complete trial sequence with delayed reward delivery.

        Returns:
            List of TrialStepData objects detailing each step of the trial.
        """
        rng = np.random.default_rng(trial_seed) if trial_seed is not None else self.rng
        steps: List[TrialStepData] = []
        step_idx = 0

        # Target: Context 1 (A) requires Action +1.0; Context 2 (B) requires Action -1.0
        target_action = 1.0 if context_id == 1 else -1.0
        ctx_vec = self.ctx_a_vec if context_id == 1 else self.ctx_b_vec

        # 1. Context Presentation Phase
        for _ in range(self.t_ctx):
            noise = rng.normal(0.0, self.distractor_std, self.obs_dim).astype(np.float32)
            steps.append(
                TrialStepData(
                    step=step_idx,
                    sensory_input=np.clip(ctx_vec + noise, 0.0, 1.0),
                    target_action=target_action,
                    is_action_step=False,
                    is_reward_step=False,
                    reward_value=0.0,
                    context_id=context_id,
                    phase_name="CONTEXT",
                )
            )
            step_idx += 1

        # 2. Inter-Stimulus Delay Phase 1 (Maintenance of latent context)
        for _ in range(self.t_delay1):
            noise = rng.normal(0.5, self.distractor_std, self.obs_dim).astype(np.float32)
            noise[-1] = 0.0
            steps.append(
                TrialStepData(
                    step=step_idx,
                    sensory_input=np.clip(noise, 0.0, 1.0),
                    target_action=target_action,
                    is_action_step=False,
                    is_reward_step=False,
                    reward_value=0.0,
                    context_id=context_id,
                    phase_name="DELAY_1",
                )
            )
            step_idx += 1

        # 3. Ambiguous Cue Phase (Identical Cue X presented)
        for _ in range(self.t_cue):
            noise = rng.normal(0.0, self.distractor_std, self.obs_dim).astype(np.float32)
            steps.append(
                TrialStepData(
                    step=step_idx,
                    sensory_input=np.clip(self.cue_x_vec + noise, 0.0, 1.0),
                    target_action=target_action,
                    is_action_step=False,
                    is_reward_step=False,
                    reward_value=0.0,
                    context_id=context_id,
                    phase_name="AMBIGUOUS_CUE",
                )
            )
            step_idx += 1

        # 4. Action / Decision Step
        steps.append(
            TrialStepData(
                step=step_idx,
                sensory_input=self.probe_vec.copy(),
                target_action=target_action,
                is_action_step=True,
                is_reward_step=(reward_delay == 0),
                reward_value=0.0,  # Populated dynamically upon action evaluation
                context_id=context_id,
                phase_name="ACTION",
            )
        )
        step_idx += 1

        # 5. Reward Delay Interval
        for d in range(reward_delay):
            is_final_delay_step = (d == reward_delay - 1)
            noise = rng.normal(0.5, self.distractor_std, self.obs_dim).astype(np.float32)
            noise[-1] = 0.0
            steps.append(
                TrialStepData(
                    step=step_idx,
                    sensory_input=np.clip(noise, 0.0, 1.0),
                    target_action=target_action,
                    is_action_step=False,
                    is_reward_step=is_final_delay_step,
                    reward_value=0.0,  # Populated when reward is delivered
                    context_id=context_id,
                    phase_name="REWARD_DELAY" if not is_final_delay_step else "REWARD_DELIVERY",
                )
            )
            step_idx += 1

        return steps
