"""Unit tests for LongDelayContextualCreditTask (Phase 10)."""

import numpy as np
import pytest

from benchmarks.delayed_credit_assignment import (
    LongDelayContextualCreditTask,
    ContinualRegime,
)
from benchmarks.bptt_baseline import BPTTGradientBaseline


def test_long_delay_task_trial_generation():
    task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=20, t_choice=2, seed=42)

    trial_ctx1 = task.generate_trial(context_id=1, delay_steps=20, reward_delay=5)
    # Total steps = 5 (ctx) + 20 (delay) + 2 (choice) + 5 (reward delay) = 32 steps
    assert len(trial_ctx1) == 32

    # Check phases
    assert trial_ctx1[0].phase_name == "CONTEXT"
    assert trial_ctx1[5].phase_name == "DELAY"
    assert trial_ctx1[25].phase_name == "CHOICE"
    assert trial_ctx1[26].is_choice_step == True
    assert trial_ctx1[-1].is_reward_step == True
    assert trial_ctx1[-1].phase_name == "REWARD_DELIVERY"


def test_continual_learning_regimes():
    task = LongDelayContextualCreditTask(obs_dim=8, seed=42)

    # Stationary
    t_stat = task.generate_trial(context_id=1, regime=ContinualRegime.STATIONARY, trial_index=0)
    assert t_stat[0].target_action == 1.0

    # Abrupt reversal at trial 35
    t_rev = task.generate_trial(context_id=1, regime=ContinualRegime.ABRUPT_REVERSAL, trial_index=35)
    assert t_rev[0].target_action == -1.0


def test_causal_relevance_mask():
    task = LongDelayContextualCreditTask(obs_dim=8, seed=42)
    mask = task.compute_causal_relevance_mask(hidden_dim=20, context_id=1)
    assert mask.shape == (20, 20)
    assert mask.dtype == bool
    assert np.any(mask == True)
    assert np.any(mask == False)


def test_bptt_gradient_baseline():
    task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=3, t_delay=5, t_choice=1, seed=42)
    model = BPTTGradientBaseline(input_dim=8, hidden_dim=15, output_dim=1, seed=42)

    trial = task.generate_trial(context_id=1, delay_steps=5, reward_delay=0)
    x_seq = np.array([s.sensory_input for s in trial], dtype=np.float32)
    choice_idx = [i for i, s in enumerate(trial) if s.is_choice_step][0]

    loss = model.train_trial(x_seq, target_action=1.0, choice_step_idx=choice_idx)
    assert loss >= 0.0
    assert model.parameter_count() > 0
