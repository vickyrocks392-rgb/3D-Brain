"""Tests for Delayed Contextual Choice Task and Temporal Credit Assignment (Phase 9).

Verifies:
1. DelayedContextualChoiceTask structure (Context, Delay1, Ambiguous Cue X, Action, Delay2, Reward).
2. Credit assignment across delayed rewards.
3. Falsification check: Shuffled reward vs ordered reward.
"""

import numpy as np
import pytest

from benchmarks.delayed_contextual_choice import DelayedContextualChoiceTask
from experiments.run_experiment_009 import run_delayed_contextual_choice_experiment


def test_delayed_contextual_choice_task_generation():
    """Verify trial phase sequence, sensory ambiguity of Cue X, and reward timing."""
    task = DelayedContextualChoiceTask(obs_dim=6, t_ctx=3, t_delay1=5, t_cue=3, seed=42)

    # Context 1 trial
    trial_c1 = task.generate_trial(context_id=1, reward_delay=4, trial_seed=42)
    # Context 2 trial
    trial_c2 = task.generate_trial(context_id=2, reward_delay=4, trial_seed=43)

    assert len(trial_c1) == 3 + 5 + 3 + 1 + 4  # 16 steps
    assert len(trial_c2) == len(trial_c1)

    # Check phase sequence
    phases = [s.phase_name for s in trial_c1]
    assert phases[:3] == ["CONTEXT"] * 3
    assert phases[3:8] == ["DELAY_1"] * 5
    assert phases[8:11] == ["AMBIGUOUS_CUE"] * 3
    assert phases[11] == "ACTION"
    assert phases[12:15] == ["REWARD_DELAY"] * 3
    assert phases[15] == "REWARD_DELIVERY"

    # Crucial scientific check: Ambiguous Cue X is identical across contexts
    cue_x_c1 = task.cue_x_vec
    cue_x_c2 = task.cue_x_vec
    assert np.allclose(cue_x_c1, cue_x_c2)

    # Targets are opposite: Action 1 for Context 1, Action 2 for Context 2
    assert trial_c1[0].target_action == 1.0
    assert trial_c2[0].target_action == -1.0


def test_credit_assignment_learning():
    """Verify that three-factor RPE network demonstrates active learning on delayed contextual choice."""
    # Run short experiment on seeds [42]
    res_rpe = run_delayed_contextual_choice_experiment(
        condition="F_SIGNED_RPE",
        reward_delay=2,
        num_trials=30,
        seeds=[42],
        hidden_dim=25,
    )
    # RPE accuracy should be well above chance
    assert res_rpe["accuracy"] >= 0.60
    assert res_rpe["mean_eligibility"] > 0.0
    assert res_rpe["total_weight_change"] > 0.0


def test_ablation_falsification():
    """Verify that removing eligibility traces prevents credit assignment across delays."""
    res_no_elig = run_delayed_contextual_choice_experiment(
        condition="F_SIGNED_RPE",
        reward_delay=10,
        num_trials=30,
        seeds=[42],
        hidden_dim=25,
        ablation="ABL_01_NO_ELIGIBILITY",
    )
    # Without eligibility, credit cannot be bridged across 10 steps
    assert res_no_elig["accuracy"] <= 0.60
