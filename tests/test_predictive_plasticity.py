"""Tests for PredictiveLearningRules in Phase 5."""

import numpy as np
import pytest

from brain.predictive_plasticity import (
    FrozenPredictive,
    GradientPredictiveBaseline,
    LocalPredictivePlasticity,
    NeuromodulatedPredictivePlasticity,
)


def test_frozen_predictive_rule():
    rule = FrozenPredictive()
    H, S, A = 10, 8, 4
    W_hh = np.ones((H, H), dtype=np.float32)
    W_pred = np.ones((H + A, S), dtype=np.float32)
    b_pred = np.zeros(S, dtype=np.float32)
    adj = np.ones((H, H), dtype=bool)

    dW_hh, dW_pred, db_pred, metrics = rule.update(
        pre_act=np.ones(H, dtype=np.float32),
        post_act=np.ones(H, dtype=np.float32),
        post_cand=np.ones(H, dtype=np.float32),
        W_hh=W_hh,
        W_pred=W_pred,
        b_pred=b_pred,
        action_onehot=np.zeros(A, dtype=np.float32),
        target_s_next=np.ones(S, dtype=np.float32),
        pred_s_next=np.zeros(S, dtype=np.float32),
        error_vec=np.ones(S, dtype=np.float32),
        adjacency_mask=adj,
    )
    assert np.all(dW_hh == 0.0)
    assert np.all(dW_pred == 0.0)
    assert np.all(db_pred == 0.0)
    assert metrics["dw_norm"] == 0.0


def test_local_predictive_plasticity():
    H, S, A = 12, 8, 4
    rule = LocalPredictivePlasticity(hidden_dim=H, obs_dim=S, lr=0.01, lr_pred=0.02, seed=42)

    W_hh = np.zeros((H, H), dtype=np.float32)
    W_pred = np.zeros((H + A, S), dtype=np.float32)
    b_pred = np.zeros(S, dtype=np.float32)
    adj = np.eye(H, dtype=bool)  # Only diagonal active

    pre_act = np.ones(H, dtype=np.float32) * 0.5
    post_act = np.ones(H, dtype=np.float32) * 0.5
    error_vec = np.ones(S, dtype=np.float32) * 0.2

    dW_hh, dW_pred, db_pred, metrics = rule.update(
        pre_act=pre_act,
        post_act=post_act,
        post_cand=post_act,
        W_hh=W_hh,
        W_pred=W_pred,
        b_pred=b_pred,
        action_onehot=np.zeros(A, dtype=np.float32),
        target_s_next=np.ones(S, dtype=np.float32) * 0.7,
        pred_s_next=np.ones(S, dtype=np.float32) * 0.5,
        error_vec=error_vec,
        adjacency_mask=adj,
    )

    # dW_hh must respect adjacency mask (only non-zero on active edges)
    assert np.all(dW_hh[~adj] == 0.0)
    assert np.any(dW_hh[adj] != 0.0)
    assert np.any(dW_pred != 0.0)
    assert metrics["pred_mse"] > 0.0


def test_neuromodulated_predictive_plasticity():
    H, S, A = 10, 8, 4
    rule = NeuromodulatedPredictivePlasticity(hidden_dim=H, lr=0.01, beta=2.0)

    W_hh = np.zeros((H, H), dtype=np.float32)
    W_pred = np.zeros((H + A, S), dtype=np.float32)
    b_pred = np.zeros(S, dtype=np.float32)
    adj = np.ones((H, H), dtype=bool)

    pre_act = np.ones(H, dtype=np.float32) * 0.5
    post_act = np.ones(H, dtype=np.float32) * 0.5

    # Case 1: Large error -> High modulation
    large_err = np.ones(S, dtype=np.float32) * 1.0
    _, _, _, metrics_large = rule.update(
        pre_act=pre_act,
        post_act=post_act,
        post_cand=post_act,
        W_hh=W_hh,
        W_pred=W_pred,
        b_pred=b_pred,
        action_onehot=np.zeros(A, dtype=np.float32),
        target_s_next=np.ones(S, dtype=np.float32),
        pred_s_next=np.zeros(S, dtype=np.float32),
        error_vec=large_err,
        adjacency_mask=adj,
    )

    # Case 2: Zero error -> Zero modulation
    zero_err = np.zeros(S, dtype=np.float32)
    _, _, _, metrics_zero = rule.update(
        pre_act=pre_act,
        post_act=post_act,
        post_cand=post_act,
        W_hh=W_hh,
        W_pred=W_pred,
        b_pred=b_pred,
        action_onehot=np.zeros(A, dtype=np.float32),
        target_s_next=np.zeros(S, dtype=np.float32),
        pred_s_next=np.zeros(S, dtype=np.float32),
        error_vec=zero_err,
        adjacency_mask=adj,
    )

    assert metrics_large["modulation"] > 0.5
    assert metrics_zero["modulation"] == pytest.approx(0.0, abs=1e-5)


def test_gradient_predictive_baseline():
    H, S, A = 8, 4, 2
    rule = GradientPredictiveBaseline(lr=0.01, lr_pred=0.05)

    W_hh = np.zeros((H, H), dtype=np.float32)
    W_pred = np.ones((H + A, S), dtype=np.float32) * 0.5
    b_pred = np.zeros(S, dtype=np.float32)
    adj = np.ones((H, H), dtype=bool)

    pre_act = np.ones(H, dtype=np.float32) * 0.5
    post_act = np.ones(H, dtype=np.float32) * 0.5
    error_vec = np.ones(S, dtype=np.float32) * 0.3

    dW_hh, dW_pred, db_pred, metrics = rule.update(
        pre_act=pre_act,
        post_act=post_act,
        post_cand=post_act,
        W_hh=W_hh,
        W_pred=W_pred,
        b_pred=b_pred,
        action_onehot=np.zeros(A, dtype=np.float32),
        target_s_next=np.ones(S, dtype=np.float32) * 0.8,
        pred_s_next=np.ones(S, dtype=np.float32) * 0.5,
        error_vec=error_vec,
        adjacency_mask=adj,
    )
    assert np.any(dW_hh != 0.0)
    assert np.any(dW_pred != 0.0)
