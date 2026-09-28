"""Experiment runner for reproducible training, evaluation, and telemetry export.

Satisfies requirements of:
experiments/EXPERIMENT_001_SPECIFICATION.md
docs/PHASE_1_IMPLEMENTATION_SPEC.md
"""

from dataclasses import asdict, dataclass
import datetime
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Type
import numpy as np

from benchmarks.delayed_match import DelayedMatchTask, TaskConfig
from brain.base import AdamWOptimizer, BrainModel
from brain.gru import BaseGRU
from brain.liquid import CandLiquid
from brain.rnn import BaseRNN
from visualization.telemetry import (
    build_frame,
    build_topology,
    export_brainlog,
    validate_brainlog,
)


@dataclass
class ExperimentConfig:
    experiment_id: str = "EXP001"
    model_type: str = "BASE-RNN"  # "BASE-RNN", "BASE-GRU", "CAND-LI"
    target_budget: int = 300  # 300, 1000, 3000
    t_delay: int = 25
    distractor_sigma: float = 0.1
    seed: int = 42
    epochs: int = 80
    batch_size: int = 32
    num_train_batches: int = 15  # 15 * 32 = 480 trials/epoch
    num_val_samples: int = 120
    num_test_samples: int = 200
    lr: float = 0.005
    clip_norm: float = 1.0


# Map model type and budget to exact hidden dimension
BUDGET_MAP = {
    # target ~300
    ("BASE-RNN", 300): 16,
    ("BASE-GRU", 300): 9,
    ("CAND-LI", 300): 15,
    # target ~1000
    ("BASE-RNN", 1000): 30,
    ("BASE-GRU", 1000): 16,
    ("CAND-LI", 1000): 29,
    # target ~3000
    ("BASE-RNN", 3000): 52,
    ("BASE-GRU", 3000): 30,
    ("CAND-LI", 3000): 51,
}

MODEL_CLASSES: Dict[str, Type[BrainModel]] = {
    "BASE-RNN": BaseRNN,
    "BASE-GRU": BaseGRU,
    "CAND-LI": CandLiquid,
}


def instantiate_model(model_type: str, budget: int, seed: int) -> BrainModel:
    cls = MODEL_CLASSES[model_type]
    hidden_dim = BUDGET_MAP.get((model_type, budget))
    if hidden_dim is None:
        raise ValueError(f"Unknown configuration: model {model_type}, budget {budget}")
    return cls(input_dim=2, hidden_dim=hidden_dim, output_dim=1, seed=seed)


def compute_metrics(
    model: BrainModel,
    X: np.ndarray,
    Y_target: np.ndarray,
    mask: np.ndarray,
    metadata_list: list,
    task_config: TaskConfig,
) -> Dict[str, float]:
    """Compute accuracy, MSE, and Memory Trace Stability Ratio (MSR)."""
    B, T, _ = X.shape
    Y_pred, H, _, _ = model.forward_sequence(X)

    # Response decision is taken at probe onset (or mean over probe steps)
    probe_start = task_config.t_cue + task_config.t_delay
    probe_preds = np.mean(Y_pred[:, probe_start:T, 0], axis=1)  # [B]
    probe_targets = np.mean(Y_target[:, probe_start:T, 0], axis=1)  # [B]

    decisions = np.where(probe_preds >= 0.0, 1.0, -1.0)
    acc = float(np.mean(decisions == probe_targets))

    # Masked MSE during response
    mask_count = float(np.sum(mask)) + 1e-8
    mse = float(np.sum(((Y_pred - Y_target) * mask) ** 2) / mask_count)

    # Compute Memory Trace Stability Ratio (MSR)
    # Measures the separation of hidden state attractors during delay between cue=+1 and cue=-1
    cues = np.array([m["s_cue"] for m in metadata_list])
    idx_pos = np.where(cues > 0)[0]
    idx_neg = np.where(cues < 0)[0]

    delay_start = task_config.t_cue
    delay_end = delay_start + task_config.t_delay

    if len(idx_pos) > 0 and len(idx_neg) > 0 and delay_end > delay_start:
        h_pos_delay = np.mean(H[idx_pos, delay_start:delay_end, :], axis=(0, 1))
        h_neg_delay = np.mean(H[idx_neg, delay_start:delay_end, :], axis=(0, 1))
        dist = float(np.linalg.norm(h_pos_delay - h_neg_delay))
        h_std = float(np.std(H[:, delay_start:delay_end, :])) + 1e-6
        msr = dist / h_std
    else:
        msr = 0.0

    return {
        "accuracy": round(acc, 4),
        "mse": round(mse, 4),
        "msr": round(float(msr), 4),
    }


def run_experiment(config: ExperimentConfig) -> Tuple[dict, Optional[Path]]:
    """Execute complete training, evaluation, and telemetry export for one experiment configuration."""
    start_time = time.time()
    task_config = TaskConfig(
        t_cue=5,
        t_delay=config.t_delay,
        t_probe=5,
        distractor_sigma=config.distractor_sigma,
        seed=config.seed,
    )
    task = DelayedMatchTask(task_config)

    # Instantiate model
    model = instantiate_model(config.model_type, config.target_budget, config.seed)
    param_count = model.parameter_count()

    # Pre-generate validation and test sets deterministically
    val_seed = config.seed + 1000
    test_seed = config.seed + 2000
    X_val, Y_val, mask_val, meta_val = task.generate_dataset(
        config.num_val_samples, seed=val_seed
    )
    X_test, Y_test, mask_test, meta_test = task.generate_dataset(
        config.num_test_samples, seed=test_seed
    )

    optimizer = AdamWOptimizer(lr=config.lr, clip_norm=config.clip_norm)
    train_rng = np.random.default_rng(config.seed)

    train_loss_history = []
    val_acc_history = []

    best_val_acc = -1.0
    best_weights = model.get_weights()

    for epoch in range(config.epochs):
        epoch_losses = []
        for _ in range(config.num_train_batches):
            # Generate fresh batch
            X_b = np.zeros((config.batch_size, task_config.total_steps, 2), dtype=np.float32)
            Y_b = np.zeros((config.batch_size, task_config.total_steps, 1), dtype=np.float32)
            mask_b = np.zeros((config.batch_size, task_config.total_steps, 1), dtype=np.float32)

            for b in range(config.batch_size):
                x_i, y_i, m_i, _ = task.generate_trial(train_rng)
                X_b[b] = x_i
                Y_b[b] = y_i
                mask_b[b] = m_i

            Y_pred, _, _, cache = model.forward_sequence(X_b)
            loss = float(
                np.sum(((Y_pred - Y_b) * mask_b) ** 2) / (np.sum(mask_b) + 1e-8)
            )
            epoch_losses.append(loss)

            grads = model.backward_sequence(X_b, Y_pred, Y_b, mask_b, cache)
            weights = model.get_weights()
            optimizer.step(weights, grads)
            model.set_weights(weights)

        avg_loss = float(np.mean(epoch_losses))
        train_loss_history.append(avg_loss)

        # Validation evaluation
        val_metrics = compute_metrics(
            model, X_val, Y_val, mask_val, meta_val, task_config
        )
        val_acc = val_metrics["accuracy"]
        val_acc_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()

        if val_acc >= 0.99 and epoch >= 20:
            # Converged early
            break

    # Restore best checkpoint
    model.set_weights(best_weights)

    # Final test evaluation
    test_metrics = compute_metrics(
        model, X_test, Y_test, mask_test, meta_test, task_config
    )
    runtime_sec = time.time() - start_time

    pes = (test_metrics["accuracy"] / param_count) * 1000.0

    result = {
        "experiment_id": config.experiment_id,
        "model_id": config.model_type,
        "parameter_budget_target": config.target_budget,
        "parameter_count": param_count,
        "seed": config.seed,
        "delay": config.t_delay,
        "epochs_trained": len(train_loss_history),
        "final_train_loss": round(float(train_loss_history[-1]), 4),
        "val_accuracy": round(float(best_val_acc), 4),
        "test_accuracy": test_metrics["accuracy"],
        "test_mse": test_metrics["mse"],
        "msr": test_metrics["msr"],
        "pes": round(float(pes), 4),
        "runtime_sec": round(runtime_sec, 2),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Generate and export sample .brainlog for Match & Mismatch test trials
    brainlog_path = None
    try:
        brainlog_path = export_sample_brainlog(model, task, config, test_metrics)
    except Exception as e:
        print(f"Warning: Telemetry export failed: {e}")

    return result, brainlog_path


def export_sample_brainlog(
    model: BrainModel,
    task: DelayedMatchTask,
    config: ExperimentConfig,
    test_metrics: dict,
) -> Path:
    """Generate and serialize a complete .brainlog trajectory for visual inspection."""
    rng = np.random.default_rng(config.seed + 999)

    # Generate 1 match trial and 1 mismatch trial
    trials = []
    while len(trials) < 2:
        x_i, y_i, m_i, meta_i = task.generate_trial(rng)
        if len(trials) == 0 and meta_i["is_match"] == 1.0:
            trials.append((x_i, y_i, m_i, meta_i))
        elif len(trials) == 1 and meta_i["is_match"] == 0.0:
            trials.append((x_i, y_i, m_i, meta_i))

    # Pick the match trial as canonical representation
    X_single, Y_single, mask_single, meta_single = trials[0]
    X_in = X_single[np.newaxis, :, :]  # [1, T, 2]

    Y_pred, H, A, _ = model.forward_sequence(X_in)
    T = X_single.shape[0]

    topology = build_topology(model, config.model_type)
    frames = []

    dt_ms = 40.0  # 25 fps virtual timeline
    for t in range(T):
        inp = X_single[t]
        h_vec = H[0, t]
        a_vec = A[0, t]
        y_val = float(Y_pred[0, t, 0])
        target_val = float(Y_single[t, 0])
        m_val = float(mask_single[t, 0])

        event = None
        if t == 0:
            event = {"type": "CUE_START", "node_id": 0, "val": float(inp[0])}
        elif t == task.config.t_cue:
            event = {"type": "DELAY_START"}
        elif t == task.config.t_cue + task.config.t_delay:
            event = {"type": "PROBE_START", "node_id": 0, "val": float(inp[0])}
        elif t == T - 1:
            event = {
                "type": "DECISION_COMMITTED",
                "node_id": len(topology["nodes"]) - 1,
                "prediction": round(y_val, 3),
                "target": round(target_val, 3),
            }

        frame = build_frame(
            step=t,
            time_ms=t * dt_ms,
            input_vec=inp,
            h_vec=h_vec,
            a_vec=a_vec,
            y_val=y_val,
            target_val=target_val,
            mask_val=m_val,
            edges=topology["edges"],
            event=event,
        )
        frames.append(frame)

    metadata = {
        "experiment_id": config.experiment_id,
        "model_id": config.model_type,
        "budget": config.target_budget,
        "parameter_count": model.parameter_count(),
        "seed": config.seed,
        "delay": config.t_delay,
        "total_steps": T,
        "dt_ms": dt_ms,
        "trial_metadata": meta_single,
        "test_accuracy": test_metrics["accuracy"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    log_filename = f"{config.experiment_id}_{config.model_type}_b{config.target_budget}_d{config.t_delay}_s{config.seed}.brainlog"
    out_dir = Path("experiments/results/brainlogs")
    out_path = out_dir / log_filename
    export_brainlog(out_path, metadata, topology, frames)

    # Validate generated brainlog
    valid, reason = validate_brainlog(out_path)
    if not valid:
        raise ValueError(f"Generated brainlog failed validation: {reason}")

    return out_path
