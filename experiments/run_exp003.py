"""Experiment 003 Runner: Continual Adaptation & Online Plasticity Matrix.

Executes the formal benchmark protocol specified in:
experiments/EXPERIMENT_003_SPECIFICATION.md
"""

import json
from pathlib import Path
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.base import AdamWOptimizer
from brain.plasticity import (
    FrozenPlasticity,
    HebbianPlasticity,
    HomeostaticPlasticity,
    ModulatedPlasticity,
    OjaPlasticity,
    PlasticRNN,
    PlasticityRule,
)
from benchmarks.continual_adaptation import (
    ContinualCondition,
    ContinualConfig,
    ContinualDelayedAssociationTask,
)
from benchmarks.delayed_match import DelayedMatchTask, TaskConfig
from observability.brainlog import BrainLog, BrainLogSerializer, BrainLogValidator
from observability.capture import StateCapture
from observability.events import ComputationalEvent, EventType
from observability.topology import TopologyExtractor

PLASTICITY_RULES = {
    "Frozen": lambda: FrozenPlasticity(),
    "Hebbian": lambda: HebbianPlasticity(lr=5e-4, weight_decay=1e-4),
    "Oja": lambda: OjaPlasticity(lr=1e-3),
    "Modulated": lambda: ModulatedPlasticity(lr=2e-3),
    "Homeostatic": lambda: HomeostaticPlasticity(lr=1e-3, target_activity=0.2, gamma=1e-4),
}

BUDGET_DIM_MAP = {
    300: 15,    # 15^2 + 5*15 + 1 = 301 params
    1000: 29,   # 29^2 + 5*29 + 1 = 987 params
    3000: 52,   # 52^2 + 5*52 + 1 = 2965 params
}


def pretrain_on_env_a(
    model: PlasticRNN,
    task_config: ContinualConfig,
    seed: int,
    epochs: int = 25,
    num_samples: int = 120,
) -> float:
    """Pretrain model on Environment A (identity association) with offline AdamW (plasticity disabled)."""
    model.online_plasticity_enabled = False
    task = ContinualDelayedAssociationTask(task_config)
    rng = np.random.default_rng(seed)

    # Generate training dataset on Env A
    total = task_config.total_steps
    X = np.zeros((num_samples, total, 2), dtype=np.float32)
    Y = np.zeros((num_samples, total, 1), dtype=np.float32)
    mask = np.zeros((num_samples, total, 1), dtype=np.float32)

    for i in range(num_samples):
        x_i, y_i, m_i, _ = task.generate_trial(i, "ENV_A", rng)
        X[i] = x_i
        Y[i] = y_i
        mask[i] = m_i

    optimizer = AdamWOptimizer(lr=0.015, weight_decay=1e-4)

    for epoch in range(epochs):
        perm = rng.permutation(num_samples)
        batch_size = 16
        for start_idx in range(0, num_samples, batch_size):
            b_idxs = perm[start_idx : start_idx + batch_size]
            b_X, b_Y, b_mask = X[b_idxs], Y[b_idxs], mask[b_idxs]

            B, T, _ = b_X.shape
            H = np.zeros((B, T, model.hidden_dim), dtype=np.float32)
            cand_H = np.zeros((B, T, model.hidden_dim), dtype=np.float32)
            A = np.zeros((B, T, model.hidden_dim), dtype=np.float32)
            Y_pred = np.zeros((B, T, model.output_dim), dtype=np.float32)

            alpha = model.alpha
            h_prev = np.zeros((B, model.hidden_dim), dtype=np.float32)

            for t in range(T):
                x_t = b_X[:, t, :]
                raw = np.dot(x_t, model.W_xh) + np.dot(h_prev, model.W_hh) + model.b_h
                c_t = np.tanh(raw)
                cand_H[:, t, :] = c_t
                h_t = (1.0 - alpha) * h_prev + alpha * c_t
                H[:, t, :] = h_t
                a_t = np.tanh(h_t)
                A[:, t, :] = a_t
                y_t = np.dot(a_t, model.W_hy) + model.b_y
                Y_pred[:, t, :] = y_t
                h_prev = h_t

            cache = {"H": H, "cand_H": cand_H, "A": A}
            grads = model.backward_sequence(b_X, Y_pred, b_Y, b_mask, cache)
            params = model.get_weights()
            optimizer.step(params, grads)
            model.set_weights(params)

    # Save pretrained weights as the baseline snapshot
    model.initial_W_hh = model.W_hh.copy()
    model.online_plasticity_enabled = True

    # Compute validation accuracy on Env A
    val_samples = 40
    correct = 0
    for i in range(val_samples):
        x_i, y_i, m_i, _ = task.generate_trial(i + 1000, "ENV_A", rng)
        model.reset()
        # Evaluate without updating
        model.online_plasticity_enabled = False
        y_seq, _, _ = model.forward_sequence(x_i[np.newaxis, :, :])
        model.online_plasticity_enabled = True
        probe_steps = np.where(m_i[:, 0] > 0)[0]
        pred_sign = np.sign(np.mean(y_seq[0, probe_steps, 0]))
        target_sign = np.sign(y_i[probe_steps[0], 0])
        if pred_sign == target_sign:
            correct += 1

    return correct / val_samples


def measure_msr(model: PlasticRNN, delay: int = 15, seed: int = 42, num_trials: int = 20) -> float:
    """Measure Mean Signal Retention (MSR) across cue and delay periods."""
    cfg = TaskConfig(t_cue=5, t_delay=delay, t_probe=5, distractor_sigma=0.0, seed=seed)
    task = DelayedMatchTask(cfg)
    rng = np.random.default_rng(seed)
    X, _, _, meta = task.generate_dataset(num_trials, seed)

    ratios = []
    # Temporarily freeze plasticity during retention probe
    saved_flag = model.online_plasticity_enabled
    model.online_plasticity_enabled = False

    for i in range(num_trials):
        model.reset()
        _, H, _ = model.forward_sequence(X[i : i + 1])
        h_cue = H[0, cfg.t_cue - 1]
        h_delay_end = H[0, cfg.t_cue + delay - 1]
        norm_cue = float(np.linalg.norm(h_cue)) + 1e-6
        norm_delay = float(np.linalg.norm(h_delay_end))
        ratios.append(norm_delay / norm_cue)

    model.online_plasticity_enabled = saved_flag
    return float(np.mean(ratios))


def run_continual_experiment(
    rule_name: str,
    condition: ContinualCondition,
    budget: int,
    seed: int,
    export_brainlog: bool = False,
) -> Dict[str, Any]:
    """Execute a single continual adaptation run under a specified condition."""
    hidden_dim = BUDGET_DIM_MAP[budget]
    rule = PLASTICITY_RULES[rule_name]()
    model = PlasticRNN(
        input_dim=2,
        hidden_dim=hidden_dim,
        output_dim=1,
        plasticity_rule=rule,
        seed=seed,
    )

    task_config = ContinualConfig(
        condition=condition,
        t_cue=5,
        t_delay=15,
        t_probe=5,
        distractor_sigma=0.05,
        seed=seed,
    )

    # 1. Pretrain on Env A
    pretrain_acc = pretrain_on_env_a(model, task_config, seed=seed, epochs=25)
    pre_msr = measure_msr(model, delay=15, seed=seed)

    # 2. Setup task sequence
    task = ContinualDelayedAssociationTask(task_config)
    X, Y, mask, metadata_list = task.generate_sequence(seed=seed)
    num_trials = X.shape[0]

    # Telemetry capture setup if requested
    capture = None
    topology = None
    if export_brainlog:
        topology = TopologyExtractor().extract(model, f"{rule_name}_{hidden_dim}units")
        capture = StateCapture(topology, dt_ms=40.0)

    # 3. Continual Online Adaptation Loop
    trial_correct = []
    trial_errors = []
    phase_accuracies: Dict[str, List[float]] = {}
    phase_trial_counts: Dict[str, int] = {}
    current_phase_name = None

    step_counter = 0

    for tr_idx in range(num_trials):
        x_tr = X[tr_idx]
        y_tr = Y[tr_idx]
        m_tr = mask[tr_idx]
        meta_tr = metadata_list[tr_idx]
        env_name = meta_tr["env_name"]

        if env_name not in phase_accuracies:
            phase_accuracies[env_name] = []
            phase_trial_counts[env_name] = 0
        phase_trial_counts[env_name] += 1

        model.reset()
        seq_len = x_tr.shape[0]
        y_pred_tr = np.zeros(seq_len, dtype=np.float32)

        # Single trial step-by-step with online plasticity
        for t in range(seq_len):
            x_t = x_tr[t : t + 1]
            y_target_t = y_tr[t, 0]
            mask_t = m_tr[t, 0]

            # Predictive error modulation signal for ModulatedPlasticity
            # Modulation is active during probe decision phase based on instantaneous error
            mod_val = None
            if mask_t > 0:
                # Immediate feedback prediction error
                est_err = abs(float(model.a[0, 0]) - float(y_target_t))
                mod_val = float(np.tanh(est_err * 2.0))
            elif meta_tr.get("env_switch", 0.0) > 0 and t < 5:
                # Environmental transition novelty pulse
                mod_val = 1.0

            y_t = model.step(x_t, modulation=mod_val)
            y_pred_tr[t] = y_t[0, 0]

            if capture:
                phase_label = "RUNNING"
                if meta_tr.get("env_switch", 0.0) > 0 and t == 0:
                    phase_label = f"ENV_SWITCH_{env_name}"
                elif t == 0:
                    phase_label = "CUE_START"
                elif t == task_config.t_cue + task_config.t_delay:
                    phase_label = "PROBE_START"

                capture.capture_step(
                    step=step_counter,
                    input_vec=x_t[0],
                    model=model,
                    y_val=float(y_t[0, 0]),
                    target_val=float(y_target_t),
                    mask_val=float(mask_t),
                    phase_label=phase_label,
                )
            step_counter += 1

        probe_steps = np.where(m_tr[:, 0] > 0)[0]
        mean_probe_pred = float(np.mean(y_pred_tr[probe_steps]))
        target_sign = np.sign(y_tr[probe_steps[0], 0])
        pred_sign = np.sign(mean_probe_pred)
        is_correct = float(pred_sign == target_sign)
        err = float(np.mean((y_pred_tr[probe_steps] - y_tr[probe_steps, 0]) ** 2))

        trial_correct.append(is_correct)
        trial_errors.append(err)
        phase_accuracies[env_name].append(is_correct)

    # 4. Measure Post-Adaptation Metrics
    overall_acc = float(np.mean(trial_correct))
    post_msr = measure_msr(model, delay=15, seed=seed)
    delta_msr = post_msr - pre_msr

    # Synaptic Plasticity Statistics
    w_initial = model.initial_W_hh
    w_final = model.W_hh
    dw_matrix = w_final - w_initial
    mean_abs_dw = float(np.mean(np.abs(dw_matrix)))
    max_dw = float(np.max(np.abs(dw_matrix)))
    w_norm = float(np.linalg.norm(w_final))
    fraction_plastic = float(np.mean(np.abs(dw_matrix) > 1e-4))

    # Stability checks
    is_stable = bool(w_norm < 3.0 * np.linalg.norm(w_initial) and not np.isnan(w_norm))

    # Condition-Specific Metrics
    adaptation_time = None
    recovery_time = None
    forgetting_a = None
    backward_transfer = None

    if condition == ContinualCondition.ABRUPT_SHIFT:
        # Phase 1: Env A (trials 0-49), Phase 2: Env B (trials 50-99)
        env_b_accs = trial_correct[50:100]
        # Find adaptation time (window of 5 trials with acc >= 0.8)
        for i in range(len(env_b_accs) - 4):
            if np.mean(env_b_accs[i : i + 5]) >= 0.8:
                adaptation_time = i
                break
        if adaptation_time is None:
            adaptation_time = 50

    elif condition == ContinualCondition.RECURRING:
        # Phase 1: Env A (0-39), Phase 2: Env B (40-79), Phase 3: Env A (80-119)
        acc_a_initial = float(np.mean(trial_correct[0:40]))
        acc_b = float(np.mean(trial_correct[40:80]))
        acc_a_recurring = float(np.mean(trial_correct[80:120]))
        # Catastrophic Forgetting = initial performance on A - recurring performance on A
        forgetting_a = float(acc_a_initial - acc_a_recurring)
        backward_transfer = float(acc_a_recurring - acc_a_initial)

        # Recovery time in Phase 3
        phase3_accs = trial_correct[80:120]
        for i in range(len(phase3_accs) - 4):
            if np.mean(phase3_accs[i : i + 5]) >= 0.85:
                recovery_time = i
                break
        if recovery_time is None:
            recovery_time = 40

    result = {
        "experiment_id": "EXP003",
        "plasticity_rule": rule_name,
        "condition": condition.value,
        "budget": budget,
        "parameter_count": model.parameter_count(),
        "seed": seed,
        "pretrain_accuracy": float(pretrain_acc),
        "overall_accuracy": round(overall_acc, 4),
        "mean_error": round(float(np.mean(trial_errors)), 4),
        "pre_msr": round(pre_msr, 4),
        "post_msr": round(post_msr, 4),
        "delta_msr": round(delta_msr, 4),
        "mean_abs_dw": round(mean_abs_dw, 6),
        "max_dw": round(max_dw, 6),
        "weight_norm": round(w_norm, 4),
        "fraction_plastic": round(fraction_plastic, 4),
        "is_stable": is_stable,
        "adaptation_time": adaptation_time,
        "recovery_time": recovery_time,
        "forgetting_a": round(forgetting_a, 4) if forgetting_a is not None else None,
        "backward_transfer": round(backward_transfer, 4) if backward_transfer is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save .brainlog if requested
    if export_brainlog and capture and topology:
        brainlog_path = Path(f"experiments/results/brainlogs/EXP003_{rule_name}_rep.brainlog")
        blog = BrainLog(
            metadata={
                "experiment_id": "EXP003",
                "model_id": f"PLASTIC_{rule_name.upper()}",
                "rule": rule_name,
                "condition": condition.value,
                "budget": budget,
                "parameter_count": model.parameter_count(),
                "seed": seed,
                "overall_accuracy": overall_acc,
                "mean_abs_dw": mean_abs_dw,
                "timestamp": result["timestamp"],
            },
            topology=topology,
            frames=capture.frames,
            events=capture.all_events,
            snapshots={},
            version="2.1",
        )
        blog.save(brainlog_path)
        result["brainlog_file"] = str(brainlog_path)

    return result


def run_exp003_matrix() -> List[Dict[str, Any]]:
    """Execute complete Experiment 003 matrix across rules, conditions, budgets, and seeds."""
    print("=" * 86)
    print("3D-BRAIN — EXPERIMENT 003: ONLINE PLASTICITY & CONTINUAL ADAPTATION MATRIX")
    print("=" * 86)

    results: List[Dict[str, Any]] = []
    rules = ["Frozen", "Hebbian", "Oja", "Modulated", "Homeostatic"]
    conditions = [
        ContinualCondition.STATIONARY,
        ContinualCondition.ABRUPT_SHIFT,
        ContinualCondition.RECURRING,
        ContinualCondition.MULTIPLE,
        ContinualCondition.GRADUAL_DRIFT,
    ]
    seeds = [42, 137, 256]
    budgets = [300, 1000]

    # Pre-select representative configurations for .brainlog export
    rep_exported = set()

    total_runs = len(rules) * len(conditions) * len(seeds) + len(rules) * 2 * 2  # core + scaling
    run_idx = 0

    # 1. Core Matrix: Budget 300, all rules, all conditions, 3 seeds (75 runs)
    for rule_name in rules:
        for cond in conditions:
            for seed in seeds:
                run_idx += 1
                should_export = (rule_name not in rep_exported and cond == ContinualCondition.RECURRING and seed == 42)
                if should_export:
                    rep_exported.add(rule_name)

                t0 = time.time()
                res = run_continual_experiment(
                    rule_name=rule_name,
                    condition=cond,
                    budget=300,
                    seed=seed,
                    export_brainlog=should_export,
                )
                dt = time.time() - t0
                results.append(res)

                forg_str = f"| Forg: {res['forgetting_a']:+.2f}" if res['forgetting_a'] is not None else ""
                print(
                    f"[{run_idx:02d}] {rule_name:<11} | Cond: {cond.value:<13} | "
                    f"Acc: {res['overall_accuracy']*100:5.1f}% | ΔW: {res['mean_abs_dw']:7.5f} {forg_str} | {dt:.1f}s"
                )

    # 2. Scaling Ablation: Budgets 1000 and 3000 on RECURRING condition (20 runs)
    print("\n--- Scaling Ablation: Parameter Budgets 1000 & 3000 ---")
    for b in [1000, 3000]:
        for rule_name in rules:
            for seed in [42, 137]:
                run_idx += 1
                t0 = time.time()
                res = run_continual_experiment(
                    rule_name=rule_name,
                    condition=ContinualCondition.RECURRING,
                    budget=b,
                    seed=seed,
                    export_brainlog=False,
                )
                dt = time.time() - t0
                results.append(res)
                print(
                    f"[{run_idx:02d}] {rule_name:<11} | Budget: {b:4d} | Seed: {seed:3d} | "
                    f"Acc: {res['overall_accuracy']*100:5.1f}% | Forg: {res['forgetting_a']:+.2f} | {dt:.1f}s"
                )

    # Save complete results
    results_path = Path("experiments/results/exp003_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 86)
    print(f"Saved complete machine-readable results to: {results_path}")
    print(f"Total runs executed: {len(results)}")
    print("=" * 86)

    return results


if __name__ == "__main__":
    run_exp003_matrix()
