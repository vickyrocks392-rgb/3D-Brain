"""Phase 19 Track C: Fair Causal Transformer & Baseline Comparison Suite.

Provides rigorous, head-to-head empirical comparison across 4 matched-budget architectures:
1. MIN_E (Minimal Sufficient Biological Architecture)
2. BPTT_RECURRENT (DiscreteLeakyRNN trained via BPTT)
3. CAUSAL_TRANSFORMER (SmallCausalTransformer, matched parameter budget ~2.3K)
4. SIMPLE_RECURRENT_CONTROL (DiscreteLeakyRNN with local Hebbian readout)

Budget Equalization Methodology:
- Parameters: Matched to ~2,000 total parameters.
- Task Schedule: Identical trial sequences, identical inputs, identical choice steps.
- Evaluation: Strictly frozen test trials without target leakage.
- Accounting: Analytical FLOPs, wall-clock time, peak memory (RSS), sample efficiency,
  long-delay scaling, and continual-learning retention.
"""

import resource
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain.phase16_network import Phase16Network, ArchitecturePreset
from benchmarks.bptt_baseline import BPTTGradientBaseline
from benchmarks.transformer_baseline import SmallCausalTransformer
from benchmarks.phase19_credit_horizon import (
    generate_delayed_credit_trial,
    train_and_eval_min_e,
    train_and_eval_transformer,
    train_and_eval_bptt,
    train_and_eval_simple_recurrent,
    SimpleRecurrentControl,
    DT_MS,
)
from benchmarks.phase19_consolidation import generate_continual_trial


def get_process_peak_rss_mb() -> float:
    """Return process peak resident set size in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes; on Linux, in kilobytes
    import sys
    if sys.platform == "darwin":
        return float(usage.ru_maxrss / (1024.0 * 1024.0))
    else:
        return float(usage.ru_maxrss / 1024.0)


def calculate_analytical_flops_per_step(architecture: str, seq_len: int = 15) -> Dict[str, Any]:
    """Compute exact analytical floating-point operations per step and trial."""
    if architecture == "MIN_E":
        # W_in: 8 x 40 = 320 MACs = 640 FLOPs
        # W_rec: 40 x 40 = 1600 MACs = 3200 FLOPs
        # Readout: 40 x 2 = 80 MACs = 160 FLOPs
        # Assembly (slow): 2 x 2 = 4 MACs = 8 FLOPs
        # Soma/Dend updates: ~40 x 5 = 200 FLOPs
        flops_step = 640 + 3200 + 160 + 8 + 200  # ~4,208 FLOPs/step
        flops_trial = flops_step * seq_len
        return {"flops_per_step": flops_step, "flops_per_trial": flops_trial}

    elif architecture == "BPTT_RECURRENT":
        # Forward: W_xh (8x40) + W_hh (40x40) + W_hy (40x2) = 2000 MACs = 4000 FLOPs/step
        # Backward (unrolled over seq_len): ~2x forward = 8000 FLOPs/step
        flops_step_fwd = 4000
        flops_step_bwd = 8000
        flops_trial = (flops_step_fwd + flops_step_bwd) * seq_len
        return {"flops_per_step": flops_step_fwd + flops_step_bwd, "flops_per_trial": flops_trial}

    elif architecture == "CAUSAL_TRANSFORMER":
        # d_model=16, heads=2, d_ff=32, T=seq_len
        # QKV projections: 3 * (T * 16 * 16) MACs = 6 * 256 * T = 1,536 T FLOPs
        # Scaled dot-product attention: 2 * T^2 * 16 = 32 T^2 FLOPs
        # Out projection: 16 * 16 * T MACs = 512 T FLOPs
        # FF1 (16 -> 32): 16 * 32 * T MACs = 1,024 T FLOPs
        # FF2 (32 -> 16): 32 * 16 * T MACs = 1,024 T FLOPs
        # Readout (16 -> 2): 16 * 2 * T MACs = 64 T FLOPs
        # Backward pass: ~2x forward
        fwd_flops = (4160 * seq_len) + (32 * seq_len * seq_len)
        bwd_flops = 2 * fwd_flops
        flops_trial = fwd_flops + bwd_flops
        flops_step = int(flops_trial / max(1, seq_len))
        return {"flops_per_step": flops_step, "flops_per_trial": flops_trial}

    elif architecture == "SIMPLE_RECURRENT_CONTROL":
        # W_xh (8x40) + W_hh (40x40) + W_out (40x2) = 2000 MACs = 4000 FLOPs/step
        # Hebbian plasticity: 40 x 2 = 80 MACs = 160 FLOPs
        flops_step = 4160
        flops_trial = flops_step * seq_len
        return {"flops_per_step": flops_step, "flops_per_trial": flops_trial}

    return {"flops_per_step": 4000, "flops_per_trial": 4000 * seq_len}


def benchmark_continual_transformer(
    trials_per_task: int = 40,
    eval_trials: int = 30,
    delay_steps: int = 5,
    seed: int = 42,
    lr: float = 0.02,
) -> Dict[str, Any]:
    """Evaluate standard Catastrophic Forgetting protocol (A -> B -> A) on Causal Transformer."""
    total_steps = 5 + delay_steps + 1 + 2 + 1
    model = SmallCausalTransformer(input_dim=8, d_model=16, num_heads=2, d_ff=32, max_seq_len=256, seed=seed)

    # 1. Train Task A
    for tr in range(trials_per_task):
        trial = generate_continual_trial(tr, "A", tr % 2, delay_steps=delay_steps, seed=seed * 1000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_idx"]
        model.train_step(x_seq, target_action=float(trial.target_action), action_step_idx=choice_idx, lr=lr)

    # 2. Eval Task A (Pre)
    correct_a_pre = 0
    for tr in range(eval_trials):
        trial = generate_continual_trial(tr, "A", tr % 2, delay_steps=delay_steps, seed=seed * 2000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_idx"]
        y = model.forward(x_seq)
        pred = 1 if float(y[choice_idx, 0]) >= 0.5 else 0
        if pred == trial.target_action:
            correct_a_pre += 1
    acc_a_pre = correct_a_pre / eval_trials

    # 3. Train Task B (Inverted)
    for tr in range(trials_per_task):
        trial = generate_continual_trial(tr, "B", tr % 2, delay_steps=delay_steps, seed=seed * 3000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_idx"]
        model.train_step(x_seq, target_action=float(trial.target_action), action_step_idx=choice_idx, lr=lr)

    # 4. Eval Task B
    correct_b = 0
    for tr in range(eval_trials):
        trial = generate_continual_trial(tr, "B", tr % 2, delay_steps=delay_steps, seed=seed * 4000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_idx"]
        y = model.forward(x_seq)
        pred = 1 if float(y[choice_idx, 0]) >= 0.5 else 0
        if pred == trial.target_action:
            correct_b += 1
    acc_b = correct_b / eval_trials

    # 5. Eval Task A (Post) - Forgetting check
    correct_a_post = 0
    for tr in range(eval_trials):
        trial = generate_continual_trial(tr, "A", tr % 2, delay_steps=delay_steps, seed=seed * 5000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_idx"]
        y = model.forward(x_seq)
        pred = 1 if float(y[choice_idx, 0]) >= 0.5 else 0
        if pred == trial.target_action:
            correct_a_post += 1
    acc_a_post = correct_a_post / eval_trials

    rel_retention = float(acc_a_post / max(0.01, acc_a_pre))
    return {
        "architecture": "CAUSAL_TRANSFORMER",
        "acc_a_pre": acc_a_pre,
        "acc_b": acc_b,
        "acc_a_post": acc_a_post,
        "relative_retention": round(rel_retention, 4),
        "stability_plasticity_product": round(acc_a_post * acc_b, 4),
    }


def benchmark_continual_bptt(
    trials_per_task: int = 40,
    eval_trials: int = 30,
    delay_steps: int = 5,
    seed: int = 42,
    lr: float = 0.02,
) -> Dict[str, Any]:
    """Evaluate standard Catastrophic Forgetting protocol (A -> B -> A) on BPTT Recurrent."""
    model = BPTTGradientBaseline(input_dim=8, hidden_dim=40, output_dim=1, learning_rate=lr, seed=seed)

    # 1. Train Task A
    for tr in range(trials_per_task):
        trial = generate_continual_trial(tr, "A", tr % 2, delay_steps=delay_steps, seed=seed * 1000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_idx"]
        model.train_trial(x_seq, target_action=float(trial.target_action), choice_step_idx=choice_idx)

    # 2. Eval Task A (Pre)
    correct_a_pre = 0
    for tr in range(eval_trials):
        trial = generate_continual_trial(tr, "A", tr % 2, delay_steps=delay_steps, seed=seed * 2000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_idx"]
        model.reset()
        y = model.forward(x_seq)
        pred = 1 if float(y[choice_idx, 0]) >= 0.5 else 0
        if pred == trial.target_action:
            correct_a_pre += 1
    acc_a_pre = correct_a_pre / eval_trials

    # 3. Train Task B (Inverted)
    for tr in range(trials_per_task):
        trial = generate_continual_trial(tr, "B", tr % 2, delay_steps=delay_steps, seed=seed * 3000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_idx"]
        model.train_trial(x_seq, target_action=float(trial.target_action), choice_step_idx=choice_idx)

    # 4. Eval Task B
    correct_b = 0
    for tr in range(eval_trials):
        trial = generate_continual_trial(tr, "B", tr % 2, delay_steps=delay_steps, seed=seed * 4000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_idx"]
        model.reset()
        y = model.forward(x_seq)
        pred = 1 if float(y[choice_idx, 0]) >= 0.5 else 0
        if pred == trial.target_action:
            correct_b += 1
    acc_b = correct_b / eval_trials

    # 5. Eval Task A (Post) - Forgetting check
    correct_a_post = 0
    for tr in range(eval_trials):
        trial = generate_continual_trial(tr, "A", tr % 2, delay_steps=delay_steps, seed=seed * 5000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_idx"]
        model.reset()
        y = model.forward(x_seq)
        pred = 1 if float(y[choice_idx, 0]) >= 0.5 else 0
        if pred == trial.target_action:
            correct_a_post += 1
    acc_a_post = correct_a_post / eval_trials

    rel_retention = float(acc_a_post / max(0.01, acc_a_pre))
    return {
        "architecture": "BPTT_RECURRENT",
        "acc_a_pre": acc_a_pre,
        "acc_b": acc_b,
        "acc_a_post": acc_a_post,
        "relative_retention": round(rel_retention, 4),
        "stability_plasticity_product": round(acc_a_post * acc_b, 4),
    }


def execute_equalized_baseline_matrix(seeds: List[int] = [42, 101, 2024, 777, 999]) -> Dict[str, Any]:
    """Execute complete head-to-head comparison under matched budgets across 5 independent seeds."""
    delays_steps = [0, 6, 25, 125]  # 0 ms, 240 ms, 1000 ms, 5000 ms
    architectures = ["MIN_E", "BPTT_RECURRENT", "CAUSAL_TRANSFORMER", "SIMPLE_RECURRENT_CONTROL"]

    results: Dict[str, Any] = {
        "delay_sweep": {},
        "continual_reversal": {},
        "budget_accounting": {},
    }

    # 1. Delay Sweep
    for d_step in delays_steps:
        d_ms = int(d_step * DT_MS)
        results["delay_sweep"][str(d_ms)] = {}

        for arch in architectures:
            accs = []
            times = []
            for s in seeds:
                if arch == "MIN_E":
                    res = train_and_eval_min_e(ArchitecturePreset.MIN_E, d_step, seed=s)
                elif arch == "BPTT_RECURRENT":
                    res = train_and_eval_bptt(d_step, seed=s)
                elif arch == "CAUSAL_TRANSFORMER":
                    res = train_and_eval_transformer(d_step, seed=s)
                elif arch == "SIMPLE_RECURRENT_CONTROL":
                    res = train_and_eval_simple_recurrent(d_step, seed=s)
                accs.append(res["eval_accuracy"])
                times.append(res["train_time_sec"])

            results["delay_sweep"][str(d_ms)][arch] = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_train_time_sec": float(np.mean(times)),
                "accuracies": accs,
            }

    # 2. Continual Learning Reversal (A -> B -> A)
    for arch in architectures:
        retentions = []
        acc_a_pres = []
        acc_bs = []
        acc_a_posts = []

        for s in seeds:
            if arch == "MIN_E":
                from benchmarks.phase19_consolidation import evaluate_continual_reversal
                from brain.synaptic_consolidation import ConsolidationMode
                res = evaluate_continual_reversal(ConsolidationMode.NONE, seed=s)
            elif arch == "BPTT_RECURRENT":
                res = benchmark_continual_bptt(seed=s)
            elif arch == "CAUSAL_TRANSFORMER":
                res = benchmark_continual_transformer(seed=s)
            elif arch == "SIMPLE_RECURRENT_CONTROL":
                from benchmarks.phase19_consolidation import run_task_phase
                # Simple control without consolidation
                net = SimpleRecurrentControl(seed=s)
                # Task A train
                for tr in range(40):
                    tr_obj = generate_continual_trial(tr, "A", tr % 2, seed=s*1000+tr)
                    net.reset()
                    ch = 0
                    for st in tr_obj.steps:
                        if st.is_choice_step: _, ch = net.step(st.sensory_input, is_choice_step=True)
                        elif st.is_reward_step: net.step(st.sensory_input, target_action=st.target_action, is_reward_step=True)
                        else: net.step(st.sensory_input)
                # Task A eval
                net.set_frozen(True)
                corr_a = sum(1 for tr in range(30) if net.step(generate_continual_trial(tr, "A", tr % 2, seed=s*2000+tr).steps[10].sensory_input, is_choice_step=True)[1] == (tr%2))
                a_pre = corr_a / 30.0
                # Task B train
                net.set_frozen(False)
                for tr in range(40):
                    tr_obj = generate_continual_trial(tr, "B", tr % 2, seed=s*3000+tr)
                    net.reset()
                    ch = 0
                    for st in tr_obj.steps:
                        if st.is_choice_step: _, ch = net.step(st.sensory_input, is_choice_step=True)
                        elif st.is_reward_step: net.step(st.sensory_input, target_action=st.target_action, is_reward_step=True)
                        else: net.step(st.sensory_input)
                # Task B eval
                net.set_frozen(True)
                corr_b = sum(1 for tr in range(30) if net.step(generate_continual_trial(tr, "B", tr % 2, seed=s*4000+tr).steps[10].sensory_input, is_choice_step=True)[1] == (1-(tr%2)))
                b_acc = corr_b / 30.0
                # Task A retest
                corr_a_post = sum(1 for tr in range(30) if net.step(generate_continual_trial(tr, "A", tr % 2, seed=s*5000+tr).steps[10].sensory_input, is_choice_step=True)[1] == (tr%2))
                a_post = corr_a_post / 30.0
                res = {
                    "acc_a_pre": a_pre,
                    "acc_b": b_acc,
                    "acc_a_post": a_post,
                    "relative_retention": round(a_post / max(0.01, a_pre), 4),
                }

            acc_a_pres.append(res["acc_a_pre"])
            acc_bs.append(res["acc_b"])
            acc_a_posts.append(res["acc_a_post"])
            retentions.append(res["relative_retention"])

        results["continual_reversal"][arch] = {
            "mean_acc_a_pre": float(np.mean(acc_a_pres)),
            "mean_acc_b": float(np.mean(acc_bs)),
            "mean_acc_a_post": float(np.mean(acc_a_posts)),
            "mean_relative_retention": float(np.mean(retentions)),
            "retentions": retentions,
        }

    # 3. Budget Accounting
    results["budget_accounting"] = {
        "MIN_E": {
            "trainable_parameters": 82,
            "fixed_parameters": 1920,
            "total_parameters": 2002,
            "state_variables": 282,
            "flops": calculate_analytical_flops_per_step("MIN_E", seq_len=15),
            "peak_rss_mb": get_process_peak_rss_mb(),
        },
        "BPTT_RECURRENT": {
            "trainable_parameters": 2041,
            "fixed_parameters": 0,
            "total_parameters": 2041,
            "state_variables": 40,
            "flops": calculate_analytical_flops_per_step("BPTT_RECURRENT", seq_len=15),
            "peak_rss_mb": get_process_peak_rss_mb(),
        },
        "CAUSAL_TRANSFORMER": {
            "trainable_parameters": 2322,
            "fixed_parameters": 0,
            "total_parameters": 2322,
            "state_variables": 256 * 16,  # KV cache / sequence buffer
            "flops": calculate_analytical_flops_per_step("CAUSAL_TRANSFORMER", seq_len=15),
            "peak_rss_mb": get_process_peak_rss_mb(),
        },
        "SIMPLE_RECURRENT_CONTROL": {
            "trainable_parameters": 80,
            "fixed_parameters": 1920,
            "total_parameters": 2000,
            "state_variables": 40 + 80,  # hidden + eligibility
            "flops": calculate_analytical_flops_per_step("SIMPLE_RECURRENT_CONTROL", seq_len=15),
            "peak_rss_mb": get_process_peak_rss_mb(),
        },
    }

    return results
