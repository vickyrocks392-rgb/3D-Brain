"""Phase 23A Master Experiment Runner: Sample Efficiency Benchmark & Resource Accounting.

Executes head-to-head sample efficiency comparison across:
- 6 Architectures:
  1. MIN_E (Minimal Sufficient Biological Architecture)
  2. MIN_E_PLUS_EPISODIC (MIN_E + Sparse Episodic Memory)
  3. SIMPLE_RECURRENT (Discrete Leaky RNN with local Hebbian readout)
  4. GRU (Standard Gated Recurrent Unit)
  5. CAUSAL_TRANSFORMER (Small Causal Attention Architecture)
  6. TYPED_PROBABILISTIC_DECISION (System One-Inspired Categorical Decision Model)

- 5 Controlled Tasks:
  1. STATIC_CLASSIFICATION
  2. TEMPORAL_CLASSIFICATION
  3. EPISODE_DEPENDENT_DECISION
  4. DELAYED_CONTEXT_DECISION
  5. SIMPLE_COMPOSITIONAL_DECISION

- 7 Sample Budgets:
  [5, 10, 25, 50, 100, 250, 500]

- 5 Independent Seeds:
  [42, 101, 2024, 777, 999]

Calculates:
- Full sample-efficiency curves
- Threshold examples to 70%, 80%, 90% accuracy
- Catastrophic forgetting (Task 1 retention after sequential learning)
- Zero-shot / compositional generalization
- Rigorous statistical summarization: mean, sample std, 95% CI, Cohen's d effect sizes

Emits results to:
  experiments/results/PHASE_23A_RESULTS.json
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.phase23a_sample_efficiency import (
    BenchmarkTrial,
    SampleEfficiencyRunner,
    SystemArchitecture,
    TaskType,
    TRAINING_BUDGETS,
    EVAL_TRIALS_COUNT,
    ACCURACY_THRESHOLDS,
    get_system_accounting,
    get_task_datasets,
    get_process_peak_rss_mb,
)

SEEDS: List[int] = [42, 101, 2024, 777, 999]
TASKS_TO_RUN: List[TaskType] = [
    TaskType.STATIC_CLASSIFICATION,
    TaskType.TEMPORAL_CLASSIFICATION,
    TaskType.EPISODE_DEPENDENT_DECISION,
    TaskType.DELAYED_CONTEXT_DECISION,
    TaskType.SIMPLE_COMPOSITIONAL_DECISION,
]
SYSTEMS_TO_RUN: List[SystemArchitecture] = [
    SystemArchitecture.MIN_E,
    SystemArchitecture.MIN_E_PLUS_EPISODIC,
    SystemArchitecture.SIMPLE_RECURRENT,
    SystemArchitecture.GRU,
    SystemArchitecture.CAUSAL_TRANSFORMER,
    SystemArchitecture.TYPED_PROBABILISTIC_DECISION,
]


def summarize(values: List[float]) -> Dict[str, Any]:
    """Calculate mean, sample std, 95% CI (t-critical for n=5 is 2.776), and exact values."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": [0.0, 0.0], "n": 0, "values": []}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        tcrit = 2.776 if n == 5 else float(np.abs(np.percentile(np.random.standard_t(df=n-1, size=100000), 97.5)))
        margin = tcrit * sd / np.sqrt(n)
    else:
        margin = 0.0
    return {
        "mean": round(mean, 4),
        "std": round(sd, 4),
        "ci95": [round(float(mean - margin), 4), round(float(mean + margin), 4)],
        "n": n,
        "values": [round(float(v), 4) for v in arr],
    }


def cohens_d(a: List[float], b: List[float]) -> float:
    """Calculate paired Cohen's d effect size."""
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    if sd < 1e-9:
        return 0.0
    return float(np.mean(diffs) / sd)


def compute_examples_to_threshold(
    curve: Dict[int, float],
    threshold: float,
) -> Optional[int]:
    """Determine minimum training examples required to stably reach accuracy threshold."""
    for budget in sorted(curve.keys()):
        if curve[budget] >= threshold:
            return budget
    return None


def run_catastrophic_forgetting_evaluation(seeds: List[int]) -> Dict[str, Any]:
    """Measure retention on Task A after sequential learning on Task B."""
    results: Dict[str, List[float]] = defaultdict(list)

    for seed in seeds:
        runner = SampleEfficiencyRunner(seed=seed)
        # Task 1: Static classification
        tr_a, ev_a = get_task_datasets(TaskType.STATIC_CLASSIFICATION, train_budget=50, eval_count=30, seed=seed)
        # Task 2: Temporal classification
        tr_b, _ = get_task_datasets(TaskType.TEMPORAL_CLASSIFICATION, train_budget=50, eval_count=30, seed=seed)

        for sys_arch in SYSTEMS_TO_RUN:
            # 1. Train on Task A
            model, _ = runner._train_model(sys_arch, TaskType.STATIC_CLASSIFICATION, tr_a, seed=seed)
            # 2. Intervening training on Task B
            # Sequential training step
            if sys_arch == SystemArchitecture.MIN_E:
                for tr in tr_b:
                    model.reset()
                    for t in range(len(tr.inputs)):
                        model.step(tr.inputs[t], target_action=int(tr.target_action) if t == tr.eval_step else None)
            elif sys_arch == SystemArchitecture.MIN_E_PLUS_EPISODIC:
                net, mem = model
                for tr in tr_b:
                    net.reset()
                    mem.open_episode(
                        episode_id=len(mem.records),
                        context_vector=tr.inputs[0],
                        event_sequence=tr.inputs[0],
                        chosen_action=tr.target_action,
                        timestamp_ms=0.0,
                        task_name="TaskB",
                    )
                    mem.close_episode(prediction_error=0.0, reward_val=1.0)
                    for t in range(len(tr.inputs)):
                        net.step(tr.inputs[t], target_action=int(tr.target_action) if t == tr.eval_step else None)
            elif sys_arch == SystemArchitecture.SIMPLE_RECURRENT:
                rnn, W_ro = model
                for tr in tr_b:
                    rnn.reset()
                    for t in range(len(tr.inputs)):
                        rnn.step(tr.inputs[t])
                        if t == tr.eval_step:
                            h = rnn.a[0]
                            pred = h @ W_ro
                            t_onehot = np.zeros(2, dtype=np.float32)
                            t_onehot[tr.target_action] = 1.0
                            W_ro += 0.05 * np.outer(h, t_onehot - pred)
            elif sys_arch == SystemArchitecture.GRU:
                for tr in tr_b:
                    X = np.expand_dims(tr.inputs, axis=0)
                    Y, H, _, _ = model.forward_sequence(X)
                    t_oh = np.zeros(2, dtype=np.float32)
                    t_oh[tr.target_action] = 1.0
                    grad_out = Y[0, tr.eval_step] - t_oh
                    model.W_hy -= 0.04 * np.outer(H[0, tr.eval_step], grad_out)
                    model.b_y -= 0.04 * grad_out
            elif sys_arch == SystemArchitecture.CAUSAL_TRANSFORMER:
                for tr in tr_b:
                    model.train_step(tr.inputs, float(tr.target_action), action_step_idx=tr.eval_step, lr=0.03)
            elif sys_arch == SystemArchitecture.TYPED_PROBABILISTIC_DECISION:
                for tr in tr_b:
                    model.train_step(tr.inputs.mean(axis=0), tr.target_action)

            # 3. Retest on Task A (Frozen retention)
            correct = 0
            for tr in ev_a:
                pred = runner._eval_step(sys_arch, model, tr)
                if pred == tr.target_action:
                    correct += 1
            retention_acc = float(correct / len(ev_a))
            results[sys_arch.value].append(retention_acc)

    return {sys_k: summarize(accs) for sys_k, accs in results.items()}


def run_compositional_generalization_evaluation(seeds: List[int]) -> Dict[str, Any]:
    """Measure zero-shot transfer on held-out compositional combinations."""
    results: Dict[str, List[float]] = defaultdict(list)

    for seed in seeds:
        runner = SampleEfficiencyRunner(seed=seed)
        all_comp_trials = get_task_datasets(TaskType.SIMPLE_COMPOSITIONAL_DECISION, train_budget=100, eval_count=40, seed=seed)
        train_trials, eval_trials = all_comp_trials

        for sys_arch in SYSTEMS_TO_RUN:
            model, _ = runner._train_model(sys_arch, TaskType.SIMPLE_COMPOSITIONAL_DECISION, train_trials, seed=seed)
            correct = 0
            for tr in eval_trials:
                pred = runner._eval_step(sys_arch, model, tr)
                if pred == tr.target_action:
                    correct += 1
            results[sys_arch.value].append(float(correct / len(eval_trials)))

    return {sys_k: summarize(accs) for sys_k, accs in results.items()}


def run_phase23a_experiments() -> Dict[str, Any]:
    """Run all Phase 23A benchmarks, compute statistical metrics, and format results."""
    print("=" * 80)
    print("PHASE 23A: SAMPLE-EFFICIENCY BENCHMARK & HUMAN-DRIVEN OBSERVABILITY")
    print("=" * 80)
    start_total_time = time.time()

    runner = SampleEfficiencyRunner()
    raw_results: Dict[str, Any] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    total_runs = len(TASKS_TO_RUN) * len(SYSTEMS_TO_RUN) * len(TRAINING_BUDGETS) * len(SEEDS)
    completed_runs = 0
    print(f"Beginning {total_runs} controlled training & frozen evaluation runs across {len(SEEDS)} seeds...\n")

    for task in TASKS_TO_RUN:
        print(f"--- Task: {task.value} ---")
        for sys_arch in SYSTEMS_TO_RUN:
            for budget in TRAINING_BUDGETS:
                for s in SEEDS:
                    res = runner.train_and_eval_system(
                        system=sys_arch,
                        task=task,
                        train_budget=budget,
                        eval_count=EVAL_TRIALS_COUNT,
                        seed=s,
                    )
                    raw_results[task.value][sys_arch.value][str(budget)].append(res)
                    completed_runs += 1
            # Brief progress update
            mean_50 = np.mean([r["test_accuracy"] for r in raw_results[task.value][sys_arch.value]["50"]])
            print(f"  {sys_arch.value:<30} | Budget 50 Acc: {mean_50*100:.1f}%")

    print("\n--- Running Catastrophic Forgetting & Generalization Protocols ---")
    forgetting_results = run_catastrophic_forgetting_evaluation(SEEDS)
    generalization_results = run_compositional_generalization_evaluation(SEEDS)

    # Compile analyzed results
    analyzed_tasks: Dict[str, Any] = {}
    resource_accounting_summary: Dict[str, Any] = {}

    for sys_arch in SYSTEMS_TO_RUN:
        resource_accounting_summary[sys_arch.value] = get_system_accounting(sys_arch, input_dim=16, seq_len=6)

    for task in TASKS_TO_RUN:
        t_key = task.value
        task_summary: Dict[str, Any] = {
            "systems": {},
            "sample_efficiency_curves": {},
            "threshold_analysis": {},
            "effect_sizes_vs_mine": {},
        }

        # Calculate sample efficiency curves & threshold analysis
        for sys_arch in SYSTEMS_TO_RUN:
            s_key = sys_arch.value
            curve_points: Dict[str, Any] = {}
            mean_curve: Dict[int, float] = {}

            for budget in TRAINING_BUDGETS:
                b_str = str(budget)
                runs = raw_results[t_key][s_key][b_str]
                accs = [r["test_accuracy"] for r in runs]
                flops_list = [r["training_flops"] for r in runs]
                wall_clock_list = [r["training_wall_clock_sec"] for r in runs]

                summary_acc = summarize(accs)
                mean_curve[budget] = summary_acc["mean"]
                curve_points[b_str] = {
                    "accuracy": summary_acc,
                    "training_flops_mean": int(np.mean(flops_list)),
                    "training_wall_clock_sec_mean": round(float(np.mean(wall_clock_list)), 5),
                }

            # Threshold analysis
            thresholds_reached: Dict[str, Any] = {}
            for th in ACCURACY_THRESHOLDS:
                req_b = compute_examples_to_threshold(mean_curve, th)
                thresholds_reached[f"threshold_{int(th*100)}pct"] = req_b

            task_summary["systems"][s_key] = {
                "curve": curve_points,
                "thresholds": thresholds_reached,
                "accounting": resource_accounting_summary[s_key],
            }
            task_summary["sample_efficiency_curves"][s_key] = {
                b: curve_points[str(b)]["accuracy"]["mean"] for b in TRAINING_BUDGETS
            }
            task_summary["threshold_analysis"][s_key] = thresholds_reached

        # Compute effect sizes vs MIN_E at Budget 50 & Budget 100
        mine_key = SystemArchitecture.MIN_E.value
        effect_sizes: Dict[str, Any] = {}
        for s_key in task_summary["systems"]:
            if s_key == mine_key:
                continue
            mine_accs_50 = [r["test_accuracy"] for r in raw_results[t_key][mine_key]["50"]]
            base_accs_50 = [r["test_accuracy"] for r in raw_results[t_key][s_key]["50"]]
            d_50 = cohens_d(mine_accs_50, base_accs_50)

            mine_accs_100 = [r["test_accuracy"] for r in raw_results[t_key][mine_key]["100"]]
            base_accs_100 = [r["test_accuracy"] for r in raw_results[t_key][s_key]["100"]]
            d_100 = cohens_d(mine_accs_100, base_accs_100)

            effect_sizes[s_key] = {
                "cohens_d_budget_50": round(d_50, 3),
                "cohens_d_budget_100": round(d_100, 3),
            }
        task_summary["effect_sizes_vs_mine"] = effect_sizes
        analyzed_tasks[t_key] = task_summary

    total_wall_clock = time.time() - start_total_time
    print(f"\nCompleted in {total_wall_clock:.2f} seconds.")

    final_payload: Dict[str, Any] = {
        "metadata": {
            "phase": "23A",
            "title": "Sample-Efficiency Benchmark & Human-Driven Brain Observability",
            "date": "2026-10-06",
            "seeds": SEEDS,
            "training_budgets": TRAINING_BUDGETS,
            "tasks_evaluated": [t.value for t in TASKS_TO_RUN],
            "architectures_evaluated": [s.value for s in SYSTEMS_TO_RUN],
            "evaluation_trials_per_run": EVAL_TRIALS_COUNT,
            "total_runs": total_runs,
            "wall_clock_total_sec": round(total_wall_clock, 2),
            "disclaimer_typed_probabilistic": (
                "The TYPED_PROBABILISTIC_DECISION model is an independent, non-proprietary "
                "baseline inspired only by publicly describable System One interface characteristics. "
                "It is NOT the proprietary JEV model from Typesafe AI."
            ),
        },
        "resource_accounting": resource_accounting_summary,
        "task_results": analyzed_tasks,
        "catastrophic_forgetting": forgetting_results,
        "compositional_generalization": generalization_results,
    }

    out_file = Path("experiments/results/PHASE_23A_RESULTS.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nSuccessfully wrote results to: {out_file.resolve()}")
    return final_payload


if __name__ == "__main__":
    run_phase23a_experiments()
