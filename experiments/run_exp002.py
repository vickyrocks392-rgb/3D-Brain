"""CLI driver for executing Experiment 002: Leaky Dynamics & Gate Initialization Ablation.

Compares BASE-RNN, DISC-LRNN, CAND-LI, and BASE-GRU-MemBias across delays T in {10, 25, 50, 100}
under matched parameter budgets.
"""

from collections import defaultdict
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.runner import ExperimentConfig, run_experiment


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def run_exp002_matrix(
    models=("BASE-RNN", "DISC-LRNN", "CAND-LI", "BASE-GRU-MemBias"),
    budgets=(300, 1000),
    delays=(10, 25, 50, 100),
    seeds=(42, 137, 256),
    epochs=70,
):
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    brainlog_dir = results_dir / "brainlogs"
    brainlog_dir.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()

    all_results = []
    total_runs = len(models) * len(budgets) * len(delays) * len(seeds)
    run_idx = 0

    print("=" * 86)
    print("3D-BRAIN: EXPERIMENT 002 - LEAKY DYNAMICS & GATE INITIALIZATION ABLATION")
    print(f"Git commit: {git_commit} | Total experimental runs: {total_runs}")
    print("=" * 86)

    for delay in delays:
        for budget in budgets:
            for model_type in models:
                for seed in seeds:
                    run_idx += 1
                    cfg = ExperimentConfig(
                        experiment_id="EXP002",
                        model_type=model_type,
                        target_budget=budget,
                        t_delay=delay,
                        distractor_sigma=0.1,
                        seed=seed,
                        epochs=epochs,
                        batch_size=32,
                        num_train_batches=15,
                        num_val_samples=100,
                        num_test_samples=200,
                        lr=0.005,
                    )
                    print(
                        f"[{run_idx:2d}/{total_runs:2d}] {model_type:16s} | Budget: {budget:4d} | Delay: {delay:3d} | Seed: {seed:4d} ...",
                        end="",
                        flush=True,
                    )
                    res, blog_path = run_experiment(cfg)
                    res["git_commit"] = git_commit
                    all_results.append(res)
                    print(
                        f" Acc: {res['test_accuracy']*100:5.1f}% | MSR: {res['msr']:5.2f} | PES: {res['pes']:5.2f} | {res['runtime_sec']:.1f}s"
                    )

                    # Export representative trace for each model family
                    model_rep_blog = brainlog_dir / f"EXP002_{model_type}_rep.brainlog"
                    if blog_path and not model_rep_blog.exists():
                        shutil.copy(blog_path, model_rep_blog)

    # Save full machine-readable results
    results_file = results_dir / "exp002_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 86)
    print(f"Saved complete machine-readable results to: {results_file}")
    print("=" * 86)

    print_aggregate_summary(all_results)
    return all_results


def print_aggregate_summary(results):
    grouped = defaultdict(list)
    for r in results:
        key = (r["model_id"], r["parameter_budget_target"], r["delay"])
        grouped[key].append(r)

    print(
        f"\n{'Model':<16} | {'Budget':<6} | {'Delay':<5} | {'Params':<6} | {'Test Acc (Mean±Std)':<20} | {'MSR (Mean)':<10} | {'PES (Mean)':<10}"
    )
    print("-" * 92)

    for (model, budget, delay), runs in sorted(grouped.items(), key=lambda x: (x[0][2], x[0][1], x[0][0])):
        accs = [r["test_accuracy"] * 100.0 for r in runs]
        msrs = [r["msr"] for r in runs]
        pes_vals = [r["pes"] for r in runs]
        params = runs[0]["parameter_count"]

        acc_str = f"{np.mean(accs):5.1f}% ± {np.std(accs):4.1f}%"
        print(
            f"{model:<16} | {budget:<6} | {delay:<5} | {params:<6} | {acc_str:<20} | {np.mean(msrs):<10.2f} | {np.mean(pes_vals):<10.2f}"
        )


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if quick:
        run_exp002_matrix(
            models=("BASE-RNN", "DISC-LRNN", "CAND-LI", "BASE-GRU-MemBias"),
            budgets=(300,),
            delays=(10, 25),
            seeds=(42, 137),
            epochs=20,
        )
    else:
        run_exp002_matrix()
