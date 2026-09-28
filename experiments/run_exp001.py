"""CLI driver for executing the full Experiment 001 evaluation matrix.

Evaluates BASE-RNN, BASE-GRU, and CAND-LI across parameter budgets, delays, and random seeds.
"""

from collections import defaultdict
import json
import sys
from pathlib import Path
import shutil
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.runner import ExperimentConfig, run_experiment


def run_matrix(
    models=("BASE-RNN", "BASE-GRU", "CAND-LI"),
    budgets=(300, 1000),
    delays=(10, 25, 50),
    seeds=(42, 137, 256),
    epochs=60,
):
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    brainlog_dir = results_dir / "brainlogs"
    brainlog_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_runs = len(models) * len(budgets) * len(delays) * len(seeds)
    run_idx = 0

    print("=" * 80)
    print("3D-BRAIN: EXPERIMENT 001 - CONTINUOUS DELAYED MATCH-TO-SAMPLE")
    print(f"Total experimental runs to execute: {total_runs}")
    print("=" * 80)

    for delay in delays:
        for budget in budgets:
            for model_type in models:
                for seed in seeds:
                    run_idx += 1
                    cfg = ExperimentConfig(
                        experiment_id="EXP001",
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
                        f"[{run_idx}/{total_runs}] Running {model_type:8s} | Budget: {budget:4d} | Delay: {delay:2d} | Seed: {seed} ...",
                        end="",
                        flush=True,
                    )
                    res, blog_path = run_experiment(cfg)
                    all_results.append(res)
                    print(
                        f" Acc: {res['test_accuracy']*100:5.1f}% | MSR: {res['msr']:4.2f} | PES: {res['pes']:5.2f} | {res['runtime_sec']:.1f}s"
                    )

                    # Save first successful brainlog as default for visualizer
                    default_blog = brainlog_dir / "EXP001_default.brainlog"
                    if blog_path and not default_blog.exists():
                        shutil.copy(blog_path, default_blog)

    # Save full machine-readable results
    results_file = results_dir / "exp001_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Saved complete machine-readable results to: {results_file}")
    print("=" * 80)

    # Print Aggregate Summary Table
    print_aggregate_summary(all_results)
    return all_results


def print_aggregate_summary(results):
    grouped = defaultdict(list)
    for r in results:
        key = (r["model_id"], r["parameter_budget_target"], r["delay"])
        grouped[key].append(r)

    print(
        f"\n{'Model':<10} | {'Budget':<6} | {'Delay':<5} | {'Params':<6} | {'Test Acc (Mean±Std)':<20} | {'MSR (Mean)':<10} | {'PES (Mean)':<10}"
    )
    print("-" * 84)

    for (model, budget, delay), runs in sorted(grouped.items(), key=lambda x: (x[0][2], x[0][1], x[0][0])):
        accs = [r["test_accuracy"] * 100.0 for r in runs]
        msrs = [r["msr"] for r in runs]
        pes_vals = [r["pes"] for r in runs]
        params = runs[0]["parameter_count"]

        acc_str = f"{np.mean(accs):5.1f}% ± {np.std(accs):4.1f}%"
        print(
            f"{model:<10} | {budget:<6} | {delay:<5} | {params:<6} | {acc_str:<20} | {np.mean(msrs):<10.2f} | {np.mean(pes_vals):<10.2f}"
        )


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if quick:
        run_matrix(
            models=("BASE-RNN", "BASE-GRU", "CAND-LI"),
            budgets=(300,),
            delays=(10, 25),
            seeds=(42, 137),
            epochs=25,
        )
    else:
        run_matrix()
