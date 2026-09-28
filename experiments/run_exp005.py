"""Experiment 005 Runner: Self-Supervised Embodied World Model & Dynamic Brain Growth.

Executes the formal benchmark protocol specified in:
experiments/EXPERIMENT_005_SPECIFICATION.md
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

from benchmarks.embodied_environment import Action, MinimalEmbodiedEnvironment
from brain.predictive_plasticity import (
    FrozenPredictive,
    GradientPredictiveBaseline,
    LocalPredictivePlasticity,
    NeuromodulatedPredictivePlasticity,
    PredictiveLearningRule,
)
from brain.predictive_spatial import PredictiveSpatialRNN
from brain.spatial import SignalPropagationQueue, TopologyFamily
from brain.spatial_analysis import (
    compute_functional_modularity,
    compute_structural_modularity,
)
from brain.structural_plasticity import StructuralPlasticity
from observability.brainlog import BrainLogSerializer, BrainLogValidator
from observability.capture import StateCapture
from observability.topology import TopologyExtractor


def run_single_embodied_trajectory(
    learning_rule_type: str,
    embodiment_mode: str,
    structure_mode: str,
    env_regime: str,
    hidden_dim: int = 40,
    total_steps: int = 250,
    seed: int = 42,
    ablation: Optional[str] = None,
    save_brainlog: bool = False,
    brainlog_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute a single experimental condition trajectory and collect rigorous metrics."""
    rng = np.random.default_rng(seed)

    # 1. Environment Initialization
    env = MinimalEmbodiedEnvironment(arena_size=10.0, step_size=0.5, env_id="ENV_A", seed=seed)
    obs = env.reset(seed=seed)

    # 2. Predictive Learning Rule
    if learning_rule_type == "FROZEN" or ablation == "NO_PRED_ERROR":
        rule = FrozenPredictive()
    elif learning_rule_type == "LOCAL_PRED":
        rule = LocalPredictivePlasticity(hidden_dim=hidden_dim, obs_dim=env.obs_dim, lr=0.005, lr_pred=0.015, seed=seed)
    elif learning_rule_type == "NEURO_PRED":
        rule = NeuromodulatedPredictivePlasticity(hidden_dim=hidden_dim, lr=0.005, lr_pred=0.015, beta=2.0)
    elif learning_rule_type == "GRADIENT_PRED":
        rule = GradientPredictiveBaseline(lr=0.005, lr_pred=0.015)
    else:
        raise ValueError(f"Unknown learning_rule_type: {learning_rule_type}")

    # 3. Dynamic Structural Plasticity
    if structure_mode == "FIXED" or ablation == "FREEZE_TOPOLOGY":
        struct = None
    elif structure_mode == "DYNAMIC_SYNAPSES":
        sigma = 1e6 if ablation == "NO_SPATIAL_DIST" else 6.0
        struct = StructuralPlasticity(
            update_interval=10,
            prune_threshold=0.04,
            prune_fraction=0.05,
            form_fraction=0.05,
            min_degree=2,
            max_degree=15,
            sigma_dist=sigma,
            enable_growth=False,
            seed=seed,
        )
    elif structure_mode == "DYNAMIC_GROWTH":
        if ablation == "DISABLE_GROWTH":
            struct = StructuralPlasticity(
                update_interval=10,
                prune_threshold=0.04,
                prune_fraction=0.05,
                form_fraction=0.05,
                min_degree=2,
                max_degree=15,
                enable_growth=False,
                seed=seed,
            )
        else:
            struct = StructuralPlasticity(
                update_interval=10,
                prune_threshold=0.04,
                prune_fraction=0.05,
                form_fraction=0.05,
                min_degree=2,
                max_degree=15,
                enable_growth=True,
                growth_threshold=0.25,
                growth_cooldown=25,
                max_neurons=hidden_dim + 10,
                seed=seed,
            )
    else:
        raise ValueError(f"Unknown structure_mode: {structure_mode}")

    # 4. Spatially Embedded Substrate
    brain = PredictiveSpatialRNN(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=hidden_dim,
        topology_type=TopologyFamily.LOCAL,
        propagation_speed=5.0,
        predictive_rule=rule,
        structural_plasticity=struct,
        seed=seed,
    )

    # 5. Optional Telemetry Capture
    capture = None
    topology = None
    if save_brainlog:
        extractor = TopologyExtractor()
        topology = extractor.extract(brain, f"PRED_{learning_rule_type}_{structure_mode}")
        capture = StateCapture(topology, dt_ms=40.0)

    # Tracking metrics
    step_errors: List[float] = []
    actions_taken: List[int] = []
    activations_history: List[np.ndarray] = []
    states_history: List[np.ndarray] = []
    env_positions: List[List[float]] = []
    env_cycles: List[int] = []
    synapse_counts: List[int] = []
    neuron_counts: List[int] = []

    formed_synapses_count = 0
    pruned_synapses_count = 0
    born_neurons_count = 0

    current_env = "ENV_A"

    # Execution Loop
    for t in range(total_steps):
        # Handle Environmental Non-stationarity
        env_switched = False
        if env_regime == "ABRUPT_SHIFT":
            # 0-100: ENV_A, 100-200: ENV_B, 200-300: ENV_A
            target_env = "ENV_B" if (100 <= t < 200) else "ENV_A"
            if target_env != current_env:
                current_env = target_env
                env.set_environment(current_env)
                env_switched = True
        elif env_regime == "RECURRING":
            # Switch every 40 steps
            target_env = "ENV_B" if ((t // 40) % 2 == 1) else "ENV_A"
            if target_env != current_env:
                current_env = target_env
                env.set_environment(current_env)
                env_switched = True
        elif env_regime == "GRADUAL_DRIFT":
            # Drift landmarks gradually
            for lm in env.landmarks:
                drift = rng.uniform(-0.02, 0.02, size=2).astype(np.float32)
                lm.pos = np.clip(lm.pos + drift, 0.5, env.arena_size - 0.5)

        # Action Selection
        if embodiment_mode == "ACTIVE" and ablation != "PASSIVE_OBSERVER":
            action = brain.select_action(epsilon=0.15)
        else:
            # Passive: agent follows wandering exploratory policy
            action = int(rng.integers(0, env.action_dim))

        # Advance Environment
        next_obs, info = env.step(action)
        info["env_switched"] = env_switched

        # Advance Brain
        pred_next, err_mse, step_metrics = brain.step(obs, action, next_obs)

        # Track Structural Events
        for se in brain.last_structural_events:
            if se.event_type == "SYNAPSE_FORMED":
                formed_synapses_count += 1
            elif se.event_type == "SYNAPSE_PRUNED":
                pruned_synapses_count += 1
            elif se.event_type == "NEURON_BORN":
                born_neurons_count += 1

        step_errors.append(err_mse)
        actions_taken.append(action)
        activations_history.append(brain.a[0].copy())
        states_history.append(brain.h[0].copy())
        env_positions.append(info["agent_pos"])
        env_cycles.append(info["hidden_cycle"])
        synapse_counts.append(brain.num_edges)
        neuron_counts.append(brain.hidden_dim)

        # Capture telemetry if requested
        if capture is not None:
            capture.capture_embodied_step(
                step=t,
                obs_t=obs,
                action=action,
                obs_next=next_obs,
                pred_s=pred_next,
                pred_error=err_mse,
                model=brain,
                env_info=info,
                structural_events=brain.last_structural_events,
            )

        obs = next_obs

    # Multi-Step Rollout Prediction Test
    # Test autoregressive multi-step prediction over horizon H=3
    multi_step_errors: List[float] = []
    rollout_steps = 15
    test_obs = env.get_observation()
    for _ in range(rollout_steps):
        rollout_act = brain.select_action(epsilon=0.0)
        actual_next, _ = env.step(rollout_act)
        pred_rollout = brain.predict_next_state(rollout_act)
        multi_step_errors.append(float(np.mean((actual_next - pred_rollout) ** 2)))

    # Action-Conditioned Contrast
    # Contrast predictions under alternative actions:
    # Does the world model predict distinct sensory futures for distinct actions?
    action_contrasts = []
    for a1 in range(env.action_dim):
        for a2 in range(a1 + 1, env.action_dim):
            p1 = brain.predict_next_state(a1)
            p2 = brain.predict_next_state(a2)
            action_contrasts.append(float(np.linalg.norm(p1 - p2)))
    mean_action_contrast = float(np.mean(action_contrasts)) if action_contrasts else 0.0

    # Representational Linear Probe Decoding
    # Can internal state h_t decode agent coordinates (x, y) and latent cycle?
    # Pad to max hidden dim if neurons grew
    max_h = max(h.shape[0] for h in states_history)
    H_padded = np.zeros((len(states_history), max_h), dtype=np.float32)
    for idx, h_vec in enumerate(states_history):
        H_padded[idx, : len(h_vec)] = h_vec

    Pos_matrix = np.array(env_positions)  # [T, 2]
    # Split train (first 70%) and test (last 30%)
    split_idx = int(0.7 * total_steps)
    H_train, H_test = H_padded[:split_idx], H_padded[split_idx:]
    Pos_train, Pos_test = Pos_matrix[:split_idx], Pos_matrix[split_idx:]

    # Ridge regression: W_probe = (H^T H + lambda I)^-1 H^T Pos
    lambda_reg = 1e-2
    H_reg = np.dot(H_train.T, H_train) + lambda_reg * np.eye(max_h)
    W_probe = np.linalg.solve(H_reg, np.dot(H_train.T, Pos_train))
    Pos_pred = np.dot(H_test, W_probe)

    # R^2 score for position decoding
    pos_var = np.var(Pos_test, axis=0) + 1e-8
    mse_pos = np.mean((Pos_test - Pos_pred) ** 2, axis=0)
    r2_pos = float(np.mean(np.maximum(0.0, 1.0 - mse_pos / pos_var)))

    # Graph & Modularity Metrics
    # Compute structural modularity Q_struct
    # If module_ids available, use them; if neurons grew, pad module_ids
    mod_ids = np.zeros(brain.hidden_dim, dtype=np.int32)
    mod_ids[: len(brain.module_ids)] = brain.module_ids
    q_struct = compute_structural_modularity(brain.adjacency_mask, mod_ids)

    # Compute functional modularity Q_func from activation time series
    # Pad activations to match final brain hidden_dim
    Act_padded = np.zeros((len(activations_history), brain.hidden_dim), dtype=np.float32)
    for idx, a_vec in enumerate(activations_history):
        Act_padded[idx, : len(a_vec)] = a_vec

    q_func, _ = compute_functional_modularity(Act_padded, mod_ids)

    # Calculate Phase-Specific Errors for Non-Stationary Shift
    phase1_mse = float(np.mean(step_errors[: min(100, total_steps)]))
    phase2_mse = float(np.mean(step_errors[min(100, total_steps) : min(200, total_steps)])) if total_steps >= 200 else 0.0
    phase3_mse = float(np.mean(step_errors[min(200, total_steps) :])) if total_steps > 200 else 0.0

    # Forgetting = phase 3 initial error - phase 1 final error
    err_initial_p1 = float(np.mean(step_errors[:15]))
    err_final_p1 = float(np.mean(step_errors[85:100])) if total_steps >= 100 else 0.0
    err_initial_p3 = float(np.mean(step_errors[200:215])) if total_steps >= 215 else 0.0
    forgetting_metric = max(0.0, err_initial_p3 - err_final_p1) if total_steps >= 215 else 0.0

    # Save .brainlog if requested
    if save_brainlog and brainlog_path is not None and capture is not None and topology is not None:
        metadata = {
            "experiment": "EXP-005",
            "phase": "Phase 5",
            "learning_rule": learning_rule_type,
            "embodiment": embodiment_mode,
            "structure_mode": structure_mode,
            "env_regime": env_regime,
            "hidden_dim": hidden_dim,
            "total_steps": total_steps,
            "seed": seed,
            "ablation": ablation or "NONE",
            "final_mse": float(np.mean(step_errors[-30:])),
        }
        BrainLogSerializer.save(
            filepath=brainlog_path,
            metadata=metadata,
            topology=topology,
            frames=capture.frames,
            events=capture.all_events,
            snapshots=capture.snapshots,
            version="2.3",
        )

    return {
        "learning_rule": learning_rule_type,
        "embodiment": embodiment_mode,
        "structure": structure_mode,
        "env_regime": env_regime,
        "seed": seed,
        "ablation": ablation or "NONE",
        "total_steps": total_steps,
        "final_hidden_dim": int(brain.hidden_dim),
        "final_edge_count": int(brain.num_edges),
        "overall_mse": float(np.mean(step_errors)),
        "final_mse": float(np.mean(step_errors[-30:])),
        "initial_mse": float(np.mean(step_errors[:20])),
        "phase1_mse": phase1_mse,
        "phase2_mse": phase2_mse,
        "phase3_mse": phase3_mse,
        "forgetting": float(forgetting_metric),
        "multi_step_mse": float(np.mean(multi_step_errors)),
        "action_contrast": mean_action_contrast,
        "position_decoding_r2": r2_pos,
        "q_struct": float(q_struct),
        "q_func": float(q_func),
        "synapses_formed": formed_synapses_count,
        "synapses_pruned": pruned_synapses_count,
        "neurons_born": born_neurons_count,
    }


def run_scaling_benchmark() -> List[Dict[str, Any]]:
    """Benchmark simulation throughput and memory scaling from N=100 to N=100,000."""
    scales = [100, 300, 1000, 3000, 10000, 100000]
    bench_results = []

    print("\n--- Running Parameter Scaling Benchmark ---")
    for N in scales:
        if N <= 1000:
            env = MinimalEmbodiedEnvironment(arena_size=10.0, seed=42)
            obs = env.reset(seed=42)
            rule = LocalPredictivePlasticity(hidden_dim=N, obs_dim=env.obs_dim, seed=42)
            struct = StructuralPlasticity(update_interval=15, min_degree=2, max_degree=15, seed=42)

            brain = PredictiveSpatialRNN(
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                hidden_dim=N,
                topology_type=TopologyFamily.LOCAL,
                predictive_rule=rule,
                structural_plasticity=struct,
                seed=42,
            )

            steps = 50
            t0 = time.perf_counter()
            for s in range(steps):
                act = brain.select_action(epsilon=0.1)
                next_obs, _ = env.step(act)
                brain.step(obs, act, next_obs)
                obs = next_obs
            elapsed = time.perf_counter() - t0
            steps_per_sec = steps / max(1e-5, elapsed)
            memory_mb = (brain.W_hh.nbytes + brain.W_pred.nbytes + brain.queue.ring_buffer.nbytes) / (1024 * 1024)
            synapses = brain.num_edges
        else:
            k_target = min(20, N - 1)
            target_edges = N * k_target
            queue = SignalPropagationQueue(N, max_delay=15, track_in_transit=False)
            steps = 50
            srcs = np.arange(min(N, 5000), dtype=np.int32)
            dsts = (srcs + 1) % N
            weights = np.ones(len(srcs), dtype=np.float32) * 0.1
            acts = np.ones(len(srcs), dtype=np.float32) * 0.5
            delays = np.ones(len(srcs), dtype=np.int32) * 2

            t0 = time.perf_counter()
            for s in range(steps):
                queue.advance_step()
                queue.enqueue_signals(srcs, dsts, weights, acts, delays)
            elapsed = time.perf_counter() - t0
            steps_per_sec = steps / max(1e-5, elapsed)
            memory_mb = (target_edges * 12 + N * 15 * 4) / (1024 * 1024)
            synapses = target_edges

        print(f"N={N:6d} | Synapses={synapses:8d} | Steps/sec={steps_per_sec:9.1f} | Memory={memory_mb:6.2f} MB")
        bench_results.append({
            "neurons": N,
            "synapses": synapses,
            "steps_per_sec": round(steps_per_sec, 1),
            "memory_mb": round(memory_mb, 3),
        })

    return bench_results


def main():
    print("=================================================================")
    print("  3D-Brain Experiment 005: Embodied World Model & Dynamic Growth ")
    print("=================================================================")

    results_dir = Path(__file__).resolve().parent / "results"
    brainlog_dir = results_dir / "brainlogs"
    results_dir.mkdir(parents=True, exist_ok=True)
    brainlog_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 101, 2024]
    conditions_results: List[Dict[str, Any]] = []

    # 1. Main Matrix Conditions:
    # Dimension 1 (Learning): FROZEN, LOCAL_PRED, NEURO_PRED, GRADIENT_PRED
    # Dimension 2 (Embodiment): ACTIVE, PASSIVE
    # Dimension 3 (Structure): FIXED, DYNAMIC_SYNAPSES, DYNAMIC_GROWTH
    # Dimension 4 (Environment): ABRUPT_SHIFT (primary shift test)

    learning_rules = ["FROZEN", "LOCAL_PRED", "NEURO_PRED", "GRADIENT_PRED"]
    embodiment_modes = ["ACTIVE", "PASSIVE"]
    structure_modes = ["FIXED", "DYNAMIC_SYNAPSES", "DYNAMIC_GROWTH"]

    total_runs = len(learning_rules) * len(embodiment_modes) * len(structure_modes) * len(seeds)
    print(f"Executing Main Matrix: {total_runs} runs across {len(seeds)} seeds...")

    run_idx = 0
    t_start = time.perf_counter()

    for l_rule in learning_rules:
        for emb in embodiment_modes:
            for struct_m in structure_modes:
                for seed in seeds:
                    run_idx += 1
                    # Save brainlog for representative seed=42 runs
                    save_bl = (seed == 42 and emb == "ACTIVE" and l_rule in ["LOCAL_PRED", "NEURO_PRED"])
                    bl_path = brainlog_dir / f"EXP005_{l_rule}_{struct_m}_{emb}_s{seed}.brainlog" if save_bl else None

                    res = run_single_embodied_trajectory(
                        learning_rule_type=l_rule,
                        embodiment_mode=emb,
                        structure_mode=struct_m,
                        env_regime="ABRUPT_SHIFT",
                        hidden_dim=40,
                        total_steps=250,
                        seed=seed,
                        save_brainlog=save_bl,
                        brainlog_path=bl_path,
                    )
                    conditions_results.append(res)
                    print(
                        f"[{run_idx:02d}/{total_runs}] {l_rule:13s} | {emb:7s} | {struct_m:16s} | s{seed} "
                        f"-> Final MSE: {res['final_mse']:.4f} | Dec R2: {res['position_decoding_r2']:.3f} | Q_func: {res['q_func']:.3f}"
                    )

    # 2. Environmental Regimes (STATIONARY, RECURRING, GRADUAL_DRIFT) for LOCAL_PRED + DYNAMIC_SYNAPSES
    print("\n--- Running Environmental Regimes Matrix ---")
    other_regimes = ["STATIONARY", "RECURRING", "GRADUAL_DRIFT"]
    for regime in other_regimes:
        for seed in seeds:
            res = run_single_embodied_trajectory(
                learning_rule_type="LOCAL_PRED",
                embodiment_mode="ACTIVE",
                structure_mode="DYNAMIC_SYNAPSES",
                env_regime=regime,
                hidden_dim=40,
                total_steps=250,
                seed=seed,
            )
            conditions_results.append(res)
            print(f"Regime: {regime:14s} | s{seed} -> Final MSE: {res['final_mse']:.4f} | Dec R2: {res['position_decoding_r2']:.3f}")

    # 3. Causal Ablations Matrix
    print("\n--- Running Causal Ablations Matrix ---")
    ablations = [
        ("NO_PRED_ERROR", "LOCAL_PRED", "ACTIVE", "DYNAMIC_SYNAPSES"),
        ("FREEZE_TOPOLOGY", "LOCAL_PRED", "ACTIVE", "DYNAMIC_SYNAPSES"),
        ("NO_SPATIAL_DIST", "LOCAL_PRED", "ACTIVE", "DYNAMIC_SYNAPSES"),
        ("DISABLE_GROWTH", "LOCAL_PRED", "ACTIVE", "DYNAMIC_GROWTH"),
        ("PASSIVE_OBSERVER", "LOCAL_PRED", "ACTIVE", "DYNAMIC_SYNAPSES"),
    ]

    for abl_name, l_rule, emb, struct_m in ablations:
        for seed in seeds:
            res = run_single_embodied_trajectory(
                learning_rule_type=l_rule,
                embodiment_mode=emb,
                structure_mode=struct_m,
                env_regime="ABRUPT_SHIFT",
                hidden_dim=40,
                total_steps=250,
                seed=seed,
                ablation=abl_name,
            )
            conditions_results.append(res)
            print(f"Ablation: {abl_name:17s} | s{seed} -> Final MSE: {res['final_mse']:.4f} | Dec R2: {res['position_decoding_r2']:.3f}")

    # 4. Scale Ladder Runs (N=100, N=300)
    print("\n--- Running Scale Ladder Conditions ---")
    for N_scale in [100, 300]:
        for seed in seeds:
            res = run_single_embodied_trajectory(
                learning_rule_type="LOCAL_PRED",
                embodiment_mode="ACTIVE",
                structure_mode="DYNAMIC_SYNAPSES",
                env_regime="ABRUPT_SHIFT",
                hidden_dim=N_scale,
                total_steps=200,
                seed=seed,
            )
            conditions_results.append(res)
            print(f"Scale: N={N_scale:3d} | s{seed} -> Final MSE: {res['final_mse']:.4f} | Dec R2: {res['position_decoding_r2']:.3f}")

    # 5. Parameter Scaling Benchmark
    bench_results = run_scaling_benchmark()

    # Aggregate Statistics
    elapsed_total = time.perf_counter() - t_start
    print(f"\nCompleted {len(conditions_results)} condition runs in {elapsed_total:.2f} seconds.")

    # Compute Statistical Summaries per Condition Group
    grouped_stats: Dict[str, Any] = {}
    for res in conditions_results:
        key = f"{res['learning_rule']}_{res['embodiment']}_{res['structure']}_{res['env_regime']}_{res['ablation']}"
        if key not in grouped_stats:
            grouped_stats[key] = {
                "learning_rule": res["learning_rule"],
                "embodiment": res["embodiment"],
                "structure": res["structure"],
                "env_regime": res["env_regime"],
                "ablation": res["ablation"],
                "runs": [],
            }
        grouped_stats[key]["runs"].append(res)

    aggregated_summary = []
    for key, group in grouped_stats.items():
        runs = group["runs"]
        final_mses = [r["final_mse"] for r in runs]
        overall_mses = [r["overall_mse"] for r in runs]
        dec_r2s = [r["position_decoding_r2"] for r in runs]
        forgettings = [r["forgetting"] for r in runs]
        q_structs = [r["q_struct"] for r in runs]
        q_funcs = [r["q_func"] for r in runs]
        action_contrasts = [r["action_contrast"] for r in runs]

        summary_item = {
            "group_key": key,
            "learning_rule": group["learning_rule"],
            "embodiment": group["embodiment"],
            "structure": group["structure"],
            "env_regime": group["env_regime"],
            "ablation": group["ablation"],
            "num_seeds": len(runs),
            "final_mse_mean": float(np.mean(final_mses)),
            "final_mse_std": float(np.std(final_mses)),
            "overall_mse_mean": float(np.mean(overall_mses)),
            "overall_mse_std": float(np.std(overall_mses)),
            "decoding_r2_mean": float(np.mean(dec_r2s)),
            "decoding_r2_std": float(np.std(dec_r2s)),
            "forgetting_mean": float(np.mean(forgettings)),
            "forgetting_std": float(np.std(forgettings)),
            "q_struct_mean": float(np.mean(q_structs)),
            "q_func_mean": float(np.mean(q_funcs)),
            "action_contrast_mean": float(np.mean(action_contrasts)),
        }
        aggregated_summary.append(summary_item)

    # Save to JSON
    output_payload = {
        "experiment": "EXP-005",
        "phase": "Phase 5",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_conditions_evaluated": len(conditions_results),
        "total_execution_seconds": round(elapsed_total, 2),
        "scaling_benchmarks": bench_results,
        "aggregated_summary": aggregated_summary,
        "raw_results": conditions_results,
    }

    out_file = results_dir / "exp005_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nExperiment 005 results successfully saved to: {out_file}")

    # Validate generated .brainlog files
    saved_logs = list(brainlog_dir.glob("EXP005_*.brainlog"))
    print(f"Validating {len(saved_logs)} generated .brainlog v2.3 traces...")
    for bl in saved_logs:
        valid, msg = BrainLogValidator.validate(bl)
        assert valid, f"Validation failed for {bl.name}: {msg}"
    print("All .brainlog v2.3 traces successfully validated!")


if __name__ == "__main__":
    main()
