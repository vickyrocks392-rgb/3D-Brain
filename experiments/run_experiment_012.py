"""Experiment 012 Matrix Runner: Learnable Context-to-Action Interface & Causal Action Readout.

Executes:
1. Controlled Condition Matrix (Controls, Local Learning Rules, Competitions, Representations)
2. Representation Probe & Pure Context Decode Decomposition
3. Delay Sweeps (Delta t in [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000 ms])
4. Credit Localization & Leakage Comparison (Interface vs Collateral vs Recurrent)
5. Causal Interventions Suite (Scramble, Swap, Lesion, Irrelevant, Label Permute, Freeze)
6. Phase 11 Lesion Paradox Replication & Mechanistic Dissection
7. Computational Scaling Ladder (N = 30, 50, 100, 300, 1K, 10K)
8. BrainLog v3.0 Telemetry Generation and Schema Validation
"""

from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.base import BrainModel
from brain.spatial import TopologyFamily
from brain.phase12_network import Phase12Network
from brain.two_compartment import TwoCompartmentPopulation
from brain.context_action_interface import (
    ContextActionInterface,
    RepresentationSource,
    ReadoutArchitecture,
    ActionCompetitionMode,
)
from benchmarks.delayed_credit_assignment import (
    LongDelayContextualCreditTask,
    ContinualRegime,
    StepRecord,
)
from benchmarks.pure_context_decode import PureContextDecodeTask
from benchmarks.bptt_baseline import BPTTGradientBaseline
from observability.capture import StateCapture
from observability.topology import TopologyExtractor
from observability.brainlog import BrainLogSerializer, BrainLogValidator


def create_phase12_condition_model(
    condition_name: str,
    hidden_dim: int = 50,
    output_dim: int = 2,
    seed: int = 42,
) -> BrainModel:
    """Instantiate Phase12Network configured precisely for experimental conditions."""
    input_dim = 8

    # 1. Controls
    if condition_name == "COND_01_PHASE11_ROUTING_BASELINE":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R7_PHASE11_ROUTING,
            competition_mode=ActionCompetitionMode.BASELINE_INHIBITION,
            seed=seed,
        )
    elif condition_name == "COND_02_FIXED_RANDOM_READOUT":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R0_RANDOM,
            competition_mode=ActionCompetitionMode.BASELINE_INHIBITION,
            seed=seed,
        )
    elif condition_name == "COND_03_LINEAR_SUPERVISED_READOUT":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R1_LINEAR_SUPERVISED,
            competition_mode=ActionCompetitionMode.BASELINE_INHIBITION,
            learning_rate=0.03,
            seed=seed,
        )

    # 2. Local Learning Rules
    elif condition_name == "COND_04_LOCAL_HEBBIAN":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R2_HEBBIAN,
            competition_mode=ActionCompetitionMode.BASELINE_INHIBITION,
            learning_rate=0.01,
            seed=seed,
        )
    elif condition_name == "COND_05_THREE_FACTOR_INTERFACE":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R3_THREE_FACTOR,
            competition_mode=ActionCompetitionMode.BASELINE_INHIBITION,
            learning_rate=0.025,
            seed=seed,
        )
    elif condition_name == "COND_06_PRED_ERROR_GATED":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R4_PRED_ERROR,
            competition_mode=ActionCompetitionMode.BASELINE_INHIBITION,
            learning_rate=0.025,
            seed=seed,
        )
    elif condition_name == "COND_07_DELAYED_REWARD_ELIGIBILITY":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R5_DELAYED_REWARD,
            competition_mode=ActionCompetitionMode.BASELINE_INHIBITION,
            learning_rate=0.025,
            tau_eligibility_ms=60.0,
            seed=seed,
        )
    elif condition_name == "REF_08_ANALYTICAL_BPTT_READOUT":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R6_ANALYTICAL_BPTT,
            competition_mode=ActionCompetitionMode.BASELINE_INHIBITION,
            learning_rate=0.03,
            seed=seed,
        )

    # 3. Action Competition Ablations
    elif condition_name == "ABL_09_NO_ACTION_INHIBITION":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R3_THREE_FACTOR,
            competition_mode=ActionCompetitionMode.NO_INHIBITION,
            seed=seed,
        )
    elif condition_name == "ABL_10_WEAK_ACTION_INHIBITION":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R3_THREE_FACTOR,
            competition_mode=ActionCompetitionMode.WEAK_INHIBITION,
            seed=seed,
        )
    elif condition_name == "ABL_11_STRONG_ACTION_INHIBITION":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R3_THREE_FACTOR,
            competition_mode=ActionCompetitionMode.STRONG_INHIBITION,
            seed=seed,
        )
    elif condition_name == "ABL_12_NO_ACTION_RECURRENCE":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.ATTRACTOR,
            architecture=ReadoutArchitecture.R3_THREE_FACTOR,
            competition_mode=ActionCompetitionMode.NO_RECURRENCE,
            seed=seed,
        )

    # 4. Representation Source Ablations
    elif condition_name == "REP_13_SOMATIC_STATE":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.SOMATIC,
            architecture=ReadoutArchitecture.R3_THREE_FACTOR,
            seed=seed,
        )
    elif condition_name == "REP_14_DENDRITIC_STATE":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.DENDRITIC,
            architecture=ReadoutArchitecture.R3_THREE_FACTOR,
            seed=seed,
        )
    elif condition_name == "REP_15_NMDA_STATE":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.NMDA,
            architecture=ReadoutArchitecture.R3_THREE_FACTOR,
            seed=seed,
        )
    elif condition_name == "REP_16_CONCATENATED_STATE":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.CONCATENATED,
            architecture=ReadoutArchitecture.R3_THREE_FACTOR,
            seed=seed,
        )
    elif condition_name == "REP_17_PCA_COMPRESSED":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.PCA,
            architecture=ReadoutArchitecture.R3_THREE_FACTOR,
            seed=seed,
        )
    elif condition_name == "REP_18_RANDOM_PROJECTION":
        return Phase12Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            source=RepresentationSource.RANDOM_PROJ,
            architecture=ReadoutArchitecture.R3_THREE_FACTOR,
            seed=seed,
        )

    raise ValueError(f"Unknown condition: {condition_name}")


def run_condition_trial_sequence(
    model: Phase12Network,
    task: LongDelayContextualCreditTask,
    num_trials: int = 40,
    delay_steps: int = 15,
    reward_delay: int = 3,
    trial_seed: int = 42,
) -> Dict[str, Any]:
    """Execute a sequence of delayed contextual choice trials on a Phase12Network."""
    correct_count = 0
    total_trials = num_trials
    csi_list = []
    prec_list = []
    leak_list = []
    rec_leak_list = []
    margin_list = []
    entropy_list = []
    confusion = np.zeros((2, 2), dtype=int)

    rng = np.random.RandomState(trial_seed)

    for tr in range(num_trials):
        ctx_id = int(rng.choice([0, 1]))
        target_act = ctx_id  # Context 0 -> Action 0, Context 1 -> Action 1
        trial = task.generate_trial(
            context_id=ctx_id,
            delay_steps=delay_steps,
            reward_delay=reward_delay,
            trial_seed=trial_seed + tr * 17,
        )

        model.reset()
        chosen_action = 0
        choice_made = False

        rel_mask = (model.attractor_circuit.pool_ids == ctx_id)

        for step_sample in trial:
            rew = 0.0
            if step_sample.is_reward_step:
                rew = 1.0 if (chosen_action == target_act) else -1.0

            v_act, sel_act = model.step(
                x=step_sample.sensory_input,
                reward=rew,
                target_action=target_act,
                context_id=ctx_id,
                relevant_mask=rel_mask,
                is_choice_step=step_sample.is_choice_step,
                is_reward_step=step_sample.is_reward_step,
            )

            if step_sample.is_choice_step:
                chosen_action = sel_act
                choice_made = True

        if chosen_action == target_act:
            correct_count += 1
        confusion[target_act, chosen_action] += 1

        t = model.last_telemetry
        if t:
            csi_list.append(float(t.get("csi", 0.0)))
            prec_list.append(float(t.get("precision", 0.0)))
            leak_list.append(float(t.get("leakage", 0.0)))
            rec_leak_list.append(float(t.get("recurrent_leakage", 0.0)))
            margin_list.append(float(t.get("winner_margin", 0.0)))
            entropy_list.append(float(t.get("action_entropy", 0.0)))

    acc = float(correct_count / max(1, total_trials))
    return {
        "accuracy": acc,
        "mean_csi": float(np.mean(csi_list)) if csi_list else 0.0,
        "mean_precision": float(np.mean(prec_list)) if prec_list else 0.0,
        "mean_leakage": float(np.mean(leak_list)) if leak_list else 0.0,
        "mean_recurrent_leakage": float(np.mean(rec_leak_list)) if rec_leak_list else 0.0,
        "mean_winner_margin": float(np.mean(margin_list)) if margin_list else 0.0,
        "mean_action_entropy": float(np.mean(entropy_list)) if entropy_list else 0.0,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_18_conditions_matrix(
    seeds: List[int] = [42, 43, 44],
    trials_per_seed: int = 35,
) -> Dict[str, Any]:
    """Evaluate full matrix across multiple deterministic seeds."""
    conditions = [
        "COND_01_PHASE11_ROUTING_BASELINE",
        "COND_02_FIXED_RANDOM_READOUT",
        "COND_03_LINEAR_SUPERVISED_READOUT",
        "COND_04_LOCAL_HEBBIAN",
        "COND_05_THREE_FACTOR_INTERFACE",
        "COND_06_PRED_ERROR_GATED",
        "COND_07_DELAYED_REWARD_ELIGIBILITY",
        "REF_08_ANALYTICAL_BPTT_READOUT",
        "ABL_09_NO_ACTION_INHIBITION",
        "ABL_10_WEAK_ACTION_INHIBITION",
        "ABL_11_STRONG_ACTION_INHIBITION",
        "ABL_12_NO_ACTION_RECURRENCE",
        "REP_13_SOMATIC_STATE",
        "REP_14_DENDRITIC_STATE",
        "REP_15_NMDA_STATE",
        "REP_16_CONCATENATED_STATE",
        "REP_17_PCA_COMPRESSED",
        "REP_18_RANDOM_PROJECTION",
    ]

    results = {}
    print(f"=== Running Phase 12 Condition Matrix ({len(conditions)} conditions, {len(seeds)} seeds) ===")

    for c_idx, cond in enumerate(conditions, start=1):
        t0 = time.time()
        accs = []
        csis = []
        precs = []
        leaks = []
        rec_leaks = []
        margins = []
        entropies = []

        for seed in seeds:
            task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=15, t_choice=2, seed=seed)
            model = create_phase12_condition_model(cond, hidden_dim=50, output_dim=2, seed=seed)
            res = run_condition_trial_sequence(
                model=model,
                task=task,
                num_trials=trials_per_seed,
                delay_steps=15,
                reward_delay=3,
                trial_seed=seed,
            )
            accs.append(res["accuracy"])
            csis.append(res["mean_csi"])
            precs.append(res["mean_precision"])
            leaks.append(res["mean_leakage"])
            rec_leaks.append(res["mean_recurrent_leakage"])
            margins.append(res["mean_winner_margin"])
            entropies.append(res["mean_action_entropy"])

        elapsed = time.time() - t0
        results[cond] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "csi_mean": float(np.mean(csis)),
            "precision_mean": float(np.mean(precs)),
            "leakage_mean": float(np.mean(leaks)),
            "recurrent_leakage_mean": float(np.mean(rec_leaks)),
            "winner_margin_mean": float(np.mean(margins)),
            "action_entropy_mean": float(np.mean(entropies)),
            "elapsed_seconds": round(elapsed, 2),
        }
        print(
            f"[{c_idx:02d}/{len(conditions):02d}] {cond:<35} | Acc: {np.mean(accs)*100:5.1f}% ± {np.std(accs)*100:4.1f}% | "
            f"CSI: {np.mean(csis):+5.2f} | Prec: {np.mean(precs)*100:4.1f}% | Leak: {np.mean(leaks)*100:4.1f}% | "
            f"RecLeak: {np.mean(rec_leaks)*100:4.1f}% | Time: {elapsed:4.1f}s"
        )

    return results


def evaluate_pure_context_decode(seeds: List[int] = [42, 43, 44]) -> Dict[str, Any]:
    """Stage A & B: Pure Context Decode & Geometrical Separability Analysis."""
    print("\n=== Running Pure Context Decode & Representation Decomposition ===")
    sources = [
        RepresentationSource.SOMATIC,
        RepresentationSource.DENDRITIC,
        RepresentationSource.NMDA,
        RepresentationSource.ATTRACTOR,
        RepresentationSource.CONCATENATED,
    ]
    decode_task = PureContextDecodeTask(obs_dim=8, noise_std=0.05, seed=42)
    source_results = {}

    for src in sources:
        reps_a = []
        reps_b = []
        for seed in seeds:
            model = Phase12Network(
                input_dim=8,
                hidden_dim=50,
                output_dim=2,
                source=src,
                seed=seed,
            )
            # Evaluate across 15 trials of Context A and Context B
            for tr in range(15):
                model.reset()
                for _ in range(8):
                    xa, _ = decode_task.generate_sample(context_id=0)
                    model.step(xa)
                reps_a.append(model.action_interface.extract_context(model.observable_state()))

                model.reset()
                for _ in range(8):
                    xb, _ = decode_task.generate_sample(context_id=1)
                    model.step(xb)
                reps_b.append(model.action_interface.extract_context(model.observable_state()))

        reps_a_arr = np.array(reps_a)
        reps_b_arr = np.array(reps_b)
        geom = decode_task.evaluate_representation_separability(reps_a_arr, reps_b_arr)
        source_results[src.value] = geom
        print(
            f"  {src.value:<14} | D_sep: {geom['centroid_distance']:6.2f} | "
            f"SepRatio: {geom['separation_ratio']:5.2f} | ProbeAcc: {geom['linear_probe_accuracy']*100:5.1f}% | "
            f"MI: {geom['mutual_information_bits']:5.2f} bits"
        )

    return source_results


def evaluate_delay_sweeps(seeds: List[int] = [42, 43, 44]) -> Dict[str, Any]:
    """Stage D: Post-choice delay sweeps (0, 1, 5, 10, 25, 50, 100, 250, 500, 1000 ms)."""
    delays = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000]
    print(f"\n=== Running Post-Choice Delay Sweeps across {len(delays)} delays ===")

    architectures = [
        ("Phase 11 Baseline", ReadoutArchitecture.R7_PHASE11_ROUTING),
        ("Three-Factor Interface (R3)", ReadoutArchitecture.R3_THREE_FACTOR),
        ("Delayed Reward + Eligibility (R5)", ReadoutArchitecture.R5_DELAYED_REWARD),
        ("Analytical BPTT (R6)", ReadoutArchitecture.R6_ANALYTICAL_BPTT),
    ]

    sweep_results = {}
    for name, arch in architectures:
        delay_accs = {}
        for d in delays:
            acc_list = []
            for seed in seeds:
                task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=10, seed=seed)
                model = Phase12Network(
                    input_dim=8,
                    hidden_dim=50,
                    output_dim=2,
                    architecture=arch,
                    learning_rate=0.03,
                    seed=seed,
                )
                res = run_condition_trial_sequence(
                    model=model,
                    task=task,
                    num_trials=25,
                    delay_steps=10,
                    reward_delay=max(0, d // 10),
                    trial_seed=seed,
                )
                acc_list.append(res["accuracy"])
            delay_accs[str(d)] = float(np.mean(acc_list))
        sweep_results[name] = delay_accs
        print(f"  {name:<32} | 0ms: {delay_accs['0']*100:4.1f}% | 100ms: {delay_accs['100']*100:4.1f}% | 1000ms: {delay_accs['1000']*100:4.1f}%")

    return sweep_results


def evaluate_causal_interventions(seed: int = 42) -> Dict[str, Any]:
    """Section 13: Causal Interventions (C1..C6) on the Best Local Learning Interface."""
    print("\n=== Running Causal Interventions Suite (C1..C6) ===")
    task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=15, seed=seed)

    # 1. Train intact baseline model first
    model = Phase12Network(
        input_dim=8,
        hidden_dim=50,
        output_dim=2,
        architecture=ReadoutArchitecture.R3_THREE_FACTOR,
        learning_rate=0.04,
        seed=seed,
    )
    # Pre-train for 30 trials
    run_condition_trial_sequence(model, task, num_trials=30, trial_seed=seed)

    # Evaluate across causal intervention modes
    interventions = [
        ("INTACT_TRAINED", lambda m: None),
        ("C1_CONTEXT_SCRAMBLE", lambda m: m.action_interface.apply_context_scramble(True)),
        ("C2_CONTEXT_SWAP", lambda m: m.action_interface.apply_context_swap(True)),
        ("C3_LESION_25_PCT", lambda m: m.action_interface.apply_readout_lesion(0.25)),
        ("C3_LESION_50_PCT", lambda m: m.action_interface.apply_readout_lesion(0.50)),
        ("C3_LESION_75_PCT", lambda m: m.action_interface.apply_readout_lesion(0.75)),
        ("C3_LESION_100_PCT", lambda m: m.action_interface.apply_readout_lesion(1.00)),
        ("C5_LABEL_PERMUTATION", lambda m: m.action_interface.apply_label_permutation(True)),
        ("C6_FREEZE_PLASTICITY", lambda m: m.action_interface.freeze_plasticity(True)),
    ]

    results = {}
    for name, fn in interventions:
        # Clone current learned weights
        w_curr = model.action_interface.W_ctx_to_act.copy()
        fn(model)
        # Test performance under intervention (with plasticity temporarily paused to isolate causal readout)
        model.action_interface.freeze_plasticity(True)
        res = run_condition_trial_sequence(model, task, num_trials=30, trial_seed=seed + 99)
        results[name] = {
            "accuracy": res["accuracy"],
            "winner_margin": res["mean_winner_margin"],
            "action_entropy": res["mean_action_entropy"],
        }
        # Restore weights and flags
        model.action_interface.restore_lesion()
        model.action_interface.W_ctx_to_act = w_curr.copy()
        model.action_interface.freeze_plasticity(False)
        print(f"  {name:<25} | Accuracy: {res['accuracy']*100:5.1f}% | Margin: {res['mean_winner_margin']:5.2f} | Entropy: {res['mean_action_entropy']:4.2f}")

    return results


def evaluate_phase11_lesion_paradox(seed: int = 42) -> Dict[str, Any]:
    """Section 14: Systematic replication and diagnosis of Phase 11 Lesion Paradox."""
    print("\n=== Running Phase 11 Lesion Paradox Replication & Analysis ===")
    task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=15, seed=seed)

    conditions = [
        "INTACT",
        "RELEVANT_LESION",
        "IRRELEVANT_LESION",
        "BOTH_LESION",
        "CROSS_INH_DISABLED",
        "CROSS_INH_REVERSED",
    ]

    results = {}
    for cond in conditions:
        model = Phase12Network(
            input_dim=8,
            hidden_dim=50,
            output_dim=2,
            architecture=ReadoutArchitecture.R7_PHASE11_ROUTING,
            seed=seed,
        )
        model.action_interface.configure_lesion_paradox_condition(cond)
        res = run_condition_trial_sequence(model, task, num_trials=30, trial_seed=seed)
        results[cond] = {
            "accuracy": res["accuracy"],
            "winner_margin": res["mean_winner_margin"],
            "action_entropy": res["mean_action_entropy"],
        }
        print(f"  {cond:<22} | Accuracy: {res['accuracy']*100:5.1f}% | Margin: {res['mean_winner_margin']:5.2f}")

    return results


def evaluate_scaling_ladder() -> Dict[str, Any]:
    """Section 17: Computational Scaling Ladder (N = 30, 50, 100, 300, 1K, 10K)."""
    print("\n=== Running Computational Scaling Ladder (N = 30 to 10,000) ===")
    scales = [30, 50, 100, 300, 1000, 10000]
    results = {}

    for n in scales:
        if n <= 300:
            model = Phase12Network(
                input_dim=8,
                hidden_dim=n,
                output_dim=2,
                architecture=ReadoutArchitecture.R3_THREE_FACTOR,
                seed=42,
            )
            param_count = model.W_in.size + model.W_rec.size + model.action_interface.W_ctx_to_act.size + n + 2
            mem_mb = (param_count * 4 + n * 4 * 10) / (1024 * 1024)

            x = np.ones(8, dtype=np.float32)
            n_steps = 100
            t0 = time.time()
            total_spikes = 0
            for _ in range(n_steps):
                _, _ = model.step(x)
                total_spikes += int(np.sum(model.last_spikes))
            dt_exec = time.time() - t0

            steps_sec = float(n_steps / max(1e-6, dt_exec))
            spikes_sec = float(total_spikes / max(1e-6, dt_exec))
            syn_ops_sec = float(spikes_sec * (n * 0.15))
        else:
            # For N >= 1000, benchmark biophysical population + action interface directly (avoiding O(N^2) distance matrix)
            pop = TwoCompartmentPopulation(num_neurons=n, seed=42)
            iface = ContextActionInterface(hidden_dim=n, num_actions=2, seed=42)
            # Sparse parameters assuming ~100 synapses per neuron
            param_count = n * 100 + iface.W_ctx_to_act.size + n + 2
            mem_mb = (n * 8 * 14 + iface.W_ctx_to_act.size * 4) / (1024 * 1024)

            i_e = np.full(n, 1.5, dtype=np.float32)
            i_i = np.zeros(n, dtype=np.float32)
            i_soma = np.full(n, 0.5, dtype=np.float32)

            n_steps = 50 if n == 1000 else 25
            t0 = time.time()
            total_spikes = 0
            for s in range(n_steps):
                spks = pop.step(i_e, i_i, i_soma, current_step=s)
                total_spikes += int(np.sum(spks))
                # Step interface with subset representation
                ctx_sub = np.maximum(0.0, pop.v_soma[:iface.context_dim])
                iface.step(ctx_sub)
            dt_exec = time.time() - t0

            steps_sec = float(n_steps / max(1e-6, dt_exec))
            spikes_sec = float(total_spikes / max(1e-6, dt_exec))
            syn_ops_sec = float(spikes_sec * 100.0)

        results[str(n)] = {
            "parameters": int(param_count),
            "memory_mb": round(mem_mb, 3),
            "steps_per_sec": round(steps_sec, 1),
            "spikes_per_sec": round(spikes_sec, 1),
            "synaptic_ops_per_sec": round(syn_ops_sec, 1),
        }
        print(f"  N = {n:<5} | Params: {param_count:8d} | RAM: {mem_mb:6.2f} MB | {steps_sec:7.1f} steps/s | {spikes_sec/1e3:6.1f} kSpk/s")

    return results


def generate_phase12_brainlog(output_path: Path) -> Path:
    """Generate BrainLog v3.0 telemetry capturing the learnable action interface."""
    print(f"\n=== Generating BrainLog v3.0 Trace: {output_path.name} ===")
    task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=10, seed=42)
    model = Phase12Network(
        input_dim=8,
        hidden_dim=30,
        output_dim=2,
        architecture=ReadoutArchitecture.R3_THREE_FACTOR,
        learning_rate=0.03,
        seed=42,
    )

    topology = TopologyExtractor.from_network(model, "EXP012_ContextActionInterface")
    capture = StateCapture(topology=topology, dt_ms=1.0)

    # Run 2 complete contextual choice trials
    step_global = 0
    correct_trials = 0
    for tr in range(2):
        ctx_id = tr % 2
        target_act = ctx_id
        trial = task.generate_trial(context_id=ctx_id, delay_steps=10, reward_delay=2, trial_seed=42 + tr)
        model.reset()
        chosen = 0

        for step_record in trial:
            rew = 0.0
            if step_record.is_reward_step:
                rew = 1.0 if (chosen == target_act) else -1.0

            v_act, sel_act = model.step(
                x=step_record.sensory_input,
                reward=rew,
                target_action=target_act,
                context_id=ctx_id,
                is_choice_step=step_record.is_choice_step,
                is_reward_step=step_record.is_reward_step,
            )
            if step_record.is_choice_step:
                chosen = sel_act

            capture.capture_phase12_step(
                step=step_global,
                input_vec=step_record.sensory_input,
                model=model,
                y_val=v_act,
                target_val=float(target_act),
                mask_val=1.0 if step_record.is_choice_step else 0.0,
                reward=rew,
                context_id=ctx_id,
                interface_telemetry=model.last_telemetry,
                phase_label="PHASE12_TRACE",
            )
            step_global += 1

        if chosen == target_act:
            correct_trials += 1

    capture.complete_trial(final_acc=float(correct_trials / 2.0))

    saved_path = BrainLogSerializer.save(
        filepath=output_path,
        metadata={
            "experiment_id": "EXP012",
            "phase": "12",
            "architecture": "Phase12Network",
            "readout_rule": "R3_THREE_FACTOR",
            "substrate_status": "FROZEN_RECURRENT",
        },
        topology=topology,
        frames=capture.frames,
        events=capture.all_events,
        snapshots=capture.snapshots,
        version="3.0",
    )

    valid, reason = BrainLogValidator.validate(saved_path)
    assert valid, f"BrainLog v3.0 validation error: {reason}"
    print(f"  Saved BrainLog v3.0 trace ({saved_path.stat().st_size / 1024:.1f} KB) - VALIDATED: {valid}")
    return saved_path


def main():
    start_time = time.time()
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    brainlog_dir = results_dir / "brainlogs"
    brainlog_dir.mkdir(parents=True, exist_ok=True)

    # 1. 18 Conditions Controlled Matrix
    matrix_results = evaluate_18_conditions_matrix(seeds=[42, 43, 44], trials_per_seed=35)

    # 2. Pure Context Decode & Geometry Decomposition
    decode_results = evaluate_pure_context_decode(seeds=[42, 43, 44])

    # 3. Post-Choice Delay Sweeps
    delay_results = evaluate_delay_sweeps(seeds=[42, 43, 44])

    # 4. Causal Interventions (C1..C6)
    causal_results = evaluate_causal_interventions(seed=42)

    # 5. Phase 11 Lesion Paradox Replication
    lesion_results = evaluate_phase11_lesion_paradox(seed=42)

    # 6. Computational Scaling Ladder
    scaling_results = evaluate_scaling_ladder()

    # 7. BrainLog v3.0 Generation
    blog_path = brainlog_dir / "EXP012_action_interface.brainlog"
    generate_phase12_brainlog(blog_path)

    # Assemble comprehensive metrics
    total_elapsed = time.time() - start_time
    full_output = {
        "metadata": {
            "experiment": "EXP012",
            "phase": 12,
            "title": "Learnable Context-to-Action Interface & Causal Action Readout",
            "git_commit": "HEAD",
            "python_version": sys.version,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "seeds": [42, 43, 44],
        },
        "conditions": matrix_results,
        "pure_context_decode": decode_results,
        "delay_sweeps": delay_results,
        "causal_interventions": causal_results,
        "phase11_lesion_paradox": lesion_results,
        "scaling_ladder": scaling_results,
    }

    metrics_file = results_dir / "exp012_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n=======================================================")
    print(f"EXP012 execution completed successfully in {total_elapsed:.1f}s.")
    print(f"Metrics saved to: {metrics_file}")
    print(f"BrainLog saved to: {blog_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
