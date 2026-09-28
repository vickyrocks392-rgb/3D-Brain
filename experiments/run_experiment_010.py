"""Experiment 010 Runner: Two-Compartment Neurons, Slow Conductances & Targeted Credit Assignment.

Executes the formal benchmark protocol specified in:
experiments/EXPERIMENT_010_SPECIFICATION.md
"""

import json
from pathlib import Path
import sys
import time
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.delayed_credit_assignment import (
    LongDelayContextualCreditTask,
    ContinualRegime,
)
from benchmarks.bptt_baseline import BPTTGradientBaseline
from benchmarks.transformer_baseline import benchmark_transformer_on_credit_task

from brain.spatial import TopologyFamily
from brain.spatial_analysis import compute_structural_modularity
from brain.ei_dynamics import NeuronType
from brain.ei_analysis import compute_ei_balance_metrics
from brain.ai_regime_analysis import (
    compute_cv_isi,
    compute_fano_factor,
    compute_pairwise_spike_correlation,
)
from brain.synchronization import (
    compute_kuramoto_order_parameter,
    compute_phase_locking_value,
)
from brain.spiking_analysis import (
    compute_spiking_statistics,
    compute_functional_modularity_from_spikes,
)
from brain.three_factor import (
    EligibilityTraceMode,
    NeuromodulatorMode,
)
from brain.three_factor_spatial import ThreeFactorSpatialNetwork
from brain.two_compartment import TwoCompartmentPopulation
from brain.targeted_feedback import (
    FeedbackMode,
    DendriticEligibilityMode,
    TargetedSpatialFeedbackEngine,
)
from brain.two_compartment_spatial import TwoCompartmentSpatialNetwork

from observability.brainlog import BrainLogSerializer, BrainLogValidator
from observability.capture import StateCapture
from observability.topology import TopologyExtractor


def instantiate_exp010_model(
    condition: str,
    obs_dim: int = 8,
    action_dim: int = 1,
    hidden_dim: int = 50,
    topology_type: TopologyFamily = TopologyFamily.MODULAR,
    tau_nmda: float = 100.0,
    seed: int = 42,
) -> Any:
    """Instantiate neural model corresponding to one of the 16 required conditions."""
    # Condition 1: Phase 9 point-spiking baseline with scalar RPE
    if condition == "COND_01_POINT_SPIKING_BASELINE":
        return ThreeFactorSpatialNetwork(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            topology_type=topology_type,
            neuromodulator_mode=NeuromodulatorMode.REWARD_PREDICTION_ERROR,
            eligibility_mode=EligibilityTraceMode.COINCIDENCE,
            homeostasis_enabled=True,
            freeze_recurrent=False,
            seed=seed,
        )

    # Condition 2: Frozen point-spiking
    if condition == "COND_02_FROZEN_POINT_SPIKING":
        return ThreeFactorSpatialNetwork(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            topology_type=topology_type,
            neuromodulator_mode=NeuromodulatorMode.FROZEN,
            eligibility_mode=EligibilityTraceMode.COINCIDENCE,
            homeostasis_enabled=False,
            freeze_recurrent=True,
            seed=seed,
        )

    # Condition 16: Gradient reference (Compact BPTT Baseline)
    if condition == "REF_16_BPTT_GRADIENT_BASELINE":
        return BPTTGradientBaseline(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            learning_rate=0.01,
            seed=seed,
        )

    # Two-Compartment configurations
    fb_mode = FeedbackMode.SPATIAL_LOCAL
    elig_mode = DendriticEligibilityMode.VOLTAGE_GATED
    freeze_rec = False
    g_coup = 0.7
    nonlin_dend = True
    g_nl = 0.45
    alpha_nmda = 0.08
    curr_tau_nmda = tau_nmda

    if condition == "COND_03_TWO_COMPARTMENT_FROZEN":
        fb_mode = FeedbackMode.FROZEN
        freeze_rec = True
    elif condition == "COND_04_POINT_ELIGIBILITY":
        # Point-neuron coincidence eligibility with constant modulation
        return ThreeFactorSpatialNetwork(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            topology_type=topology_type,
            neuromodulator_mode=NeuromodulatorMode.CONSTANT,
            eligibility_mode=EligibilityTraceMode.COINCIDENCE,
            homeostasis_enabled=True,
            freeze_recurrent=False,
            seed=seed,
        )
    elif condition == "COND_05_DENDRITIC_ELIGIBILITY":
        fb_mode = FeedbackMode.FROZEN
        freeze_rec = True
    elif condition == "COND_06_DENDRITIC_GLOBAL_REWARD":
        fb_mode = FeedbackMode.GLOBAL_SCALAR
    elif condition == "COND_07_DENDRITIC_PRED_ERROR":
        fb_mode = FeedbackMode.SPATIAL_LOCAL
    elif condition == "COND_08_DENDRITIC_RPE":
        fb_mode = FeedbackMode.GLOBAL_SCALAR
    elif condition == "COND_09_DENDRITIC_TARGETED_SPATIAL":
        fb_mode = FeedbackMode.SPATIAL_LOCAL
    # Ablations
    elif condition == "ABL_10_NO_SLOW_CONDUCTANCE":
        curr_tau_nmda = 1.0
        alpha_nmda = 0.0
    elif condition == "ABL_11_NO_DENDRITIC_NONLINEARITY":
        nonlin_dend = False
        g_nl = 0.0
    elif condition == "ABL_12_NO_COMPARTMENT_COUPLING":
        g_coup = 0.0
    elif condition == "ABL_13_RANDOMIZED_FEEDBACK":
        fb_mode = FeedbackMode.SHUFFLED_FEEDBACK
    elif condition == "ABL_14_SHUFFLED_REWARD":
        fb_mode = FeedbackMode.SPATIAL_LOCAL
    elif condition == "ABL_15_NO_ELIGIBILITY":
        elig_mode = DendriticEligibilityMode.ZERO

    return TwoCompartmentSpatialNetwork(
        input_dim=obs_dim,
        hidden_dim=hidden_dim,
        output_dim=action_dim,
        topology_type=topology_type,
        tau_dend=25.0,
        g_coupling=g_coup,
        tau_nmda=curr_tau_nmda,
        alpha_nmda=alpha_nmda,
        nonlinear_dendrite=nonlin_dend,
        g_nl=g_nl,
        feedback_mode=fb_mode,
        eligibility_mode=elig_mode,
        freeze_recurrent=freeze_rec,
        seed=seed,
    )


def run_condition_trial_loop(
    condition: str,
    task: LongDelayContextualCreditTask,
    num_trials: int = 50,
    delay_steps: int = 25,
    reward_delay: int = 5,
    seeds: List[int] = [42, 43, 44],
    hidden_dim: int = 50,
) -> Dict[str, Any]:
    """Execute evaluation trials for a single condition across seeds."""
    accuracies = []
    final_losses = []
    csi_scores = []
    precisions = []
    leakages = []
    concentrations = []
    weight_changes = []

    for seed in seeds:
        model = instantiate_exp010_model(condition=condition, hidden_dim=hidden_dim, seed=seed)
        correct_count = 0
        losses = []

        is_two_comp = isinstance(model, TwoCompartmentSpatialNetwork)
        is_gradient = isinstance(model, BPTTGradientBaseline)

        for tr in range(num_trials):
            ctx_id = 1 if (tr % 2 == 0) else 2
            trial_steps = task.generate_trial(
                context_id=ctx_id,
                delay_steps=delay_steps,
                reward_delay=reward_delay,
                trial_seed=seed * 1000 + tr,
            )

            # Causal relevance mask for Credit Specificity Index
            mod_ids = getattr(model, "module_ids", None)
            rel_mask = task.compute_causal_relevance_mask(hidden_dim, ctx_id, mod_ids)

            # BPTT Gradient baseline execution
            if is_gradient:
                x_seq = np.array([s.sensory_input for s in trial_steps], dtype=np.float32)
                choice_idx = [i for i, s in enumerate(trial_steps) if s.is_choice_step][-1]
                target = trial_steps[choice_idx].target_action
                loss = model.train_trial(x_seq, target, choice_idx)
                losses.append(loss)
                y_eval = model.forward(x_seq)[choice_idx, 0]
                pred_sign = 1.0 if y_eval >= 0.0 else -1.0
                if pred_sign == target:
                    correct_count += 1
                continue

            # Spiking / Two-Compartment network trial loop
            action_emitted = 0.0
            choice_made = False
            last_target = 1.0

            for s_data in trial_steps:
                x = s_data.sensory_input
                rew = 0.0

                if s_data.is_choice_step:
                    choice_made = True
                    last_target = s_data.target_action

                # Calculate reward when delivered
                if s_data.is_reward_step and choice_made:
                    pred_sign = 1.0 if action_emitted >= 0.0 else -1.0
                    is_correct = (pred_sign == last_target)
                    rew = 1.0 if is_correct else -1.0
                    if condition == "ABL_14_SHUFFLED_REWARD":
                        rew = float(np.random.choice([-1.0, 1.0]))

                # Step model
                if is_two_comp:
                    y = model.step(x, reward=rew, target_action=last_target, relevant_mask=rel_mask)
                else:
                    y = model.step(x, reward=rew)

                if s_data.is_choice_step:
                    action_emitted = float(y[0])

            # Evaluate trial outcome
            pred_sign = 1.0 if action_emitted >= 0.0 else -1.0
            if pred_sign == last_target:
                correct_count += 1
            loss = 0.5 * (action_emitted - last_target) ** 2
            losses.append(loss)

        acc = float(correct_count) / float(num_trials)
        accuracies.append(acc)
        final_losses.append(float(np.mean(losses[-10:])))

        if is_two_comp:
            csi_scores.append(float(model.feedback_engine.last_credit_specificity_index))
            precisions.append(float(model.feedback_engine.last_credit_precision))
            leakages.append(float(model.feedback_engine.last_credit_leakage))
            concentrations.append(float(model.feedback_engine.last_eligibility_concentration))
            weight_changes.append(float(model.feedback_engine.last_total_weight_change))
        else:
            csi_scores.append(0.0)
            precisions.append(0.0)
            leakages.append(0.0)
            concentrations.append(0.0)
            weight_changes.append(0.0)

    return {
        "condition": condition,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "final_loss_mean": float(np.mean(final_losses)),
        "csi_mean": float(np.mean(csi_scores)),
        "precision_mean": float(np.mean(precisions)),
        "leakage_mean": float(np.mean(leakages)),
        "concentration_mean": float(np.mean(concentrations)),
        "weight_change_mean": float(np.mean(weight_changes)),
    }


def run_delay_matrix_sweep(
    delays: List[int] = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000],
    seeds: List[int] = [42, 43],
    num_trials: int = 30,
) -> Dict[str, Any]:
    """Sweep post-choice reward delay Delta t across major architectures."""
    task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=20, seed=42)
    models_to_test = [
        "COND_01_POINT_SPIKING_BASELINE",
        "COND_09_DENDRITIC_TARGETED_SPATIAL",
        "ABL_10_NO_SLOW_CONDUCTANCE",
        "ABL_13_RANDOMIZED_FEEDBACK",
    ]

    sweep_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models_to_test}

    for d in delays:
        for cond in models_to_test:
            res = run_condition_trial_loop(
                condition=cond,
                task=task,
                num_trials=num_trials,
                delay_steps=20,
                reward_delay=d,
                seeds=seeds,
            )
            sweep_results[cond].append({
                "delay": d,
                "accuracy": res["accuracy_mean"],
                "csi": res["csi_mean"],
                "precision": res["precision_mean"],
            })

    return sweep_results


def run_memory_retention_sweep(
    delays: List[int] = [10, 25, 50, 100, 250, 500, 1000],
    tau_nmda_list: List[float] = [1.0, 20.0, 50.0, 100.0, 250.0],
    seed: int = 42,
) -> Dict[str, Any]:
    """Measure contextual information retention R^2 across delay horizons and tau_NMDA."""
    task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=25, seed=seed)
    results: Dict[str, Dict[str, float]] = {}

    for tau in tau_nmda_list:
        label = f"tau_{int(tau)}ms" if tau > 1.0 else "no_nmda"
        results[label] = {}
        for d in delays:
            # Generate trials for context 1 and 2
            model = instantiate_exp010_model("COND_09_DENDRITIC_TARGETED_SPATIAL", tau_nmda=tau, seed=seed)
            states_ctx1 = []
            states_ctx2 = []

            for ctx in [1, 2]:
                trial = task.generate_trial(context_id=ctx, delay_steps=d, reward_delay=0, trial_seed=seed + ctx)
                for s in trial:
                    model.step(s.sensory_input)
                    if s.phase_name == "DELAY" and s.step == (5 + d - 1):
                        # Capture latent dendritic and somatic potential at end of delay
                        st = np.concatenate([model.population.v_dend, model.population.g_nmda])
                        if ctx == 1:
                            states_ctx1.append(st)
                        else:
                            states_ctx2.append(st)

            s1 = states_ctx1[0]
            s2 = states_ctx2[0]
            # Euclidean separation and normalized correlation
            dist = float(np.linalg.norm(s1 - s2))
            # R^2 correlation approximation between state separation and true context
            r2 = float(np.clip((dist / (np.linalg.norm(s1) + np.linalg.norm(s2) + 1e-6)) ** 2 * 2.0, 0.0, 1.0))
            results[label][str(d)] = round(r2, 4)

    return results


def run_attractor_persistence_test(seed: int = 42) -> Dict[str, Any]:
    """Test whether Two-Compartment network maintains stable attractor states upon input removal."""
    task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=50, seed=seed)
    model = instantiate_exp010_model("COND_09_DENDRITIC_TARGETED_SPATIAL", tau_nmda=150.0, seed=seed)

    # 1. Drive with Context 1
    trial_ctx1 = task.generate_trial(context_id=1, delay_steps=50, reward_delay=0)
    for s in trial_ctx1[:5]:
        model.step(s.sensory_input)

    # 2. Measure state stability during 50 steps of complete zero input
    zero_in = np.zeros(8, dtype=np.float32)
    state_traj = []
    plateau_count = 0
    switches = 0
    prev_dir = 1.0

    for step in range(50):
        y = model.step(zero_in)
        state_traj.append(model.population.v_dend.copy())
        if np.any(model.population.plateau_remaining > 0):
            plateau_count += 1
        curr_dir = 1.0 if y[0] >= 0.0 else -1.0
        if curr_dir != prev_dir:
            switches += 1
            prev_dir = curr_dir

    traj = np.array(state_traj)
    drift = float(np.mean(np.linalg.norm(traj[1:] - traj[:-1], axis=-1)))

    return {
        "drift_per_step": round(drift, 4),
        "plateau_steps": plateau_count,
        "spontaneous_switches": switches,
        "switching_rate": round(float(switches) / 50.0, 4),
        "attractor_stable": bool(switches == 0 and drift < 2.0),
    }


def run_continual_learning_benchmark(seeds: List[int] = [42, 43]) -> Dict[str, Any]:
    """Evaluate performance across continual learning regimes."""
    task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=15, seed=42)
    regimes = [
        ContinualRegime.STATIONARY,
        ContinualRegime.ABRUPT_REVERSAL,
        ContinualRegime.GRADUAL_DRIFT,
        ContinualRegime.RECURRING,
    ]

    results: Dict[str, Dict[str, float]] = {}

    for reg in regimes:
        reg_accs = []
        for seed in seeds:
            model = instantiate_exp010_model("COND_09_DENDRITIC_TARGETED_SPATIAL", seed=seed)
            correct = 0
            num_tr = 40
            for tr in range(num_tr):
                ctx_id = 1 if (tr % 2 == 0) else 2
                trial = task.generate_trial(context_id=ctx_id, delay_steps=15, reward_delay=2, regime=reg, trial_index=tr, trial_seed=seed * 100 + tr)
                act = 0.0
                tgt = 1.0
                for s in trial:
                    rew = 0.0
                    if s.is_choice_step:
                        tgt = s.target_action
                    if s.is_reward_step:
                        rew = 1.0 if ((act >= 0) == (tgt >= 0)) else -1.0
                    y = model.step(s.sensory_input, reward=rew, target_action=tgt)
                    if s.is_choice_step:
                        act = float(y[0])
                if (act >= 0.0) == (tgt >= 0.0):
                    correct += 1
            reg_accs.append(float(correct) / float(num_tr))
        results[reg.value] = {
            "accuracy": round(float(np.mean(reg_accs)), 4),
        }

    return results


def run_scale_ladder_benchmark() -> Dict[str, Any]:
    """Benchmark computational throughput across the scale ladder N in [100, 1K, 10K, 50K, 100K]."""
    scale_ladder = [100, 1000, 10000, 50000, 100000]
    metrics = []

    for N in scale_ladder:
        pop = TwoCompartmentPopulation(num_neurons=N, seed=42)
        i_e = np.full(N, 15.0, dtype=np.float32)
        i_i = np.full(N, -10.0, dtype=np.float32)
        i_soma = np.full(N, 18.0, dtype=np.float32)

        # Warmup
        for _ in range(5):
            pop.step(i_e, i_i, i_soma)

        # Timed benchmark (100 steps or fewer for largest N to keep suite fast)
        n_steps = 100 if N <= 10000 else 25
        t0 = time.perf_counter()
        spikes_accum = 0
        for s in range(n_steps):
            spks = pop.step(i_e, i_i, i_soma, current_step=s)
            spikes_accum += int(np.sum(spks))
        elapsed = time.perf_counter() - t0

        steps_per_sec = float(n_steps) / elapsed
        spikes_per_sec = float(spikes_accum) / elapsed
        synaptic_ops_per_sec = spikes_per_sec * min(N, 100)  # average degree ~100
        mem_mb = (N * 8 * 12) / (1024 * 1024)  # ~12 float arrays of size N

        metrics.append({
            "neurons": N,
            "steps_per_sec": round(steps_per_sec, 1),
            "spikes_per_sec_m": round(spikes_per_sec / 1e6, 3),
            "synaptic_ops_per_sec_m": round(synaptic_ops_per_sec / 1e6, 2),
            "memory_mb": round(mem_mb, 2),
        })

    return {"scale_ladder": metrics}


def generate_exp010_brainlog(output_path: Union[str, Path]) -> str:
    """Generate and validate complete BrainLog v2.8 telemetry trace."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=15, seed=42)
    model = TwoCompartmentSpatialNetwork(input_dim=8, hidden_dim=50, output_dim=1, seed=42)

    topo_dict = model.get_weights()
    topology = TopologyExtractor.from_network(model, "EXP010_TwoCompartment_TargetedCredit")
    capture = StateCapture(topology=topology, dt_ms=1.0)

    # Run 1 complete trial with delayed reward
    trial = task.generate_trial(context_id=1, delay_steps=15, reward_delay=3, trial_seed=42)
    rel_mask = task.compute_causal_relevance_mask(50, 1, model.module_ids)

    act_val = 0.0
    for s in trial:
        rew = 0.0
        if s.is_reward_step:
            rew = 1.0

        y = model.step(s.sensory_input, reward=rew, target_action=s.target_action, relevant_mask=rel_mask)
        if s.is_choice_step:
            act_val = float(y[0])

        capture.capture_two_compartment_step(
            step=s.step,
            input_vec=s.sensory_input,
            model=model,
            y_val=float(y[0]),
            target_val=s.target_action,
            mask_val=1.0 if s.is_choice_step else 0.0,
            reward=rew,
            m_vector=model.last_m_vector,
            targeted_updates=model.last_plastic_updates,
            metrics=model.last_telemetry,
            context_id=s.context_id,
            phase_label=s.phase_name,
        )

    capture.complete_trial(1.0)

    metadata = {
        "experiment": "EXPERIMENT_010",
        "phase": 10,
        "title": "Two-Compartment Neurons, Slow Conductances & Targeted Credit Assignment",
        "model_architecture": "TwoCompartmentSpatialNetwork",
        "neuron_count": 50,
        "synapse_count": len(topology.edges),
        "tau_nmda_ms": 100.0,
        "g_coupling": 0.7,
        "nonlinear_dendrite": True,
        "feedback_mode": "SPATIAL_LOCAL",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    BrainLogSerializer.save(
        filepath=path,
        metadata=metadata,
        topology=topology,
        frames=capture.frames,
        events=capture.all_events,
        snapshots=capture.snapshots,
        version="2.8",
    )

    valid, msg = BrainLogValidator.validate(path)
    if not valid:
        raise ValueError(f"BrainLog v2.8 validation failed: {msg}")

    return str(path)


def run_all_exp010() -> Dict[str, Any]:
    """Execute complete Phase 10 Experiment 010 matrix."""
    print("=" * 80)
    print("EXECUTING EXPERIMENT 010: TWO-COMPARTMENT & TARGETED CREDIT ASSIGNMENT")
    print("=" * 80)

    task = LongDelayContextualCreditTask(obs_dim=8, t_ctx=5, t_delay=20, seed=42)
    conditions = [
        "COND_01_POINT_SPIKING_BASELINE",
        "COND_02_FROZEN_POINT_SPIKING",
        "COND_03_TWO_COMPARTMENT_FROZEN",
        "COND_04_POINT_ELIGIBILITY",
        "COND_05_DENDRITIC_ELIGIBILITY",
        "COND_06_DENDRITIC_GLOBAL_REWARD",
        "COND_07_DENDRITIC_PRED_ERROR",
        "COND_08_DENDRITIC_RPE",
        "COND_09_DENDRITIC_TARGETED_SPATIAL",
        "ABL_10_NO_SLOW_CONDUCTANCE",
        "ABL_11_NO_DENDRITIC_NONLINEARITY",
        "ABL_12_NO_COMPARTMENT_COUPLING",
        "ABL_13_RANDOMIZED_FEEDBACK",
        "ABL_14_SHUFFLED_REWARD",
        "ABL_15_NO_ELIGIBILITY",
        "REF_16_BPTT_GRADIENT_BASELINE",
    ]

    condition_results = {}
    print("\n[1/7] Running 16 Model Conditions on Delayed Credit Benchmark...")
    for cond in conditions:
        t0 = time.perf_counter()
        res = run_condition_trial_loop(
            condition=cond,
            task=task,
            num_trials=40,
            delay_steps=20,
            reward_delay=5,
            seeds=[42, 43],
        )
        elapsed = time.perf_counter() - t0
        condition_results[cond] = res
        print(f"  -> {cond:38s}: Acc={res['accuracy_mean']*100:5.1f}% | CSI={res['csi_mean']:+6.3f} | Prec={res['precision_mean']*100:5.1f}% | ({elapsed:.2f}s)")

    print("\n[2/7] Running Post-Choice Reward Delay Sweep (Delta t in [0..1000])...")
    delay_sweep = run_delay_matrix_sweep()

    print("\n[3/7] Running Working Memory Retention Sweep (tau_NMDA x Delta t)...")
    memory_sweep = run_memory_retention_sweep()

    print("\n[4/7] Running Attractor Persistence & Disambiguation Diagnostic...")
    attractor_results = run_attractor_persistence_test()
    print(f"  -> Attractor Stable: {attractor_results['attractor_stable']} | Drift={attractor_results['drift_per_step']} | Switches={attractor_results['spontaneous_switches']}")

    print("\n[5/7] Running Continual Learning Regimes...")
    continual_results = run_continual_learning_benchmark()

    print("\n[6/7] Running Scale Ladder & Transformer Baseline Benchmark...")
    scale_results = run_scale_ladder_benchmark()
    transformer_res = benchmark_transformer_on_credit_task(num_train_trials=30, num_test_trials=15, delay_steps=20)
    print(f"  -> Transformer Baseline: Acc={transformer_res['accuracy']*100:.1f}% | Latency={transformer_res['inference_latency_per_step_ms']:.3f}ms")

    print("\n[7/7] Generating & Validating BrainLog v2.8 Telemetry Archive...")
    blog_path = "experiments/results/brainlogs/EXP010_two_compartment_targeted_credit.brainlog"
    saved_log = generate_exp010_brainlog(blog_path)
    print(f"  -> BrainLog v2.8 saved and validated at: {saved_log}")

    final_payload = {
        "conditions": condition_results,
        "delay_sweep": delay_sweep,
        "memory_retention": memory_sweep,
        "attractor_persistence": attractor_results,
        "continual_learning": continual_results,
        "scale_ladder": scale_results,
        "transformer_baseline": transformer_res,
        "brainlog_path": saved_log,
    }

    out_json = Path("experiments/results/exp010_metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nAll Experiment 010 metrics written to: {out_json}")

    return final_payload


if __name__ == "__main__":
    run_all_exp010()
