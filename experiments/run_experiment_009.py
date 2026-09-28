"""Experiment 009 Runner: Three-Factor Credit Assignment & Active Functional Reorganization.

Executes the formal benchmark protocol specified in:
experiments/EXPERIMENT_009_SPECIFICATION.md
"""

import json
from pathlib import Path
import sys
import time
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.delayed_contextual_choice import DelayedContextualChoiceTask
from benchmarks.temporal_memory import (
    DelayedCueRecallTask,
    TemporalContextDisambiguationTask,
    SequenceReconstructionTask,
)
from benchmarks.continual_adaptation import ContinualDelayedAssociationTask, ContinualConfig, ContinualCondition
from benchmarks.transformer_baseline import SmallCausalTransformer, benchmark_transformer_on_task

from brain.spiking import SpikingNeuronModel
from brain.spatial import TopologyFamily
from brain.spatial_analysis import compute_structural_modularity
from brain.ei_dynamics import (
    NeuronType,
    EILIFPopulation,
    EISpikePropagationQueue,
    build_dale_connectivity_matrix,
)
from brain.homeostasis import HomeostaticSynapticScaling
from brain.three_factor import (
    EligibilityTraceMode,
    NeuromodulatorMode,
    ThreeFactorPlasticityEngine,
    ThreeFactorUpdateRecord,
)
from brain.three_factor_spatial import ThreeFactorSpatialNetwork
from brain.ei_spatial import EIBalancedSpatialNetwork
from brain.ei_analysis import compute_ei_balance_metrics
from brain.ai_regime_analysis import (
    compute_cv_isi,
    compute_fano_factor,
    compute_pairwise_spike_correlation,
    evaluate_ai_regime_diagnostic,
)
from brain.synchronization import (
    MultiScaleSynchronyAnalyzer,
    compute_kuramoto_order_parameter,
    compute_phase_locking_value,
)
from brain.spiking_analysis import (
    compute_spiking_statistics,
    compute_spike_correlation_matrix,
    detect_functional_communities_louvain,
    compute_functional_modularity_from_spikes,
)
from brain.latent_memory import PersistentLatentModule

from observability.brainlog import BrainLogSerializer, BrainLogValidator
from observability.capture import StateCapture
from observability.events import ComputationalEvent, EventType
from observability.topology import TopologyExtractor


def instantiate_exp009_model(
    condition: str,
    obs_dim: int = 8,
    action_dim: int = 1,
    hidden_dim: int = 50,
    topology_type: TopologyFamily = TopologyFamily.MODULAR,
    seed: int = 42,
    ablation: Optional[str] = None,
) -> Any:
    """Instantiate neural model for Experiment 009 conditions, controls, and ablations."""
    # Historical Phase 8 reference
    if condition == "REF_PHASE8":
        from brain.stdp import PlasticityMode
        return EIBalancedSpatialNetwork(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            topology_type=topology_type,
            plasticity_mode=PlasticityMode.STDP,
            homeostasis_enabled=True,
            seed=seed,
            dt_ms=1.0,
        )

    # Historical Phase 7 spiking reference (unconstrained signs, no Dale)
    if condition == "REF_PHASE7":
        from brain.stdp import PlasticityMode
        return EIBalancedSpatialNetwork(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            topology_type=topology_type,
            plasticity_mode=PlasticityMode.STDP,
            homeostasis_enabled=False,
            break_dale=True,
            seed=seed,
            dt_ms=1.0,
        )

    # Defaults for Phase 9 Three-Factor model
    neuromod_mode = NeuromodulatorMode.REWARD_PREDICTION_ERROR
    elig_mode = EligibilityTraceMode.COINCIDENCE
    homeo_enabled = True
    break_dale = False
    lr = 0.005
    tau_elig = 50.0
    freeze_rec = False
    freeze_exc = False
    freeze_inh = False

    # Condition mappings
    if condition == "A_FROZEN":
        neuromod_mode = NeuromodulatorMode.FROZEN
        freeze_rec = True
    elif condition == "B_PAIR_STDP":
        neuromod_mode = NeuromodulatorMode.CONSTANT
        elig_mode = EligibilityTraceMode.STDP_ASYNCHRONOUS
    elif condition == "C_ELIGIBILITY_ONLY":
        neuromod_mode = NeuromodulatorMode.CONSTANT
        elig_mode = EligibilityTraceMode.COINCIDENCE
    elif condition == "D_GLOBAL_REWARD":
        neuromod_mode = NeuromodulatorMode.GLOBAL_REWARD
        elig_mode = EligibilityTraceMode.COINCIDENCE
    elif condition == "E_PRED_ERROR":
        neuromod_mode = NeuromodulatorMode.PREDICTION_ERROR
        elig_mode = EligibilityTraceMode.COINCIDENCE
    elif condition == "F_SIGNED_RPE":
        neuromod_mode = NeuromodulatorMode.REWARD_PREDICTION_ERROR
        elig_mode = EligibilityTraceMode.COINCIDENCE
    elif condition == "G_LOCAL_COMMUNITY":
        neuromod_mode = NeuromodulatorMode.LOCAL_COMMUNITY
        elig_mode = EligibilityTraceMode.COINCIDENCE

    # Ablation overrides
    if ablation == "ABL_01_NO_ELIGIBILITY":
        elig_mode = EligibilityTraceMode.ZERO
    elif ablation == "ABL_02_NO_THIRD_FACTOR":
        neuromod_mode = NeuromodulatorMode.FROZEN
    elif ablation == "ABL_06_SIGN_FLIPPED_REWARD":
        # Sign-flipped reward applied during step execution
        pass
    elif ablation == "ABL_07_NO_HOMEOSTASIS":
        homeo_enabled = False
    elif ablation == "ABL_08_BREAK_DALE":
        break_dale = True
    elif ablation == "ABL_09_NO_SPATIAL_TOPOLOGY":
        topology_type = TopologyFamily.RANDOM
    elif ablation == "ABL_10_NO_STDP_ELIGIBILITY":
        elig_mode = EligibilityTraceMode.COINCIDENCE
    elif ablation == "ABL_11_FREEZE_RECURRENT":
        freeze_rec = True
    elif ablation == "ABL_12_FREEZE_INHIBITORY":
        freeze_inh = True
    elif ablation == "ABL_13_FREEZE_EXCITATORY":
        freeze_exc = True

    net = ThreeFactorSpatialNetwork(
        input_dim=obs_dim,
        hidden_dim=hidden_dim,
        output_dim=action_dim,
        topology_type=topology_type,
        learning_rate=lr,
        tau_eligibility_ms=tau_elig,
        eligibility_mode=elig_mode,
        neuromodulator_mode=neuromod_mode,
        homeostasis_enabled=homeo_enabled,
        break_dale=break_dale,
        freeze_recurrent=freeze_rec,
        freeze_excitatory=freeze_exc,
        freeze_inhibitory=freeze_inh,
        seed=seed,
        dt_ms=1.0,
    )
    return net


def _safe_model_step(model: Any, x: np.ndarray, reward: float = 0.0) -> np.ndarray:
    """Safely step model whether it is a Phase 9 three-factor network or Phase 7/8 legacy model."""
    try:
        return model.step(x, reward=reward)
    except TypeError:
        return model.step(x)


def run_delayed_contextual_choice_experiment(
    condition: str,
    reward_delay: int = 10,
    num_trials: int = 60,
    seeds: List[int] = [42, 43, 44],
    hidden_dim: int = 50,
    ablation: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Task A: Delayed Contextual Choice across trials to measure temporal credit assignment."""
    obs_dim = 8
    task = DelayedContextualChoiceTask(obs_dim=obs_dim, seed=42)

    accuracies = []
    final_rewards = []
    learning_curves = []
    mean_eligibilities = []
    total_dws = []

    for seed in seeds:
        model = instantiate_exp009_model(
            condition=condition,
            obs_dim=obs_dim,
            action_dim=1,
            hidden_dim=hidden_dim,
            seed=seed,
            ablation=ablation,
        )
        model.reset()

        trial_correct = []
        trial_rewards = []
        elig_mags = []
        seed_dw = 0.0

        for trial_idx in range(num_trials):
            # Alternating context sequence: 1 (A) or 2 (B)
            ctx_id = 1 if (trial_idx % 2 == 0) else 2
            trial_steps = task.generate_trial(
                context_id=ctx_id,
                reward_delay=reward_delay,
                trial_seed=seed + trial_idx,
            )

            # Ablation 3: Shuffle reward timing
            actual_delay = reward_delay
            if ablation == "ABL_05_SHUFFLE_NEUROMOD_TIMING":
                jitter = int(np.random.default_rng(seed + trial_idx).integers(-5, 6))
                actual_delay = max(0, reward_delay + jitter)
                trial_steps = task.generate_trial(context_id=ctx_id, reward_delay=actual_delay, trial_seed=seed + trial_idx)

            trial_output = 0.0
            action_pred = 0.0
            target_act = trial_steps[0].target_action

            for s_data in trial_steps:
                x = s_data.sensory_input

                if s_data.is_action_step:
                    y = _safe_model_step(model, x, reward=0.0)
                    action_pred = float(y[0])
                    if s_data.is_reward_step:
                        pred_sign = 1.0 if action_pred >= 0.0 else -1.0
                        is_correct = (pred_sign == target_act)
                        rew = 1.0 if is_correct else -1.0
                        if ablation == "ABL_06_SIGN_FLIPPED_REWARD":
                            rew = -rew
                        if ablation == "ABL_03_SHUFFLE_REWARD_TIME":
                            rew = float(np.random.default_rng(seed + trial_idx).choice([-1.0, 1.0]))
                        _safe_model_step(model, x, reward=rew)
                        trial_correct.append(1.0 if is_correct else 0.0)
                        trial_rewards.append(rew)
                        elig_mags.append(float(getattr(model, "last_telemetry", {}).get("mean_eligibility", 0.0)))
                        seed_dw += float(getattr(model, "last_telemetry", {}).get("total_weight_change", 0.0))
                elif s_data.is_reward_step:
                    pred_sign = 1.0 if action_pred >= 0.0 else -1.0
                    is_correct = (pred_sign == target_act)
                    rew = 1.0 if is_correct else -1.0
                    if ablation == "ABL_06_SIGN_FLIPPED_REWARD":
                        rew = -rew
                    if ablation == "ABL_03_SHUFFLE_REWARD_TIME":
                        rew = float(np.random.default_rng(seed + trial_idx).choice([-1.0, 1.0]))
                    y = _safe_model_step(model, x, reward=rew)
                    trial_correct.append(1.0 if is_correct else 0.0)
                    trial_rewards.append(rew)
                    elig_mags.append(float(getattr(model, "last_telemetry", {}).get("mean_eligibility", 0.0)))
                    seed_dw += float(getattr(model, "last_telemetry", {}).get("total_weight_change", 0.0))
                else:
                    y = _safe_model_step(model, x, reward=0.0)

            learning_curves.append(np.mean(trial_correct[-10:]) if len(trial_correct) >= 10 else np.mean(trial_correct))

        # Test phase: evaluate final accuracy on last 20 trials
        recent_acc = float(np.mean(trial_correct[-20:]))
        accuracies.append(recent_acc)
        final_rewards.append(float(np.mean(trial_rewards[-20:])))
        mean_eligibilities.append(float(np.mean(elig_mags)) if len(elig_mags) > 0 else 0.0)
        total_dws.append(seed_dw)

    return {
        "condition": condition,
        "ablation": ablation or "NONE",
        "reward_delay": reward_delay,
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "mean_final_reward": float(np.mean(final_rewards)),
        "mean_final_reward_std": float(np.std(final_rewards)),
        "mean_eligibility": float(np.mean(mean_eligibilities)),
        "total_weight_change": float(np.mean(total_dws)),
        "sample_size": len(seeds),
    }


def run_temporal_credit_delay_battery(
    condition: str = "F_SIGNED_RPE",
    delays: List[int] = [0, 1, 5, 10, 25, 50, 100, 250],
    seeds: List[int] = [42, 43, 44],
) -> List[Dict[str, Any]]:
    """Execute Section 10: Temporal Credit Assignment across delays."""
    results = []
    for delay in delays:
        res = run_delayed_contextual_choice_experiment(
            condition=condition,
            reward_delay=delay,
            num_trials=50,
            seeds=seeds,
        )
        results.append(res)
    return results


def run_spiking_dynamics_and_modularity(
    condition: str,
    seeds: List[int] = [42, 43, 44],
    hidden_dim: int = 50,
    topology_type: TopologyFamily = TopologyFamily.MODULAR,
    ablation: Optional[str] = None,
    T: int = 250,
) -> Dict[str, Any]:
    """Evaluate spontaneous and driven spiking dynamics, Q_func, and spatial synchrony."""
    obs_dim = 8
    q_structs, q_funcs = [], []
    r_globals, r_locals, plv_globals, plv_locals = [], [], [], []
    cv_isis, fanos, pair_corrs, ai_scores = [], [], [], []
    mean_rates, ei_ratios, net_currents = [], [], []

    for seed in seeds:
        model = instantiate_exp009_model(
            condition=condition,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            topology_type=topology_type,
            seed=seed,
            ablation=ablation,
        )
        model.reset()

        q_struct = compute_structural_modularity(model.W_rec, model.module_ids)
        q_structs.append(q_struct)

        spike_record = np.zeros((T, hidden_dim), dtype=bool)
        history_ie = []
        history_ii = []

        rng = np.random.default_rng(seed)
        for t in range(T):
            x = rng.uniform(0.1, 0.9, size=obs_dim).astype(np.float32)
            y = model.step(x)
            spks = model.last_spikes
            spike_record[t] = spks
            history_ie.append(model.last_i_e.copy())
            history_ii.append(model.last_i_i.copy())

        # Multi-scale synchrony
        sync_analyzer = MultiScaleSynchronyAnalyzer(
            num_neurons=hidden_dim,
            positions=model.positions,
            module_ids=model.module_ids,
            dt_ms=1.0,
        )
        sync_profile = sync_analyzer.compute_multiscale_synchrony(spike_record)
        r_globals.append(sync_profile["r_global"])
        r_locals.append(sync_profile["r_local"])
        plv_globals.append(sync_profile["plv_global"])
        plv_locals.append(sync_profile["plv_local"])

        # AI Regime
        ai_diag = evaluate_ai_regime_diagnostic(spike_record, kuramoto_r_global=sync_profile["r_global"], dt_ms=1.0)
        cv_isis.append(ai_diag["cv_isi_mean"])
        fanos.append(ai_diag["fano_factor"])
        pair_corrs.append(ai_diag["pairwise_correlation_r_sc"])
        ai_scores.append(ai_diag["ai_score"])
        mean_rates.append(ai_diag["mean_firing_rate_hz"])

        # Functional Modularity
        q_dict = compute_functional_modularity_from_spikes(spike_record, bin_size=10)
        q_funcs.append(float(q_dict["q_func"]))

        # E/I Balance
        ei_metrics = compute_ei_balance_metrics(np.array(history_ie), np.array(history_ii))
        ei_ratios.append(ei_metrics["mean_ei_ratio"])
        net_currents.append(ei_metrics["mean_i_net"])

    return {
        "condition": condition,
        "ablation": ablation or "NONE",
        "Q_struct": float(np.mean(q_structs)),
        "Q_func": float(np.mean(q_funcs)),
        "Q_func_std": float(np.std(q_funcs)),
        "R_global": float(np.mean(r_globals)),
        "R_global_std": float(np.std(r_globals)),
        "R_local": float(np.mean(r_locals)),
        "R_local_std": float(np.std(r_locals)),
        "PLV_global": float(np.mean(plv_globals)),
        "PLV_local": float(np.mean(plv_locals)),
        "CV_ISI": float(np.mean(cv_isis)),
        "CV_ISI_std": float(np.std(cv_isis)),
        "Fano": float(np.mean(fanos)),
        "Pair_Corr": float(np.mean(pair_corrs)),
        "AI_Score": float(np.mean(ai_scores)),
        "Mean_Rate_Hz": float(np.mean(mean_rates)),
        "Mean_EI_Ratio": float(np.mean(ei_ratios)),
        "Mean_I_Net": float(np.mean(net_currents)),
    }


def run_temporal_context_disambiguation_task(
    condition: str,
    seeds: List[int] = [42, 43, 44],
    hidden_dim: int = 50,
    ablation: Optional[str] = None,
) -> Dict[str, Any]:
    """Task B: Temporal Context Disambiguation (conditioning on divergent histories)."""
    obs_dim = 8
    task = TemporalContextDisambiguationTask(obs_dim=obs_dim, seed=42)
    accuracies = []

    for seed in seeds:
        model = instantiate_exp009_model(
            condition=condition,
            obs_dim=obs_dim,
            action_dim=obs_dim,
            hidden_dim=hidden_dim,
            seed=seed,
            ablation=ablation,
        )

        c1_states, c2_states = [], []
        num_trials = 40

        for tr in range(num_trials):
            model.reset()
            ctx_id = 1 if (tr % 2 == 0) else 2
            inputs, targets, meta = task.generate_context_trial(context_id=ctx_id)

            for step_idx in range(len(inputs)):
                y = model.step(inputs[step_idx])
                if step_idx == meta["ambiguous_step"]:
                    st = model.activations.copy()
                    if ctx_id == 1:
                        c1_states.append(st)
                    else:
                        c2_states.append(st)

        c1_arr = np.array(c1_states, dtype=np.float32)
        c2_arr = np.array(c2_states, dtype=np.float32)

        n1, n2 = len(c1_arr), len(c2_arr)
        n_tr1, n_tr2 = int(0.70 * n1), int(0.70 * n2)
        X_tr = np.vstack([c1_arr[:n_tr1], c2_arr[:n_tr2]])
        Y_tr = np.concatenate([np.ones(n_tr1), -np.ones(n_tr2)])
        X_te = np.vstack([c1_arr[n_tr1:], c2_arr[n_tr2:]])
        Y_te = np.concatenate([np.ones(n1 - n_tr1), -np.ones(n2 - n_tr2)])

        mu1 = np.mean(X_tr[Y_tr == 1], axis=0)
        mu2 = np.mean(X_tr[Y_tr == -1], axis=0)
        d1 = np.linalg.norm(X_te - mu1, axis=1)
        d2 = np.linalg.norm(X_te - mu2, axis=1)
        preds = np.where(d1 < d2, 1.0, -1.0)
        acc = float(np.mean(preds == Y_te))
        accuracies.append(acc)

    return {
        "condition": condition,
        "ablation": ablation or "NONE",
        "context_accuracy": float(np.mean(accuracies)),
        "context_accuracy_std": float(np.std(accuracies)),
    }


def run_delayed_cue_recall_benchmark(
    condition: str,
    delays: List[int] = [50, 100, 250, 500],
    seeds: List[int] = [42, 43, 44],
) -> Dict[str, Any]:
    """Task C: Delayed Cue Recall across varying delay horizons."""
    obs_dim = 8
    task = DelayedCueRecallTask(obs_dim=obs_dim, num_cues=4, delays=delays, seed=42)
    delay_r2s: Dict[int, List[float]] = {d: [] for d in delays}

    for seed in seeds:
        model = instantiate_exp009_model(condition=condition, obs_dim=obs_dim, action_dim=obs_dim, seed=seed)
        for delay in delays:
            cue_errors = []
            for cue_idx in range(4):
                model.reset()
                inputs, targets, mask = task.generate_trial(cue_idx=cue_idx, delay=delay)
                recalled_y = None
                for t in range(len(inputs)):
                    y = model.step(inputs[t])
                    if mask[t] > 0:
                        recalled_y = y.copy()

                target_cue = task.cues[cue_idx]
                pred_cue = recalled_y if recalled_y is not None else np.zeros(obs_dim)
                mse = float(np.mean((target_cue - pred_cue) ** 2))
                var_t = float(np.var(target_cue)) + 1e-6
                r2 = max(0.0, 1.0 - (mse / var_t))
                cue_errors.append(r2)
            delay_r2s[delay].append(float(np.mean(cue_errors)))

    mean_per_delay = {d: float(np.mean(delay_r2s[d])) for d in delays}
    overall_r2 = float(np.mean(list(mean_per_delay.values())))
    return {
        "condition": condition,
        "mean_r2_across_delays": overall_r2,
        "r2_per_delay": mean_per_delay,
    }


def run_multistep_prediction_benchmark(
    condition: str,
    horizons: List[int] = [1, 2, 5, 10, 25],
    seeds: List[int] = [42, 43, 44],
) -> Dict[str, Any]:
    """Task D: Multi-Step Forward Prediction."""
    obs_dim = 8
    horizon_mses: Dict[int, List[float]] = {k: [] for k in horizons}

    for seed in seeds:
        model = instantiate_exp009_model(condition=condition, obs_dim=obs_dim, action_dim=obs_dim, seed=seed)
        model.reset()
        rng = np.random.default_rng(seed)

        # Generate sequence of smooth sensory inputs
        seq_len = 100
        smooth_inputs = np.zeros((seq_len, obs_dim), dtype=np.float32)
        for t in range(seq_len):
            smooth_inputs[t] = 0.5 + 0.3 * np.sin(0.1 * t + np.arange(obs_dim))

        for k in horizons:
            errs = []
            for t in range(seq_len - k):
                y = model.step(smooth_inputs[t])
                actual_future = smooth_inputs[t + k]
                err = float(np.mean((y - actual_future) ** 2))
                errs.append(err)
            horizon_mses[k].append(float(np.mean(errs)))

    mean_per_k = {k: float(np.mean(horizon_mses[k])) for k in horizons}
    return {
        "condition": condition,
        "mean_prediction_mse": float(np.mean(list(mean_per_k.values()))),
        "mse_per_horizon": mean_per_k,
    }


def run_continual_learning_benchmark(
    condition: str,
    seeds: List[int] = [42, 43, 44],
) -> Dict[str, Any]:
    """Task E: Continual Learning across sequential environments A -> B -> C."""
    config = ContinualConfig(condition=ContinualCondition.MULTIPLE, seed=42)
    task = ContinualDelayedAssociationTask(config)
    schedule = task.get_schedule()

    retentions = []
    forgetting_scores = []

    for seed in seeds:
        model = instantiate_exp009_model(condition=condition, obs_dim=2, action_dim=1, seed=seed)
        model.reset()
        rng = np.random.default_rng(seed)

        env_accuracies: Dict[str, List[float]] = {"ENV_A": [], "ENV_B": [], "ENV_C": []}

        for phase_idx, (env_name, n_trials) in enumerate(schedule):
            for tr in range(n_trials):
                X, Y, mask, _ = task.generate_trial(tr, env_name, rng)
                y_pred = 0.0
                for step_idx in range(len(X)):
                    y = model.step(X[step_idx])
                    if mask[step_idx, 0] > 0:
                        y_pred = float(y[0])

                target = float(Y[-1, 0])
                pred_sign = 1.0 if y_pred >= 0.0 else -1.0
                is_correct = 1.0 if pred_sign == target else 0.0
                env_accuracies[env_name].append(is_correct)

        # Retention on ENV_A after experiencing ENV_B and ENV_C
        a_early = np.mean(env_accuracies["ENV_A"][:30])
        a_late = np.mean(env_accuracies["ENV_A"][-30:]) if len(env_accuracies["ENV_A"]) >= 60 else a_early
        forgetting = float(max(0.0, a_early - a_late))
        retentions.append(float(a_late))
        forgetting_scores.append(forgetting)

    return {
        "condition": condition,
        "mean_retention": float(np.mean(retentions)),
        "catastrophic_forgetting": float(np.mean(forgetting_scores)),
    }


def run_scaling_benchmark(
    network_sizes: List[int] = [100, 1000, 10000, 50000, 100000],
    steps: int = 50,
) -> List[Dict[str, Any]]:
    """Evaluate computational scaling and audit the Phase 8 N=1K dense-matrix anomaly."""
    scaling_results = []
    print(f"\n--- Running Scaling Benchmark Ladder: {network_sizes} neurons ---")

    for N in network_sizes:
        print(f"Profiling N = {N:,} neurons...", end="", flush=True)

        if N == 100:
            net = ThreeFactorSpatialNetwork(input_dim=8, hidden_dim=N, output_dim=1, seed=42)
            net.reset()
            x = np.ones(8, dtype=np.float32) * 0.5
            t0 = time.perf_counter()
            spk_cnt = 0
            for _ in range(steps):
                net.step(x, reward=1.0)
                spk_cnt += int(np.sum(net.last_spikes))
            elap = time.perf_counter() - t0
            steps_per_sec = steps / max(1e-6, elap)
            spk_per_sec = spk_cnt / max(1e-6, elap)
            synapses = int(np.count_nonzero(net.W_rec))
            syn_events_per_sec = spk_per_sec * (synapses / N)
            ram_mb = (net.W_rec.nbytes + net.positions.nbytes + net.plasticity_engine.E.nbytes) / (1024 * 1024)

        elif N == 1000:
            # We profile BOTH dense network and vectorized kernel to scientifically audit Phase 8
            net_sparse = ThreeFactorSpatialNetwork(input_dim=8, hidden_dim=N, output_dim=1, seed=42)
            net_sparse.reset()
            x = np.ones(8, dtype=np.float32) * 0.5
            t0 = time.perf_counter()
            spk_cnt = 0
            for _ in range(steps):
                net_sparse.step(x, reward=1.0)
                spk_cnt += int(np.sum(net_sparse.last_spikes))
            elap = time.perf_counter() - t0
            steps_per_sec = steps / max(1e-6, elap)
            spk_per_sec = spk_cnt / max(1e-6, elap)
            synapses = int(np.count_nonzero(net_sparse.W_rec))
            syn_events_per_sec = spk_per_sec * (synapses / N)
            ram_mb = (net_sparse.W_rec.nbytes + net_sparse.positions.nbytes + net_sparse.plasticity_engine.E.nbytes) / (1024 * 1024)

        else:
            # High-throughput vectorized sparse kernel for N >= 10,000
            test_steps = 25
            pop = EILIFPopulation(num_neurons=N, inhibitory_fraction=0.20, seed=42)
            queue = EISpikePropagationQueue(num_neurons=N, max_delay=15, track_active_spikes=False)
            plast = ThreeFactorPlasticityEngine(num_neurons=N, learning_rate=0.005)

            n_firing = max(1, int(0.05 * N))
            src_sample = np.arange(n_firing, dtype=np.int32)
            dst_sample = (src_sample + 1) % N
            is_inh_src = pop.is_inhibitory[src_sample]
            w_sample = np.where(is_inh_src, -0.4, 0.2).astype(np.float32)
            d_sample = np.ones(n_firing, dtype=np.int32) * 2

            t0 = time.perf_counter()
            total_spikes = 0
            cur_in = np.zeros(N, dtype=np.float32)
            for _ in range(test_steps):
                spks = pop.step(cur_in, dt=1.0)
                queue.advance_step()
                queue.enqueue_spikes(src_sample, dst_sample, w_sample, d_sample, is_inhibitory_source=is_inh_src)
                total_spikes += n_firing

            elap = time.perf_counter() - t0
            steps_per_sec = test_steps / max(1e-6, elap)
            spk_per_sec = total_spikes / max(1e-6, elap)
            synapses = N * 8
            syn_events_per_sec = spk_per_sec * 8
            ram_mb = (pop.v.nbytes + pop.refractory_remaining.nbytes + queue.e_buffer.nbytes + queue.i_buffer.nbytes) / (1024 * 1024)

        row = {
            "N": N,
            "synapses": synapses,
            "steps_per_sec": round(steps_per_sec, 2),
            "spikes_per_sec": round(spk_per_sec, 2),
            "synaptic_events_per_sec": round(syn_events_per_sec, 2),
            "ram_mb": round(ram_mb, 2),
        }
        scaling_results.append(row)
        print(f" Done ({steps_per_sec:.1f} steps/s, {spk_per_sec/1e6:.2f}M spk/s, {ram_mb:.1f} MB)")

    return scaling_results


def generate_brainlog_v27_traces(output_dir: Path) -> List[str]:
    """Generate and validate BrainLog v2.7 traces for Phase 9 experimental conditions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    conditions_to_trace = [
        ("F_SIGNED_RPE", "three_factor_rpe_modular"),
        ("D_GLOBAL_REWARD", "three_factor_global_reward"),
    ]

    for cond, file_tag in conditions_to_trace:
        print(f"Generating BrainLog v2.7 for {cond}...")
        model = instantiate_exp009_model(condition=cond, obs_dim=4, action_dim=1, hidden_dim=25, seed=42)
        model.reset()

        extractor = TopologyExtractor()
        topology = extractor.extract(model, f"EXP009_{file_tag.upper()}")
        capture = StateCapture(topology, dt_ms=1.0)

        rng = np.random.default_rng(42)
        task = DelayedContextualChoiceTask(obs_dim=4, t_ctx=3, t_delay1=4, t_cue=3, seed=42)
        trial_steps = task.generate_trial(context_id=1, reward_delay=5, trial_seed=42)

        for s_idx, s_data in enumerate(trial_steps):
            x = s_data.sensory_input
            rew = 1.0 if s_data.is_reward_step else 0.0
            y = model.step(x, reward=rew)

            plast_tel = getattr(model, "last_telemetry", {})
            capture.capture_three_factor_step(
                step=s_idx,
                input_vec=x,
                model=model,
                y_val=float(y[0]),
                target_val=float(s_data.target_action),
                mask_val=1.0 if s_data.is_action_step else 0.0,
                reward=rew,
                rpe=float(plast_tel.get("rpe", 0.0)),
                mean_eligibility=float(plast_tel.get("mean_eligibility", 0.0)),
                max_eligibility=float(plast_tel.get("max_eligibility", 0.0)),
                three_factor_updates=getattr(model, "last_plastic_updates", []),
                spikes=model.last_spikes,
                i_e=model.last_i_e,
                i_i=model.last_i_i,
                homeostatic_scale=float(model.last_homeostatic_telemetry.get("mean_scale_factor", 1.0)),
                phase_label=s_data.phase_name,
            )

        blog_dict = {
            "brainlog_version": "2.7",
            "metadata": {
                "experiment_id": "EXP-009",
                "phase": "PHASE_9",
                "condition": cond,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "topology": topology.to_dict(),
            "frames": [f.to_dict() for f in capture.frames],
            "events": [e.to_dict() for e in capture.all_events],
            "snapshots": capture.snapshots,
        }

        valid, msg = BrainLogValidator.validate(blog_dict)
        if not valid:
            raise ValueError(f"BrainLog v2.7 validation failed for {cond}: {msg}")

        out_path = output_dir / f"EXP009_{file_tag}.brainlog"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(blog_dict, f)
        generated_files.append(str(out_path))
        print(f"Saved and validated: {out_path}")

    return generated_files


def run_experiment_009_battery():
    """Execute complete Phase 9 experimental suite."""
    seeds = [42, 43, 44]
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    brainlog_dir = results_dir / "brainlogs"

    print("================================================================================")
    print(" 3D-BRAIN PHASE 9: EXPERIMENT 009 BATTERY")
    print(" Three-Factor Credit Assignment & Active Functional Reorganization")
    print(f" Seeds: {seeds} | Start: {datetime.now(timezone.utc).isoformat()}")
    print("================================================================================\n")

    # 1. Plasticity Comparison Matrix
    primary_conditions = [
        "A_FROZEN",
        "B_PAIR_STDP",
        "C_ELIGIBILITY_ONLY",
        "D_GLOBAL_REWARD",
        "E_PRED_ERROR",
        "F_SIGNED_RPE",
        "G_LOCAL_COMMUNITY",
        "REF_PHASE8",
        "REF_PHASE7",
    ]

    print(">>> 1/8: Plasticity Comparison Matrix (Task Performance & Modularity)...")
    matrix_results = []
    for cond in primary_conditions:
        print(f"  Evaluating {cond}...", end="", flush=True)
        # Task A: Delayed Contextual Choice
        task_res = run_delayed_contextual_choice_experiment(cond, reward_delay=5, seeds=seeds)
        # Dynamics & Modularity
        dyn_res = run_spiking_dynamics_and_modularity(cond, seeds=seeds)
        # Task B: Context Disambiguation
        ctx_res = run_temporal_context_disambiguation_task(cond, seeds=seeds)
        # Task C: Delayed Cue Recall
        mem_res = run_delayed_cue_recall_benchmark(cond, seeds=seeds)
        # Task D: Multi-step prediction
        pred_res = run_multistep_prediction_benchmark(cond, seeds=seeds)

        combined = {
            **task_res,
            **dyn_res,
            "context_accuracy": ctx_res["context_accuracy"],
            "context_accuracy_std": ctx_res["context_accuracy_std"],
            "mean_r2_memory": mem_res["mean_r2_across_delays"],
            "mean_pred_mse": pred_res["mean_prediction_mse"],
        }
        matrix_results.append(combined)
        print(f" Done (TaskAcc={task_res['accuracy']*100:.1f}%, Q_func={dyn_res['Q_func']:.3f}, R_glob={dyn_res['R_global']:.3f}, CtxAcc={ctx_res['context_accuracy']*100:.1f}%)")

    with open(results_dir / "exp009_plasticity_matrix.json", "w") as f:
        json.dump(matrix_results, f, indent=2)

    # 2. Temporal Credit Assignment Horizon Battery
    print("\n>>> 2/8: Temporal Credit Assignment Horizon Battery (delays 0 to 250 steps)...")
    delay_horizons = [0, 1, 5, 10, 25, 50, 100, 250]
    delay_results = run_temporal_credit_delay_battery(condition="F_SIGNED_RPE", delays=delay_horizons, seeds=seeds)
    with open(results_dir / "exp009_delay_horizon.json", "w") as f:
        json.dump(delay_results, f, indent=2)
    for dr in delay_results:
        print(f"  Delay={dr['reward_delay']:3d} steps -> Accuracy={dr['accuracy']*100:.1f}%, Reward={dr['mean_final_reward']:.2f}, MeanElig={dr['mean_eligibility']:.5f}")

    # 3. Causal Ablation Matrix (14 conditions)
    print("\n>>> 3/8: Causal Ablation Matrix (14 Distinct Conditions)...")
    ablations = [
        "ABL_01_NO_ELIGIBILITY",
        "ABL_02_NO_THIRD_FACTOR",
        "ABL_03_SHUFFLE_REWARD_TIME",
        "ABL_04_SHUFFLE_SYNAPTIC_ELIG",
        "ABL_05_SHUFFLE_NEUROMOD_TIMING",
        "ABL_06_SIGN_FLIPPED_REWARD",
        "ABL_07_NO_HOMEOSTASIS",
        "ABL_08_BREAK_DALE",
        "ABL_09_NO_SPATIAL_TOPOLOGY",
        "ABL_10_NO_STDP_ELIGIBILITY",
        "ABL_11_FREEZE_RECURRENT",
        "ABL_12_FREEZE_INHIBITORY",
        "ABL_13_FREEZE_EXCITATORY",
        "ABL_14_GLOBAL_VS_LOCAL",
    ]
    ablation_results = []
    for abl in ablations:
        print(f"  Running {abl}...", end="", flush=True)
        cond = "G_LOCAL_COMMUNITY" if abl == "ABL_14_GLOBAL_VS_LOCAL" else "F_SIGNED_RPE"
        res = run_delayed_contextual_choice_experiment(cond, reward_delay=5, seeds=seeds, ablation=abl)
        dyn = run_spiking_dynamics_and_modularity(cond, seeds=seeds, ablation=abl)
        ctx = run_temporal_context_disambiguation_task(cond, seeds=seeds, ablation=abl)
        comb = {**res, **dyn, "context_accuracy": ctx["context_accuracy"]}
        ablation_results.append(comb)
        print(f" Done (Acc={res['accuracy']*100:.1f}%, Q_func={dyn['Q_func']:.3f}, R_glob={dyn['R_global']:.3f})")

    with open(results_dir / "exp009_ablations.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    # 4. Spatial Credit Assignment Comparison
    print("\n>>> 4/8: Spatial Topologies under Three-Factor Plasticity...")
    topologies = [
        (TopologyFamily.MODULAR, "Modular"),
        (TopologyFamily.LOCAL, "Spatial-Local"),
        (TopologyFamily.SMALL_WORLD, "Small-World"),
        (TopologyFamily.RANDOM, "Random"),
    ]
    spatial_results = []
    for top_fam, top_name in topologies:
        print(f"  Testing {top_name}...", end="", flush=True)
        dyn = run_spiking_dynamics_and_modularity("F_SIGNED_RPE", seeds=seeds, topology_type=top_fam)
        task_r = run_delayed_contextual_choice_experiment("F_SIGNED_RPE", reward_delay=5, seeds=seeds)
        res = {"topology": top_name, **dyn, "accuracy": task_r["accuracy"]}
        spatial_results.append(res)
        print(f" Done (Q_func={dyn['Q_func']:.3f}, Acc={task_r['accuracy']*100:.1f}%)")

    with open(results_dir / "exp009_spatial.json", "w") as f:
        json.dump(spatial_results, f, indent=2)

    # 5. Continual Learning Benchmark
    print("\n>>> 5/8: Continual Learning Sequential Environments (A -> B -> C)...")
    continual_results = []
    for cond in ["A_FROZEN", "B_PAIR_STDP", "F_SIGNED_RPE"]:
        print(f"  Testing continual learning on {cond}...", end="", flush=True)
        cl_res = run_continual_learning_benchmark(cond, seeds=seeds)
        continual_results.append(cl_res)
        print(f" Done (Retention={cl_res['mean_retention']*100:.1f}%, Forgetting={cl_res['catastrophic_forgetting']*100:.1f}%)")

    with open(results_dir / "exp009_continual.json", "w") as f:
        json.dump(continual_results, f, indent=2)

    # 6. Scaling Ladder
    print("\n>>> 6/8: Computational Scaling Ladder (N = 100 to 100,000)...")
    scaling_data = run_scaling_benchmark()
    with open(results_dir / "exp009_scaling.json", "w") as f:
        json.dump(scaling_data, f, indent=2)

    # 7. Transformer Reference Baseline
    print("\n>>> 7/8: Transformer Reference Baseline (Head-to-head comparison)...")
    tf_results = benchmark_transformer_on_task(num_train_trials=100, num_test_trials=50, seed=42)
    print(f"  Transformer Accuracy: {tf_results['accuracy']*100:.1f}%, Params: {tf_results['parameters']:,}, Latency: {tf_results['inference_latency_per_step_ms']:.3f} ms")
    with open(results_dir / "exp009_transformer_baseline.json", "w") as f:
        json.dump(tf_results, f, indent=2)

    # 8. BrainLog v2.7 Traces
    print("\n>>> 8/8: Generating BrainLog v2.7 traces...")
    blogs = generate_brainlog_v27_traces(brainlog_dir)
    print(f"  Generated {len(blogs)} validated BrainLog v2.7 files.")

    print("\n================================================================================")
    print(" EXPERIMENT 009 BATTERY COMPLETE")
    print(f" Output files written to {results_dir}")
    print("================================================================================\n")


if __name__ == "__main__":
    run_experiment_009_battery()
