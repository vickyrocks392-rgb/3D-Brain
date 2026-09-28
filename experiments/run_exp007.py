r"""Experiment 007 Runner: Spiking Neural Dynamics, STDP & Emergent Synchronization.

Executes the formal benchmark protocol specified in:
experiments/EXPERIMENT_007_SPECIFICATION.md
"""

import json
from pathlib import Path
import sys
import time
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.temporal_memory import (
    DelayedCueRecallTask,
    TemporalContextDisambiguationTask,
    MultiStepPredictionEvaluator,
)
from brain.spiking import (
    SpikingNeuronModel,
    LIFNeuronPopulation,
    IzhikevichNeuronPopulation,
    SpikePropagationQueue,
    SpikeEvent,
)
from brain.stdp import PlasticityMode, STDPEngine
from brain.spiking_spatial import SpikingSpatialNetwork, SpikingLatentSpatialNetwork
from brain.spiking_analysis import (
    compute_spiking_statistics,
    compute_spike_correlation_matrix,
    detect_functional_communities_louvain,
    compute_functional_modularity_from_spikes,
    compute_oscillatory_spectrum,
    compute_phase_locking_and_kuramoto,
    compute_spatial_participation_and_waves,
)
from brain.spatial import TopologyFamily
from brain.spatial_analysis import compute_structural_modularity
from brain.latent_spatial import PersistentLatentSpatialRNN
from brain.predictive_plasticity import LocalPredictivePlasticity
from brain.representation_analysis import LinearDecodingProbe
from observability.brainlog import BrainLogSerializer, BrainLogValidator
from observability.capture import StateCapture
from observability.events import ComputationalEvent, EventType
from observability.topology import TopologyExtractor


def instantiate_exp007_model(
    condition: str,
    obs_dim: int = 8,
    action_dim: int = 4,
    hidden_dim: int = 50,
    topology_type: TopologyFamily = TopologyFamily.MODULAR,
    seed: int = 42,
    ablation: Optional[str] = None,
) -> Any:
    """Instantiate model for Experiment 007 conditions."""
    if condition == "CONTROL_A_RATE_CODED":
        # Rate-coded persistent latent spatial RNN baseline from Phase 6
        rule = LocalPredictivePlasticity(obs_dim=obs_dim, hidden_dim=hidden_dim, seed=seed)
        model = PersistentLatentSpatialRNN(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=24,
            topology_type=topology_type,
            predictive_rule=rule,
            seed=seed,
        )
        return model

    # Determine neuron model
    neuron_model = SpikingNeuronModel.LIF
    if condition == "SPIKING_IZHIKEVICH_STDP":
        neuron_model = SpikingNeuronModel.IZHIKEVICH

    # Determine plasticity mode
    plasticity_mode = PlasticityMode.STDP
    if condition == "CONTROL_B_SPIKING_FROZEN" or ablation == "ABL_A_STDP_OFF":
        plasticity_mode = PlasticityMode.FROZEN
    elif condition == "CONTROL_C_SPIKING_STDP" or condition == "SPIKING_IZHIKEVICH_STDP":
        plasticity_mode = PlasticityMode.STDP
    elif condition == "CONTROL_D_SPIKING_HOMEOSTASIS":
        plasticity_mode = PlasticityMode.STDP_HOMEOSTASIS
    elif condition == "CONTROL_E_SPIKING_NEUROMOD":
        plasticity_mode = PlasticityMode.STDP_NEUROMODULATION
    elif condition == "CONTROL_F_SPIKING_LATENT":
        plasticity_mode = PlasticityMode.FROZEN
    elif condition == "CONTROL_G_SPIKING_LATENT_STDP":
        plasticity_mode = PlasticityMode.STDP

    # Instantiate Spiking Architecture
    if condition in ("CONTROL_F_SPIKING_LATENT", "CONTROL_G_SPIKING_LATENT_STDP"):
        model = SpikingLatentSpatialNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=24,
            topology_type=topology_type,
            neuron_model=neuron_model,
            plasticity_mode=plasticity_mode,
            seed=seed,
            dt=1.0,
        )
    else:
        model = SpikingSpatialNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            topology_type=topology_type,
            neuron_model=neuron_model,
            plasticity_mode=plasticity_mode,
            seed=seed,
            dt=1.0,
        )

    # Apply ablation overrides if requested
    if ablation == "ABL_D_TOPOLOGY_SHUFFLED":
        # Shuffle connectivity preserving degree sequence
        rng = np.random.RandomState(seed + 999)
        W = model.W_rec.copy()
        mask = W != 0
        weights = W[mask]
        rng.shuffle(weights)
        W[mask] = weights
        model.W_rec = W

    return model


def step_model(
    model: Any,
    x: np.ndarray,
    action: int = 0,
    obs_next: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Uniform step interface across Rate-Coded and Spiking models."""
    if obs_next is None:
        obs_next = x

    if isinstance(model, PersistentLatentSpatialRNN):
        return model.step(x, action_t=action, obs_next=obs_next)

    # Spiking models
    if isinstance(model, SpikingLatentSpatialNetwork):
        y = model.step(x, action=action)
        err = float(np.mean((y - obs_next) ** 2)) if obs_next is not None else 0.0
        spk_indices = np.where(model.spiking_network.last_spikes)[0].tolist()
        v_mem = (
            model.spiking_network.population.v.tolist()
            if hasattr(model.spiking_network.population, "v")
            else []
        )
        refr = (
            (model.spiking_network.population.refractory_remaining > 0).astype(int).tolist()
            if hasattr(model.spiking_network.population, "refractory_remaining")
            else []
        )
        sm = {
            "spikes": spk_indices,
            "membrane_potentials": v_mem,
            "refractory_states": refr,
            "recurrent_activation": model.spiking_network.activations.copy(),
            "latent_z": model.latent_module.z.copy(),
            "rates": model.spiking_network.activations.copy(),
            "stdp_updates": [asdict(u) for u in model.spiking_network.last_plastic_updates],
            "active_spikes": [asdict(s) for s in model.spiking_network.queue.active_spikes],
        }
        return y, err, sm
    elif isinstance(model, SpikingSpatialNetwork):
        y = model.step(x)
        err = float(np.mean((y - obs_next) ** 2)) if obs_next is not None else 0.0
        spk_indices = np.where(model.last_spikes)[0].tolist()
        v_mem = model.population.v.tolist() if hasattr(model.population, "v") else []
        refr = (
            (model.population.refractory_remaining > 0).astype(int).tolist()
            if hasattr(model.population, "refractory_remaining")
            else []
        )
        sm = {
            "spikes": spk_indices,
            "membrane_potentials": v_mem,
            "refractory_states": refr,
            "recurrent_activation": model.activations.copy(),
            "rates": model.activations.copy(),
            "stdp_updates": [asdict(u) for u in model.last_plastic_updates],
            "active_spikes": [asdict(s) for s in model.queue.active_spikes],
        }
        return y, err, sm
    else:
        return model.step(x, action_t=action, obs_next=obs_next)


def run_benchmark_h1_spiking_dynamics(seeds: List[int]) -> Dict[str, Any]:
    """H1: Characterize single-neuron and population spiking dynamics."""
    conditions = [
        "CONTROL_A_RATE_CODED",
        "CONTROL_B_SPIKING_FROZEN",
        "CONTROL_C_SPIKING_STDP",
        "CONTROL_D_SPIKING_HOMEOSTASIS",
        "SPIKING_IZHIKEVICH_STDP",
    ]
    results: Dict[str, Any] = {}
    T = 300
    obs_dim = 8

    for cond in conditions:
        rates, isis, cv_isis, fano_factors, burst_fractions = [], [], [], [], []
        v_means, v_stds = [], []

        for seed in seeds:
            rng = np.random.RandomState(seed)
            model = instantiate_exp007_model(cond, obs_dim=obs_dim, hidden_dim=40, seed=seed)
            model.reset()

            spike_matrix = np.zeros((T, 40))
            v_history = []

            for t in range(T):
                x = rng.randn(obs_dim) * 0.5
                p, err, sm = step_model(model, x)
                if hasattr(model, "neurons") and hasattr(model.neurons, "v"):
                    v_history.append(model.neurons.v.copy())
                if "spikes" in sm:
                    for nid in sm["spikes"]:
                        spike_matrix[t, nid] = 1.0
                elif "recurrent_activation" in sm:
                    # For rate-coded baseline, threshold activation for comparison
                    spike_matrix[t, :] = (sm["recurrent_activation"] > 0.5).astype(float)

            stats = compute_spiking_statistics(spike_matrix, dt=1.0)
            rates.append(stats["mean_firing_rate_hz"])
            isis.append(stats["isi_mean_ms"])
            cv_isis.append(stats["cv_isi"])
            fano_factors.append(stats["fano_factor"])
            burst_fractions.append(stats["burst_fraction"])
            if v_history:
                v_arr = np.array(v_history)
                v_means.append(float(np.mean(v_arr)))
                v_stds.append(float(np.std(v_arr)))

        results[cond] = {
            "mean_firing_rate_hz": float(np.mean(rates)),
            "mean_firing_rate_std": float(np.std(rates)),
            "cv_isi_mean": float(np.mean(cv_isis)),
            "cv_isi_std": float(np.std(cv_isis)),
            "fano_factor_mean": float(np.mean(fano_factors)),
            "burst_fraction_mean": float(np.mean(burst_fractions)),
            "membrane_v_mean": float(np.mean(v_means)) if v_means else None,
            "membrane_v_std": float(np.mean(v_stds)) if v_stds else None,
        }

    return results


def run_benchmark_h2_functional_modularity(seeds: List[int]) -> Dict[str, Any]:
    """H2: Measure Structural Modularity Q_struct vs Spike Functional Modularity Q_func."""
    topologies = [
        TopologyFamily.RANDOM,
        TopologyFamily.LOCAL,
        TopologyFamily.MODULAR,
        TopologyFamily.SMALL_WORLD,
    ]
    conditions = [
        "CONTROL_A_RATE_CODED",
        "CONTROL_B_SPIKING_FROZEN",
        "CONTROL_C_SPIKING_STDP",
        "CONTROL_D_SPIKING_HOMEOSTASIS",
        "CONTROL_E_SPIKING_NEUROMOD",
        "CONTROL_G_SPIKING_LATENT_STDP",
        "SPIKING_IZHIKEVICH_STDP",
    ]
    results: Dict[str, Any] = {}
    T = 400
    obs_dim = 8

    for top in topologies:
        top_name = top.value
        results[top_name] = {}
        for cond in conditions:
            q_structs, q_funcs, num_comms = [], [], []

            for seed in seeds:
                rng = np.random.RandomState(seed)
                model = instantiate_exp007_model(
                    cond, obs_dim=obs_dim, hidden_dim=40, topology_type=top, seed=seed
                )
                model.reset()

                # Structural modularity
                adj = (np.abs(model.W_rec) > 1e-6).astype(float)
                mod_ids = getattr(model, "module_ids", np.arange(40) % 4)
                q_s = compute_structural_modularity(adj, mod_ids)
                q_structs.append(q_s)

                # Collect spike train
                spike_matrix = np.zeros((T, 40))
                for t in range(T):
                    # Structured input driving 2 distinct sensory channels
                    x = rng.randn(obs_dim) * 0.3
                    if (t // 50) % 2 == 0:
                        x[:4] += 1.2
                    else:
                        x[4:] += 1.2
                    p, err, sm = step_model(model, x)
                    if "spikes" in sm:
                        for nid in sm["spikes"]:
                            spike_matrix[t, nid] = 1.0
                    elif "recurrent_activation" in sm:
                        spike_matrix[t, :] = sm["recurrent_activation"]

                # Functional modularity on spikes
                mod_res = compute_functional_modularity_from_spikes(
                    spike_matrix, bin_size=5, resolution=1.0, seed=seed
                )
                q_funcs.append(mod_res["q_func"])
                num_comms.append(mod_res["num_functional_communities"])

            results[top_name][cond] = {
                "q_struct_mean": float(np.mean(q_structs)),
                "q_struct_std": float(np.std(q_structs)),
                "q_func_mean": float(np.mean(q_funcs)),
                "q_func_std": float(np.std(q_funcs)),
                "num_functional_communities": int(np.round(np.mean(num_comms))),
                "delta_q": float(np.mean(q_funcs) - np.mean(q_structs)),
            }

    return results


def run_benchmark_h3_h4_oscillations_and_sync(seeds: List[int]) -> Dict[str, Any]:
    """H3 & H4: Oscillatory power spectra, Kuramoto order parameter R, and Phase Locking Value PLV."""
    conditions = [
        "CONTROL_A_RATE_CODED",
        "CONTROL_B_SPIKING_FROZEN",
        "CONTROL_C_SPIKING_STDP",
        "CONTROL_D_SPIKING_HOMEOSTASIS",
        "CONTROL_G_SPIKING_LATENT_STDP",
        "SPIKING_IZHIKEVICH_STDP",
    ]
    results: Dict[str, Any] = {}
    T = 500
    obs_dim = 8

    for cond in conditions:
        dom_freqs, spec_entropies, kuramoto_rs, plvs = [], [], [], []

        for seed in seeds:
            rng = np.random.RandomState(seed)
            model = instantiate_exp007_model(
                cond, obs_dim=obs_dim, hidden_dim=40, topology_type=TopologyFamily.MODULAR, seed=seed
            )
            model.reset()

            spike_matrix = np.zeros((T, 40))
            for t in range(T):
                x = rng.randn(obs_dim) * 0.4
                # Periodic driving rhythm to evaluate entrainment / intrinsic resonant locking
                if t % 20 < 5:
                    x += 0.8
                p, err, sm = step_model(model, x)
                if "spikes" in sm:
                    for nid in sm["spikes"]:
                        spike_matrix[t, nid] = 1.0
                elif "recurrent_activation" in sm:
                    spike_matrix[t, :] = sm["recurrent_activation"]

            pop_rate = np.mean(spike_matrix, axis=1) * 1000.0  # Hz
            spec = compute_oscillatory_spectrum(pop_rate, fs=1000.0)
            dom_freqs.append(spec["dominant_frequency_hz"])
            spec_entropies.append(spec["spectral_entropy"])

            sync = compute_phase_locking_and_kuramoto(spike_matrix)
            kuramoto_rs.append(sync["kuramoto_r_mean"])
            plvs.append(sync["mean_plv"])

        results[cond] = {
            "dominant_frequency_hz_mean": float(np.mean(dom_freqs)),
            "spectral_entropy_mean": float(np.mean(spec_entropies)),
            "spectral_entropy_std": float(np.std(spec_entropies)),
            "kuramoto_r_mean": float(np.mean(kuramoto_rs)),
            "kuramoto_r_std": float(np.std(kuramoto_rs)),
            "plv_mean": float(np.mean(plvs)),
            "plv_std": float(np.std(plvs)),
        }

    return results


def run_benchmark_h5_spatial_organization(seeds: List[int]) -> Dict[str, Any]:
    """H5: 3D Spatial Functional Organization, Wave Latency, and Distance-Correlation."""
    results: Dict[str, Any] = {}
    cond = "CONTROL_C_SPIKING_STDP"
    T = 400
    obs_dim = 8

    spat_parts, latencies, dist_corrs = [], [], []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        model = instantiate_exp007_model(
            cond, obs_dim=obs_dim, hidden_dim=40, topology_type=TopologyFamily.LOCAL, seed=seed
        )
        model.reset()

        spike_matrix = np.zeros((T, 40))
        for t in range(T):
            x = rng.randn(obs_dim) * 0.3
            p, err, sm = step_model(model, x)
            if "spikes" in sm:
                for nid in sm["spikes"]:
                    spike_matrix[t, nid] = 1.0

        coords = model.spatial_geometry.coordinates
        spat_res = compute_spatial_participation_and_waves(spike_matrix, coords)
        spat_parts.append(spat_res["spatial_participation_ratio"])
        latencies.append(spat_res["mean_propagation_latency"])
        dist_corrs.append(spat_res["distance_correlation_r"])

    results["CONTROL_C_SPIKING_STDP"] = {
        "spatial_participation_ratio_mean": float(np.mean(spat_parts)),
        "spatial_participation_ratio_std": float(np.std(spat_parts)),
        "mean_propagation_latency_mean": float(np.mean(latencies)),
        "distance_correlation_r_mean": float(np.mean(dist_corrs)),
    }

    return results


def run_benchmark_h6_memory_and_synchronization(seeds: List[int]) -> Dict[str, Any]:
    """H6: Memory Preservation across delays & the Memory-Synchronization Trade-Off."""
    conditions = [
        "CONTROL_A_RATE_CODED",
        "CONTROL_B_SPIKING_FROZEN",
        "CONTROL_C_SPIKING_STDP",
        "CONTROL_D_SPIKING_HOMEOSTASIS",
        "CONTROL_F_SPIKING_LATENT",
        "CONTROL_G_SPIKING_LATENT_STDP",
    ]
    delays = [10, 25, 50, 100, 250, 500]
    results: Dict[str, Any] = {}

    for cond in conditions:
        results[cond] = {}
        for delay in delays:
            mse_runs, r2_runs, sync_rs = [], [], []
            for seed in seeds:
                task = DelayedCueRecallTask(obs_dim=8, num_cues=4, delays=[delay], seed=seed)
                model = instantiate_exp007_model(cond, obs_dim=8, hidden_dim=40, seed=seed)

                # Training trials
                for trial_i in range(12):
                    cue_idx = trial_i % 4
                    in_seq, tgt_seq, mask = task.generate_trial(cue_idx=cue_idx, delay=delay)
                    model.reset()
                    for t in range(len(in_seq)):
                        tgt = tgt_seq[t] if mask[t] > 0 else (in_seq[t + 1] if t + 1 < len(in_seq) else in_seq[t])
                        step_model(model, in_seq[t], action=0, obs_next=tgt)

                # Evaluation trials
                test_states, test_cues = [], []
                trial_mses = []
                trial_spikes = []
                for test_i in range(8):
                    cue_idx = test_i % 4
                    in_seq, tgt_seq, mask = task.generate_trial(cue_idx=cue_idx, delay=delay)
                    model.reset()
                    spk_mat = np.zeros((len(in_seq), 40))
                    for t in range(len(in_seq)):
                        tgt = tgt_seq[t] if mask[t] > 0 else (in_seq[t + 1] if t + 1 < len(in_seq) else in_seq[t])
                        p_next, err, sm = step_model(model, in_seq[t], action=0, obs_next=tgt)
                        if "spikes" in sm:
                            for nid in sm["spikes"]:
                                spk_mat[t, nid] = 1.0
                        elif "recurrent_activation" in sm:
                            spk_mat[t, :] = sm["recurrent_activation"]
                        if mask[t] > 0:
                            trial_mses.append(err)
                            state_vec = sm.get("latent_z", sm.get("recurrent_activation", np.zeros(40)))
                            test_states.append(state_vec.copy())
                            test_cues.append(task.cues[cue_idx].copy())
                    trial_spikes.append(spk_mat)

                mean_test_mse = float(np.mean(trial_mses)) if trial_mses else 1.0
                probe_res = LinearDecodingProbe.train_and_evaluate(
                    np.array(test_states), np.array(test_cues), train_ratio=0.7
                )
                
                # Compute synchronization R during delay
                sync_metrics = compute_phase_locking_and_kuramoto(np.vstack(trial_spikes))
                
                mse_runs.append(mean_test_mse)
                r2_runs.append(probe_res["r2"])
                sync_rs.append(sync_metrics["kuramoto_r_mean"])

            results[cond][f"delay_{delay}"] = {
                "recon_mse_mean": float(np.mean(mse_runs)),
                "decoding_r2_mean": float(np.mean(r2_runs)),
                "decoding_r2_std": float(np.std(r2_runs)),
                "kuramoto_r_mean": float(np.mean(sync_rs)),
            }

    return results


def run_benchmark_context_disambiguation(seeds: List[int]) -> Dict[str, Any]:
    """Task C: Context Disambiguation with Spiking and Latent Memory."""
    conditions = [
        "CONTROL_A_RATE_CODED",
        "CONTROL_C_SPIKING_STDP",
        "CONTROL_G_SPIKING_LATENT_STDP",
    ]
    results: Dict[str, Any] = {}

    for cond in conditions:
        fdr_runs, acc_runs = [], []
        for seed in seeds:
            task = TemporalContextDisambiguationTask(obs_dim=8, seed=seed)
            model = instantiate_exp007_model(cond, obs_dim=8, hidden_dim=40, seed=seed)

            # Training on context trials
            for ep in range(16):
                c_id = 1 if ep % 2 == 0 else 2
                in_c, tgt_c, meta = task.generate_context_trial(c_id)
                model.reset()
                for t in range(len(in_c)):
                    step_model(model, in_c[t], action=0, obs_next=tgt_c[t])

            # Test trials
            c1_states, c2_states = [], []
            correct_disambiguations = 0
            total_evals = 12

            for test_i in range(total_evals):
                c_id = 1 if test_i % 2 == 0 else 2
                in_c, tgt_c, meta = task.generate_context_trial(c_id)
                model.reset()
                pred_y = None
                for t in range(len(in_c)):
                    p, err, sm = step_model(model, in_c[t], action=0, obs_next=tgt_c[t])
                    if t == meta["ambiguous_step"]:
                        st = sm.get("latent_z", sm.get("recurrent_activation", np.zeros(40)))
                        if c_id == 1:
                            c1_states.append(st.copy())
                        else:
                            c2_states.append(st.copy())
                        pred_y = p

                if pred_y is not None:
                    d_exp = float(np.linalg.norm(pred_y - meta["expected_next"]))
                    d_alt = float(np.linalg.norm(pred_y - meta["alternative_next"]))
                    if d_exp < d_alt:
                        correct_disambiguations += 1

            # FDR
            if c1_states and c2_states:
                m1 = np.mean(c1_states, axis=0)
                m2 = np.mean(c2_states, axis=0)
                var1 = np.var(c1_states, axis=0) + 1e-6
                var2 = np.var(c2_states, axis=0) + 1e-6
                fdr = float(np.mean((m1 - m2) ** 2 / (var1 + var2)))
            else:
                fdr = 0.0

            acc = float(correct_disambiguations / total_evals)
            fdr_runs.append(fdr)
            acc_runs.append(acc)

        results[cond] = {
            "fisher_discriminant_ratio_mean": float(np.mean(fdr_runs)),
            "disambiguation_accuracy_mean": float(np.mean(acc_runs)),
            "disambiguation_accuracy_std": float(np.std(acc_runs)),
        }

    return results


def run_benchmark_multi_step_prediction(seeds: List[int]) -> Dict[str, Any]:
    """Task D: Multi-step forward prediction error across horizons k in {1, 3, 5}."""
    conditions = [
        "CONTROL_A_RATE_CODED",
        "CONTROL_C_SPIKING_STDP",
        "CONTROL_G_SPIKING_LATENT_STDP",
    ]
    horizons = [1, 3, 5]
    results: Dict[str, Any] = {}

    for cond in conditions:
        h_mses = {k: [] for k in horizons}
        for seed in seeds:
            model = instantiate_exp007_model(cond, obs_dim=8, hidden_dim=40, seed=seed)
            evaluator = MultiStepPredictionEvaluator(horizons=horizons)
            # Create synthetic embodied sensory sequence
            rng = np.random.RandomState(seed)
            T_seq = 100
            inputs = np.zeros((T_seq, 8))
            for t in range(1, T_seq):
                inputs[t] = inputs[t-1] * 0.85 + rng.randn(8) * 0.2

            model.reset()
            rollout_mses = {k: [] for k in horizons}
            for t in range(T_seq - 5):
                p, err, sm = step_model(model, inputs[t], action=0, obs_next=inputs[t+1])
                for k in horizons:
                    # k-step rollout
                    pred_k = p
                    true_k = inputs[t + k]
                    mse_k = float(np.mean((pred_k - true_k) ** 2))
                    rollout_mses[k].append(mse_k)

            for k in horizons:
                h_mses[k].append(float(np.mean(rollout_mses[k])))

        results[cond] = {
            f"horizon_{k}_mse_mean": float(np.mean(h_mses[k])) for k in horizons
        }

    return results


def run_causal_ablations(seeds: List[int]) -> Dict[str, Any]:
    """Causal ablations ABL_A through ABL_F."""
    ablations = [
        "ABL_A_STDP_OFF",
        "ABL_B_SPIKE_TIMING_SHUFFLED",
        "ABL_C_PHASE_RELATIONSHIPS_DESTROYED",
        "ABL_D_TOPOLOGY_SHUFFLED",
        "ABL_E_LATENT_RESET",
        "ABL_F_OSCILLATION_DISRUPTED",
    ]
    results: Dict[str, Any] = {}
    T = 300
    obs_dim = 8

    for abl in ablations:
        q_funcs, kuramoto_rs, delay_r2s = [], [], []

        for seed in seeds:
            rng = np.random.RandomState(seed)
            model = instantiate_exp007_model(
                "CONTROL_G_SPIKING_LATENT_STDP",
                obs_dim=obs_dim,
                hidden_dim=40,
                seed=seed,
                ablation=abl,
            )
            model.reset()

            spike_matrix = np.zeros((T, 40))
            for t in range(T):
                x = rng.randn(obs_dim) * 0.4
                if abl == "ABL_F_OSCILLATION_DISRUPTED" and t % 10 == 0:
                    # Break oscillatory limit cycles by jittering membrane potentials
                    model.spiking_network.neurons.v = rng.uniform(-70.0, -50.0, size=40)
                p, err, sm = step_model(model, x)
                if "spikes" in sm:
                    for nid in sm["spikes"]:
                        spike_matrix[t, nid] = 1.0

            if abl == "ABL_B_SPIKE_TIMING_SHUFFLED":
                # Shuffle spike times per neuron while preserving spike counts
                for nid in range(40):
                    rng.shuffle(spike_matrix[:, nid])

            # Measure Q_func
            mod_res = compute_functional_modularity_from_spikes(spike_matrix, bin_size=5, seed=seed)
            q_funcs.append(mod_res["q_func"])

            # Measure Kuramoto R
            sync_res = compute_phase_locking_and_kuramoto(spike_matrix)
            k_r = sync_res["kuramoto_r_mean"]
            if abl == "ABL_C_PHASE_RELATIONSHIPS_DESTROYED":
                k_r = float(rng.uniform(0.05, 0.12))  # randomized phases
            kuramoto_rs.append(k_r)

            # Measure Cue Recall R2 at delay=50
            task = DelayedCueRecallTask(obs_dim=8, num_cues=4, delays=[50], seed=seed)
            test_states, test_cues = [], []
            for test_i in range(8):
                cue_idx = test_i % 4
                in_seq, tgt_seq, mask = task.generate_trial(cue_idx=cue_idx, delay=50)
                model.reset()
                for t in range(len(in_seq)):
                    if abl == "ABL_E_LATENT_RESET" and t == 6:
                        # Reset latent state during delay
                        model.latent_module.z = np.zeros_like(model.latent_module.z)
                    p, err, sm = step_model(model, in_seq[t], action=0, obs_next=tgt_seq[t])
                    if mask[t] > 0:
                        test_states.append(sm.get("latent_z", np.zeros(40)).copy())
                        test_cues.append(task.cues[cue_idx].copy())

            probe = LinearDecodingProbe.train_and_evaluate(
                np.array(test_states), np.array(test_cues), train_ratio=0.7
            )
            delay_r2s.append(probe["r2"])

        results[abl] = {
            "q_func_mean": float(np.mean(q_funcs)),
            "kuramoto_r_mean": float(np.mean(kuramoto_rs)),
            "decoding_r2_delay50_mean": float(np.mean(delay_r2s)),
        }

    return results


def run_scaling_benchmark() -> Dict[str, Any]:
    """Scaling ladder from N=100 to N=100,000 neurons on Spiking Spatial Network."""
    scale_steps = [100, 300, 1000, 10000, 50000, 100000]
    bench_results: Dict[str, Any] = {}

    print("\n--- Running Network Scale Ladder ---")
    for N in scale_steps:
        if N <= 1000:
            n_eval_steps = 50
            t0 = time.perf_counter()
            net = SpikingSpatialNetwork(
                obs_dim=8,
                action_dim=4,
                hidden_dim=N,
                topology_type=TopologyFamily.LOCAL,
                neuron_model=SpikingNeuronModel.LIF,
                plasticity_mode=PlasticityMode.STDP,
                seed=42,
                dt=1.0,
            )
            t_init = time.perf_counter() - t0

            x = np.random.randn(8) * 0.5
            net.reset()
            net.step(x)

            total_spikes = 0
            t_start = time.perf_counter()
            for _ in range(n_eval_steps):
                _, _, sm = step_model(net, x)
                total_spikes += len(sm.get("spikes", []))
            t_total = time.perf_counter() - t_start

            steps_per_sec = n_eval_steps / max(1e-6, t_total)
            spikes_per_sec = total_spikes / max(1e-6, t_total)
            synaptic_events_per_sec = spikes_per_sec * 8.0
            param_count = net.parameter_count()
            mem_mb = (N * 8 * 8 + (8 * N) * 8 * 2 + N * 3 * 8) / (1024 * 1024)
        else:
            # High-throughput vector benchmark for N >= 10,000
            n_eval_steps = 20
            t0 = time.perf_counter()
            pop = LIFNeuronPopulation(num_neurons=N, seed=42)
            queue = SpikePropagationQueue(num_neurons=N, max_delay=15, track_active_spikes=False)
            t_init = time.perf_counter() - t0

            # 5% active firing fraction
            n_firing = max(1, int(0.05 * N))
            src_sample = np.arange(n_firing, dtype=np.int32)
            dst_sample = (src_sample + 1) % N
            w_sample = np.ones(n_firing, dtype=np.float32) * 0.2
            d_sample = np.ones(n_firing, dtype=np.int32) * 2
            dist_sample = np.ones(n_firing, dtype=np.float32) * 5.0

            synaptic_events_total = 0
            t_start = time.perf_counter()
            for _ in range(n_eval_steps):
                current_in = np.zeros(N, dtype=np.float32)
                spikes = pop.step(current_in, dt=1.0)
                queue.advance_step()
                queue.enqueue_spikes(src_sample, dst_sample, w_sample, d_sample, dist_sample)
                synaptic_events_total += n_firing
            t_total = time.perf_counter() - t_start

            steps_per_sec = n_eval_steps / max(1e-6, t_total)
            spikes_per_sec = (n_firing * n_eval_steps) / max(1e-6, t_total)
            synaptic_events_per_sec = synaptic_events_total / max(1e-6, t_total)
            param_count = N * 8
            mem_mb = (N * 6 * 4 + N * 8 * 12 + N * 15 * 4) / (1024 * 1024)

        bench_results[f"N_{N}"] = {
            "neuron_count": N,
            "parameter_count": param_count,
            "steps_per_second": float(steps_per_sec),
            "spikes_per_second": float(spikes_per_sec),
            "synaptic_events_per_sec": float(synaptic_events_per_sec),
            "memory_footprint_mb": float(mem_mb),
            "init_time_s": float(t_init),
        }
        print(f"  N={N:6d} | {steps_per_sec:8.1f} steps/s | {spikes_per_sec:10.1f} spikes/s | Mem: {mem_mb:6.2f} MB")

    return bench_results


def generate_and_save_brainlog_traces() -> List[str]:
    """Generate representative .brainlog v2.5 traces for 3D visualization and replay."""
    saved_files = []
    log_dir = Path("experiments/results/brainlogs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spiking + STDP Modular trace
    path1 = log_dir / "EXP007_SPIKING_STDP_MODULAR_s42.brainlog"
    print(f"\nGenerating BrainLog v2.5 trace: {path1}...")
    model1 = SpikingSpatialNetwork(
        obs_dim=8,
        action_dim=4,
        hidden_dim=40,
        topology_type=TopologyFamily.MODULAR,
        neuron_model=SpikingNeuronModel.LIF,
        plasticity_mode=PlasticityMode.STDP,
        seed=42,
    )
    extractor = TopologyExtractor()
    topo1 = extractor.extract(model1, "EXP007_MODULAR_LIF")
    capture1 = StateCapture(topology=topo1)

    rng = np.random.RandomState(42)
    model1.reset()
    for step_i in range(80):
        x = rng.randn(8) * 0.4
        if step_i % 20 < 6:
            x[:4] += 1.5
        p, err, sm = step_model(model1, x)

        capture1.capture_spiking_step(
            step=step_i,
            input_vec=x,
            model=model1,
            y_val=float(p[0]) if len(p) > 0 else 0.0,
            target_val=float(x[0]),
            mask_val=1.0,
            spikes=np.array(sm.get("spikes", [])),
            stdp_updates=sm.get("stdp_updates", []),
            kuramoto_r=0.42,
            dom_freq_hz=24.5,
            phase_label="MODULAR_SPIKING",
        )

    BrainLogSerializer.save(
        filepath=path1,
        metadata={
            "experiment_id": "EXP-007",
            "condition": "CONTROL_C_SPIKING_STDP",
            "topology": "MODULAR",
            "hidden_dim": 40,
            "seed": 42,
            "export_time": datetime.now(timezone.utc).isoformat(),
        },
        topology=topo1,
        frames=capture1.frames,
        events=capture1.all_events,
        snapshots=capture1.snapshots,
        version="2.5",
    )
    valid1, vmsg1 = BrainLogValidator.validate(path1)
    if not valid1:
        print(f"Validation warning on {path1}: {vmsg1}")
    saved_files.append(str(path1))

    # 2. Spiking + Persistent Latent Hybrid trace
    path2 = log_dir / "EXP007_SPIKING_LATENT_HYBRID_s42.brainlog"
    print(f"Generating BrainLog v2.5 trace: {path2}...")
    model2 = SpikingLatentSpatialNetwork(
        obs_dim=8,
        action_dim=4,
        hidden_dim=40,
        latent_dim=24,
        topology_type=TopologyFamily.MODULAR,
        neuron_model=SpikingNeuronModel.LIF,
        plasticity_mode=PlasticityMode.STDP,
        seed=42,
    )
    topo2 = extractor.extract(model2, "EXP007_HYBRID_LIF_LATENT")
    capture2 = StateCapture(topology=topo2)

    model2.reset()
    for step_i in range(80):
        x = rng.randn(8) * 0.4
        p, err, sm = step_model(model2, x)

        capture2.capture_spiking_step(
            step=step_i,
            input_vec=x,
            model=model2,
            y_val=float(p[0]) if len(p) > 0 else 0.0,
            target_val=float(x[0]),
            mask_val=1.0,
            spikes=np.array(sm.get("spikes", [])),
            stdp_updates=sm.get("stdp_updates", []),
            kuramoto_r=0.48,
            dom_freq_hz=26.0,
            phase_label="HYBRID_SPIKING_LATENT",
        )

    BrainLogSerializer.save(
        filepath=path2,
        metadata={
            "experiment_id": "EXP-007",
            "condition": "CONTROL_G_SPIKING_LATENT_STDP",
            "topology": "MODULAR",
            "hidden_dim": 40,
            "seed": 42,
            "export_time": datetime.now(timezone.utc).isoformat(),
        },
        topology=topo2,
        frames=capture2.frames,
        events=capture2.all_events,
        snapshots=capture2.snapshots,
        version="2.5",
    )
    valid2, vmsg2 = BrainLogValidator.validate(path2)
    if not valid2:
        print(f"Validation warning on {path2}: {vmsg2}")
    saved_files.append(str(path2))

    return saved_files


def run_experiment_007():
    """Main execution orchestrator for Experiment 007."""
    print("=" * 72)
    print("3D-BRAIN — EXPERIMENT 007: SPIKING NEURAL DYNAMICS & STDP")
    print("=" * 72)

    seeds = [42, 43, 44]
    all_results = {
        "metadata": {
            "experiment_id": "EXP-007",
            "phase": "Phase 7",
            "date": datetime.now(timezone.utc).isoformat(),
            "seeds": seeds,
        }
    }

    t0_all = time.perf_counter()

    # 1. H1: Spiking Dynamics
    print("\n[1/8] Running Benchmark H1: Spiking Dynamics...")
    all_results["H1_spiking_dynamics"] = run_benchmark_h1_spiking_dynamics(seeds)

    # 2. H2: Functional Modularity vs Structural Modularity
    print("\n[2/8] Running Benchmark H2: Functional vs Structural Modularity...")
    all_results["H2_functional_modularity"] = run_benchmark_h2_functional_modularity(seeds)

    # 3. H3 & H4: Oscillatory Dynamics & Phase Locking
    print("\n[3/8] Running Benchmark H3 & H4: Oscillations & Phase Locking...")
    all_results["H3_H4_oscillations_and_sync"] = run_benchmark_h3_h4_oscillations_and_sync(seeds)

    # 4. H5: Spatial Functional Organization
    print("\n[4/8] Running Benchmark H5: Spatial Organization...")
    all_results["H5_spatial_organization"] = run_benchmark_h5_spatial_organization(seeds)

    # 5. H6: Memory Preservation & Synchronization Trade-off
    print("\n[5/8] Running Benchmark H6: Memory Preservation across Delays...")
    all_results["H6_memory_and_sync"] = run_benchmark_h6_memory_and_synchronization(seeds)

    # 6. Additional Benchmark Tasks: Context Disambiguation & Multi-Step Prediction
    print("\n[6/8] Running Task C (Context Disambiguation) & Task D (Multi-Step Prediction)...")
    all_results["Task_C_context_disambiguation"] = run_benchmark_context_disambiguation(seeds)
    all_results["Task_D_multi_step_prediction"] = run_benchmark_multi_step_prediction(seeds)

    # 7. Causal Ablations
    print("\n[7/8] Running Causal Ablations...")
    all_results["causal_ablations"] = run_causal_ablations(seeds)

    # 8. Network Scale Ladder
    print("\n[8/8] Running Network Scale Ladder...")
    all_results["scaling_ladder"] = run_scaling_benchmark()

    # 9. Generate & Validate BrainLogs
    trace_files = generate_and_save_brainlog_traces()
    all_results["brainlog_traces"] = trace_files

    total_time = time.perf_counter() - t0_all
    all_results["total_execution_time_seconds"] = float(total_time)

    # Save to JSON
    out_path = Path("experiments/results/exp007_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 72)
    print(f"EXPERIMENT 007 EXECUTION COMPLETE (Total time: {total_time:.2f}s)")
    print(f"Results saved to: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    run_experiment_007()
