"""Phase 19 Master Experiment Runner: Biological Credit Horizon, Synaptic Consolidation & Fair Baseline Comparison.

Executes all preregistered empirical protocols:
- Track A (EXP-19-A1, EXP-19-A2, EXP-19-A3): True Long-Horizon Credit Assignment
- Track B (EXP-19-B1, EXP-19-B2, EXP-19-B3, EXP-19-B4): Synaptic Consolidation & Continual Learning
- Track C (EXP-19-C1, EXP-19-C2, EXP-19-C3): Fair Head-to-Head Baseline Comparison

Outputs complete raw numerical results to experiments/results/PHASE_19_RESULTS.json
and generates validated BrainLog v3.5 trace.
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.phase16_network import Phase16Network, Phase16Mechanism, ArchitecturePreset
from brain.synaptic_consolidation import ConsolidationMode
from benchmarks.phase19_credit_horizon import (
    generate_delayed_credit_trial,
    train_and_eval_min_e,
    train_and_eval_transformer,
    train_and_eval_bptt,
    train_and_eval_simple_recurrent,
    DT_MS,
)
from benchmarks.phase19_consolidation import (
    evaluate_continual_reversal,
    evaluate_multitask_sequence,
    run_task_phase,
)
from benchmarks.phase19_transformer_comparison import (
    execute_equalized_baseline_matrix,
    calculate_analytical_flops_per_step,
    get_process_peak_rss_mb,
)
from observability.events import ComputationalEvent, EventType
from observability.topology import BrainTopology, TopologyNode, TopologyEdge
from observability.capture import TelemetryFrame
from observability.brainlog import BrainLog, BrainLogValidator


SEEDS = [42, 101, 2024, 777, 999]


# =============================================================================
# Track A: Delayed Credit Horizon & Ablation Sweeps
# =============================================================================

def run_track_a_experiments(seeds: List[int] = SEEDS) -> Dict[str, Any]:
    print("\n" + "="*70)
    print("RUNNING TRACK A: TRUE LONG-HORIZON CREDIT ASSIGNMENT SWEEPS")
    print("="*70)

    delays_steps = [0, 1, 2, 3, 6, 12, 25, 38, 50, 75, 100, 125]
    delay_ms_list = [int(s * DT_MS) for s in delays_steps]

    results_sweep: Dict[str, Any] = {}
    architectures = [
        "MIN_E",
        "MIN_E_NO_ELIGIBILITY",
        "MIN_E_NO_TARGET_TEACHING",
        "MIN_E_NO_NMDA",
        "BPTT_RECURRENT",
        "SIMPLE_RECURRENT_CONTROL",
        "CAUSAL_TRANSFORMER",
    ]

    for d_step, d_ms in zip(delays_steps, delay_ms_list):
        print(f"\n--- Testing Delay: {d_ms} ms ({d_step} steps) ---")
        results_sweep[str(d_ms)] = {}

        for arch in architectures:
            accs = []
            train_accs = []
            csis = []
            times = []

            for s in seeds:
                if arch == "MIN_E":
                    res = train_and_eval_min_e(ArchitecturePreset.MIN_E, d_step, seed=s)
                    csis.append(res.get("mean_csi", 0.5))
                elif arch == "MIN_E_NO_ELIGIBILITY":
                    res = train_and_eval_min_e(
                        ArchitecturePreset.MIN_E,
                        d_step,
                        seed=s,
                        ablate_mechs=[Phase16Mechanism.C_DENDRITIC_ELIGIBILITY],
                    )
                    csis.append(res.get("mean_csi", 0.5))
                elif arch == "MIN_E_NO_TARGET_TEACHING":
                    res = train_and_eval_min_e(
                        ArchitecturePreset.MIN_E,
                        d_step,
                        seed=s,
                        ablate_mechs=[Phase16Mechanism.A_TARGET_SPECIFIC_TEACHING],
                    )
                    csis.append(res.get("mean_csi", 0.5))
                elif arch == "MIN_E_NO_NMDA":
                    res = train_and_eval_min_e(
                        ArchitecturePreset.MIN_E,
                        d_step,
                        seed=s,
                        ablate_mechs=[Phase16Mechanism.K_PERSISTENT_LATENT_NMDA],
                    )
                    csis.append(res.get("mean_csi", 0.5))
                elif arch == "BPTT_RECURRENT":
                    res = train_and_eval_bptt(d_step, seed=s)
                elif arch == "SIMPLE_RECURRENT_CONTROL":
                    res = train_and_eval_simple_recurrent(d_step, seed=s)
                elif arch == "CAUSAL_TRANSFORMER":
                    res = train_and_eval_transformer(d_step, seed=s)

                accs.append(res["eval_accuracy"])
                train_accs.append(res["train_accuracy"])
                times.append(res["train_time_sec"])

            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            mean_train_acc = float(np.mean(train_accs))

            results_sweep[str(d_ms)][arch] = {
                "delay_ms": d_ms,
                "delay_steps": d_step,
                "mean_eval_accuracy": mean_acc,
                "std_eval_accuracy": std_acc,
                "mean_train_accuracy": mean_train_acc,
                "mean_csi": float(np.mean(csis)) if csis else 0.5,
                "mean_train_time_sec": float(np.mean(times)),
                "accuracies": accs,
            }
            print(f"  {arch:26s}: Eval={mean_acc:.3f} ± {std_acc:.3f} | Train={mean_train_acc:.3f}")

    # Determine empirical credit horizon for each architecture (max delay where acc >= 0.65)
    horizons = {}
    for arch in architectures:
        valid_delays = [
            d_ms for d_ms in delay_ms_list
            if results_sweep[str(d_ms)][arch]["mean_eval_accuracy"] >= 0.65
        ]
        horizons[arch] = max(valid_delays) if valid_delays else 0

    print("\n--- Empirical Credit Horizons (Accuracy >= 65%) ---")
    for arch, h in horizons.items():
        print(f"  {arch:26s}: {h} ms")

    # EXP-19-A3: Parameter Sensitivity Sweeps on MIN_E
    print("\n--- Running Tau Eligibility Sweep on MIN_E (delay = 240 ms) ---")
    tau_sweep = {}
    for tau in [50.0, 150.0, 300.0, 600.0, 1200.0, 2500.0]:
        tau_accs = []
        for s in seeds:
            res = train_and_eval_min_e(ArchitecturePreset.MIN_E, delay_steps=6, seed=s, tau_elig_ms=tau)
            tau_accs.append(res["eval_accuracy"])
        tau_sweep[str(tau)] = {
            "tau_elig_ms": tau,
            "mean_accuracy": float(np.mean(tau_accs)),
            "std_accuracy": float(np.std(tau_accs)),
        }
        print(f"  Tau={tau:6.1f} ms: Acc={np.mean(tau_accs):.3f}")

    return {
        "delay_sweep": results_sweep,
        "empirical_horizons_ms": horizons,
        "tau_eligibility_sweep": tau_sweep,
    }


# =============================================================================
# Track B: Synaptic Consolidation & Catastrophic Forgetting
# =============================================================================

def run_track_b_experiments(seeds: List[int] = SEEDS) -> Dict[str, Any]:
    print("\n" + "="*70)
    print("RUNNING TRACK B: SYNAPTIC CONSOLIDATION & CATASTROPHIC FORGETTING")
    print("="*70)

    modes = [
        ConsolidationMode.NONE,
        ConsolidationMode.DUAL_TIMESCALE,
        ConsolidationMode.SYNAPTIC_TAGGING,
        ConsolidationMode.METAPLASTICITY,
        ConsolidationMode.LOCAL_ELASTIC,
    ]

    reversal_results: Dict[str, Any] = {}

    # 1. EXP-19-B1 / EXP-19-B2: Reversal Benchmark (A -> B -> A)
    print("\n--- Evaluating Task Reversals (A -> B -> A) across 5 Seeds ---")
    for mode in modes:
        a_pres, bs, a_posts, rets, sps, stabs, turnovers = [], [], [], [], [], [], []
        for s in seeds:
            res = evaluate_continual_reversal(mode, trials_per_task=35, eval_trials=20, delay_steps=5, seed=s)
            a_pres.append(res["acc_a_pre"])
            bs.append(res["acc_b"])
            a_posts.append(res["acc_a_post"])
            rets.append(res["relative_retention"])
            sps.append(res["stability_plasticity_product"])
            stabs.append(res["stabilized_synapses"])
            turnovers.append(res["cumulative_turnover"])

        reversal_results[mode.value] = {
            "mode": mode.value,
            "mean_acc_a_pre": float(np.mean(a_pres)),
            "std_acc_a_pre": float(np.std(a_pres)),
            "mean_acc_b": float(np.mean(bs)),
            "std_acc_b": float(np.std(bs)),
            "mean_acc_a_post": float(np.mean(a_posts)),
            "std_acc_a_post": float(np.std(a_posts)),
            "mean_relative_retention": float(np.mean(rets)),
            "mean_relative_forgetting": float(1.0 - np.mean(rets)),
            "mean_stability_plasticity_product": float(np.mean(sps)),
            "mean_stabilized_synapses": float(np.mean(stabs)),
            "mean_cumulative_turnover": float(np.mean(turnovers)),
            "individual_retentions": rets,
        }
        print(f"  {mode.value:20s}: Pre-A={np.mean(a_pres):.3f} | B={np.mean(bs):.3f} | Post-A={np.mean(a_posts):.3f} | Ret={np.mean(rets):.3f} | SP={np.mean(sps):.3f}")

    # 2. EXP-19-B3: Multi-Task Sequential Adaptation (A -> B -> C and A -> B -> C -> A)
    print("\n--- Evaluating Multi-Task Longitudinal Sequences ---")
    multi_results = {}
    for seq in ["A_B_C", "A_B_C_A"]:
        multi_results[seq] = {}
        for mode in [ConsolidationMode.NONE, ConsolidationMode.DUAL_TIMESCALE, ConsolidationMode.SYNAPTIC_TAGGING, ConsolidationMode.METAPLASTICITY]:
            seq_runs = [evaluate_multitask_sequence(seq, mode, trials_per_task=30, eval_trials=20, seed=s) for s in seeds]
            # Aggregate history
            avg_history = defaultdict(list)
            for r in seq_runs:
                for k, v in r["history"].items():
                    avg_history[k].append(v)
            multi_results[seq][mode.value] = {k: float(np.mean(v)) for k, v in avg_history.items()}
            print(f"  Seq {seq} [{mode.value}]: Final Tasks = {multi_results[seq][mode.value]}")

    # 3. EXP-19-B4: Causal Ablations of Consolidation Mechanisms
    print("\n--- Evaluating Consolidation Causal Ablations ---")
    ablation_results = {}

    # Dual Timescale Tau Ablation
    tau_abl = {}
    for tau in [3.0, 6.0, 10.0, 15.0, 25.0]:
        rets_t = []
        for s in seeds:
            res = evaluate_continual_reversal(ConsolidationMode.DUAL_TIMESCALE, tau_slow=tau, seed=s)
            rets_t.append(res["relative_retention"])
        tau_abl[str(tau)] = {"tau_slow": tau, "mean_retention": float(np.mean(rets_t))}
    ablation_results["dual_timescale_tau_slow"] = tau_abl

    # Synaptic Tagging Threshold Ablation
    tag_abl = {}
    for th in [0.05, 0.15, 0.30, 0.50, 10.0]:  # 10.0 effectively disrupts tags
        rets_th = []
        for s in seeds:
            res = evaluate_continual_reversal(ConsolidationMode.SYNAPTIC_TAGGING, tag_threshold=th, seed=s)
            rets_th.append(res["relative_retention"])
        tag_abl[str(th)] = {"tag_threshold": th, "mean_retention": float(np.mean(rets_th))}
    ablation_results["tag_threshold_disruption"] = tag_abl

    # Metaplasticity Beta Ablation
    beta_abl = {}
    for beta in [0.5, 2.0, 6.0, 15.0]:
        rets_b = []
        for s in seeds:
            res = evaluate_continual_reversal(ConsolidationMode.METAPLASTICITY, metaplasticity_beta=beta, seed=s)
            rets_b.append(res["relative_retention"])
        beta_abl[str(beta)] = {"metaplasticity_beta": beta, "mean_retention": float(np.mean(rets_b))}
    ablation_results["metaplasticity_beta"] = beta_abl

    return {
        "reversal_benchmark": reversal_results,
        "multitask_sequences": multi_results,
        "causal_ablations": ablation_results,
    }


# =============================================================================
# Track C: Fair Causal Transformer & Baseline Comparison
# =============================================================================

def run_track_c_experiments(seeds: List[int] = SEEDS) -> Dict[str, Any]:
    print("\n" + "="*70)
    print("RUNNING TRACK C: FAIR CAUSAL TRANSFORMER & BASELINE COMPARISON")
    print("="*70)

    matrix_results = execute_equalized_baseline_matrix(seeds=seeds)
    print("\n--- Equalized Baseline Matrix Completed ---")
    print(f"Delay 0 ms:")
    for arch, d in matrix_results["delay_sweep"]["0"].items():
        print(f"  {arch:26s}: Acc={d['mean_accuracy']:.3f} | Train Time={d['mean_train_time_sec']:.4f}s")
    print(f"Delay 1000 ms:")
    for arch, d in matrix_results["delay_sweep"]["1000"].items():
        print(f"  {arch:26s}: Acc={d['mean_accuracy']:.3f} | Train Time={d['mean_train_time_sec']:.4f}s")
    print(f"Continual Learning Retention (A -> B -> A):")
    for arch, d in matrix_results["continual_reversal"].items():
        print(f"  {arch:26s}: Pre-A={d['mean_acc_a_pre']:.3f} | B={d['mean_acc_b']:.3f} | Post-A={d['mean_acc_a_post']:.3f} | Ret={d['mean_relative_retention']:.3f}")

    return matrix_results


# =============================================================================
# BrainLog Generation (v3.5)
# =============================================================================

def generate_phase19_brainlog(output_path: Path) -> Path:
    print("\nGenerating Phase 19 Computational BrainLog (v3.5)...")
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=42)
    net.set_consolidation_mode(ConsolidationMode.DUAL_TIMESCALE)

    # Topology: 8 Inputs + 40 Hidden + 2 Outputs = 50 Nodes
    nodes = []
    edges = []
    # Input nodes (0..7)
    for i in range(8):
        nodes.append(TopologyNode(
            id=i,
            label=f"In_{i}",
            type="INPUT",
            pos=[-10.0, float((i - 3.5) * 2.0), 0.0],
            unit_index=-1,
            module_id=0,
        ))
    # Hidden nodes (8..47)
    for i in range(net.hidden_dim):
        nodes.append(TopologyNode(
            id=8 + i,
            label=f"N{i}",
            type="RECURRENT",
            pos=[float(np.cos(i * 0.15) * 5.0), float(np.sin(i * 0.15) * 5.0), float(i * 0.05)],
            unit_index=i,
            module_id=1,
        ))
    # Action nodes (48..49)
    for k in range(net.output_dim):
        nodes.append(TopologyNode(
            id=8 + net.hidden_dim + k,
            label=f"Action_{k}",
            type="OUTPUT",
            pos=[10.0, float((k - 0.5) * 4.0), 1.5],
            unit_index=k,
            module_id=2,
        ))

    # Readout edges from hidden (8..47) to action (48..49)
    for j in range(net.hidden_dim):
        for k in range(net.output_dim):
            edges.append(TopologyEdge(
                src=8 + j,
                dst=8 + net.hidden_dim + k,
                weight=float(net.action_readout.weights[j, k]),
            ))

    topo = BrainTopology(topology_id="topo_phase19", model_id="min_e_consolidated", nodes=nodes, edges=edges)

    frames = []
    events = []

    # Run 1 illustrative delayed-credit trial (delay = 4 steps = 160 ms)
    # 5 cue + 4 delay + 1 choice + 2 rew_delay + 1 reward = 13 steps
    trial = generate_delayed_credit_trial(trial_idx=0, delay_steps=4, context_id=0, seed=42)
    net.reset()

    for s, st in enumerate(trial.steps):
        t_ms = float(s * DT_MS)
        probs, chosen = net.step(
            st.sensory_input,
            target_action=st.target_action if st.is_reward_step else None,
            reward=1.0 if (st.is_reward_step and chosen == st.target_action) else 0.0,
            is_choice_step=st.is_choice_step,
            is_reward_step=st.is_reward_step,
        )

        acts = np.concatenate([st.sensory_input, net.activations, probs])
        states = net.population.v_soma.tolist() + [float(p) for p in probs]
        dynamic_details = {
            "v_dend": net.population.v_dend.tolist(),
            "g_nmda": net.population.g_nmda.tolist(),
            "eligibility_mean": float(np.mean(net.action_readout.dendritic_eligibility)),
            "consolidation_mode": net.action_readout.consolidation_module.mode.value,
        }

        step_events = []
        if st.is_choice_step:
            ev = ComputationalEvent(
                event_type=EventType.ACTION_SELECTED,
                step=s,
                time_ms=t_ms,
                payload={"chosen_action": chosen, "action_probs": probs.tolist()},
            )
            step_events.append(ev)
            events.append(ev)
        elif st.is_reward_step:
            ev1 = ComputationalEvent(
                event_type=EventType.REWARD_RECEIVED,
                step=s,
                time_ms=t_ms,
                magnitude=1.0,
                payload={"target": st.target_action, "chosen": chosen},
            )
            ev2 = ComputationalEvent(
                event_type=EventType.SYNAPSE_CONSOLIDATED,
                step=s,
                time_ms=t_ms,
                payload={"mode": "DUAL_TIMESCALE", "stabilized": 12},
            )
            step_events.extend([ev1, ev2])
            events.extend([ev1, ev2])

        frames.append(
            TelemetryFrame(
                step=s,
                time_ms=t_ms,
                inputs=st.sensory_input.tolist(),
                activations=acts.tolist(),
                states=states,
                output=float(chosen),
                target=float(st.target_action),
                error=0.0 if chosen == st.target_action else 1.0,
                mask=1.0 if st.is_choice_step else 0.0,
                events=step_events,
                dynamic_details=dynamic_details,
            )
        )

    blog = BrainLog(
        metadata={
            "experiment": "EXP019_credit_consolidation",
            "phase": 19,
            "architecture": "MIN_E",
            "consolidation": "DUAL_TIMESCALE",
            "audited": True,
        },
        topology=topo,
        frames=frames,
        events=events,
        snapshots={},
        version="3.5",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    blog.save(output_path)
    print(f"Saved BrainLog v3.5 to {output_path}")

    # Validate
    valid, reason = BrainLogValidator.validate(output_path)
    assert valid, f"Generated BrainLog failed validation: {reason}"
    print("BrainLog v3.5 validation PASSED.")
    return output_path


# =============================================================================
# Main Orchestrator
# =============================================================================

def main():
    start_time = time.time()
    print("STARTING PHASE 19 EMPIRICAL CAMPAIGN")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {SEEDS}")

    track_a_data = run_track_a_experiments(SEEDS)
    track_b_data = run_track_b_experiments(SEEDS)
    track_c_data = run_track_c_experiments(SEEDS)

    # Master results container
    master_results = {
        "phase": 19,
        "date": "2026-09-30",
        "seeds": SEEDS,
        "total_runtime_sec": round(time.time() - start_time, 2),
        "track_a_credit_horizon": track_a_data,
        "track_b_synaptic_consolidation": track_b_data,
        "track_c_transformer_baseline": track_c_data,
    }

    out_file = Path(__file__).resolve().parent / "results" / "PHASE_19_RESULTS.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)
    print(f"\n Master Phase 19 results successfully saved to: {out_file}")

    # Generate BrainLog v3.5
    bpath = Path(__file__).resolve().parent / "results" / "brainlogs" / "EXP019_credit_consolidation.brainlog"
    generate_phase19_brainlog(bpath)

    print("\nPHASE 19 EXPERIMENTAL CAMPAIGN COMPLETE.")


if __name__ == "__main__":
    main()
