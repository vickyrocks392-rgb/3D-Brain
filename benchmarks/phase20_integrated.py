"""Phase 20 Track C Benchmark: Integrated System & Head-to-Head Baseline Comparison.

Evaluates the integrated interaction of Context-Gated Modular Routing (Track A),
Sparse Hippocampal-Style Replay (Track B), and Local Synaptic Consolidation:

Conditions:
- C0: MIN_E Baseline
- C1: Best Routing Only (A4)
- C2: Best Replay Only (B5)
- C3: Routing + Replay (A4 + B3)
- C4: Routing + Replay + Consolidation (A4 + B5)

External Baselines:
- Simple Recurrent Control (DiscreteLeakyRNN)
- Recurrent BPTT (BPTTGradientBaseline)
- Matched Causal Transformer (SmallCausalTransformer)

Causal Ablation Matrix:
- Full System (C4)
- Routing Removed (Replay + Consolidation)
- Replay Removed (Routing + Consolidation)
- Consolidation Removed (Routing + Replay)
- Routing + Replay Removed (Consolidation Only)
"""

from collections import defaultdict
import math
import os
import resource
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain.phase16_network import Phase16Network, ArchitecturePreset
from brain.modular_routing import RoutingCondition
from brain.hippocampal_replay import HippocampalReplayBuffer, ReplayCondition
from brain.synaptic_consolidation import ConsolidationMode
from benchmarks.phase20_routing import (
    generate_routing_trial,
    run_routing_task_phase,
    evaluate_routing_reversal,
)
from benchmarks.phase20_replay import (
    generate_replay_trial,
    train_and_eval_replay_credit,
)
from benchmarks.bptt_baseline import BPTTGradientBaseline
from benchmarks.transformer_baseline import SmallCausalTransformer
from brain.discrete_leaky_rnn import DiscreteLeakyRNN


def get_peak_rss_mb() -> float:
    """Return process peak resident set size in megabytes."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports bytes in ru_maxrss
    return float(rusage.ru_maxrss / (1024.0 * 1024.0))


def run_integrated_continual_experiment(
    condition_id: str,  # 'C0', 'C1', 'C2', 'C3', 'C4'
    trials_per_task: int = 35,
    eval_trials: int = 25,
    delay_steps: int = 5,
    seed: int = 42,
    causal_ablation: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute integrated continual learning and credit assignment protocol."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=seed)

    # Configure Routing
    if condition_id in ("C1", "C3", "C4") and causal_ablation not in ("REMOVE_ROUTING", "REMOVE_ROUTING_AND_REPLAY"):
        net.set_routing_mode(RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY, num_branches=3)
    else:
        net.set_routing_mode(RoutingCondition.A0_SHARED_LINEAR, num_branches=1)

    # Configure Consolidation
    if condition_id in ("C2", "C4") and causal_ablation not in ("REMOVE_CONSOLIDATION",):
        net.set_consolidation_mode(ConsolidationMode.DUAL_TIMESCALE)
    else:
        net.set_consolidation_mode(ConsolidationMode.NONE)

    # Configure Replay
    if condition_id in ("C2", "C3", "C4") and causal_ablation not in ("REMOVE_REPLAY", "REMOVE_ROUTING_AND_REPLAY"):
        r_cond = ReplayCondition.B5_PRIORITIZED_CONSOLIDATION if condition_id == "C4" else ReplayCondition.B3_PRIORITIZED_REPLAY
        buffer = HippocampalReplayBuffer(capacity=64, replay_budget=4, seed=seed)
        net.set_replay_buffer(buffer, condition=r_cond)
    else:
        r_cond = ReplayCondition.B0_NO_REPLAY
        net.set_replay_buffer(None, condition=r_cond)

    start_time = time.perf_counter()

    # 1. Train on Task A
    for tr in range(trials_per_task):
        ctx_id = tr % 2
        trial = generate_routing_trial(tr, "A", ctx_id, delay_steps=delay_steps, seed=seed * 100 + tr)
        net.reset()
        for st in trial.steps:
            if st.is_choice_step:
                probs, chosen = net.step(st.sensory_input, context_id=0, is_choice_step=True)
                if net.replay_buffer is not None:
                    net.replay_buffer.store_choice(
                        episode_id=tr,
                        task_id="A",
                        context_vector=net.action_readout.last_input.copy(),
                        chosen_action=chosen,
                        delay_steps=delay_steps,
                        delay_ms=float(delay_steps * 40.0),
                        eligibility_snapshot=net.action_readout.dendritic_eligibility.copy(),
                        timestamp_ms=float(st.step_idx * 40.0),
                    )
            elif st.is_reward_step:
                rew = 1.0 if (chosen == st.target_action) else -0.5
                net.step(st.sensory_input, target_action=st.target_action, context_id=0, reward=rew, is_reward_step=True)
                if net.replay_buffer is not None:
                    net.replay_buffer.record_outcome(
                        target_action=st.target_action,
                        reward_val=rew,
                        prediction_error=abs(1.0 - probs[st.target_action]),
                    )
            else:
                net.step(st.sensory_input, context_id=0, reward=0.0)

        # Quiescent rest step with replay
        net.step_quiescent_rest(rest_steps=10)

    # 2. Evaluate Task A (Pre)
    eval_a_pre = run_routing_task_phase(net, "A", eval_trials, is_training=False, seed_offset=seed * 200, delay_steps=delay_steps)
    acc_a_pre = eval_a_pre["accuracy"]
    net.set_frozen(False)

    # 3. Train on Task B (Conflicting mapping)
    for tr in range(trials_per_task):
        ctx_id = tr % 2
        trial = generate_routing_trial(tr, "B", ctx_id, delay_steps=delay_steps, seed=seed * 300 + tr)
        net.reset()
        for st in trial.steps:
            if st.is_choice_step:
                probs, chosen = net.step(st.sensory_input, context_id=1, is_choice_step=True)
                if net.replay_buffer is not None:
                    net.replay_buffer.store_choice(
                        episode_id=trials_per_task + tr,
                        task_id="B",
                        context_vector=net.action_readout.last_input.copy(),
                        chosen_action=chosen,
                        delay_steps=delay_steps,
                        delay_ms=float(delay_steps * 40.0),
                        eligibility_snapshot=net.action_readout.dendritic_eligibility.copy(),
                        timestamp_ms=float(st.step_idx * 40.0),
                    )
            elif st.is_reward_step:
                rew = 1.0 if (chosen == st.target_action) else -0.5
                net.step(st.sensory_input, target_action=st.target_action, context_id=1, reward=rew, is_reward_step=True)
                if net.replay_buffer is not None:
                    net.replay_buffer.record_outcome(
                        target_action=st.target_action,
                        reward_val=rew,
                        prediction_error=abs(1.0 - probs[st.target_action]),
                    )
            else:
                net.step(st.sensory_input, context_id=1, reward=0.0)

        net.step_quiescent_rest(rest_steps=10)

    # 4. Evaluate Task B
    eval_b = run_routing_task_phase(net, "B", eval_trials, is_training=False, seed_offset=seed * 400, delay_steps=delay_steps)
    acc_b = eval_b["accuracy"]
    net.set_frozen(False)

    # 5. Retest Task A (Post) - Catastrophic Forgetting Probe
    eval_a_post = run_routing_task_phase(net, "A", eval_trials, is_training=False, seed_offset=seed * 500, delay_steps=delay_steps)
    acc_a_post = eval_a_post["accuracy"]

    elapsed = time.perf_counter() - start_time
    retention_ratio = float(acc_a_post / max(1e-4, acc_a_pre))
    interference = float(max(0.0, acc_a_pre - acc_a_post))

    replay_footprint = net.replay_buffer.calculate_memory_footprint() if net.replay_buffer is not None else {
        "total_buffer_bytes": 0,
        "total_replay_events": 0,
        "total_replay_flops": 0,
    }

    param_info = net.action_readout.modular_readout.count_parameters() if net.action_readout.modular_readout is not None else {
        "total_trainable": net.action_readout.weights.size + net.action_readout.biases.size,
        "dynamic_state_vars": net.action_readout.eligibility.size * 2,
        "total_memory_bytes": (net.action_readout.weights.size + net.action_readout.biases.size + net.action_readout.eligibility.size * 2) * 4,
    }

    return {
        "condition": condition_id,
        "causal_ablation": causal_ablation or "NONE",
        "acc_a_pre": acc_a_pre,
        "acc_b": acc_b,
        "acc_a_post": acc_a_post,
        "retention_ratio": round(retention_ratio, 4),
        "interference_index": round(interference, 4),
        "stability_plasticity_product": round(float(acc_a_post * acc_b), 4),
        "branch_selectivity": eval_a_post["mean_selectivity"],
        "elapsed_sec": round(elapsed, 3),
        "replay_events": replay_footprint["total_replay_events"],
        "replay_flops": replay_footprint["total_replay_flops"],
        "peak_rss_mb": round(get_peak_rss_mb(), 2),
        "parameters": param_info,
    }
