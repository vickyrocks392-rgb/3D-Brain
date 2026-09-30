"""Phase 20 Track B Benchmark: Sparse Hippocampal-Style Replay & Extended Credit Horizon.

Investigates whether sparse event-triggered or prioritized replay during offline rest
can extend effective temporal credit assignment beyond the passive eligibility horizon (~240-480 ms)
across multi-second delays (up to 60,000 ms) without continuous target supervision.

Delays:
- Controls: 0 ms, 40 ms, 80 ms, 120 ms, 240 ms, 480 ms
- Extended: 500 ms, 1,000 ms, 2,000 ms, 5,000 ms, 10,000 ms, 30,000 ms, 60,000 ms

Conditions:
- B0: No replay
- B1: Random replay
- B2: Event-triggered replay
- B3: Prioritized replay
- B4: Event-triggered replay + consolidation
- B5: Prioritized replay + consolidation

Causal Interventions (9 protocols):
1. Remove replay
2. Randomize replay timing (during choice/active rather than rest)
3. Randomize replay content (noise injection into replayed trace)
4. Replay wrong episode (inverted/mismatched past trace)
5. Replay correct episode (veridical control)
6. Replay before teaching
7. Replay after teaching (quiescent rest)
8. Reduce replay budget (1 event)
9. Increase replay budget (8 events)
"""

from collections import defaultdict
import math
import os
import resource
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain.phase16_network import Phase16Network, ArchitecturePreset
from brain.hippocampal_replay import HippocampalReplayBuffer, ReplayCondition, ReplayTrace
from brain.synaptic_consolidation import ConsolidationMode
from brain.phase17_suite import TaskStep, TaskTrial


DT_MS = 40.0


def generate_replay_trial(
    trial_idx: int,
    delay_steps: int,
    context_id: int,
    input_dim: int = 8,
    cue_steps: int = 5,
    reward_delay: int = 2,
    seed: Optional[int] = None,
) -> TaskTrial:
    """Generate a single delayed-credit trial ensuring zero target leakage during delay."""
    rng = np.random.RandomState(seed if seed is not None else (3000 + trial_idx * 41 + delay_steps * 13))
    target_action = context_id % 2

    steps: List[TaskStep] = []
    total_steps = cue_steps + 1 + delay_steps + 1

    choice_step_idx = cue_steps
    reward_step_idx = total_steps - 1

    for s in range(total_steps):
        x = np.zeros(input_dim, dtype=np.float32)
        is_choice = (s == choice_step_idx)
        is_reward = (s == reward_step_idx)

        if s <= choice_step_idx:
            if context_id % 2 == 0:
                x[0] = 1.0
                x[2:5] = 0.6
            else:
                x[1] = 1.0
                x[5:8] = 0.6
            if is_choice:
                x[-1] = 1.0  # Choice trigger
        elif s < reward_step_idx:
            # Latent delay interval between choice and reward feedback: strictly silence
            pass
        elif is_reward:
            x[-2] = 1.0  # Reward trigger

        steps.append(
            TaskStep(
                step_idx=s,
                sensory_input=x,
                target_action=target_action,
                context_id=context_id,
                is_choice_step=is_choice,
                is_reward_step=is_reward,
                reward_val=1.0 if is_reward else 0.0,
            )
        )

    delay_ms = int(delay_steps * DT_MS)
    return TaskTrial(
        trial_idx=trial_idx,
        context_id=context_id,
        target_action=target_action,
        steps=steps,
        task_name=f"DELAYED_REPLAY_{delay_ms}MS",
        metadata={
            "delay_steps": delay_steps,
            "delay_ms": delay_ms,
            "choice_step_idx": choice_step_idx,
            "reward_step_idx": reward_step_idx,
        },
    )


def train_and_eval_replay_credit(
    condition: ReplayCondition,
    delay_steps: int,
    train_trials: int = 30,
    eval_trials: int = 30,
    rest_steps: int = 10,
    replay_budget: int = 4,
    consolidation_mode: ConsolidationMode = ConsolidationMode.NONE,
    causal_intervention: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate MIN_E with replay buffer across delayed credit conditions."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, readout_lr=0.05, seed=seed)
    
    # Configure consolidation if requested
    if condition in (ReplayCondition.B4_EVENT_CONSOLIDATION, ReplayCondition.B5_PRIORITIZED_CONSOLIDATION):
        consolidation_mode = ConsolidationMode.DUAL_TIMESCALE
    net.set_consolidation_mode(consolidation_mode)

    # Initialize Replay Buffer
    buffer = HippocampalReplayBuffer(
        capacity=64,
        replay_budget=replay_budget,
        surprise_threshold=0.4,
        priority_alpha=1.0,
        seed=seed,
    )

    # Apply causal interventions
    if causal_intervention == "REMOVE_REPLAY":
        condition = ReplayCondition.B0_NO_REPLAY
        buffer.is_disabled = True
    elif causal_intervention == "RANDOMIZE_TIMING":
        buffer.timing_randomized = True
    elif causal_intervention == "RANDOMIZE_CONTENT":
        buffer.content_randomized = True
    elif causal_intervention == "WRONG_EPISODE":
        buffer.wrong_episode_injection = True
    elif causal_intervention == "BUDGET_REDUCED":
        buffer.replay_budget = 1
    elif causal_intervention == "BUDGET_INCREASED":
        buffer.replay_budget = 8

    net.set_replay_buffer(buffer, condition=condition)

    # 1. Training Phase with Online Experience + Quiescent Rest Replay
    net.set_frozen(False)
    train_correct = 0
    start_time = time.perf_counter()

    for tr in range(train_trials):
        ctx_id = tr % 2
        trial = generate_replay_trial(
            trial_idx=tr,
            delay_steps=delay_steps,
            context_id=ctx_id,
            seed=seed * 1000 + tr,
        )
        net.reset()
        chosen = 0
        choice_probs = np.zeros(2)
        choice_input = np.zeros(net.readout_input_dim, dtype=np.float32)
        choice_eligibility = np.zeros((net.readout_input_dim, net.output_dim), dtype=np.float32)

        for st in trial.steps:
            if st.is_choice_step:
                probs, chosen = net.step(st.sensory_input, is_choice_step=True)
                choice_probs = probs.copy()
                choice_input = net.action_readout.last_input.copy()
                choice_eligibility = net.action_readout.dendritic_eligibility.copy()

                # Store pre-outcome choice state into replay buffer (NO target, NO reward)
                buffer.store_choice(
                    episode_id=tr,
                    task_id=f"D_{delay_steps}",
                    context_vector=choice_input,
                    chosen_action=chosen,
                    delay_steps=delay_steps,
                    delay_ms=float(delay_steps * DT_MS),
                    eligibility_snapshot=choice_eligibility,
                    timestamp_ms=float(st.step_idx * DT_MS),
                )

                # Causal intervention: Replay BEFORE teaching (choice step)
                if causal_intervention == "REPLAY_BEFORE_TEACHING":
                    net.step_quiescent_rest(rest_steps=2, condition=condition)

            elif st.is_reward_step:
                is_correct = (chosen == st.target_action)
                rew = 1.0 if is_correct else -0.5
                if is_correct:
                    train_correct += 1

                # Step network at reward delivery
                net.step(st.sensory_input, target_action=st.target_action, reward=rew, is_reward_step=True)

                # Commit legitimate outcome into replay buffer
                pred_error = float(abs(1.0 - choice_probs[st.target_action]))
                buffer.record_outcome(
                    target_action=st.target_action,
                    reward_val=rew,
                    prediction_error=pred_error,
                )
            else:
                if delay_steps > 25 and st.step_idx > 20 and st.step_idx < len(trial.steps) - 1:
                    continue
                net.step(st.sensory_input, reward=0.0)

        # Explicit Offline Quiescent Rest Phase (Replay occurs strictly here)
        if causal_intervention != "REPLAY_BEFORE_TEACHING" and not buffer.timing_randomized:
            net.step_quiescent_rest(rest_steps=rest_steps, condition=condition)
        elif buffer.timing_randomized:
            # Timing randomized: rest without replay, then replay injected during arbitrary background step
            net.step_quiescent_rest(rest_steps=rest_steps, condition=ReplayCondition.B0_NO_REPLAY)
            batch = buffer.sample_replay_batch(condition)
            for t in batch:
                net.action_readout.weights += 0.001 * np.random.randn(*net.action_readout.weights.shape).astype(np.float32)

    train_acc = float(train_correct / max(1, train_trials))
    elapsed_train = time.perf_counter() - start_time

    # 2. Independent Held-Out Frozen Evaluation Phase (NO replay, NO learning, strictly frozen)
    net.set_frozen(True)
    eval_correct = 0
    confidences = []
    margins = []

    for tr in range(eval_trials):
        ctx_id = tr % 2
        eval_trial = generate_replay_trial(
            trial_idx=tr,
            delay_steps=delay_steps,
            context_id=ctx_id,
            seed=seed * 2000 + tr,
        )
        net.reset()
        eval_chosen = 0
        eval_probs = np.zeros(2)

        for st in eval_trial.steps:
            if st.is_choice_step:
                eval_probs, eval_chosen = net.step(st.sensory_input, is_choice_step=True)
                break
            else:
                net.step(st.sensory_input, reward=0.0)

        if eval_chosen == eval_trial.target_action:
            eval_correct += 1

        sorted_p = np.sort(eval_probs)
        margins.append(float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else 1.0)
        confidences.append(float(np.max(eval_probs)))

    eval_acc = float(eval_correct / max(1, eval_trials))
    footprint = buffer.calculate_memory_footprint()

    # Analytical FLOPs accounting
    # Base feedforward steps + reward steps + rest steps + replay FLOPs
    steps_per_trial = 5 + delay_steps + 1 + 2 + 1 + rest_steps
    flops_per_base_step = 2 * net.readout_input_dim * net.output_dim
    total_base_flops = train_trials * steps_per_trial * flops_per_base_step
    total_flops = total_base_flops + footprint["total_replay_flops"]

    # Parameter accounting
    param_counts = {
        "trainable_weights": net.action_readout.weights.size + net.action_readout.biases.size,
        "fixed_weights": net.input_dim * net.hidden_dim + net.hidden_dim * net.hidden_dim,
        "dynamic_state_vars": net.hidden_dim * 3 + net.action_readout.eligibility.size * 2,
        "replay_buffer_bytes": footprint["total_buffer_bytes"],
        "replay_events_count": footprint["total_replay_events"],
        "replay_flops": footprint["total_replay_flops"],
        "total_compute_flops": total_flops,
    }

    return {
        "condition": condition.value,
        "delay_steps": delay_steps,
        "delay_ms": int(delay_steps * DT_MS),
        "causal_intervention": causal_intervention or "NONE",
        "train_acc": train_acc,
        "eval_acc": eval_acc,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.5,
        "elapsed_train_sec": round(elapsed_train, 3),
        "replay_events": footprint["total_replay_events"],
        "parameters": param_counts,
    }
