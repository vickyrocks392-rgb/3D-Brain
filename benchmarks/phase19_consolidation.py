"""Phase 19 Track B: Synaptic Consolidation & Catastrophic Forgetting Benchmark.

Investigates whether biologically plausible local synaptic consolidation mechanisms
(Dual-Timescale Synapses, Synaptic Tagging & Capture, Metaplasticity Cascade, and Local Elastic Stabilization)
can protect previously learned context-action memories while maintaining new learning capacity.

Measures:
- Pre-task accuracy (acc_A_pre)
- New task acquisition (acc_B)
- Retest retention (acc_A_post)
- Relative retention ratio (acc_A_post / acc_A_pre)
- Stability-Plasticity Product (acc_A_post * acc_B)
- Number of stabilized synapses
- Synaptic turnover and weight drift
- Sequential task protocols: A->B, B->A, A->B->A, A->B->C, A->B->C->A
"""

from collections import defaultdict
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain.phase16_network import Phase16Network, ArchitecturePreset
from brain.synaptic_consolidation import ConsolidationMode
from brain.phase17_suite import TaskStep, TaskTrial


DT_MS = 40.0


def generate_continual_trial(
    trial_idx: int,
    task_id: str,  # 'A', 'B', or 'C'
    context_id: int,
    delay_steps: int = 5,
    cue_steps: int = 5,
    reward_delay: int = 2,
    seed: Optional[int] = None,
) -> TaskTrial:
    """Generate a single trial for continual learning with zero target leakage."""
    rng = np.random.RandomState(seed if seed is not None else (100 + trial_idx * 43))
    
    # Task mapping definitions
    if task_id == "A":
        # Task A: standard mapping (Ctx 0 -> Act 0, Ctx 1 -> Act 1)
        target_action = context_id % 2
    elif task_id == "B":
        # Task B: exact inverted mapping (Ctx 0 -> Act 1, Ctx 1 -> Act 0)
        target_action = 1 - (context_id % 2)
    elif task_id == "C":
        # Task C: orthogonal context mapping
        target_action = context_id % 2
    else:
        target_action = context_id % 2

    input_dim = 8
    total_steps = cue_steps + delay_steps + 1 + reward_delay + 1
    choice_idx = cue_steps + delay_steps
    reward_idx = total_steps - 1

    steps: List[TaskStep] = []
    for s in range(total_steps):
        x = np.zeros(input_dim, dtype=np.float32)
        is_choice = (s == choice_idx)
        is_reward = (s == reward_idx)

        if s < cue_steps:
            if task_id in ("A", "B"):
                if context_id % 2 == 0:
                    x[0] = 1.0
                    x[2:5] = 0.6
                else:
                    x[1] = 1.0
                    x[5:8] = 0.6
            elif task_id == "C":
                # Shifted sensory features for Task C
                if context_id % 2 == 0:
                    x[2] = 1.0
                    x[4] = 0.7
                else:
                    x[3] = 1.0
                    x[5] = 0.7
        elif is_choice:
            x[-1] = 1.0  # Choice trigger
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

    return TaskTrial(
        trial_idx=trial_idx,
        context_id=context_id,
        target_action=target_action,
        steps=steps,
        task_name=f"TASK_{task_id}",
        metadata={"task_id": task_id, "choice_idx": choice_idx, "reward_idx": reward_idx},
    )


def run_task_phase(
    net: Phase16Network,
    task_id: str,
    num_trials: int,
    is_training: bool,
    seed_offset: int = 0,
    delay_steps: int = 5,
) -> Dict[str, Any]:
    """Execute a block of trials on a specific task either with learning or frozen evaluation."""
    if not is_training:
        net.set_frozen(True)

    correct_count = 0
    margins = []
    confidences = []

    for tr in range(num_trials):
        ctx_id = tr % 2
        trial = generate_continual_trial(
            trial_idx=tr,
            task_id=task_id,
            context_id=ctx_id,
            delay_steps=delay_steps,
            seed=seed_offset + tr,
        )
        net.reset()
        chosen = 0
        choice_probs = np.zeros(2)

        for st in trial.steps:
            if st.is_choice_step:
                probs, chosen = net.step(st.sensory_input, is_choice_step=True)
                choice_probs = probs.copy()
            elif st.is_reward_step:
                rew = 1.0 if (chosen == st.target_action) else -0.5
                target = st.target_action if is_training else None
                net.step(st.sensory_input, target_action=target, reward=rew, is_reward_step=True)
            else:
                net.step(st.sensory_input, reward=0.0)

        if chosen == trial.target_action:
            correct_count += 1
        sorted_p = np.sort(choice_probs)
        margins.append(float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else 1.0)
        confidences.append(float(np.max(choice_probs)))

    acc = float(correct_count / max(1, num_trials))
    return {
        "task_id": task_id,
        "is_training": is_training,
        "accuracy": acc,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.5,
    }


def evaluate_continual_reversal(
    mode: ConsolidationMode,
    trials_per_task: int = 40,
    eval_trials: int = 30,
    delay_steps: int = 5,
    seed: int = 42,
    **consol_kwargs,
) -> Dict[str, Any]:
    """Evaluate standard Catastrophic Forgetting protocol (A -> B -> A)."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=seed)
    net.set_consolidation_mode(mode, **consol_kwargs)

    # 1. Train on Task A
    run_task_phase(net, "A", trials_per_task, is_training=True, seed_offset=seed * 1000, delay_steps=delay_steps)
    
    # 2. Evaluate Task A (Pre)
    eval_a_pre = run_task_phase(net, "A", eval_trials, is_training=False, seed_offset=seed * 2000, delay_steps=delay_steps)
    acc_a_pre = eval_a_pre["accuracy"]
    net.set_frozen(False)

    # 3. Train on Task B (Inverted condition)
    run_task_phase(net, "B", trials_per_task, is_training=True, seed_offset=seed * 3000, delay_steps=delay_steps)
    
    # 4. Evaluate Task B
    eval_b = run_task_phase(net, "B", eval_trials, is_training=False, seed_offset=seed * 4000, delay_steps=delay_steps)
    acc_b = eval_b["accuracy"]
    net.set_frozen(False)

    # 5. Retest Task A (Post) - Catastrophic Forgetting Probe
    eval_a_post = run_task_phase(net, "A", eval_trials, is_training=False, seed_offset=seed * 5000, delay_steps=delay_steps)
    acc_a_post = eval_a_post["accuracy"]

    # 6. Re-exposure / Recovery on Task A (10 trials)
    net.set_frozen(False)
    run_task_phase(net, "A", 10, is_training=True, seed_offset=seed * 6000, delay_steps=delay_steps)
    eval_a_rec = run_task_phase(net, "A", eval_trials, is_training=False, seed_offset=seed * 7000, delay_steps=delay_steps)
    acc_a_recovery = eval_a_rec["accuracy"]

    # Metrics
    rel_retention = float(acc_a_post / max(0.01, acc_a_pre))
    rel_forgetting = float(max(0.0, 1.0 - rel_retention))
    stability_plasticity_product = float(acc_a_post * acc_b)
    backward_transfer = float(acc_a_post - acc_a_pre)

    consol_state = net.action_readout.consolidation_module.observable_state()

    return {
        "mode": mode.value,
        "seed": seed,
        "acc_a_pre": acc_a_pre,
        "acc_b": acc_b,
        "acc_a_post": acc_a_post,
        "acc_a_recovery": acc_a_recovery,
        "relative_retention": round(rel_retention, 4),
        "relative_forgetting": round(rel_forgetting, 4),
        "backward_transfer": round(backward_transfer, 4),
        "stability_plasticity_product": round(stability_plasticity_product, 4),
        "stabilized_synapses": consol_state["stabilized_synapses"],
        "mean_stability": consol_state["mean_stability"],
        "cumulative_turnover": consol_state["cumulative_turnover"],
    }


def evaluate_multitask_sequence(
    sequence: str,  # 'A_B_C' or 'A_B_C_A'
    mode: ConsolidationMode,
    trials_per_task: int = 35,
    eval_trials: int = 25,
    delay_steps: int = 5,
    seed: int = 42,
    **consol_kwargs,
) -> Dict[str, Any]:
    """Evaluate longitudinal multi-task sequences: A->B->C or A->B->C->A."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=seed)
    net.set_consolidation_mode(mode, **consol_kwargs)

    stages = sequence.split("_")
    history: Dict[str, List[float]] = defaultdict(list)

    for stage_idx, t_name in enumerate(stages):
        # Train on current task
        net.set_frozen(False)
        run_task_phase(net, t_name, trials_per_task, is_training=True, seed_offset=seed * 10000 + stage_idx * 1000, delay_steps=delay_steps)

        # Evaluate all seen tasks so far
        seen_tasks = set(stages[:stage_idx + 1])
        for t_eval in seen_tasks:
            ev = run_task_phase(net, t_eval, eval_trials, is_training=False, seed_offset=seed * 20000 + stage_idx * 500, delay_steps=delay_steps)
            history[f"{t_eval}_after_stage_{stage_idx}_{t_name}"].append(ev["accuracy"])

    return {
        "sequence": sequence,
        "mode": mode.value,
        "seed": seed,
        "history": {k: float(np.mean(v)) for k, v in history.items()},
        "final_consolidation_state": net.action_readout.consolidation_module.observable_state(),
    }
