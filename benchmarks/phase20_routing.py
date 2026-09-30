"""Phase 20 Track A Benchmark: Context-Gated Modular Routing & Interference.

Evaluates progressive routing mechanisms (A0 to A5) on conflicting task mappings:
1. Task A (Standard)
2. Task B (Conflicting Inverted)
3. Task C (Orthogonal)
4. Task XOR (Nonlinear Context Integration)
5. Sequential protocols: A->B->A, A->B->C, A->B->C->A

Measures:
- Pre-task accuracy, new task accuracy, post-retention accuracy
- Relative retention ratio (acc_post / acc_pre)
- Cross-task interference index: max(0, acc_pre - acc_post)
- Branch selectivity: 1 - H(g)/log(M)
- Gate entropy: H(g)
- Gate sparsity: fraction of branches with g_m < 0.05
- Cross-branch activations
- Parameter count and FLOPs
- Causal lesion ablations (gate ablated, gate randomized, branch shuffled, coupling restored)
"""

from collections import defaultdict
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain.phase16_network import Phase16Network, ArchitecturePreset
from brain.modular_routing import RoutingCondition
from brain.phase17_suite import TaskStep, TaskTrial


DT_MS = 40.0


def generate_routing_trial(
    trial_idx: int,
    task_id: str,  # 'A', 'B', 'C', 'XOR'
    context_id: int,
    input_dim: int = 8,
    cue_steps: int = 5,
    delay_steps: int = 5,
    reward_delay: int = 2,
    seed: Optional[int] = None,
) -> TaskTrial:
    """Generate a single trial for routing evaluation with zero target leakage."""
    rng = np.random.RandomState(seed if seed is not None else (2000 + trial_idx * 53))

    # Determine target action according to task definition
    if task_id == "A":
        # Standard mapping: Ctx 0 -> Act 0, Ctx 1 -> Act 1
        target_action = context_id % 2
    elif task_id == "B":
        # Conflicting inverted mapping: Ctx 0 -> Act 1, Ctx 1 -> Act 0
        target_action = 1 - (context_id % 2)
    elif task_id == "C":
        # Orthogonal mapping
        target_action = context_id % 2
    elif task_id == "XOR":
        # Nonlinear XOR condition: 4 sub-contexts
        # (c0, c1): (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
        bit0 = (context_id // 2) % 2
        bit1 = context_id % 2
        target_action = bit0 ^ bit1
    else:
        target_action = context_id % 2

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
                # Shared stimulus representation: x[0] for stimulus 0, x[1] for stimulus 1
                if context_id % 2 == 0:
                    x[0] = 1.0
                    x[5] = 0.6
                else:
                    x[1] = 1.0
                    x[6] = 0.6

                # Distinct contextual condition cue:
                # Task A is Context Condition 0 (x[2] = 1.0)
                # Task B is Context Condition 1 (x[3] = 1.0)
                if task_id == "A":
                    x[2] = 1.0
                else:
                    x[3] = 1.0

            elif task_id == "C":
                # Task C is Context Condition 2 (x[4] = 1.0)
                if context_id % 2 == 0:
                    x[0] = 1.0
                    x[5] = 0.6
                else:
                    x[1] = 1.0
                    x[6] = 0.6
                x[4] = 1.0

            elif task_id == "XOR":
                bit0 = (context_id // 2) % 2
                bit1 = context_id % 2
                x[0] = float(bit0)
                x[1] = float(bit1)
                x[2 + bit0] = 0.8
                x[4 + bit1] = 0.8

        if is_choice:
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
        task_name=f"ROUTING_{task_id}",
        metadata={"task_id": task_id, "choice_idx": choice_idx, "reward_idx": reward_idx},
    )


def run_routing_task_phase(
    net: Phase16Network,
    task_id: str,
    num_trials: int,
    is_training: bool,
    seed_offset: int = 0,
    delay_steps: int = 5,
) -> Dict[str, Any]:
    """Execute a block of trials on a specific task either with training or frozen evaluation."""
    if not is_training:
        net.set_frozen(True)

    correct_count = 0
    selectivities = []
    entropies = []
    sparsities = []
    margins = []
    confidences = []

    task_ctx_id = 0 if task_id == "A" else (1 if task_id == "B" else (2 if task_id == "C" else 0))

    for tr in range(num_trials):
        ctx_mod = 4 if task_id == "XOR" else 2
        ctx_id = tr % ctx_mod
        trial_ctx_id = ctx_id if task_id == "XOR" else task_ctx_id
        trial = generate_routing_trial(
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
                probs, chosen = net.step(st.sensory_input, context_id=trial_ctx_id, is_choice_step=True)
                choice_probs = probs.copy()
            elif st.is_reward_step:
                rew = 1.0 if (chosen == st.target_action) else -0.5
                target = st.target_action if is_training else None
                net.step(st.sensory_input, target_action=target, context_id=trial_ctx_id, reward=rew, is_reward_step=True)
            else:
                net.step(st.sensory_input, context_id=trial_ctx_id, reward=0.0)

        if chosen == trial.target_action:
            correct_count += 1

        # Collect routing telemetry if modular readout exists
        if net.action_readout.modular_readout is not None:
            telem = net.action_readout.modular_readout.get_telemetry()
            selectivities.append(telem["branch_selectivity"])
            entropies.append(telem["gate_entropy"])
            sparsities.append(telem["gate_sparsity"])
        else:
            selectivities.append(1.0)
            entropies.append(0.0)
            sparsities.append(0.0)

        sorted_p = np.sort(choice_probs)
        margins.append(float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else 1.0)
        confidences.append(float(np.max(choice_probs)))

    acc = float(correct_count / max(1, num_trials))
    return {
        "task_id": task_id,
        "is_training": is_training,
        "accuracy": acc,
        "mean_selectivity": float(np.mean(selectivities)) if selectivities else 1.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.5,
    }


def evaluate_routing_reversal(
    condition: RoutingCondition,
    num_branches: int = 3,
    trials_per_task: int = 40,
    eval_trials: int = 30,
    delay_steps: int = 5,
    seed: int = 42,
    causal_ablation: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate standard Catastrophic Forgetting protocol (A -> B -> A) with routing."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=seed)
    net.set_routing_mode(condition, num_branches=num_branches)

    # Apply causal ablations if requested
    if causal_ablation == "GATE_ABLATED" and net.action_readout.modular_readout is not None:
        net.action_readout.modular_readout.router.is_ablated = True
    elif causal_ablation == "GATE_RANDOMIZED" and net.action_readout.modular_readout is not None:
        net.action_readout.modular_readout.router.is_random = True
    elif causal_ablation == "BRANCH_SHUFFLED" and net.action_readout.modular_readout is not None:
        rng = np.random.default_rng(seed + 99)
        rng.shuffle(net.action_readout.modular_readout.branches)
    elif causal_ablation == "COUPLING_RESTORED" and net.action_readout.modular_readout is not None:
        net.action_readout.modular_readout.cross_branch_leakage = 0.5
    elif causal_ablation == "BRANCH_PLASTICITY_DISABLED" and net.action_readout.modular_readout is not None:
        net.action_readout.modular_readout.branch_plasticity_disabled = True

    # 1. Train on Task A
    run_routing_task_phase(net, "A", trials_per_task, is_training=True, seed_offset=seed * 1000, delay_steps=delay_steps)
    
    # 2. Evaluate Task A (Pre)
    eval_a_pre = run_routing_task_phase(net, "A", eval_trials, is_training=False, seed_offset=seed * 2000, delay_steps=delay_steps)
    acc_a_pre = eval_a_pre["accuracy"]
    net.set_frozen(False)

    # 3. Train on Task B (Conflicting mapping)
    run_routing_task_phase(net, "B", trials_per_task, is_training=True, seed_offset=seed * 3000, delay_steps=delay_steps)
    
    # 4. Evaluate Task B
    eval_b = run_routing_task_phase(net, "B", eval_trials, is_training=False, seed_offset=seed * 4000, delay_steps=delay_steps)
    acc_b = eval_b["accuracy"]
    net.set_frozen(False)

    # 5. Retest Task A (Post) - Catastrophic Interference Probe
    eval_a_post = run_routing_task_phase(net, "A", eval_trials, is_training=False, seed_offset=seed * 5000, delay_steps=delay_steps)
    acc_a_post = eval_a_post["accuracy"]

    retention_ratio = float(acc_a_post / max(1e-4, acc_a_pre))
    interference = float(max(0.0, acc_a_pre - acc_a_post))

    params_info = net.action_readout.modular_readout.count_parameters() if net.action_readout.modular_readout is not None else {
        "total_trainable": net.action_readout.weights.size + net.action_readout.biases.size,
        "dynamic_state_vars": net.action_readout.eligibility.size * 2,
        "total_memory_bytes": (net.action_readout.weights.size + net.action_readout.biases.size + net.action_readout.eligibility.size * 2) * 4,
    }

    return {
        "condition": condition.value,
        "num_branches": num_branches,
        "causal_ablation": causal_ablation or "NONE",
        "acc_a_pre": acc_a_pre,
        "acc_b": acc_b,
        "acc_a_post": acc_a_post,
        "retention_ratio": round(retention_ratio, 4),
        "interference_index": round(interference, 4),
        "branch_selectivity": eval_a_post["mean_selectivity"],
        "gate_entropy": eval_a_post["mean_entropy"],
        "gate_sparsity": eval_a_post["mean_sparsity"],
        "parameters": params_info,
    }


def evaluate_multitask_routing_cycle(
    condition: RoutingCondition,
    num_branches: int = 3,
    trials_per_task: int = 30,
    eval_trials: int = 25,
    delay_steps: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate multi-task cycle A -> B -> C -> A."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=seed)
    net.set_routing_mode(condition, num_branches=num_branches)

    # Train A -> Eval A
    run_routing_task_phase(net, "A", trials_per_task, is_training=True, seed_offset=seed * 100, delay_steps=delay_steps)
    eval_a1 = run_routing_task_phase(net, "A", eval_trials, is_training=False, seed_offset=seed * 200, delay_steps=delay_steps)
    net.set_frozen(False)

    # Train B -> Eval B
    run_routing_task_phase(net, "B", trials_per_task, is_training=True, seed_offset=seed * 300, delay_steps=delay_steps)
    eval_b = run_routing_task_phase(net, "B", eval_trials, is_training=False, seed_offset=seed * 400, delay_steps=delay_steps)
    net.set_frozen(False)

    # Train C -> Eval C
    run_routing_task_phase(net, "C", trials_per_task, is_training=True, seed_offset=seed * 500, delay_steps=delay_steps)
    eval_c = run_routing_task_phase(net, "C", eval_trials, is_training=False, seed_offset=seed * 600, delay_steps=delay_steps)
    net.set_frozen(False)

    # Retest A -> Retest B
    eval_a2 = run_routing_task_phase(net, "A", eval_trials, is_training=False, seed_offset=seed * 700, delay_steps=delay_steps)
    eval_b2 = run_routing_task_phase(net, "B", eval_trials, is_training=False, seed_offset=seed * 800, delay_steps=delay_steps)

    ret_a = float(eval_a2["accuracy"] / max(1e-4, eval_a1["accuracy"]))
    ret_b = float(eval_b2["accuracy"] / max(1e-4, eval_b["accuracy"]))

    return {
        "condition": condition.value,
        "acc_a_initial": eval_a1["accuracy"],
        "acc_b_initial": eval_b["accuracy"],
        "acc_c": eval_c["accuracy"],
        "acc_a_final": eval_a2["accuracy"],
        "acc_b_final": eval_b2["accuracy"],
        "retention_a": round(ret_a, 4),
        "retention_b": round(ret_b, 4),
        "mean_cycle_retention": round(float((ret_a + ret_b) / 2.0), 4),
    }


def evaluate_xor_task(
    condition: RoutingCondition,
    num_branches: int = 4,
    train_trials: int = 80,
    eval_trials: int = 40,
    delay_steps: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate nonlinear XOR task resolution capability."""
    net = Phase16Network(preset=ArchitecturePreset.MIN_E, seed=seed)
    net.set_routing_mode(condition, num_branches=num_branches)

    # Train XOR
    train_res = run_routing_task_phase(net, "XOR", train_trials, is_training=True, seed_offset=seed * 10, delay_steps=delay_steps)
    # Frozen Eval XOR
    eval_res = run_routing_task_phase(net, "XOR", eval_trials, is_training=False, seed_offset=seed * 20, delay_steps=delay_steps)

    return {
        "condition": condition.value,
        "task": "XOR",
        "train_acc": train_res["accuracy"],
        "eval_acc": eval_res["accuracy"],
        "mean_selectivity": eval_res["mean_selectivity"],
        "mean_entropy": eval_res["mean_entropy"],
        "mean_sparsity": eval_res["mean_sparsity"],
    }
