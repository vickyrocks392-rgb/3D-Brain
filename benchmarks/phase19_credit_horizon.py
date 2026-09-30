"""Phase 19 Track A: True Long-Horizon Credit Assignment Benchmark.

Investigates the actual temporal credit-assignment horizon of local dendritic eligibility +
target-specific teaching when all continuous supervision, target leakage, and test-time learning
are completely removed.

Causal Trial Structure:
    INPUT / CUE (t_cue=5 steps)
         │
    LATENT RETENTION (t_delay steps, SILENCE, NO target, NO cue)
         │
    ACTION OPPORTUNITY (t_choice=1 step, model selects action, NO target)
         │
    REWARD DELAY (t_rew_delay=2 steps, SILENCE, NO feedback)
         │
    SINGLE SPARSE OUTCOME (t_reward=1 step, reward R in {+1.0, -0.5} + local teaching signal)

Evaluates:
1. MIN_E (Audited Minimal Architecture)
2. MIN_E_NO_ELIGIBILITY (Ablate Mechanism C)
3. MIN_E_NO_TARGET_TEACHING (Ablate Mechanism A -> Global Scalar Reward G0)
4. MIN_E_NO_NMDA (Ablate Mechanism K)
5. BPTT_RECURRENT (DiscreteLeakyRNN trained via BPTT)
6. SIMPLE_RECURRENT (DiscreteLeakyRNN with reward-modulated Hebbian control)
7. CAUSAL_TRANSFORMER (SmallCausalTransformer, matched budget)
"""

from collections import defaultdict
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from brain.phase16_network import Phase16Network, Phase16Mechanism, ArchitecturePreset
from brain.phase17_suite import TaskStep, TaskTrial
from benchmarks.bptt_baseline import BPTTGradientBaseline
from benchmarks.transformer_baseline import SmallCausalTransformer
from brain.discrete_leaky_rnn import DiscreteLeakyRNN


# Standard step duration in ms
DT_MS = 40.0


def generate_delayed_credit_trial(
    trial_idx: int,
    delay_steps: int,
    context_id: int,
    input_dim: int = 8,
    cue_steps: int = 5,
    reward_delay: int = 2,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> TaskTrial:
    """Generate a single delayed-credit trial ensuring zero target leakage during delay."""
    rng = np.random.RandomState(seed if seed is not None else (42 + trial_idx * 31 + delay_steps * 7))
    target_action = context_id % 2

    steps: List[TaskStep] = []
    total_steps = cue_steps + delay_steps + 1 + reward_delay + 1

    choice_step_idx = cue_steps + delay_steps
    reward_step_idx = total_steps - 1

    for s in range(total_steps):
        x = np.zeros(input_dim, dtype=np.float32)
        is_choice = (s == choice_step_idx)
        is_reward = (s == reward_step_idx)

        if s < cue_steps:
            # Cue presentation
            if context_id % 2 == 0:
                x[0] = 1.0
                x[2:5] = 0.6
            else:
                x[1] = 1.0
                x[5:8] = 0.6
        elif s < choice_step_idx:
            # Delay interval: strictly silence, NO cues, NO target
            pass
        elif is_choice:
            # Action opportunity trigger probe
            x[-1] = 1.0
        elif is_reward:
            # Reward step trigger probe
            x[-2] = 1.0

        if noise_std > 0.0:
            x += rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)

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
        task_name=f"DELAYED_CREDIT_{delay_ms}MS",
        metadata={
            "delay_steps": delay_steps,
            "delay_ms": delay_ms,
            "choice_step_idx": choice_step_idx,
            "reward_step_idx": reward_step_idx,
        },
    )


# =============================================================================
# Simple Recurrent Control Baseline
# =============================================================================

class SimpleRecurrentControl:
    """Matched-parameter recurrent control with reward-modulated Hebbian readout."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 40, output_dim: int = 2, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.rnn = DiscreteLeakyRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, seed=seed)
        scale = 1.0 / np.sqrt(hidden_dim)
        self.W_out = self.rng.normal(0.0, scale, (hidden_dim, output_dim)).astype(np.float32)
        self.b_out = np.zeros(output_dim, dtype=np.float32)
        self.eligibility = np.zeros((hidden_dim, output_dim), dtype=np.float32)
        self.decay_elig = float(np.exp(-DT_MS / 300.0))
        self.lr = 0.02
        self.is_frozen = False
        self.last_choice = 0
        self.last_probs = np.zeros(output_dim, dtype=np.float32)

    def reset(self) -> None:
        self.rnn.reset()
        self.eligibility.fill(0.0)

    def set_frozen(self, frozen: bool) -> None:
        self.is_frozen = frozen

    def step(
        self,
        x: np.ndarray,
        target_action: Optional[int] = None,
        reward: float = 0.0,
        is_choice_step: bool = False,
        is_reward_step: bool = False,
    ) -> Tuple[np.ndarray, int]:
        h_step = self.rnn.step(x)
        h = self.rnn.h[0]

        logits = np.dot(h, self.W_out) + self.b_out
        shifted = logits - np.max(logits)
        exp_l = np.exp(np.clip(shifted, -20.0, 20.0))
        probs = exp_l / (np.sum(exp_l) + 1e-12)
        selected = int(np.argmax(probs))

        if is_choice_step:
            self.last_choice = selected
            self.last_probs = probs.copy()

        # Update eligibility trace
        self.eligibility = self.decay_elig * self.eligibility + (1.0 - self.decay_elig) * np.outer(h, probs)

        # Plasticity step at reward delivery
        if is_reward_step and not self.is_frozen:
            if target_action is not None:
                teaching = np.full(self.output_dim, -1.0, dtype=np.float32)
                teaching[target_action] = 1.0
                delta_w = self.lr * (self.eligibility * teaching[np.newaxis, :])
            else:
                delta_w = self.lr * reward * self.eligibility
            self.W_out += delta_w
            self.W_out = np.clip(self.W_out, -2.0, 2.0)

        return probs, selected


# =============================================================================
# Execution Protocols for Delayed Credit
# =============================================================================

def train_and_eval_min_e(
    preset: ArchitecturePreset,
    delay_steps: int,
    train_trials: int = 30,
    eval_trials: int = 30,
    seed: int = 42,
    ablate_mechs: Optional[List[Phase16Mechanism]] = None,
    tau_elig_ms: Optional[float] = None,
    consolidation_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Train and evaluate MIN_E (or ablation) on Delayed Credit task under un-leaked protocol."""
    net = Phase16Network(preset=preset, seed=seed)
    if ablate_mechs:
        for m in ablate_mechs:
            net.ablate_mechanism(m)
    if tau_elig_ms is not None:
        net.action_readout.tau_eligibility_ms = tau_elig_ms
        net.action_readout.decay_elig = float(np.exp(-DT_MS / max(1.0, tau_elig_ms)))
    if consolidation_mode is not None:
        net.set_consolidation_mode(consolidation_mode)

    # 1. Training Phase
    train_correct = 0
    train_csis = []
    t_train0 = time.perf_counter()

    for tr in range(train_trials):
        ctx_id = tr % 2
        trial = generate_delayed_credit_trial(
            trial_idx=tr,
            delay_steps=delay_steps,
            context_id=ctx_id,
            seed=seed * 1000 + tr,
        )
        net.reset()
        chosen = 0

        for st in trial.steps:
            if st.is_choice_step:
                probs, chosen = net.step(st.sensory_input, target_action=None, is_choice_step=True)
            elif st.is_reward_step:
                rew = 1.0 if (chosen == st.target_action) else -0.5
                net.step(
                    st.sensory_input,
                    target_action=st.target_action,
                    reward=rew,
                    is_reward_step=True,
                )
            else:
                net.step(st.sensory_input, target_action=None, reward=0.0)

        if chosen == trial.target_action:
            train_correct += 1
        telem = net.last_phase15_telemetry or {}
        train_csis.append(telem.get("credit_specificity_index", 0.5))

    train_time = time.perf_counter() - t_train0
    train_acc = float(train_correct / max(1, train_trials))

    # 2. Evaluation Phase (Strictly Frozen)
    net.set_frozen(True)
    eval_correct = 0
    eval_margins = []
    eval_confidences = []
    t_eval0 = time.perf_counter()

    for tr in range(eval_trials):
        ctx_id = tr % 2
        trial = generate_delayed_credit_trial(
            trial_idx=tr,
            delay_steps=delay_steps,
            context_id=ctx_id,
            seed=seed * 2000 + tr + 500,
        )
        net.reset()
        chosen = 0
        choice_probs = np.zeros(2)

        for st in trial.steps:
            if st.is_choice_step:
                probs, chosen = net.step(st.sensory_input, target_action=None, is_choice_step=True)
                choice_probs = probs.copy()
            elif st.is_reward_step:
                # Evaluation only: no learning occurs
                rew = 1.0 if (chosen == st.target_action) else -0.5
                net.step(st.sensory_input, target_action=None, reward=rew, is_reward_step=True)
            else:
                net.step(st.sensory_input, target_action=None, reward=0.0)

        if chosen == trial.target_action:
            eval_correct += 1
        sorted_p = np.sort(choice_probs)
        eval_margins.append(float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else 1.0)
        eval_confidences.append(float(np.max(choice_probs)))

    eval_time = time.perf_counter() - t_eval0
    eval_acc = float(eval_correct / max(1, eval_trials))

    return {
        "architecture": preset.value,
        "delay_steps": delay_steps,
        "delay_ms": int(delay_steps * DT_MS),
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
        "mean_csi": float(np.mean(train_csis)) if train_csis else 0.5,
        "mean_margin": float(np.mean(eval_margins)) if eval_margins else 0.0,
        "mean_confidence": float(np.mean(eval_confidences)) if eval_confidences else 0.5,
        "train_time_sec": round(train_time, 4),
        "eval_time_sec": round(eval_time, 4),
        "complexity": net.get_complexity_stats(),
    }


def train_and_eval_transformer(
    delay_steps: int,
    train_trials: int = 30,
    eval_trials: int = 30,
    seed: int = 42,
    lr: float = 0.02,
) -> Dict[str, Any]:
    """Train and evaluate Causal Transformer on Delayed Credit task under matched budget."""
    total_steps = 5 + delay_steps + 1 + 2 + 1
    max_len = max(256, total_steps + 32)
    model = SmallCausalTransformer(input_dim=8, d_model=16, num_heads=2, d_ff=32, max_seq_len=max_len, seed=seed)

    # 1. Training Phase
    t_train0 = time.perf_counter()
    train_correct = 0

    for tr in range(train_trials):
        ctx_id = tr % 2
        trial = generate_delayed_credit_trial(tr, delay_steps, ctx_id, seed=seed * 1000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_step_idx"]
        target = float(trial.target_action)

        # Forward pass to check prediction
        y_seq = model.forward(x_seq)
        pred = 1 if float(y_seq[choice_idx, 0]) >= 0.5 else 0
        if pred == trial.target_action:
            train_correct += 1

        # Train step from reward step / choice step
        model.train_step(x_seq, target_action=target, action_step_idx=choice_idx, lr=lr)

    train_time = time.perf_counter() - t_train0
    train_acc = float(train_correct / max(1, train_trials))

    # 2. Evaluation Phase
    t_eval0 = time.perf_counter()
    eval_correct = 0
    confidences = []

    for tr in range(eval_trials):
        ctx_id = tr % 2
        trial = generate_delayed_credit_trial(tr, delay_steps, ctx_id, seed=seed * 2000 + tr + 500)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_step_idx"]

        y_seq = model.forward(x_seq)
        logit = float(y_seq[choice_idx, 0])
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0)))
        pred = 1 if prob >= 0.5 else 0
        if pred == trial.target_action:
            eval_correct += 1
        confidences.append(float(max(prob, 1.0 - prob)))

    eval_time = time.perf_counter() - t_eval0
    eval_acc = float(eval_correct / max(1, eval_trials))

    return {
        "architecture": "CAUSAL_TRANSFORMER",
        "delay_steps": delay_steps,
        "delay_ms": int(delay_steps * DT_MS),
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.5,
        "train_time_sec": round(train_time, 4),
        "eval_time_sec": round(eval_time, 4),
        "parameters": model.parameter_count(),
    }


def train_and_eval_bptt(
    delay_steps: int,
    train_trials: int = 30,
    eval_trials: int = 30,
    seed: int = 42,
    lr: float = 0.02,
) -> Dict[str, Any]:
    """Train and evaluate BPTT Recurrent baseline on Delayed Credit task."""
    model = BPTTGradientBaseline(input_dim=8, hidden_dim=40, output_dim=1, learning_rate=lr, seed=seed)

    t_train0 = time.perf_counter()
    train_correct = 0

    for tr in range(train_trials):
        ctx_id = tr % 2
        trial = generate_delayed_credit_trial(tr, delay_steps, ctx_id, seed=seed * 1000 + tr)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_step_idx"]
        target = float(trial.target_action)

        model.reset()
        outputs = model.forward(x_seq)
        pred = 1 if float(outputs[choice_idx, 0]) >= 0.5 else 0
        if pred == trial.target_action:
            train_correct += 1

        model.train_trial(x_seq, target_action=target, choice_step_idx=choice_idx)

    train_time = time.perf_counter() - t_train0
    train_acc = float(train_correct / max(1, train_trials))

    t_eval0 = time.perf_counter()
    eval_correct = 0

    for tr in range(eval_trials):
        ctx_id = tr % 2
        trial = generate_delayed_credit_trial(tr, delay_steps, ctx_id, seed=seed * 2000 + tr + 500)
        x_seq = np.array([st.sensory_input for st in trial.steps], dtype=np.float32)
        choice_idx = trial.metadata["choice_step_idx"]

        model.reset()
        outputs = model.forward(x_seq)
        pred = 1 if float(outputs[choice_idx, 0]) >= 0.5 else 0
        if pred == trial.target_action:
            eval_correct += 1

    eval_time = time.perf_counter() - t_eval0
    eval_acc = float(eval_correct / max(1, eval_trials))

    return {
        "architecture": "BPTT_RECURRENT",
        "delay_steps": delay_steps,
        "delay_ms": int(delay_steps * DT_MS),
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
        "train_time_sec": round(train_time, 4),
        "eval_time_sec": round(eval_time, 4),
        "parameters": 40 * 40 + 8 * 40 + 40 * 1 + 41,  # ~2,001
    }


def train_and_eval_simple_recurrent(
    delay_steps: int,
    train_trials: int = 30,
    eval_trials: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate Simple Recurrent Control on Delayed Credit task."""
    model = SimpleRecurrentControl(input_dim=8, hidden_dim=40, output_dim=2, seed=seed)

    t_train0 = time.perf_counter()
    train_correct = 0

    for tr in range(train_trials):
        ctx_id = tr % 2
        trial = generate_delayed_credit_trial(tr, delay_steps, ctx_id, seed=seed * 1000 + tr)
        model.reset()
        chosen = 0

        for st in trial.steps:
            if st.is_choice_step:
                probs, chosen = model.step(st.sensory_input, is_choice_step=True)
            elif st.is_reward_step:
                rew = 1.0 if (chosen == st.target_action) else -0.5
                model.step(st.sensory_input, target_action=st.target_action, reward=rew, is_reward_step=True)
            else:
                model.step(st.sensory_input, reward=0.0)

        if chosen == trial.target_action:
            train_correct += 1

    train_time = time.perf_counter() - t_train0
    train_acc = float(train_correct / max(1, train_trials))

    model.set_frozen(True)
    t_eval0 = time.perf_counter()
    eval_correct = 0

    for tr in range(eval_trials):
        ctx_id = tr % 2
        trial = generate_delayed_credit_trial(tr, delay_steps, ctx_id, seed=seed * 2000 + tr + 500)
        model.reset()
        chosen = 0

        for st in trial.steps:
            if st.is_choice_step:
                probs, chosen = model.step(st.sensory_input, is_choice_step=True)
            elif st.is_reward_step:
                model.step(st.sensory_input, is_reward_step=True)
            else:
                model.step(st.sensory_input)

        if chosen == trial.target_action:
            eval_correct += 1

    eval_time = time.perf_counter() - t_eval0
    eval_acc = float(eval_correct / max(1, eval_trials))

    return {
        "architecture": "SIMPLE_RECURRENT_CONTROL",
        "delay_steps": delay_steps,
        "delay_ms": int(delay_steps * DT_MS),
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
        "train_time_sec": round(train_time, 4),
        "eval_time_sec": round(eval_time, 4),
        "parameters": 40 * 40 + 8 * 40 + 40 * 2,  # ~2,000
    }
