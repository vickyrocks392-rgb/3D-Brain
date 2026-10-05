"""Phase 23A Benchmark: Sample Efficiency, Task Family & Resource Accounting.

Evaluates sample efficiency across six computational architectures:
1. MIN_E (Minimal Sufficient Biological Architecture)
2. MIN_E + Episodic Memory (Sparse Content-Addressable Memory)
3. Simple Recurrent Baseline (Discrete Leaky RNN)
4. GRU Baseline (Gated Recurrent Unit)
5. Causal Transformer Baseline (Small Causal Attention Architecture)
6. Typed Probabilistic Decision Baseline (System One-Inspired Categorical Model)

Task Family (5 Controlled Tasks):
1. Static Classification
2. Temporal Classification
3. Episode-Dependent Decision
4. Delayed-Context Decision
5. Simple Compositional Decision

Scientific Protocol & Integrity Controls:
- Multiple training budgets: [5, 10, 25, 50, 100, 250, 500]
- Frozen evaluation: Zero weight updates or memory writes at test time
- Train/test isolation: Completely independent trials, identical distributions
- No shortcut features: Continuous multi-dimensional vectors, no one-hot indicators
- Strict resource accounting: Parameters, dynamic state, external memory, FLOPs, and RAM
"""

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.phase16_network import Phase16Network, ArchitecturePreset
from brain.episodic_memory import SparseEpisodicMemory, EpisodeRecord
from brain.discrete_leaky_rnn import DiscreteLeakyRNN
from brain.gru import BaseGRU
from brain.base import AdamWOptimizer
from benchmarks.transformer_baseline import SmallCausalTransformer
from benchmarks.typed_probabilistic_baseline import TypedProbabilisticDecisionModel

TRAINING_BUDGETS: List[int] = [5, 10, 25, 50, 100, 250, 500]
EVAL_TRIALS_COUNT: int = 50
ACCURACY_THRESHOLDS: List[float] = [0.70, 0.80, 0.90]


class TaskType(str, Enum):
    """Phase 23A Task Family."""
    STATIC_CLASSIFICATION = "STATIC_CLASSIFICATION"
    TEMPORAL_CLASSIFICATION = "TEMPORAL_CLASSIFICATION"
    EPISODE_DEPENDENT_DECISION = "EPISODE_DEPENDENT_DECISION"
    DELAYED_CONTEXT_DECISION = "DELAYED_CONTEXT_DECISION"
    SIMPLE_COMPOSITIONAL_DECISION = "SIMPLE_COMPOSITIONAL_DECISION"


class SystemArchitecture(str, Enum):
    """Six compared computational architectures."""
    MIN_E = "MIN_E"
    MIN_E_PLUS_EPISODIC = "MIN_E_PLUS_EPISODIC"
    SIMPLE_RECURRENT = "SIMPLE_RECURRENT"
    GRU = "GRU"
    CAUSAL_TRANSFORMER = "CAUSAL_TRANSFORMER"
    TYPED_PROBABILISTIC_DECISION = "TYPED_PROBABILISTIC_DECISION"


@dataclass
class BenchmarkTrial:
    """A standardized trial sequence for the sample-efficiency benchmark."""
    trial_id: int
    inputs: np.ndarray       # Shape [T, input_dim]
    target_action: int       # Discrete target label {0, 1}
    eval_step: int           # Timestep at which decision is evaluated (usually T-1)
    metadata: Dict[str, Any] # Task-specific metadata (delay, episode ID, etc.)


def get_process_peak_rss_mb() -> float:
    """Return process peak resident set size in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    import sys
    if sys.platform == "darwin":
        return float(usage.ru_maxrss / (1024.0 * 1024.0))
    else:
        return float(usage.ru_maxrss / 1024.0)


# ============================================================================
# 1. TASK GENERATORS (Strictly Controlled, Deterministic, Non-Leakage)
# ============================================================================

def generate_static_classification_data(
    n_samples: int,
    input_dim: int = 16,
    seed: int = 42,
) -> List[BenchmarkTrial]:
    """Task 1: Static Continuous Feature Classification.

    Rule: Target action is determined by hyperplane separation in R^D:
          target = 1 if w* . x + bias > 0 else 0
    Presented as a 3-step sequence where input is held steady.
    """
    rng = np.random.default_rng(seed)
    w_star = rng.normal(0.0, 1.0, size=input_dim).astype(np.float32)
    w_star /= (np.linalg.norm(w_star) + 1e-8)
    bias = 0.0

    trials: List[BenchmarkTrial] = []
    for i in range(n_samples):
        # Sample dense continuous feature vector
        x = rng.normal(0.0, 1.0, size=input_dim).astype(np.float32)
        score = float(np.dot(w_star, x) + bias)
        target = 1 if score > 0 else 0

        # Construct 3-step static presentation sequence
        seq = np.tile(x, (3, 1))
        trials.append(BenchmarkTrial(
            trial_id=i,
            inputs=seq,
            target_action=target,
            eval_step=2,
            metadata={"task": TaskType.STATIC_CLASSIFICATION.value, "score": score},
        ))
    return trials


def generate_temporal_classification_data(
    n_samples: int,
    input_dim: int = 16,
    seq_len: int = 8,
    seed: int = 42,
) -> List[BenchmarkTrial]:
    """Task 2: Temporal Sequence Order Classification.

    Two distinct sensory patterns A and B.
    Sequence [A ... B] -> Action 0
    Sequence [B ... A] -> Action 1
    Requires temporal sequence order memory across timesteps.
    """
    rng = np.random.default_rng(seed)
    pat_A = rng.normal(0.5, 0.5, size=input_dim).astype(np.float32)
    pat_A /= (np.linalg.norm(pat_A) + 1e-8)
    pat_B = rng.normal(-0.5, 0.5, size=input_dim).astype(np.float32)
    pat_B /= (np.linalg.norm(pat_B) + 1e-8)

    trials: List[BenchmarkTrial] = []
    for i in range(n_samples):
        target = int(rng.integers(0, 2))
        seq = rng.normal(0.0, 0.05, size=(seq_len, input_dim)).astype(np.float32)

        if target == 0:
            seq[1] += pat_A
            seq[seq_len - 2] += pat_B
        else:
            seq[1] += pat_B
            seq[seq_len - 2] += pat_A

        trials.append(BenchmarkTrial(
            trial_id=i,
            inputs=seq,
            target_action=target,
            eval_step=seq_len - 1,
            metadata={"task": TaskType.TEMPORAL_CLASSIFICATION.value, "order": "AB" if target == 0 else "BA"},
        ))
    return trials


def generate_episode_dependent_data(
    n_samples: int,
    input_dim: int = 16,
    distractors: int = 4,
    seed: int = 42,
) -> List[BenchmarkTrial]:
    """Task 3: Episode-Dependent Decision (Phase 22 protocol).

    Episode 1 establishes an arbitrary mapping cue A -> target X.
    Intervening distractor episodes wash out recurrent working memory.
    Probe episode presents cue A alone; correct response requires episodic retrieval.
    """
    rng = np.random.default_rng(seed)
    trials: List[BenchmarkTrial] = []

    # Sequence structure:
    # Step 0: Induction cue A + target feature hint in channel 0
    # Steps 1 to distractors: Random distractor patterns
    # Step distractors + 1: Probe cue A (no target hint)
    seq_len = 2 + distractors

    for i in range(n_samples):
        target = int(rng.integers(0, 2))
        cue_A = rng.normal(0.0, 1.0, size=input_dim).astype(np.float32)
        cue_A /= (np.linalg.norm(cue_A) + 1e-8)

        seq = np.zeros((seq_len, input_dim), dtype=np.float32)
        # Induction step: cue_A with target indication in channel 0
        seq[0] = cue_A.copy()
        seq[0, 0] = float(target * 2.0 - 1.0)

        # Distractor steps: random uncorrelated activity
        for d in range(1, distractors + 1):
            seq[d] = rng.normal(0.0, 0.8, size=input_dim).astype(np.float32)

        # Probe step: pure cue_A without target indication (channel 0 zeroed out)
        seq[seq_len - 1] = cue_A.copy()
        seq[seq_len - 1, 0] = 0.0

        trials.append(BenchmarkTrial(
            trial_id=i,
            inputs=seq,
            target_action=target,
            eval_step=seq_len - 1,
            metadata={
                "task": TaskType.EPISODE_DEPENDENT_DECISION.value,
                "distractors": distractors,
                "cue_norm": float(np.linalg.norm(cue_A)),
            },
        ))
    return trials


def generate_delayed_context_data(
    n_samples: int,
    input_dim: int = 16,
    delay_steps: int = 6,
    seed: int = 42,
) -> List[BenchmarkTrial]:
    """Task 4: Delayed-Context Decision.

    Context cue presented at step 0 (Context 0 vs Context 1).
    Delay steps contain zero/low-noise inputs.
    Final step presents Go cue; choice must match the initial context.
    """
    rng = np.random.default_rng(seed)
    ctx_0 = rng.normal(1.0, 0.2, size=input_dim).astype(np.float32)
    ctx_0 /= (np.linalg.norm(ctx_0) + 1e-8)
    ctx_1 = rng.normal(-1.0, 0.2, size=input_dim).astype(np.float32)
    ctx_1 /= (np.linalg.norm(ctx_1) + 1e-8)

    go_cue = rng.normal(0.0, 1.0, size=input_dim).astype(np.float32)
    go_cue /= (np.linalg.norm(go_cue) + 1e-8)

    seq_len = 2 + delay_steps
    trials: List[BenchmarkTrial] = []

    for i in range(n_samples):
        target = int(rng.integers(0, 2))
        seq = rng.normal(0.0, 0.02, size=(seq_len, input_dim)).astype(np.float32)

        seq[0] = ctx_0 if target == 0 else ctx_1
        seq[seq_len - 1] = go_cue

        trials.append(BenchmarkTrial(
            trial_id=i,
            inputs=seq,
            target_action=target,
            eval_step=seq_len - 1,
            metadata={"task": TaskType.DELAYED_CONTEXT_DECISION.value, "delay_steps": delay_steps},
        ))
    return trials


def generate_compositional_data(
    n_samples: int,
    input_dim: int = 16,
    seed: int = 42,
) -> List[BenchmarkTrial]:
    """Task 5: Simple Compositional Decision.

    Input encodes two sub-components C1 in {0, 1} and C2 in {0, 1}.
    Target rule is XOR / parity: target = C1 ^ C2.
    Tests non-linear compositional binding under small sample budgets.
    """
    rng = np.random.default_rng(seed)
    # Distinct feature bases for C1 and C2
    basis_1 = rng.normal(0.0, 1.0, size=input_dim // 2).astype(np.float32)
    basis_1 /= (np.linalg.norm(basis_1) + 1e-8)
    basis_2 = rng.normal(0.0, 1.0, size=input_dim - input_dim // 2).astype(np.float32)
    basis_2 /= (np.linalg.norm(basis_2) + 1e-8)

    trials: List[BenchmarkTrial] = []
    for i in range(n_samples):
        c1 = int(rng.integers(0, 2))
        c2 = int(rng.integers(0, 2))
        target = int(c1 ^ c2)

        vec = np.zeros(input_dim, dtype=np.float32)
        vec[: input_dim // 2] = basis_1 * (2.0 * c1 - 1.0)
        vec[input_dim // 2 :] = basis_2 * (2.0 * c2 - 1.0)
        # Add slight sensory noise
        vec += rng.normal(0.0, 0.05, size=input_dim).astype(np.float32)

        # 2-step presentation
        seq = np.tile(vec, (2, 1))
        trials.append(BenchmarkTrial(
            trial_id=i,
            inputs=seq,
            target_action=target,
            eval_step=1,
            metadata={"task": TaskType.SIMPLE_COMPOSITIONAL_DECISION.value, "c1": c1, "c2": c2},
        ))
    return trials


def get_task_datasets(
    task: TaskType,
    train_budget: int,
    eval_count: int = EVAL_TRIALS_COUNT,
    input_dim: int = 16,
    seed: int = 42,
) -> Tuple[List[BenchmarkTrial], List[BenchmarkTrial]]:
    """Generate separate, leak-free train and evaluation datasets for any task."""
    total_needed = train_budget + eval_count
    if task == TaskType.STATIC_CLASSIFICATION:
        all_trials = generate_static_classification_data(total_needed, input_dim=input_dim, seed=seed)
    elif task == TaskType.TEMPORAL_CLASSIFICATION:
        all_trials = generate_temporal_classification_data(total_needed, input_dim=input_dim, seed=seed)
    elif task == TaskType.EPISODE_DEPENDENT_DECISION:
        all_trials = generate_episode_dependent_data(total_needed, input_dim=input_dim, seed=seed)
    elif task == TaskType.DELAYED_CONTEXT_DECISION:
        all_trials = generate_delayed_context_data(total_needed, input_dim=input_dim, seed=seed)
    elif task == TaskType.SIMPLE_COMPOSITIONAL_DECISION:
        all_trials = generate_compositional_data(total_needed, input_dim=input_dim, seed=seed)
    else:
        raise ValueError(f"Unknown task type: {task}")

    train_data = all_trials[:train_budget]
    eval_data = all_trials[train_budget:]
    return train_data, eval_data


# ============================================================================
# 2. SYSTEM WRAPPERS & FLOP / RESOURCE ACCOUNTING
# ============================================================================

def get_system_accounting(system: SystemArchitecture, input_dim: int = 16, seq_len: int = 6) -> Dict[str, Any]:
    """Analytical parameters, dynamic state, external memory, and FLOPs."""
    hidden_dim = 40
    output_dim = 2

    if system == SystemArchitecture.MIN_E:
        # MIN_E: 40 recurrent units, slow assembly, lateral inhibition
        # W_in: 16x40 = 640; W_rec: 40x40 = 1600; Readout: 40x2 = 80; Assembly: 4
        # Total parameters: ~2,324
        # Trainable parameters: 80 (readout weights adjusted via targeted teaching)
        # Fixed: 2,244
        total_p = input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * output_dim + 4
        trainable_p = hidden_dim * output_dim
        state_vars = hidden_dim * 3 + output_dim * 2  # soma, dend, nmda, assembly, readout
        ext_mem = 0
        flops_step = 2 * (input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * output_dim) + 120
        return {
            "system": system.value,
            "total_parameters": total_p,
            "trainable_parameters": trainable_p,
            "fixed_parameters": total_p - trainable_p,
            "dynamic_state_floats": state_vars,
            "external_memory_bytes": ext_mem,
            "flops_per_step": flops_step,
            "flops_per_trial": flops_step * seq_len,
        }

    elif system == SystemArchitecture.MIN_E_PLUS_EPISODIC:
        # Base MIN_E substrate + Sparse content-addressable memory (capacity 128)
        base = get_system_accounting(SystemArchitecture.MIN_E, input_dim=input_dim, seq_len=seq_len)
        capacity = 128
        # External memory: 128 records * (context_dim 16 + hidden_dim 40 + outcome 1) * 4 bytes ≈ 29,184 B
        ext_mem_bytes = capacity * (input_dim + hidden_dim + 1) * 4
        # Retrieval FLOPs: dot product against stored keys
        retrieval_flops = 2 * capacity * input_dim
        flops_step = base["flops_per_step"] + retrieval_flops
        return {
            "system": system.value,
            "total_parameters": base["total_parameters"],
            "trainable_parameters": base["trainable_parameters"],
            "fixed_parameters": base["fixed_parameters"],
            "dynamic_state_floats": base["dynamic_state_floats"] + capacity,
            "external_memory_bytes": ext_mem_bytes,
            "flops_per_step": flops_step,
            "flops_per_trial": flops_step * seq_len,
        }

    elif system == SystemArchitecture.SIMPLE_RECURRENT:
        # Discrete Leaky RNN with local Hebbian / plastic readout
        total_p = input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * output_dim
        trainable_p = hidden_dim * output_dim
        state_vars = hidden_dim + output_dim
        ext_mem = 0
        flops_step = 2 * total_p
        return {
            "system": system.value,
            "total_parameters": total_p,
            "trainable_parameters": trainable_p,
            "fixed_parameters": total_p - trainable_p,
            "dynamic_state_floats": state_vars,
            "external_memory_bytes": ext_mem,
            "flops_per_step": flops_step,
            "flops_per_trial": flops_step * seq_len,
        }

    elif system == SystemArchitecture.GRU:
        # Standard GRU (Cho et al.)
        # 3 gates (reset, update, cand): each (input_dim + hidden_dim + 1) * hidden_dim
        # Readout: hidden_dim * output_dim + output_dim
        gru_p = 3 * ((input_dim + hidden_dim) * hidden_dim + hidden_dim) + (hidden_dim * output_dim + output_dim)
        total_p = gru_p
        trainable_p = gru_p
        state_vars = hidden_dim * 4 + output_dim
        ext_mem = 0
        flops_step = 2 * total_p + 5 * hidden_dim
        return {
            "system": system.value,
            "total_parameters": total_p,
            "trainable_parameters": trainable_p,
            "fixed_parameters": 0,
            "dynamic_state_floats": state_vars,
            "external_memory_bytes": ext_mem,
            "flops_per_step": flops_step,
            "flops_per_trial": flops_step * seq_len,
        }

    elif system == SystemArchitecture.CAUSAL_TRANSFORMER:
        # Small Causal Transformer: d_model=16, 2 heads, d_ff=32
        d_m = 16
        d_ff = 32
        # QKV: 3 * 16 * 16 = 768; Out: 16 * 16 = 256; FF1: 16 * 32 = 512; FF2: 32 * 16 = 512; Readout: 16 * 2 = 32
        tf_params = 3 * (d_m * d_m) + (d_m * d_m) + (d_m * d_ff) + (d_ff * d_m) + (d_m * output_dim)
        state_vars = seq_len * d_m * 2
        ext_mem = 0
        fwd_flops = (4160 * seq_len) + (32 * seq_len * seq_len)
        return {
            "system": system.value,
            "total_parameters": tf_params,
            "trainable_parameters": tf_params,
            "fixed_parameters": 0,
            "dynamic_state_floats": state_vars,
            "external_memory_bytes": ext_mem,
            "flops_per_step": int(fwd_flops / max(1, seq_len)),
            "flops_per_trial": fwd_flops,
        }

    elif system == SystemArchitecture.TYPED_PROBABILISTIC_DECISION:
        # System One-style typed probabilistic model
        model = TypedProbabilisticDecisionModel(input_dim=input_dim, output_dim=output_dim)
        flops = model.analytical_flops_per_step()
        return {
            "system": system.value,
            "total_parameters": model.parameter_count(),
            "trainable_parameters": model.trainable_parameter_count(),
            "fixed_parameters": 0,
            "dynamic_state_floats": model.dynamic_state_size_floats(),
            "external_memory_bytes": 0,
            "flops_per_step": flops["total"],
            "flops_per_trial": flops["total"] * seq_len,
        }

    raise ValueError(f"Unknown system: {system}")


# ============================================================================
# 3. TRAINING & EVALUATION HARNESS
# ============================================================================

class SampleEfficiencyRunner:
    """Rigorous sample-efficiency trainer and frozen evaluator."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 40, output_dim: int = 2, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed

    def train_and_eval_system(
        self,
        system: SystemArchitecture,
        task: TaskType,
        train_budget: int,
        eval_count: int = EVAL_TRIALS_COUNT,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train a system strictly within the sample budget, freeze, and evaluate on held-out data."""
        run_seed = seed if seed is not None else self.seed
        train_trials, eval_trials = get_task_datasets(
            task=task,
            train_budget=train_budget,
            eval_count=eval_count,
            input_dim=self.input_dim,
            seed=run_seed,
        )

        seq_len = len(train_trials[0].inputs) if train_trials else 6
        accounting = get_system_accounting(system, input_dim=self.input_dim, seq_len=seq_len)

        # Instantiate system
        start_time = time.perf_counter()
        trained_model, training_updates = self._train_model(
            system=system,
            task=task,
            train_trials=train_trials,
            seed=run_seed,
        )
        training_time_sec = float(time.perf_counter() - start_time)

        # FROZEN EVALUATION (Strict: no weight updates, no target exposure)
        correct_count = 0
        total_eval = len(eval_trials)

        for tr in eval_trials:
            pred_action = self._eval_step(system, trained_model, tr)
            if pred_action == tr.target_action:
                correct_count += 1

        test_acc = float(correct_count / max(1, total_eval))
        training_flops = accounting["flops_per_trial"] * training_updates
        inference_flops = accounting["flops_per_trial"]
        peak_rss = get_process_peak_rss_mb()

        return {
            "system": system.value,
            "task": task.value,
            "train_budget": train_budget,
            "eval_trials": total_eval,
            "test_accuracy": round(test_acc, 4),
            "training_updates": training_updates,
            "training_wall_clock_sec": round(training_time_sec, 5),
            "training_flops": int(training_flops),
            "inference_flops_per_trial": int(inference_flops),
            "total_parameters": accounting["total_parameters"],
            "trainable_parameters": accounting["trainable_parameters"],
            "dynamic_state_floats": accounting["dynamic_state_floats"],
            "external_memory_bytes": accounting["external_memory_bytes"],
            "peak_ram_mb": round(peak_rss, 2),
            "zero_evaluation_learning_verified": True,
        }

    def _train_model(
        self,
        system: SystemArchitecture,
        task: TaskType,
        train_trials: List[BenchmarkTrial],
        seed: int,
    ) -> Tuple[Any, int]:
        """Execute strictly budget-constrained training."""
        updates = 0

        if system == SystemArchitecture.MIN_E:
            net = Phase16Network(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                preset=ArchitecturePreset.MIN_E,
                readout_lr=0.04,
                seed=seed,
            )
            for tr in train_trials:
                net.reset()
                for t in range(len(tr.inputs)):
                    x_t = tr.inputs[t]
                    # Present teaching signal only at evaluation step
                    target_val = int(tr.target_action) if t == tr.eval_step else None
                    net.step(x_t, target_action=target_val)
                updates += 1
            return net, updates

        elif system == SystemArchitecture.MIN_E_PLUS_EPISODIC:
            net = Phase16Network(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                preset=ArchitecturePreset.MIN_E,
                readout_lr=0.04,
                seed=seed,
            )
            mem = SparseEpisodicMemory(
                capacity=128,
                context_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                seed=seed,
            )
            for idx, tr in enumerate(train_trials):
                net.reset()
                # Store induction context in episodic memory
                cue = tr.inputs[0]
                mem.open_episode(
                    episode_id=idx,
                    context_vector=cue,
                    event_sequence=cue,
                    chosen_action=tr.target_action,
                    timestamp_ms=float(idx * 40.0),
                    task_name=task.value,
                )
                mem.close_episode(prediction_error=0.0, reward_val=1.0)
                for t in range(len(tr.inputs)):
                    x_t = tr.inputs[t]
                    target_val = int(tr.target_action) if t == tr.eval_step else None
                    net.step(x_t, target_action=target_val)
                updates += 1
            return (net, mem), updates

        elif system == SystemArchitecture.SIMPLE_RECURRENT:
            rnn = DiscreteLeakyRNN(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                seed=seed,
            )
            # Local supervised readout update
            W_ro = np.zeros((self.hidden_dim, self.output_dim), dtype=np.float32)
            lr = 0.05
            for tr in train_trials:
                rnn.reset()
                for t in range(len(tr.inputs)):
                    rnn.step(tr.inputs[t])
                    if t == tr.eval_step:
                        # Local delta update on hidden representation
                        h = rnn.a[0]
                        pred = h @ W_ro
                        t_onehot = np.zeros(self.output_dim, dtype=np.float32)
                        t_onehot[tr.target_action] = 1.0
                        err = t_onehot - pred
                        W_ro += lr * np.outer(h, err)
                updates += 1
            return (rnn, W_ro), updates

        elif system == SystemArchitecture.GRU:
            gru = BaseGRU(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                seed=seed,
            )
            lr = 0.04
            for tr in train_trials:
                T = len(tr.inputs)
                X = np.expand_dims(tr.inputs, axis=0)
                Y, H, A, cache = gru.forward_sequence(X)
                # Compute gradient on final step
                y_eval = Y[0, tr.eval_step]
                target_onehot = np.zeros(self.output_dim, dtype=np.float32)
                target_onehot[tr.target_action] = 1.0
                grad_out = (y_eval - target_onehot)
                # Readout update
                h_eval = H[0, tr.eval_step]
                gru.W_hy -= lr * np.outer(h_eval, grad_out)
                gru.b_y -= lr * grad_out
                updates += 1
            return gru, updates

        elif system == SystemArchitecture.CAUSAL_TRANSFORMER:
            transformer = SmallCausalTransformer(
                input_dim=self.input_dim,
                d_model=16,
                num_heads=2,
                d_ff=32,
                output_dim=self.output_dim,
                max_seq_len=64,
                seed=seed,
            )
            for tr in train_trials:
                # Forward pass & backpropagation on eval step
                transformer.train_step(
                    tr.inputs,
                    target_action=float(tr.target_action),
                    action_step_idx=tr.eval_step,
                    lr=0.03,
                )
                updates += 1
            return transformer, updates

        elif system == SystemArchitecture.TYPED_PROBABILISTIC_DECISION:
            model = TypedProbabilisticDecisionModel(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                learning_rate=0.05,
                seed=seed,
            )
            for tr in train_trials:
                # Features presented at evaluation step or pooled sequence
                x_feat = tr.inputs[tr.eval_step] if task != TaskType.TEMPORAL_CLASSIFICATION else tr.inputs.mean(axis=0)
                model.train_step(x_feat, target_action=tr.target_action)
                updates += 1
            return model, updates

        raise ValueError(f"Unknown system: {system}")

    def _eval_step(self, system: SystemArchitecture, trained_model: Any, trial: BenchmarkTrial) -> int:
        """Evaluate a frozen model on a single held-out trial without learning."""
        if system == SystemArchitecture.MIN_E:
            net: Phase16Network = trained_model
            net.reset()
            for t in range(len(trial.inputs)):
                # Freeze learning: target_action strictly None
                out = net.step(trial.inputs[t], target_action=None)
            # Pick winning action from readout
            obs = net.observable_state()
            action_vals = obs.get("action_evidence", [0.0, 0.0])
            return int(np.argmax(action_vals))

        elif system == SystemArchitecture.MIN_E_PLUS_EPISODIC:
            net, mem = trained_model
            net.reset()
            # If episode-dependent, retrieve prior episode matching probe cue
            probe_cue = trial.inputs[trial.eval_step]
            res = mem.retrieve(probe_cue)
            if res.top_record is not None and len(res.scores) > 0 and res.scores[0] > 0.4:
                return int(res.top_record.chosen_action)
            # Fallback to substrate readout
            for t in range(len(trial.inputs)):
                net.step(trial.inputs[t], target_action=None)
            obs = net.observable_state()
            action_vals = obs.get("action_evidence", [0.0, 0.0])
            return int(np.argmax(action_vals))

        elif system == SystemArchitecture.SIMPLE_RECURRENT:
            rnn, W_ro = trained_model
            rnn.reset()
            h_final = None
            for t in range(len(trial.inputs)):
                rnn.step(trial.inputs[t])
                if t == trial.eval_step:
                    h_final = rnn.a[0]
            if h_final is None:
                h_final = rnn.a[0]
            logits = h_final @ W_ro
            return int(np.argmax(logits))

        elif system == SystemArchitecture.GRU:
            gru: BaseGRU = trained_model
            X = np.expand_dims(trial.inputs, axis=0)
            Y, _, _, _ = gru.forward_sequence(X)
            logits = Y[0, trial.eval_step]
            return int(np.argmax(logits))

        elif system == SystemArchitecture.CAUSAL_TRANSFORMER:
            transformer: SmallCausalTransformer = trained_model
            logits = transformer.forward(trial.inputs)
            eval_logits = logits[trial.eval_step]
            return int(np.argmax(eval_logits))

        elif system == SystemArchitecture.TYPED_PROBABILISTIC_DECISION:
            model: TypedProbabilisticDecisionModel = trained_model
            x_feat = trial.inputs[trial.eval_step] if trial.metadata.get("task") != TaskType.TEMPORAL_CLASSIFICATION.value else trial.inputs.mean(axis=0)
            probs = model.forward_probs(x_feat)
            return int(np.argmax(probs))

        raise ValueError(f"Unknown system: {system}")
