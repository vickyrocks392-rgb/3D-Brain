"""Phase 17 Generalization & Transfer Task Suite and Substrate.

Provides rigorous, deterministic task generators and the Phase17Network substrate
to answer the Phase 17 scientific question:
"Does the Phase 16 minimal architecture generalize its persistent memory, target-specific
local credit assignment, temporal representation, and stable action selection across
qualitatively different tasks and temporal regimes without task-specific architectural
expansion — or is it merely specialized to delayed contextual choice?"
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.phase16_network import (
    Phase16Network,
    Phase16Mechanism,
    ArchitecturePreset,
)
from brain.action_readout import (
    ActionReadout,
    InputRepresentationType,
    LearningMechanism,
    TeachingSignalType,
    RetrogradeSignalType,
    DendriticSTDPType,
)
from brain.multi_timescale_assembly import (
    MultiTimescaleAssembly,
    AssemblyTimescaleMode,
)
from brain.information_measurement_audit import InformationMeasurementAuditor
from observability.events import ComputationalEvent, EventType
from benchmarks.delayed_credit_assignment import (
    LongDelayContextualCreditTask,
    StepRecord,
)
from benchmarks.embodied_environment import MinimalEmbodiedEnvironment, Action as EmbodiedAction


# =============================================================================
# 1. Unified Task Step & Evaluation Data Structures
# =============================================================================

@dataclass
class TaskStep:
    """Single time-step sample for any Phase 17 task."""
    step_idx: int
    sensory_input: np.ndarray        # [input_dim]
    target_action: int               # Ground-truth action label
    context_id: int                  # Context / condition index
    is_choice_step: bool             # When true, agent must commit to an action
    is_reward_step: bool             # When true, teaching reward is delivered
    reward_val: float = 0.0          # Reward magnitude delivered at this step
    metadata: Dict[str, Any] = None


@dataclass
class TaskTrial:
    """A complete sequence of TaskSteps."""
    trial_idx: int
    context_id: int
    target_action: int
    steps: List[TaskStep]
    task_name: str
    metadata: Dict[str, Any] = None


# =============================================================================
# 2. Task Family Generators
# =============================================================================

class TaskFamilyA_DelayedChoice:
    """Task Family A: Delayed Contextual Choice (Expanded Distribution).
    
    Tests within-task generalization across familiar and unseen delays, unseen contexts,
    and noisy/distractor observations.
    Uses canonical 5-step cue encoding (t_ctx=5) to ensure proper biophysical NMDA latching.
    """

    def __init__(self, input_dim: int = 8, num_actions: int = 2, seed: int = 42):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_trial(
        self,
        trial_idx: int,
        cue_steps: int = 5,
        delay_steps: int = 10,
        reward_delay: int = 2,
        context_id: int = 0,
        noise_std: float = 0.0,
        distractor_prob: float = 0.0,
        seed: Optional[int] = None,
    ) -> TaskTrial:
        trial_rng = np.random.RandomState(seed if seed is not None else (self.seed + trial_idx * 17))
        target_action = context_id % self.num_actions

        steps: List[TaskStep] = []
        total_steps = cue_steps + delay_steps + 1 + reward_delay + 1

        for s in range(total_steps):
            x = np.zeros(self.input_dim, dtype=np.float32)
            is_choice = (s == cue_steps + delay_steps)
            is_reward = (s == total_steps - 1)

            if s < cue_steps:
                # Canonical cue presentation
                if context_id % 2 == 0:
                    x[0] = 1.0
                    x[2:5] = 0.6
                else:
                    x[1] = 1.0
                    x[5:8] = 0.6
            elif s < cue_steps + delay_steps:
                # Delay interval: distractor pulses if enabled
                if distractor_prob > 0.0 and trial_rng.rand() < distractor_prob:
                    ch = trial_rng.randint(2, 6)
                    x[ch] = float(trial_rng.uniform(0.3, 0.7))
            elif is_choice:
                # Choice trigger probe
                x[-1] = 1.0
            elif is_reward:
                # Reward delivery trigger
                x[-2] = 1.0

            if noise_std > 0.0:
                x += trial_rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)

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
            task_name="TASK_A_DELAYED_CHOICE",
            metadata={"delay_steps": delay_steps, "noise_std": noise_std},
        )


class TaskFamilyB_SequencePrediction:
    """Task Family B: Sequence Prediction & Temporal Pattern Discrimination.
    
    Tests whether MIN_E can maintain temporal order and predict next states
    across variable horizons.
    Pattern 0: C0 (5 steps) -> Delay -> Common Ambiguous C1 (5 steps) -> Target 0
    Pattern 1: C2 (5 steps) -> Delay -> Common Ambiguous C1 (5 steps) -> Target 1
    """

    def __init__(self, input_dim: int = 8, num_actions: int = 2, seed: int = 42):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.seed = seed

    def generate_trial(
        self,
        trial_idx: int,
        pattern_id: int = 0,
        cue_steps: int = 5,
        inter_cue_delay: int = 4,
        probe_delay: int = 6,
        seed: Optional[int] = None,
    ) -> TaskTrial:
        trial_rng = np.random.RandomState(seed if seed is not None else (self.seed + trial_idx * 23))
        target_action = pattern_id % self.num_actions

        steps: List[TaskStep] = []
        total_steps = cue_steps + inter_cue_delay + cue_steps + probe_delay + 1 + 2

        t_first_end = cue_steps
        t_ambiguous_start = cue_steps + inter_cue_delay
        t_ambiguous_end = t_ambiguous_start + cue_steps
        t_choice = t_ambiguous_end + probe_delay
        t_reward = total_steps - 1

        for s in range(total_steps):
            x = np.zeros(self.input_dim, dtype=np.float32)
            is_choice = (s == t_choice)
            is_reward = (s == t_reward)

            if s < t_first_end:
                if pattern_id == 0:
                    x[0] = 1.0
                    x[2:5] = 0.6
                else:
                    x[2] = 1.0
                    x[5:8] = 0.6
            elif t_ambiguous_start <= s < t_ambiguous_end:
                x[1] = 1.0  # Common ambiguous cue
                x[4] = 0.5
            elif is_choice:
                x[-1] = 1.0  # Probe
            elif is_reward:
                x[-2] = 1.0

            steps.append(
                TaskStep(
                    step_idx=s,
                    sensory_input=x,
                    target_action=target_action,
                    context_id=pattern_id,
                    is_choice_step=is_choice,
                    is_reward_step=is_reward,
                    reward_val=1.0 if is_reward else 0.0,
                )
            )

        return TaskTrial(
            trial_idx=trial_idx,
            context_id=pattern_id,
            target_action=target_action,
            steps=steps,
            task_name="TASK_B_SEQUENCE_PREDICTION",
            metadata={"pattern_id": pattern_id, "inter_cue_delay": inter_cue_delay, "probe_delay": probe_delay},
        )


class TaskFamilyC_EmbodiedSpatial:
    """Task Family C: Embodied / Spatial Landmark-Conditioned Navigation.
    
    Reuses MinimalEmbodiedEnvironment. Agent observes 4 wall distance sensors +
    4 landmark sensors (8D input). Target action is to turn toward landmark A (action 0)
    or landmark B (action 1) conditioned on a prior cue.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.env = MinimalEmbodiedEnvironment(arena_size=10.0, step_size=0.5, seed=seed)

    def generate_trial(
        self,
        trial_idx: int,
        target_landmark_id: int = 0,
        cue_steps: int = 5,
        navigation_steps: int = 8,
        seed: Optional[int] = None,
    ) -> TaskTrial:
        trial_rng = np.random.RandomState(seed if seed is not None else (self.seed + trial_idx * 31))
        self.env.reset(seed=(seed if seed is not None else self.seed + trial_idx))

        steps: List[TaskStep] = []
        target_action = target_landmark_id % 2

        # Cue presentation for cue_steps to prime NMDA
        for s in range(cue_steps):
            obs = self.env.get_observation()
            cue_x = obs.copy()
            if target_landmark_id % 2 == 0:
                cue_x[0] += 1.5
                cue_x[2:5] += 0.6
            else:
                cue_x[1] += 1.5
                cue_x[5:8] += 0.6

            steps.append(
                TaskStep(
                    step_idx=s,
                    sensory_input=cue_x,
                    target_action=target_action,
                    context_id=target_landmark_id,
                    is_choice_step=False,
                    is_reward_step=False,
                )
            )

        # Navigation steps
        for s in range(cue_steps, cue_steps + navigation_steps):
            act_enum = EmbodiedAction(trial_rng.randint(0, 4))
            obs, _ = self.env.step(act_enum)

            is_choice = (s == cue_steps + navigation_steps - 1)
            steps.append(
                TaskStep(
                    step_idx=s,
                    sensory_input=obs.copy(),
                    target_action=target_action,
                    context_id=target_landmark_id,
                    is_choice_step=is_choice,
                    is_reward_step=False,
                )
            )

        # Reward delivery step
        reward_obs = self.env.get_observation()
        reward_obs[-2] = 1.0
        steps.append(
            TaskStep(
                step_idx=cue_steps + navigation_steps,
                sensory_input=reward_obs,
                target_action=target_action,
                context_id=target_landmark_id,
                is_choice_step=False,
                is_reward_step=True,
                reward_val=1.0,
            )
        )

        return TaskTrial(
            trial_idx=trial_idx,
            context_id=target_landmark_id,
            target_action=target_action,
            steps=steps,
            task_name="TASK_C_EMBODIED_SPATIAL",
            metadata={"target_landmark_id": target_landmark_id},
        )


class TaskFamilyD_ContinualLearning:
    """Task Family D: Continual Adaptation Across Sequential Task Switches.
    
    Sequence: Task A (mapping 0->0, 1->1) -> Task B (inverted mapping 0->1, 1->0) ->
              Task C (delayed cue) -> Return to Task A (re-test retention / forgetting).
    """

    def __init__(self, input_dim: int = 8, seed: int = 42):
        self.input_dim = input_dim
        self.seed = seed

    def generate_trial(
        self,
        trial_idx: int,
        regime: str = "REGIME_A",
        context_id: int = 0,
        cue_steps: int = 5,
        delay_steps: int = 8,
        seed: Optional[int] = None,
    ) -> TaskTrial:
        trial_rng = np.random.RandomState(seed if seed is not None else (self.seed + trial_idx * 41))

        if regime == "REGIME_A":
            target_action = context_id % 2
        elif regime == "REGIME_B":
            target_action = (1 - (context_id % 2))  # Inverted mapping
        elif regime == "REGIME_C":
            target_action = context_id % 2          # Alternate sensory channel
        else:
            target_action = context_id % 2

        steps: List[TaskStep] = []
        total_steps = cue_steps + delay_steps + 1 + 2

        for s in range(total_steps):
            x = np.zeros(self.input_dim, dtype=np.float32)
            is_choice = (s == cue_steps + delay_steps)
            is_reward = (s == total_steps - 1)

            if s < cue_steps:
                if regime in ("REGIME_A", "REGIME_B"):
                    if context_id % 2 == 0:
                        x[0] = 1.0
                        x[2:5] = 0.6
                    else:
                        x[1] = 1.0
                        x[5:8] = 0.6
                else:
                    if context_id % 2 == 0:
                        x[4] = 1.0
                        x[6] = 0.6
                    else:
                        x[5] = 1.0
                        x[7] = 0.6
            elif is_choice:
                x[-1] = 1.0
            elif is_reward:
                x[-2] = 1.0

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
            task_name="TASK_D_CONTINUAL_LEARNING",
            metadata={"regime": regime},
        )


class TaskFamilyE_CompositionalContext:
    """Task Family E: Compositional Context (Context A x Context B -> Action).
    
    Tests compositional generalization.
    Features: Context A in {0, 1}, Context B in {0, 1}.
    Target: XOR rule: (A != B) -> action 1, else action 0.
    Train on {(0,0), (0,1), (1,0)}; held-out test on {(1,1)}.
    """

    def __init__(self, input_dim: int = 8, seed: int = 42):
        self.input_dim = input_dim
        self.seed = seed

    def generate_trial(
        self,
        trial_idx: int,
        ctx_a: int = 0,
        ctx_b: int = 0,
        cue_steps: int = 5,
        delay_steps: int = 8,
        seed: Optional[int] = None,
    ) -> TaskTrial:
        target_action = 1 if (ctx_a != ctx_b) else 0
        comb_id = ctx_a * 2 + ctx_b

        steps: List[TaskStep] = []
        total_steps = cue_steps + delay_steps + 1 + 2

        for s in range(total_steps):
            x = np.zeros(self.input_dim, dtype=np.float32)
            is_choice = (s == cue_steps + delay_steps)
            is_reward = (s == total_steps - 1)

            if s < cue_steps:
                # Independent channels: Context A in [0, 1], Context B in [2, 3]
                x[0 if ctx_a == 0 else 1] = 1.0
                x[2 if ctx_b == 0 else 3] = 1.0
                x[4 + comb_id] = 0.4
            elif is_choice:
                x[-1] = 1.0
            elif is_reward:
                x[-2] = 1.0

            steps.append(
                TaskStep(
                    step_idx=s,
                    sensory_input=x,
                    target_action=target_action,
                    context_id=comb_id,
                    is_choice_step=is_choice,
                    is_reward_step=is_reward,
                    reward_val=1.0 if is_reward else 0.0,
                )
            )

        return TaskTrial(
            trial_idx=trial_idx,
            context_id=comb_id,
            target_action=target_action,
            steps=steps,
            task_name="TASK_E_COMPOSITIONAL_CONTEXT",
            metadata={"ctx_a": ctx_a, "ctx_b": ctx_b, "comb_id": comb_id},
        )


class TaskFamilyF_DistractorInterference:
    """Task Family F: Distractor & Interference Robustness.
    
    Presents task cues accompanied by irrelevant sensory channels, distractor temporal pulses,
    and deceptive intermediate rewards.
    """

    def __init__(self, input_dim: int = 8, seed: int = 42):
        self.input_dim = input_dim
        self.seed = seed

    def generate_trial(
        self,
        trial_idx: int,
        context_id: int = 0,
        cue_steps: int = 5,
        delay_steps: int = 10,
        interference_level: str = "HIGH",  # LOW, MEDIUM, HIGH
        seed: Optional[int] = None,
    ) -> TaskTrial:
        trial_rng = np.random.RandomState(seed if seed is not None else (self.seed + trial_idx * 53))
        target_action = context_id % 2

        steps: List[TaskStep] = []
        total_steps = cue_steps + delay_steps + 1 + 2

        # Noise & distractor parameters
        dist_rate = 0.4 if interference_level == "HIGH" else (0.2 if interference_level == "MEDIUM" else 0.05)
        noise_std = 0.10 if interference_level == "HIGH" else (0.05 if interference_level == "MEDIUM" else 0.01)

        for s in range(total_steps):
            x = np.zeros(self.input_dim, dtype=np.float32)
            is_choice = (s == cue_steps + delay_steps)
            is_reward = (s == total_steps - 1)

            if s < cue_steps:
                # True cue
                if context_id % 2 == 0:
                    x[0] = 1.0
                    x[2:5] = 0.6
                else:
                    x[1] = 1.0
                    x[5:8] = 0.6
                # Distractor cue in irrelevant channel
                x[trial_rng.randint(2, 6)] = float(trial_rng.uniform(0.3, 0.7))
            elif s < cue_steps + delay_steps:
                # Distractor temporal pulses
                if trial_rng.rand() < dist_rate:
                    ch = trial_rng.randint(2, 6)
                    x[ch] = float(trial_rng.uniform(0.3, 0.7))
            elif is_choice:
                x[-1] = 1.0
            elif is_reward:
                x[-2] = 1.0

            x += trial_rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)

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
            task_name="TASK_F_DISTRACTOR_INTERFERENCE",
            metadata={"interference_level": interference_level},
        )


# =============================================================================
# 3. Phase 17 Network Substrate with Generalization & BrainLog Telemetry
# =============================================================================

class Phase17Network(Phase16Network):
    """Phase 17 Generalized Substrate:
    - Maintains exact MIN_E minimal architecture by default.
    - Supports freezing (MIN_E-FROZEN) vs few-shot adaptation (MIN_E-ADAPTED).
    - Records explicit Phase 17 BrainLog computational events.
    - Supports causal lesions and mechanism inspections across task families.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 40,
        output_dim: int = 2,
        preset: ArchitecturePreset = ArchitecturePreset.MIN_E,
        frozen: bool = False,
        readout_lr: float = 0.015,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            preset=preset,
            readout_lr=readout_lr,
            seed=seed,
            **kwargs,
        )
        self.is_frozen = frozen
        if self.is_frozen:
            self.action_readout.is_frozen = True
            self.action_readout.learning_mech = LearningMechanism.D1_FROZEN

        self.current_task_name: str = "NONE"
        self.phase17_events: List[ComputationalEvent] = []

    def set_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze synaptic plasticity."""
        self.is_frozen = frozen
        self.action_readout.is_frozen = frozen
        if frozen:
            self.action_readout.learning_mech = LearningMechanism.D1_FROZEN
        else:
            self._synchronize_readout_subsystems()

    def record_phase17_event(
        self,
        event_type: EventType,
        step: int,
        time_ms: float,
        magnitude: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> ComputationalEvent:
        """Create and record an explicit Phase 17 computational event."""
        evt = ComputationalEvent(
            event_type=event_type,
            step=step,
            time_ms=time_ms,
            magnitude=magnitude,
            payload=payload or {},
        )
        self.phase17_events.append(evt)
        return evt

    def execute_trial(
        self,
        trial: TaskTrial,
        record_stages: bool = False,
        step_offset: int = 0,
    ) -> Dict[str, Any]:
        """Execute a full TaskTrial through the minimal substrate."""
        self.reset()
        chosen_action = 0
        choice_probs = np.zeros(self.output_dim)
        confidence = 0.0
        margin = 0.0
        switches = 0

        for st in trial.steps:
            rew = 0.0
            if st.is_reward_step:
                rew = 1.0 if (chosen_action == st.target_action) else -0.5

            probs, sel_act = self.step(
                x=st.sensory_input,
                target_action=st.target_action,
                context_id=st.context_id,
                reward=rew,
                is_choice_step=st.is_choice_step,
                is_reward_step=st.is_reward_step,
            )

            if st.is_choice_step:
                chosen_action = sel_act
                choice_probs = probs.copy()
                sorted_p = np.sort(probs)
                margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else 1.0
                confidence = float(np.max(probs))

        is_correct = (chosen_action == trial.target_action)
        telem = self.last_phase15_telemetry or {}

        return {
            "trial_idx": trial.trial_idx,
            "task_name": trial.task_name,
            "context_id": trial.context_id,
            "target_action": trial.target_action,
            "chosen_action": chosen_action,
            "is_correct": is_correct,
            "choice_probs": choice_probs.tolist(),
            "margin": margin,
            "confidence": confidence,
            "csi": float(telem.get("credit_specificity_index", 0.35)),
            "precision": float(telem.get("credit_precision", 0.0)),
            "leakage": float(telem.get("credit_leakage", 0.65)),
            "switches": float(telem.get("total_switches", 0)),
        }


# =============================================================================
# 4. Standard Unified Trial Runner & Evaluator
# =============================================================================

def evaluate_task_trials(
    model: Phase17Network,
    trials: List[TaskTrial],
    record_audit_stages: bool = False,
) -> Dict[str, Any]:
    """Run an entire trial battery on Phase17Network and aggregate metrics."""
    correct_count = 0
    margins, confs, csis, precs, leaks, switches = [], [], [], [], [], []

    for tr in trials:
        res = model.execute_trial(tr, record_stages=record_audit_stages)
        if res["is_correct"]:
            correct_count += 1
        margins.append(res["margin"])
        confs.append(res["confidence"])
        csis.append(res["csi"])
        precs.append(res["precision"])
        leaks.append(res["leakage"])
        switches.append(res["switches"])

    total = len(trials)
    acc = float(correct_count / max(1, total))

    return {
        "accuracy": acc,
        "total_trials": total,
        "correct_count": correct_count,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
        "csi": float(np.mean(csis)) if csis else 0.5,
        "precision": float(np.mean(precs)) if precs else 0.0,
        "leakage": float(np.mean(leaks)) if leaks else 0.5,
        "switches": float(np.mean(switches)) if switches else 0.0,
    }
