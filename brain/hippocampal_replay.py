"""Phase 20 Track B: Sparse Hippocampal-Style Replay & Offline Rest Consolidation.

Implements an explicit episodic replay buffer that stores sparse event/state traces
legitimately available during active experience, replaying them exclusively during
quiescent REST periods to bridge multi-second credit assignment without continuous supervision:

       ACTIVE EXPERIENCE (Trial t)
                 │
                 ├──► Store Choice State: <x_t, a_t, e_t, tau_t> (Pre-outcome)
                 │
                 ▼ (Delay interval: strictly silent, no targets, no leakage)
                 │
                 └──► Outcome Observed: <R_t, delta_t, M_t>
                            │
                            ▼
              EPISODIC REPLAY BUFFER (Capacity N_mem)
                            │
       QUIESCENT REST PERIOD (x = 0, R = 0, no targets)
                            │
                            ▼
       SPARSE EVENT-TRIGGERED / PRIORITIZED REPLAY
       (B0 None, B1 Random, B2 Event, B3 Prioritized, B4/B5 Consolidation)
                            │
                            ▼
       COUNTERFACTUAL DENDRITIC CREDIT ASSIGNMENT
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.synaptic_consolidation import ConsolidationMode


class ReplayCondition(str, Enum):
    """Replay condition for Track B credit horizon evaluation."""
    B0_NO_REPLAY = "B0_NO_REPLAY"
    B1_RANDOM_REPLAY = "B1_RANDOM_REPLAY"
    B2_EVENT_TRIGGERED = "B2_EVENT_TRIGGERED"
    B3_PRIORITIZED_REPLAY = "B3_PRIORITIZED_REPLAY"
    B4_EVENT_CONSOLIDATION = "B4_EVENT_CONSOLIDATION"
    B5_PRIORITIZED_CONSOLIDATION = "B5_PRIORITIZED_CONSOLIDATION"


@dataclass
class ReplayTrace:
    """Verifiable episodic trace recorded legitimately without future target leakage."""
    episode_id: int
    task_id: str
    context_vector: np.ndarray          # Legitimate upstream representation x in R^Din
    chosen_action: int                  # Actual action selected at choice time
    delay_steps: int                    # Duration of delay bridged
    delay_ms: float
    eligibility_snapshot: np.ndarray    # Dendritic eligibility trace at choice time [Din, K]
    timestamp_ms: float
    target_action: Optional[int] = None # Added ONLY at outcome/reward step
    reward_val: float = 0.0             # Added ONLY at outcome/reward step
    prediction_error: float = 0.0       # Surprise magnitude |delta|
    is_replayed: bool = False
    replay_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "chosen_action": self.chosen_action,
            "target_action": self.target_action,
            "delay_ms": round(float(self.delay_ms), 1),
            "reward_val": round(float(self.reward_val), 3),
            "prediction_error": round(float(self.prediction_error), 4),
            "replay_count": self.replay_count,
        }


class HippocampalReplayBuffer:
    """Sparse episodic memory buffer for offline rest credit assignment."""

    def __init__(
        self,
        capacity: int = 64,
        replay_budget: int = 4,
        surprise_threshold: float = 0.5,
        priority_alpha: float = 1.0,
        seed: int = 42,
    ):
        self.capacity = max(4, int(capacity))
        self.replay_budget = max(0, int(replay_budget))
        self.surprise_threshold = float(surprise_threshold)
        self.priority_alpha = float(priority_alpha)
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.traces: List[ReplayTrace] = []
        self._current_pending_trace: Optional[ReplayTrace] = None

        # Causal ablation controls
        self.timing_randomized: bool = False
        self.content_randomized: bool = False
        self.wrong_episode_injection: bool = False
        self.is_disabled: bool = False

        # Telemetry
        self.total_replay_events: int = 0
        self.total_replay_flops: int = 0

    def reset(self) -> None:
        """Clear episodic buffer and reset metrics."""
        self.traces.clear()
        self._current_pending_trace = None
        self.total_replay_events = 0
        self.total_replay_flops = 0

    def store_choice(
        self,
        episode_id: int,
        task_id: str,
        context_vector: np.ndarray,
        chosen_action: int,
        delay_steps: int,
        delay_ms: float,
        eligibility_snapshot: np.ndarray,
        timestamp_ms: float,
    ) -> None:
        """Capture choice state before outcome. Strictly NO target or reward stored."""
        if self.is_disabled:
            return

        trace = ReplayTrace(
            episode_id=episode_id,
            task_id=task_id,
            context_vector=context_vector.copy().astype(np.float32),
            chosen_action=int(chosen_action),
            delay_steps=int(delay_steps),
            delay_ms=float(delay_ms),
            eligibility_snapshot=eligibility_snapshot.copy().astype(np.float32),
            timestamp_ms=float(timestamp_ms),
            target_action=None,
            reward_val=0.0,
            prediction_error=0.0,
        )
        self._current_pending_trace = trace

    def record_outcome(
        self,
        target_action: Optional[int],
        reward_val: float,
        prediction_error: float,
    ) -> Optional[ReplayTrace]:
        """Record legitimate outcome at causal feedback step and commit trace to buffer."""
        if self.is_disabled or self._current_pending_trace is None:
            return None

        trace = self._current_pending_trace
        trace.target_action = target_action
        trace.reward_val = float(reward_val)
        trace.prediction_error = float(abs(prediction_error))

        if len(self.traces) >= self.capacity:
            self.traces.pop(0)
        self.traces.append(trace)
        self._current_pending_trace = None
        return trace

    def sample_replay_batch(
        self,
        condition: ReplayCondition,
        budget: Optional[int] = None,
    ) -> List[ReplayTrace]:
        """Sample a batch of traces according to the specified replay condition."""
        if self.is_disabled or condition == ReplayCondition.B0_NO_REPLAY or len(self.traces) == 0:
            return []

        k = self.replay_budget if budget is None else max(0, int(budget))
        if k == 0:
            return []

        k = min(k, len(self.traces))

        if condition == ReplayCondition.B1_RANDOM_REPLAY:
            # Uniform random sampling
            indices = self.rng.choice(len(self.traces), size=k, replace=False)
            selected = [self.traces[i] for i in indices]

        elif condition in (ReplayCondition.B2_EVENT_TRIGGERED, ReplayCondition.B4_EVENT_CONSOLIDATION):
            # Triggered on high-surprise events
            surprising = [t for t in self.traces if t.prediction_error >= self.surprise_threshold]
            if len(surprising) == 0:
                # Fallback to the latest trial
                selected = [self.traces[-1]]
            else:
                k_sel = min(k, len(surprising))
                indices = self.rng.choice(len(surprising), size=k_sel, replace=False)
                selected = [surprising[i] for i in indices]

        elif condition in (ReplayCondition.B3_PRIORITIZED_REPLAY, ReplayCondition.B5_PRIORITIZED_CONSOLIDATION):
            # Prioritized replay based on prediction error magnitude |delta|^alpha
            errors = np.array([max(1e-4, t.prediction_error) for t in self.traces], dtype=np.float64)
            priorities = errors ** self.priority_alpha
            probs = priorities / np.sum(priorities)
            indices = self.rng.choice(len(self.traces), size=k, replace=False, p=probs)
            selected = [self.traces[i] for i in indices]

        else:
            selected = []

        # Causal ablation modifications
        if self.content_randomized:
            # Corrupt replayed context and eligibility with Gaussian noise
            corrupted = []
            for t in selected:
                c_trace = ReplayTrace(
                    episode_id=t.episode_id,
                    task_id=t.task_id,
                    context_vector=self.rng.normal(0.0, 1.0, size=t.context_vector.shape).astype(np.float32),
                    chosen_action=int(self.rng.integers(0, 2)),
                    delay_steps=t.delay_steps,
                    delay_ms=t.delay_ms,
                    eligibility_snapshot=self.rng.normal(0.0, 0.5, size=t.eligibility_snapshot.shape).astype(np.float32),
                    timestamp_ms=t.timestamp_ms,
                    target_action=int(self.rng.integers(0, 2)),
                    reward_val=t.reward_val,
                    prediction_error=t.prediction_error,
                )
                corrupted.append(c_trace)
            selected = corrupted

        if self.wrong_episode_injection:
            # Invert the target action (simulate wrong episode association)
            wrong = []
            for t in selected:
                inv_target = 1 - t.target_action if t.target_action is not None else 0
                w_trace = ReplayTrace(
                    episode_id=t.episode_id,
                    task_id=t.task_id + "_WRONG",
                    context_vector=t.context_vector,
                    chosen_action=1 - t.chosen_action,
                    delay_steps=t.delay_steps,
                    delay_ms=t.delay_ms,
                    eligibility_snapshot=t.eligibility_snapshot,
                    timestamp_ms=t.timestamp_ms,
                    target_action=inv_target,
                    reward_val=-0.5,
                    prediction_error=1.0,
                )
                wrong.append(w_trace)
            selected = wrong

        return selected

    def calculate_memory_footprint(self) -> Dict[str, Any]:
        """Compute exact memory accounting for replay buffer."""
        if len(self.traces) == 0:
            bytes_per_trace = 0
        else:
            t0 = self.traces[0]
            bytes_per_trace = (
                t0.context_vector.nbytes
                + t0.eligibility_snapshot.nbytes
                + sys.getsizeof(t0)
            )

        total_bytes = bytes_per_trace * len(self.traces)
        return {
            "capacity_traces": self.capacity,
            "current_traces": len(self.traces),
            "bytes_per_trace": bytes_per_trace,
            "total_buffer_bytes": total_bytes,
            "total_replay_events": self.total_replay_events,
            "total_replay_flops": self.total_replay_flops,
        }
