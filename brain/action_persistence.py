"""Biophysically Plausible Action Persistence & Attractor Switching (Phase 13).

Prevents rapid action chatter and premature decision abandonment by providing:
1. Hysteresis Switching Barrier: Changing action requires evidence exceeding the current
   selected action by a dynamic switching threshold Delta_switch.
2. Refractory Action Lockout: Once an action is triggered, an absolute lockout period
   (T_lockout) protects the action from transient subthreshold fluctuations.
3. Recurrent Self-Sustenance: Intra-pool positive feedback maintains action population
   firing during brief sensory pauses or delays.
4. Comprehensive Persistence Metrics:
   - Action duration
   - Unwanted switching rate
   - Action chatter (high-frequency toggling)
   - False transitions
   - Selection latency
   - Persistence stability
   - Context-dependent switching accuracy
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ActionPersistenceState(str, Enum):
    """Dynamical state of the action persistence controller."""
    IDLE = "IDLE"
    ONSET = "ONSET"
    SELECTION = "SELECTION"
    MAINTENANCE = "MAINTENANCE"
    REFRACTORY_LOCKOUT = "REFRACTORY_LOCKOUT"
    TERMINATION = "TERMINATION"
    SWITCHING = "SWITCHING"


class ActionPersistenceController:
    """Manages temporal persistence, hysteresis, and switching of selected actions."""

    def __init__(
        self,
        num_actions: int = 2,
        dt_ms: float = 1.0,
        tau_persistence_ms: float = 120.0,
        lockout_duration_ms: float = 30.0,
        hysteresis_margin: float = 0.15,
        decay_rate: float = 0.02,
        max_duration_ms: float = 500.0,
        seed: int = 42,
    ):
        self.num_actions = max(2, int(num_actions))
        self.dt_ms = float(dt_ms)
        self.tau_persistence_ms = float(tau_persistence_ms)
        self.lockout_duration_ms = float(lockout_duration_ms)
        self.hysteresis_margin = float(hysteresis_margin)
        self.decay_rate = float(decay_rate)
        self.max_duration_ms = float(max_duration_ms)
        self.seed = seed

        # State tracking
        self.state = ActionPersistenceState.IDLE
        self.current_action: Optional[int] = None
        self.action_onset_time_ms = 0.0
        self.current_action_duration_ms = 0.0
        self.lockout_remaining_ms = 0.0
        self.total_time_ms = 0.0

        # Memory traces
        self.action_activations = np.zeros(self.num_actions, dtype=np.float32)
        self.persistence_activations = np.zeros(self.num_actions, dtype=np.float32)

        # Telemetry & metrics history
        self.history_actions: List[int] = []
        self.history_durations: List[float] = []
        self.history_switches: List[Tuple[float, int, int]] = []  # (time, from, to)
        self.chatter_events = 0
        self.unwanted_switches = 0
        self.false_transitions = 0
        self.selection_latencies: List[float] = []
        self.context_switch_successes = 0
        self.context_switch_attempts = 0

        # Current context ID for tracking context-dependent stability
        self._current_context_id: Optional[int] = None
        self._last_context_id: Optional[int] = None
        self._context_onset_time_ms = 0.0

    def reset(self) -> None:
        """Reset internal dynamic states and histories."""
        self.state = ActionPersistenceState.IDLE
        self.current_action = None
        self.action_onset_time_ms = 0.0
        self.current_action_duration_ms = 0.0
        self.lockout_remaining_ms = 0.0
        self.total_time_ms = 0.0
        self.action_activations.fill(0.0)
        self.persistence_activations.fill(0.0)
        self.history_actions.clear()
        self.history_durations.clear()
        self.history_switches.clear()
        self.chatter_events = 0
        self.unwanted_switches = 0
        self.false_transitions = 0
        self.selection_latencies.clear()
        self.context_switch_successes = 0
        self.context_switch_attempts = 0
        self._current_context_id = None
        self._last_context_id = None
        self._context_onset_time_ms = 0.0

    def set_context(self, context_id: int) -> None:
        """Notify controller of current task context."""
        if self._current_context_id != context_id:
            self._last_context_id = self._current_context_id
            self._current_context_id = context_id
            self._context_onset_time_ms = self.total_time_ms
            if self._last_context_id is not None:
                self.context_switch_attempts += 1

    def step(
        self,
        raw_evidence: np.ndarray,
        spikes: Optional[np.ndarray] = None,
        target_action: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update action persistence and resolve active winning action.

        Args:
            raw_evidence: Instantaneous evidence or firing rate per action [num_actions]
            spikes: Optional population spike raster
            target_action: Ground truth target action if known (for false switch analysis)

        Returns:
            Dict containing selected_action, state, duration, persistence_levels, and metrics.
        """
        self.total_time_ms += self.dt_ms
        ev = np.asarray(raw_evidence, dtype=np.float32)
        if len(ev) != self.num_actions:
            raise ValueError(f"Expected evidence of length {self.num_actions}, got {len(ev)}")

        # 1. Update Leaky Persistence Dynamics with Recurrent Attractor Latching
        alpha_p = float(np.exp(-self.dt_ms / max(1.0, self.tau_persistence_ms)))
        self.persistence_activations = self.persistence_activations * alpha_p + (1.0 - alpha_p) * ev

        # Add active boost to currently maintained action
        if self.current_action is not None and self.state in (
            ActionPersistenceState.MAINTENANCE,
            ActionPersistenceState.REFRACTORY_LOCKOUT,
        ):
            # Recurrent attractor self-excitation: maintain active pool
            self.persistence_activations[self.current_action] += 0.05 * self.dt_ms

        # 2. Handle Lockout Timer
        if self.lockout_remaining_ms > 0.0:
            self.lockout_remaining_ms = max(0.0, self.lockout_remaining_ms - self.dt_ms)
            in_lockout = True
        else:
            in_lockout = False

        # 3. Decision & Winner Evaluation
        best_candidate = int(np.argmax(self.persistence_activations))
        best_evidence = float(self.persistence_activations[best_candidate])

        # 4. State Machine Execution
        if self.current_action is None:
            # Initial Action Selection
            if best_evidence > 0.01:
                self.current_action = best_candidate
                self.action_onset_time_ms = self.total_time_ms
                self.current_action_duration_ms = 0.0
                self.lockout_remaining_ms = self.lockout_duration_ms
                self.state = ActionPersistenceState.ONSET
                latency = self.total_time_ms - self._context_onset_time_ms
                self.selection_latencies.append(latency)
            else:
                self.state = ActionPersistenceState.IDLE
        else:
            self.current_action_duration_ms += self.dt_ms
            current_ev = float(self.persistence_activations[self.current_action])

            # Check for timeout termination
            if self.current_action_duration_ms >= self.max_duration_ms:
                self.state = ActionPersistenceState.TERMINATION
                self.history_durations.append(self.current_action_duration_ms)
                # Soft reset of persistence for current action to allow switching
                self.persistence_activations[self.current_action] *= 0.1
                self.current_action = None
                self.current_action_duration_ms = 0.0

            elif in_lockout:
                # Lockout protects against any switching
                self.state = ActionPersistenceState.REFRACTORY_LOCKOUT

            else:
                # Outside lockout: Evaluate hysteresis switching condition
                # Must exceed current action evidence by hysteresis_margin
                if best_candidate != self.current_action and (best_evidence - current_ev) > self.hysteresis_margin:
                    # Action Switch Triggered!
                    old_action = self.current_action
                    new_action = best_candidate
                    self.history_switches.append((self.total_time_ms, old_action, new_action))
                    self.history_durations.append(self.current_action_duration_ms)

                    # Check for action chatter (switching again within 20 ms)
                    if self.current_action_duration_ms < 20.0:
                        self.chatter_events += 1

                    # Check if switch is unwanted (context has not changed)
                    if self._last_context_id is not None and self._current_context_id == self._last_context_id:
                        self.unwanted_switches += 1

                    # Check if switch is a false transition (switched away from correct target)
                    if target_action is not None and old_action == target_action:
                        self.false_transitions += 1

                    # Check context-switching success
                    if target_action is not None and new_action == target_action:
                        self.context_switch_successes += 1

                    self.current_action = new_action
                    self.action_onset_time_ms = self.total_time_ms
                    self.current_action_duration_ms = 0.0
                    self.lockout_remaining_ms = self.lockout_duration_ms
                    self.state = ActionPersistenceState.SWITCHING
                else:
                    self.state = ActionPersistenceState.MAINTENANCE

        chosen_action = self.current_action if self.current_action is not None else best_candidate
        self.history_actions.append(chosen_action)

        return {
            "selected_action": chosen_action,
            "state": self.state.value,
            "duration_ms": self.current_action_duration_ms,
            "lockout_remaining_ms": self.lockout_remaining_ms,
            "persistence_activations": self.persistence_activations.copy(),
            "best_evidence": best_evidence,
        }

    def get_persistence_metrics(self) -> Dict[str, float]:
        """Compute aggregate stability and persistence metrics."""
        total_steps = max(1, len(self.history_actions))
        total_switches = len(self.history_switches)

        mean_duration = float(np.mean(self.history_durations)) if self.history_durations else self.current_action_duration_ms
        std_duration = float(np.std(self.history_durations)) if self.history_durations else 0.0
        mean_latency = float(np.mean(self.selection_latencies)) if self.selection_latencies else 0.0

        # Switching rate per second
        sim_time_sec = max(0.001, self.total_time_ms / 1000.0)
        switching_rate_hz = float(total_switches / sim_time_sec)

        # Chatter rate: chatter events per switch
        chatter_ratio = float(self.chatter_events / max(1, total_switches))

        # Persistence stability: fraction of steps where an action was sustained >= 50ms
        stable_steps = 0
        cur_len = 0
        for i in range(1, len(self.history_actions)):
            if self.history_actions[i] == self.history_actions[i - 1]:
                cur_len += 1
            else:
                if cur_len >= 50:
                    stable_steps += cur_len
                cur_len = 0
        if cur_len >= 50:
            stable_steps += cur_len
        persistence_stability = float(stable_steps / total_steps)

        # Context-dependent switching accuracy
        if self.context_switch_attempts > 0:
            context_switching_acc = float(self.context_switch_successes / self.context_switch_attempts)
        else:
            context_switching_acc = 1.0

        return {
            "action_duration_mean_ms": round(mean_duration, 2),
            "action_duration_std_ms": round(std_duration, 2),
            "selection_latency_ms": round(mean_latency, 2),
            "switching_rate_hz": round(switching_rate_hz, 3),
            "chatter_events": self.chatter_events,
            "chatter_ratio": round(chatter_ratio, 3),
            "unwanted_switches": self.unwanted_switches,
            "false_transitions": self.false_transitions,
            "persistence_stability": round(persistence_stability, 4),
            "context_switching_accuracy": round(context_switching_acc, 4),
        }
