"""Multi-Timescale Action Assemblies (Phase 15).

Implements competitive neural assemblies operating across multiple distinct timescales:
1. FAST (tens of ms, tau ~ 25 ms): Responsive to instantaneous bursts, rapid switching
2. MEDIUM (hundreds of ms, tau ~ 250 ms): Deliberative evidence accumulation, attractor latching
3. SLOW (seconds, tau ~ 1500 ms): Long-delay persistence, bridging multi-second delays

Architecture:
  READOUT DRIVE u_k(t)
          │
          ├───► FAST ASSEMBLY (tau_fast = 25 ms) ────┐
          │                                          │
          ├───► MEDIUM ASSEMBLY (tau_med = 250 ms) ──┼──► POPULATION RECEPTOR / LATERAL INHIBITION
          │                                          │      a_k(t) = sum_m alpha_m a_k,m(t)
          └───► SLOW ASSEMBLY (tau_slow = 1500 ms) ──┘      p = softmax(a)
                                                            Winner Margin / Confidence
"""

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class AssemblyTimescaleMode(str, Enum):
    """Timescale combination regime."""
    A0_FAST_ONLY = "A0_FAST_ONLY"
    A1_MEDIUM_ONLY = "A1_MEDIUM_ONLY"
    A2_SLOW_ONLY = "A2_SLOW_ONLY"
    A3_FAST_MEDIUM = "A3_FAST_MEDIUM"
    A4_MEDIUM_SLOW = "A4_MEDIUM_SLOW"
    A5_FAST_MEDIUM_SLOW = "A5_FAST_MEDIUM_SLOW"


class MultiTimescaleAssembly:
    """Manages multi-timescale competitive action assemblies."""

    def __init__(
        self,
        num_actions: int = 2,
        mode: AssemblyTimescaleMode = AssemblyTimescaleMode.A5_FAST_MEDIUM_SLOW,
        tau_fast_ms: float = 25.0,
        tau_med_ms: float = 250.0,
        tau_slow_ms: float = 1500.0,
        w_self: float = 0.20,
        w_cross_inh: float = 0.50,
        dt_ms: float = 1.0,
        seed: int = 42,
    ):
        self.num_actions = max(2, int(num_actions))
        self.mode = mode
        self.tau_fast_ms = float(tau_fast_ms)
        self.tau_med_ms = float(tau_med_ms)
        self.tau_slow_ms = float(tau_slow_ms)
        self.w_self = float(w_self)
        self.w_cross_inh = float(w_cross_inh)
        self.dt_ms = float(dt_ms)
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        # Decay factors per timestep: exp(-dt / tau)
        self.decay_fast = float(np.exp(-dt_ms / max(1.0, self.tau_fast_ms)))
        self.decay_med = float(np.exp(-dt_ms / max(1.0, self.tau_med_ms)))
        self.decay_slow = float(np.exp(-dt_ms / max(1.0, self.tau_slow_ms)))

        # Multi-timescale state vectors: [num_actions]
        self.act_fast = np.zeros(self.num_actions, dtype=np.float32)
        self.act_med = np.zeros(self.num_actions, dtype=np.float32)
        self.act_slow = np.zeros(self.num_actions, dtype=np.float32)
        self.combined_activations = np.zeros(self.num_actions, dtype=np.float32)

        # Probabilities & Decision metrics
        self.action_probs = np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        self.selected_action: int = 0
        self.decision_confidence: float = 0.0
        self.action_margin: float = 0.0
        self.action_entropy: float = math.log(self.num_actions)

        # Switching & persistence tracking
        self.current_winner: int = 0
        self.winner_duration_steps: int = 0
        self.total_switches: int = 0
        self.history_winners: List[int] = []

    def set_mode(self, mode: AssemblyTimescaleMode) -> None:
        """Set timescale assembly mode A0 to A5."""
        self.mode = mode

    def reset(self) -> None:
        """Reset internal activations and histories."""
        self.act_fast.fill(0.0)
        self.act_med.fill(0.0)
        self.act_slow.fill(0.0)
        self.combined_activations.fill(0.0)
        self.action_probs.fill(1.0 / self.num_actions)
        self.selected_action = 0
        self.decision_confidence = 0.0
        self.action_margin = 0.0
        self.action_entropy = math.log(self.num_actions)
        self.current_winner = 0
        self.winner_duration_steps = 0
        self.total_switches = 0
        self.history_winners.clear()

    def step(
        self,
        readout_drives: np.ndarray,
        target_action: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Step multi-timescale assembly integration and lateral competition.
        
        Args:
            readout_drives: [num_actions] continuous evidence from action readout
            target_action: optional target action index for supervision controls
            
        Returns:
            dict containing combined activations, probabilities, winner, confidence, and telemetry
        """
        u = np.asarray(readout_drives, dtype=np.float32)
        K = self.num_actions

        # 1. Update Fast assembly (tau ~ 25 ms)
        # da_fast/dt = -a_fast/tau_fast + u + self_exc - cross_inh
        inh_fast = self.w_cross_inh * (np.sum(np.maximum(0.0, self.act_fast)) - np.maximum(0.0, self.act_fast))
        self.act_fast = (
            self.decay_fast * self.act_fast
            + (1.0 - self.decay_fast) * (u + self.w_self * np.maximum(0.0, self.act_fast) - inh_fast)
        )

        # 2. Update Medium assembly (tau ~ 250 ms)
        inh_med = self.w_cross_inh * (np.sum(np.maximum(0.0, self.act_med)) - np.maximum(0.0, self.act_med))
        self.act_med = (
            self.decay_med * self.act_med
            + (1.0 - self.decay_med) * (u + self.w_self * np.maximum(0.0, self.act_med) - inh_med)
        )

        # 3. Update Slow assembly (tau ~ 1500 ms)
        inh_slow = self.w_cross_inh * (np.sum(np.maximum(0.0, self.act_slow)) - np.maximum(0.0, self.act_slow))
        self.act_slow = (
            self.decay_slow * self.act_slow
            + (1.0 - self.decay_slow) * (u + self.w_self * np.maximum(0.0, self.act_slow) - inh_slow)
        )

        # Clip activations to prevent explosion
        self.act_fast = np.clip(self.act_fast, -10.0, 10.0)
        self.act_med = np.clip(self.act_med, -10.0, 10.0)
        self.act_slow = np.clip(self.act_slow, -10.0, 10.0)

        # 4. Assembly Mode Weighting (A0 to A5)
        m = self.mode
        if m == AssemblyTimescaleMode.A0_FAST_ONLY:
            combined = self.act_fast.copy()
        elif m == AssemblyTimescaleMode.A1_MEDIUM_ONLY:
            combined = self.act_med.copy()
        elif m == AssemblyTimescaleMode.A2_SLOW_ONLY:
            combined = self.act_slow.copy()
        elif m == AssemblyTimescaleMode.A3_FAST_MEDIUM:
            combined = 0.40 * self.act_fast + 0.60 * self.act_med
        elif m == AssemblyTimescaleMode.A4_MEDIUM_SLOW:
            combined = 0.50 * self.act_med + 0.50 * self.act_slow
        else:  # A5_FAST_MEDIUM_SLOW
            combined = 0.20 * self.act_fast + 0.30 * self.act_med + 0.50 * self.act_slow

        self.combined_activations = combined.astype(np.float32)

        # 5. Softmax competition & Probabilities
        shifted = combined - np.max(combined)
        exp_c = np.exp(np.clip(shifted, -20.0, 20.0))
        sum_exp = np.sum(exp_c) + 1e-12
        probs = exp_c / sum_exp
        self.action_probs = probs.astype(np.float32)

        # 6. Winner selection & Action Margin
        sorted_indices = np.argsort(combined)[::-1]
        winner = int(sorted_indices[0])
        runner_up = int(sorted_indices[1]) if K > 1 else winner

        self.action_margin = float(combined[winner] - combined[runner_up])
        # Mathematically defined confidence: probability difference
        self.decision_confidence = float(np.clip(probs[winner] - probs[runner_up], 0.0, 1.0))

        # Action entropy: H(P) = -sum(p * log(p))
        clip_p = np.clip(probs, 1e-12, 1.0)
        self.action_entropy = float(-np.sum(clip_p * np.log(clip_p)))

        # 7. Persistence and Switching tracking
        if winner != self.current_winner:
            self.total_switches += 1
            self.current_winner = winner
            self.winner_duration_steps = 1
        else:
            self.winner_duration_steps += 1

        self.selected_action = winner
        self.history_winners.append(winner)
        if len(self.history_winners) > 200:
            self.history_winners.pop(0)

        return {
            "selected_action": self.selected_action,
            "action_probs": self.action_probs.copy(),
            "combined_activations": self.combined_activations.copy(),
            "act_fast": self.act_fast.copy(),
            "act_med": self.act_med.copy(),
            "act_slow": self.act_slow.copy(),
            "action_margin": round(self.action_margin, 4),
            "decision_confidence": round(self.decision_confidence, 4),
            "action_entropy": round(self.action_entropy, 4),
            "winner_duration_steps": self.winner_duration_steps,
            "total_switches": self.total_switches,
            "timescale_mode": self.mode.value,
        }
