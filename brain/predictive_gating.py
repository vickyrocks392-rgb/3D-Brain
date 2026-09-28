"""Predictive Dendritic Disinhibition & Top-Down Gating Engine (Phase 11).

Implements biologically grounded dendritic disinhibition:
- In cortical microcircuits (VIP-SOM-Pyramidal motif), top-down predictive projections
  activate VIP interneurons, which inhibit SOM interneurons that target apical dendrites.
- The net effect is dendritic disinhibition: reducing the hyperpolarizing inhibitory shunt
  on specific dendritic branches, allowing preserved NMDA state and plateau events to couple
  into the soma and trigger action selection.
- Irrelevant pathways retain full dendritic inhibition, shunting noise and distractor inputs.
- Provides explicit causal modes:
  * GATE_ON: Active predictive disinhibition derived from ongoing context/attractor state
  * GATE_OFF: Gating clamped to 0 (no disinhibition, passive full inhibition)
  * GATE_SHUFFLED: Gating vector randomly permuted across neurons/trials
  * GATE_DELAYED: Gating vector delayed in time by delta_t steps
  * GATE_REVERSED: Inverted gating vector (disinhibiting the wrong pathway)
  * NO_DISINHIBITION: Raw inhibition without modulatory gating
"""

from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class GateMode(str, Enum):
    GATE_ON = "GATE_ON"
    GATE_OFF = "GATE_OFF"
    GATE_SHUFFLED = "GATE_SHUFFLED"
    GATE_DELAYED = "GATE_DELAYED"
    GATE_REVERSED = "GATE_REVERSED"
    NO_DISINHIBITION = "NO_DISINHIBITION"


class PredictiveDendriticGating:
    """Computes endogenous top-down predictive signals to gate dendritic inhibitory conductances."""

    def __init__(
        self,
        num_neurons: int,
        hidden_dim: int,
        gate_mode: GateMode = GateMode.GATE_ON,
        disinhibition_strength: float = 0.85,
        gate_delay_steps: int = 5,
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.hidden_dim = hidden_dim
        self.gate_mode = GateMode(gate_mode)
        self.disinhibition_strength = float(disinhibition_strength)
        self.gate_delay_steps = int(gate_delay_steps)
        self.rng = np.random.RandomState(seed)

        # Top-down predictive projection weights (context attractor -> dendritic gate)
        # Maps attractor population state to gating vector
        self.W_pred = self.rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, num_neurons)).astype(np.float32)
        self.b_pred = np.zeros(num_neurons, dtype=np.float32)

        # Circular buffer for GATE_DELAYED mode
        self._delay_queue: deque = deque(maxlen=max(1, self.gate_delay_steps + 1))
        for _ in range(self.gate_delay_steps):
            self._delay_queue.append(np.zeros(num_neurons, dtype=np.float32))

        # Telemetry cache
        self.last_gate_vector = np.zeros(num_neurons, dtype=np.float32)
        self.last_disinhibition_factor = np.ones(num_neurons, dtype=np.float32)
        self.last_gate_selectivity = 0.0

    def reset(self) -> None:
        """Reset internal buffers and history."""
        self._delay_queue.clear()
        for _ in range(self.gate_delay_steps):
            self._delay_queue.append(np.zeros(self.num_neurons, dtype=np.float32))
        self.last_gate_vector.fill(0.0)
        self.last_disinhibition_factor.fill(1.0)
        self.last_gate_selectivity = 0.0

    def compute_gate(
        self,
        attractor_state: np.ndarray,
        relevant_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the raw predictive gating vector G in [0, 1]^N from current attractor state.

        Args:
            attractor_state: Current hidden/attractor population state [hidden_dim]
            relevant_mask: Optional boolean mask of causally relevant neurons [num_neurons]

        Returns:
            gate_vector: Gating vector G in [0, 1]^N
        """
        s = np.asarray(attractor_state, dtype=np.float32)
        if s.ndim == 2:
            s = s[0]

        # Raw linear projection followed by sigmoid activation
        z = s @ self.W_pred + self.b_pred
        # Sigmoid: 1 / (1 + exp(-z))
        z_clipped = np.clip(z, -15.0, 15.0)
        g_raw = 1.0 / (1.0 + np.exp(-z_clipped))

        # Store into delay buffer
        self._delay_queue.append(g_raw.copy())

        # Apply gate mode
        if self.gate_mode == GateMode.GATE_OFF or self.gate_mode == GateMode.NO_DISINHIBITION:
            g_eff = np.zeros(self.num_neurons, dtype=np.float32)
        elif self.gate_mode == GateMode.GATE_SHUFFLED:
            g_eff = self.rng.permutation(g_raw)
        elif self.gate_mode == GateMode.GATE_DELAYED:
            g_eff = self._delay_queue[0]
        elif self.gate_mode == GateMode.GATE_REVERSED:
            g_eff = 1.0 - g_raw
        else:  # GATE_ON
            g_eff = g_raw

        self.last_gate_vector = g_eff.copy()

        # Compute Gate Selectivity if relevant mask is provided
        if relevant_mask is not None:
            rel = np.asarray(relevant_mask, dtype=bool)
            if np.any(rel) and np.any(~rel):
                mean_rel = float(np.mean(g_eff[rel]))
                mean_irrel = float(np.mean(g_eff[~rel]))
                denom = mean_rel + mean_irrel + 1e-8
                self.last_gate_selectivity = float((mean_rel - mean_irrel) / denom)
            else:
                self.last_gate_selectivity = 0.0

        return g_eff

    def apply_disinhibition(
        self,
        i_syn_i: np.ndarray,
        gate_vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Modulate dendritic inhibitory synaptic current via predictive disinhibition.

        I_syn_i_gated = I_syn_i * (1.0 - gamma_disinhib * G)
        When G = 1, dendritic inhibition is attenuated by gamma_disinhib (e.g. 85%),
        disinhibiting the dendritic compartment and allowing NMDA depolarizations to couple into soma.

        Args:
            i_syn_i: Dendritic inhibitory current [num_neurons] (<= 0.0)
            gate_vector: Optional override for gating vector [num_neurons]

        Returns:
            i_syn_i_gated: Gated inhibitory current [num_neurons] (<= 0.0)
        """
        i_i = np.asarray(i_syn_i, dtype=np.float32)
        if self.gate_mode == GateMode.NO_DISINHIBITION:
            self.last_disinhibition_factor.fill(1.0)
            return i_i

        g = gate_vector if gate_vector is not None else self.last_gate_vector
        # Disinhibition factor: 1.0 - gamma * G in [1 - gamma, 1.0]
        factor = 1.0 - self.disinhibition_strength * np.clip(g, 0.0, 1.0)
        self.last_disinhibition_factor = factor.copy()

        return i_i * factor
