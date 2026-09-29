"""Phase 14 Dendritic Calcium Coincidence Detection Engine.

Implements biophysical coincidence detection at dendritic spines and apical tufts:
1. Single isolated synaptic input (subthreshold EPSP)
2. Spatially separated multi-synaptic inputs (sublinear or linear summation)
3. Temporally coincident synaptic inputs within window Delta t (supralinear thresholding)
4. NMDA state + Intracellular calcium state coincidence (plateau initiation)
5. Larkum-style back-propagating action potential (bAP) + distal apical input coincidence

Controlled coincidence window sweep:
Delta t in [0, 1, 2, 5, 10, 20, 50] ms.
"""

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class CoincidenceTriggerType(str, Enum):
    """Specific biophysical coincidence trigger types."""
    SUBTHRESHOLD_SINGLE = "SUBTHRESHOLD_SINGLE"
    SPATIAL_SEPARATED = "SPATIAL_SEPARATED"
    TEMPORAL_COINCIDENCE = "TEMPORAL_COINCIDENCE"
    NMDA_CALCIUM_GATE = "NMDA_CALCIUM_GATE"
    BAP_DENDRITIC_PLATEAU = "BAP_DENDRITIC_PLATEAU"


class DendriticCoincidenceDetector:
    """Evaluates dendritic and somatic spatio-temporal coincidence.

    Features:
    - Finite temporal integration window Delta t (ms)
    - Supralinear boost when inputs arrive within Delta t
    - NMDA slow conductance + Calcium accumulation thresholding
    - Back-propagating action potential (bAP) feedback arriving at apical dendrite
    """

    def __init__(
        self,
        num_compartments: int = 50,
        delta_t_ms: float = 10.0,
        dt_ms: float = 1.0,
        supralinear_factor: float = 2.4,
        nmda_thresh: float = 0.25,
        ca_thresh: float = 0.35,
        plateau_duration_ms: float = 40.0,
        seed: int = 42,
    ):
        self.num_compartments = int(num_compartments)
        self.delta_t_ms = float(delta_t_ms)
        self.dt_ms = float(dt_ms)
        self.supralinear_factor = float(supralinear_factor)
        self.nmda_thresh = float(nmda_thresh)
        self.ca_thresh = float(ca_thresh)
        self.plateau_duration_ms = float(plateau_duration_ms)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Recent input arrival timestamps per compartment: List[float]
        self.input_history: List[List[float]] = [[] for _ in range(self.num_compartments)]
        self.bap_history: List[float] = []  # somatic bAP timestamps

        # Active dendritic plateaus: remaining duration in ms
        self.plateau_active = np.zeros(self.num_compartments, dtype=bool)
        self.plateau_remaining_ms = np.zeros(self.num_compartments, dtype=np.float32)

        self.current_time_ms = 0.0
        self.coincidence_events_count = 0
        self.last_trigger_types: List[CoincidenceTriggerType] = []

    def reset(self) -> None:
        """Reset coincidence buffers and active plateaus."""
        self.input_history = [[] for _ in range(self.num_compartments)]
        self.bap_history.clear()
        self.plateau_active.fill(False)
        self.plateau_remaining_ms.fill(0.0)
        self.current_time_ms = 0.0
        self.coincidence_events_count = 0
        self.last_trigger_types.clear()

    def set_delta_t(self, delta_t_ms: float) -> None:
        """Adjust temporal coincidence window (for parametric delta_t sweeps)."""
        self.delta_t_ms = max(0.0, float(delta_t_ms))

    def step(
        self,
        synaptic_inputs: np.ndarray,
        v_dend: np.ndarray,
        v_soma: np.ndarray,
        g_nmda: np.ndarray,
        calcium: np.ndarray,
        somatic_spikes: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Process one timestep of dendritic coincidence detection.

        Args:
            synaptic_inputs: Sensory/recurrent inputs per compartment [num_compartments]
            v_dend: Dendritic voltages [num_compartments]
            v_soma: Somatic voltages [num_compartments]
            g_nmda: Slow NMDA conductances [num_compartments]
            calcium: Calcium levels [num_compartments]
            somatic_spikes: Optional binary spikes from soma triggering bAP [num_compartments]

        Returns:
            Dict containing effective dendritic drive, plateau states, and coincidence flags.
        """
        self.current_time_ms += self.dt_ms
        n = self.num_compartments
        inputs = np.asarray(synaptic_inputs, dtype=np.float32)
        if len(inputs) < n:
            padded = np.zeros(n, dtype=np.float32)
            padded[: len(inputs)] = inputs
            inputs = padded
        elif len(inputs) > n:
            inputs = inputs[:n]

        # Update active plateaus
        self.plateau_remaining_ms = np.maximum(0.0, self.plateau_remaining_ms - self.dt_ms)
        self.plateau_active = self.plateau_remaining_ms > 0.0

        # Record bAP events
        if somatic_spikes is not None:
            if np.any(somatic_spikes > 0):
                self.bap_history.append(self.current_time_ms)
        # Purge bAP history older than max window (50 ms)
        self.bap_history = [t for t in self.bap_history if self.current_time_ms - t <= 60.0]

        effective_drive = inputs.copy()
        coincidence_detected = np.zeros(n, dtype=bool)
        plateau_triggered = np.zeros(n, dtype=bool)
        trigger_types: List[CoincidenceTriggerType] = []

        for i in range(n):
            # Record input event if input > threshold
            if inputs[i] > 0.15:
                self.input_history[i].append(self.current_time_ms)

            # Purge events older than 60 ms
            self.input_history[i] = [t for t in self.input_history[i] if self.current_time_ms - t <= 60.0]
            recent_events = self.input_history[i]

            # 1. Condition 1 & 3: Temporal Coincidence within compartment in Delta t
            temporal_coincident = False
            if len(recent_events) >= 2:
                # Check if last two events occurred within delta_t_ms
                if (recent_events[-1] - recent_events[-2]) <= self.delta_t_ms:
                    temporal_coincident = True

            # 2. Condition 4: NMDA + Calcium coincidence
            nmda_ca_coincident = (g_nmda[i] >= self.nmda_thresh) and (calcium[i] >= self.ca_thresh)

            # 3. Condition 5: bAP + Dendritic Depolarization Coincidence (Larkum mechanism)
            bap_dend_coincident = False
            if self.bap_history and v_dend[i] > -50.0:
                last_bap = self.bap_history[-1]
                # Coincidence within delta_t_ms
                if abs(self.current_time_ms - last_bap) <= self.delta_t_ms:
                    bap_dend_coincident = True

            # Classify trigger
            assigned_type = CoincidenceTriggerType.SUBTHRESHOLD_SINGLE
            if bap_dend_coincident:
                assigned_type = CoincidenceTriggerType.BAP_DENDRITIC_PLATEAU
            elif nmda_ca_coincident:
                assigned_type = CoincidenceTriggerType.NMDA_CALCIUM_GATE
            elif temporal_coincident:
                assigned_type = CoincidenceTriggerType.TEMPORAL_COINCIDENCE
            elif inputs[i] > 0.4:
                assigned_type = CoincidenceTriggerType.SPATIAL_SEPARATED

            # Supralinear boost upon coincidence
            is_coincident = (
                bap_dend_coincident
                or nmda_ca_coincident
                or (temporal_coincident and self.delta_t_ms > 0.0)
            )

            if is_coincident:
                coincidence_detected[i] = True
                effective_drive[i] *= self.supralinear_factor
                self.coincidence_events_count += 1
                trigger_types.append(assigned_type)

                # Trigger plateau potential if high NMDA/Ca or bAP coincidence
                if (nmda_ca_coincident or bap_dend_coincident) and not self.plateau_active[i]:
                    self.plateau_active[i] = True
                    self.plateau_remaining_ms[i] = self.plateau_duration_ms
                    plateau_triggered[i] = True

            # If compartment is in active plateau, inject sustained depolarizing current
            if self.plateau_active[i]:
                effective_drive[i] += 0.50

        self.last_trigger_types = trigger_types

        return {
            "effective_drive": effective_drive,
            "coincidence_detected": coincidence_detected,
            "plateau_active": self.plateau_active.copy(),
            "plateau_triggered": plateau_triggered,
            "total_coincidences": self.coincidence_events_count,
            "trigger_types": [t.value for t in trigger_types],
        }
