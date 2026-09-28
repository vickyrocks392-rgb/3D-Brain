"""Principled computational event model for 3D-Brain.

Events represent discrete, verifiable mathematical operations or threshold crossings
within the computational substrate. No synthetic 'thoughts' or decorative animations.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class EventType(str, Enum):
    TRIAL_STARTED = "TRIAL_STARTED"
    INPUT_PRESENTED = "INPUT_PRESENTED"
    STATE_UPDATED = "STATE_UPDATED"
    ACTIVATION_CHANGED = "ACTIVATION_CHANGED"
    SIGNAL_PROPAGATED = "SIGNAL_PROPAGATED"
    MEMORY_RETAINED = "MEMORY_RETAINED"
    OUTPUT_COMPUTED = "OUTPUT_COMPUTED"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    TRIAL_COMPLETED = "TRIAL_COMPLETED"
    
    # Phase 3: Plasticity & Continual Adaptation Events
    PLASTICITY_UPDATE = "PLASTICITY_UPDATE"
    WEIGHT_INCREASE = "WEIGHT_INCREASE"
    WEIGHT_DECREASE = "WEIGHT_DECREASE"
    MODULATION_ON = "MODULATION_ON"
    MODULATION_OFF = "MODULATION_OFF"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    TASK_SWITCH = "TASK_SWITCH"
    MEMORY_RETENTION_CHECK = "MEMORY_RETENTION_CHECK"
    FORGETTING_MEASUREMENT = "FORGETTING_MEASUREMENT"
    STABILITY_WARNING = "STABILITY_WARNING"

    # Phase 4: Spatial Topology & Latency Events
    SIGNAL_EMITTED = "SIGNAL_EMITTED"
    SIGNAL_ARRIVED = "SIGNAL_ARRIVED"
    PHASE_SYNC_BURST = "PHASE_SYNC_BURST"
    COMMUNITY_FORMATION = "COMMUNITY_FORMATION"

    # Phase 5: Embodied World Model & Dynamic Brain Growth Events
    ACTION_SELECTED = "ACTION_SELECTED"
    PREDICTION_MADE = "PREDICTION_MADE"
    SYNAPSE_FORMED = "SYNAPSE_FORMED"
    SYNAPSE_PRUNED = "SYNAPSE_PRUNED"
    NEURON_BORN = "NEURON_BORN"
    NEURON_PRUNED = "NEURON_PRUNED"




@dataclass(frozen=True)
class ComputationalEvent:
    """Verifiable computational event record."""

    event_type: EventType
    step: int
    time_ms: float
    node_id: Optional[int] = None
    source_node: Optional[int] = None
    target_node: Optional[int] = None
    magnitude: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": self.event_type.value,
            "step": self.step,
            "time_ms": round(float(self.time_ms), 1),
        }
        if self.node_id is not None:
            d["node_id"] = self.node_id
        if self.source_node is not None:
            d["source_node"] = self.source_node
        if self.target_node is not None:
            d["target_node"] = self.target_node
        if self.magnitude is not None:
            d["magnitude"] = round(float(self.magnitude), 4)
        if self.payload:
            d["payload"] = self.payload
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ComputationalEvent":
        return cls(
            event_type=EventType(data["type"]),
            step=int(data["step"]),
            time_ms=float(data["time_ms"]),
            node_id=data.get("node_id"),
            source_node=data.get("source_node"),
            target_node=data.get("target_node"),
            magnitude=data.get("magnitude"),
            payload=data.get("payload", {}),
        )
