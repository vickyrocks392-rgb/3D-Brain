"""Observability Engine for 3D-Brain.

Provides passive state capture, computational event generation, static topology
extraction, .brainlog serialization/validation, and standalone replay.
"""

from observability.brainlog import BrainLog, BrainLogSerializer, BrainLogValidator
from observability.capture import StateCapture, TelemetryFrame, TelemetrySignal
from observability.events import ComputationalEvent, EventType
from observability.replay import EdgeInspection, NodeInspection, ReplayEngine
from observability.topology import BrainTopology, TopologyEdge, TopologyExtractor, TopologyNode

__all__ = [
    "BrainLog",
    "BrainLogSerializer",
    "BrainLogValidator",
    "StateCapture",
    "TelemetryFrame",
    "TelemetrySignal",
    "ComputationalEvent",
    "EventType",
    "EdgeInspection",
    "NodeInspection",
    "ReplayEngine",
    "BrainTopology",
    "TopologyEdge",
    "TopologyExtractor",
    "TopologyNode",
]
