"""Standalone Replay Engine for 3D-Brain.

Decoupled from browser UI; provides programmatic execution replay, inspection,
and seeking over recorded .brainlog trajectories.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from observability.brainlog import BrainLogValidator
from observability.events import ComputationalEvent


@dataclass
class NodeInspection:
    node_id: int
    label: str
    node_type: str
    pos: List[float]
    unit_index: int
    current_activation: float
    current_state: float
    incoming_connections: List[dict]
    outgoing_connections: List[dict]
    dynamic_vars: Dict[str, Any]


@dataclass
class EdgeInspection:
    src: int
    dst: int
    weight: float
    src_label: str
    dst_label: str
    current_src_activation: float
    current_signal_magnitude: float
    is_active: bool


class ReplayEngine:
    """Headless replay engine managing trajectory time-travel and component inspection."""

    def __init__(self, brainlog_data: Optional[dict] = None):
        self.data: Optional[dict] = None
        self.current_step: int = 0
        self.is_playing: bool = False
        self.playback_speed: float = 1.0

        if brainlog_data:
            self.load(brainlog_data)

    def load(self, source: Union[str, Path, dict]) -> None:
        """Load and validate a .brainlog file or dictionary."""
        valid, msg = BrainLogValidator.validate(source)
        if not valid:
            raise ValueError(f"Invalid .brainlog data: {msg}")

        if isinstance(source, (str, Path)):
            import json
            with open(source, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = source

        self.current_step = 0
        self.is_playing = False

    @property
    def total_steps(self) -> int:
        if not self.data:
            return 0
        return len(self.data["frames"])

    @property
    def metadata(self) -> dict:
        return self.data["metadata"] if self.data else {}

    @property
    def topology(self) -> dict:
        return self.data["topology"] if self.data else {}

    def seek(self, step: int) -> dict:
        """Seek to an exact simulation time step."""
        if not self.data:
            raise RuntimeError("No .brainlog loaded in ReplayEngine")
        if step < 0 or step >= self.total_steps:
            raise IndexError(f"Step {step} out of bounds [0, {self.total_steps - 1}]")
        self.current_step = step
        return self.current_frame()

    def step_forward(self) -> dict:
        """Step one time increment forward."""
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
        return self.current_frame()

    def step_backward(self) -> dict:
        """Step one time increment backward."""
        if self.current_step > 0:
            self.current_step -= 1
        return self.current_frame()

    def current_frame(self) -> dict:
        """Return full telemetry frame at current time step."""
        if not self.data:
            return {}
        return self.data["frames"][self.current_step]

    def get_events_at_step(self, step: Optional[int] = None) -> List[dict]:
        """Return all computational events emitted at step (default: current step)."""
        s = self.current_step if step is None else step
        frame = self.data["frames"][s]
        return frame.get("events", [])

    def get_all_events(self) -> List[dict]:
        """Return entire event stream for timeline scrubber indexing."""
        if not self.data:
            return []
        if "events" in self.data:
            return self.data["events"]
        # Fallback to gathering from frames
        events = []
        for f in self.data["frames"]:
            events.extend(f.get("events", []))
        return events

    def inspect_node(self, node_id: int) -> NodeInspection:
        """Inspect a specific computational neuron at current step."""
        if not self.data:
            raise RuntimeError("No .brainlog loaded")
        nodes = self.data["topology"]["nodes"]
        if node_id < 0 or node_id >= len(nodes):
            raise IndexError(f"Node ID {node_id} out of bounds")

        node_info = nodes[node_id]
        frame = self.current_frame()

        act = frame["activations"][node_id]
        state = frame["states"][node_id]

        edges = self.data["topology"]["edges"]
        in_edges = [
            {"src": e["src"], "weight": e["weight"], "src_label": nodes[e["src"]]["label"]}
            for e in edges
            if e["dst"] == node_id
        ]
        out_edges = [
            {"dst": e["dst"], "weight": e["weight"], "dst_label": nodes[e["dst"]]["label"]}
            for e in edges
            if e["src"] == node_id
        ]

        dynamic_vars = {}
        dyn_details = frame.get("dynamic_details", {})
        unit_idx = node_info.get("unit_index", -1)
        if unit_idx >= 0:
            for var_name, var_list in dyn_details.items():
                if isinstance(var_list, list) and unit_idx < len(var_list):
                    dynamic_vars[var_name] = var_list[unit_idx]

        return NodeInspection(
            node_id=node_id,
            label=node_info["label"],
            node_type=node_info["type"],
            pos=node_info["pos"],
            unit_index=unit_idx,
            current_activation=act,
            current_state=state,
            incoming_connections=in_edges,
            outgoing_connections=out_edges,
            dynamic_vars=dynamic_vars,
        )

    def inspect_edge(self, src: int, dst: int) -> Optional[EdgeInspection]:
        """Inspect a specific directed connection at current step."""
        if not self.data:
            return None
        nodes = self.data["topology"]["nodes"]
        edges = self.data["topology"]["edges"]

        match = [e for e in edges if e["src"] == src and e["dst"] == dst]
        if not match:
            return None

        e = match[0]
        frame = self.current_frame()
        src_act = frame["activations"][src]
        signal_mag = src_act * e["weight"]
        is_active = abs(signal_mag) >= 0.07

        return EdgeInspection(
            src=src,
            dst=dst,
            weight=e["weight"],
            src_label=nodes[src]["label"],
            dst_label=nodes[dst]["label"],
            current_src_activation=src_act,
            current_signal_magnitude=round(float(signal_mag), 4),
            is_active=is_active,
        )
