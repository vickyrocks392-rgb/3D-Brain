"""Serialization, deserialization, and schema validation for .brainlog (v2.0).

Maintains strict determinism, forward compatibility, and backward compatibility with v1.0 traces.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from observability.capture import TelemetryFrame
from observability.events import ComputationalEvent
from observability.topology import BrainTopology


class BrainLog:
    """Represents a validated, in-memory .brainlog data container."""

    def __init__(
        self,
        metadata: Dict[str, Any],
        topology: BrainTopology,
        frames: List[TelemetryFrame],
        events: List[ComputationalEvent],
        snapshots: Dict[int, dict],
        version: str = "2.5",
    ):
        self.version = version
        self.metadata = metadata
        self.topology = topology
        self.frames = frames
        self.events = events
        self.snapshots = snapshots

    def to_dict(self) -> dict:
        return {
            "brainlog_version": self.version,
            "metadata": self.metadata,
            "topology": self.topology.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "events": [e.to_dict() for e in self.events],
            "snapshots": self.snapshots,
        }

    def save(self, filepath: Union[str, Path]) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


class BrainLogValidator:
    """Strict schema validator for .brainlog files."""

    @staticmethod
    def validate(filepath_or_dict: Union[str, Path, dict]) -> Tuple[bool, str]:
        if isinstance(filepath_or_dict, (str, Path)):
            p = Path(filepath_or_dict)
            if not p.exists():
                return False, f"File does not exist: {p}"
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                return False, f"JSON parse failure: {e}"
        else:
            data = filepath_or_dict

        version = data.get("brainlog_version")
        if version not in ["1.0", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"]:
            return False, f"Missing, incompatible or unsupported brainlog_version '{version}' (expected '1.0', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', or '2.7')"

        if "metadata" not in data or not isinstance(data["metadata"], dict):
            return False, "Missing or invalid 'metadata' object"

        if "topology" not in data or not isinstance(data["topology"], dict):
            return False, "Missing or invalid 'topology' object"

        topo = data["topology"]
        if "nodes" not in topo or not isinstance(topo["nodes"], list):
            return False, "Topology missing 'nodes' list"
        if "edges" not in topo or not isinstance(topo["edges"], list):
            return False, "Topology missing 'edges' list"

        if "frames" not in data or not isinstance(data["frames"], list) or len(data["frames"]) == 0:
            return False, "Missing or empty 'frames' sequence"

        node_count = len(topo["nodes"])
        f0 = data["frames"][0]
        for field_name in ["step", "time_ms", "inputs", "activations", "states", "output", "target"]:
            if field_name not in f0:
                return False, f"Frame 0 missing required field '{field_name}'"

        if len(f0["activations"]) != node_count:
            return (
                False,
                f"Activation length mismatch: {len(f0['activations'])} vs {node_count} nodes in topology",
            )

        if version in ["2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"]:
            if "events" not in data or not isinstance(data["events"], list):
                return False, f"v{version} brainlog requires 'events' list"

        return True, "Valid .brainlog trace"


class BrainLogSerializer:
    """Helper to save and load .brainlog archives."""

    @staticmethod
    def save(
        filepath: Union[str, Path],
        metadata: Dict[str, Any],
        topology: BrainTopology,
        frames: List[TelemetryFrame],
        events: List[ComputationalEvent],
        snapshots: Dict[int, dict],
        version: str = "2.7",
    ) -> Path:
        blog = BrainLog(
            metadata=metadata,
            topology=topology,
            frames=frames,
            events=events,
            snapshots=snapshots,
            version=version,
        )
        return blog.save(filepath)

    @staticmethod
    def load(filepath: Union[str, Path]) -> dict:
        path = Path(filepath)
        valid, msg = BrainLogValidator.validate(path)
        if not valid:
            raise ValueError(f"BrainLog validation error in {path.name}: {msg}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
