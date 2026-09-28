"""Telemetry emitter, 3D topology generator, and .brainlog serializer/validator.

Strictly implements the telemetry data contract specified in
visualization/VISUALIZATION_CONCEPT_AND_TELEMETRY.md.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from brain.base import BrainModel


def generate_fibonacci_ellipsoid(
    n: int, rx: float = 3.5, ry: float = 2.5, rz: float = 2.5
) -> List[List[float]]:
    """Distribute n points smoothly over an ellipsoid surface using the golden ratio."""
    points = []
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
    for i in range(n):
        theta = 2.0 * np.pi * i / phi
        s = (2.0 * i + 1.0) / n - 1.0
        s = np.clip(s, -1.0, 1.0)
        phi_angle = np.arcsin(s)
        x = rx * np.cos(phi_angle) * np.cos(theta)
        y = ry * np.cos(phi_angle) * np.sin(theta)
        z = rz * np.sin(phi_angle)
        points.append([round(float(x), 3), round(float(y), 3), round(float(z), 3)])
    return points


def build_topology(
    model: BrainModel, model_id: str, edge_weight_threshold: float = 0.05
) -> dict:
    """Generate 3D spatial node coordinates and connection edges from actual model weights."""
    nodes = []
    edges = []

    # 1. Sensory Input Nodes (Z = -8.0)
    nodes.append({
        "id": 0,
        "type": "INPUT",
        "label": "Stimulus_Input",
        "pos": [-2.5, 0.0, -8.0],
    })
    nodes.append({
        "id": 1,
        "type": "INPUT",
        "label": "Phase_Gate",
        "pos": [2.5, 0.0, -8.0],
    })

    # 2. Recurrent Memory Nodes (Z in [-2.5, 2.5])
    H = model.hidden_dim
    hidden_positions = generate_fibonacci_ellipsoid(H, rx=3.5, ry=2.5, rz=2.5)
    for i in range(H):
        nodes.append({
            "id": 2 + i,
            "type": "RECURRENT",
            "label": f"Unit_{i:02d}",
            "pos": hidden_positions[i],
        })

    # 3. Decision Readout Node (Z = +8.0)
    output_idx = 2 + H
    nodes.append({
        "id": output_idx,
        "type": "OUTPUT",
        "label": "Decision_Readout",
        "pos": [0.0, 0.0, 8.0],
    })

    # Extract actual weights
    weights = model.get_weights()
    W_xh = weights.get("W_xh")
    if W_xh is None and "W_xz" in weights:
        W_xh = weights["W_xz"]  # GRU proxy

    W_hh = weights.get("W_hh")
    if W_hh is None and "W_hz" in weights:
        W_hh = weights["W_hz"]

    W_hy = weights.get("W_hy")

    # Add input -> hidden edges
    if W_xh is not None:
        for in_j in range(model.input_dim):
            for h_i in range(H):
                w = float(W_xh[in_j, h_i])
                if abs(w) >= edge_weight_threshold:
                    edges.append({
                        "src": in_j,
                        "dst": 2 + h_i,
                        "weight": round(w, 4),
                        "latency_steps": 1,
                    })

    # Add recurrent hidden -> hidden edges
    if W_hh is not None:
        for src_k in range(H):
            for dst_i in range(H):
                w = float(W_hh[src_k, dst_i])
                if abs(w) >= edge_weight_threshold:
                    edges.append({
                        "src": 2 + src_k,
                        "dst": 2 + dst_i,
                        "weight": round(w, 4),
                        "latency_steps": 1,
                    })

    # Add hidden -> output edges
    if W_hy is not None:
        for h_i in range(H):
            w = float(W_hy[h_i, 0])
            if abs(w) >= edge_weight_threshold:
                edges.append({
                    "src": 2 + h_i,
                    "dst": output_idx,
                    "weight": round(w, 4),
                    "latency_steps": 1,
                })

    return {
        "topology_id": f"{model_id}_{H}units",
        "model_id": model_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def build_frame(
    step: int,
    time_ms: float,
    input_vec: np.ndarray,
    h_vec: np.ndarray,
    a_vec: np.ndarray,
    y_val: float,
    target_val: float,
    mask_val: float,
    edges: list,
    event: Optional[dict] = None,
) -> dict:
    """Build a single telemetry frame strictly from actual computational outputs."""
    # Concatenate all node activations: [in_0, in_1, h_0 ... h_{H-1}, out_0]
    in_acts = [round(float(x), 4) for x in input_vec]
    h_acts = [round(float(a), 4) for a in a_vec]
    out_act = [round(float(y_val), 4)]
    all_activations = in_acts + h_acts + out_act

    in_states = [round(float(x), 4) for x in input_vec]
    h_states = [round(float(h), 4) for h in h_vec]
    all_states = in_states + h_states + out_act

    err = abs(float(y_val) - float(target_val)) if mask_val > 0 else 0.0

    # Compute signal pulses across active edges
    signals = []
    for edge in edges:
        src = edge["src"]
        dst = edge["dst"]
        w = edge["weight"]
        src_act = all_activations[src]
        signal_mag = src_act * w
        if abs(signal_mag) > 0.08:
            signals.append({
                "src": src,
                "dst": dst,
                "progress": 1.0,
                "mag": round(float(signal_mag), 4),
            })

    frame = {
        "step": int(step),
        "time_ms": round(float(time_ms), 1),
        "inputs": [round(float(x), 4) for x in input_vec],
        "activations": all_activations,
        "states": all_states,
        "output": round(float(y_val), 4),
        "target": round(float(target_val), 4),
        "mask": round(float(mask_val), 1),
        "error": round(float(err), 4),
        "signals": signals[:40],  # Cap active pulses for rendering performance
        "events": [event] if event else [],
    }
    return frame


def export_brainlog(
    filepath: Union[str, Path], metadata: dict, topology: dict, frames: List[dict]
) -> None:
    """Serialize execution trajectory into a validated .brainlog JSON file."""
    data = {
        "brainlog_version": "1.0",
        "metadata": metadata,
        "topology": topology,
        "frames": frames,
    }
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def validate_brainlog(filepath_or_dict: Union[str, Path, dict]) -> Tuple[bool, str]:
    """Validate that a .brainlog file or dictionary conforms strictly to the specification."""
    if isinstance(filepath_or_dict, (str, Path)):
        p = Path(filepath_or_dict)
        if not p.exists():
            return False, f"File not found: {p}"
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return False, f"JSON parse error: {e}"
    else:
        data = filepath_or_dict

    if data.get("brainlog_version") != "1.0":
        return False, "Missing or incompatible 'brainlog_version' (expected '1.0')"

    if "metadata" not in data or not isinstance(data["metadata"], dict):
        return False, "Missing or invalid 'metadata' block"

    if "topology" not in data or not isinstance(data["topology"], dict):
        return False, "Missing or invalid 'topology' block"

    topology = data["topology"]
    if "nodes" not in topology or not isinstance(topology["nodes"], list):
        return False, "Topology missing 'nodes' list"
    if "edges" not in topology or not isinstance(topology["edges"], list):
        return False, "Topology missing 'edges' list"

    if "frames" not in data or not isinstance(data["frames"], list) or len(data["frames"]) == 0:
        return False, "Missing or empty 'frames' sequence"

    first_frame = data["frames"][0]
    required_keys = ["step", "time_ms", "inputs", "activations", "states", "output", "target"]
    for k in required_keys:
        if k not in first_frame:
            return False, f"Frame missing required field '{k}'"

    node_count = len(topology["nodes"])
    if len(first_frame["activations"]) != node_count:
        return (
            False,
            f"Activation count mismatch: frame has {len(first_frame['activations'])} but topology has {node_count} nodes",
        )

    return True, "Valid .brainlog trace"
