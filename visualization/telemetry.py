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
    if hasattr(model, "positions") and model.positions is not None:
        hidden_positions = [[round(float(p), 3) for p in pos] for pos in model.positions]
        module_ids = getattr(model, "module_ids", None)
    else:
        hidden_positions = generate_fibonacci_ellipsoid(H, rx=3.5, ry=2.5, rz=2.5)
        module_ids = None

    neuron_types = getattr(model, "neuron_types", None)
    for i in range(H):
        n_type = str(neuron_types[i]) if (neuron_types is not None and i < len(neuron_types)) else "EXCITATORY"
        n_label = f"[{n_type[:1]}] Unit_{i:02d}" if neuron_types is not None else f"Unit_{i:02d}"
        node_d = {
            "id": 2 + i,
            "type": "RECURRENT",
            "label": n_label,
            "pos": hidden_positions[i],
            "metadata": {"neuron_type": n_type},
        }
        if module_ids is not None:
            node_d["module_id"] = int(module_ids[i])
        nodes.append(node_d)

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

    dist_mat = getattr(model, "distance_matrix", None)
    delay_mat = getattr(model, "delay_matrix", None)

    # Add input -> hidden edges
    if W_xh is not None:
        for in_j in range(model.input_dim):
            for h_i in range(H):
                w = float(W_xh[in_j, h_i])
                if abs(w) >= edge_weight_threshold:
                    edge_d = {
                        "src": in_j,
                        "dst": 2 + h_i,
                        "weight": round(w, 4),
                        "latency_steps": 1,
                    }
                    edges.append(edge_d)

    # Add recurrent hidden -> hidden edges
    if W_hh is not None:
        for src_k in range(H):
            for dst_i in range(H):
                w = float(W_hh[src_k, dst_i])
                if abs(w) >= edge_weight_threshold:
                    edge_d = {
                        "src": 2 + src_k,
                        "dst": 2 + dst_i,
                        "weight": round(w, 4),
                        "latency_steps": 1,
                    }
                    if dist_mat is not None and delay_mat is not None:
                        edge_d["distance"] = round(float(dist_mat[src_k, dst_i]), 3)
                        edge_d["propagation_delay"] = int(delay_mat[src_k, dst_i])
                        edge_d["latency_steps"] = int(delay_mat[src_k, dst_i])
                    edges.append(edge_d)

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
    from observability.brainlog import BrainLogValidator
    return BrainLogValidator.validate(filepath_or_dict)

