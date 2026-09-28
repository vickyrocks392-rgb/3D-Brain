"""Static topology extraction and spatial 3D graph construction for 3D-Brain."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from brain.base import BrainModel


@dataclass
class TopologyNode:
    id: int
    type: str  # "INPUT", "RECURRENT", "OUTPUT"
    label: str
    pos: List[float]  # [x, y, z]
    unit_index: int  # index inside model hidden layer, or -1 for I/O
    module_id: int = 0
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "pos": [round(float(p), 3) for p in self.pos],
            "unit_index": self.unit_index,
            "module_id": int(self.module_id),
            "metadata": self.metadata,
        }
        return d


@dataclass
class TopologyEdge:
    src: int
    dst: int
    weight: float
    latency_steps: int = 1
    distance: Optional[float] = None
    propagation_delay: Optional[int] = None

    def to_dict(self) -> dict:
        d = {
            "src": self.src,
            "dst": self.dst,
            "weight": round(float(self.weight), 4),
            "latency_steps": self.latency_steps,
        }
        if self.distance is not None:
            d["distance"] = round(float(self.distance), 3)
        if self.propagation_delay is not None:
            d["propagation_delay"] = int(self.propagation_delay)
        return d


@dataclass
class BrainTopology:
    topology_id: str
    model_id: str
    nodes: List[TopologyNode]
    edges: List[TopologyEdge]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict:
        return {
            "topology_id": self.topology_id,
            "model_id": self.model_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


def generate_fibonacci_ellipsoid(
    n: int, rx: float = 3.5, ry: float = 2.5, rz: float = 2.5
) -> List[List[float]]:
    """Distribute n points deterministically over an ellipsoid surface using the golden ratio."""
    points = []
    phi = (1.0 + np.sqrt(5.0)) / 2.0
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


class TopologyExtractor:
    """Extracts static 3D network topology from a computational substrate."""

    def __init__(self, edge_weight_threshold: float = 0.04):
        self.edge_weight_threshold = edge_weight_threshold

    def extract(self, model: BrainModel, model_id: str) -> BrainTopology:
        nodes: List[TopologyNode] = []
        edges: List[TopologyEdge] = []

        # 1. Sensory Input Cluster (Z = -8.0)
        in_dim = model.input_dim
        if in_dim == 3:
            nodes.append(TopologyNode(id=0, type="INPUT", label="Stimulus_Left", pos=[-3.5, 0.0, -8.0], unit_index=-1, module_id=0))
            nodes.append(TopologyNode(id=1, type="INPUT", label="Stimulus_Right", pos=[3.5, 0.0, -8.0], unit_index=-1, module_id=1))
            nodes.append(TopologyNode(id=2, type="INPUT", label="Phase_Gate", pos=[0.0, 0.0, -8.0], unit_index=-1, module_id=0))
            recurrent_id_offset = 3
        elif in_dim == 2:
            nodes.append(TopologyNode(id=0, type="INPUT", label="Stimulus_Input", pos=[-2.5, 0.0, -8.0], unit_index=-1, module_id=0))
            nodes.append(TopologyNode(id=1, type="INPUT", label="Phase_Gate", pos=[2.5, 0.0, -8.0], unit_index=-1, module_id=0))
            recurrent_id_offset = 2
        else:
            # Multi-channel sensory inputs (e.g. 8-channel embodied environment)
            recurrent_id_offset = in_dim
            for j in range(in_dim):
                x_pos = -4.0 + (8.0 * j / max(1, in_dim - 1))
                nodes.append(TopologyNode(id=j, type="INPUT", label=f"Sensor_{j}", pos=[round(x_pos, 3), 0.0, -8.0], unit_index=-1, module_id=j % 4))

        # 2. Recurrent Memory Population
        H = model.hidden_dim
        if hasattr(model, "positions") and model.positions is not None:
            positions = [[round(float(p), 3) for p in pos] for pos in model.positions]
            module_ids = getattr(model, "module_ids", None)
        else:
            positions = generate_fibonacci_ellipsoid(H, rx=3.5, ry=2.5, rz=2.5)
            module_ids = None

        # Extract static parameters per unit if available (e.g. tau or alpha)
        weights = model.get_weights()
        unit_meta = [{} for _ in range(H)]
        neuron_types = getattr(model, "neuron_types", None)
        if neuron_types is not None:
            for i in range(min(H, len(neuron_types))):
                unit_meta[i]["neuron_type"] = str(neuron_types[i])
        if "rho" in weights:
            from brain.liquid import sigmoid
            alphas = sigmoid(weights["rho"])
            for i in range(H):
                unit_meta[i]["initial_alpha"] = round(float(alphas[i]), 4)
                unit_meta[i]["initial_tau"] = round(float(1.0 / (alphas[i] + 1e-7)), 2)

        for i in range(H):
            mod_id = int(module_ids[i]) if module_ids is not None else 0
            n_type_label = f"[{unit_meta[i].get('neuron_type', 'E')[:1]}] " if "neuron_type" in unit_meta[i] else ""
            nodes.append(
                TopologyNode(
                    id=recurrent_id_offset + i,
                    type="RECURRENT",
                    label=f"{n_type_label}Unit_{i:02d}",
                    pos=positions[i],
                    unit_index=i,
                    module_id=mod_id,
                    metadata=unit_meta[i],
                )
            )

        # 3. Decision / Predictive Readout Cluster (Z = +8.0)
        out_dim = getattr(model, "output_dim", 1)
        output_start_id = recurrent_id_offset + H
        if out_dim == 1:
            nodes.append(
                TopologyNode(
                    id=output_start_id,
                    type="OUTPUT",
                    label="Decision_Readout",
                    pos=[0.0, 0.0, 8.0],
                    unit_index=-1,
                    module_id=0,
                )
            )
        else:
            for k in range(out_dim):
                x_pos = -4.0 + (8.0 * k / max(1, out_dim - 1))
                nodes.append(
                    TopologyNode(
                        id=output_start_id + k,
                        type="OUTPUT",
                        label=f"Pred_{k}",
                        pos=[round(x_pos, 3), 0.0, 8.0],
                        unit_index=-1,
                        module_id=k % 4,
                    )
                )

        # Extract connections
        W_xh = weights.get("W_xh")
        if W_xh is None and "W_in" in weights:
            W_xh = weights["W_in"]
        elif W_xh is None and "W_xz" in weights:
            W_xh = weights["W_xz"]

        W_hh = weights.get("W_hh")
        if W_hh is None and "weights" in weights:
            W_hh = weights["weights"]
        elif W_hh is None and "W_hz" in weights and getattr(weights["W_hz"], "shape", None) == (H, H):
            W_hh = weights["W_hz"]

        W_hy = weights.get("W_hy")
        if W_hy is None and "W_pred" in weights:
            # Predictive head: weights from hidden to output
            W_hy = weights["W_pred"][:H, :]
        elif W_hy is None and "W_out" in weights:
            W_hy = weights["W_out"]

        dist_mat = getattr(model, "distance_matrix", None)
        delay_mat = getattr(model, "delay_matrix", None)

        # Input -> Hidden edges
        if W_xh is not None:
            for in_j in range(model.input_dim):
                p_in = np.array(nodes[in_j].pos, dtype=np.float32)
                for h_i in range(H):
                    if W_xh.shape == (model.input_dim, H):
                        w = float(W_xh[in_j, h_i])
                    elif W_xh.shape == (H, model.input_dim):
                        w = float(W_xh[h_i, in_j])
                    else:
                        continue
                    if abs(w) >= self.edge_weight_threshold:
                        p_h = np.array(nodes[recurrent_id_offset + h_i].pos, dtype=np.float32)
                        d = float(np.linalg.norm(p_in - p_h))
                        edges.append(
                            TopologyEdge(
                                src=in_j,
                                dst=recurrent_id_offset + h_i,
                                weight=w,
                                latency_steps=1,
                                distance=d,
                                propagation_delay=1,
                            )
                        )

        # Recurrent Hidden -> Hidden edges
        if W_hh is not None:
            for src_k in range(H):
                for dst_i in range(H):
                    w = float(W_hh[src_k, dst_i])
                    if abs(w) >= self.edge_weight_threshold:
                        if dist_mat is not None and delay_mat is not None:
                            d = float(dist_mat[src_k, dst_i])
                            lat = int(delay_mat[src_k, dst_i])
                        else:
                            p_src = np.array(positions[src_k], dtype=np.float32)
                            p_dst = np.array(positions[dst_i], dtype=np.float32)
                            d = float(np.linalg.norm(p_src - p_dst))
                            lat = 1
                        edges.append(
                            TopologyEdge(
                                src=recurrent_id_offset + src_k,
                                dst=recurrent_id_offset + dst_i,
                                weight=w,
                                latency_steps=lat,
                                distance=d,
                                propagation_delay=lat,
                            )
                        )

        # Hidden -> Output edges
        if W_hy is not None:
            for out_k in range(out_dim):
                out_node_id = output_start_id + out_k
                p_out = np.array(nodes[out_node_id].pos, dtype=np.float32)
                for h_i in range(H):
                    if W_hy.shape == (H, out_dim):
                        w = float(W_hy[h_i, out_k])
                    elif W_hy.shape == (out_dim, H):
                        w = float(W_hy[out_k, h_i])
                    elif W_hy.ndim == 1 and out_k == 0:
                        w = float(W_hy[h_i])
                    else:
                        continue
                    if abs(w) >= self.edge_weight_threshold:
                        p_h = np.array(nodes[recurrent_id_offset + h_i].pos, dtype=np.float32)
                        d = float(np.linalg.norm(p_h - p_out))
                        edges.append(
                            TopologyEdge(
                                src=recurrent_id_offset + h_i,
                                dst=out_node_id,
                                weight=w,
                                latency_steps=1,
                                distance=d,
                                propagation_delay=1,
                            )
                        )

        topology_id = f"{model_id}_{H}units"
        return BrainTopology(
            topology_id=topology_id,
            model_id=model_id,
            nodes=nodes,
            edges=edges,
        )
