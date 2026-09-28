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
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "pos": [round(float(p), 3) for p in self.pos],
            "unit_index": self.unit_index,
            "metadata": self.metadata,
        }


@dataclass
class TopologyEdge:
    src: int
    dst: int
    weight: float
    latency_steps: int = 1

    def to_dict(self) -> dict:
        return {
            "src": self.src,
            "dst": self.dst,
            "weight": round(float(self.weight), 4),
            "latency_steps": self.latency_steps,
        }


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
        nodes.append(
            TopologyNode(
                id=0,
                type="INPUT",
                label="Stimulus_Input",
                pos=[-2.5, 0.0, -8.0],
                unit_index=-1,
            )
        )
        nodes.append(
            TopologyNode(
                id=1,
                type="INPUT",
                label="Phase_Gate",
                pos=[2.5, 0.0, -8.0],
                unit_index=-1,
            )
        )

        # 2. Recurrent Memory Population (Z in [-2.5, 2.5])
        H = model.hidden_dim
        positions = generate_fibonacci_ellipsoid(H, rx=3.5, ry=2.5, rz=2.5)

        # Extract static parameters per unit if available (e.g. tau or alpha)
        weights = model.get_weights()
        unit_meta = [{} for _ in range(H)]
        if "rho" in weights:
            from brain.liquid import sigmoid
            alphas = sigmoid(weights["rho"])
            for i in range(H):
                unit_meta[i]["initial_alpha"] = round(float(alphas[i]), 4)
                unit_meta[i]["initial_tau"] = round(float(1.0 / (alphas[i] + 1e-7)), 2)

        for i in range(H):
            nodes.append(
                TopologyNode(
                    id=2 + i,
                    type="RECURRENT",
                    label=f"Unit_{i:02d}",
                    pos=positions[i],
                    unit_index=i,
                    metadata=unit_meta[i],
                )
            )

        # 3. Decision Readout Cluster (Z = +8.0)
        output_id = 2 + H
        nodes.append(
            TopologyNode(
                id=output_id,
                type="OUTPUT",
                label="Decision_Readout",
                pos=[0.0, 0.0, 8.0],
                unit_index=-1,
            )
        )

        # Extract connections
        W_xh = weights.get("W_xh")
        if W_xh is None and "W_xz" in weights:
            W_xh = weights["W_xz"]

        W_hh = weights.get("W_hh")
        if W_hh is None and "W_hz" in weights:
            W_hh = weights["W_hz"]

        W_hy = weights.get("W_hy")

        # Input -> Hidden edges
        if W_xh is not None:
            for in_j in range(model.input_dim):
                for h_i in range(H):
                    w = float(W_xh[in_j, h_i])
                    if abs(w) >= self.edge_weight_threshold:
                        edges.append(TopologyEdge(src=in_j, dst=2 + h_i, weight=w))

        # Recurrent Hidden -> Hidden edges
        if W_hh is not None:
            for src_k in range(H):
                for dst_i in range(H):
                    w = float(W_hh[src_k, dst_i])
                    if abs(w) >= self.edge_weight_threshold:
                        edges.append(TopologyEdge(src=2 + src_k, dst=2 + dst_i, weight=w))

        # Hidden -> Output edges
        if W_hy is not None:
            for h_i in range(H):
                w = float(W_hy[h_i, 0])
                if abs(w) >= self.edge_weight_threshold:
                    edges.append(TopologyEdge(src=2 + h_i, dst=output_id, weight=w))

        topology_id = f"{model_id}_{H}units"
        return BrainTopology(
            topology_id=topology_id,
            model_id=model_id,
            nodes=nodes,
            edges=edges,
        )
