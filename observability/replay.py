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
    module_id: int = 0


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
    initial_weight: Optional[float] = None
    current_weight: Optional[float] = None
    delta_weight: Optional[float] = None
    recent_dw: Optional[float] = None
    distance: Optional[float] = None
    propagation_delay: Optional[int] = None
    signal_progress: Optional[float] = None



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
            module_id=node_info.get("module_id", 0),
        )

    def get_plasticity_events(self) -> List[dict]:
        """Return all events related to synaptic plasticity and weight updates."""
        all_events = self.get_all_events()
        plasticity_types = {
            "PLASTICITY_UPDATE",
            "WEIGHT_INCREASE",
            "WEIGHT_DECREASE",
            "MODULATION_ON",
            "MODULATION_OFF",
            "ENVIRONMENT_CHANGE",
            "TASK_SWITCH",
        }
        return [e for e in all_events if e.get("type") in plasticity_types]

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
        base_w = float(e["weight"])
        
        # Check dynamic weight details if present in frame
        curr_w = base_w
        dyn_details = frame.get("dynamic_details", {})
        recent_dw = None
        if "mean_abs_dw" in dyn_details:
            recent_dw = float(dyn_details["mean_abs_dw"])

        signal_mag = src_act * curr_w
        is_active = abs(signal_mag) >= 0.07

        # Check in-transit signal on this edge
        sig_progress = None
        for s in frame.get("signals", []):
            if s.get("src") == src and s.get("dst") == dst:
                sig_progress = s.get("progress", 1.0)
                signal_mag = s.get("mag", signal_mag)
                break

        return EdgeInspection(
            src=src,
            dst=dst,
            weight=round(curr_w, 4),
            src_label=nodes[src]["label"],
            dst_label=nodes[dst]["label"],
            current_src_activation=src_act,
            current_signal_magnitude=round(float(signal_mag), 4),
            is_active=is_active,
            initial_weight=round(base_w, 4),
            current_weight=round(curr_w, 4),
            delta_weight=round(curr_w - base_w, 6),
            recent_dw=recent_dw,
            distance=e.get("distance"),
            propagation_delay=e.get("propagation_delay", e.get("latency_steps", 1)),
            signal_progress=sig_progress,
        )

    def get_signals_in_transit(self, step: Optional[int] = None) -> List[dict]:
        """Return all signals actively propagating along connections at step."""
        if not self.data:
            return []
        s = self.current_step if step is None else step
        frame = self.data["frames"][s]
        return frame.get("signals", [])

    def get_spatial_metrics(self, step: Optional[int] = None) -> dict:
        """Return spatial dynamic metrics at given step (modularity, synchrony, etc)."""
        if not self.data:
            return {}
        s = self.current_step if step is None else step
        frame = self.data["frames"][s]
        return frame.get("dynamic_details", {})

    def get_environment_state(self, step: Optional[int] = None) -> Optional[dict]:
        """Return embodied environment state at step (agent_pos, heading, env_id, hidden_cycle)."""
        if not self.data:
            return None
        s = self.current_step if step is None else step
        frame = self.data["frames"][s]
        return frame.get("dynamic_details", {}).get("environment")

    def get_action(self, step: Optional[int] = None) -> Optional[int]:
        """Return action executed at step."""
        env_state = self.get_environment_state(step)
        if env_state and "action" in env_state:
            return int(env_state["action"])
        events = self.get_events_at_step(step)
        for e in events:
            if e.get("type") == "ACTION_SELECTED":
                return int(e.get("payload", {}).get("action", e.get("magnitude", 0)))
        return None

    def get_prediction(self, step: Optional[int] = None) -> Optional[dict]:
        """Return prediction details at step: pred_s, obs_next, pred_mse."""
        if not self.data:
            return None
        s = self.current_step if step is None else step
        frame = self.data["frames"][s]
        dyn = frame.get("dynamic_details", {})
        if "pred_s" in dyn and "obs_next" in dyn:
            return {
                "pred_s": dyn["pred_s"],
                "obs_next": dyn["obs_next"],
                "pred_mse": dyn.get("pred_mse", frame.get("error", 0.0)),
            }
        return None

    def get_structural_events(self, step: Optional[int] = None) -> List[dict]:
        """Return structural plasticity events (SYNAPSE_FORMED, SYNAPSE_PRUNED, NEURON_BORN)."""
        struct_types = {"SYNAPSE_FORMED", "SYNAPSE_PRUNED", "NEURON_BORN", "NEURON_PRUNED"}
        if step is not None:
            return [e for e in self.get_events_at_step(step) if e.get("type") in struct_types]
        all_events = self.get_all_events()
        return [e for e in all_events if e.get("type") in struct_types]

    def get_active_topology_at_step(self, step: Optional[int] = None) -> dict:
        """Compute active topology accounting for synaptogenesis and pruning up to step."""
        if not self.data:
            return {}
        target_step = self.current_step if step is None else step
        base_topo = self.data["topology"]
        nodes = [dict(n) for n in base_topo["nodes"]]
        edges = {(e["src"], e["dst"]): dict(e) for e in base_topo["edges"]}

        all_events = self.get_all_events()
        for evt in all_events:
            if evt.get("step", 0) > target_step:
                break
            etype = evt.get("type")
            payload = evt.get("payload", {})
            if etype == "SYNAPSE_FORMED":
                src = payload.get("src")
                dst = payload.get("dst")
                if src is not None and dst is not None:
                    edges[(src, dst)] = {
                        "src": src,
                        "dst": dst,
                        "weight": payload.get("weight", 0.1),
                        "distance": payload.get("distance", 1.0),
                        "propagation_delay": payload.get("delay", 1),
                    }
            elif etype == "SYNAPSE_PRUNED":
                src = payload.get("src")
                dst = payload.get("dst")
                if (src, dst) in edges:
                    del edges[(src, dst)]
            elif etype == "NEURON_BORN":
                nid = payload.get("neuron_id")
                pos = payload.get("pos", [0.0, 0.0, 0.0])
                if nid is not None and nid >= len(nodes):
                    nodes.append({
                        "id": nid,
                        "type": "RECURRENT",
                        "label": f"Unit_{nid:02d}",
                        "pos": pos,
                        "unit_index": nid,
                        "module_id": 0,
                    })

        return {
            "step": target_step,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": list(edges.values()),
        }

    def get_latent_state(self, step: Optional[int] = None) -> Optional[dict]:
        """Return latent state vector z_t, norm, and persistence metrics at step."""
        if not self.data:
            return None
        s = self.current_step if step is None else step
        frame = self.data["frames"][s]
        dyn = frame.get("dynamic_details", {})
        if "latent_z" in dyn:
            return {
                "step": s,
                "latent_z": dyn["latent_z"],
                "latent_norm": dyn.get("latent_norm", 0.0),
                "context_id": dyn.get("context_id"),
            }
        return None

    def get_multistep_predictions(self, step: Optional[int] = None) -> Optional[dict]:
        """Return multi-step predictive forecasts and errors at step."""
        if not self.data:
            return None
        s = self.current_step if step is None else step
        frame = self.data["frames"][s]
        dyn = frame.get("dynamic_details", {})
        return {
            "step": s,
            "multistep_preds": dyn.get("multistep_preds", {}),
            "multistep_mse": dyn.get("multistep_mse", {}),
        }

    def compare_histories(self, step_a: int, step_b: int) -> dict:
        """Compare internal states at two timesteps producing identical or similar current observations.

        Demonstrates whether the internal latent state encodes historical context
        that is unavailable from the instantaneous sensory input alone.
        """
        if not self.data:
            raise RuntimeError("No .brainlog loaded in ReplayEngine")
        if step_a >= self.total_steps or step_b >= self.total_steps:
            raise IndexError("Step index out of bounds")

        import numpy as np

        frame_a = self.data["frames"][step_a]
        frame_b = self.data["frames"][step_b]

        in_a = np.array(frame_a["inputs"], dtype=np.float32)
        in_b = np.array(frame_b["inputs"], dtype=np.float32)
        act_a = np.array(frame_a["activations"], dtype=np.float32)
        act_b = np.array(frame_b["activations"], dtype=np.float32)

        dyn_a = frame_a.get("dynamic_details", {})
        dyn_b = frame_b.get("dynamic_details", {})

        z_a = np.array(dyn_a.get("latent_z", []), dtype=np.float32)
        z_b = np.array(dyn_b.get("latent_z", []), dtype=np.float32)

        input_diff = float(np.linalg.norm(in_a - in_b))
        act_diff = float(np.linalg.norm(act_a - act_b))
        latent_diff = float(np.linalg.norm(z_a - z_b)) if len(z_a) > 0 and len(z_b) > 0 else 0.0

        # Cosine similarity between latent vectors
        cos_sim = 1.0
        if len(z_a) > 0 and len(z_b) > 0:
            denom = float(np.linalg.norm(z_a) * np.linalg.norm(z_b))
            if denom > 1e-6:
                cos_sim = float(np.dot(z_a, z_b) / denom)

        return {
            "step_a": step_a,
            "step_b": step_b,
            "input_distance": input_diff,
            "inputs_identical": bool(input_diff < 1e-4),
            "recurrent_activation_distance": act_diff,
            "latent_state_distance": latent_diff,
            "latent_cosine_similarity": cos_sim,
            "context_disambiguated": bool(input_diff < 1e-3 and latent_diff > 0.05),
            "pred_a": dyn_a.get("pred_s"),
            "pred_b": dyn_b.get("pred_s"),
        }

    # Alias for camelCase callers
    compareHistories = compare_histories

    # Phase 7: Spiking Neural Dynamics & STDP Inspection
    def get_spike_events(self, step: Optional[int] = None) -> List[dict]:
        """Return all spike emitted/arrived events at the given step."""
        if not self.data:
            return []
        s = self.current_step if step is None else step
        frame = self.data["frames"][s]
        return [
            e for e in frame.get("events", [])
            if e.get("type") in ["SPIKE_EMITTED", "SPIKE_ARRIVED"]
        ]

    def get_membrane_states(self, step: Optional[int] = None) -> dict:
        """Return membrane potential states and active spike counts at the given step."""
        if not self.data:
            return {}
        s = self.current_step if step is None else step
        frame = self.data["frames"][s]
        dyn = frame.get("dynamic_details", {})
        return {
            "step": s,
            "states": frame.get("states", []),
            "activations": frame.get("activations", []),
            "v_mem_mean": dyn.get("v_mem_mean"),
            "v_mem_std": dyn.get("v_mem_std"),
            "spikes_active": dyn.get("spikes_active", 0),
        }

    def get_stdp_updates(self, step: Optional[int] = None) -> List[dict]:
        """Return synaptic plasticity STDP events at the given step."""
        if not self.data:
            return []
        s = self.current_step if step is None else step
        frame = self.data["frames"][s]
        return [e for e in frame.get("events", []) if e.get("type") == "STDP_UPDATE"]

    def get_phase_metrics(self, step: Optional[int] = None) -> dict:
        """Return Kuramoto phase order parameter and oscillatory spectrum metrics."""
        if not self.data:
            return {}
        s = self.current_step if step is None else step
        frame = self.data["frames"][s]
        dyn = frame.get("dynamic_details", {})
        return {
            "step": s,
            "kuramoto_r": dyn.get("kuramoto_r"),
            "dominant_freq_hz": dyn.get("dominant_freq_hz"),
        }



