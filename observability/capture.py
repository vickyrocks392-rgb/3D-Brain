"""Passive state capture and telemetry frame construction for 3D-Brain."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from brain.base import BrainModel
from observability.events import ComputationalEvent, EventType
from observability.topology import BrainTopology


@dataclass
class TelemetrySignal:
    src: int
    dst: int
    progress: float
    magnitude: float
    distance: Optional[float] = None
    delay: Optional[int] = None

    def to_dict(self) -> dict:
        d = {
            "src": self.src,
            "dst": self.dst,
            "progress": round(float(self.progress), 2),
            "mag": round(float(self.magnitude), 4),
        }
        if self.distance is not None:
            d["distance"] = round(float(self.distance), 2)
        if self.delay is not None:
            d["delay"] = int(self.delay)
        return d


@dataclass
class DynamicNodeState:
    unit_id: int
    activation: float
    state: float
    dynamic_vars: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "unit_id": self.unit_id,
            "act": round(float(self.activation), 4),
            "state": round(float(self.state), 4),
        }
        if self.dynamic_vars:
            d["vars"] = {k: round(float(v), 4) for k, v in self.dynamic_vars.items()}
        return d


@dataclass
class TelemetryFrame:
    step: int
    time_ms: float
    inputs: List[float]
    activations: List[float]
    states: List[float]
    output: float
    target: float
    error: float
    mask: float
    signals: List[TelemetrySignal] = field(default_factory=list)
    events: List[ComputationalEvent] = field(default_factory=list)
    dynamic_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "time_ms": round(float(self.time_ms), 1),
            "inputs": [round(float(x), 4) for x in self.inputs],
            "activations": [round(float(a), 4) for a in self.activations],
            "states": [round(float(s), 4) for s in self.states],
            "output": round(float(self.output), 4),
            "target": round(float(self.target), 4),
            "error": round(float(self.error), 4),
            "mask": round(float(self.mask), 1),
            "signals": [s.to_dict() for s in self.signals],
            "events": [e.to_dict() for e in self.events],
            "dynamic_details": self.dynamic_details,
        }


class StateCapture:
    """Passive state capture engine observing computational substrates without side effects."""

    def __init__(self, topology: BrainTopology, dt_ms: float = 40.0, max_active_signals: int = 40):
        self.topology = topology
        self.dt_ms = dt_ms
        self.max_active_signals = max_active_signals
        self.frames: List[TelemetryFrame] = []
        self.all_events: List[ComputationalEvent] = []
        self.snapshots: Dict[int, dict] = {}

    def capture_step(
        self,
        step: int,
        input_vec: np.ndarray,
        model: BrainModel,
        y_val: float,
        target_val: float,
        mask_val: float,
        phase_label: str = "RUNNING",
    ) -> TelemetryFrame:
        """Capture a single computational step and generate verifiable events."""
        time_ms = step * self.dt_ms
        obs = model.observable_state()

        h_vec = obs["state"][0]
        a_vec = obs["activation"][0]

        # Construct full node array: [in_0, in_1, h_0 ... h_{H-1}, out_0]
        in_acts = [float(x) for x in input_vec]
        h_acts = [float(a) for a in a_vec]
        out_act = [float(y_val)]
        all_acts = in_acts + h_acts + out_act

        in_states = [float(x) for x in input_vec]
        h_states = [float(h) for h in h_vec]
        all_states = in_states + h_states + out_act

        err = abs(float(y_val) - float(target_val)) if mask_val > 0 else 0.0

        # Calculate actual signal pulses across connection edges
        signals: List[TelemetrySignal] = []
        if hasattr(model, "queue") and model.queue is not None and len(model.queue.active_signals) > 0:
            offset = 3 if model.input_dim == 3 else 2
            for sig in model.queue.active_signals:
                s_id = offset + sig.source
                d_id = offset + sig.target
                prog = sig.progress(step)
                signals.append(
                    TelemetrySignal(
                        src=s_id,
                        dst=d_id,
                        progress=prog,
                        magnitude=sig.magnitude,
                        distance=sig.distance,
                        delay=sig.delay,
                    )
                )
        else:
            for edge in self.topology.edges:
                src = edge.src
                dst = edge.dst
                w = edge.weight
                src_act = all_acts[src]
                signal_mag = src_act * w
                if abs(signal_mag) >= 0.07:
                    signals.append(
                        TelemetrySignal(
                            src=src,
                            dst=dst,
                            progress=1.0,
                            magnitude=signal_mag,
                            distance=edge.distance,
                            delay=edge.propagation_delay or edge.latency_steps,
                        )
                    )

        # Cap active signals for serialization performance
        signals = signals[: self.max_active_signals]

        # Computational Event Detection
        step_events: List[ComputationalEvent] = []

        if step == 0:
            step_events.append(
                ComputationalEvent(
                    event_type=EventType.TRIAL_STARTED,
                    step=step,
                    time_ms=time_ms,
                    payload={"stimulus": float(input_vec[0])},
                )
            )
            step_events.append(
                ComputationalEvent(
                    event_type=EventType.INPUT_PRESENTED,
                    step=step,
                    time_ms=time_ms,
                    node_id=0,
                    magnitude=float(input_vec[0]),
                    payload={"phase": "CUE"},
                )
            )
        elif phase_label == "DELAY_START":
            step_events.append(
                ComputationalEvent(
                    event_type=EventType.MEMORY_RETAINED,
                    step=step,
                    time_ms=time_ms,
                    payload={"phase": "DELAY"},
                )
            )
        elif phase_label == "PROBE_START":
            step_events.append(
                ComputationalEvent(
                    event_type=EventType.INPUT_PRESENTED,
                    step=step,
                    time_ms=time_ms,
                    node_id=0,
                    magnitude=float(input_vec[0]),
                    payload={"phase": "PROBE", "probe_flag": float(input_vec[1])},
                )
            )
        elif phase_label.startswith("ENV_SWITCH"):
            step_events.append(
                ComputationalEvent(
                    event_type=EventType.ENVIRONMENT_CHANGE,
                    step=step,
                    time_ms=time_ms,
                    magnitude=1.0,
                    payload={"label": phase_label},
                )
            )

        if mask_val > 0:
            step_events.append(
                ComputationalEvent(
                    event_type=EventType.OUTPUT_COMPUTED,
                    step=step,
                    time_ms=time_ms,
                    node_id=self.topology.node_count - 1,
                    magnitude=float(y_val),
                    payload={"target": float(target_val), "error": float(err)},
                )
            )
            if err >= 0.4:
                step_events.append(
                    ComputationalEvent(
                        event_type=EventType.PREDICTION_ERROR,
                        step=step,
                        time_ms=time_ms,
                        node_id=self.topology.node_count - 1,
                        magnitude=float(err),
                    )
                )

        # Extract dynamic variable details for inspection
        dynamic_details = {}
        if "alpha" in obs:
            dynamic_details["alpha"] = [round(float(v), 4) for v in obs["alpha"]]
        if "tau" in obs:
            dynamic_details["tau"] = [round(float(v), 2) for v in obs["tau"]]
        if "gate_z" in obs:
            dynamic_details["gate_z"] = [round(float(v), 4) for v in obs["gate_z"][0]]
        if "gate_r" in obs:
            dynamic_details["gate_r"] = [round(float(v), 4) for v in obs["gate_r"][0]]
        if "candidate_state" in obs:
            dynamic_details["cand_h"] = [round(float(v), 4) for v in obs["candidate_state"][0]]
        if "cand_h" in obs:
            dynamic_details["cand_h"] = [round(float(v), 4) for v in obs["cand_h"][0]]
        if "previous_state" in obs:
            dynamic_details["prev_h"] = [round(float(v), 4) for v in obs["previous_state"][0]]
        if "prev_h" in obs:
            dynamic_details["prev_h"] = [round(float(v), 4) for v in obs["prev_h"][0]]
        if "mean_abs_dw" in obs:
            mean_abs_dw = float(obs["mean_abs_dw"][0])
            dynamic_details["mean_abs_dw"] = round(mean_abs_dw, 6)
            if "max_dw" in obs:
                dynamic_details["max_dw"] = round(float(obs["max_dw"][0]), 6)
            if "weight_norm" in obs:
                dynamic_details["weight_norm"] = round(float(obs["weight_norm"][0]), 4)
            if mean_abs_dw > 1e-5:
                step_events.append(
                    ComputationalEvent(
                        event_type=EventType.PLASTICITY_UPDATE,
                        step=step,
                        time_ms=time_ms,
                        magnitude=mean_abs_dw,
                        payload={
                            "max_dw": float(obs.get("max_dw", [0])[0]),
                            "weight_norm": float(obs.get("weight_norm", [0])[0]),
                        },
                    )
                )

        frame = TelemetryFrame(
            step=step,
            time_ms=time_ms,
            inputs=in_acts,
            activations=all_acts,
            states=all_states,
            output=float(y_val),
            target=float(target_val),
            error=float(err),
            mask=float(mask_val),
            signals=signals,
            events=step_events,
            dynamic_details=dynamic_details,
        )

        self.frames.append(frame)
        self.all_events.extend(step_events)

        # Snapshot full state every 10 steps for fast bidirectional scrubbing
        if step % 10 == 0:
            self.snapshots[step] = {
                "step": step,
                "states": all_states,
                "activations": all_acts,
                "weights_digest": len(self.topology.edges),
            }

        return frame

    def capture_embodied_step(
        self,
        step: int,
        obs_t: np.ndarray,
        action: int,
        obs_next: np.ndarray,
        pred_s: np.ndarray,
        pred_error: float,
        model: BrainModel,
        env_info: Optional[Dict[str, Any]] = None,
        structural_events: Optional[List[Any]] = None,
    ) -> TelemetryFrame:
        """Capture a single embodied sensorimotor step with forward predictions and structural plasticity."""
        time_ms = step * self.dt_ms
        obs = model.observable_state()

        h_vec = obs["state"][0]
        a_vec = obs["activation"][0]

        in_acts = [float(x) for x in obs_t]
        h_acts = [float(a) for a in a_vec]
        out_acts = [float(p) for p in pred_s]
        all_acts = in_acts + h_acts + out_acts

        in_states = [float(x) for x in obs_t]
        h_states = [float(h) for h in h_vec]
        all_states = in_states + h_states + out_acts

        # Signals actively propagating along synaptic connections
        signals: List[TelemetrySignal] = []
        if hasattr(model, "queue") and model.queue is not None and len(model.queue.active_signals) > 0:
            offset = len(in_acts)
            for sig in model.queue.active_signals:
                s_id = offset + sig.source
                d_id = offset + sig.target
                prog = sig.progress(step)
                signals.append(
                    TelemetrySignal(
                        src=s_id,
                        dst=d_id,
                        progress=prog,
                        magnitude=sig.magnitude,
                        distance=sig.distance,
                        delay=sig.delay,
                    )
                )
        signals = signals[: self.max_active_signals]

        # Computational Events
        step_events: List[ComputationalEvent] = []

        # 1. Action Event
        step_events.append(
            ComputationalEvent(
                event_type=EventType.ACTION_SELECTED,
                step=step,
                time_ms=time_ms,
                magnitude=float(action),
                payload={"action": int(action)},
            )
        )

        # 2. Prediction Event
        step_events.append(
            ComputationalEvent(
                event_type=EventType.PREDICTION_MADE,
                step=step,
                time_ms=time_ms,
                magnitude=float(pred_error),
                payload={
                    "pred_s": [round(float(p), 4) for p in pred_s],
                    "obs_next": [round(float(s), 4) for s in obs_next],
                    "pred_error": round(float(pred_error), 4),
                },
            )
        )

        if pred_error >= 0.25:
            step_events.append(
                ComputationalEvent(
                    event_type=EventType.PREDICTION_ERROR,
                    step=step,
                    time_ms=time_ms,
                    magnitude=float(pred_error),
                    payload={"pred_error": round(float(pred_error), 4)},
                )
            )

        # 3. Environment Change Event if env_id switched
        if env_info and "env_switched" in env_info and env_info["env_switched"]:
            step_events.append(
                ComputationalEvent(
                    event_type=EventType.ENVIRONMENT_CHANGE,
                    step=step,
                    time_ms=time_ms,
                    magnitude=1.0,
                    payload={"env_id": env_info.get("env_id", "")},
                )
            )

        # 4. Structural Plasticity Events
        if structural_events:
            for se in structural_events:
                se_dict = se.to_dict() if hasattr(se, "to_dict") else se
                evt_type_str = se_dict.get("type", "")
                if evt_type_str == "SYNAPSE_FORMED":
                    evt_type = EventType.SYNAPSE_FORMED
                elif evt_type_str == "SYNAPSE_PRUNED":
                    evt_type = EventType.SYNAPSE_PRUNED
                elif evt_type_str == "NEURON_BORN":
                    evt_type = EventType.NEURON_BORN
                else:
                    evt_type = EventType.PLASTICITY_UPDATE

                offset = len(in_acts)
                s_node = offset + se_dict["src"] if "src" in se_dict else None
                d_node = offset + se_dict["dst"] if "dst" in se_dict else None
                n_node = offset + se_dict["neuron_id"] if "neuron_id" in se_dict else None

                step_events.append(
                    ComputationalEvent(
                        event_type=evt_type,
                        step=step,
                        time_ms=time_ms,
                        node_id=n_node,
                        source_node=s_node,
                        target_node=d_node,
                        magnitude=se_dict.get("weight", 1.0),
                        payload=se_dict,
                    )
                )

        # Dynamic Details
        dynamic_details: Dict[str, Any] = {
            "pred_mse": round(float(pred_error), 6),
            "pred_s": [round(float(p), 4) for p in pred_s],
            "obs_next": [round(float(s), 4) for s in obs_next],
        }
        if env_info:
            dynamic_details["environment"] = env_info
        if "alpha" in obs:
            dynamic_details["alpha"] = [round(float(v), 4) for v in obs["alpha"]]
        if "tau" in obs:
            dynamic_details["tau"] = [round(float(v), 2) for v in obs["tau"]]
        if "cand_h" in obs:
            dynamic_details["cand_h"] = [round(float(v), 4) for v in obs["cand_h"][0]]
        if "prev_h" in obs:
            dynamic_details["prev_h"] = [round(float(v), 4) for v in obs["prev_h"][0]]
        if structural_events:
            dynamic_details["structural_events"] = [
                se.to_dict() if hasattr(se, "to_dict") else se for se in structural_events
            ]

        frame = TelemetryFrame(
            step=step,
            time_ms=time_ms,
            inputs=in_acts,
            activations=all_acts,
            states=all_states,
            output=float(pred_s[0]),
            target=float(obs_next[0]),
            error=float(pred_error),
            mask=1.0,
            signals=signals,
            events=step_events,
            dynamic_details=dynamic_details,
        )

        self.frames.append(frame)
        self.all_events.extend(step_events)

        if step % 10 == 0:
            self.snapshots[step] = {
                "step": step,
                "states": all_states,
                "activations": all_acts,
                "weights_digest": len(self.topology.edges),
            }

        return frame

    def complete_trial(self, final_acc: float) -> None:
        """Mark trial completion and emit final event."""
        if len(self.frames) > 0:
            last_frame = self.frames[-1]
            completion_event = ComputationalEvent(
                event_type=EventType.TRIAL_COMPLETED,
                step=last_frame.step,
                time_ms=last_frame.time_ms,
                payload={"test_accuracy": round(float(final_acc), 4)},
            )
            last_frame.events.append(completion_event)
            self.all_events.append(completion_event)

