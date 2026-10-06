"""Tests for Scientific Telemetry Provenance and Fallback Prevention.

Verifies:
A. Missing action_probs must NOT silently become [0.5, 0.5].
B. Missing decision_latency_ms must NOT silently become 11.5.
C. Missing selected_action must NOT silently become 0.
D. Genuine upstream action_probs must survive unchanged.
E. Genuine upstream latency must survive unchanged.
F. Genuine upstream selected_action must survive unchanged.
G. BrainLog metadata indicates whether each telemetry field is:
   GENUINE / GENUINE_MEASURED / DETERMINISTICALLY_DERIVED / UNAVAILABLE / FALLBACK.
H. ZERO-SHOT VERIFIED fails if decision telemetry is fallback-derived or incomplete.
I. Browser UI logic never presents fallback telemetry as genuine scientific output.
J. Explicit regression test using EXP023B verifying genuine computational provenance.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from observability.capture import StateCapture
from observability.topology import BrainTopology, TopologyNode, TopologyEdge
from brain.phase16_network import Phase16Network, ArchitecturePreset


def _dummy_topology() -> BrainTopology:
    return BrainTopology(
        topology_id="test_topology",
        model_id="test_model",
        nodes=[
            TopologyNode(id=0, label="Input", type="INPUT", pos=[0, 0, 0], unit_index=-1),
            TopologyNode(id=1, label="Output", type="OUTPUT", pos=[1, 1, 1], unit_index=-1),
        ],
        edges=[TopologyEdge(src=0, dst=1, weight=0.5)],
    )


class DummyModel:
    def __init__(self, obs_dict=None):
        self._obs = obs_dict if obs_dict is not None else {}

    def observable_state(self):
        return dict(self._obs)


class TestScientificTelemetryProvenance:
    def setup_method(self):
        self.topology = _dummy_topology()
        self.capture = StateCapture(topology=self.topology, dt_ms=40.0)

    def test_a_missing_action_probs_not_silently_fallback(self):
        """A. Missing action_probs must NOT silently become [0.5, 0.5]."""
        model = DummyModel({"selected_action": 1, "decision_latency_ms": 2.5})
        frame = self.capture.capture_phase23b_step(
            step=10,
            input_vec=np.zeros(2),
            model=model,
            is_zero_shot_test=True,
        )
        dyn = frame.dynamic_details
        assert dyn["action_probs"] == [], f"Expected empty action_probs, got: {dyn['action_probs']}"
        assert dyn["confidence"] is None, f"Expected None confidence, got: {dyn['confidence']}"
        assert dyn["telemetry_provenance"]["action_probs"] == "UNAVAILABLE"
        assert dyn["telemetry_provenance"]["confidence"] == "UNAVAILABLE"

    def test_b_missing_decision_latency_ms_not_silently_fallback(self):
        """B. Missing decision_latency_ms must NOT silently become 11.5."""
        model = DummyModel({"selected_action": 1, "action_probs": [0.3, 0.7]})
        frame = self.capture.capture_phase23b_step(
            step=10,
            input_vec=np.zeros(2),
            model=model,
            is_zero_shot_test=True,
        )
        dyn = frame.dynamic_details
        assert dyn["decision_latency_ms"] is None, f"Expected None latency, got: {dyn['decision_latency_ms']}"
        assert dyn["telemetry_provenance"]["decision_latency_ms"] == "UNAVAILABLE"

    def test_c_missing_selected_action_not_silently_fallback(self):
        """C. Missing selected_action must NOT silently become 0."""
        # Model without selected_action and without action_probs/evidence
        model = DummyModel({"decision_latency_ms": 5.0})
        frame = self.capture.capture_phase23b_step(
            step=10,
            input_vec=np.zeros(2),
            model=model,
            is_zero_shot_test=True,
        )
        dyn = frame.dynamic_details
        assert dyn["selected_action"] is None, f"Expected None action, got: {dyn['selected_action']}"
        assert dyn["telemetry_provenance"]["selected_action"] == "UNAVAILABLE"

    def test_d_genuine_upstream_action_probs_survive_unchanged(self):
        """D. Genuine upstream action_probs must survive unchanged."""
        raw_probs = [0.2345, 0.7655]
        model = DummyModel({
            "selected_action": 1,
            "action_probs": raw_probs,
            "decision_latency_ms": 1.23,
        })
        frame = self.capture.capture_phase23b_step(
            step=10,
            input_vec=np.zeros(2),
            model=model,
            is_zero_shot_test=True,
        )
        dyn = frame.dynamic_details
        assert dyn["action_probs"] == [0.2345, 0.7655]
        assert dyn["confidence"] == 0.7655
        assert dyn["telemetry_provenance"]["action_probs"] == "GENUINE"
        assert dyn["telemetry_provenance"]["confidence"] == "DETERMINISTICALLY_DERIVED"

    def test_e_genuine_upstream_latency_survives_unchanged(self):
        """E. Genuine upstream latency must survive unchanged."""
        measured_lat = 0.456
        model = DummyModel({
            "selected_action": 1,
            "action_probs": [0.4, 0.6],
            "decision_latency_ms": measured_lat,
        })
        frame = self.capture.capture_phase23b_step(
            step=10,
            input_vec=np.zeros(2),
            model=model,
            is_zero_shot_test=True,
        )
        dyn = frame.dynamic_details
        assert dyn["decision_latency_ms"] == 0.456
        assert dyn["telemetry_provenance"]["decision_latency_ms"] == "GENUINE_MEASURED"

    def test_f_genuine_upstream_selected_action_survives_unchanged(self):
        """F. Genuine upstream selected_action must survive unchanged."""
        model = DummyModel({
            "selected_action": 1,
            "action_probs": [0.3, 0.7],
            "decision_latency_ms": 2.1,
        })
        frame = self.capture.capture_phase23b_step(
            step=10,
            input_vec=np.zeros(2),
            model=model,
            is_zero_shot_test=True,
        )
        dyn = frame.dynamic_details
        assert dyn["selected_action"] == 1
        assert dyn["telemetry_provenance"]["selected_action"] == "GENUINE"

    def test_g_brainlog_telemetry_provenance_metadata(self):
        """G. BrainLog metadata must indicate provenance status for each field."""
        model = DummyModel({
            "selected_action": 0,
            "action_probs": [0.8, 0.2],
            "decision_latency_ms": 3.4,
        })
        frame = self.capture.capture_phase23b_step(
            step=10,
            input_vec=np.zeros(2),
            model=model,
            is_zero_shot_test=True,
        )
        prov = frame.dynamic_details["telemetry_provenance"]
        assert "selected_action" in prov
        assert "action_probs" in prov
        assert "confidence" in prov
        assert "decision_latency_ms" in prov
        assert prov["selected_action"] in ["GENUINE", "DETERMINISTICALLY_DERIVED", "UNAVAILABLE", "FALLBACK"]
        assert prov["action_probs"] in ["GENUINE", "DETERMINISTICALLY_DERIVED", "UNAVAILABLE", "FALLBACK"]
        assert prov["decision_latency_ms"] in ["GENUINE_MEASURED", "UNAVAILABLE", "FALLBACK"]

    def test_h_zero_shot_verification_fails_if_telemetry_fallback(self):
        """H. ZERO-SHOT VERIFIED must fail if decision telemetry is fallback-derived or incomplete."""
        # Simulate evaluateZeroShotStatus logic in UI/Observatory
        def evaluate_zero_shot_badge(dyn):
            prov = dyn.get("telemetry_provenance", {})
            is_action_genuine = prov.get("selected_action") in ["GENUINE", "DETERMINISTICALLY_DERIVED"]
            is_prob_genuine = prov.get("confidence") in ["GENUINE", "DETERMINISTICALLY_DERIVED"] or prov.get("action_probs") == "GENUINE"
            has_fallback = any(v == "FALLBACK" for v in prov.values())
            is_dw_zero = dyn.get("delta_w_zero", False)
            is_zero_shot = dyn.get("is_zero_shot_test", False) or dyn.get("evaluation_status") == "FROZEN"

            if is_zero_shot:
                if is_action_genuine and is_prob_genuine and not has_fallback and is_dw_zero:
                    return "ZERO-SHOT VERIFIED"
                elif has_fallback:
                    return "UNVERIFIED (FALLBACK TELEMETRY)"
                else:
                    return "UNVERIFIED (INCOMPLETE TELEMETRY)"
            return "COMPUTED"

        # Case 1: Incomplete / missing telemetry
        dyn_incomplete = {
            "is_zero_shot_test": True,
            "delta_w_zero": True,
            "telemetry_provenance": {
                "selected_action": "UNAVAILABLE",
                "action_probs": "UNAVAILABLE",
                "confidence": "UNAVAILABLE",
                "decision_latency_ms": "UNAVAILABLE",
            }
        }
        assert evaluate_zero_shot_badge(dyn_incomplete) == "UNVERIFIED (INCOMPLETE TELEMETRY)"

        # Case 2: Fallback telemetry
        dyn_fallback = {
            "is_zero_shot_test": True,
            "delta_w_zero": True,
            "telemetry_provenance": {
                "selected_action": "FALLBACK",
                "action_probs": "FALLBACK",
                "confidence": "FALLBACK",
                "decision_latency_ms": "FALLBACK",
            }
        }
        assert evaluate_zero_shot_badge(dyn_fallback) == "UNVERIFIED (FALLBACK TELEMETRY)"

        # Case 3: Genuine scientific telemetry
        dyn_genuine = {
            "is_zero_shot_test": True,
            "delta_w_zero": True,
            "telemetry_provenance": {
                "selected_action": "GENUINE",
                "action_probs": "GENUINE",
                "confidence": "DETERMINISTICALLY_DERIVED",
                "decision_latency_ms": "GENUINE_MEASURED",
            }
        }
        assert evaluate_zero_shot_badge(dyn_genuine) == "ZERO-SHOT VERIFIED"

    def test_i_browser_ui_provenance_contract_in_player_html(self):
        """I. Browser UI must contain explicit provenance checks preventing fallback rendering."""
        player_path = Path("visualization/player.html")
        assert player_path.exists()
        html = player_path.read_text(encoding="utf-8")

        assert "telemetry_provenance" in html, "player.html must check telemetry_provenance"
        assert "UNVERIFIED (FALLBACK TELEMETRY)" in html, "player.html must flag fallback telemetry"
        assert "UNVERIFIED (INCOMPLETE TELEMETRY)" in html, "player.html must flag incomplete telemetry"
        assert "isActionGenuine" in html, "player.html must verify action genuineness"
        assert "isProbGenuine" in html, "player.html must verify probability genuineness"

    def test_j_exp023b_regression_test(self):
        """J. Explicit regression test using EXP023B verifying genuine computational provenance."""
        brainlog_path = Path("experiments/results/brainlogs/EXP023B_compositional.brainlog")
        assert brainlog_path.exists(), f"EXP023B BrainLog not found at {brainlog_path}"
        with open(brainlog_path, "r") as f:
            data = json.load(f)

        assert len(data["frames"]) == 120, f"Expected 120 frames, found {len(data['frames'])}"
        f119 = data["frames"][119]
        dyn = f119["dynamic_details"]

        # Step 119 must NOT contain silent fallback constants
        assert dyn["selected_action"] != 0 or dyn["telemetry_provenance"]["selected_action"] == "GENUINE", (
            "Action must not be silent fallback 0"
        )
        assert dyn["action_probs"] != [0.5, 0.5], "action_probs must not be fallback [0.5, 0.5]"
        assert dyn["confidence"] != 0.5, "confidence must not be fallback 0.5"
        assert dyn["decision_latency_ms"] != 11.5, "decision_latency_ms must not be fallback constant 11.5"

        # Frame 119 must have genuine verified provenance
        prov = dyn["telemetry_provenance"]
        assert prov["selected_action"] == "GENUINE", f"Unexpected action provenance: {prov['selected_action']}"
        assert prov["action_probs"] == "GENUINE", f"Unexpected probs provenance: {prov['action_probs']}"
        assert prov["confidence"] == "DETERMINISTICALLY_DERIVED", f"Unexpected conf provenance: {prov['confidence']}"
        assert prov["decision_latency_ms"] == "GENUINE_MEASURED", f"Unexpected latency provenance: {prov['decision_latency_ms']}"

        # Decision values: retrieved Episode 15 (BLUE) produces RIGHT (1) with >50% confidence
        assert dyn["selected_action"] == 1, f"Expected action 1 (RIGHT), got {dyn['selected_action']}"
        assert dyn["confidence"] > 0.50, f"Expected confidence > 50%, got {dyn['confidence']}"
        assert dyn["decision_latency_ms"] < 5.0, f"Expected measured latency < 5ms, got {dyn['decision_latency_ms']}"
        assert dyn["delta_w_zero"] is True, "Delta W must be strictly zero"
        assert dyn["evaluation_status"] == "FROZEN", "Evaluation must be FROZEN"
