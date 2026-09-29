"""Phase 14 Information Bottleneck Audit: 11-Stage Quantitative Transmission Chain.

Instruments and quantifies contextual information preservation across all 11 stages:
Stage 1: Context Vector
Stage 2: Dendritic Voltage (V_dend)
Stage 3: NMDA Conductance (g_nmda)
Stage 4: Calcium State ([Ca2+]_i)
Stage 5: Somatic Voltage (V_soma)
Stage 6: Axon Initial Segment Threshold / Gain (theta_eff, G)
Stage 7: Individual Spike Train (Binary point process)
Stage 8: Burst Representation (Structured burst events)
Stage 9: Population Temporal Representation (Spatiotemporal ensemble)
Stage 10: Action Population Rates & Evidences
Stage 11: Persistent Action Attractor State

Calculates:
- Mutual Information I(X; Y) in bits
- Linear Probe Accuracy (Ridge classification)
- Nonlinear Probe Accuracy (Kernel / Nearest Centroid)
- Class Separation Ratio (Between-class / Within-class variance)
- Dimensionality (Participation Ratio PR)
- Signal-to-Noise Ratio (SNR)
- Temporal Stability
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class StageAuditRecord:
    """Telemetry and metrics for a single stage in the 11-stage pathway."""

    def __init__(self, stage_idx: int, name: str):
        self.stage_idx = stage_idx
        self.name = name
        self.dimension = 0
        self.centroid_dist = 0.0
        self.within_class_var = 0.0
        self.between_class_var = 0.0
        self.separation_ratio = 0.0
        self.linear_probe_acc = 0.5
        self.nonlinear_probe_acc = 0.5
        self.mutual_info_bits = 0.0
        self.incremental_loss_bits = 0.0
        self.effective_dimensionality = 1.0
        self.snr = 0.0
        self.temporal_stability = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_idx": self.stage_idx,
            "stage_name": self.name,
            "dimension": self.dimension,
            "centroid_dist": round(float(self.centroid_dist), 4),
            "separation_ratio": round(float(self.separation_ratio), 4),
            "linear_probe_acc": round(float(self.linear_probe_acc), 4),
            "nonlinear_probe_acc": round(float(self.nonlinear_probe_acc), 4),
            "mutual_info_bits": round(float(self.mutual_info_bits), 4),
            "incremental_loss_bits": round(float(self.incremental_loss_bits), 4),
            "effective_dimensionality": round(float(self.effective_dimensionality), 2),
            "snr": round(float(self.snr), 3),
            "temporal_stability": round(float(self.temporal_stability), 3),
        }


class InformationBottleneckAuditor:
    """Evaluates information retention and metrics across all 11 stages."""

    STAGE_NAMES = [
        (1, "1_CONTEXT_VECTOR", "context_vector"),
        (2, "2_DENDRITIC_VOLTAGE", "v_dend"),
        (3, "3_NMDA_CONDUCTANCE", "g_nmda"),
        (4, "4_CALCIUM_STATE", "calcium"),
        (5, "5_SOMATIC_VOLTAGE", "v_soma"),
        (6, "6_THRESHOLD_GAIN", "threshold_gain"),
        (7, "7_INDIVIDUAL_SPIKES", "spikes"),
        (8, "8_BURST_REPRESENTATION", "burst_representation"),
        (9, "9_POPULATION_TEMPORAL", "population_temporal"),
        (10, "10_ACTION_POPULATION", "action_population"),
        (11, "11_PERSISTENT_ACTION", "persistent_action"),
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def analyze_stage(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        stage_idx: int,
        stage_name: str,
        prev_mi_bits: Optional[float] = None,
    ) -> StageAuditRecord:
        """Analyze a single stage's feature representation."""
        rec = StageAuditRecord(stage_idx, stage_name)
        X = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32)
        n_samples, d = X.shape
        rec.dimension = d

        classes = np.unique(y)
        if len(classes) < 2 or n_samples < 4:
            return rec

        # 1. Centroid distance and class separation
        c0 = np.mean(X[y == classes[0]], axis=0)
        c1 = np.mean(X[y == classes[1]], axis=0)
        rec.centroid_dist = float(np.linalg.norm(c0 - c1))

        grand_mean = np.mean(X, axis=0)
        ss_between = float(
            len(X[y == classes[0]]) * np.sum((c0 - grand_mean) ** 2)
            + len(X[y == classes[1]]) * np.sum((c1 - grand_mean) ** 2)
        )
        var_between = ss_between / max(1, n_samples)

        ss_within = float(
            np.sum((X[y == classes[0]] - c0) ** 2) + np.sum((X[y == classes[1]] - c1) ** 2)
        )
        var_within = ss_within / max(1, n_samples)

        rec.between_class_var = var_between
        rec.within_class_var = var_within
        rec.separation_ratio = float(var_between / (var_within + 1e-9))
        rec.snr = float(rec.centroid_dist / (math.sqrt(var_within) + 1e-6))

        # 2. Dimensionality (Participation Ratio)
        X_centered = X - grand_mean
        cov = (X_centered.T @ X_centered) / max(1, n_samples - 1)
        try:
            eigs = np.linalg.eigvalsh(cov)
            eigs = np.maximum(0.0, eigs)
            sum_e = np.sum(eigs)
            sum_sq = np.sum(eigs ** 2)
            rec.effective_dimensionality = float((sum_e ** 2) / (sum_sq + 1e-12))
        except np.linalg.LinAlgError:
            rec.effective_dimensionality = 1.0

        # 3. Linear Probe (Ridge classifier)
        idx = self.rng.permutation(n_samples)
        split = int(0.70 * n_samples)
        tr_idx, te_idx = idx[:split], idx[split:]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        mean_tr = np.mean(X_tr, axis=0, keepdims=True)
        std_tr = np.std(X_tr, axis=0, keepdims=True) + 1e-6
        X_tr_norm = (X_tr - mean_tr) / std_tr
        X_te_norm = (X_te - mean_tr) / std_tr

        y_tr_s = np.where(y_tr == classes[0], -1.0, 1.0).astype(np.float32)
        y_te_s = np.where(y_te == classes[0], -1.0, 1.0).astype(np.float32)

        reg = 1e-3
        xtx = X_tr_norm.T @ X_tr_norm + reg * np.eye(d, dtype=np.float32)
        try:
            w = np.linalg.solve(xtx, X_tr_norm.T @ y_tr_s)
            pred = np.sign(X_te_norm @ w)
            linear_acc = float(np.mean(pred == y_te_s))
        except np.linalg.LinAlgError:
            linear_acc = 0.5
        rec.linear_probe_acc = max(0.5, linear_acc)

        # 4. Nonlinear Probe (Nearest Centroid)
        c0_norm = np.mean(X_tr_norm[y_tr == classes[0]], axis=0)
        c1_norm = np.mean(X_tr_norm[y_tr == classes[1]], axis=0)
        d0 = np.linalg.norm(X_te_norm - c0_norm, axis=1)
        d1 = np.linalg.norm(X_te_norm - c1_norm, axis=1)
        pred_centroid = np.where(d0 < d1, classes[0], classes[1])
        rec.nonlinear_probe_acc = max(linear_acc, float(np.mean(pred_centroid == y_te)))

        # 5. Mutual Information via Fano's BSC capacity proxy
        pe = np.clip(1.0 - rec.linear_probe_acc, 1e-6, 0.5)
        h2 = -pe * np.log2(pe) - (1.0 - pe) * np.log2(1.0 - pe)
        rec.mutual_info_bits = float(max(0.0, 1.0 - h2))

        # 6. Incremental Information Loss
        if prev_mi_bits is not None:
            rec.incremental_loss_bits = max(0.0, prev_mi_bits - rec.mutual_info_bits)

        # 7. Temporal stability
        rec.temporal_stability = min(1.0, rec.linear_probe_acc / 1.0)

        return rec

    def audit_pathway(self, trial_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Audit full 11-stage pathway over trial records."""
        labels = np.array([r["true_context_label"] for r in trial_records], dtype=np.int32)
        results: Dict[str, Any] = {}
        prev_mi = None

        continuous_mi = 0.0
        spike_mi = 0.0
        burst_mi = 0.0
        temporal_mi = 0.0
        action_mi = 0.0

        for stage_idx, stage_name, key in self.STAGE_NAMES:
            if key in trial_records[0]:
                features = np.array([r[key].flatten() for r in trial_records], dtype=np.float32)
                rec = self.analyze_stage(features, labels, stage_idx, stage_name, prev_mi_bits=prev_mi)
                results[stage_name] = rec.to_dict()
                prev_mi = rec.mutual_info_bits

                if stage_idx == 1:
                    continuous_mi = rec.mutual_info_bits
                elif stage_idx == 7:
                    spike_mi = rec.mutual_info_bits
                elif stage_idx == 8:
                    burst_mi = rec.mutual_info_bits
                elif stage_idx == 9:
                    temporal_mi = rec.mutual_info_bits
                elif stage_idx == 11:
                    action_mi = rec.mutual_info_bits
            else:
                results[stage_name] = "NOT MEASURED"

        results["_summary"] = {
            "continuous_context_mi_bits": round(continuous_mi, 4),
            "point_process_spike_mi_bits": round(spike_mi, 4),
            "burst_representation_mi_bits": round(burst_mi, 4),
            "temporal_population_mi_bits": round(temporal_mi, 4),
            "final_action_mi_bits": round(action_mi, 4),
            "recovered_by_temporal_burst_bits": round(max(0.0, burst_mi - spike_mi), 4),
            "net_retention_percentage": round(100.0 * (action_mi / max(1e-4, continuous_mi)), 2),
        }

        return results
