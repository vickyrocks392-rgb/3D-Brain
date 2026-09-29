"""Biophysical Spike-Generation Audit & Multi-Stage Information Preservation (Phase 13).

Instruments and analyzes the complete pathway:
1. Context State (s_ctx)
2. Dendritic Potential (V_dend) & NMDA Conductance (g_nmda)
3. Somatic Integration (V_soma)
4. Axon Initial Segment (AIS) Adaptive Threshold (theta_eff) & Dynamic Gain (g)
5. Spike Generation (Spike train S(t), Firing Rate, ISI distribution)
6. Action Population Activity (Rate distribution, Synchrony)
7. Competition Dynamics (Lateral Inhibition, Margin)
8. Persistent Action Attractor State

Quantifies information preservation and identifies the exact locus of information loss:
Continuous Representation -> Spike Representation -> Action Selection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class StageMetrics:
    """Quantitative metrics for a single biophysical processing stage."""

    def __init__(self, name: str):
        self.name = name
        self.dimension = 0
        self.centroid_dist = 0.0
        self.within_class_var = 0.0
        self.between_class_var = 0.0
        self.separation_ratio = 0.0
        self.linear_probe_acc = 0.5
        self.nonlinear_probe_acc = 0.5
        self.mutual_info_bits = 0.0
        self.information_loss_from_prev_bits = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.name,
            "dimension": self.dimension,
            "centroid_dist": round(float(self.centroid_dist), 4),
            "within_class_var": round(float(self.within_class_var), 4),
            "between_class_var": round(float(self.between_class_var), 4),
            "separation_ratio": round(float(self.separation_ratio), 4),
            "linear_probe_acc": round(float(self.linear_probe_acc), 4),
            "nonlinear_probe_acc": round(float(self.nonlinear_probe_acc), 4),
            "mutual_info_bits": round(float(self.mutual_info_bits), 4),
            "information_loss_bits": round(float(self.information_loss_from_prev_bits), 4),
        }


class BiophysicalPathwayAuditor:
    """Instruments and computes information preservation metrics across biophysical stages."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def analyze_stage_representations(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        stage_name: str,
        prev_mi_bits: Optional[float] = None,
    ) -> StageMetrics:
        """Evaluate geometric separation, probe decoding, and mutual information for a stage.

        Args:
            features: [num_samples, feature_dim]
            labels: [num_samples] class labels (e.g. Context 0 vs 1)
            stage_name: Name of the stage
            prev_mi_bits: Mutual info of preceding stage to measure information loss

        Returns:
            StageMetrics instance.
        """
        metrics = StageMetrics(stage_name)
        X = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32)
        n_samples, d = X.shape
        metrics.dimension = d

        classes = np.unique(y)
        if len(classes) < 2 or n_samples < 4:
            return metrics

        # 1. Centroid Distance and Variance Metrics
        c0 = np.mean(X[y == classes[0]], axis=0)
        c1 = np.mean(X[y == classes[1]], axis=0)
        d_ab = float(np.linalg.norm(c0 - c1))
        metrics.centroid_dist = d_ab

        grand_mean = np.mean(X, axis=0)
        ss_between = float(len(X[y == classes[0]]) * np.sum((c0 - grand_mean) ** 2) + len(X[y == classes[1]]) * np.sum((c1 - grand_mean) ** 2))
        var_between = ss_between / max(1, n_samples)

        ss_within = float(np.sum((X[y == classes[0]] - c0) ** 2) + np.sum((X[y == classes[1]] - c1) ** 2))
        var_within = ss_within / max(1, n_samples)

        metrics.between_class_var = var_between
        metrics.within_class_var = var_within
        metrics.separation_ratio = float(var_between / (var_within + 1e-9))

        # 2. Linear Probe Accuracy (Regularized Least-Squares / Perceptron classifier)
        # 5-fold cross-validation or train/test split
        idx = self.rng.permutation(n_samples)
        split = int(0.75 * n_samples)
        train_idx, test_idx = idx[:split], idx[split:]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # Standardize features
        mean_tr = np.mean(X_tr, axis=0, keepdims=True)
        std_tr = np.std(X_tr, axis=0, keepdims=True) + 1e-6
        X_tr_norm = (X_tr - mean_tr) / std_tr
        X_te_norm = (X_te - mean_tr) / std_tr

        # Ridge regression classifier: w = (X^T X + lambda I)^-1 X^T y
        reg = 1e-3
        y_tr_signed = np.where(y_tr == classes[0], -1.0, 1.0).astype(np.float32)
        y_te_signed = np.where(y_te == classes[0], -1.0, 1.0).astype(np.float32)

        xtx = X_tr_norm.T @ X_tr_norm + reg * np.eye(d, dtype=np.float32)
        try:
            w = np.linalg.solve(xtx, X_tr_norm.T @ y_tr_signed)
            pred_te = np.sign(X_te_norm @ w)
            linear_acc = float(np.mean(pred_te == y_te_signed))
        except np.linalg.LinAlgError:
            linear_acc = 0.5

        metrics.linear_probe_acc = max(0.5, linear_acc)

        # 3. Nonlinear Probe Accuracy (Kernel RBF / Nearest Centroid / 2nd order)
        # Nearest centroid with feature normalization
        c0_norm = np.mean(X_tr_norm[y_tr == classes[0]], axis=0)
        c1_norm = np.mean(X_tr_norm[y_tr == classes[1]], axis=0)
        d0 = np.linalg.norm(X_te_norm - c0_norm, axis=1)
        d1 = np.linalg.norm(X_te_norm - c1_norm, axis=1)
        pred_centroid = np.where(d0 < d1, classes[0], classes[1])
        centroid_acc = float(np.mean(pred_centroid == y_te))
        metrics.nonlinear_probe_acc = max(linear_acc, centroid_acc)

        # 4. Mutual Information I(X; y) Estimation
        # Binary entropy: H(y) - H(y|x)
        # Using linear/probe classification error pe as proxy:
        # Fano's bound / binary symmetric channel channel capacity:
        # I(X; y) = 1 - H_2(p_error)
        pe = np.clip(1.0 - metrics.linear_probe_acc, 1e-6, 0.5)
        h2 = -pe * np.log2(pe) - (1.0 - pe) * np.log2(1.0 - pe)
        metrics.mutual_info_bits = float(max(0.0, 1.0 - h2))

        # 5. Information Loss from Previous Stage
        if prev_mi_bits is not None:
            metrics.information_loss_from_prev_bits = max(0.0, prev_mi_bits - metrics.mutual_info_bits)

        return metrics

    def run_pathway_audit(
        self,
        trial_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Audit the complete biophysical transmission chain across trials.

        Expected stages in trial records:
        - context_vector
        - v_dend
        - g_nmda
        - v_soma
        - theta_eff
        - action_spikes
        - action_pool_rates
        - persistent_action_state
        - true_context_label
        """
        stages = [
            ("1_CONTEXT_VECTOR", "context_vector"),
            ("2_DENDRITIC_POTENTIAL", "v_dend"),
            ("3_NMDA_CONDUCTANCE", "g_nmda"),
            ("4_SOMATIC_INTEGRATION", "v_soma"),
            ("5_ADAPTIVE_THRESHOLD", "theta_eff"),
            ("6_SPIKE_ENCODING", "action_spikes"),
            ("7_ACTION_POPULATION_RATES", "action_pool_rates"),
            ("8_PERSISTENT_ACTION", "persistent_action_state"),
        ]

        labels = np.array([r["true_context_label"] for r in trial_records], dtype=np.int32)
        results: Dict[str, Any] = {}
        prev_mi = None

        total_continuous_mi = 0.0
        total_spike_mi = 0.0
        total_action_mi = 0.0

        for stage_name, key in stages:
            features = np.array([r[key].flatten() for r in trial_records], dtype=np.float32)
            m = self.analyze_stage_representations(features, labels, stage_name, prev_mi_bits=prev_mi)
            results[stage_name] = m.to_dict()
            prev_mi = m.mutual_info_bits

            if stage_name == "4_SOMATIC_INTEGRATION":
                total_continuous_mi = m.mutual_info_bits
            elif stage_name == "6_SPIKE_ENCODING":
                total_spike_mi = m.mutual_info_bits
            elif stage_name == "8_PERSISTENT_ACTION":
                total_action_mi = m.mutual_info_bits

        # Summary transitions
        loss_continuous_to_spike = max(0.0, total_continuous_mi - total_spike_mi)
        loss_spike_to_action = max(0.0, total_spike_mi - total_action_mi)
        total_information_loss = max(0.0, total_continuous_mi - total_action_mi)

        results["_summary"] = {
            "continuous_soma_mi_bits": round(total_continuous_mi, 4),
            "spike_encoding_mi_bits": round(total_spike_mi, 4),
            "action_selection_mi_bits": round(total_action_mi, 4),
            "loss_continuous_to_spike_bits": round(loss_continuous_to_spike, 4),
            "loss_spike_to_action_bits": round(loss_spike_to_action, 4),
            "total_pathway_loss_bits": round(total_information_loss, 4),
            "retention_percentage": round(100.0 * (total_action_mi / max(1e-4, total_continuous_mi)), 2),
        }

        return results
