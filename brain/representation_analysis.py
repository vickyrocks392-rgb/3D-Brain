r"""Representational Analysis Engine for 3D-Brain Phase 6.

Quantifies internal latent state representations across time, history, and physical space:
1. Linear Decoding Probe: Ridge/OLS decodability of past sensory cues from z_t vs h_t.
2. Representational Similarity Analysis (RSA): Dissimilarity matrices across conditions.
3. Temporal Stability: Autocorrelation decay across temporal lags \tau.
4. State Separability: Fisher Discriminant Ratio separating divergent historical contexts.
5. History Dependence: Partial variance in z_t explained by history conditioning.
6. Dimensionality: Participation Ratio PR = (\sum \lambda_i)^2 / \sum \lambda_i^2 from PCA eigenvalues.
7. Spatial Participation: Correlation of latent activity with 3D metric coordinates.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class LinearDecodingProbe:
    """Linear regression / classification probe to decode historical information from internal states."""

    @staticmethod
    def train_and_evaluate(
        features: np.ndarray,
        targets: np.ndarray,
        train_ratio: float = 0.8,
        ridge_alpha: float = 1e-4,
    ) -> Dict[str, float]:
        r"""Train linear readout W to predict targets Y from state features X.

        Args:
            features: [N_samples, feature_dim] e.g. z_t or h_t.
            targets: [N_samples, target_dim] e.g. cue identity or spatial position.

        Returns:
            Dictionary with R^2 score, MSE, and Pearson correlation.
        """
        N = len(features)
        if N < 5:
            return {"r2": 0.0, "mse": 1.0, "corr": 0.0}

        n_train = max(3, int(N * train_ratio))
        X_train, X_test = features[:n_train], features[n_train:]
        Y_train, Y_test = targets[:n_train], targets[n_train:]

        if len(X_test) == 0:
            X_test, Y_test = X_train, Y_train

        # Add bias column
        X_tr = np.hstack([X_train, np.ones((len(X_train), 1), dtype=np.float32)])
        X_te = np.hstack([X_test, np.ones((len(X_test), 1), dtype=np.float32)])

        # Ridge regression solution: W = (X^T X + \alpha I)^{-1} X^T Y
        reg = ridge_alpha * np.eye(X_tr.shape[1], dtype=np.float32)
        reg[-1, -1] = 0.0  # Do not regularize bias
        try:
            W = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ Y_train)
            Y_pred = X_te @ W
        except np.linalg.LinAlgError:
            W = np.linalg.pinv(X_tr) @ Y_train
            Y_pred = X_te @ W

        mse = float(np.mean((Y_test - Y_pred) ** 2))
        var_y = float(np.var(Y_test))
        r2 = float(1.0 - mse / var_y) if var_y > 1e-8 else 0.0

        # Mean Pearson correlation across target dimensions
        corrs = []
        for d in range(Y_test.shape[1] if Y_test.ndim > 1 else 1):
            y_t = Y_test[:, d] if Y_test.ndim > 1 else Y_test
            y_p = Y_pred[:, d] if Y_pred.ndim > 1 else Y_pred
            if np.std(y_t) > 1e-6 and np.std(y_p) > 1e-6:
                c = np.corrcoef(y_t, y_p)[0, 1]
                corrs.append(c if not np.isnan(c) else 0.0)
            else:
                corrs.append(0.0)
        mean_corr = float(np.mean(corrs)) if corrs else 0.0

        return {"r2": r2, "mse": mse, "corr": mean_corr}


class RepresentationalSimilarityAnalysis:
    """Computes Representational Dissimilarity Matrices (RDMs) across stimuli or conditions."""

    @staticmethod
    def compute_rdm(
        states: np.ndarray, metric: str = "correlation"
    ) -> np.ndarray:
        r"""Compute square dissimilarity matrix between condition representations.

        Args:
            states: [N_conditions, feature_dim] activation patterns.
            metric: 'correlation' or 'cosine' or 'euclidean'.

        Returns:
            [N_conditions, N_conditions] symmetric RDM.
        """
        N = len(states)
        rdm = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(i, N):
                v1, v2 = states[i], states[j]
                if metric == "correlation":
                    std1, std2 = np.std(v1), np.std(v2)
                    if std1 > 1e-6 and std2 > 1e-6:
                        corr = np.corrcoef(v1, v2)[0, 1]
                        dist = 1.0 - (corr if not np.isnan(corr) else 0.0)
                    else:
                        dist = 1.0
                elif metric == "cosine":
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-6 and n2 > 1e-6:
                        dist = 1.0 - float(np.dot(v1, v2) / (n1 * n2))
                    else:
                        dist = 1.0
                else:
                    dist = float(np.linalg.norm(v1 - v2))

                rdm[i, j] = dist
                rdm[j, i] = dist

        return rdm

    @staticmethod
    def rdm_correlation(rdm1: np.ndarray, rdm2: np.ndarray) -> float:
        """Spearman/Pearson correlation between upper triangles of two RDMs."""
        idx = np.triu_indices_from(rdm1, k=1)
        v1, v2 = rdm1[idx], rdm2[idx]
        if np.std(v1) > 1e-6 and np.std(v2) > 1e-6:
            r = np.corrcoef(v1, v2)[0, 1]
            return float(r if not np.isnan(r) else 0.0)
        return 0.0


class StateDynamicsAnalyzer:
    """Analyzes temporal stability, separability, dimensionality, and spatial distribution."""

    @staticmethod
    def autocorrelation_decay(
        trajectory: np.ndarray, max_lag: int = 50
    ) -> Tuple[np.ndarray, float]:
        r"""Compute temporal autocorrelation curve: R(\tau) = \frac{\langle z_t, z_{t+\tau} \rangle}{\|z_t\| \|z_{t+\tau}\|}.

        Returns:
            autocorr_curve: [max_lag] array.
            decay_time: Effective half-life / decay constant \tau where R(\tau) drops below 1/e.
        """
        T = len(trajectory)
        lags = min(max_lag, T - 1)
        curve = np.zeros(lags, dtype=np.float32)

        centered = trajectory - np.mean(trajectory, axis=0, keepdims=True)
        var_total = np.sum(centered ** 2)

        if var_total < 1e-8:
            return curve, 0.0

        for tau in range(lags):
            v1 = centered[: T - tau]
            v2 = centered[tau:]
            dot_prod = np.sum(v1 * v2)
            curve[tau] = float(dot_prod / (np.sqrt(np.sum(v1 ** 2) * np.sum(v2 ** 2)) + 1e-8))

        # Estimate decay time \tau where curve drops below 1/e = 0.36788
        threshold = 1.0 / np.e
        decay_time = float(lags)
        for tau in range(lags):
            if curve[tau] < threshold:
                decay_time = float(tau)
                break

        return curve, decay_time

    @staticmethod
    def state_separability(
        class_1_states: np.ndarray, class_2_states: np.ndarray
    ) -> Dict[str, float]:
        r"""Compute Fisher Discriminant Ratio separating two clusters in latent space:
        FDR = \frac{\|\mu_1 - \mu_2\|^2}{\sigma_1^2 + \sigma_2^2}
        """
        if len(class_1_states) == 0 or len(class_2_states) == 0:
            return {"fdr": 0.0, "centroid_distance": 0.0, "within_var": 0.0}

        mu1 = np.mean(class_1_states, axis=0)
        mu2 = np.mean(class_2_states, axis=0)
        between_dist_sq = float(np.sum((mu1 - mu2) ** 2))

        var1 = float(np.mean(np.sum((class_1_states - mu1) ** 2, axis=1))) if len(class_1_states) > 1 else 1e-6
        var2 = float(np.mean(np.sum((class_2_states - mu2) ** 2, axis=1))) if len(class_2_states) > 1 else 1e-6
        within_var = var1 + var2

        fdr = float(between_dist_sq / (within_var + 1e-8))
        return {
            "fdr": fdr,
            "centroid_distance": float(np.sqrt(between_dist_sq)),
            "within_var": within_var,
        }

    @staticmethod
    def participation_ratio(states: np.ndarray) -> float:
        r"""Compute effective dimensionality via Participation Ratio:
        PR = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}
        where \lambda_i are the eigenvalues of the state covariance matrix.
        """
        if len(states) < 2:
            return 1.0

        centered = states - np.mean(states, axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)

        if cov.ndim == 0:
            return 1.0

        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0.0)

        sum_e = float(np.sum(eigvals))
        sum_sq_e = float(np.sum(eigvals ** 2))

        if sum_sq_e < 1e-10:
            return 1.0

        pr = float((sum_e ** 2) / sum_sq_e)
        return min(pr, float(states.shape[1]))

    @staticmethod
    def spatial_participation(
        node_positions: np.ndarray,
        node_activations: np.ndarray,
        latent_state: np.ndarray,
    ) -> Dict[str, float]:
        r"""Measure correlation between 3D metric spatial position and latent representation participation."""
        if len(node_positions) == 0 or len(node_activations) == 0:
            return {"spatial_corr_mean": 0.0, "radius_correlation": 0.0}

        r = np.linalg.norm(node_positions, axis=1)
        mean_act = np.mean(node_activations, axis=0) if node_activations.ndim > 1 else node_activations

        if np.std(r) > 1e-6 and np.std(mean_act) > 1e-6:
            rad_corr = float(np.corrcoef(r, mean_act)[0, 1])
        else:
            rad_corr = 0.0

        return {
            "radius_correlation": rad_corr if not np.isnan(rad_corr) else 0.0,
            "mean_activity": float(np.mean(mean_act)),
            "spatial_spread": float(np.std(mean_act)),
        }
