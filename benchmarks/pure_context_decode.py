"""Pure Context Decode Benchmark: Representation vs Readout vs Credit Decomposition (Phase 12).

Diagnostic probe with:
- NO delayed reward
- NO temporal credit assignment
- NO recurrent learning

Directly tests whether the latent representations formed by the Phase 11 substrate
contain decodable information about context, isolating representation capacity from
downstream readout learning and temporal credit assignment.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DecodeProbeRecord:
    context_id: int
    target_action: int
    context_vector: np.ndarray
    predicted_action: int
    action_probs: np.ndarray
    is_correct: bool


class PureContextDecodeTask:
    """Diagnostic task evaluating immediate context representation separability and decodability."""

    def __init__(
        self,
        obs_dim: int = 8,
        num_classes: int = 2,
        noise_std: float = 0.05,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.num_classes = num_classes
        self.noise_std = noise_std
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # High-dimensional orthogonal context prototypes
        self.prototypes = np.zeros((num_classes, obs_dim), dtype=np.float32)
        if num_classes == 2:
            self.prototypes[0, 0] = 1.0
            self.prototypes[0, 2:5] = 0.6
            self.prototypes[1, 1] = 1.0
            self.prototypes[1, 5:8] = 0.6
        else:
            for k in range(num_classes):
                self.prototypes[k, k % obs_dim] = 1.0
                start = (k * 2) % obs_dim
                self.prototypes[k, start : min(obs_dim, start + 2)] = 0.5

    def generate_sample(
        self,
        context_id: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate a single noisy sensory observation of the given context."""
        gen = rng if rng is not None else self.rng
        noise = gen.normal(0.0, self.noise_std, self.obs_dim).astype(np.float32)
        x = np.clip(self.prototypes[context_id] + noise, 0.0, 1.0)
        return x, context_id

    def evaluate_representation_separability(
        self,
        representations_a: np.ndarray,
        representations_b: np.ndarray,
    ) -> Dict[str, float]:
        """Quantify geometry, centroid distance, within/between class variance, and mutual information.

        Args:
            representations_a: Extracted context representations for Context A [N_samples, D]
            representations_b: Extracted context representations for Context B [N_samples, D]

        Returns:
            metrics: Dictionary of geometrical separability metrics
        """
        ra = np.asarray(representations_a, dtype=np.float32)
        rb = np.asarray(representations_b, dtype=np.float32)

        mu_a = np.mean(ra, axis=0)
        mu_b = np.mean(rb, axis=0)
        centroid_distance = float(np.linalg.norm(mu_a - mu_b))

        var_a = float(np.mean(np.var(ra, axis=0)))
        var_b = float(np.mean(np.var(rb, axis=0)))
        within_var = 0.5 * (var_a + var_b)
        between_var = float(np.var(np.concatenate([ra, rb], axis=0), axis=0).mean())

        sep_ratio = float(centroid_distance / np.sqrt(max(1e-6, within_var)))

        # Linear separability via Fisher discriminant / linear probe on representation
        x_all = np.vstack([ra, rb])
        y_all = np.concatenate([np.zeros(len(ra)), np.ones(len(rb))])
        w_diff = mu_b - mu_a
        proj = x_all @ w_diff
        thresh = 0.5 * (np.dot(mu_a, w_diff) + np.dot(mu_b, w_diff))
        preds = (proj >= thresh).astype(int)
        linear_acc = float(np.mean(preds == y_all))

        # Discrete mutual information estimation I(Context; Quantized Proj)
        q_proj = (proj >= np.median(proj)).astype(int)
        p_c = [0.5, 0.5]
        p_q = [float(np.mean(q_proj == 0)), float(np.mean(q_proj == 1))]
        p_cq = np.zeros((2, 2))
        for c_idx in (0, 1):
            for q_idx in (0, 1):
                p_cq[c_idx, q_idx] = float(np.mean((y_all == c_idx) & (q_proj == q_idx)))

        mi = 0.0
        for c_idx in range(2):
            for q_idx in range(2):
                if p_cq[c_idx, q_idx] > 1e-12:
                    mi += p_cq[c_idx, q_idx] * np.log2(
                        p_cq[c_idx, q_idx] / (p_c[c_idx] * p_q[q_idx] + 1e-12)
                    )
        mi = max(0.0, float(mi))

        return {
            "centroid_distance": round(centroid_distance, 4),
            "within_class_variance": round(within_var, 4),
            "between_class_variance": round(between_var, 4),
            "separation_ratio": round(sep_ratio, 4),
            "linear_probe_accuracy": round(linear_acc, 4),
            "mutual_information_bits": round(mi, 4),
        }
