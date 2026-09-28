r"""Persistent Latent Memory Module for 3D-Brain Phase 6.

Implements an explicit, multi-timescale persistent latent state $z_t \in \mathbb{R}^{D_z}$
coupled with the fast spatial recurrent substrate $h_t \in \mathbb{R}^H$.

Update equations:
    \tilde{z}_t = \tanh(a_t W_{hz} + z_{t-1} W_{zz} + a_{\text{onehot}} W_{az} + b_z)
    z_t = \lambda_z \odot z_{t-1} + (1 - \lambda_z) \odot \tilde{z}_t
    I_{\text{latent}} = z_t W_{zh}

Multi-step forward prediction interface:
    \hat{s}_{t+k} = \text{clip}( [z_t, a_{\text{onehot}}] W_{\text{pred}}^{(k)} + b_{\text{pred}}^{(k)}, 0, 1)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class PersistentLatentModule:
    """Explicit persistent latent memory unit with multi-timescale retention and multi-step prediction."""

    def __init__(
        self,
        hidden_dim: int = 40,
        latent_dim: int = 24,
        obs_dim: int = 8,
        action_dim: int = 4,
        prediction_horizons: Optional[List[int]] = None,
        persistence_min: float = 0.85,
        persistence_max: float = 0.98,
        feedback_strength: float = 0.5,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0001,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.prediction_horizons = prediction_horizons or [1, 2, 5, 10, 25]
        self.feedback_strength = feedback_strength
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Multi-timescale persistence vector \lambda_z in [persistence_min, persistence_max]
        # Generates a log-linear distribution of timescales across latent dimensions
        self.persistence_rates = np.linspace(persistence_min, persistence_max, latent_dim).astype(np.float32)

        # Weight matrices
        scale_hz = np.sqrt(2.0 / (hidden_dim + latent_dim))
        scale_zz = np.sqrt(2.0 / (latent_dim + latent_dim))
        scale_az = np.sqrt(2.0 / (action_dim + latent_dim))
        scale_zh = np.sqrt(2.0 / (latent_dim + hidden_dim))

        self.W_hz = self.rng.normal(0.0, scale_hz, (hidden_dim, latent_dim)).astype(np.float32)
        self.W_zz = self.rng.normal(0.0, scale_zz, (latent_dim, latent_dim)).astype(np.float32)
        self.W_az = self.rng.normal(0.0, scale_az, (action_dim, latent_dim)).astype(np.float32)
        self.b_z = np.zeros(latent_dim, dtype=np.float32)

        # Top-down feedback projection into recurrent substrate
        self.W_zh = self.rng.normal(0.0, scale_zh, (latent_dim, hidden_dim)).astype(np.float32)

        # Multi-step prediction heads: mapping [z_t; a_{onehot}] -> \hat{s}_{t+k}
        scale_pred = np.sqrt(2.0 / (latent_dim + action_dim + obs_dim))
        self.W_preds: Dict[int, np.ndarray] = {}
        self.b_preds: Dict[int, np.ndarray] = {}
        self.B_fa: Dict[int, np.ndarray] = {}

        for k in self.prediction_horizons:
            self.W_preds[k] = self.rng.normal(0.0, scale_pred, (latent_dim + action_dim, obs_dim)).astype(np.float32)
            self.b_preds[k] = np.zeros(obs_dim, dtype=np.float32)
            # Random orthogonal feedback alignment projection for biologically plausible credit assignment
            fa_rand = self.rng.normal(0.0, 1.0, (obs_dim, latent_dim)).astype(np.float32)
            q, _ = np.linalg.qr(fa_rand.T)
            self.B_fa[k] = q.T[:obs_dim, :latent_dim].copy()

        # Neuromodulatory eligibility traces
        self.trace_zz = np.zeros((latent_dim, latent_dim), dtype=np.float32)
        self.trace_hz = np.zeros((hidden_dim, latent_dim), dtype=np.float32)
        self.trace_gamma = 0.9

        # State registers
        self.z = np.zeros(latent_dim, dtype=np.float32)
        self.prev_z = np.zeros(latent_dim, dtype=np.float32)
        self.cand_z = np.zeros(latent_dim, dtype=np.float32)
        self.last_predictions: Dict[int, np.ndarray] = {}
        self.last_errors: Dict[int, float] = {}

        # Causal ablation flags
        self.persistence_enabled = True
        self.reset_per_step = False
        self.feedback_enabled = True
        self.learning_enabled = True
        self.frozen = False

    def reset(self) -> None:
        """Reset internal dynamic latent state registers."""
        self.z.fill(0.0)
        self.prev_z.fill(0.0)
        self.cand_z.fill(0.0)
        self.trace_zz.fill(0.0)
        self.trace_hz.fill(0.0)
        self.last_predictions.clear()
        self.last_errors.clear()

    def update_latent_state(
        self,
        recurrent_act: np.ndarray,
        action: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the next persistent latent state $z_t$ and recurrent feedback $I_{\text{latent}}$.

        Args:
            recurrent_act: Current activation vector $a_t \in \mathbb{R}^H$ of the recurrent substrate.
            action: Selected discrete action index.

        Returns:
            z_t: Persistent latent state [latent_dim].
            I_latent: Feedback drive injected into recurrent substrate [hidden_dim].
        """
        if self.reset_per_step:
            self.prev_z.fill(0.0)
        else:
            self.prev_z = self.z.copy()

        action_onehot = np.zeros(self.action_dim, dtype=np.float32)
        action_onehot[action % self.action_dim] = 1.0

        # Raw candidate drive: a_t W_{hz} + z_{t-1} W_{zz} + a_t W_{az} + b_z
        drive = (
            np.dot(recurrent_act, self.W_hz)
            + np.dot(self.prev_z, self.W_zz)
            + np.dot(action_onehot, self.W_az)
            + self.b_z
        )
        self.cand_z = np.tanh(drive)

        if self.persistence_enabled:
            lambdas = self.persistence_rates
            self.z = lambdas * self.prev_z + (1.0 - lambdas) * self.cand_z
        else:
            self.z = self.cand_z.copy()

        # Compute feedback into recurrent substrate: I_latent = z_t W_{zh} * strength
        if self.feedback_enabled:
            I_latent = float(self.feedback_strength) * np.dot(self.z, self.W_zh)
        else:
            I_latent = np.zeros(self.hidden_dim, dtype=np.float32)

        return self.z, I_latent

    def predict_multi_step(self, action: int) -> Dict[int, np.ndarray]:
        r"""Generate sensory predictions for all configured horizons $k \in \{1, 2, 5, 10, 25\}$."""
        action_onehot = np.zeros(self.action_dim, dtype=np.float32)
        action_onehot[action % self.action_dim] = 1.0
        feature = np.concatenate([self.z, action_onehot], axis=0)

        preds = {}
        for k, W_p in self.W_preds.items():
            b_p = self.b_preds[k]
            raw = np.dot(feature, W_p) + b_p
            preds[k] = np.clip(raw, 0.0, 1.0).astype(np.float32)

        self.last_predictions = preds
        return preds

    def learn_step(
        self,
        recurrent_act: np.ndarray,
        action: int,
        future_observations: Dict[int, np.ndarray],
    ) -> Dict[str, float]:
        r"""Apply predictive plasticity update across multi-step prediction horizons.

        Args:
            recurrent_act: Pre-synaptic neural activation $a_t$.
            action: Action executed.
            future_observations: Dictionary mapping horizon $k \to s_{t+k}$ ground-truth observation.

        Returns:
            Dictionary of learning metrics (MSE per horizon, gradient norms).
        """
        if not self.learning_enabled or self.frozen:
            return {"mean_multistep_mse": 0.0, "dw_norm": 0.0}

        action_onehot = np.zeros(self.action_dim, dtype=np.float32)
        action_onehot[action % self.action_dim] = 1.0
        feature = np.concatenate([self.z, action_onehot], axis=0)

        total_err_mse = 0.0
        delta_z_accum = np.zeros(self.latent_dim, dtype=np.float32)
        horizon_count = 0

        # 1. Update multi-step prediction heads
        for k, target_obs in future_observations.items():
            if k not in self.W_preds:
                continue
            pred_obs = self.last_predictions.get(k)
            if pred_obs is None:
                continue

            error_vec = target_obs - pred_obs
            mse = float(np.mean(error_vec ** 2))
            self.last_errors[k] = mse
            total_err_mse += mse
            horizon_count += 1

            # Head update: outer product between feature and error
            dW_pred = self.learning_rate * np.outer(feature, error_vec) - self.weight_decay * self.W_preds[k]
            db_pred = self.learning_rate * error_vec

            self.W_preds[k] += np.clip(dW_pred, -1.0, 1.0)
            self.b_preds[k] += np.clip(db_pred, -1.0, 1.0)

            # Backproject error to latent space via Feedback Alignment matrix B_fa
            delta_z_accum += np.dot(error_vec, self.B_fa[k])

        mean_mse = total_err_mse / max(1, horizon_count)

        # 2. Update latent dynamics weights (W_hz, W_zz, W_zh)
        # Neuromodulated error signal
        modulation = float(np.tanh(mean_mse * 10.0))

        # Eligibility traces
        self.trace_hz = self.trace_gamma * self.trace_hz + np.outer(recurrent_act, self.z)
        self.trace_zz = self.trace_gamma * self.trace_zz + np.outer(self.prev_z, self.z)

        # Plasticity with feedback alignment & trace modulation
        dW_hz = (
            self.learning_rate * (np.outer(recurrent_act, delta_z_accum) + modulation * self.trace_hz)
            - self.weight_decay * self.W_hz
        )
        dW_zz = (
            self.learning_rate * (np.outer(self.prev_z, delta_z_accum) + modulation * self.trace_zz)
            - self.weight_decay * self.W_zz
        )
        dW_zh = (
            self.learning_rate * np.outer(self.z, recurrent_act)
            - self.weight_decay * self.W_zh
        )

        self.W_hz += np.clip(dW_hz, -1.0, 1.0)
        self.W_zz += np.clip(dW_zz, -1.0, 1.0)
        self.W_zh += np.clip(dW_zh, -1.0, 1.0)

        # Weight clamping for stability
        self.W_hz = np.clip(self.W_hz, -3.0, 3.0)
        self.W_zz = np.clip(self.W_zz, -3.0, 3.0)
        self.W_zh = np.clip(self.W_zh, -3.0, 3.0)

        dw_norm = float(np.linalg.norm(dW_hz) + np.linalg.norm(dW_zz))
        return {
            "mean_multistep_mse": mean_mse,
            "dw_norm": dw_norm,
            "modulation": modulation,
        }

    def adapt_dimensions(self, new_hidden_dim: int) -> None:
        """Dynamically adapt projection matrices when recurrent hidden_dim expands."""
        old_h = self.hidden_dim
        if new_hidden_dim <= old_h:
            return

        added = new_hidden_dim - old_h
        self.hidden_dim = new_hidden_dim

        # Expand W_hz
        scale_hz = np.sqrt(2.0 / (new_hidden_dim + self.latent_dim))
        new_W_hz = np.zeros((new_hidden_dim, self.latent_dim), dtype=np.float32)
        new_W_hz[:old_h, :] = self.W_hz
        new_W_hz[old_h:, :] = self.rng.normal(0.0, scale_hz, (added, self.latent_dim))
        self.W_hz = new_W_hz

        # Expand W_zh
        scale_zh = np.sqrt(2.0 / (self.latent_dim + new_hidden_dim))
        new_W_zh = np.zeros((self.latent_dim, new_hidden_dim), dtype=np.float32)
        new_W_zh[:, :old_h] = self.W_zh
        new_W_zh[:, old_h:] = self.rng.normal(0.0, scale_zh, (self.latent_dim, added))
        self.W_zh = new_W_zh

        # Expand trace
        new_trace = np.zeros((new_hidden_dim, self.latent_dim), dtype=np.float32)
        new_trace[:old_h, :] = self.trace_hz
        self.trace_hz = new_trace
