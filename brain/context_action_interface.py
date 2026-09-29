"""Learnable Context-to-Action Interface & Downstream Competitive Action Population (Phase 12).

Provides an explicit architectural separation between the frozen recurrent/attractor
substrate and a learnable downstream action-selection interface:

                  PHASE 11 SUBSTRATE (FROZEN)
                         │
                         │ stable latent state
                         ▼
              ┌──────────────────────┐
              │ CONTEXT READOUT      │
              │                      │
              │ learned interface    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ ACTION POPULATION    │
              │                      │
              │ A0 / A1 / A2 / A3    │
              └──────────┬───────────┘
                         │
                         ▼
                       ACTION

Key Features:
1. Flexible Representation Source Extraction:
   - SOMATIC, DENDRITIC, NMDA, ATTRACTOR, CONCATENATED, PCA, RANDOM_PROJECTION
2. Diverse Readout Architectures:
   - R0: Fixed Random Readout
   - R1: Linear Supervised Readout (Delta / LMS)
   - R2: Local Hebbian Readout (Correlation with decay/Oja)
   - R3: Three-Factor Interface Plasticity (Eligibility + Action Neuromodulation)
   - R4: Prediction-Error Gated Interface
   - R5: Delayed Reward + Isolated Interface Eligibility
   - R6: Analytical BPTT Reference Upper-Bound
   - R7: Phase 11 Fixed Action Routing Baseline
3. Competitive Action Population:
   - Configurable action accumulators (ACTION_0, ACTION_1, ACTION_2, ACTION_3, ...)
   - Reciprocal lateral inhibition, self-excitation, leaky integration
   - Softmax action probabilities, decision margin, action entropy
4. Credit Localization & Leakage Measurement:
   - Interface credit, collateral credit, recurrent leakage (0.0% when substrate frozen)
   - CSI, Credit Precision, Credit Leakage, Gini concentration, Causal Credit Ratio
5. Causal Intervention Suite:
   - Context scramble, context swap, readout lesion, irrelevant context injection,
     label permutation, freeze plasticity
6. Phase 11 Lesion Paradox Diagnostics Suite:
   - Intact, relevant lesion, irrelevant lesion, both lesion, cross-inh disabled, cross-inh reversed
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class RepresentationSource(str, Enum):
    """Source of latent representation extracted from the biological substrate."""
    SOMATIC = "SOMATIC"
    DENDRITIC = "DENDRITIC"
    NMDA = "NMDA"
    ATTRACTOR = "ATTRACTOR"
    CONCATENATED = "CONCATENATED"
    PCA = "PCA"
    RANDOM_PROJ = "RANDOM_PROJ"


class ReadoutArchitecture(str, Enum):
    """Readout learning and routing architecture."""
    R0_RANDOM = "R0_RANDOM"
    R1_LINEAR_SUPERVISED = "R1_LINEAR_SUPERVISED"
    R2_HEBBIAN = "R2_HEBBIAN"
    R3_THREE_FACTOR = "R3_THREE_FACTOR"
    R4_PRED_ERROR = "R4_PRED_ERROR"
    R5_DELAYED_REWARD = "R5_DELAYED_REWARD"
    R6_ANALYTICAL_BPTT = "R6_ANALYTICAL_BPTT"
    R7_PHASE11_ROUTING = "R7_PHASE11_ROUTING"


class ActionCompetitionMode(str, Enum):
    """Competitive lateral interaction regime within action population."""
    NO_INHIBITION = "NO_INHIBITION"          # w_cross = 0.0, w_self = 0.2
    WEAK_INHIBITION = "WEAK_INHIBITION"      # w_cross = 0.2, w_self = 0.2
    BASELINE_INHIBITION = "BASELINE"        # w_cross = 0.5, w_self = 0.2
    STRONG_INHIBITION = "STRONG_INHIBITION"  # w_cross = 1.0, w_self = 0.2
    NO_RECURRENT_EXC = "NO_RECURRENT_EXC"    # w_cross = 0.5, w_self = 0.0
    NO_RECURRENCE = "NO_RECURRENCE"          # w_cross = 0.0, w_self = 0.0


class ContextActionInterface:
    """Manages the learnable context-to-action interface and competitive action population."""

    def __init__(
        self,
        hidden_dim: int = 50,
        num_actions: int = 4,
        source: RepresentationSource = RepresentationSource.ATTRACTOR,
        architecture: ReadoutArchitecture = ReadoutArchitecture.R3_THREE_FACTOR,
        competition_mode: ActionCompetitionMode = ActionCompetitionMode.BASELINE_INHIBITION,
        tau_action_ms: float = 10.0,
        tau_eligibility_ms: float = 50.0,
        learning_rate: float = 0.02,
        pca_dim: int = 10,
        random_proj_dim: int = 16,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_actions = max(2, int(num_actions))
        self.source = source
        self.architecture = architecture
        self.competition_mode = competition_mode
        self.tau_action_ms = float(tau_action_ms)
        self.tau_eligibility_ms = float(tau_eligibility_ms)
        self.learning_rate = float(learning_rate)
        self.pca_dim = min(pca_dim, hidden_dim)
        self.random_proj_dim = random_proj_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 1. Determine Context Input Dimension based on Source
        self.context_dim = self._resolve_context_dim()

        # Fixed PCA / Random projection matrices if needed
        self._pca_matrix: Optional[np.ndarray] = None
        self._rand_proj_matrix: Optional[np.ndarray] = None
        if self.source == RepresentationSource.PCA:
            # Deterministic orthogonal projection
            concat_d = hidden_dim * 4
            mat = self.rng.normal(0.0, 1.0 / np.sqrt(concat_d), (concat_d, self.pca_dim)).astype(np.float32)
            q, _ = np.linalg.qr(mat)
            self._pca_matrix = q.astype(np.float32)
        elif self.source == RepresentationSource.RANDOM_PROJ:
            concat_d = hidden_dim * 4
            self._rand_proj_matrix = self.rng.normal(
                0.0, 1.0 / np.sqrt(concat_d), (concat_d, self.random_proj_dim)
            ).astype(np.float32)

        # 2. Feedforward Interface Weights: W_ctx_to_act [context_dim, num_actions]
        self.W_ctx_to_act = self._init_interface_weights()
        self.b_action = np.zeros(self.num_actions, dtype=np.float32)
        self._W_initial = self.W_ctx_to_act.copy()

        # 3. Action Population Competition Weights
        self.w_self, self.w_cross = self._resolve_competition_weights()

        # 4. Dynamic Action States
        self.v_action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_action_probs = np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        self.last_winner_margin = 0.0
        self.last_action_entropy = float(np.log(self.num_actions))
        self.last_selected_action = 0

        # 5. Interface Eligibility Traces [context_dim, num_actions]
        self.eligibility = np.zeros((self.context_dim, self.num_actions), dtype=np.float32)
        self.alpha_eligibility = float(np.exp(-1.0 / max(1.0, self.tau_eligibility_ms)))

        # 6. Plasticity Tracking & Credit Assignment
        self.plasticity_enabled = True
        self.last_dw = np.zeros_like(self.W_ctx_to_act)
        self.accumulated_causal_credit = 0.0
        self.accumulated_collateral_credit = 0.0
        self.accumulated_recurrent_leakage = 0.0  # Stays 0.0 when substrate is frozen
        self.last_csi = 0.0
        self.last_precision = 0.0
        self.last_leakage = 0.0

        # 7. Lesion & Intervention Flags
        self._is_lesioned = False
        self._lesion_mask: Optional[np.ndarray] = None
        self._context_scramble = False
        self._context_swap = False
        self._irrelevant_context = False
        self._label_permuted = False
        self._label_perm: Optional[np.ndarray] = None

    def _resolve_context_dim(self) -> int:
        """Resolve dimensionality of extracted context feature vector."""
        if self.source in (
            RepresentationSource.SOMATIC,
            RepresentationSource.DENDRITIC,
            RepresentationSource.NMDA,
            RepresentationSource.ATTRACTOR,
        ):
            return self.hidden_dim
        elif self.source == RepresentationSource.CONCATENATED:
            return self.hidden_dim * 4
        elif self.source == RepresentationSource.PCA:
            return self.pca_dim
        elif self.source == RepresentationSource.RANDOM_PROJ:
            return self.random_proj_dim
        return self.hidden_dim

    def _init_interface_weights(self) -> np.ndarray:
        """Initialize context-to-action synaptic weights based on architecture."""
        std = 1.0 / np.sqrt(max(1, self.context_dim))
        W = self.rng.normal(0.0, std, (self.context_dim, self.num_actions)).astype(np.float32)

        if self.architecture == ReadoutArchitecture.R7_PHASE11_ROUTING:
            # Replicate exact Phase 11 initial alignment
            half_h = min(self.context_dim // 2, self.context_dim)
            W[:half_h, 0] += 0.55
            if self.num_actions > 1:
                W[half_h:, 1] += 0.55
        elif self.architecture == ReadoutArchitecture.R0_RANDOM:
            # Frozen random projection - scaled for initial differentiation
            W *= 1.5

        return W

    def _resolve_competition_weights(self) -> Tuple[float, float]:
        """Configure intra-action excitation and lateral inhibition."""
        mode = self.competition_mode
        if mode == ActionCompetitionMode.NO_INHIBITION:
            return 0.20, 0.0
        elif mode == ActionCompetitionMode.WEAK_INHIBITION:
            return 0.20, 0.20
        elif mode == ActionCompetitionMode.BASELINE_INHIBITION:
            return 0.20, 0.50
        elif mode == ActionCompetitionMode.STRONG_INHIBITION:
            return 0.20, 1.00
        elif mode == ActionCompetitionMode.NO_RECURRENT_EXC:
            return 0.0, 0.50
        elif mode == ActionCompetitionMode.NO_RECURRENCE:
            return 0.0, 0.0
        return 0.20, 0.50

    def reset(self) -> None:
        """Reset dynamic accumulator and eligibility states while preserving learned weights."""
        self.v_action.fill(0.0)
        self.last_action_probs.fill(1.0 / self.num_actions)
        self.last_winner_margin = 0.0
        self.last_action_entropy = float(np.log(self.num_actions))
        self.last_selected_action = 0
        self.eligibility.fill(0.0)
        self.last_dw.fill(0.0)

    # -------------------------------------------------------------------------
    # Context Feature Extraction
    # -------------------------------------------------------------------------
    def extract_context(
        self,
        substrate_state: Union[Dict[str, Any], np.ndarray],
    ) -> np.ndarray:
        """Extract the designated context representation from biophysical substrate states."""
        if isinstance(substrate_state, np.ndarray):
            raw = substrate_state.astype(np.float32)
            if raw.ndim == 2:
                raw = raw[0]
            v_s = raw[: self.hidden_dim] if len(raw) >= self.hidden_dim else np.pad(raw, (0, self.hidden_dim - len(raw)))
            v_d = v_s.copy()
            g_n = np.zeros(self.hidden_dim, dtype=np.float32)
            act = np.maximum(0.0, v_s)
        elif isinstance(substrate_state, dict):
            v_s = np.asarray(substrate_state.get("v_soma", np.zeros(self.hidden_dim)), dtype=np.float32)
            v_d = np.asarray(substrate_state.get("v_dend", np.zeros(self.hidden_dim)), dtype=np.float32)
            g_n = np.asarray(substrate_state.get("g_nmda", np.zeros(self.hidden_dim)), dtype=np.float32)
            act = np.asarray(substrate_state.get("activations", np.zeros(self.hidden_dim)), dtype=np.float32)
        else:
            v_s = np.zeros(self.hidden_dim, dtype=np.float32)
            v_d = v_s.copy()
            g_n = v_s.copy()
            act = v_s.copy()

        src = self.source
        if src == RepresentationSource.SOMATIC:
            ctx = v_s.copy()
        elif src == RepresentationSource.DENDRITIC:
            ctx = v_d.copy()
        elif src == RepresentationSource.NMDA:
            ctx = g_n.copy()
        elif src == RepresentationSource.ATTRACTOR:
            ctx = act.copy()
        elif src in (RepresentationSource.CONCATENATED, RepresentationSource.PCA, RepresentationSource.RANDOM_PROJ):
            # Scale each feature family to comparable standard deviation
            concat = np.concatenate([
                v_s / (np.std(v_s) + 1.0),
                v_d / (np.std(v_d) + 1.0),
                g_n / (np.std(g_n) + 1e-4),
                act / (np.std(act) + 1e-4),
            ]).astype(np.float32)
            if src == RepresentationSource.CONCATENATED:
                ctx = concat
            elif src == RepresentationSource.PCA and self._pca_matrix is not None:
                ctx = concat @ self._pca_matrix
            elif src == RepresentationSource.RANDOM_PROJ and self._rand_proj_matrix is not None:
                ctx = concat @ self._rand_proj_matrix
            else:
                ctx = concat[: self.context_dim]
        else:
            ctx = act.copy()

        # Apply active causal interventions if requested
        if self._irrelevant_context:
            ctx = self.rng.normal(0.0, 1.0, len(ctx)).astype(np.float32)
        elif self._context_scramble:
            perm = self.rng.permutation(len(ctx))
            ctx = ctx[perm]
        elif self._context_swap:
            half = len(ctx) // 2
            ctx = np.concatenate([ctx[half:], ctx[:half]])

        return ctx

    # -------------------------------------------------------------------------
    # Core Step: Integration & Winner Selection
    # -------------------------------------------------------------------------
    def step(
        self,
        context_input: np.ndarray,
        dt_ms: float = 1.0,
        reward: float = 0.0,
        target_action: Optional[int] = None,
        context_id: Optional[int] = None,
        is_choice_step: bool = False,
        is_reward_step: bool = False,
    ) -> Tuple[np.ndarray, int, float]:
        """Advance action interface by one timestep dt.

        Args:
            context_input: Context feature vector [context_dim]
            dt_ms: Simulation timestep
            reward: Teaching/feedback reward R(t)
            target_action: Ground-truth target action index
            context_id: Current contextual cue ID
            is_choice_step: True if this timestep produces the trial's action choice
            is_reward_step: True if delayed reward arrives on this timestep

        Returns:
            v_action: Continuous action accumulator activations [num_actions]
            selected_action: Winning action index (0 .. num_actions-1)
            winner_margin: Decision margin between 1st and 2nd action
        """
        x_ctx = np.asarray(context_input, dtype=np.float32)
        if len(x_ctx) < self.context_dim:
            x_ctx = np.pad(x_ctx, (0, self.context_dim - len(x_ctx)))
        elif len(x_ctx) > self.context_dim:
            x_ctx = x_ctx[: self.context_dim]

        # 1. Feedforward Context Drive: I_ff = x_ctx @ W_ctx_to_act + b
        i_ff = x_ctx @ self.W_ctx_to_act + self.b_action

        # 2. Competitive Recurrent Interactions within Action Population
        pos_act = np.maximum(0.0, self.v_action)
        sum_pos = float(np.sum(pos_act))
        # Reciprocal lateral inhibition from competing action channels
        i_inh = self.w_cross * (sum_pos - pos_act)
        # Recurrent self-excitation
        i_exc = self.w_self * pos_act
        net_recurrent = i_exc - i_inh

        # 3. Leaky Integration: dv/dt = (-v + I_ff + I_rec) / tau
        decay = float(np.exp(-dt_ms / max(1.0, self.tau_action_ms)))
        self.v_action = self.v_action * decay + (1.0 - decay) * (i_ff + net_recurrent)

        # 4. Softmax Probabilities, Winner Margin & Action Entropy
        v_shifted = self.v_action - np.max(self.v_action)
        exp_v = np.exp(np.clip(v_shifted, -20.0, 20.0))
        self.last_action_probs = (exp_v / (np.sum(exp_v) + 1e-12)).astype(np.float32)

        sorted_v = np.sort(self.v_action)[::-1]
        self.last_winner_margin = float(sorted_v[0] - sorted_v[1]) if len(sorted_v) > 1 else float(sorted_v[0])
        probs = np.clip(self.last_action_probs, 1e-12, 1.0)
        self.last_action_entropy = float(-np.sum(probs * np.log(probs)))

        # 5. Winning Action (Argmax)
        self.last_selected_action = int(np.argmax(self.v_action))

        # 6. Update Synaptic Eligibility Traces
        # Outer product of pre-synaptic context and post-synaptic action activity
        pre_ctx = x_ctx[:, np.newaxis]
        post_act = self.last_action_probs[np.newaxis, :]
        self.eligibility = self.eligibility * self.alpha_eligibility + (1.0 - self.alpha_eligibility) * (pre_ctx @ post_act)

        # 7. Apply Synaptic Plasticity based on Architecture
        if self.plasticity_enabled and self.architecture != ReadoutArchitecture.R0_RANDOM and self.architecture != ReadoutArchitecture.R7_PHASE11_ROUTING:
            self._apply_interface_plasticity(
                x_ctx=x_ctx,
                target_action=target_action,
                reward=reward,
                context_id=context_id,
                is_choice_step=is_choice_step,
                is_reward_step=is_reward_step,
            )

        return self.v_action.copy(), self.last_selected_action, self.last_winner_margin

    # -------------------------------------------------------------------------
    # Interface Synaptic Plasticity Rules
    # -------------------------------------------------------------------------
    def _apply_interface_plasticity(
        self,
        x_ctx: np.ndarray,
        target_action: Optional[int],
        reward: float,
        context_id: Optional[int],
        is_choice_step: bool,
        is_reward_step: bool,
    ) -> None:
        """Execute the chosen local plasticity rule on W_ctx_to_act."""
        rule = self.architecture
        dw = np.zeros_like(self.W_ctx_to_act)

        # R1: Linear Supervised Readout (Delta / LMS Error Correction)
        if rule == ReadoutArchitecture.R1_LINEAR_SUPERVISED:
            if target_action is not None and is_choice_step:
                one_hot = np.zeros(self.num_actions, dtype=np.float32)
                tgt = target_action if not self._label_permuted else (self._label_perm[target_action] if self._label_perm is not None else target_action)
                one_hot[tgt] = 1.0
                error = one_hot - self.last_action_probs
                # Local delta rule: dW = eta * x_ctx outer error
                dw = self.learning_rate * np.outer(x_ctx, error)

        # R2: Local Hebbian / Correlation Readout (Oja-normalized)
        elif rule == ReadoutArchitecture.R2_HEBBIAN:
            # pre_context * post_action with weight decay
            post = self.last_action_probs
            hebb = np.outer(x_ctx, post)
            # Oja's stabilization: dW = eta * (x*y - y^2 * W)
            oja_decay = np.outer(np.ones(self.context_dim), post ** 2) * self.W_ctx_to_act
            dw = self.learning_rate * 0.1 * (hebb - 0.5 * oja_decay)

        # R3: Three-Factor Interface Plasticity
        elif rule == ReadoutArchitecture.R3_THREE_FACTOR:
            # Factor 1: pre_context, Factor 2: post_action (in eligibility trace)
            # Factor 3: M_action local teaching/reward signal
            m_action = 0.0
            if is_choice_step and target_action is not None:
                tgt = target_action if not self._label_permuted else (self._label_perm[target_action] if self._label_perm is not None else target_action)
                chosen = self.last_selected_action
                m_action = 1.0 if chosen == tgt else -1.0
            elif is_reward_step:
                m_action = float(reward)

            if abs(m_action) > 1e-4:
                dw = self.learning_rate * m_action * self.eligibility

        # R4: Prediction-Error Gated Interface
        elif rule == ReadoutArchitecture.R4_PRED_ERROR:
            # Action prediction error modifies only context->action synapses
            if target_action is not None and (is_choice_step or is_reward_step):
                one_hot = np.zeros(self.num_actions, dtype=np.float32)
                tgt = target_action if not self._label_permuted else (self._label_perm[target_action] if self._label_perm is not None else target_action)
                one_hot[tgt] = 1.0
                action_pe = one_hot - self.last_action_probs
                dw = self.learning_rate * (self.eligibility @ np.diag(action_pe))

        # R5: Delayed Reward + Isolated Interface Eligibility
        elif rule == ReadoutArchitecture.R5_DELAYED_REWARD:
            # Reward delivered strictly on reward step to accumulated eligibility
            if is_reward_step and abs(reward) > 1e-4:
                dw = self.learning_rate * reward * self.eligibility

        # R6: Analytical BPTT Readout (Exact Gradient Step on Action Readout Layer)
        elif rule == ReadoutArchitecture.R6_ANALYTICAL_BPTT:
            if target_action is not None and is_choice_step:
                one_hot = np.zeros(self.num_actions, dtype=np.float32)
                tgt = target_action if not self._label_permuted else (self._label_perm[target_action] if self._label_perm is not None else target_action)
                one_hot[tgt] = 1.0
                grad = self.last_action_probs - one_hot
                dw = -self.learning_rate * np.outer(x_ctx, grad)

        # Apply weight update if any
        if np.any(np.abs(dw) > 1e-8):
            self.W_ctx_to_act += dw
            # Soft weight clipping to prevent pathological unbounded growth
            np.clip(self.W_ctx_to_act, -3.0, 3.0, out=self.W_ctx_to_act)
            self.last_dw = dw

            # Compute and record synaptic credit specificity metrics
            self._update_credit_metrics(dw, context_id=context_id, target_action=target_action)

    # -------------------------------------------------------------------------
    # Credit Specificity, Precision & Leakage Computation
    # -------------------------------------------------------------------------
    def _update_credit_metrics(
        self,
        dw: np.ndarray,
        context_id: Optional[int],
        target_action: Optional[int],
    ) -> None:
        """Measure credit specificity, precision, and leakage against Phase 11 baseline."""
        abs_dw = np.abs(dw)
        total_interface_credit = float(np.sum(abs_dw))
        if total_interface_credit < 1e-9:
            return

        # Determine causal synapses for the current context
        # For binary tasks: Context 0 -> Pool 0 (first half) -> Action 0
        #                   Context 1 -> Pool 1 (second half) -> Action 1
        causal_mask = np.zeros_like(dw, dtype=bool)
        ctx_k = context_id if context_id is not None else (target_action if target_action is not None else 0)
        half = self.context_dim // self.num_actions
        start_idx = ctx_k * half
        end_idx = (ctx_k + 1) * half if ctx_k < self.num_actions - 1 else self.context_dim

        causal_act = target_action if target_action is not None else ctx_k
        if causal_act < self.num_actions:
            causal_mask[start_idx:end_idx, causal_act] = True

        causal_credit = float(np.sum(abs_dw[causal_mask]))
        collateral_credit = float(np.sum(abs_dw[~causal_mask]))
        recurrent_leakage = 0.0  # ZERO! Substrate is frozen!

        self.accumulated_causal_credit += causal_credit
        self.accumulated_collateral_credit += collateral_credit
        self.accumulated_recurrent_leakage += recurrent_leakage

        denom = causal_credit + collateral_credit + recurrent_leakage + 1e-12
        self.last_csi = float((causal_credit - (collateral_credit + recurrent_leakage)) / denom)
        self.last_precision = float(causal_credit / denom)
        self.last_leakage = float((collateral_credit + recurrent_leakage) / denom)

    def get_credit_summary(self) -> Dict[str, float]:
        """Return cumulative credit assignment metrics."""
        c_causal = self.accumulated_causal_credit
        c_collat = self.accumulated_collateral_credit
        c_rec = self.accumulated_recurrent_leakage
        total = c_causal + c_collat + c_rec + 1e-12

        csi = float((c_causal - (c_collat + c_rec)) / total)
        prec = float(c_causal / total)
        leak = float((c_collat + c_rec) / total)
        ccr = float(c_causal / (c_causal + c_collat + 1e-12))

        # Gini concentration of learned weights
        w_flat = np.sort(np.abs(self.W_ctx_to_act.ravel()))
        n = len(w_flat)
        index = np.arange(1, n + 1)
        gini = float((np.sum((2 * index - n - 1) * w_flat)) / (n * np.sum(w_flat) + 1e-12))

        return {
            "csi": round(csi, 4),
            "precision": round(prec, 4),
            "leakage": round(leak, 4),
            "recurrent_leakage": round(float(c_rec / total), 4),
            "interface_credit": round(c_causal, 5),
            "collateral_credit": round(c_collat, 5),
            "causal_credit_ratio": round(ccr, 4),
            "gini_concentration": round(gini, 4),
        }

    # -------------------------------------------------------------------------
    # Causal Interventions Suite (Section 13)
    # -------------------------------------------------------------------------
    def apply_context_scramble(self, enabled: bool = True) -> None:
        """C1: Randomly permute context representations before action decoding."""
        self._context_scramble = enabled

    def apply_context_swap(self, enabled: bool = True) -> None:
        """C2: Swap context representation halves (Context A <-> Context B)."""
        self._context_swap = enabled

    def apply_readout_lesion(self, fraction: float = 0.50, seed: Optional[int] = None) -> int:
        """C3: Zero out selected context->action interface weights."""
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        total_synapses = self.W_ctx_to_act.size
        num_to_lesion = int(round(fraction * total_synapses))
        flat_indices = rng.choice(total_synapses, size=num_to_lesion, replace=False)

        mask = np.ones(total_synapses, dtype=bool)
        mask[flat_indices] = False
        self._lesion_mask = mask.reshape(self.W_ctx_to_act.shape)

        self.W_ctx_to_act[~self._lesion_mask] = 0.0
        self._is_lesioned = True
        return num_to_lesion

    def restore_lesion(self) -> None:
        """Restore all interface weights to pre-lesion backup."""
        self.W_ctx_to_act = self._W_initial.copy()
        self._is_lesioned = False
        self._lesion_mask = None
        self._context_scramble = False
        self._context_swap = False
        self._irrelevant_context = False
        self._label_permuted = False

    def apply_irrelevant_context(self, enabled: bool = True) -> None:
        """C4: Replace context with uncorrelated noise."""
        self._irrelevant_context = enabled

    def apply_label_permutation(self, enabled: bool = True) -> None:
        """C5: Permute context-to-action contingency mapping."""
        self._label_permuted = enabled
        if enabled:
            self._label_perm = self.rng.permutation(self.num_actions)
        else:
            self._label_perm = None

    def freeze_plasticity(self, frozen: bool = True) -> None:
        """C6: Stop synaptic updates to test post-convergence stability."""
        self.plasticity_enabled = not frozen

    # -------------------------------------------------------------------------
    # Phase 11 Lesion Paradox Replication Suite (Section 14)
    # -------------------------------------------------------------------------
    def configure_lesion_paradox_condition(self, condition: str) -> None:
        """Configure exact circuit lesions for testing the Phase 11 disinhibition paradox.

        Conditions:
        - INTACT: Normal routing and mutual cross-inhibition
        - RELEVANT_LESION: Zero out projections from relevant pool (Pool A -> Action 0)
        - IRRELEVANT_LESION: Zero out projections from irrelevant pool (Pool B -> Action 1)
        - BOTH_LESION: Zero out both projection pathways
        - CROSS_INH_DISABLED: Mutual cross-inhibition W_cross = 0.0
        - CROSS_INH_REVERSED: Mutual cross-inhibition W_cross = +0.5 (reciprocal excitation)
        """
        self.restore_lesion()
        half_h = self.context_dim // 2

        if condition == "INTACT":
            self.w_cross = 0.50
        elif condition == "RELEVANT_LESION":
            self.W_ctx_to_act[:half_h, 0] = 0.0
            self.w_cross = 0.50
        elif condition == "IRRELEVANT_LESION":
            self.W_ctx_to_act[half_h:, 1] = 0.0
            self.w_cross = 0.50
        elif condition == "BOTH_LESION":
            self.W_ctx_to_act[:half_h, 0] = 0.0
            self.W_ctx_to_act[half_h:, 1] = 0.0
            self.w_cross = 0.50
        elif condition == "CROSS_INH_DISABLED":
            self.w_cross = 0.0
        elif condition == "CROSS_INH_REVERSED":
            self.w_cross = -0.50  # Sign reversed in i_inh gives reciprocal excitation
        else:
            raise ValueError(f"Unknown lesion condition: {condition}")

    # -------------------------------------------------------------------------
    # Telemetry Dictionary
    # -------------------------------------------------------------------------
    def get_telemetry(self) -> Dict[str, Any]:
        """Assemble comprehensive diagnostic telemetry for Observatory & BrainLog."""
        cred = self.get_credit_summary()
        dw_norm = float(np.linalg.norm(self.last_dw))
        e_norm = float(np.linalg.norm(self.eligibility))

        return {
            "context_source": self.source.value,
            "context_dim": self.context_dim,
            "num_actions": self.num_actions,
            "action_evidence": [round(float(v), 3) for v in self.v_action],
            "action_probs": [round(float(p), 4) for p in self.last_action_probs],
            "selected_action": self.last_selected_action,
            "winner_margin": round(self.last_winner_margin, 4),
            "action_entropy": round(self.last_action_entropy, 4),
            "learning_rule": self.architecture.value,
            "plasticity_enabled": self.plasticity_enabled,
            "dw_norm": dw_norm,
            "eligibility_norm": e_norm,
            "eligibility_updated": bool(e_norm > 1e-4),
            "credit_assigned": bool(dw_norm > 1e-6),
            "w_self": round(self.w_self, 2),
            "w_cross": round(self.w_cross, 2),
            **cred,
        }
