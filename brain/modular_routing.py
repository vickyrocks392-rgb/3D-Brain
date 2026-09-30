"""Phase 20 Track A: Context-Gated Modular Routing & Branch-Specific Plasticity.

Provides modular downstream readout branches coordinated by an inspectable context gate
to eliminate linear readout interference without arbitrary capacity growth:

   TEMPORAL CONTEXT REPRESENTATION (x in R^Din)
                 │
                 ├──► CONTEXT ROUTER (Soft / Hard WTA / Learned Oja)
                 │         │
                 │         ▼
                 │     Gate Distribution g in [0, 1]^M
                 │         │
                 ▼         ▼
       MODULAR READOUT BRANCHES (B0, B1, ... B_{M-1})
       (Branch-Specific Dendritic Plasticity & Consolidation)
                 │
                 ▼
       COMPETITIVE ACTION EVIDENCE u in R^K
"""

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from brain.synaptic_consolidation import ConsolidationMode, SynapticConsolidationModule


class RoutingCondition(str, Enum):
    """Routing mechanism condition for Track A evaluation."""
    A0_SHARED_LINEAR = "A0_SHARED_LINEAR"
    A1_FIXED_MODULAR = "A1_FIXED_MODULAR"
    A2_CONTEXT_GATED_SOFT = "A2_CONTEXT_GATED_SOFT"
    A3_LEARNED_SPARSE_GATE = "A3_LEARNED_SPARSE_GATE"
    A4_BRANCH_SPECIFIC_PLASTICITY = "A4_BRANCH_SPECIFIC_PLASTICITY"
    A5_ORACLE_GATED = "A5_ORACLE_GATED"


class ModularReadoutBranch:
    """Individual modular readout branch with dedicated synaptic weights and plasticity."""

    def __init__(
        self,
        branch_id: int,
        input_dim: int = 40,
        num_actions: int = 2,
        lr: float = 0.05,
        tau_eligibility_ms: float = 300.0,
        dt_ms: float = 40.0,
        consolidation_mode: ConsolidationMode = ConsolidationMode.NONE,
        seed: int = 42,
    ):
        self.branch_id = branch_id
        self.input_dim = int(input_dim)
        self.num_actions = max(2, int(num_actions))
        self.lr = float(lr)
        self.tau_eligibility_ms = float(tau_eligibility_ms)
        self.dt_ms = float(dt_ms)
        self.decay_elig = float(np.exp(-dt_ms / max(1.0, tau_eligibility_ms)))
        self.seed = seed + branch_id * 101

        self.rng = np.random.default_rng(self.seed)

        # Base weights [input_dim, num_actions]
        limit = 1.0 / math.sqrt(max(1, self.input_dim))
        self.weights = self.rng.uniform(-limit, limit, size=(self.input_dim, self.num_actions)).astype(np.float32)
        self.initial_weights = self.weights.copy()
        self.biases = np.zeros(self.num_actions, dtype=np.float32)

        # Eligibility traces
        self.eligibility = np.zeros((self.input_dim, self.num_actions), dtype=np.float32)
        self.dendritic_eligibility = np.zeros((self.input_dim, self.num_actions), dtype=np.float32)

        # Synaptic Consolidation Subsystem
        self.consolidation_module = SynapticConsolidationModule(
            input_dim=self.input_dim,
            num_actions=self.num_actions,
            mode=consolidation_mode,
            seed=self.seed,
        )

        # Causal ablation and lesion controls
        self.is_lesioned: bool = False
        self.lesion_mask: np.ndarray = np.ones((self.input_dim, self.num_actions), dtype=np.float32)
        self.is_frozen: bool = False

        # Telemetry
        self.last_drives: np.ndarray = np.zeros(self.num_actions, dtype=np.float32)
        self.last_delta_w: np.ndarray = np.zeros_like(self.weights)
        self.cumulative_updates: float = 0.0

    def reset(self) -> None:
        """Reset transient dynamic state."""
        self.eligibility.fill(0.0)
        self.dendritic_eligibility.fill(0.0)
        self.last_drives.fill(0.0)
        self.last_delta_w.fill(0.0)
        self.consolidation_module.reset()

    def compute_drives(self, x: np.ndarray, v_dend: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute feedforward drives for this branch."""
        w_base = self.consolidation_module.compute_effective_weights(self.weights)
        w_eff = w_base * self.lesion_mask
        raw_drives = (x @ w_eff) + self.biases

        if v_dend is not None and len(v_dend) == self.num_actions:
            dend_factor = 1.0 + 0.3 * np.tanh((v_dend + 50.0) / 10.0)
            drives = raw_drives * dend_factor
        else:
            drives = raw_drives

        self.last_drives = drives.astype(np.float32)
        return self.last_drives

    def update_eligibility(
        self,
        x: np.ndarray,
        action_probs: np.ndarray,
        v_dend: Optional[np.ndarray] = None,
    ) -> None:
        """Update branch-specific dendritic eligibility trace."""
        outer_act = np.outer(x, action_probs)
        self.eligibility = self.eligibility * self.decay_elig + outer_act
        self.dendritic_eligibility = self.eligibility.copy()

    def step_plasticity(
        self,
        teaching_signal: np.ndarray,
        reward: float,
        modulation_factor: float = 1.0,
        is_reward_step: bool = False,
        replay_eligibility: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply synaptic plasticity update to this branch."""
        if self.is_frozen or modulation_factor <= 0.0:
            self.last_delta_w.fill(0.0)
            return self.last_delta_w

        if replay_eligibility is not None:
            raw_delta = self.lr * modulation_factor * replay_eligibility
            effective_elig = np.abs(replay_eligibility)
        else:
            raw_delta = self.lr * modulation_factor * (self.dendritic_eligibility * teaching_signal[np.newaxis, :])
            effective_elig = self.dendritic_eligibility

        # Synaptic consolidation modulation
        delta_w, _ = self.consolidation_module.modulate_plasticity_step(
            base_weights=self.weights,
            raw_delta_w=raw_delta,
            eligibility=effective_elig,
            teaching_signal=teaching_signal,
            reward=reward,
            is_reward_step=is_reward_step,
        )

        self.weights += delta_w * self.lesion_mask
        self.weights = np.clip(self.weights, -2.0, 2.0)
        self.last_delta_w = delta_w
        self.cumulative_updates += float(np.sum(np.abs(delta_w)))
        return delta_w


class ContextRouter:
    """Computes dynamic context gating distribution over modular branches."""

    def __init__(
        self,
        input_dim: int = 40,
        num_branches: int = 3,
        lr_gate: float = 0.02,
        seed: int = 42,
    ):
        self.input_dim = int(input_dim)
        self.num_branches = max(1, int(num_branches))
        self.lr_gate = float(lr_gate)
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        # Gate linear weights [input_dim, num_branches]
        w = np.zeros((self.input_dim, self.num_branches), dtype=np.float32)
        for m in range(min(self.num_branches, 6)):
            w[2 + m, m] = 2.5
        w += self.rng.normal(0.0, 0.05, size=w.shape).astype(np.float32)
        self.weights_gate = w.astype(np.float32)
        self.biases_gate = np.zeros(self.num_branches, dtype=np.float32)

        # Causal ablation controls
        self.is_ablated: bool = False  # If True, gates uniformly
        self.is_random: bool = False   # If True, chooses uniformly random branch
        self.is_frozen: bool = False

        # Telemetry
        self.last_logits: np.ndarray = np.zeros(self.num_branches, dtype=np.float32)
        self.last_distribution: np.ndarray = np.ones(self.num_branches, dtype=np.float32) / self.num_branches
        self.last_active_branch: int = 0

    def reset(self) -> None:
        self.last_logits.fill(0.0)
        self.last_distribution.fill(1.0 / self.num_branches)
        self.last_active_branch = 0

    def compute_gate(
        self,
        x: np.ndarray,
        condition: RoutingCondition,
        oracle_context_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Compute branch gating distribution g and selected branch index m*."""
        M = self.num_branches

        if self.is_ablated or condition == RoutingCondition.A1_FIXED_MODULAR:
            g = np.full(M, 1.0 / M, dtype=np.float32)
            m_star = 0
            self.last_distribution = g
            self.last_active_branch = m_star
            return g, m_star

        if self.is_random:
            m_star = int(self.rng.integers(0, M))
            g = np.zeros(M, dtype=np.float32)
            g[m_star] = 1.0
            self.last_distribution = g
            self.last_active_branch = m_star
            return g, m_star

        if condition == RoutingCondition.A0_SHARED_LINEAR:
            # Single shared branch
            g = np.zeros(M, dtype=np.float32)
            g[0] = 1.0
            self.last_distribution = g
            self.last_active_branch = 0
            return g, 0

        if condition == RoutingCondition.A5_ORACLE_GATED:
            # Ground-truth context routing (Oracle Upper Bound)
            c = 0 if oracle_context_id is None else (int(oracle_context_id) % M)
            g = np.zeros(M, dtype=np.float32)
            g[c] = 1.0
            self.last_distribution = g
            self.last_active_branch = c
            return g, c

        # Compute raw affinity logits: z = x^T W_g + b_g
        z = (x @ self.weights_gate) + self.biases_gate
        self.last_logits = z.astype(np.float32)

        if condition == RoutingCondition.A2_CONTEXT_GATED_SOFT:
            # Soft gating distribution: softmax(z)
            shifted = z - np.max(z)
            exp_z = np.exp(np.clip(shifted, -20.0, 20.0))
            g = exp_z / (np.sum(exp_z) + 1e-12)
            m_star = int(np.argmax(g))
        else:
            # A3 or A4: Hard winner-take-all routing: argmax(z)
            m_star = int(np.argmax(z))
            g = np.zeros(M, dtype=np.float32)
            g[m_star] = 1.0

        self.last_distribution = g.astype(np.float32)
        self.last_active_branch = m_star
        return self.last_distribution, m_star

    def update_gate(self, x: np.ndarray, active_branch_idx: int) -> None:
        """Competitive local Hebbian / Oja update for router affinity weights."""
        if self.is_frozen or self.is_ablated or self.is_random:
            return

        m = active_branch_idx
        # Oja's normalized competitive rule: delta_w = lr * (x - w * y)
        norm_x = np.linalg.norm(x)
        if norm_x > 1e-6:
            x_unit = x / norm_x
            w_col = self.weights_gate[:, m]
            delta = self.lr_gate * (x_unit - w_col)
            self.weights_gate[:, m] += delta
            # Renormalize column to unit sphere
            col_norm = np.linalg.norm(self.weights_gate[:, m])
            if col_norm > 1e-6:
                self.weights_gate[:, m] /= col_norm

    def compute_entropy(self) -> float:
        """Compute Shannon entropy of the current gating distribution: H(g)."""
        g = self.last_distribution
        p = g[g > 1e-12]
        if len(p) == 0:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    def compute_selectivity(self) -> float:
        """Compute branch selectivity metric: 1.0 - H(g) / log2(M)."""
        if self.num_branches <= 1:
            return 1.0
        h = self.compute_entropy()
        max_h = math.log2(self.num_branches)
        if max_h <= 1e-12:
            return 1.0
        return float(np.clip(1.0 - (h / max_h), 0.0, 1.0))

    def compute_sparsity(self) -> float:
        """Compute fraction of branches with gate probability < 0.05."""
        if self.num_branches <= 1:
            return 0.0
        inactive = np.sum(self.last_distribution < 0.05)
        return float(inactive / self.num_branches)


class ModularActionReadout:
    """Coordinates modular readout branches and context-gated routing."""

    def __init__(
        self,
        input_dim: int = 40,
        num_actions: int = 2,
        num_branches: int = 3,
        routing_condition: RoutingCondition = RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY,
        lr: float = 0.05,
        tau_eligibility_ms: float = 300.0,
        dt_ms: float = 40.0,
        consolidation_mode: ConsolidationMode = ConsolidationMode.NONE,
        seed: int = 42,
    ):
        self.input_dim = int(input_dim)
        self.num_actions = max(2, int(num_actions))
        self.num_branches = max(1, int(num_branches))
        self.routing_condition = routing_condition
        self.lr = float(lr)
        self.tau_eligibility_ms = float(tau_eligibility_ms)
        self.dt_ms = float(dt_ms)
        self.consolidation_mode = consolidation_mode
        self.seed = seed

        # Router
        self.router = ContextRouter(
            input_dim=self.input_dim,
            num_branches=self.num_branches,
            lr_gate=0.02,
            seed=seed,
        )

        # Modular Branches
        self.branches: List[ModularReadoutBranch] = [
            ModularReadoutBranch(
                branch_id=m,
                input_dim=self.input_dim,
                num_actions=self.num_actions,
                lr=self.lr,
                tau_eligibility_ms=self.tau_eligibility_ms,
                dt_ms=self.dt_ms,
                consolidation_mode=self.consolidation_mode,
                seed=self.seed,
            )
            for m in range(self.num_branches)
        ]

        # Causal ablation / cross-branch coupling
        self.cross_branch_leakage: float = 0.0  # Fraction of update leaked to inactive branches
        self.branch_plasticity_disabled: bool = False
        self.is_frozen: bool = False

        # Telemetry & Trial State
        self.last_drives: np.ndarray = np.zeros(self.num_actions, dtype=np.float32)
        self.last_gate: np.ndarray = np.zeros(self.num_branches, dtype=np.float32)
        self.last_active_branch: int = 0
        self.active_context_branch: int = 0
        self.latched_choice_branch: int = 0
        self.context_detected: bool = False
        self.branch_usage_counts: np.ndarray = np.zeros(self.num_branches, dtype=np.int32)

    def reset(self) -> None:
        """Reset router, working memory, and all branches."""
        self.router.reset()
        for b in self.branches:
            b.reset()
        self.last_drives.fill(0.0)
        self.last_gate.fill(0.0)
        self.last_active_branch = 0
        self.active_context_branch = 0
        self.latched_choice_branch = 0
        self.context_detected = False

    def set_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze synaptic plasticity and gating updates."""
        self.is_frozen = frozen
        self.router.is_frozen = frozen
        for b in self.branches:
            b.is_frozen = frozen

    def set_consolidation_mode(self, mode: Union[ConsolidationMode, str], **kwargs) -> None:
        """Set consolidation mode across all branches."""
        if isinstance(mode, str):
            mode = ConsolidationMode(mode)
        self.consolidation_mode = mode
        for b in self.branches:
            b.consolidation_module = SynapticConsolidationModule(
                input_dim=self.input_dim,
                num_actions=self.num_actions,
                mode=mode,
                seed=b.seed,
                **kwargs,
            )

    def forward(
        self,
        x: np.ndarray,
        v_dend: Optional[np.ndarray] = None,
        oracle_context_id: Optional[int] = None,
        is_choice_step: bool = False,
    ) -> np.ndarray:
        """Compute action evidence using context-gated modular branches."""
        has_context = (np.max(np.abs(x[2:5])) > 0.2) or (oracle_context_id is not None)
        if has_context or not self.context_detected:
            gate_dist, m_star = self.router.compute_gate(
                x=x,
                condition=self.routing_condition,
                oracle_context_id=oracle_context_id,
            )
            self.last_gate = gate_dist.copy()
            self.last_active_branch = m_star
            self.active_context_branch = m_star
            if has_context:
                self.context_detected = True
        else:
            m_star = self.active_context_branch
            gate_dist = np.zeros(self.num_branches, dtype=np.float32)
            gate_dist[m_star] = 1.0
            self.last_gate = gate_dist
            self.last_active_branch = m_star

        if is_choice_step:
            self.latched_choice_branch = self.last_active_branch

        self.branch_usage_counts[m_star] += 1

        if self.routing_condition in (
            RoutingCondition.A3_LEARNED_SPARSE_GATE,
            RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY,
            RoutingCondition.A5_ORACLE_GATED,
        ) and not self.router.is_ablated:
            # Winner-take-all routed branch
            drives = self.branches[m_star].compute_drives(x, v_dend=v_dend)
        else:
            # Mixture across branches (A0, A1, A2)
            drives = np.zeros(self.num_actions, dtype=np.float32)
            for m, b in enumerate(self.branches):
                g_m = float(gate_dist[m])
                if g_m > 1e-6:
                    drives += g_m * b.compute_drives(x, v_dend=v_dend)

        self.last_drives = drives.astype(np.float32)
        return self.last_drives

    def update_eligibility(
        self,
        x: np.ndarray,
        action_probs: np.ndarray,
        v_dend: Optional[np.ndarray] = None,
    ) -> None:
        """Update dendritic eligibility traces across branches."""
        if self.routing_condition == RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY and not self.branch_plasticity_disabled:
            # Strictly branch-specific: active context branch updates eligibility
            m_star = self.active_context_branch
            self.branches[m_star].update_eligibility(x, action_probs, v_dend=v_dend)
        else:
            # Shared or coupled eligibility update
            for b in self.branches:
                b.update_eligibility(x, action_probs, v_dend=v_dend)

    def step_plasticity(
        self,
        x: np.ndarray,
        teaching_signal: np.ndarray,
        reward: float,
        is_reward_step: bool = False,
        replay_eligibility: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Apply context-gated plasticity step."""
        if self.is_frozen:
            return self.get_telemetry()

        cond = self.routing_condition
        m_star = self.latched_choice_branch if is_reward_step else self.active_context_branch

        # Update context gate affinity weights only for learned gate condition on context steps
        if cond == RoutingCondition.A3_LEARNED_SPARSE_GATE and np.max(np.abs(x[2:5])) > 0.2:
            self.router.update_gate(x, m_star)

        if cond == RoutingCondition.A0_SHARED_LINEAR:
            # Shared linear readout (only branch 0 exists/is used)
            self.branches[0].step_plasticity(
                teaching_signal, reward, modulation_factor=1.0, is_reward_step=is_reward_step, replay_eligibility=replay_eligibility
            )

        elif cond in (RoutingCondition.A4_BRANCH_SPECIFIC_PLASTICITY, RoutingCondition.A5_ORACLE_GATED) and not self.branch_plasticity_disabled:
            # Strictly branch-specific plasticity: ONLY branch m_star updates weights!
            self.branches[m_star].step_plasticity(
                teaching_signal, reward, modulation_factor=1.0, is_reward_step=is_reward_step, replay_eligibility=replay_eligibility
            )
            # If causal cross-branch leakage is intentionally enabled for ablation
            if self.cross_branch_leakage > 0.0:
                leak_factor = self.cross_branch_leakage / max(1, self.num_branches - 1)
                for m, b in enumerate(self.branches):
                    if m != m_star:
                        b.step_plasticity(
                            teaching_signal, reward, modulation_factor=leak_factor, is_reward_step=is_reward_step, replay_eligibility=replay_eligibility
                        )

        elif cond == RoutingCondition.A2_CONTEXT_GATED_SOFT:
            # Soft-gated plasticity: each branch updates proportional to gate g_m
            for m, b in enumerate(self.branches):
                g_m = float(self.last_gate[m])
                if g_m > 1e-4:
                    b.step_plasticity(
                        teaching_signal, reward, modulation_factor=g_m, is_reward_step=is_reward_step, replay_eligibility=replay_eligibility
                    )

        else:
            # A1 (Fixed modular) or A3 (coupled plasticity): equal update
            for b in self.branches:
                b.step_plasticity(
                    teaching_signal, reward, modulation_factor=1.0 / self.num_branches, is_reward_step=is_reward_step, replay_eligibility=replay_eligibility
                )

        return self.get_telemetry()

    def get_telemetry(self) -> Dict[str, Any]:
        """Collect rich routing and branch telemetry."""
        m_star = self.last_active_branch
        active_b = self.branches[m_star]
        branch_acts = [float(np.mean(np.abs(b.last_drives))) for b in self.branches]
        branch_cum_updates = [float(b.cumulative_updates) for b in self.branches]

        return {
            "routing_condition": self.routing_condition.value,
            "active_branch": m_star,
            "gate_distribution": [round(float(g), 4) for g in self.last_gate],
            "gate_entropy": round(self.router.compute_entropy(), 4),
            "branch_selectivity": round(self.router.compute_selectivity(), 4),
            "gate_sparsity": round(self.router.compute_sparsity(), 4),
            "branch_activations": [round(a, 4) for a in branch_acts],
            "branch_cumulative_updates": [round(u, 4) for u in branch_cum_updates],
            "active_branch_weights_norm": round(float(np.linalg.norm(active_b.weights)), 4),
            "branch_usage_counts": [int(c) for c in self.branch_usage_counts],
        }

    def count_parameters(self) -> Dict[str, int]:
        """Explicit accounting of trainable and dynamic parameters."""
        branch_weights = sum(b.weights.size for b in self.branches)
        branch_biases = sum(b.biases.size for b in self.branches)
        gate_weights = self.router.weights_gate.size
        gate_biases = self.router.biases_gate.size

        total_trainable = branch_weights + branch_biases + (gate_weights + gate_biases if self.routing_condition != RoutingCondition.A0_SHARED_LINEAR else 0)
        eligibility_vars = sum(b.eligibility.size * 2 for b in self.branches)

        return {
            "branch_weights": branch_weights,
            "branch_biases": branch_biases,
            "gate_weights": gate_weights,
            "gate_biases": gate_biases,
            "total_trainable": total_trainable,
            "dynamic_state_vars": eligibility_vars,
            "total_memory_bytes": (total_trainable + eligibility_vars) * 4,
        }
