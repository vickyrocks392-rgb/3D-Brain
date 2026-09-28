"""Selective Action-Selection Routing Circuit & Causal Pathway Lesioning (Phase 11).

Routes stabilized context attractor activity into distinct downstream action-selection populations
(Action A vs. Action B) with mutual competitive cross-inhibition.

Key Features:
1. Downstream Action Populations:
   - Action Population A: corresponds to Choice A (Target = -1.0 or Action 0)
   - Action Population B: corresponds to Choice B (Target = +1.0 or Action 1)
2. Competitive Mutual Inhibition:
   - Winner-take-all / mutually inhibitory dynamics enforce clear decision boundaries
     and prevent ambiguous co-activation.
3. Information-Theoretic Routing Metrics:
   - Action Selectivity Index
   - Context-Action Mutual Information I(Context; Action)
   - Winner Decision Margin
   - Incorrect Action Activation Ratio
4. Causal Pathway Lesion Engine:
   - Relevant Pathway Lesion: zeroes synaptic weights along the causal context -> attractor -> action route
   - Irrelevant Pathway Lesion: zeroes an equal number of matched non-causal weights
   - Enables direct empirical verification of causal specificity.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ActionSelectionCircuit:
    """Manages downstream action-selection populations and competitive mutual inhibition."""

    def __init__(
        self,
        num_hidden: int,
        num_actions: int = 2,
        tau_action: float = 10.0,
        mutual_inhibition: float = 0.50,
        decision_threshold: float = 0.35,
        seed: int = 42,
    ):
        self.num_hidden = num_hidden
        self.num_actions = num_actions
        self.tau_action = float(tau_action)
        self.mutual_inhibition = float(mutual_inhibition)
        self.decision_threshold = float(decision_threshold)
        self.rng = np.random.RandomState(seed)

        # 1. Routing Projection Weights (Hidden Attractor -> Action Populations)
        # Seeded so Pool A projects strongly to Action 0, Pool B projects strongly to Action 1
        self.W_route = self.rng.normal(0.0, 1.0 / np.sqrt(num_hidden), (num_hidden, num_actions)).astype(np.float32)
        half_h = num_hidden // 2
        # Induce natural routing alignment
        self.W_route[:half_h, 0] += 0.55
        self.W_route[half_h:, 1] += 0.55
        self.b_action = np.zeros(num_actions, dtype=np.float32)

        # Cache original weights for lesion restoration
        self._W_route_orig = self.W_route.copy()
        self._is_lesioned = False
        self._lesion_type: Optional[str] = None

        # 2. Dynamic Action Accumulator State
        self.v_action = np.zeros(num_actions, dtype=np.float32)
        self.action_history: List[int] = []

        # Telemetry Cache
        self.last_action_probs = np.full(num_actions, 1.0 / num_actions, dtype=np.float32)
        self.last_winner_margin = 0.0
        self.last_action_selectivity = 0.0

    def reset(self) -> None:
        """Reset dynamic action state registers."""
        self.v_action.fill(0.0)
        self.action_history.clear()
        self.last_action_probs.fill(1.0 / self.num_actions)
        self.last_winner_margin = 0.0
        self.last_action_selectivity = 0.0

    def step(
        self,
        attractor_state: np.ndarray,
        dt: float = 1.0,
    ) -> Tuple[np.ndarray, int, float]:
        """Update competitive action accumulators and select winning action.

        Args:
            attractor_state: Current hidden/attractor state [num_hidden]
            dt: Timestep (ms)

        Returns:
            v_action: Continuous action activations [num_actions]
            selected_action: Winning action index (0 or 1)
            margin: Margin between highest and second-highest action activations
        """
        s = np.asarray(attractor_state, dtype=np.float32)
        if s.ndim == 2:
            s = s[0]

        # 1. Feedforward Routing Drive: I_route = s @ W_route + b
        i_route = s @ self.W_route + self.b_action

        # 2. Mutual Cross-Inhibition between action channels
        # Cross-inhibition: for action k, sum of other action activations
        total_act = np.sum(np.maximum(0.0, self.v_action))
        i_mutual = self.mutual_inhibition * (total_act - np.maximum(0.0, self.v_action))

        # 3. Leaky Integration: dv/dt = (-v + I_route - I_mutual) / tau
        decay = float(np.exp(-dt / self.tau_action))
        net_drive = i_route - i_mutual
        self.v_action = self.v_action * decay + (1.0 - decay) * net_drive

        # 4. Softmax Action Probabilities & Decision Margin
        v_shifted = self.v_action - np.max(self.v_action)
        exp_v = np.exp(np.clip(v_shifted, -20.0, 20.0))
        self.last_action_probs = exp_v / (np.sum(exp_v) + 1e-8)

        sorted_acts = np.sort(self.v_action)[::-1]
        self.last_winner_margin = float(sorted_acts[0] - sorted_acts[1]) if len(sorted_acts) > 1 else float(sorted_acts[0])

        # 5. Selected Action (Argmax)
        selected_action = int(np.argmax(self.v_action))
        self.action_history.append(selected_action)

        return self.v_action.copy(), selected_action, self.last_winner_margin

    def compute_routing_metrics(
        self,
        target_action: int,
        context_id: int,
    ) -> Dict[str, float]:
        """Compute action selectivity, incorrect activation ratio, and routing precision.

        Args:
            target_action: Desired action index (0 or 1)
            context_id: Current contextual cue ID (0 or 1)

        Returns:
            metrics: Dictionary of quantitative routing metrics
        """
        target_act = float(self.v_action[target_action])
        other_idx = 1 - target_action if self.num_actions == 2 else 0
        non_target_act = float(self.v_action[other_idx])

        # Action Selectivity Index: (Act_target - Act_other) / (|Act_target| + |Act_other| + eps)
        denom = abs(target_act) + abs(non_target_act) + 1e-6
        selectivity = float((target_act - non_target_act) / denom)
        self.last_action_selectivity = selectivity

        # Incorrect activation ratio: Act_other / (|Act_target| + eps)
        inc_ratio = float(max(0.0, non_target_act) / (max(1e-4, abs(target_act))))

        return {
            "action_selectivity": round(selectivity, 4),
            "winner_margin": round(self.last_winner_margin, 4),
            "target_action_prob": round(float(self.last_action_probs[target_action]), 4),
            "incorrect_activation_ratio": round(inc_ratio, 4),
        }

    # -----------------------------------------------------------------
    # Causal Pathway Lesioning Suite (Section 13)
    # -----------------------------------------------------------------
    def lesion_relevant_pathway(self, relevant_hidden_mask: np.ndarray) -> int:
        """Ablate (zero out) routing synapses originating from causally relevant hidden neurons.

        Args:
            relevant_hidden_mask: Boolean mask [num_hidden] indicating relevant pathway units

        Returns:
            num_lesioned: Number of synapses set to zero
        """
        rel_idx = np.where(relevant_hidden_mask)[0]
        self.W_route[rel_idx, :] = 0.0
        self._is_lesioned = True
        self._lesion_type = "RELEVANT"
        return len(rel_idx) * self.num_actions

    def lesion_irrelevant_pathway(self, relevant_hidden_mask: np.ndarray) -> int:
        """Ablate (zero out) a matched number of synapses originating from IRRELEVANT hidden neurons.

        Args:
            relevant_hidden_mask: Boolean mask [num_hidden] indicating relevant pathway units

        Returns:
            num_lesioned: Number of synapses set to zero
        """
        irrel_idx = np.where(~relevant_hidden_mask)[0]
        n_to_lesion = min(len(irrel_idx), int(np.sum(relevant_hidden_mask)))
        chosen_idx = self.rng.choice(irrel_idx, size=n_to_lesion, replace=False)
        self.W_route[chosen_idx, :] = 0.0
        self._is_lesioned = True
        self._lesion_type = "IRRELEVANT"
        return len(chosen_idx) * self.num_actions

    def restore_lesions(self) -> None:
        """Restore all original routing weights from un-lesioned backup."""
        self.W_route = self._W_route_orig.copy()
        self._is_lesioned = False
        self._lesion_type = None
