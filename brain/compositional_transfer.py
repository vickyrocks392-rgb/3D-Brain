"""Phase 21 Track C: Compositional Task Transfer.

Implements **compositional** transfer testing: several component rules are
learned individually, and the system is then evaluated **frozen** on
combinations that were never trained.

    COMPONENT RULES (trained individually)
        Rule A : shape        -> action
        Rule B : colour       -> action
        Rule C : temporal     -> action
                  |
                  v
    NOVEL COMBINATIONS (never trained, evaluated frozen)
        A+B , A+C , B+C , A+B+C  (parity composition)
        plus an explicit nonlinear (XOR-like) composition

**No task-identity shortcut is used.** All conditions share an identical
sensory schema; only which feature subspaces are active differs. The forbidden
pattern `x[fixed_index] = task_id` never appears.

Zero-shot here means: no weight update, no plasticity, no replay, no target
access during evaluation. `TransferAudit.assert_frozen()` enforces this and is
called by the benchmark before every zero-shot measurement.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ComponentRule(str, Enum):
    """Component rules learned individually during training."""

    A_SHAPE = "A_SHAPE"
    B_COLOR = "B_COLOR"
    C_TEMPORAL = "C_TEMPORAL"


class Combination(str, Enum):
    """Novel combinations evaluated frozen (never trained)."""

    AB = "AB"
    AC = "AC"
    BC = "BC"
    ABC = "ABC"
    XOR = "XOR"


# Sensory subspace each component reads. Identical schema across all rules:
# the input vector always has 8 channels, only the active ones differ.
COMPONENT_CHANNELS: Dict[ComponentRule, Tuple[int, int]] = {
    ComponentRule.A_SHAPE: (0, 2),
    ComponentRule.B_COLOR: (2, 4),
    ComponentRule.C_TEMPORAL: (4, 6),
}

COMPONENTS_OF: Dict[Combination, Tuple[ComponentRule, ...]] = {
    Combination.AB: (ComponentRule.A_SHAPE, ComponentRule.B_COLOR),
    Combination.AC: (ComponentRule.A_SHAPE, ComponentRule.C_TEMPORAL),
    Combination.BC: (ComponentRule.B_COLOR, ComponentRule.C_TEMPORAL),
    Combination.ABC: (ComponentRule.A_SHAPE, ComponentRule.B_COLOR, ComponentRule.C_TEMPORAL),
    Combination.XOR: (ComponentRule.A_SHAPE, ComponentRule.B_COLOR),
}
def component_bit(rule: ComponentRule, stimulus: np.ndarray) -> int:
    """Binary activation of one component rule given a stimulus vector."""
    lo, hi = COMPONENT_CHANNELS[rule]
    return int(float(np.sum(stimulus[lo:hi])) > 0.0)


def component_action(rule: ComponentRule, stimulus: np.ndarray) -> int:
    """Learned mapping: a component rule maps its active bit to an action."""
    return component_bit(rule, stimulus)


def compose_action(
    components: Tuple[ComponentRule, ...],
    stimulus: np.ndarray,
    linear: bool = False,
) -> int:
    """Compose component actions by parity.

    Parity composition is the preregistered rule (spec 4.2). `linear=True`
    selects the additive control used to separate nonlinear composition from
    linear interpolation.
    """
    bits = [component_action(r, stimulus) for r in components]
    if linear:
        # Additive control: sum, thresholded into the action range.
        return int(sum(bits) > 0)
    return int(sum(bits) % 2)


def make_stimulus(
    active: Tuple[ComponentRule, ...],
    trial_idx: int,
    input_dim: int = 8,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Build a stimulus with only the requested component subspaces active.

    Every condition uses the same 8-channel schema; inactive channels are zero.
    No channel is ever used to encode task identity.
    """
    rng = np.random.RandomState((seed if seed is not None else 0) * 100003 + trial_idx)
    x = np.zeros(input_dim, dtype=np.float32)
    for r in active:
        lo, hi = COMPONENT_CHANNELS[r]
        x[lo:hi] = rng.uniform(0.6, 1.0, size=hi - lo).astype(np.float32)
    # Neutral filler in the non-semantic tail channel (identical across rules).
    if input_dim > 6:
        x[6] = 0.25
        x[7] = 0.0
    return x


class TransferAudit:
    """Tracks frozen-evaluation integrity for zero-shot claims.

    Any write to the model during evaluation invalidates the zero-shot label,
    so the audit records a weight hash before and after and compares it.
    """

    def __init__(self) -> None:
        self.weight_hash_before: Optional[str] = None
        self.weight_hash_after: Optional[str] = None
        self.plasticity_calls_during_eval = 0
        self.replay_calls_during_eval = 0
        self.target_access_during_eval = 0

    @staticmethod
    def hash_weights(weights: np.ndarray) -> str:
        """Stable hash of a weight matrix used to verify the model is frozen."""
        w = np.asarray(weights, dtype=np.float64)
        return f"{float(np.sum(w)):.12e}:{float(np.sum(np.abs(w))):.12e}"

    def begin(self, weights: np.ndarray) -> None:
        self.weight_hash_before = self.hash_weights(weights)

    def end(self, weights: np.ndarray) -> None:
        self.weight_hash_after = self.hash_weights(weights)

    def assert_frozen(self) -> None:
        """Raise if any zero-shot guarantee was violated."""
        if self.plasticity_calls_during_eval:
            raise RuntimeError(
                f"Zero-shot violation: {self.plasticity_calls_during_eval} plasticity call(s) during evaluation"
            )
        if self.replay_calls_during_eval:
            raise RuntimeError(
                f"Zero-shot violation: {self.replay_calls_during_eval} replay call(s) during evaluation"
            )
        if self.target_access_during_eval:
            raise RuntimeError(
                f"Zero-shot violation: {self.target_access_during_eval} target access event(s) during evaluation"
            )
        if (
            self.weight_hash_before is not None
            and self.weight_hash_after is not None
            and self.weight_hash_before != self.weight_hash_after
        ):
            raise RuntimeError("Zero-shot violation: model weights changed during frozen evaluation")

    def to_dict(self) -> Dict[str, Any]:
        """Machine-readable integrity report for the results file."""
        return {
            "weights_unchanged": self.weight_hash_before == self.weight_hash_after,
            "plasticity_calls": int(self.plasticity_calls_during_eval),
            "replay_calls": int(self.replay_calls_during_eval),
            "target_access": int(self.target_access_during_eval),
            "zero_shot_valid": (
                self.plasticity_calls_during_eval == 0
                and self.replay_calls_during_eval == 0
                and self.target_access_during_eval == 0
                and self.weight_hash_before == self.weight_hash_after
            ),
        }


def linear_separability_probe(features: np.ndarray, labels: np.ndarray) -> float:
    """Least-squares linear probe accuracy.

    Used to distinguish "the system cannot express this" from "the problem is
    linearly separable and the readout simply does not capture it".
    """
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    f = np.hstack([f, np.ones((f.shape[0], 1))])
    try:
        w, *_ = np.linalg.lstsq(f, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    pred = (f @ w) > 0.5
    return float(np.mean(pred == (y > 0.5)))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two representation matrices."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    hsic = float(np.sum((a.T @ a) * (b.T @ b)))
    na = float(np.linalg.norm(a, "fro"))
    nb = float(np.linalg.norm(b, "fro"))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return hsic / (na * nb)

    for r in active:
        lo, hi = COMPONENT_CHANNELS[r]
        x[lo:hi] = rng.uniform(0.6, 1.0, size=hi - lo).astype(np.float32)
    # Neutral filler in the non-semantic tail channel (identical across rules).
    if input_dim > 6:
        x[6] = 0.25
        x[7] = 0.0
    return x



# Chance accuracy for a 2-way action choice. Reported explicitly and never used
# as a hidden normalizer.
CHANCE_ACCURACY = 0.5

