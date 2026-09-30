"""Phase 21 Track C Benchmark: Compositional Transfer (Zero-Shot, Frozen).

Trains component rules individually, then evaluates **unseen combinations**
with the model fully frozen. The zero-shot contract is enforced by
`TransferAudit`, which hashes the weights before and after evaluation and
rejects any plasticity, replay, or target access during the frozen phase.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain.compositional_transfer import (
    CHANCE_ACCURACY,
    COMPONENTS_OF,
    Combination,
    ComponentRule,
    TransferAudit,
    compose_action,
    linear_cka,
    linear_separability_probe,
    make_stimulus,
)
from brain.episodic_memory import EpisodicCondition, MemoryAblation, SparseEpisodicMemory

NUM_ACTIONS = 2


class ComponentReadout:
    """Minimal local readout with one weight block per component rule.

    Each component rule owns its own weights, so a component can be learned in
    isolation. Composition is performed by an explicit parity combiner at read
    time; the combiner has NO trained parameters of its own, so zero-shot
    composition cannot be attributed to having trained on the combination.
    """

    def __init__(self, dim: int = 8, lr: float = 0.4, seed: int = 42, threshold: float = 0.5):
        self.dim = int(dim)
        self.lr = float(lr)
        self.threshold = float(threshold)
        self.rng = np.random.default_rng(seed)
        self.W: Dict[ComponentRule, np.ndarray] = {
            r: self.rng.normal(0.0, 0.05, size=self.dim).astype(np.float32)
            for r in ComponentRule
        }

    def component_scores(self, x: np.ndarray) -> Dict[ComponentRule, float]:
        """Per-rule activation score for a stimulus."""
        return {r: float(self.W[r] @ x) for r in ComponentRule}

    def predict_single(self, rule: ComponentRule, x: np.ndarray) -> int:
        """Threshold the component score at the label midpoint (0.5).

        Component labels are encoded as 0/1, so the decision boundary is 0.5
        rather than 0. Using 0 would make near-zero scores on inactive
        distractor trials count as false positives.
        """
        return int(self.component_scores(x)[rule] > self.threshold)

    def predict_composition(self, combo: Combination, x: np.ndarray) -> int:
        """Compose the active components by parity (no trained parameters)."""
        components = COMPONENTS_OF[combo]
        bits = [self.predict_single(r, x) for r in components]
        return int(sum(bits) % NUM_ACTIONS)

    # -- plasticity -------------------------------------------------------------
    def train_rule(
        self,
        rule: ComponentRule,
        items: List[Tuple[np.ndarray, int]],
        epochs: int = 60,
    ) -> None:
        """Train ONE component rule in isolation (local two-factor update)."""
        for ep in range(epochs):
            for x, y in items:
                s = float(self.W[rule] @ x)
                err = float(np.clip(y - s, -1.0, 1.0))
                self.W[rule] += self.lr * err * x.astype(np.float32)

    def freeze(self) -> np.ndarray:
        """Immutable snapshot of all learned parameters."""
        return np.concatenate([self.W[r].ravel() for r in ComponentRule]).copy()

    def restore(self, snapshot: np.ndarray) -> None:
        """Restore parameters from a frozen snapshot."""
        i = 0
        for r in ComponentRule:
            n = self.W[r].size
            self.W[r] = snapshot[i:i + n].reshape(self.W[r].shape).astype(np.float32)
            i += n

    def parameter_count(self) -> int:
        return int(sum(self.W[r].size for r in ComponentRule))


def build_rule_dataset(
    rule: ComponentRule,
    n_per_case: int = 40,
    seed: int = 42,
    split: str = "train",
) -> List[Tuple[np.ndarray, int]]:
    """Balanced dataset for ONE component rule on the shared sensory schema.

    The label is 0 when the rule's subspace is inactive and 1 when it is
    active, so the rule must be read from the stimulus itself. Split streams
    are disjoint.
    """
    offsets = {"train": 0, "valid": 300000, "test": 700000}
    base = offsets.get(split, 0)
    items: List[Tuple[np.ndarray, int]] = []
    others = [r for r in ComponentRule if r is not rule]
    for case in (0, 1):
        for i in range(n_per_case):
            active: Tuple[ComponentRule, ...] = (rule,) if case == 1 else ()
            # Present distractor subspaces so activity alone cannot identify
            # which rule is being queried.
            if i % 2 == 1 and others:
                active = active + (others[(i // 2) % len(others)],)
            x = make_stimulus(active, trial_idx=base + i, seed=seed)
            items.append((x, case))
    return items


def train_components(seed: int = 42, epochs: int = 60, lr: float = 0.4) -> ComponentReadout:
    """Train all three component rules individually (never a combination)."""
    ro = ComponentReadout(seed=seed, lr=lr)
    for rule in ComponentRule:
        ro.train_rule(rule, build_rule_dataset(rule, seed=seed, split="train"), epochs=epochs)
    return ro


def evaluate_component_rules(ro: ComponentReadout, seed: int = 42) -> Dict[str, float]:
    """Frozen accuracy on each individually-trained rule (sanity check)."""
    out: Dict[str, float] = {}
    for rule in ComponentRule:
        items = build_rule_dataset(rule, seed=seed, split="test")
        correct = sum(1 for x, y in items if ro.predict_single(rule, x) == y)
        out[rule.value] = correct / max(1, len(items))
    return out


def evaluate_compositional_transfer(
    seed: int = 42,
    epochs: int = 60,
    lr: float = 0.4,
    use_memory: bool = True,
    memory_ablation: MemoryAblation = MemoryAblation.NONE,
    n_episodes: int = 120,
) -> Dict[str, Any]:
    """Zero-shot evaluation of unseen combinations under a frozen model.

    The model is frozen before evaluation; the audit verifies that no weight
    change, plasticity call, replay, or target access occurred.
    """
    ro = train_components(seed=seed, epochs=epochs, lr=lr)
    component_acc = evaluate_component_rules(ro, seed=seed)

    mem = SparseEpisodicMemory(capacity=n_episodes, seed=seed)
    mem.set_ablation(memory_ablation)
    for rule in ComponentRule:
        for x, y in build_rule_dataset(rule, n_per_case=20, seed=seed, split="train"):
            ctx = np.pad(x, (0, max(0, mem.context_dim - x.size)))[: mem.context_dim]
            mem.open_episode(
                episode_id=len(mem), context_vector=ctx,
                event_sequence=np.stack([x, x * 0.5]),
                chosen_action=y, timestamp_ms=float(len(mem) * 40.0),
                task_name=rule.value,
            )
            mem.close_episode(prediction_error=0.3, reward_val=1.0)

    audit = TransferAudit()
    audit.begin(ro.freeze())

    results: Dict[str, Any] = {}
    feats: Dict[str, List[float]] = {}
    labels: Dict[str, List[int]] = {}

    for combo in [Combination.AB, Combination.AC, Combination.BC, Combination.ABC, Combination.XOR]:
        correct, total = 0, 0
        f_list: List[float] = []
        l_list: List[int] = []
        for i in range(80):
            x = make_stimulus(COMPONENTS_OF[combo], trial_idx=700000 + i, seed=seed)
            y = compose_action(COMPONENTS_OF[combo], x)
            pred = ro.predict_composition(combo, x)
            correct += int(pred == y)
            total += 1
            f_list.append(float(sum(
                float(x.astype(np.float64) @ ro.W[r].astype(np.float64))
                for r in COMPONENTS_OF[combo]
            )))
            l_list.append(y)
            if use_memory:
                # Retrieval is READ-ONLY: no plasticity, no target access.
                q = np.pad(x, (0, max(0, mem.context_dim - x.size)))[: mem.context_dim]
                mem.retrieve(q, EpisodicCondition.A3_SPARSE_CONTENT)
        feats[combo.value] = f_list
        labels[combo.value] = l_list
        results[combo.value] = {
            "accuracy": correct / max(1, total),
            "chance": CHANCE_ACCURACY,
            "n_trials": total,
            "linear_probe_accuracy": linear_separability_probe(np.array(f_list), np.array(l_list)),
        }

    # Shuffled-label control: labels replaced by chance draws.
    rng = np.random.default_rng(seed)
    shuffled_correct = 0
    for i in range(80):
        x = make_stimulus(COMPONENTS_OF[Combination.AB], trial_idx=700000 + i, seed=seed)
        y_shuf = int(rng.integers(0, NUM_ACTIONS))
        shuffled_correct += int(ro.predict_composition(Combination.AB, x) == y_shuf)

    audit.end(ro.freeze())
    audit.assert_frozen()   # raises if any zero-shot guarantee was violated

    cka_ab_bc = linear_cka(np.array(feats["AB"]), np.array(feats["BC"]))
    linear_control = linear_separability_probe(np.array(feats["AB"]), np.array(labels["AB"]))

    return {
        "component_rule_accuracy": component_acc,
        "combinations": results,
        "mean_combination_accuracy": float(np.mean([
            v["accuracy"] for k, v in results.items() if k != "XOR"
        ])),
        "chance": CHANCE_ACCURACY,
        "shuffled_label_accuracy": shuffled_correct / 80.0,
        "linear_control_probe": round(linear_control, 6),
        "cka_ab_vs_bc": round(cka_ab_bc, 6),
        "zero_shot_audit": audit.to_dict(),
        "episodic_memory_bytes": mem.memory_bytes(),
        "episodic_memory_records": len(mem),
        "parameters": ro.parameter_count(),
    }


def evaluate_transfer_retrieval_dependence(
    seed: int = 42, epochs: int = 60, lr: float = 0.4
) -> Dict[str, Any]:
    """Does transfer actually depend on episodic retrieval? (H-C2)"""
    out: Dict[str, Any] = {}
    for tag, use_mem, abl in (
        ("with_memory", True, MemoryAblation.NONE),
        ("memory_removed", True, MemoryAblation.REMOVE_MEMORY),
        ("retrieval_randomized", True, MemoryAblation.RANDOMIZE_RETRIEVAL),
        ("no_memory", False, MemoryAblation.NONE),
    ):
        r = evaluate_compositional_transfer(
            seed=seed, epochs=epochs, lr=lr,
            use_memory=use_mem, memory_ablation=abl,
        )
        out[tag] = {
            "mean_combination_accuracy": r["mean_combination_accuracy"],
            "ab": r["combinations"]["AB"]["accuracy"],
            "abc": r["combinations"]["ABC"]["accuracy"],
        }
    return out
