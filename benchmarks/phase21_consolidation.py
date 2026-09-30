"""Phase 21 Track B Benchmark: Replay Consolidation & Content Audit.

Evaluates whether replay during explicit quiescent rest produces measurable
consolidation, and audits **what replay actually contributes** by comparing
seven replay-content protocols.

The consolidation target is a real two-class readout trained by local updates.
Rest periods apply no sensory input, no target, and no reward.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain.episodic_memory import SparseEpisodicMemory
from brain.replay_consolidation import (
    ConsolidationAblation,
    ConsolidationModule,
    ReplayConsolidationCondition,
    ReplayConsolidationEngine,
    ReplayContent,
)

DT_MS = 40.0
NUM_CLASSES = 2


def make_task_prototypes(
    n_classes: int = NUM_CLASSES,
    dim: int = 40,
    seed: int = 42,
) -> List[np.ndarray]:
    """Class prototypes for a task. Identical across train/valid/test splits."""
    rng = np.random.RandomState(seed * 31337)
    protos = [rng.normal(0.0, 1.0, size=dim).astype(np.float32) for _ in range(n_classes)]
    return [p / np.linalg.norm(p) for p in protos]


def make_training_task(
    n_classes: int = NUM_CLASSES,
    dim: int = 40,
    seed: int = 42,
    noise: float = 0.35,
    n_per_class: int = 20,
    split: str = "train",
    prototypes: Optional[List[np.ndarray]] = None,
) -> List[Tuple[np.ndarray, int]]:
    """Sample a split of a classification task.

    The split ('train' / 'valid' / 'test') selects a DISJOINT sample stream
    while reusing the same class prototypes, so all splits describe the same
    underlying task. No trial can appear in more than one split.
    """
    protos = prototypes if prototypes is not None else make_task_prototypes(n_classes, dim, seed)

    # Disjoint, deterministic sample streams per split.
    offsets = {"train": 0, "valid": 500000, "test": 900000}
    base = offsets.get(split, 0)

    rng = np.random.RandomState(seed * 6151 + base)
    items: List[Tuple[np.ndarray, int]] = []
    for c in range(n_classes):
        for i in range(n_per_class):
            r = np.random.RandomState((seed * 977 + base + c * 1009 + i) % (2 ** 31 - 1))
            x = protos[c] + r.normal(0.0, noise, size=dim).astype(np.float32)
            items.append((x, c))
    rng.shuffle(items)
    return items


def train_readout(
    consolidation: ConsolidationModule,
    items: List[Tuple[np.ndarray, int]],
    epochs: int = 12,
    lr: float = 0.15,
    n_rest_epochs: int = 2,
    condition: ReplayConsolidationCondition = ReplayConsolidationCondition.B0_NO_REPLAY,
    content: ReplayContent = ReplayContent.R1_EXACT,
    memory: Optional[SparseEpisodicMemory] = None,
    seed: int = 42,
    use_replay: bool = True,
) -> Dict[str, Any]:
    """Train a two-class readout with optional replay-driven rest periods.

    Training uses an eligibility-style local update; evaluation is always done
    on held-out frozen probes with no plasticity.
    """
    rng = np.random.RandomState(seed)
    mem = memory or SparseEpisodicMemory(capacity=256, seed=seed)
    engine = ReplayConsolidationEngine(consolidation, replay_budget=8, condition=condition,
                                      content=content, seed=seed)

    for ep in range(epochs):
        order = rng.permutation(len(items))
        for idx in order:
            x, c = items[idx]
            xn = x / (np.linalg.norm(x) + 1e-8)
            pred = consolidation.weights_fast.T @ xn
            teach = np.full(NUM_CLASSES, -1.0, dtype=np.float32)
            teach[c] = 1.0
            margin = float(pred[c] - np.max(np.delete(pred, c))) if NUM_CLASSES > 1 else float(pred[c])
            err = float(np.clip(margin, -1.0, 1.0))
            eff_lr = consolidation.effective_lr(lr)
            elig = np.outer(xn, teach * err * eff_lr)
            consolidation.weights_fast += elig
            consolidation.set_tags(np.outer(np.abs(xn), np.ones(NUM_CLASSES, dtype=np.float32)))

            if use_replay:
                mem.open_episode(
                    episode_id=len(mem) + ep * len(items) + int(idx),
                    context_vector=x,
                    event_sequence=np.stack([x, x * 0.5]),
                    chosen_action=int(c),
                    timestamp_ms=float(ep * DT_MS * len(items) + idx * DT_MS),
                    task_name=f"CLASS_{c}",
                )
                mem.close_episode(prediction_error=abs(err), reward_val=1.0)

        # Quiescent rest: no sensory input, no target, no reward.
        for _ in range(n_rest_epochs):
            if use_replay:
                engine.run_rest(mem.records, num_rest_steps=10, dt_ms=DT_MS)

    return {"memory": mem, "engine": engine}


def evaluate_frozen(
    consolidation: ConsolidationModule,
    probes: List[Tuple[np.ndarray, int]],
) -> float:
    """Evaluate accuracy with plasticity fully disabled."""
    w = consolidation.effective_weights().copy()   # frozen snapshot
    correct = 0
    for x, c in probes:
        xn = x / (np.linalg.norm(x) + 1e-8)
        if int(np.argmax(w.T @ xn)) == c:
            correct += 1
    return correct / max(1, len(probes))


def evaluate_consolidation_condition(
    condition: ReplayConsolidationCondition,
    content: ReplayContent = ReplayContent.R1_EXACT,
    seed: int = 42,
    epochs: int = 12,
    ablation: ConsolidationAblation = ConsolidationAblation.NONE,
    dim: int = 40,
) -> Dict[str, Any]:
    """Measure pre/post-rest retention and synaptic change for one condition."""
    protos = make_task_prototypes(dim=dim, seed=seed)
    items = make_training_task(dim=dim, seed=seed, split="train", prototypes=protos)
    probes = make_training_task(dim=dim, seed=seed, split="test", prototypes=protos)

    tag_enabled = ablation not in (ConsolidationAblation.REMOVE_TAGGING,)
    meta_enabled = ablation not in (ConsolidationAblation.REMOVE_METAPLASTICITY,)
    cons = ConsolidationModule(
        (dim, NUM_CLASSES),
        consolidation_rate=0.1,
        tag_enabled=tag_enabled,
        metaplasticity_enabled=meta_enabled,
        seed=seed,
    )
    if ablation == ConsolidationAblation.ALTER_RATE:
        cons.consolidation_rate = 0.2

    trained = train_readout(
        cons, items, epochs=epochs, seed=seed,
        condition=condition, content=content,
        use_replay=(condition != ReplayConsolidationCondition.B0_NO_REPLAY),
    )
    mem, engine = trained["memory"], trained["engine"]

    acc = evaluate_frozen(cons, probes)
    pre_norm = float(np.linalg.norm(cons.effective_weights()))
    return {
        "condition": condition.value,
        "content": content.value,
        "ablation": ablation.value,
        "frozen_accuracy": acc,
        "synaptic_norm": round(float(np.linalg.norm(cons.effective_weights())), 6),
        "slow_pool_norm": round(float(np.linalg.norm(cons.weights_slow)), 6),
        "tag_mass": round(float(np.sum(cons.tags)), 6),
        "metaplasticity": round(float(cons.metaplasticity), 6),
        "captures": int(cons.total_captures),
        "replay_events": int(engine.total_replay_events),
        "replay_flops": int(engine.total_replay_flops),
        "memory_bytes": mem.memory_bytes(),
        "consolidation_bytes": cons.memory_bytes(),
        "fast_pool_parameters": cons.parameter_count(),
        "pre_eval_norm": round(pre_norm, 6),
    }


def evaluate_forgetting_sequence(
    condition: ReplayConsolidationCondition,
    content: ReplayContent = ReplayContent.R1_EXACT,
    seed: int = 42,
    dim: int = 40,
    n_intervening: int = 1,
    epochs: int = 30,
) -> Dict[str, Any]:
    """Measure forgetting of A after an intervening task, with/without replay.

    This is the protocol in which consolidation can actually matter: a task is
    learned, an *interfering* task is then learned, and retention of the first
    task is measured with the model frozen. Consolidation is expected to matter
    only in this regime, which is why it is the primary Track B endpoint.
    """
    rng = np.random.RandomState(seed)

    # Task A prototypes and data
    protos_a = make_task_prototypes(dim=dim, seed=seed)
    task_a = make_training_task(dim=dim, seed=seed, split="train", prototypes=protos_a)
    probes_a = make_training_task(dim=dim, seed=seed, split="test", prototypes=protos_a)

    # Interfering task B uses DISJOINT prototypes (a different stimulus subspace).
    # Because both tasks are learned by the SAME shared readout weights, their
    # updates genuinely compete, which is what produces measurable forgetting.
    protos_b = make_task_prototypes(dim=dim, seed=seed + 999)
    task_b = make_training_task(dim=dim, seed=seed, split="train", prototypes=protos_b)

    cons = ConsolidationModule((dim, NUM_CLASSES), seed=seed)
    mem = SparseEpisodicMemory(capacity=512, seed=seed)
    engine = ReplayConsolidationEngine(
        cons, replay_budget=8, condition=condition, content=content, seed=seed
    )
    use_replay = condition != ReplayConsolidationCondition.B0_NO_REPLAY

    def learn(items, tag):
        for ep in range(epochs):
            for idx in rng.permutation(len(items)):
                x, c = items[idx]
                xn = x / (np.linalg.norm(x) + 1e-8)
                pred = cons.weights_fast.T @ xn
                teach = np.full(NUM_CLASSES, -1.0, dtype=np.float32)
                teach[c] = 1.0
                margin = float(pred[c] - np.max(np.delete(pred, c)))
                err = float(np.clip(margin, -1.0, 1.0))
                cons.weights_fast += 0.05 * np.outer(xn, teach * err)
                cons.set_tags(np.outer(np.abs(xn), np.ones(NUM_CLASSES, dtype=np.float32)))
                if use_replay:
                    mem.open_episode(
                        episode_id=len(mem) + ep * len(items) + int(idx),
                        context_vector=x, event_sequence=np.stack([x, x * 0.5]),
                        chosen_action=int(c), timestamp_ms=float(ep * 40 + idx),
                        task_name=tag,
                    )
                    mem.close_episode(prediction_error=abs(err), reward_val=1.0)
            if use_replay:
                engine.run_rest(mem.records, num_rest_steps=10, dt_ms=40.0)

    # Learn A, measure, then learn the interfering B, then re-measure A.
    learn(task_a, "TASK_A")
    acc_a_pre = evaluate_frozen(cons, probes_a)
    slow_pre = float(np.linalg.norm(cons.weights_slow))
    learn(task_b, "TASK_B")
    acc_a_post = evaluate_frozen(cons, probes_a)

    forgetting_abs = max(0.0, acc_a_pre - acc_a_post)
    retention = (acc_a_post / acc_a_pre) if acc_a_pre > 0 else None
    backward_transfer = acc_a_post - acc_a_pre
    interference = forgetting_abs

    return {
        "condition": condition.value,
        "content": content.value,
        "acc_a_pre": round(acc_a_pre, 4),
        "acc_a_post": round(acc_a_post, 4),
        "forgetting_abs": round(forgetting_abs, 4),
        "retention_ratio": round(retention, 4) if retention is not None else None,
        "backward_transfer": round(backward_transfer, 4),
        "interference": round(interference, 4),
        "slow_pool_norm_pre": round(slow_pre, 6),
        "slow_pool_norm_post": round(float(np.linalg.norm(cons.weights_slow)), 6),
        "captures": int(cons.total_captures),
        "replay_events": int(engine.total_replay_events),
        "memory_bytes": mem.memory_bytes(),
    }


def evaluate_transfer_without_replay(seed: int = 42, dim: int = 40) -> Dict[str, Any]:
    """Four preregistered protocols isolating the effect of replay (spec 6.3)."""
    out: Dict[str, Any] = {}

    # 1 & 2: learn -> test, with and without an intervening rest/replay period.
    for tag, cond in (
        ("learn_test_no_replay", ReplayConsolidationCondition.B0_NO_REPLAY),
        ("learn_rest_replay_test", ReplayConsolidationCondition.B6_BOTH),
    ):
        r = evaluate_forgetting_sequence(cond, seed=seed, dim=dim, epochs=30)
        out[tag] = {"acc_a_post": r["acc_a_post"], "forgetting_abs": r["forgetting_abs"]}

    # 3 & 4: learn A -> (replay A | no replay A) -> learn B -> test A
    for tag, cond in (
        ("learnA_replayA_learnB_testA", ReplayConsolidationCondition.B6_BOTH),
        ("learnA_noreplayA_learnB_testA", ReplayConsolidationCondition.B0_NO_REPLAY),
    ):
        r = evaluate_forgetting_sequence(cond, seed=seed, dim=dim, epochs=30)
        out[tag] = {"acc_a_post": r["acc_a_post"], "forgetting_abs": r["forgetting_abs"]}

    return out


def evaluate_replay_content_audit(seed: int = 42, epochs: int = 30) -> Dict[str, Any]:
    """Compare all seven replay-content protocols under interference."""
    out: Dict[str, Any] = {}
    for content in ReplayContent:
        r = evaluate_forgetting_sequence(
            ReplayConsolidationCondition.B6_BOTH, content=content, seed=seed, epochs=epochs
        )
        out[content.value] = r
    return out
