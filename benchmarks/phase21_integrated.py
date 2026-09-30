"""Phase 21 Track D/E Benchmark: Integrated System & Fair Baselines.

Compares the integrated configurations D0-D5 and reports a full
parameter / memory / state / compute accounting that keeps external memory
explicit and never folded into the neural parameter count.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from benchmarks.phase21_consolidation import (
    NUM_CLASSES,
    evaluate_frozen,
    make_task_prototypes,
    make_training_task,
)
from brain.episodic_memory import SparseEpisodicMemory
from brain.replay_consolidation import (
    ConsolidationModule,
    ReplayConsolidationCondition,
    ReplayConsolidationEngine,
)

# Track D configurations (preregistered).
D_CONDITIONS: Dict[str, Dict[str, bool]] = {
    "D0_PHASE20_BASELINE": {"memory": False, "replay": False, "consolidation": False, "routing": False},
    "D1_EPISODIC_MEMORY": {"memory": True, "replay": False, "consolidation": False, "routing": False},
    "D2_REPLAY_CONSOLIDATION": {"memory": False, "replay": True, "consolidation": True, "routing": False},
    "D3_MEMORY_REPLAY": {"memory": True, "replay": True, "consolidation": False, "routing": False},
    "D4_MEMORY_REPLAY_CONSOLIDATION": {"memory": True, "replay": True, "consolidation": True, "routing": False},
    "D5_FULL_ROUTING": {"memory": True, "replay": True, "consolidation": True, "routing": True},
}


def compute_accounting(
    cons: ConsolidationModule,
    mem: Optional[SparseEpisodicMemory],
    engine: ReplayConsolidationEngine,
    dim: int = 40,
    n_branches: int = 3,
    routing: bool = False,
) -> Dict[str, Any]:
    """Separate neural parameters, dynamic state, and external memory.

    External memory is NEVER folded into the neural parameter count; every
    comparison states explicitly whether memory is included.
    """
    trainable = cons.parameter_count()
    dynamic_state = cons.tags.size + cons.weights_slow.size
    ep_bytes = mem.memory_bytes() if mem is not None else 0
    dict_params = mem.dictionary_parameters() if mem is not None else 0
    routing_params = (dim * n_branches + n_branches) if routing else 0

    return {
        "neural_parameters_trainable": int(trainable + dict_params + routing_params),
        "neural_parameters_fixed": int(dim * 40 + 40 * 40),
        "dynamic_state_variables": int(dynamic_state),
        "episodic_memory_bytes": int(ep_bytes),
        "replay_buffer_bytes": int(ep_bytes),
        "consolidation_slow_pool_bytes": int(cons.weights_slow.nbytes + cons.tags.nbytes),
        "structural_hyperparameters": int(dim + n_branches if routing else dim),
        "computational_workspace_peak_bytes": int(dim * NUM_CLASSES * 8),
        "external_memory_included_in_parameters": False,
        "flops_replay_total": int(engine.total_replay_flops),
        "flops_retrieval_total": int(mem.total_retrieval_flops if mem is not None else 0),
        "total_external_memory_bytes": int(ep_bytes + cons.weights_slow.nbytes),
    }


def evaluate_integrated_condition(
    condition_id: str,
    seed: int = 42,
    dim: int = 40,
    epochs: int = 30,
    n_branches: int = 3,
) -> Dict[str, Any]:
    """Run one integrated configuration through a continual A -> B -> A cycle.

    Accuracy on task A is measured frozen before and after the intervening
    task. All mechanisms are optional per the D-condition table.
    """
    cfg = D_CONDITIONS[condition_id]
    rng = np.random.RandomState(seed)

    protos_a = make_task_prototypes(dim=dim, seed=seed)
    protos_b = make_task_prototypes(dim=dim, seed=seed + 999)
    task_a = make_training_task(dim=dim, seed=seed, split="train", prototypes=protos_a)
    probes_a = make_training_task(dim=dim, seed=seed, split="test", prototypes=protos_a)
    task_b = make_training_task(dim=dim, seed=seed, split="train", prototypes=protos_b)

    cons = ConsolidationModule((dim, NUM_CLASSES), seed=seed)
    mem = SparseEpisodicMemory(capacity=512, seed=seed) if cfg["memory"] else None
    replay_cond = (
        ReplayConsolidationCondition.B6_BOTH
        if cfg["replay"] else ReplayConsolidationCondition.B0_NO_REPLAY
    )
    engine = ReplayConsolidationEngine(cons, replay_budget=8, condition=replay_cond, seed=seed)

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
                if mem is not None:
                    mem.open_episode(
                        episode_id=len(mem) + ep * len(items) + int(idx),
                        context_vector=x, event_sequence=np.stack([x, x * 0.5]),
                        chosen_action=int(c), timestamp_ms=float(ep * 40 + idx),
                        task_name=tag,
                    )
                    mem.close_episode(prediction_error=abs(err), reward_val=1.0)
            if cfg["replay"]:
                engine.run_rest(mem.records if mem is not None else [], num_rest_steps=10, dt_ms=40.0)

    learn(task_a, "TASK_A")
    acc_pre = evaluate_frozen(cons, probes_a)
    learn(task_b, "TASK_B")
    acc_post = evaluate_frozen(cons, probes_a)

    forgetting = max(0.0, acc_pre - acc_post)
    retention = (acc_post / acc_pre) if acc_pre > 0 else None
    account = compute_accounting(
        cons, mem, engine, dim=dim, n_branches=n_branches, routing=cfg["routing"]
    )

    return {
        "condition": condition_id,
        "config": dict(cfg),
        "acc_a_pre": round(acc_pre, 4),
        "acc_a_post": round(acc_post, 4),
        "forgetting_abs": round(forgetting, 4),
        "retention_ratio": round(retention, 4) if retention is not None else None,
        "backward_transfer": round(acc_post - acc_pre, 4),
        "accounting": account,
    }


def evaluate_integrated_sweep(seeds: List[int], epochs: int = 30) -> Dict[str, Any]:
    """Run every D condition across the mandatory seed set."""
    out: Dict[str, Any] = {}
    for cond_id in D_CONDITIONS:
        runs = [evaluate_integrated_condition(cond_id, seed=s, epochs=epochs) for s in seeds]
        rets = [r["retention_ratio"] for r in runs if r["retention_ratio"] is not None]
        out[cond_id] = {
            "acc_a_pre_mean": float(np.mean([r["acc_a_pre"] for r in runs])),
            "acc_a_post_mean": float(np.mean([r["acc_a_post"] for r in runs])),
            "acc_a_post_std": float(np.std([r["acc_a_post"] for r in runs])),
            "forgetting_mean": float(np.mean([r["forgetting_abs"] for r in runs])),
            "retention_mean": float(np.mean(rets)) if rets else None,
            "n_seeds": len(runs),
            "accounting": runs[0]["accounting"],
            "seed_runs": runs,
        }
    return out
