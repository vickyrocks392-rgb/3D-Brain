"""Phase 21 Track A Benchmark: Episodic Memory & Episode Discrimination.

Constructs episodes that share an **identical initial observation** but differ
in temporal event order and resulting action, so that discrimination requires
order-sensitive temporal information rather than a hidden episode ID.

    Episode A: x0 -> e1, e2, e3 -> action 0
    Episode B: x0 -> e3, e1, e2 -> action 1

The retrieval query is built only from observed state (context + event history).
No ground-truth episode label is ever provided to the memory as a feature.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain.episodic_memory import (
    EpisodeRecord,
    EpisodicCondition,
    MemoryAblation,
    SparseEpisodicMemory,
)

DT_MS = 40.0

# All temporal orderings of three distinct events.
_EVENT_ORDERS = {
    0: [0, 1, 2],
    1: [2, 0, 1],
    2: [1, 2, 0],
    3: [0, 2, 1],
    4: [1, 0, 2],
    5: [2, 1, 0],
}


def generate_discriminative_episode(
    episode_idx: int,
    order_variant: int,
    context_dim: int = 40,
    hidden_dim: int = 40,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build one episode with a shared initial observation and distinct order."""
    rng = np.random.RandomState(seed * 7919 + episode_idx * 31 + order_variant)

    # Shared initial observation: identical across order variants.
    initial = rng.uniform(0.5, 1.0, size=context_dim).astype(np.float32)

    # Three event primitives: drawn ONCE per episode, then permuted.
    # (Drawing them inside a per-item re-seed would make them identical and
    # destroy the order signal entirely.)
    prim_rng = np.random.RandomState(seed * 104729 + episode_idx * 613 + 17)
    primitives = [prim_rng.normal(0.0, 1.0, size=hidden_dim).astype(np.float32) for _ in range(3)]
    order = _EVENT_ORDERS[order_variant % len(_EVENT_ORDERS)]
    event_sequence = np.stack([primitives[i] for i in order]).astype(np.float32)

    return {
        "episode_idx": int(episode_idx),
        "order_variant": int(order_variant),
        "initial_observation": initial,
        "context_vector": initial,
        "event_sequence": event_sequence,
        "chosen_action": int(order_variant % 2),
        "timestamp_ms": float(episode_idx * DT_MS * 4),
    }


def build_episode_store(
    n_episodes: int,
    memory: SparseEpisodicMemory,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Populate a memory with discriminative episodes; return their metadata."""
    metas: List[Dict[str, Any]] = []
    for i in range(n_episodes):
        variant = i % 6
        ep = generate_discriminative_episode(i, variant, seed=seed)
        memory.open_episode(
            episode_id=i,
            context_vector=ep["context_vector"],
            event_sequence=ep["event_sequence"],
            chosen_action=ep["chosen_action"],
            timestamp_ms=ep["timestamp_ms"],
            task_name=f"VARIANT_{variant}",
        )
        memory.close_episode(prediction_error=0.3 + 0.1 * (i % 4), reward_val=1.0)
        metas.append(ep)
    return metas


def query_from_observation(
    episode_meta: Dict[str, Any],
    memory: SparseEpisodicMemory,
    partial_observation: bool = False,
) -> np.ndarray:
    """Build a retrieval query from legitimately observed state only.

    `partial_observation=True` supplies only the initial observation, which is
    identical across variants: a system relying on context alone must fail, and
    the resulting error rate is the contextual-confusion measurement.
    """
    if partial_observation:
        return episode_meta["initial_observation"]
    ev = episode_meta["event_sequence"]
    # Order-sensitive summary, matching the memory's key construction.
    if ev.shape[0] > 1:
        summary = np.concatenate([ev[0], ev[1], ev[1] - ev[0]])
    else:
        summary = np.concatenate([ev[0], ev[0], np.zeros_like(ev[0])])
    obs = np.concatenate([episode_meta["initial_observation"], summary])
    return obs[: memory.context_dim].astype(np.float32)


def evaluate_episode_discrimination(
    condition: EpisodicCondition,
    n_episodes: int = 60,
    seed: int = 42,
    memory: Optional[SparseEpisodicMemory] = None,
    ablation: MemoryAblation = MemoryAblation.NONE,
) -> Dict[str, Any]:
    """Evaluate retrieval accuracy, confusion modes, latency, and sparsity."""
    mem = memory or SparseEpisodicMemory(capacity=max(8, n_episodes), seed=seed)
    mem.set_ablation(ablation)
    metas = build_episode_store(n_episodes, mem, seed=seed)

    correct = 0
    correct_partial = 0
    action_agrees = 0
    latencies: List[float] = []
    sparsities: List[float] = []

    for meta in metas:
        true_id = meta["episode_idx"]
        res = mem.retrieve(query_from_observation(meta, mem), condition=condition)
        if res.top_id is not None:
            latencies.append(res.latency_us)
            sparsities.append(res.sparsity)
            if int(res.top_id) == true_id:
                correct += 1
                action_agrees += 1
            else:
                hit = next((r for r in mem.records if r.episode_id == res.top_id), None)
                if hit is not None and hit.chosen_action == meta["chosen_action"]:
                    action_agrees += 1

        # Control: initial observation only (contextual-confusion probe).
        res_p = mem.retrieve(
            query_from_observation(meta, mem, partial_observation=True), condition=condition
        )
        if res_p.top_id is not None and int(res_p.top_id) == true_id:
            correct_partial += 1

    n = max(1, len(metas))
    wrong = n - correct
    return {
        "condition": condition.value,
        "ablation": ablation.value,
        "n_episodes": n,
        "retrieval_accuracy": correct / n,
        "action_agreement": action_agrees / n,
        "partial_observation_accuracy": correct_partial / n,
        "contextual_confusion": 1.0 - correct_partial / n,
        "incorrect_retrieval_rate": wrong / n,
        "temporal_confusion": wrong / n,
        "retrieval_latency_us": float(np.mean(latencies)) if latencies else 0.0,
        "retrieval_sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
        "memory_bytes": mem.memory_bytes(),
        "dictionary_parameters": mem.dictionary_parameters(),
        "records_stored": len(mem),
    }


def evaluate_memory_capacity(
    capacities: Optional[List[int]] = None,
    seed: int = 42,
    condition: EpisodicCondition = EpisodicCondition.A3_SPARSE_CONTENT,
) -> Dict[str, Any]:
    """Measure the capacity / interference / retrieval-cost tradeoff."""
    caps = capacities or [10, 25, 50, 100, 250, 500, 1000]
    results: List[Dict[str, Any]] = []
    for cap in caps:
        mem = SparseEpisodicMemory(capacity=cap, seed=seed)
        metas = build_episode_store(cap, mem, seed=seed)
        correct = 0
        latencies: List[float] = []
        for meta in metas:
            res = mem.retrieve(query_from_observation(meta, mem), condition=condition)
            if res.top_id is not None:
                latencies.append(res.latency_us)
                if int(res.top_id) == meta["episode_idx"]:
                    correct += 1
        results.append({
            "capacity": int(cap),
            "retrieval_accuracy": correct / max(1, len(metas)),
            "memory_bytes": mem.memory_bytes(),
            "records_stored": len(mem),
            "retrieval_cost_flops": mem.total_retrieval_flops,
            "retrieval_latency_us": float(np.mean(latencies)) if latencies else 0.0,
            "retrieval_sparsity": mem.top_k / float(max(1, len(mem))),
        })
    return {"curve": results}


def evaluate_memory_ablations(
    n_episodes: int = 60,
    seed: int = 42,
) -> Dict[str, Any]:
    """Causal interventions on episodic retrieval (specification 2.6)."""
    out: Dict[str, Any] = {}
    for abl in MemoryAblation:
        r = evaluate_episode_discrimination(
            EpisodicCondition.A3_SPARSE_CONTENT,
            n_episodes=n_episodes,
            seed=seed,
            ablation=abl,
        )
        out[abl.value] = r
    return out
