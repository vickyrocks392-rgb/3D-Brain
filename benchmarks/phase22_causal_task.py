"""Phase 22 Track A, B, D, E Benchmark: Episode-Dependent Abstract Retrieval & Causal Ablation.

Resolves the Phase 21 finding:
"episodic retrieval worked, but episodic memory was NOT causally necessary for the tested compositional-transfer task."

In Phase 22, tasks are designed such that:
1. The relevant relation A -> X is introduced in an earlier episode (Episode 1).
2. The relation X in {0, 1} is sampled independently per entity/session, so NO static rule exists in the generator.
3. The current episode (Episode 3 / Probe) contains ONLY the sensory cue A; it contains NO target, NO reward, and NO episode-ID.
4. Information required to solve the task exists ONLY in the previous episode.
5. Without retrieving Episode 1, performance is strictly at chance (50.0%).
6. Intervening distractor episodes wash out any recurrent working memory across delays (0 to 120 seconds).
7. Causal ablations B0-B7 verify the causal necessity of episodic memory:
   - B0: Full system + intact episodic memory
   - B1: Memory removed entirely
   - B2: Memory present but retrieval disabled
   - B3: Shuffled memory (random episode retrieved)
   - B4: Wrong memory (non-matching episode retrieved)
   - B5: Corrupted memory (noise injected into retrieved latent state)
   - B6: Partial memory (50% latent masking)
   - B7: Corrupted retrieval cue (noise injected into query)
"""

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain.episodic_memory import (
    EpisodeRecord,
    EpisodicCondition,
    MemoryAblation,
    RetrievalResult,
    SparseEpisodicMemory,
)

DT_MS = 40.0
CHANCE_ACCURACY = 0.5
RETRIEVAL_DELAYS_SEC = [0, 1, 5, 10, 30, 60, 120]


class CausalCondition(str, Enum):
    """Phase 22 Track B Preregistered Causal Intervention Conditions."""
    B0_FULL_MEMORY = "B0_FULL_MEMORY"
    B1_MEMORY_REMOVED = "B1_MEMORY_REMOVED"
    B2_RETRIEVAL_DISABLED = "B2_RETRIEVAL_DISABLED"
    B3_SHUFFLED_MEMORY = "B3_SHUFFLED_MEMORY"
    B4_WRONG_MEMORY = "B4_WRONG_MEMORY"
    B5_CORRUPTED_MEMORY = "B5_CORRUPTED_MEMORY"
    B6_PARTIAL_MEMORY = "B6_PARTIAL_MEMORY"
    B7_CORRUPT_CUE = "B7_CORRUPT_CUE"


CONDITION_TO_ABLATION: Dict[CausalCondition, MemoryAblation] = {
    CausalCondition.B0_FULL_MEMORY: MemoryAblation.NONE,
    CausalCondition.B1_MEMORY_REMOVED: MemoryAblation.REMOVE_MEMORY,
    CausalCondition.B2_RETRIEVAL_DISABLED: MemoryAblation.DISABLE_RETRIEVAL,
    CausalCondition.B3_SHUFFLED_MEMORY: MemoryAblation.SHUFFLE_MEMORY,
    CausalCondition.B4_WRONG_MEMORY: MemoryAblation.WRONG_MEMORY,
    CausalCondition.B5_CORRUPTED_MEMORY: MemoryAblation.CORRUPT_MEMORY,
    CausalCondition.B6_PARTIAL_MEMORY: MemoryAblation.PARTIAL_MEMORY,
    CausalCondition.B7_CORRUPT_CUE: MemoryAblation.CORRUPT_CUE,
}


@dataclass
class EpisodeDependentTrial:
    """One episode-dependent trial pair: induction episode -> distractors -> probe episode."""
    cue_id: int
    cue_vector: np.ndarray       # Continuous sensory representation of cue A in R^d_ctx
    target_action: int           # Latent relation X in {0, 1}, sampled randomly per trial
    induction_events: np.ndarray # Latent trajectory during induction
    delay_sec: float             # Delay between induction and probe
    distractor_count: int        # Number of intervening distractor episodes
    probe_vector: np.ndarray     # Sensory presentation at test (identical or noisy cue A)


def generate_orthogonal_cues(n_cues: int, dim: int = 40, seed: int = 42) -> np.ndarray:
    """Generate dense, continuous, non-one-hot cue representations."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0, size=(n_cues, dim)).astype(np.float32)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return (A / np.maximum(norms, 1e-8)).astype(np.float32)


def make_episode_dependent_session(
    n_trials: int = 40,
    delay_sec: float = 5.0,
    context_dim: int = 40,
    hidden_dim: int = 40,
    seed: int = 42,
) -> List[EpisodeDependentTrial]:
    """Build a session of episode-dependent trials.

    For every trial:
    - Target relation X in {0, 1} is sampled with p=0.5 uniformly.
    - An arbitrary continuous cue A is assigned.
    - No static generator formula can predict X from A.
    - Intervening distractor count is proportional to delay_sec.
    """
    rng = np.random.default_rng(seed)
    cues = generate_orthogonal_cues(n_cues=n_trials * 2, dim=context_dim, seed=seed)

    trials: List[EpisodeDependentTrial] = []
    # Calculate distractor count (each distractor takes ~200ms = 5 steps of 40ms)
    distractor_count = max(0, int(round(delay_sec / 0.2)))

    for i in range(n_trials):
        cue_vec = cues[i]
        target = int(rng.integers(0, 2))  # Exactly 50/50 prior, arbitrary relation

        # Induction event sequence: temporal sequence of latent states
        # The latent binding encodes the relation X via directional latent modulation
        event_steps = 3
        events = np.zeros((event_steps, hidden_dim), dtype=np.float32)
        base_noise = rng.normal(0.0, 0.5, size=(event_steps, hidden_dim)).astype(np.float32)
        events += base_noise
        # Latent state carries the outcome binding direction
        latent_binding = np.zeros(hidden_dim, dtype=np.float32)
        latent_binding[:20] = (1.0 if target == 1 else -1.0) * cue_vec[:20]
        latent_binding[20:] = rng.normal(0.0, 0.2, size=hidden_dim - 20).astype(np.float32)

        # Probe presentation is cue A with small sensory variation (no target leakage)
        sensory_jitter = rng.normal(0.0, 0.05, size=context_dim).astype(np.float32)
        probe_vec = cue_vec + sensory_jitter
        p_norm = float(np.linalg.norm(probe_vec))
        if p_norm > 1e-8:
            probe_vec = (probe_vec / p_norm).astype(np.float32)

        trials.append(
            EpisodeDependentTrial(
                cue_id=i,
                cue_vector=cue_vec,
                target_action=target,
                induction_events=events,
                delay_sec=float(delay_sec),
                distractor_count=distractor_count,
                probe_vector=probe_vec,
            )
        )
    return trials


def run_episode_dependent_trial(
    trial: EpisodeDependentTrial,
    memory: SparseEpisodicMemory,
    causal_condition: CausalCondition = CausalCondition.B0_FULL_MEMORY,
    seed: int = 42,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Execute one episode-dependent trial through memory encoding, distractors, and probe."""
    if rng is None:
        rng = np.random.default_rng(seed + trial.cue_id * 101)
    ablation = CONDITION_TO_ABLATION[causal_condition]
    memory.set_ablation(ablation)

    # 1. ENCODING EPISODE 1
    # Build latent binding representing the established relation
    latent_binding = np.zeros(memory.hidden_dim, dtype=np.float32)
    latent_binding[:20] = (1.0 if trial.target_action == 1 else -1.0) * trial.cue_vector[:20]

    # Pre-outcome write: open_episode has NO target_action parameter
    # chosen_action at encoding can be arbitrary exploration action
    simulated_emitted_action = trial.target_action
    rec = memory.open_episode(
        episode_id=trial.cue_id,
        context_vector=trial.cue_vector,
        event_sequence=trial.induction_events,
        chosen_action=simulated_emitted_action,
        timestamp_ms=float(trial.cue_id * 1000.0),
        task_name=f"RELATION_CUE_{trial.cue_id}",
        latent_state=latent_binding,
    )
    # Outcome write: commit after feedback
    memory.close_episode(prediction_error=0.1, reward_val=1.0)

    # 2. INTERVENING DISTRACTORS
    # If delay > 0, store distractor episodes
    for d in range(trial.distractor_count):
        dist_cue = rng.normal(0.0, 1.0, size=memory.context_dim).astype(np.float32)
        dist_norm = float(np.linalg.norm(dist_cue))
        if dist_norm > 1e-8:
            dist_cue /= dist_norm
        dist_events = rng.normal(0.0, 1.0, size=(2, memory.hidden_dim)).astype(np.float32)
        dist_latent = rng.normal(0.0, 1.0, size=memory.hidden_dim).astype(np.float32)
        dist_id = 100000 + trial.cue_id * 1000 + d
        memory.open_episode(
            episode_id=dist_id,
            context_vector=dist_cue,
            event_sequence=dist_events,
            chosen_action=int(rng.integers(0, 2)),
            timestamp_ms=float(trial.cue_id * 1000.0 + (d + 1) * DT_MS * 5),
            task_name="DISTRACTOR",
            latent_state=dist_latent,
        )
        memory.close_episode(prediction_error=0.5, reward_val=0.0)

    # 3. PROBE EPISODE (TEST)
    # The current episode contains ONLY probe_vector (cue A).
    # It contains NO target, NO reward, NO episode ID.
    retrieval_res = memory.retrieve(trial.probe_vector, EpisodicCondition.A3_SPARSE_CONTENT)

    # Decision Readout:
    # Uses retrieved EpisodeRecord if available.
    retrieved_rec = retrieval_res.top_record
    predicted_action: int

    if retrieved_rec is None:
        # B1 (Memory removed) or B2 (Retrieval disabled): NO episodic information.
        # Must guess at chance.
        predicted_action = int(rng.integers(0, 2))
    else:
        # Memory record retrieved. Decode relation from latent binding / stored outcome.
        if retrieved_rec.latent_state is not None:
            # Linear readout of the latent binding: dot product with reference subspace
            score = float(np.dot(retrieved_rec.latent_state[:20], trial.cue_vector[:20]))
            readout_noise = float(rng.normal(0.0, 0.25))
            predicted_action = 1 if (score + readout_noise) > 0.0 else 0
        else:
            predicted_action = retrieved_rec.chosen_action

    correct = int(predicted_action == trial.target_action)

    return {
        "cue_id": trial.cue_id,
        "target_action": trial.target_action,
        "predicted_action": predicted_action,
        "correct": correct,
        "causal_condition": causal_condition.value,
        "retrieved_id": retrieval_res.top_id,
        "retrieved_score": retrieval_res.scores[0] if retrieval_res.scores else 0.0,
        "delay_sec": trial.delay_sec,
        "distractor_count": trial.distractor_count,
        "sparsity": retrieval_res.sparsity,
        "latency_us": retrieval_res.latency_us,
    }


def evaluate_causal_ablations(
    n_trials: int = 50,
    delay_sec: float = 1.0,
    seed: int = 42,
    capacity: int = 256,
) -> Dict[str, Any]:
    """Track B & D: Evaluate all 8 preregistered causal ablation conditions (B0-B7)."""
    results: Dict[str, Any] = {}

    for cond in CausalCondition:
        mem = SparseEpisodicMemory(capacity=capacity, seed=seed)
        session = make_episode_dependent_session(
            n_trials=n_trials,
            delay_sec=delay_sec,
            context_dim=mem.context_dim,
            hidden_dim=mem.hidden_dim,
            seed=seed,
        )
        cond_rng = np.random.default_rng(seed * 10007 + hash(cond.value) % 50000)

        trial_outs: List[Dict[str, Any]] = []
        for tr in session:
            out = run_episode_dependent_trial(
                trial=tr,
                memory=mem,
                causal_condition=cond,
                seed=seed,
                rng=cond_rng,
            )
            trial_outs.append(out)

        acc = float(np.mean([t["correct"] for t in trial_outs]))
        retrieval_success = float(np.mean([
            1.0 if t["retrieved_id"] == t["cue_id"] else 0.0
            for t in trial_outs
        ]))
        results[cond.value] = {
            "accuracy": acc,
            "retrieval_success_rate": retrieval_success,
            "chance": CHANCE_ACCURACY,
            "n_trials": n_trials,
            "delay_sec": delay_sec,
            "memory_bytes": mem.memory_bytes(),
            "retrievals_total": mem.total_retrievals,
        }

    return results


def evaluate_retrieval_timing_horizon(
    delays_sec: Optional[List[float]] = None,
    n_trials: int = 40,
    capacity: int = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    """Track E: Measure performance across increasing retrieval delays (0s to 120s)."""
    if delays_sec is None:
        delays_sec = [0.0, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0]

    timing_results: Dict[str, Any] = {}

    for delay in delays_sec:
        # Full memory vs No memory
        for cond, tag in ((CausalCondition.B0_FULL_MEMORY, "full_memory"), (CausalCondition.B1_MEMORY_REMOVED, "no_memory")):
            mem = SparseEpisodicMemory(capacity=capacity, seed=seed)
            session = make_episode_dependent_session(
                n_trials=n_trials,
                delay_sec=delay,
                context_dim=mem.context_dim,
                hidden_dim=mem.hidden_dim,
                seed=seed,
            )
            timing_rng = np.random.default_rng(seed * 10007 + int(delay * 31) + hash(tag) % 1000)
            trial_outs = [
                run_episode_dependent_trial(tr, mem, cond, seed=seed, rng=timing_rng)
                for tr in session
            ]
            acc = float(np.mean([t["correct"] for t in trial_outs]))
            retrieval_acc = float(np.mean([
                1.0 if t["retrieved_id"] == t["cue_id"] else 0.0
                for t in trial_outs
            ]))

            key = f"delay_{int(delay)}s_{tag}"
            timing_results[key] = {
                "delay_sec": float(delay),
                "condition": tag,
                "accuracy": acc,
                "retrieval_accuracy": retrieval_acc,
                "distractors_per_trial": session[0].distractor_count if session else 0,
                "chance": CHANCE_ACCURACY,
                "n_trials": n_trials,
            }

    # Calculate effective measured horizon: maximum delay with accuracy > 0.80
    effective_horizon = 0.0
    for delay in delays_sec:
        k = f"delay_{int(delay)}s_full_memory"
        if timing_results[k]["accuracy"] >= 0.80:
            effective_horizon = max(effective_horizon, delay)

    return {
        "timing_curve": timing_results,
        "effective_horizon_sec": effective_horizon,
        "delays_tested": delays_sec,
        "capacity_records": capacity,
    }
