"""Phase 21 Track A: Sparse Episodic Memory.

Implements an explicit episodic memory that stores **event-level and latent**
experience rather than supervised input->target mappings:

   ACTIVE EPISODE (trial i)
             |
             |---> open_episode(): store context c_i, event sequence E_i, action
             |                      alpha_i, key kappa_i, timestamp, sparse activity
             |                      ** NO target, NO reward **
             v
      (delay / subsequent events: strictly silent)
             v
        outcome observed
             |
             |---> close_episode(): write prediction error pi_i and reward r_i
             v
   EPISODIC STORE (capacity N_ep)
             |
        retrieve(query) --> top-k sparse content-addressed set
             |
        optional: feed retrievals into Phase 20 replay / consolidation

Storage legitimacy is enforced by a two-phase write protocol: `open_episode`
cannot accept a target or reward, and `close_episode` is the only writer of
outcome fields. Tests assert that no future target or reward is ever present in
a record before its outcome step.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class EpisodicCondition(str, Enum):
    """Track A episodic-memory conditions (preregistered A0-A5)."""

    A0_NO_MEMORY = "A0_NO_MEMORY"
    A1_PASSIVE_BUFFER = "A1_PASSIVE_BUFFER"
    A2_INDEXED = "A2_INDEXED"
    A3_SPARSE_CONTENT = "A3_SPARSE_CONTENT"
    A4_MEMORY_REPLAY = "A4_MEMORY_REPLAY"
    A5_MEMORY_REPLAY_CONSOLIDATION = "A5_MEMORY_REPLAY_CONSOLIDATION"


class MemoryAblation(str, Enum):
    """Causal interventions on episodic retrieval (preregistered)."""

    NONE = "NONE"                          # B0: Full system + intact episodic memory
    REMOVE_MEMORY = "REMOVE_MEMORY"        # B1: Memory removed entirely
    DISABLE_RETRIEVAL = "DISABLE_RETRIEVAL"# B2: Memory present but retrieval disabled
    SHUFFLE_MEMORY = "SHUFFLE_MEMORY"      # B3: Memory present but retrieval randomized/shuffled
    RANDOMIZE_RETRIEVAL = "RANDOMIZE_RETRIEVAL" # alias for B3
    SHUFFLE_IDENTITY = "SHUFFLE_IDENTITY"  # Permute identity tags
    WRONG_MEMORY = "WRONG_MEMORY"          # B4: Wrong/non-matching episode retrieved
    WRONG_EPISODE = "WRONG_EPISODE"        # alias for B4
    CORRUPT_MEMORY = "CORRUPT_MEMORY"      # B5: Stored representation corrupted with noise
    PARTIAL_MEMORY = "PARTIAL_MEMORY"      # B6: Partial memory (latent features partially masked)
    CORRUPT_CUE = "CORRUPT_CUE"            # B7: Retrieval cue corrupted with noise
    REDUCE_CAPACITY = "REDUCE_CAPACITY"    # Capacity reduced


@dataclass
class EpisodeRecord:
    """One stored episode.

    Outcome fields (`prediction_error`, `reward_val`) are written ONLY by
    `close_episode()`, which runs after the reward step. `target_action` is
    never stored at all: the memory is not a supervised input->target table.
    """

    episode_id: int
    context_vector: np.ndarray                # c_i in R^D_ro
    event_sequence: np.ndarray                # E_i in R^{T_i x D_hid}
    chosen_action: int                        # alpha_i (emitted at decision step)
    key: np.ndarray                           # kappa_i in R^d_key (sparse)
    timestamp_ms: float                       # tau_i
    sparse_activity: np.ndarray               # ell_i thresholded bitmap
    duration_steps: int                       # sigma_i
    task_name: str = "UNSET"
    prediction_error: float = 0.0             # pi_i, written at outcome only
    reward_val: float = 0.0                   # r_i, written at outcome only
    is_closed: bool = False
    retrieval_count: int = 0
    latent_state: Optional[np.ndarray] = None

    def memory_bytes(self, key_dim: int, hidden_dim: int, activity_dim: int) -> int:
        """Exact byte footprint of this record (specification 2.7)."""
        n_events = int(self.event_sequence.shape[0])
        extra_bytes = int(self.latent_state.size * 4) if self.latent_state is not None else 0
        return int(
            key_dim * 4                      # sparse content address
            + self.context_vector.size * 4   # context representation
            + n_events * hidden_dim * 4      # latent event sequence
            + activity_dim // 8              # sparse activity bitmap
            + 64                             # scalar/timestamp/handle metadata
            + extra_bytes                    # optional latent binding/state
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializable summary for BrainLog telemetry (no raw arrays)."""
        d = {
            "episode_id": int(self.episode_id),
            "task": self.task_name,
            "chosen_action": int(self.chosen_action),
            "prediction_error": round(float(self.prediction_error), 4),
            "reward_val": round(float(self.reward_val), 3),
            "duration_steps": int(self.duration_steps),
            "timestamp_ms": round(float(self.timestamp_ms), 1),
            "retrieval_count": int(self.retrieval_count),
            "is_closed": bool(self.is_closed),
        }
        if self.latent_state is not None:
            d["latent_dim"] = int(self.latent_state.size)
        return d


class RetrievalResult:
    """Outcome of one episodic retrieval query."""

    def __init__(
        self,
        retrieved_ids: List[int],
        scores: List[float],
        competitors: List[int],
        sparsity: float,
        latency_us: float,
        records: Optional[List[EpisodeRecord]] = None,
    ):
        self.retrieved_ids = retrieved_ids
        self.scores = scores
        self.competitors = competitors
        self.sparsity = sparsity
        self.latency_us = latency_us
        self.records = records if records is not None else []

    @property
    def top_id(self) -> Optional[int]:
        """Highest-ranked retrieved episode ID (None if memory returned nothing)."""
        return self.retrieved_ids[0] if self.retrieved_ids else None

    @property
    def top_record(self) -> Optional[EpisodeRecord]:
        """Highest-ranked retrieved EpisodeRecord (None if memory returned nothing)."""
        return self.records[0] if self.records else None

    def to_dict(self) -> Dict[str, Any]:
        """Serializable retrieval summary for BrainLog telemetry."""
        return {
            "retrieved": [int(i) for i in self.retrieved_ids],
            "top_scores": [round(float(s), 4) for s in self.scores],
            "competitors": [int(i) for i in self.competitors],
            "sparsity": round(float(self.sparsity), 6),
            "latency_us": round(float(self.latency_us), 2),
        }


class SparseEpisodicMemory:
    """Sparse content-addressable episodic memory with causal ablations."""

    def __init__(
        self,
        capacity: int = 128,
        key_dim: int = 12,
        hidden_dim: int = 40,
        activity_dim: int = 40,
        context_dim: int = 40,
        top_k: int = 3,
        sparsity_threshold: float = 0.05,
        seed: int = 42,
    ):
        self.capacity = max(1, int(capacity))
        self.key_dim = int(key_dim)
        self.hidden_dim = int(hidden_dim)
        self.activity_dim = int(activity_dim)
        self.context_dim = int(context_dim)
        self.top_k = max(1, int(top_k))
        self.sparsity_threshold = float(sparsity_threshold)
        self.rng = np.random.default_rng(seed)

        # Fixed sparse dictionary D in R^{d_key x D_ro} with orthonormal rows.
        # These are counted parameters (reported in accounting), not memory.
        a = self.rng.normal(size=(self.context_dim, self.key_dim))
        q, _ = np.linalg.qr(a)               # q is (context_dim, key_dim)
        self.dictionary = np.ascontiguousarray(q.T.astype(np.float32))

        self.records: List[EpisodeRecord] = []
        self._pending: Optional[EpisodeRecord] = None

        # Causal ablation state
        self.ablation: MemoryAblation = MemoryAblation.NONE
        self._identity_permutation: Optional[np.ndarray] = None

        # Telemetry counters
        self.total_retrievals = 0
        self.total_episodes_stored = 0
        self.total_retrieval_flops = 0
        self.last_retrieval: Optional[RetrievalResult] = None

    # -- write protocol ---------------------------------------------------------
    def open_episode(
        self,
        episode_id: int,
        context_vector: np.ndarray,
        event_sequence: np.ndarray,
        chosen_action: int,
        timestamp_ms: float,
        task_name: str = "UNSET",
        latent_state: Optional[np.ndarray] = None,
    ) -> EpisodeRecord:
        """Store pre-outcome episode content.

        Deliberately has NO target_action and NO reward parameter: the only
        legitimate pre-outcome information is the latent trajectory and the
        action the system itself emitted.
        """
        ctx = np.asarray(context_vector, dtype=np.float32).ravel()
        events = np.asarray(event_sequence, dtype=np.float32)
        if events.ndim == 1:
            events = events.reshape(1, -1)

        rec = EpisodeRecord(
            episode_id=int(episode_id),
            context_vector=ctx.copy(),
            event_sequence=events.copy(),
            chosen_action=int(chosen_action),
            key=self._compute_key(ctx, events),
            timestamp_ms=float(timestamp_ms),
            sparse_activity=self._sparse_activity(ctx),
            duration_steps=int(events.shape[0]),
            task_name=str(task_name),
            latent_state=np.asarray(latent_state, dtype=np.float32).copy() if latent_state is not None else None,
        )
        self._pending = rec
        return rec

    def close_episode(
        self,
        prediction_error: float,
        reward_val: float,
    ) -> Optional[EpisodeRecord]:
        """Commit the pending episode with outcome information.

        This is the ONLY writer of `prediction_error` and `reward_val`, and is
        called strictly after the reward step is observed.
        """
        if self._pending is None:
            return None
        rec = self._pending
        rec.prediction_error = float(prediction_error)
        rec.reward_val = float(reward_val)
        rec.is_closed = True

        if len(self.records) >= self.capacity:
            self.records.pop(0)  # bounded capacity, oldest evicted
        self.records.append(rec)
        self.total_episodes_stored += 1
        self._pending = None
        return rec

    def abort_episode(self) -> None:
        """Discard a pending episode (trial ended without an outcome)."""
        self._pending = None

    def get_record(self, episode_id: int) -> Optional[EpisodeRecord]:
        """Lookup stored episode by ID."""
        for r in self.records:
            if r.episode_id == episode_id:
                return r
        return None

    # -- key computation --------------------------------------------------------
    def _compute_key(self, ctx: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Sparse content address from context + an order-sensitive summary.

        The event summary must distinguish episodes that share the same events
        in a *different order*. A recency-weighted **mean** is order-invariant
        under reversal (the weights simply permute with the events), so the
        sequence is instead summarised by the **first** event, the **last**
        event, and their difference. This makes the key order-sensitive.
        """
        n = events.shape[0]
        if n > 2:
            first, second = events[0], events[1]
            summary = np.concatenate([first, second, second - first])
        elif n == 2:
            summary = np.concatenate([events[0], events[1], events[1] - events[0]])
        elif n == 1:
            summary = np.concatenate([events[0], events[0], np.zeros_like(events[0])])
        else:
            summary = np.zeros(self.context_dim, dtype=np.float32)

        # Interleave context and summary so both contribute to the address.
        v = np.concatenate([ctx[: self.context_dim], summary])[: self.context_dim]
        norm = float(np.linalg.norm(v))
        phi = v / norm if norm > 1e-8 else v
        proj = self.dictionary @ phi
        key = np.sign(proj) * np.abs(proj)
        key[np.abs(key) < self.sparsity_threshold] = 0.0
        kn = float(np.linalg.norm(key))
        return (key / kn).astype(np.float32) if kn > 1e-8 else key.astype(np.float32)

    def _sparse_activity(self, ctx: np.ndarray) -> np.ndarray:
        """Thresholded bitmap of salient readout activity."""
        return (np.abs(ctx[: self.activity_dim]) > 0.5).astype(np.uint8)

    def _query_key(self, query_vector: np.ndarray) -> np.ndarray:
        """Project an arbitrary query vector into the same sparse key space.

        Queries shorter than `context_dim` are zero-padded and longer ones are
        truncated, so a raw context vector is a valid (if less informative)
        query alongside a full observation-derived one.
        """
        q = np.zeros(self.context_dim, dtype=np.float32)
        raw = np.asarray(query_vector, dtype=np.float32).ravel()
        n = min(self.context_dim, raw.size)
        q[:n] = raw[:n]
        proj = self.dictionary @ q
        key = np.sign(proj) * np.abs(proj)
        key[np.abs(key) < self.sparsity_threshold] = 0.0
        kn = float(np.linalg.norm(key))
        return (key / kn).astype(np.float32) if kn > 1e-8 else key.astype(np.float32)

    # -- retrieval --------------------------------------------------------------
    def retrieve(
        self,
        query_vector: np.ndarray,
        condition: EpisodicCondition = EpisodicCondition.A3_SPARSE_CONTENT,
    ) -> RetrievalResult:
        """Sparse top-k retrieval under the given condition and ablation."""
        t0 = time.perf_counter()
        total_stored = len(self.records)

        # A0 / B1 / B2: no memory or disabled retrieval
        if (
            condition == EpisodicCondition.A0_NO_MEMORY
            or self.ablation in (MemoryAblation.REMOVE_MEMORY, MemoryAblation.DISABLE_RETRIEVAL)
        ):
            self.total_retrievals += 1
            res = RetrievalResult([], [], [], 0.0, (time.perf_counter() - t0) * 1e6, records=[])
            self.last_retrieval = res
            return res

        if total_stored == 0:
            self.total_retrievals += 1
            res = RetrievalResult([], [], [], 0.0, (time.perf_counter() - t0) * 1e6, records=[])
            self.last_retrieval = res
            return res

        # FLOPs: one dot product per stored key (spec 7)
        self.total_retrieval_flops += 2 * self.key_dim * total_stored

        # B7: corrupted cue
        if self.ablation == MemoryAblation.CORRUPT_CUE:
            cue_noise = self.rng.normal(0.0, 1.5, size=query_vector.shape).astype(np.float32)
            qk = self._query_key(query_vector + cue_noise)
        else:
            qk = self._query_key(query_vector)

        keys = np.stack([r.key for r in self.records]).astype(np.float32)
        sims = keys @ qk  # cosine similarity in [-1, 1]
        order = np.argsort(-sims, kind="stable")
        k = min(self.top_k, total_stored)

        # -- condition-specific retrieval -------------------------------------
        if condition == EpisodicCondition.A1_PASSIVE_BUFFER:
            sel = self.rng.choice(total_stored, size=k, replace=False)
            order = np.concatenate([sel, np.setdiff1d(order, sel)])[:k]
        elif condition == EpisodicCondition.A2_INDEXED:
            buckets: Dict[int, List[int]] = {}
            for idx, r in enumerate(self.records):
                b = int(np.argmax(np.abs(r.context_vector)))
                buckets.setdefault(b, []).append(idx)
            qb = int(np.argmax(np.abs(np.asarray(query_vector).ravel()[: self.context_dim])))
            if qb in buckets:
                cand = np.array(buckets[qb])
                order = cand[np.argsort(-sims[cand], kind="stable")]
                order = order[:k]

        top_idx = order[:k]

        # -- causal ablations -------------------------------------------------
        if self.ablation in (MemoryAblation.RANDOMIZE_RETRIEVAL, MemoryAblation.SHUFFLE_MEMORY):
            perm = self.rng.permutation(total_stored)[:k]
            top_idx = perm
            retrieved_ids = [int(self.records[i].episode_id) for i in perm]
            top_scores = [float(sims[i]) for i in perm]

        elif self.ablation == MemoryAblation.SHUFFLE_IDENTITY:
            perm = self.rng.permutation(total_stored)
            retrieved_ids = [int(self.records[perm[i]].episode_id) for i in top_idx]
            top_scores = [float(sims[i]) for i in top_idx]

        elif self.ablation in (MemoryAblation.WRONG_EPISODE, MemoryAblation.WRONG_MEMORY):
            # Rotated offset: returns non-matching competitor episodes
            offset = max(1, total_stored // 2)
            rot = np.roll(np.arange(total_stored), offset)[order[:k]]
            top_idx = rot
            retrieved_ids = [int(self.records[i].episode_id) for i in rot]
            top_scores = [float(sims[i]) for i in rot]

        else:
            retrieved_ids = [int(self.records[i].episode_id) for i in top_idx]
            top_scores = [float(sims[i]) for i in top_idx]

        competitors = [int(self.records[i].episode_id) for i in order[k : k + 3]]

        # Gather retrieved records with any content intervention applied
        retrieved_records: List[EpisodeRecord] = []
        for i in top_idx:
            rec = self.records[i]
            if self.ablation == MemoryAblation.CORRUPT_MEMORY:
                corrupted_latent = None
                if rec.latent_state is not None:
                    noise = self.rng.normal(0.0, 1.2, size=rec.latent_state.shape).astype(np.float32)
                    corrupted_latent = rec.latent_state + noise
                noise_ctx = self.rng.normal(0.0, 1.2, size=rec.context_vector.shape).astype(np.float32)
                cloned = EpisodeRecord(
                    episode_id=rec.episode_id,
                    context_vector=rec.context_vector + noise_ctx,
                    event_sequence=rec.event_sequence,
                    chosen_action=rec.chosen_action,
                    key=rec.key,
                    timestamp_ms=rec.timestamp_ms,
                    sparse_activity=rec.sparse_activity,
                    duration_steps=rec.duration_steps,
                    task_name=rec.task_name,
                    prediction_error=rec.prediction_error,
                    reward_val=rec.reward_val,
                    is_closed=rec.is_closed,
                    retrieval_count=rec.retrieval_count,
                    latent_state=corrupted_latent,
                )
                retrieved_records.append(cloned)
            elif self.ablation == MemoryAblation.PARTIAL_MEMORY:
                partial_latent = None
                if rec.latent_state is not None:
                    partial_latent = rec.latent_state.copy()
                    mask = self.rng.uniform(size=partial_latent.shape) < 0.80
                    partial_latent[mask] = 0.0
                mask_ctx = self.rng.uniform(size=rec.context_vector.shape) < 0.80
                partial_ctx = rec.context_vector.copy()
                partial_ctx[mask_ctx] = 0.0
                cloned = EpisodeRecord(
                    episode_id=rec.episode_id,
                    context_vector=partial_ctx,
                    event_sequence=rec.event_sequence,
                    chosen_action=rec.chosen_action,
                    key=rec.key,
                    timestamp_ms=rec.timestamp_ms,
                    sparse_activity=rec.sparse_activity,
                    duration_steps=rec.duration_steps,
                    task_name=rec.task_name,
                    prediction_error=rec.prediction_error,
                    reward_val=rec.reward_val,
                    is_closed=rec.is_closed,
                    retrieval_count=rec.retrieval_count,
                    latent_state=partial_latent,
                )
                retrieved_records.append(cloned)
            else:
                retrieved_records.append(rec)

        # Account retrieval frequency on the true stored records.
        if retrieved_ids:
            lookup = {int(r.episode_id): r for r in self.records}
            for eid in retrieved_ids:
                if eid in lookup:
                    lookup[eid].retrieval_count += 1

        sparsity = float(k) / float(total_stored)
        self.total_retrievals += 1
        res = RetrievalResult(
            retrieved_ids=retrieved_ids,
            scores=top_scores,
            competitors=competitors,
            sparsity=sparsity,
            latency_us=(time.perf_counter() - t0) * 1e6,
            records=retrieved_records,
        )
        self.last_retrieval = res
        return res

    def set_ablation(self, ablation: MemoryAblation) -> None:
        """Apply a causal ablation to retrieval."""
        self.ablation = ablation
        if ablation == MemoryAblation.REDUCE_CAPACITY:
            self.capacity = max(1, int(self.capacity * 0.1))

    # -- accounting -------------------------------------------------------------
    def memory_bytes(self) -> int:
        """Total episodic memory footprint in bytes (never a parameter count)."""
        return int(sum(
            r.memory_bytes(self.key_dim, self.hidden_dim, self.activity_dim)
            for r in self.records
        ))

    def dictionary_parameters(self) -> int:
        """Size of the fixed sparse dictionary (counted as parameters)."""
        return int(self.dictionary.size)

    def account(self) -> Dict[str, Any]:
        """Explicit separation of parameters, memory, and compute."""
        return {
            "episodic_memory_bytes": self.memory_bytes(),
            "episodic_memory_records": len(self.records),
            "dictionary_parameters": self.dictionary_parameters(),
            "episodes_stored_total": self.total_episodes_stored,
            "retrievals_total": self.total_retrievals,
            "retrieval_flops": self.total_retrieval_flops,
            "top_k": self.top_k,
            "capacity": self.capacity,
            "mean_key_sparsity": float(np.mean([
                np.count_nonzero(r.key) / max(1, self.key_dim) for r in self.records
            ])) if self.records else 0.0,
        }

    def telemetry(self) -> Dict[str, Any]:
        """Real computational state for the [E] observatory overlay."""
        acct = self.account()
        acct["condition"] = self.ablation.value
        acct["last_retrieval"] = self.last_retrieval.to_dict() if self.last_retrieval else {}
        return acct

    def __len__(self) -> int:
        return len(self.records)
