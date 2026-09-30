"""Phase 21 Track B: Replay Consolidation & Quiescent Rest Mechanisms.

Extends Phase 20 replay with explicit **consolidation** mechanisms so that
replay during silent rest is distinguishable from mere re-presentation of
stored answers:

   QUIESCENT REST (x = 0, no target, no reward, no supervision)
                    |
                    |---> replay selection (random / event / prioritized)
                    |
                    |---> reactivation restores dendritic tags from storage
                    |
                    |---> dual-timescale capture (fast pool -> slow pool)
                    |
                    |---> metaplasticity update of the effective learning rate

Key scientific distinction: `B0` (no replay) versus `B6` (replay + both
consolidation mechanisms) versus `R7` (random synthetic replay). If synthetic
replay matches exact replay, episodic specificity is NOT claimed.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain.episodic_memory import EpisodeRecord


class ReplayConsolidationCondition(str, Enum):
    """Track B conditions (preregistered B0-B6)."""

    B0_NO_REPLAY = "B0_NO_REPLAY"
    B1_RANDOM = "B1_RANDOM"
    B2_EVENT_TRIGGERED = "B2_EVENT_TRIGGERED"
    B3_PRIORITIZED = "B3_PRIORITIZED"
    B4_DUAL_TIMESCALE = "B4_DUAL_TIMESCALE"
    B5_METAPLASTICITY = "B5_METAPLASTICITY"
    B6_BOTH = "B6_BOTH"


class ReplayContent(str, Enum):
    """Replay content audit protocols (preregistered R1-R7)."""

    R1_EXACT = "R1_EXACT"
    R2_PARTIAL = "R2_PARTIAL"
    R3_LATENT_ONLY = "R3_LATENT_ONLY"
    R4_SHUFFLED = "R4_SHUFFLED"
    R5_REVERSED = "R5_REVERSED"
    R6_WRONG_EPISODE = "R6_WRONG_EPISODE"
    R7_RANDOM_SYNTHETIC = "R7_RANDOM_SYNTHETIC"


class ConsolidationAblation(str, Enum):
    """Causal interventions on consolidation (preregistered)."""

    NONE = "NONE"
    REMOVE_CONSOLIDATION = "REMOVE_CONSOLIDATION"
    ALTER_RATE = "ALTER_RATE"
    REMOVE_TAGGING = "REMOVE_TAGGING"
    REMOVE_METAPLASTICITY = "REMOVE_METAPLASTICITY"


class ConsolidationModule:
    """Dual-timescale synaptic pools with tagging and metaplasticity.

    Weights are split into a fast pool `W_fast` and a protected slow pool
    `W_slow`. Only synapses carrying an eligibility tag above threshold are
    captured into the slow pool, so consolidation requires genuine prior
    eligibility rather than indiscriminate copying.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        consolidation_rate: float = 0.1,
        tag_threshold: float = 0.05,
        tau_tag_ms: float = 1000.0,
        metaplasticity_lr: float = 0.01,
        tag_enabled: bool = True,
        metaplasticity_enabled: bool = True,
        seed: int = 42,
    ):
        self.shape = shape
        self.consolidation_rate = float(consolidation_rate)
        self.tag_threshold = float(tag_threshold)
        self.tau_tag_ms = float(tau_tag_ms)
        self.metaplasticity_lr = float(metaplasticity_lr)
        self.tag_enabled = bool(tag_enabled)
        self.metaplasticity_enabled = bool(metaplasticity_enabled)

        rng = np.random.default_rng(seed)
        limit = 1.0 / np.sqrt(max(1, shape[0]))
        self.weights_fast = rng.uniform(-limit, limit, size=shape).astype(np.float32)
        self.weights_slow = np.zeros(shape, dtype=np.float32)
        self.tags = np.zeros(shape, dtype=np.float32)
        self.metaplasticity = 0.5

        # Telemetry
        self.total_captures = 0
        self.total_tagged_synapses = 0
        self.last_captured_fraction = 0.0
        self.last_delta_norm = 0.0

    def effective_weights(self) -> np.ndarray:
        """Weights actually used for inference: fast + slow pool."""
        return self.weights_fast + self.weights_slow

    def set_tags(self, eligibility: np.ndarray) -> None:
        """Set synaptic tags from a dendritic eligibility snapshot."""
        if not self.tag_enabled:
            return
        self.tags = np.maximum(self.tags, np.abs(eligibility).astype(np.float32))

    def decay_tags(self, dt_ms: float) -> None:
        """Exponential decay of synaptic tags between rest steps."""
        if not self.tag_enabled:
            self.tags.fill(0.0)
            return
        self.tags *= np.exp(-dt_ms / max(1.0, self.tau_tag_ms))

    def update_metaplasticity(self, mean_eligibility: float) -> None:
        """BCM-like slow-state update of the effective learning-rate scale."""
        if not self.metaplasticity_enabled:
            return
        self.metaplasticity += self.metaplasticity_lr * (float(mean_eligibility) - self.metaplasticity)

    def effective_lr(self, base_lr: float) -> float:
        """Scale the base learning rate by the metaplastic state."""
        if not self.metaplasticity_enabled:
            return float(base_lr)
        return float(base_lr) * float(np.tanh(self.metaplasticity) + 1.0)

    def consolidate(self) -> Dict[str, Any]:
        """Capture tagged fast-pool synapses into the slow pool.

        Returns real capture statistics; when tagging is disabled, no capture
        occurs, which is the causal signature of `REMOVE_TAGGING`.
        """
        if not self.tag_enabled:
            self.last_captured_fraction = 0.0
            self.last_delta_norm = 0.0
            return {
                "captured_fraction": 0.0,
                "delta_norm": 0.0,
                "captured_synapses": 0,
            }

        eligible = self.tags >= self.tag_threshold
        n_eligible = int(np.count_nonzero(eligible))
        total = int(self.tags.size)
        frac = float(n_eligible) / float(max(1, total))

        before = float(np.linalg.norm(self.weights_slow))
        mask = eligible.astype(np.float32)
        self.weights_slow += self.consolidation_rate * mask * self.weights_fast
        # Tags are consumed by capture (capture-and-clear).
        self.tags *= (1.0 - mask)
        after = float(np.linalg.norm(self.weights_slow))

        self.total_captures += n_eligible
        self.total_tagged_synapses += n_eligible
        self.last_captured_fraction = frac
        self.last_delta_norm = abs(after - before)

        return {
            "captured_fraction": frac,
            "delta_norm": self.last_delta_norm,
            "captured_synapses": n_eligible,
        }

    def synaptic_norm(self) -> float:
        return float(np.linalg.norm(self.effective_weights()))

    def state_summary(self) -> Dict[str, Any]:
        """Real consolidation state for the [C] observatory overlay."""
        return {
            "fast_norm": round(float(np.linalg.norm(self.weights_fast)), 6),
            "slow_norm": round(float(np.linalg.norm(self.weights_slow)), 6),
            "tag_mass": round(float(np.sum(self.tags)), 6),
            "tagged_fraction": round(
                float(np.count_nonzero(self.tags >= self.tag_threshold)) / float(max(1, self.tags.size)), 6
            ),
            "metaplasticity": round(float(self.metaplasticity), 6),
            "captured_fraction_last": round(float(self.last_captured_fraction), 6),
            "delta_norm_last": round(float(self.last_delta_norm), 6),
            "total_captures": int(self.total_captures),
            "tagging_enabled": bool(self.tag_enabled),
            "metaplasticity_enabled": bool(self.metaplasticity_enabled),
            "consolidation_rate": float(self.consolidation_rate),
        }

    def memory_bytes(self) -> int:
        """Bytes of slow pool + tags (consolidation substrate, not parameters)."""
        return int(self.weights_slow.nbytes + self.tags.nbytes)

    def parameter_count(self) -> int:
        """Trainable fast-pool weights only; slow pool is dynamic state."""
        return int(self.weights_fast.size)


class ReplayConsolidationEngine:
    """Replay selection + content transformation + consolidation driver.

    Replay operates on stored :class:`EpisodeRecord` objects. The content audit
    protocols deliberately corrupt the replayed material so that we can
    determine *what replay actually contributes*.
    """

    def __init__(
        self,
        consolidation: ConsolidationModule,
        replay_budget: int = 4,
        surprise_threshold: float = 0.4,
        priority_alpha: float = 1.0,
        replay_step_lr: float = 0.002,
        condition: ReplayConsolidationCondition = ReplayConsolidationCondition.B0_NO_REPLAY,
        content: ReplayContent = ReplayContent.R1_EXACT,
        seed: int = 42,
    ):
        self.consolidation = consolidation
        self.replay_budget = max(1, int(replay_budget))
        self.surprise_threshold = float(surprise_threshold)
        self.priority_alpha = float(priority_alpha)
        self.replay_step_lr = float(replay_step_lr)
        self.condition = condition
        self.content = content
        self.rng = np.random.default_rng(seed)

        # Telemetry
        self.total_replay_events = 0
        self.total_replay_flops = 0
        self.total_distinct_replayed = 0
        self.replay_frequency = 0.0
        self.replay_diversity = 0.0
        self.replay_selectivity = 0.0
        self.last_replayed_ids: List[int] = []
        self.last_replay_detail: Dict[str, Any] = {}

    # -- selection --------------------------------------------------------------
    def select(self, records: List[EpisodeRecord]) -> List[EpisodeRecord]:
        """Select a replay batch under the active sampling condition."""
        pool = [r for r in records if r.is_closed]
        if not pool or self.condition == ReplayConsolidationCondition.B0_NO_REPLAY:
            return []

        k = min(self.replay_budget, len(pool))

        if self.condition == ReplayConsolidationCondition.B1_RANDOM:
            idx = self.rng.choice(len(pool), size=k, replace=False)
            return [pool[i] for i in idx]

        if self.condition == ReplayConsolidationCondition.B2_EVENT_TRIGGERED:
            surprising = [r for r in pool if r.prediction_error >= self.surprise_threshold]
            if not surprising:
                return []
            idx = self.rng.choice(len(surprising), size=min(k, len(surprising)), replace=False)
            return [surprising[i] for i in idx]

        # B3-B6: prioritized by prediction-error magnitude
        errors = np.array([max(1e-4, r.prediction_error) for r in pool], dtype=np.float64)
        w = errors ** self.priority_alpha
        probs = w / float(np.sum(w))
        idx = self.rng.choice(len(pool), size=k, replace=False, p=probs)
        return [pool[i] for i in idx]

    # -- content transformation --------------------------------------------------
    def _transform(self, records: List[EpisodeRecord]) -> List[Dict[str, np.ndarray]]:
        """Apply the content-audit transformation to a replay batch.

        Returns dicts of the material actually used for the replay update:
        `context` (readout input) and `latent` (mean event state).
        """
        out: List[Dict[str, np.ndarray]] = []
        for r in records:
            ctx = np.asarray(r.context_vector, dtype=np.float32).copy()
            ev = np.asarray(r.event_sequence, dtype=np.float32)
            latent = ev.mean(axis=0) if ev.size else ctx

            if self.content == ReplayContent.R5_REVERSED:
                # Reverse temporal order, preserving membership.
                rev = ev[::-1] if ev.size else ev
                latent = rev.mean(axis=0) if rev.size else ctx
                ctx = (ctx + latent) / 2.0
            elif self.content == ReplayContent.R6_WRONG_EPISODE:
                # Keep the latent, scramble the outcome association.
                ctx = latent.astype(np.float32)
            elif self.content == ReplayContent.R7_RANDOM_SYNTHETIC:
                # Synthetic material drawn from no episode at all.
                ctx = self.rng.normal(0.0, 1.0, size=ctx.shape).astype(np.float32)
                latent = self.rng.normal(0.0, 1.0, size=latent.shape).astype(np.float32)

            out.append({
                "context": ctx.astype(np.float32),
                "latent": np.asarray(latent, dtype=np.float32),
            })

        if self.content == ReplayContent.R4_SHUFFLED:
            perm = self.rng.permutation(len(out))
            out = [out[i] for i in perm]
        elif self.content == ReplayContent.R2_PARTIAL:
            for item in out:
                item["context"] = item["latent"].astype(np.float32)

        return out

    # -- rest-period driver ------------------------------------------------------
    def run_rest(
        self,
        records: List[EpisodeRecord],
        num_rest_steps: int = 10,
        dt_ms: float = 40.0,
    ) -> Dict[str, Any]:
        """Execute one quiescent rest period with replay and consolidation.

        No sensory input, no target, and no reward are supplied here: the only
        information used is what was legitimately stored during active
        experience.
        """
        batch = self.select(records)
        if not batch:
            return {
                "rest_steps": int(num_rest_steps),
                "replayed": 0,
                "capture": self.consolidation.consolidate(),
                "distinct_replayed": 0,
            }

        items = self._transform(batch)
        n_cols = self.consolidation.shape[1]
        ones = np.ones(n_cols, dtype=np.float32)
        W = self.consolidation.weights_fast

        for item in items:
            ctx = item["context"]
            # Restore dendritic tags from the reactivated state.
            elig = np.outer(np.abs(ctx), ones)
            self.consolidation.update_metaplasticity(float(np.mean(np.abs(ctx))))
            self.consolidation.set_tags(elig)

            # Zero-centred covariance replay drive. The step is normalised by
            # the drive magnitude so that repeated replay cannot swamp the
            # task-relevant weights with an unbounded random walk.
            centred = ctx - float(np.mean(ctx))
            drive = np.outer(centred, ones)
            scale = float(np.linalg.norm(drive))
            if scale > 1e-8:
                drive = drive / scale * self.consolidation.weights_fast.shape[0] ** 0.5
            W += self.replay_step_lr * drive

        self.consolidation.decay_tags(dt_ms * num_rest_steps)
        capture = self.consolidation.consolidate()

        # -- telemetry --------------------------------------------------------
        self.total_replay_events += len(batch)
        self.total_distinct_replayed += len(set(r.episode_id for r in batch))
        self.total_replay_flops += 4 * self.consolidation.shape[0] * n_cols * len(batch)
        self.replay_frequency = len(batch) / float(max(1, num_rest_steps))
        self.replay_diversity = len(set(r.episode_id for r in batch)) / float(len(batch))
        self.last_replayed_ids = [int(r.episode_id) for r in batch]
        self.last_replay_detail = {
            "content": self.content.value,
            "episode_ids": self.last_replayed_ids,
            "source_tasks": [r.task_name for r in batch],
            "prediction_errors": [round(float(r.prediction_error), 4) for r in batch],
            "rewards": [round(float(r.reward_val), 3) for r in batch],
            "timestamps_ms": [round(float(r.timestamp_ms), 1) for r in batch],
            "sequence_length": int(len(batch)),
        }

        return {
            "rest_steps": int(num_rest_steps),
            "replayed": len(batch),
            "capture": capture,
            "distinct_replayed": len(set(r.episode_id for r in batch)),
        }

    def telemetry(self) -> Dict[str, Any]:
        """Real computational state for the [R] and [C] observatory overlays."""
        return {
            "condition": self.condition.value,
            "content": self.content.value,
            "replay_budget": self.replay_budget,
            "total_replay_events": int(self.total_replay_events),
            "total_replay_flops": int(self.total_replay_flops),
            "replay_frequency": round(float(self.replay_frequency), 4),
            "replay_diversity": round(float(self.replay_diversity), 4),
            "last_replay": self.last_replay_detail,
        }
