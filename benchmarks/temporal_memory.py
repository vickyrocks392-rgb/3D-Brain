r"""Temporal Memory Benchmark Suite for 3D-Brain Phase 6.

Contains 4 controlled empirical tasks designed to test persistent internal latent memory:
- Task A: Delayed Cue Recall (delays: 10, 25, 50, 100, 250, 500)
- Task B: Sequence Reconstruction (overlapping sequences preventing trivial Markovian recall)
- Task C: Temporal Context Disambiguation (identical observation X conditioned on divergent histories)
- Task D: Multi-Step Forward Prediction (horizons: 1, 2, 5, 10, 25)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class DelayedCueRecallTask:
    """Task A: Delayed Cue Recall across varying distractor intervals."""

    def __init__(
        self,
        obs_dim: int = 8,
        num_cues: int = 4,
        delays: Optional[List[int]] = None,
        cue_duration: int = 5,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.num_cues = num_cues
        self.delays = delays or [10, 25, 50, 100, 250, 500]
        self.cue_duration = cue_duration
        self.rng = np.random.default_rng(seed)

        # Generate orthogonal/distinct prototype cue vectors in [0.2, 0.8]
        self.cues = np.zeros((num_cues, obs_dim), dtype=np.float32)
        for i in range(num_cues):
            vec = self.rng.uniform(0.2, 0.8, obs_dim).astype(np.float32)
            vec[0] = float(i) / max(1, num_cues - 1)  # Cue ID channel
            vec[-1] = 0.0  # Recall trigger channel (inactive during cue)
            self.cues[i] = vec

    def generate_trial(
        self, cue_idx: int, delay: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Generate a single trial sequence.

        Returns:
            inputs: [T, obs_dim] sensory inputs
            targets: [T, obs_dim] target reconstruction outputs
            mask: [T] binary evaluation mask (1.0 only at recall step)
        """
        T = self.cue_duration + delay + 1
        inputs = np.zeros((T, self.obs_dim), dtype=np.float32)
        targets = np.zeros((T, self.obs_dim), dtype=np.float32)
        mask = np.zeros(T, dtype=np.float32)

        target_cue = self.cues[cue_idx]

        # 1. Cue Presentation Period
        for t in range(self.cue_duration):
            inputs[t] = target_cue.copy()
            targets[t] = target_cue.copy()

        # 2. Distractor Period: zero or low-amplitude distractor noise
        for t in range(self.cue_duration, self.cue_duration + delay):
            # Distractor noise with channel 0 = 0.5 and recall channel = 0.0
            distractor = self.rng.normal(0.5, 0.05, self.obs_dim).astype(np.float32)
            distractor[-1] = 0.0
            inputs[t] = np.clip(distractor, 0.0, 1.0)
            targets[t] = np.zeros(self.obs_dim, dtype=np.float32)

        # 3. Recall Probe: Trigger channel active
        recall_step = T - 1
        probe = np.zeros(self.obs_dim, dtype=np.float32)
        probe[-1] = 1.0  # Trigger recall signal
        inputs[recall_step] = probe
        targets[recall_step] = target_cue.copy()
        mask[recall_step] = 1.0

        return inputs, targets, mask


class SequenceReconstructionTask:
    """Task B: Sequence Reconstruction with overlapping alphabet elements."""

    def __init__(
        self,
        obs_dim: int = 8,
        seq_length: int = 4,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.seq_length = seq_length
        self.rng = np.random.default_rng(seed)

        # 10 discrete alphabet symbols mapped to random sensory vectors
        self.alphabet_size = 10
        self.symbols = np.zeros((self.alphabet_size, obs_dim), dtype=np.float32)
        for i in range(self.alphabet_size):
            vec = self.rng.uniform(0.1, 0.9, obs_dim).astype(np.float32)
            vec[0] = float(i) / float(self.alphabet_size)
            self.symbols[i] = vec

        # Overlapping sequences:
        # Seq 0: 0 -> 1 -> 2 -> 3  (A -> B -> C -> D)
        # Seq 1: 4 -> 1 -> 2 -> 5  (E -> B -> C -> F)  (Shares B, C with Seq 0)
        # Seq 2: 6 -> 7 -> 2 -> 3  (G -> H -> C -> D)  (Shares C, D with Seq 0)
        # Seq 3: 0 -> 7 -> 8 -> 9  (A -> H -> I -> J)  (Shares A with Seq 0, H with Seq 2)
        self.sequences = [
            [0, 1, 2, 3],
            [4, 1, 2, 5],
            [6, 7, 2, 3],
            [0, 7, 8, 9],
        ]

    def generate_sequence_trial(
        self, seq_idx: int, delay: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        seq = self.sequences[seq_idx % len(self.sequences)]
        T = len(seq) + delay + 2
        inputs = np.zeros((T, self.obs_dim), dtype=np.float32)
        targets = np.zeros((T, self.obs_dim), dtype=np.float32)
        mask = np.zeros(T, dtype=np.float32)

        # Present sequence
        for t, sym in enumerate(seq):
            inputs[t] = self.symbols[sym]
            if t + 1 < len(seq):
                targets[t] = self.symbols[seq[t + 1]]

        # Delay
        for t in range(len(seq), len(seq) + delay):
            inputs[t] = np.zeros(self.obs_dim, dtype=np.float32)

        # Query 1: Predict sequence final element given previous
        q1_step = len(seq) + delay
        inputs[q1_step] = self.symbols[seq[-2]]  # Probe with element 2
        targets[q1_step] = self.symbols[seq[-1]]  # Target is element 3
        mask[q1_step] = 1.0

        # Query 2: Reconstruct initial sequence identity
        q2_step = q1_step + 1
        inputs[q2_step] = np.ones(self.obs_dim, dtype=np.float32) * 0.5
        inputs[q2_step, -1] = 1.0  # Identity query channel
        targets[q2_step] = self.symbols[seq[0]]  # Target is sequence start
        mask[q2_step] = 1.0

        meta = {
            "seq_idx": seq_idx,
            "seq": seq,
            "target_final": seq[-1],
            "target_start": seq[0],
        }
        return inputs, targets, mask, meta


class TemporalContextDisambiguationTask:
    """Task C: Temporal Context Disambiguation.

    Tests whether the network can disambiguate an identical sensory observation X
    that has different historical antecedents:
        History 1: A -> B -> X -> Y_1
        History 2: C -> D -> X -> Y_2
    """

    def __init__(self, obs_dim: int = 8, seed: int = 42):
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)

        # Define distinct sensory representations for events A, B, C, D, X, Y_1, Y_2
        self.A = self.rng.uniform(0.1, 0.9, obs_dim).astype(np.float32)
        self.B = self.rng.uniform(0.1, 0.9, obs_dim).astype(np.float32)
        self.C = self.rng.uniform(0.1, 0.9, obs_dim).astype(np.float32)
        self.D = self.rng.uniform(0.1, 0.9, obs_dim).astype(np.float32)
        # Identical ambiguous event X
        self.X = self.rng.uniform(0.1, 0.9, obs_dim).astype(np.float32)
        self.X[0] = 0.5
        self.X[1] = 0.5
        # Divergent future outcomes
        self.Y1 = self.rng.uniform(0.1, 0.9, obs_dim).astype(np.float32)
        self.Y1[0] = 0.9
        self.Y2 = self.rng.uniform(0.1, 0.9, obs_dim).astype(np.float32)
        self.Y2[0] = 0.1

    def generate_context_trial(
        self, context_id: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        r"""Generate sequence for context 1 (A->B->X->Y1) or context 2 (C->D->X->Y2).

        Step 0: A or C
        Step 1: B or D
        Step 2: X (IDENTICAL OBSERVATION)
        Step 3: Target is Y1 (for context 1) or Y2 (for context 2)
        """
        inputs = np.zeros((3, self.obs_dim), dtype=np.float32)
        targets = np.zeros((3, self.obs_dim), dtype=np.float32)

        if context_id == 1:
            inputs[0] = self.A
            inputs[1] = self.B
            inputs[2] = self.X  # Ambiguous step
            targets[0] = self.B
            targets[1] = self.X
            targets[2] = self.Y1  # Target conditioned on History 1
            expected_next = self.Y1
        else:
            inputs[0] = self.C
            inputs[1] = self.D
            inputs[2] = self.X  # Ambiguous step
            targets[0] = self.D
            targets[1] = self.X
            targets[2] = self.Y2  # Target conditioned on History 2
            expected_next = self.Y2

        meta = {
            "context_id": context_id,
            "ambiguous_step": 2,
            "expected_next": expected_next,
            "alternative_next": self.Y2 if context_id == 1 else self.Y1,
        }
        return inputs, targets, meta


class MultiStepPredictionEvaluator:
    """Task D: Multi-step prediction evaluator across horizons k in {1, 2, 5, 10, 25}."""

    def __init__(self, horizons: Optional[List[int]] = None):
        self.horizons = horizons or [1, 2, 5, 10, 25]

    def evaluate_trajectory(
        self,
        model: Any,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Evaluate multi-step prediction error over a recorded trajectory.

        Args:
            model: Neural model (PersistentLatentSpatialRNN or baseline).
            observations: [T, obs_dim] recorded environmental observations.
            actions: [T] recorded discrete actions.

        Returns:
            Dictionary of MSE per horizon k and error growth curve.
        """
        T = len(observations)
        max_k = max(self.horizons)
        horizon_errors: Dict[int, List[float]] = {k: [] for k in self.horizons}

        model.reset()
        for t in range(T - max_k - 1):
            obs_t = observations[t]
            act_t = int(actions[t])
            obs_next = observations[t + 1]

            # Prepare future targets
            future_dict = {k: observations[t + k] for k in self.horizons if t + k < T}

            # Step model
            if hasattr(model, "step"):
                try:
                    pred_next, err, step_m = model.step(
                        obs_t, act_t, obs_next, future_observations=future_dict
                    )
                    multistep_preds = step_m.get("multistep_preds", {})
                except TypeError:
                    # Baseline 1 / 2 without future_observations argument
                    pred_next, err, step_m = model.step(obs_t, act_t, obs_next)
                    multistep_preds = {}
            else:
                pred_next = np.zeros_like(obs_next)
                multistep_preds = {}

            # Measure error per horizon
            for k in self.horizons:
                target_k = observations[t + k]
                if k in multistep_preds:
                    pred_k = multistep_preds[k]
                else:
                    # 1-step baseline naive persistent forecast (s_t or pred_next)
                    pred_k = pred_next if k == 1 else obs_t

                mse_k = float(np.mean((target_k - pred_k) ** 2))
                horizon_errors[k].append(mse_k)

        summary = {}
        for k in self.horizons:
            errs = horizon_errors[k]
            summary[f"horizon_{k}_mse_mean"] = float(np.mean(errs)) if errs else 0.0
            summary[f"horizon_{k}_mse_std"] = float(np.std(errs)) if errs else 0.0

        return summary
