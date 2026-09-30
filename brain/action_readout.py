"""Action Readout Architecture & Target-Specific Dendritic Learning (Phase 15).

Provides a distinct, modular, and inspectable action-readout component that maps
temporally integrated neural representations into competitive action populations:

  TEMPORAL CONTEXT REPRESENTATION (I0 - I6)
                 │
                 ▼
        ACTION READOUT SYNAPSES
    (Target-Specific Dendritic Plasticity,
     Retrograde Signaling & Dendritic STDP)
                 │
                 ▼
       ACTION POPULATIONS (A0, A1, ...)
                 │
                 ▼
     MULTI-TIMESCALE COMPETITION (Fast / Med / Slow)
                 │
                 ▼
         PERSISTENT ACTION

Key Capabilities:
- Experimental isolation: freeze upstream brain, freeze readout, lesion, scramble, replace
- Target-Specific Learning (D1 - D8)
- Target-Specific Teaching Signals (G0 - G5)
- Postsynaptic-to-Dendritic Retrograde Signaling (R0 - R4)
- Dendritic Spike-Timing-Dependent Plasticity (S0 - S4)
- Input Representation Selectors (I0 - I6)
- Fine-grained Synapse-Level Inspection & Credit Telemetry
"""

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


from brain.synaptic_consolidation import ConsolidationMode, SynapticConsolidationModule


class InputRepresentationType(str, Enum):
    """Source representation feeding into the action readout."""
    I0_INSTANTANEOUS_SPIKES = "I0_INSTANTANEOUS_SPIKES"
    I1_SPIKE_COUNT = "I1_SPIKE_COUNT"
    I2_BURST_REPRESENTATION = "I2_BURST_REPRESENTATION"
    I3_TEMPORAL_POPULATION = "I3_TEMPORAL_POPULATION"
    I4_PHASE_REPRESENTATION = "I4_PHASE_REPRESENTATION"
    I5_LOCAL_RESONANCE = "I5_LOCAL_RESONANCE"
    I6_COMBINED_TEMPORAL = "I6_COMBINED_TEMPORAL"


class LearningMechanism(str, Enum):
    """Plasticity mechanisms for the action readout."""
    D1_FROZEN = "D1_FROZEN"
    D2_LOCAL_HEBBIAN = "D2_LOCAL_HEBBIAN"
    D3_ELIGIBILITY_ONLY = "D3_ELIGIBILITY_ONLY"
    D4_GLOBAL_REWARD = "D4_GLOBAL_REWARD"
    D5_TARGET_SPECIFIC_REWARD = "D5_TARGET_SPECIFIC_REWARD"
    D6_PRED_ERROR_GATED = "D6_PRED_ERROR_GATED"
    D7_DENDRITIC_TEACHING = "D7_DENDRITIC_TEACHING"
    D8_TEMPORAL_TARGET_DENDRITIC = "D8_TEMPORAL_TARGET_DENDRITIC"


class TeachingSignalType(str, Enum):
    """Modulatory teaching signal structure Mk(t)."""
    G0_GLOBAL_SCALAR = "G0_GLOBAL_SCALAR"
    G1_ACTION_TARGET_SCALAR = "G1_ACTION_TARGET_SCALAR"
    G2_POPULATION_LOCAL_SCALAR = "G2_POPULATION_LOCAL_SCALAR"
    G3_DENDRITIC_TARGET_SIGNAL = "G3_DENDRITIC_TARGET_SIGNAL"
    G4_SHUFFLED_TARGET = "G4_SHUFFLED_TARGET"
    G5_SIGN_FLIPPED = "G5_SIGN_FLIPPED"


class RetrogradeSignalType(str, Enum):
    """Retrograde signaling from postsynaptic action compartment to dendritic synapses."""
    R0_NONE = "R0_NONE"
    R1_SCALAR = "R1_SCALAR"
    R2_ACTION_SPECIFIC = "R2_ACTION_SPECIFIC"
    R3_COMPARTMENT_SPECIFIC = "R3_COMPARTMENT_SPECIFIC"
    R4_DELAYED = "R4_DELAYED"


class DendriticSTDPType(str, Enum):
    """Dendritic timing-dependent plasticity rules."""
    S0_NONE = "S0_NONE"
    S1_SOMATIC_PAIR = "S1_SOMATIC_PAIR"
    S2_DENDRITIC_PAIR = "S2_DENDRITIC_PAIR"
    S3_ELIGIBILITY_STDP = "S3_ELIGIBILITY_STDP"
    S4_TARGET_SPECIFIC_STDP = "S4_TARGET_SPECIFIC_STDP"


class ReadoutSynapseRecord:
    """Detailed biophysical inspection record for an individual readout synapse."""

    def __init__(
        self,
        source_idx: int,
        target_action: int,
        dendritic_compartment: str = "basal",
        distance_um: float = 120.0,
        delay_ms: float = 2.0,
        weight: float = 0.1,
    ):
        self.source_idx = source_idx
        self.target_action = target_action
        self.dendritic_compartment = dendritic_compartment
        self.distance_um = distance_um
        self.delay_ms = delay_ms
        self.weight = float(weight)
        self.initial_weight = float(weight)
        self.eligibility = 0.0
        self.target_signal = 0.0
        self.retrograde_signal = 0.0
        self.stdp_contribution = 0.0
        self.last_update = 0.0
        self.cumulative_update = 0.0
        self.causal_relevance = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_idx": self.source_idx,
            "target_action": self.target_action,
            "dendritic_compartment": self.dendritic_compartment,
            "distance_um": round(self.distance_um, 2),
            "delay_ms": round(self.delay_ms, 2),
            "weight": round(float(self.weight), 5),
            "initial_weight": round(float(self.initial_weight), 5),
            "eligibility": round(float(self.eligibility), 5),
            "target_signal": round(float(self.target_signal), 5),
            "retrograde_signal": round(float(self.retrograde_signal), 5),
            "stdp_contribution": round(float(self.stdp_contribution), 5),
            "last_update": round(float(self.last_update), 6),
            "cumulative_update": round(float(self.cumulative_update), 6),
            "causal_relevance": round(float(self.causal_relevance), 4),
        }


class ActionReadout:
    """Distinct, inspectable Action Readout component for Phase 15.
    
    Translates high-dimensional temporal context representation into action drives
    via dendritic synapses with target-specific plasticity and retrograde signaling.
    """

    def __init__(
        self,
        input_dim: int = 40,
        num_actions: int = 2,
        input_type: InputRepresentationType = InputRepresentationType.I6_COMBINED_TEMPORAL,
        learning_mech: LearningMechanism = LearningMechanism.D8_TEMPORAL_TARGET_DENDRITIC,
        teaching_signal_type: TeachingSignalType = TeachingSignalType.G3_DENDRITIC_TARGET_SIGNAL,
        retrograde_type: RetrogradeSignalType = RetrogradeSignalType.R2_ACTION_SPECIFIC,
        stdp_type: DendriticSTDPType = DendriticSTDPType.S4_TARGET_SPECIFIC_STDP,
        lr: float = 0.015,
        tau_eligibility_ms: float = 300.0,
        dt_ms: float = 1.0,
        w_min: float = -2.0,
        w_max: float = 2.0,
        retro_delay_steps: int = 2,
        seed: int = 42,
    ):
        self.input_dim = int(input_dim)
        self.num_actions = max(2, int(num_actions))
        self.input_type = input_type
        self.learning_mech = learning_mech
        self.teaching_signal_type = teaching_signal_type
        self.retrograde_type = retrograde_type
        self.stdp_type = stdp_type

        self.lr = float(lr)
        self.tau_eligibility_ms = float(tau_eligibility_ms)
        self.dt_ms = float(dt_ms)
        self.decay_elig = float(np.exp(-dt_ms / max(1.0, tau_eligibility_ms)))
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.retro_delay_steps = max(1, int(retro_delay_steps))
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        # Weight matrices: [input_dim, num_actions]
        limit = 1.0 / math.sqrt(max(1, self.input_dim))
        self.weights = self.rng.uniform(-limit, limit, size=(self.input_dim, self.num_actions)).astype(np.float32)
        self.initial_weights = self.weights.copy()
        self.biases = np.zeros(self.num_actions, dtype=np.float32)

        # Eligibility matrix: [input_dim, num_actions]
        self.eligibility = np.zeros((self.input_dim, self.num_actions), dtype=np.float32)
        self.dendritic_eligibility = np.zeros((self.input_dim, self.num_actions), dtype=np.float32)

        # Pre/post activity history for STDP
        self.pre_history: List[np.ndarray] = []
        self.post_history: List[np.ndarray] = []
        self.retro_history: List[np.ndarray] = []

        # Inspection synapse records
        self.synapse_records: Dict[Tuple[int, int], ReadoutSynapseRecord] = {}
        self._init_synapse_records()

        # Telemetry & Credit localization metrics
        self.last_input: np.ndarray = np.zeros(self.input_dim, dtype=np.float32)
        self.last_drives: np.ndarray = np.zeros(self.num_actions, dtype=np.float32)
        self.last_teaching_signals: np.ndarray = np.zeros(self.num_actions, dtype=np.float32)
        self.last_retrograde_signals: np.ndarray = np.zeros(self.num_actions, dtype=np.float32)
        self.last_weight_delta: np.ndarray = np.zeros_like(self.weights)

        self.credit_specificity_index: float = 0.0
        self.credit_precision: float = 0.0
        self.credit_leakage: float = 0.0
        self.gini_concentration: float = 0.0

        # State flags
        self.is_frozen: bool = (self.learning_mech == LearningMechanism.D1_FROZEN)
        self.is_lesioned: bool = False
        self.lesion_mask: np.ndarray = np.ones((self.input_dim, self.num_actions), dtype=np.float32)

        # Synaptic Consolidation Subsystem (Track B)
        self.consolidation_module = SynapticConsolidationModule(
            input_dim=self.input_dim,
            num_actions=self.num_actions,
            mode=ConsolidationMode.NONE,
            seed=seed,
        )

    def _init_synapse_records(self) -> None:
        """Create structured records for synapse inspection."""
        for j in range(self.input_dim):
            for k in range(self.num_actions):
                compartment = "apical" if (j % 2 == 0) else "basal"
                dist = 60.0 + (j % 10) * 15.0
                delay = 1.0 + (j % 5) * 0.5
                rec = ReadoutSynapseRecord(
                    source_idx=j,
                    target_action=k,
                    dendritic_compartment=compartment,
                    distance_um=dist,
                    delay_ms=delay,
                    weight=self.weights[j, k],
                )
                self.synapse_records[(j, k)] = rec

    def freeze(self, freeze_state: bool = True) -> None:
        """Freeze or unfreeze readout plasticity."""
        self.is_frozen = freeze_state

    def lesion(self, action_target: Optional[int] = None, fraction: float = 1.0) -> None:
        """Lesion action readout connections."""
        self.is_lesioned = True
        if action_target is not None and 0 <= action_target < self.num_actions:
            self.lesion_mask[:, action_target] = 0.0
        else:
            mask = (self.rng.uniform(0.0, 1.0, size=self.weights.shape) > fraction).astype(np.float32)
            self.lesion_mask *= mask

    def scramble(self) -> None:
        """Scramble weights across synapses."""
        flat = self.weights.flatten()
        self.rng.shuffle(flat)
        self.weights = flat.reshape(self.weights.shape)

    def replace_weights(self, new_weights: np.ndarray) -> None:
        """Replace readout weights with external reference (e.g. BPTT / offline solution)."""
        assert new_weights.shape == self.weights.shape, "Shape mismatch for new weights"
        self.weights = new_weights.astype(np.float32).copy()

    def reset_state(self) -> None:
        """Reset dynamic eligibility, STDP buffers, signals, and consolidation transients."""
        self.eligibility.fill(0.0)
        self.dendritic_eligibility.fill(0.0)
        self.pre_history.clear()
        self.post_history.clear()
        self.retro_history.clear()
        self.last_weight_delta.fill(0.0)
        self.consolidation_module.reset()

    def set_consolidation_mode(self, mode: Union[ConsolidationMode, str], **kwargs) -> None:
        """Configure active synaptic consolidation mechanism."""
        if isinstance(mode, str):
            mode = ConsolidationMode(mode)
        self.consolidation_module = SynapticConsolidationModule(
            input_dim=self.input_dim,
            num_actions=self.num_actions,
            mode=mode,
            seed=self.seed,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Input Selection (I0 to I6)
    # -------------------------------------------------------------------------
    def format_input_representation(self, substrate_state: Dict[str, Any]) -> np.ndarray:
        """Extract requested input representation from biological substrate."""
        typ = self.input_type

        if typ == InputRepresentationType.I0_INSTANTANEOUS_SPIKES:
            spk = substrate_state.get("spikes", np.zeros(self.input_dim))
            raw = np.asarray(spk, dtype=np.float32).flatten()
        elif typ == InputRepresentationType.I1_SPIKE_COUNT:
            cnt = substrate_state.get("spike_counts", np.zeros(self.input_dim))
            raw = np.asarray(cnt, dtype=np.float32).flatten()
        elif typ == InputRepresentationType.I2_BURST_REPRESENTATION:
            bursts = substrate_state.get("bursts", np.zeros(self.input_dim))
            raw = np.asarray(bursts, dtype=np.float32).flatten()
        elif typ == InputRepresentationType.I3_TEMPORAL_POPULATION:
            temp = substrate_state.get("temporal_population", np.zeros(self.input_dim))
            raw = np.asarray(temp, dtype=np.float32).flatten()
        elif typ == InputRepresentationType.I4_PHASE_REPRESENTATION:
            phase = substrate_state.get("phase_features", np.zeros(self.input_dim))
            raw = np.asarray(phase, dtype=np.float32).flatten()
        elif typ == InputRepresentationType.I5_LOCAL_RESONANCE:
            res = substrate_state.get("resonance_features", np.zeros(self.input_dim))
            raw = np.asarray(res, dtype=np.float32).flatten()
        else:  # I6_COMBINED_TEMPORAL (Default)
            # Structured fusion of context, bursts, temporal features, and NMDA conductance
            ctx = substrate_state.get("context_vector", np.zeros(min(20, self.input_dim)))
            nmda = substrate_state.get("g_nmda", np.zeros(min(10, self.input_dim)))
            temp = substrate_state.get("temporal_population", np.zeros(min(10, self.input_dim)))
            raw = np.concatenate([ctx.flatten(), nmda.flatten()[:10], temp.flatten()[:10]])

        # Dimension match: zero-pad or truncate to input_dim
        if len(raw) < self.input_dim:
            pad = np.zeros(self.input_dim - len(raw), dtype=np.float32)
            out = np.concatenate([raw, pad])
        else:
            out = raw[:self.input_dim]

        self.last_input = out.astype(np.float32)
        return self.last_input

    # -------------------------------------------------------------------------
    # Forward Readout Pass
    # -------------------------------------------------------------------------
    def forward(
        self,
        substrate_state: Dict[str, Any],
        v_dend: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute feedforward evidence for action populations.
        
        Args:
            substrate_state: dictionary containing upstream brain representations
            v_dend: optional dendritic compartment potentials of action neurons
            
        Returns:
            drives: [num_actions] continuous evidence driving action assemblies
        """
        x = self.format_input_representation(substrate_state)
        w_base = self.consolidation_module.compute_effective_weights(self.weights)
        w_eff = w_base * self.lesion_mask

        # Raw linear projection: u_k = sum_j (w_jk * x_j) + b_k
        raw_drives = (x @ w_eff) + self.biases

        # Dendritic modulation if provided
        if v_dend is not None and len(v_dend) == self.num_actions:
            # Active dendritic amplification: sigma((v_dend - theta) / k)
            dend_factor = 1.0 + 0.3 * np.tanh((v_dend + 50.0) / 10.0)
            drives = raw_drives * dend_factor
        else:
            drives = raw_drives

        self.last_drives = drives.astype(np.float32)
        return self.last_drives

    # -------------------------------------------------------------------------
    # Target-Specific Teaching Signal Computation (G0 to G5)
    # -------------------------------------------------------------------------
    def compute_teaching_signals(
        self,
        target_action: Optional[int],
        reward: float,
        action_probs: np.ndarray,
        v_dend: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute target-specific teaching signal vector M_k(t) for all actions k."""
        sig_type = self.teaching_signal_type
        K = self.num_actions
        m = np.zeros(K, dtype=np.float32)

        if sig_type == TeachingSignalType.G0_GLOBAL_SCALAR:
            # Global scalar: identical for all action targets
            m.fill(float(reward))

        elif sig_type == TeachingSignalType.G1_ACTION_TARGET_SCALAR:
            # Target action gets +1, non-targets get -1/(K-1)
            if target_action is not None and 0 <= target_action < K:
                m.fill(-1.0 / max(1, K - 1))
                m[target_action] = 1.0
                m *= abs(reward) if abs(reward) > 1e-6 else 1.0
            else:
                m.fill(float(reward))

        elif sig_type == TeachingSignalType.G2_POPULATION_LOCAL_SCALAR:
            # Locally centered modulation: R * (y_k - mean(y))
            mean_prob = float(np.mean(action_probs))
            m = float(reward) * (action_probs - mean_prob)

        elif sig_type == TeachingSignalType.G3_DENDRITIC_TARGET_SIGNAL:
            # Causally targeted: (y*_k - y_k) * dendritic_gate
            if target_action is not None and 0 <= target_action < K:
                y_star = np.zeros(K, dtype=np.float32)
                y_star[target_action] = 1.0
                err = y_star - action_probs
                if v_dend is not None and len(v_dend) == K:
                    dend_gate = 1.0 / (1.0 + np.exp(-(v_dend + 50.0) / 8.0))
                    m = err * dend_gate
                else:
                    m = err
            else:
                m.fill(float(reward))

        elif sig_type == TeachingSignalType.G4_SHUFFLED_TARGET:
            # Permuted target assignment (control)
            if target_action is not None and 0 <= target_action < K:
                perm = self.rng.permutation(K)
                shuffled_target = int(perm[target_action])
                m.fill(-1.0 / max(1, K - 1))
                m[shuffled_target] = 1.0
            else:
                m.fill(float(reward))

        elif sig_type == TeachingSignalType.G5_SIGN_FLIPPED:
            # Sign-flipped control (anti-learning)
            if target_action is not None and 0 <= target_action < K:
                y_star = np.zeros(K, dtype=np.float32)
                y_star[target_action] = 1.0
                m = -(y_star - action_probs)
            else:
                m.fill(-float(reward))

        self.last_teaching_signals = m.astype(np.float32)
        return self.last_teaching_signals

    # -------------------------------------------------------------------------
    # Retrograde Signaling (R0 to R4)
    # -------------------------------------------------------------------------
    def compute_retrograde_signals(
        self,
        action_activations: np.ndarray,
        v_dend: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute retrograde feedback signal from postsynaptic action compartment."""
        r_type = self.retrograde_type
        K = self.num_actions
        r = np.zeros(K, dtype=np.float32)

        if r_type == RetrogradeSignalType.R0_NONE:
            r.fill(0.0)

        elif r_type == RetrogradeSignalType.R1_SCALAR:
            # Scalar retrograde diffusion (non-specific)
            r.fill(float(np.mean(action_activations)))

        elif r_type == RetrogradeSignalType.R2_ACTION_SPECIFIC:
            # Action-compartment specific retrograde emission
            r = action_activations.copy()

        elif r_type == RetrogradeSignalType.R3_COMPARTMENT_SPECIFIC:
            # Gated by local dendritic compartment state
            if v_dend is not None and len(v_dend) == K:
                ca_gate = np.clip((v_dend + 65.0) / 25.0, 0.0, 2.0)
                r = action_activations * ca_gate
            else:
                r = action_activations.copy()

        elif r_type == RetrogradeSignalType.R4_DELAYED:
            # Delayed retrograde propagation
            self.retro_history.append(action_activations.copy())
            if len(self.retro_history) > self.retro_delay_steps:
                r = self.retro_history.pop(0)
            else:
                r = action_activations.copy()

        self.last_retrograde_signals = r.astype(np.float32)
        return self.last_retrograde_signals

    # -------------------------------------------------------------------------
    # Dendritic STDP (S0 to S4)
    # -------------------------------------------------------------------------
    def compute_stdp_update(
        self,
        pre_input: np.ndarray,
        action_probs: np.ndarray,
        teaching_signal: np.ndarray,
    ) -> np.ndarray:
        """Compute timing-dependent plasticity contributions."""
        stdp_type = self.stdp_type
        delta_stdp = np.zeros_like(self.weights)

        if stdp_type == DendriticSTDPType.S0_NONE:
            return delta_stdp

        self.pre_history.append(pre_input.copy())
        self.post_history.append(action_probs.copy())
        if len(self.pre_history) > 10:
            self.pre_history.pop(0)
            self.post_history.pop(0)

        if len(self.pre_history) < 2:
            return delta_stdp

        # Pairwise pre/post correlation
        pre_diff = self.pre_history[-1] - self.pre_history[-2]
        post_diff = self.post_history[-1] - self.post_history[-2]

        if stdp_type == DendriticSTDPType.S1_SOMATIC_PAIR:
            # Pre before post induces potentiation: x_pre * post_diff
            delta_stdp = 0.005 * np.outer(self.pre_history[-2], np.maximum(0.0, post_diff))

        elif stdp_type == DendriticSTDPType.S2_DENDRITIC_PAIR:
            # Dendritic timing: active derivative coupling
            delta_stdp = 0.005 * np.outer(pre_diff, post_diff)

        elif stdp_type == DendriticSTDPType.S3_ELIGIBILITY_STDP:
            # Modulates eligibility matrix rather than direct weights
            delta_stdp = 0.002 * (self.eligibility * np.outer(pre_diff, action_probs))

        elif stdp_type == DendriticSTDPType.S4_TARGET_SPECIFIC_STDP:
            # STDP gated by target-specific teaching signal M_k(t)
            base_stdp = 0.005 * np.outer(self.pre_history[-2], post_diff)
            delta_stdp = base_stdp * teaching_signal[np.newaxis, :]

        return delta_stdp.astype(np.float32)

    # -------------------------------------------------------------------------
    # Plasticity & Learning Step
    # -------------------------------------------------------------------------
    def step_plasticity(
        self,
        substrate_state: Dict[str, Any],
        target_action: Optional[int],
        reward: float,
        action_probs: np.ndarray,
        action_activations: np.ndarray,
        v_dend: Optional[np.ndarray] = None,
        is_choice_step: bool = False,
        is_reward_step: bool = False,
    ) -> Dict[str, Any]:
        """Apply target-specific dendritic learning step.
        
        Returns:
            telemetry: dictionary containing credit localization and weight statistics
        """
        if self.is_frozen:
            self.last_weight_delta.fill(0.0)
            return self._compute_credit_telemetry(target_action)

        x = self.format_input_representation(substrate_state)
        K = self.num_actions

        # 1. Update eligibility traces: e_jk(t)
        # e_jk = decay * e_jk + x_j * y_k
        outer_act = np.outer(x, action_probs)
        self.eligibility = self.eligibility * self.decay_elig + (1.0 - self.decay_elig) * outer_act

        # Dendritic compartment eligibility
        if v_dend is not None and len(v_dend) == K:
            dend_coupling = np.clip((v_dend + 65.0) / 25.0, 0.1, 2.0)
            self.dendritic_eligibility = self.eligibility * dend_coupling[np.newaxis, :]
        else:
            self.dendritic_eligibility = self.eligibility.copy()

        # 2. Teaching signal M_k(t)
        teaching = self.compute_teaching_signals(
            target_action=target_action,
            reward=reward,
            action_probs=action_probs,
            v_dend=v_dend,
        )

        # 3. Retrograde signal R_k(t)
        retrograde = self.compute_retrograde_signals(
            action_activations=action_activations,
            v_dend=v_dend,
        )

        # 4. Dendritic STDP
        delta_stdp = self.compute_stdp_update(
            pre_input=x,
            action_probs=action_probs,
            teaching_signal=teaching,
        )

        # 5. Core Plasticity Update based on Learning Mechanism (D1 to D8)
        mech = self.learning_mech
        delta_w = np.zeros_like(self.weights)

        if mech == LearningMechanism.D1_FROZEN:
            delta_w.fill(0.0)

        elif mech == LearningMechanism.D2_LOCAL_HEBBIAN:
            # Oja's rule: eta * (x_j * y_k - y_k^2 * w_jk)
            for k in range(K):
                y_k = action_probs[k]
                delta_w[:, k] = self.lr * (x * y_k - (y_k ** 2) * self.weights[:, k])

        elif mech == LearningMechanism.D3_ELIGIBILITY_ONLY:
            # Updates based purely on eligibility (unsupervised accumulation)
            delta_w = self.lr * 0.1 * self.eligibility

        elif mech == LearningMechanism.D4_GLOBAL_REWARD:
            # Standard three-factor: global scalar reward R * eligibility
            delta_w = self.lr * reward * self.eligibility

        elif mech == LearningMechanism.D5_TARGET_SPECIFIC_REWARD:
            # Target-specific reward modulation: e_jk * M_k(t)
            delta_w = self.lr * (self.eligibility * teaching[np.newaxis, :])

        elif mech == LearningMechanism.D6_PRED_ERROR_GATED:
            # Delta-rule prediction error gating
            if target_action is not None and 0 <= target_action < K:
                y_star = np.zeros(K, dtype=np.float32)
                y_star[target_action] = 1.0
                pe = y_star - action_probs
                delta_w = self.lr * np.outer(x, pe)
            else:
                delta_w = self.lr * (self.eligibility * teaching[np.newaxis, :])

        elif mech == LearningMechanism.D7_DENDRITIC_TEACHING:
            # Postsynaptic dendritic eligibility * target teaching signal
            # + retrograde boost
            retro_boost = 0.3 * np.outer(np.ones(self.input_dim), retrograde)
            effective_teaching = teaching[np.newaxis, :] + retro_boost
            delta_w = self.lr * (self.dendritic_eligibility * effective_teaching) + delta_stdp

        elif mech == LearningMechanism.D8_TEMPORAL_TARGET_DENDRITIC:
            # Phase 14 temporal population integration + target-specific dendritic learning
            # + retrograde signaling + dendritic STDP
            retro_term = 0.25 * np.outer(np.ones(self.input_dim), retrograde)
            effective_signal = teaching[np.newaxis, :] + retro_term
            delta_w = self.lr * (self.dendritic_eligibility * effective_signal) + delta_stdp

        # Modulate weight update via Synaptic Consolidation Subsystem
        delta_w, consol_telem = self.consolidation_module.modulate_plasticity_step(
            base_weights=self.weights,
            raw_delta_w=delta_w,
            eligibility=self.dendritic_eligibility,
            teaching_signal=teaching,
            reward=reward,
            is_reward_step=is_reward_step,
        )

        # Apply weight updates with bounds
        self.weights += delta_w * self.lesion_mask
        self.weights = np.clip(self.weights, self.w_min, self.w_max)
        self.last_weight_delta = delta_w

        # Update inspection synapse records
        self._update_synapse_records(teaching, retrograde, delta_stdp, delta_w, target_action)

        telemetry = self._compute_credit_telemetry(target_action)
        telemetry.update({f"consol_{k}": v for k, v in consol_telem.items()})
        return telemetry

    def _update_synapse_records(
        self,
        teaching: np.ndarray,
        retrograde: np.ndarray,
        delta_stdp: np.ndarray,
        delta_w: np.ndarray,
        target_action: Optional[int],
    ) -> None:
        """Update inspection synapse records for 3D Neural Observatory."""
        for (j, k), rec in self.synapse_records.items():
            rec.weight = float(self.weights[j, k])
            rec.eligibility = float(self.dendritic_eligibility[j, k])
            rec.target_signal = float(teaching[k])
            rec.retrograde_signal = float(retrograde[k])
            rec.stdp_contribution = float(delta_stdp[j, k])
            rec.last_update = float(delta_w[j, k])
            rec.cumulative_update += float(delta_w[j, k])
            if target_action is not None:
                rec.causal_relevance = 1.0 if (k == target_action) else -0.5
            else:
                rec.causal_relevance = 0.0

    def _compute_credit_telemetry(self, target_action: Optional[int]) -> Dict[str, Any]:
        """Compute quantitative credit localization metrics."""
        delta = self.last_weight_delta
        abs_delta = np.abs(delta)
        total_abs_update = float(np.sum(abs_delta))

        if total_abs_update > 1e-9 and target_action is not None and 0 <= target_action < self.num_actions:
            target_update = float(np.sum(abs_delta[:, target_action]))
            other_update = total_abs_update - target_update

            # CSI: Credit Specificity Index = Target / Total
            self.credit_specificity_index = float(np.clip(target_update / total_abs_update, 0.0, 1.0))
            # Credit Leakage = Non-target / Total = 1 - CSI
            self.credit_leakage = float(np.clip(other_update / total_abs_update, 0.0, 1.0))

            # Credit Precision: fraction of target weights updated in correct direction
            correct_sign = np.sum(delta[:, target_action] > 0.0)
            self.credit_precision = float(correct_sign / max(1, self.input_dim))

            # Gini concentration of weight updates
            flat_abs = np.sort(abs_delta.flatten())
            n = len(flat_abs)
            idx = np.arange(1, n + 1)
            sum_flat = np.sum(flat_abs)
            if sum_flat > 1e-9:
                self.gini_concentration = float((2.0 * np.sum(idx * flat_abs)) / (n * sum_flat) - (n + 1) / n)
            else:
                self.gini_concentration = 0.0
        else:
            self.credit_specificity_index = 0.5
            self.credit_precision = 0.5
            self.credit_leakage = 0.5
            self.gini_concentration = 0.0

        return {
            "credit_specificity_index": round(self.credit_specificity_index, 4),
            "credit_precision": round(self.credit_precision, 4),
            "credit_leakage": round(self.credit_leakage, 4),
            "gini_concentration": round(self.gini_concentration, 4),
            "mean_weight": round(float(np.mean(self.weights)), 4),
            "weight_l2_norm": round(float(np.linalg.norm(self.weights)), 4),
            "max_abs_update": round(float(np.max(abs_delta)), 6),
        }
