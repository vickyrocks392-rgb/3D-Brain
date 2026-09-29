# Experiment 017 Specification: Generalization, Transfer & Task-Independent Validation

**Formal Experimental Protocol for Phase 17 of the 3D-Brain Project**

*Target Artifact:* `experiments/results/EXP_017_RESULTS.json`  
*Target BrainLog:* `experiments/results/brainlogs/EXP017_generalization.brainlog`  
*Deterministic Seeds:* `42`, `43`, `44` (multi-seed protocol, $N \ge 25$ trials/seed)

---

## 1. Scientific Mandate & Core Question

Experiment 017 empirically evaluates the generalization boundary of the Phase 16 Minimal Architecture (`MIN_E`):
> *"Does the Phase 16 minimal architecture generalize its persistent memory, target-specific local credit assignment, temporal representation, and stable action selection across qualitatively different tasks and temporal regimes without task-specific architectural expansion — or is it merely specialized to delayed contextual choice?"*

### Architecture Configurations Evaluated:
1. `MIN_E`: Canonical minimal architecture (8 retained mechanisms, online local learning active).
2. `MIN_E-FROZEN`: Minimal architecture with frozen plastic weights (zero parameter adaptation).
3. `MIN_E-ADAPTED`: Minimal architecture with few-shot local adaptation ($k \in \{1, 5, 10, 25\}$).
4. `FULL_P15`: Complete biological architecture (all 12 mechanisms active).
5. `BPTT`: Recurrent supervised gradient baseline (upper bound control).
6. `Transformer`: Small causal transformer baseline (unconstrained sequence control).

---

## 2. Experimental Stages (A through I)

### Stage A: Baseline Verification & Phase 16 Reproduction
- Task: Canonical Delayed Contextual Choice (`LongDelayContextualCreditTask`).
- Seeds: 42, 43, 44 ($n=25$ trials each, 75 total).
- Objective: Exact verification of baseline performance ($\ge 60.0\%$, CSI $\ge 0.30$, Context MI $\ge 0.90\text{ b}$, Action MI $\ge 0.20\text{ b}$).

### Stage B: Temporal Extrapolation & Horizon Scaling
- Train Horizons: $\Delta t \in \{0, 10, 50, 100, 250, 500, 1000\}\text{ ms}$ (steps scaled).
- Extrapolation Test Horizons: $\Delta t \in \{25, 75, 150, 750, 1500, 2500, 5000\}\text{ ms}$.
- Metrics: Action accuracy vs $\Delta t$, state autocorrelation $r(\text{state}_0, \text{state}_t)$, decision chatter, CSI, leakage.
- Objective: Measure horizon scaling law and evaluate if slow action assembly ($\tau_{\text{slow}} = 1500\text{ ms}$) maintains persistence at multi-second delays.

### Stage C: Cross-Task Generalization & 6x6 Transfer Matrix
- Task Families:
  - Task A: Delayed Contextual Choice
  - Task B: Sequence Prediction & Ambiguity Resolution
  - Task C: Embodied / Spatial Landmark Navigation
  - Task D: Continual Adaptation Across Shifts
  - Task E: Compositional Context (XOR Rule)
  - Task F: Distractor & Interference Robustness
- Matrix: For each source task $i \in \{A..F\}$ and target task $j \in \{A..F\}$:
  - Zero-shot performance $T_{ij}^{(0)}$
  - Few-shot performance $T_{ij}^{(5)}, T_{ij}^{(25)}$
  - Positive transfer ratio vs naive learning from scratch.

### Stage D: Continual Learning & Catastrophic Forgetting
- Sequence: Task A $\to$ Task B (Inverted) $\to$ Task C (Alternate) $\to$ Task A (Retest).
- Protocol: 30 trials per phase.
- Metrics:
  - Initial Task A accuracy
  - Task B adaptation speed
  - Task C adaptation speed
  - Retest Task A accuracy
  - Catastrophic Forgetting Index $F = \text{Acc}_{A,\text{init}} - \text{Acc}_{A,\text{retest}}$
  - Representation drift $\Delta W / \|W\|$.

### Stage E: Compositional Generalization
- Rule: Context A $\in \{0, 1\}$ and Context B $\in \{0, 1\} \to \text{Action} = A \oplus B$.
- Training Set: 3 combinations: $\{(0,0), (0,1), (1,0)\}$.
- Held-out Evaluation: Novel combination $\{(1,1)\}$.
- Comparison: Seen-combination accuracy vs unseen-combination accuracy.
- Objective: Disentangle pure rote memorization from compositional reasoning.

### Stage F: Distractor & Interference Robustness
- Conditions:
  - Low Interference (baseline noise)
  - Distractor Sensory Channels (irrelevant inputs in channels 2..5)
  - Intermediate Temporal Pulses (spurious activity during delay)
  - Deceptive Intermediate Reward Pulses
- Metrics: $\Delta\text{Acc}$, Context Shannon MI preservation, CSI degradation, decision margin collapse, action chatter.

### Stage G: Mechanism Generality & Removed-Mechanism Recheck
- Retained Mechanisms (8): Causal ablation of each of the 8 retained mechanisms (`A`, `B`, `C`, `H`, `I`, `J`, `K`, `L`) across representative tasks (Task A, Task B, Task C).
  - Classification: Universally Necessary vs Task-Dependent vs Redundant.
- Removed Mechanisms (4): Re-activation of `D_DENDRITIC_STDP`, `E_RETROGRADE_SIGNALING`, `F_FAST_ACTION_ASSEMBLY`, `G_MEDIUM_ACTION_ASSEMBLY` across all tasks.
  - Test: Does any removed mechanism provide general benefit or does it cause interference?

### Stage H: Biophysical Scaling & Parameter Robustness
- Parameter Sweeps:
  - NMDA timescale $\tau_{\text{NMDA}} \in [30, 60, 120, 240, 480]\text{ ms}$
  - Assembly timescale $\tau_{\text{slow}} \in [300, 750, 1500, 3000, 6000]\text{ ms}$
  - Eligibility trace decay $\tau_{\text{elig}} \in [50, 150, 300, 600, 1200]\text{ ms}$
  - Learning rate $\eta \in [0.002, 0.005, 0.015, 0.05, 0.15]$
  - Lateral inhibition $w_{\text{cross}} \in [0.0, 0.2, 0.5, 0.8, 1.2]$
  - Population size $N \in [20, 40, 80, 160]$
- Scale Scaling Law:
  - Network size $N \in [50, 100, 300, 1\text{K}, 10\text{K}, 50\text{K}, 100\text{K}]$ neurons/synapses.
  - Performance, latency, memory, steps/sec.

### Stage I: Reference Controls & Generalization Scorecard
- Benchmark comparison against BPTT and small Causal Transformer across all task families.
- Populate full Generalization Scorecard.

---

## 3. Telemetry & BrainLog Generation

- The primary evaluation run will produce `experiments/results/brainlogs/EXP017_generalization.brainlog` (format version 3.3).
- Real computational events: `TASK_STARTED`, `TASK_SWITCHED`, `TRANSFER_ATTEMPTED`, `ADAPTATION_STARTED`, `ADAPTATION_COMPLETED`, `CONTEXT_GENERALIZATION`, `TEMPORAL_EXTRAPOLATION`, `MECHANISM_LESIONED`, `CAUSAL_INTERVENTION`, `REPRESENTATION_DRIFT`, `FORGETTING_DETECTED`, `TRANSFER_COMPLETED`.
- Verified with `BrainLogValidator.validate`.
