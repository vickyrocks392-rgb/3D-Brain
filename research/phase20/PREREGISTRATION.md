# Phase 20 Preregistration: Context-Gated Modular Routing & Sparse Hippocampal-Style Replay

**Document ID:** PREREG-PHASE20-001  
**Date:** September 30, 2026  
**Auditor / Lead Investigator:** Autonomous Scientific Research Agent  
**Parent Foundation:** Phase 19 Scientific Report (`research/phase19/PHASE_19_FINAL_REPORT.md`) & Phase 18 Audit  
**Repository:** `~/Documents/3D-Brain`  
**GitHub:** `https://github.com/vickyrocks392-rgb/3D-Brain.git`

---

## 1. Context and Scientific Motivation

Phase 19 established two decisive scientific findings regarding the minimal biophysical architecture (MIN_E):

1. **Finite Empirical Credit Horizon (~240–400 ms):**
   Under strict un-leaked evaluation, MIN_E achieved:
   - 0 ms: 50.0%
   - 40 ms: 50.0%
   - 80 ms: 50.0%
   - 120 ms: 60.0%
   - 240 ms: 80.0%
   - 480 ms: 51.3% (chance)
   - 1,000 ms to 5,000 ms: 50.0% (chance)
   The empirical credit horizon is governed strictly by the biophysical interaction of NMDA conductance decay ($\tau_{\text{nmda}}$) and dendritic eligibility decay ($\tau_{\text{elig}}$). The earlier headline claim of multi-second local credit assignment was definitively falsified.

2. **Linear Readout Interference & Representational Overlap:**
   While synaptic consolidation modules (Dual-Timescale, Tagging & Capture, Metaplasticity) substantially mitigated catastrophic forgetting during sequential tasks (retention ratios 0.70–1.00), conflicting tasks using identical sensory subspaces compete directly for the same linear readout parameters. In complex environments with nonlinear XOR-like task rules or multiple overlapping task contexts, a single monolithic linear readout matrix inevitably experiences representational interference.

Phase 20 investigates two distinct, biologically inspired computational mechanisms to address these bottlenecks without resorting to:
- Continuous target supervision,
- Backpropagation through time (BPTT),
- Global parameter updates, or
- Arbitrary global capacity expansion.

### Mechanism A: Context-Gated Modular Routing
In the mammalian brain, prefrontal and striatal circuits gate representations into modular cortico-striatal loops or sub-networks depending on task context. We implement modular readout branches coordinated by a context gate. By restricting forward readout and backward plasticity to task-relevant branches, conflicting mappings can be compartmentalized without global network growth.

### Mechanism B: Sparse Hippocampal-Style Replay
The biological hippocampus rapidly encodes episodic traces and replays them during quiescent rest or sharp-wave ripples (SWRs). We implement an explicit, biologically inspired episodic replay buffer that stores sparse event/state traces legitimately available during active experience. Replay occurs *only* during explicit quiescent REST periods (with zero sensory inputs, zero external targets, and zero continuous supervision) to bridge temporal credit assignment across multi-second behavioral delays.

> **Biological Interpretation Rule:**
> "Hippocampal-style" refers to computational principles inspired by sparse event-triggered episodic replay during offline rest. It is NOT presented as evidence that this implementation reproduces the biological hippocampus. We strictly separate computational results from biological interpretations.

---

## 2. Core Hypotheses

- **H1 (Routing Interference Reduction):** Context-gated modular routing significantly reduces cross-task interference on conflicting mappings ($A \to B \to A$) compared to a shared linear readout ($p < 0.01$).
- **H2 (Branch-Specific Plasticity & Sub-linear Scaling):** Context-gated branch-specific plasticity enables conflicting task mappings (including nonlinear XOR conditions) to coexist with minimal parameter overhead ($\le 1.5\times$ base readout parameters) without requiring proportional global capacity expansion.
- **H3 (Replay-Driven Credit Extension):** Sparse event-triggered replay during offline rest extends effective credit assignment beyond the passive eligibility horizon ($\tau_{\text{passive}} \approx 240\text{–}480\text{ ms}$) to delays $\ge 2,000\text{ ms}$, achieving test accuracy $\ge 70.0\%$ under strict frozen, un-leaked evaluation.
- **H4 (Zero Continuous Supervision):** Replay bridges multi-second behavioral delays without continuous target supervision, test-time plasticity, or target leakage during delay intervals.
- **H5 (Synergistic Integration):** Combining modular routing, sparse replay, and synaptic consolidation (C4) yields superior continual-task retention across multi-task cycles ($A \to B \to C \to A$) compared to any mechanism in isolation.
- **H6 (Causal Specificity):** Performance improvements depend causally on veridical replay content and timing, and collapse under temporal randomization, content shuffling, or wrong-episode replay ($p < 0.01$).

---

## 3. Non-Negotiable Methodological Rules

1. **Zero Target Leakage:** Targets must NEVER be provided during sensory cue, delay, choice, or rest intervals. Target feedback occurs solely at causal reward/feedback delivery steps.
2. **Strict Frozen Evaluation:** All test evaluation must be performed with `is_frozen = True`. No synaptic weights, eligibility traces, or consolidation states may be updated during evaluation.
3. **No Hidden State or Task-ID Cheating:** The routing gate must NEVER receive the ground-truth task label unless explicitly evaluated as an oracle upper-bound control (`ORACLE_GATED`). Routing must be driven purely by legitimately available sensory context ($x$) or persistent latent state ($g_{\text{nmda}}$).
4. **Legitimate Replay Memory:** The replay buffer may store ONLY states legitimately available prior to or at the time of choice/reward (e.g. sensory context representation, choice timestamp, chosen action, sparse latent state, eligibility snapshot). Future targets or unobserved rewards cannot be retroactively injected into pre-reward traces.
5. **Offline Quiescent Rest:** Replay occurs exclusively during explicit REST periods where sensory input is zero, reward is zero, and no external supervision exists.
6. **Reproducibility Across Deterministic Seeds:** All experiments must be conducted across the identical 5 seeds used in Phase 19: `[42, 101, 2024, 777, 999]`.
7. **Negative Results Policy:** If a mechanism fails to produce above-chance performance or fails causal ablations, the failure must be reported directly without modifying hypotheses or adjusting metrics retroactively.

---

## 4. Track A: Context-Gated Modular Routing Specifications

### 4.1 Routing Architectures
- **A0 — Shared Linear Readout (MIN_E Baseline):** Single plastic weight matrix $W \in \mathbb{R}^{D_{\text{in}} \times K_{\text{act}}}$.
- **A1 — Fixed Modular Branches:** $M$ dedicated downstream linear branches ($M=3$, each $\mathbb{R}^{D_{\text{in}} \times K_{\text{act}}}$) statically partitioned without dynamic gating.
- **A2 — Context-Gated Modular Branches (Soft Gating):** Dynamic gate $g = \text{softmax}(W_g x) \in \mathbb{R}^M$ linearly mixtures branch drives: $u = \sum_m g_m u^{(m)}$.
- **A3 — Learned Sparse Context Gate:** Top-1 competitive hard/sparse gate $g = \text{one\_hot}(\text{argmax}(W_g x))$, where gate weights $W_g$ are updated via competitive local Hebbian/Oja plasticity.
- **A4 — Context-Gated Branch-Specific Plasticity:** Learned sparse gate where BOTH feedforward readout AND synaptic plasticity are restricted to the routed branch:
  $$\Delta W^{(m)} = g_m \cdot \eta \cdot (e^{(m)} \odot M_k)$$
  Non-routed branches receive zero weight updates ($\Delta W^{(j)} = 0$ for $j \neq m$), guaranteeing structural insulation against interference.
- **A5 — Oracle-Gated Control (Upper Bound):** Ground-truth task identifier routes directly to dedicated branch.

### 4.2 Tasks & Protocols
1. **Task A:** Standard mapping (Context 0 $\to$ Action 0, Context 1 $\to$ Action 1).
2. **Task B:** Conflicting inverted mapping (Context 0 $\to$ Action 1, Context 1 $\to$ Action 0).
3. **Task C:** Orthogonal mapping (distinct sensory context channels).
4. **Task XOR:** Nonlinear XOR mapping combining context bits $(c_1, c_2) \in \{(0,0), (0,1), (1,0), (1,1)\} \to \{0, 1\}$, unsolvable by a single linear hyperplane.
5. **Sequences:** $A \to B \to A$ (Reversal), $A \to B \to C$ (Continual), $A \to B \to C \to A$ (Multi-task Cycle).

---

## 5. Track B: Sparse Hippocampal-Style Replay Specifications

### 5.1 Replay Architecture & Buffer
- **Episodic Buffer:** Capacity $N_{\text{mem}} = 64$ episode traces. Each trace contains:
  - Episode ID,
  - Sensory/Latent context vector $x \in \mathbb{R}^{D_{\text{in}}}$,
  - Chosen action $a \in \{0, \dots, K-1\}$,
  - Choice timestamp / delay interval,
  - Dendritic eligibility snapshot $e \in \mathbb{R}^{D_{\text{in}} \times K_{\text{act}}}$,
  - Observed reward outcome $R \in \{-0.5, 1.0\}$ and prediction error $\delta$.
- **Replay Scheduling:** Occurs during explicit post-trial REST phases (duration $T_{\text{rest}} = 10\text{ steps} = 400\text{ ms}$). Replay budget: $K_{\text{replay}} = 4$ events per rest phase.

### 5.2 Replay Conditions
- **B0 — No Replay:** Baseline MIN_E resting silently.
- **B1 — Random Replay:** Uniform random sampling from memory buffer during rest.
- **B2 — Event-Triggered Replay:** Replay triggered only on high-surprise or salient reward events ($|\delta| > \theta_{\text{surprise}}$).
- **B3 — Prioritized Replay:** Sampling probability proportional to surprise/salience: $P(i) \propto |\delta_i|^\alpha / \sum_j |\delta_j|^\alpha$.
- **B4 — Event-Triggered Replay + Consolidation:** Replay events trigger dual-timescale synaptic transfer into slow/protected weights.
- **B5 — Prioritized Replay + Consolidation:** Prioritized replay coupled directly with local synaptic consolidation.

### 5.3 Extended Delay Horizons
Delays tested:
- **Short-Delay Controls:** 0 ms, 40 ms, 80 ms, 120 ms, 240 ms, 480 ms
- **Long-Delay Horizons:** 500 ms, 1,000 ms, 2,000 ms, 5,000 ms, 10,000 ms, 30,000 ms, 60,000 ms

---

## 6. Track C: Integrated Architecture Specifications

### 6.1 System Configurations
- **C0 — Baseline MIN_E:** Single shared readout, no replay, no consolidation.
- **C1 — Best Routing Only:** A4 (Learned sparse context gate + branch-specific plasticity).
- **C2 — Best Replay Only:** B5 (Prioritized replay + consolidation).
- **C3 — Routing + Replay:** A4 + B3 (Context-gated routing + prioritized replay).
- **C4 — Full Integrated System:** A4 + B5 (Context-gated routing + prioritized replay + dual-timescale consolidation).

### 6.2 External Baselines
Matched parameter and compute baselines from Phase 19:
- `DiscreteLeakyRNN` (Simple Recurrent)
- `BPTTGradientBaseline` (Recurrent BPTT)
- `SmallCausalTransformer` (Causal Self-Attention, matched ~2K parameters)

---

## 7. Causal Ablation Matrix

1. **Routing Ablations:**
   - Gate Removed (fallback to uniform average across branches)
   - Gate Randomized (routing to random branch on each step)
   - Branch Identity Shuffled (scrambled branch mapping)
   - Cross-Branch Coupling Restored (leakage of plastic updates into all branches)
   - Branch-Specific Plasticity Disabled (global update across all branches)
2. **Replay Ablations:**
   - Replay Removed (zero replay during rest)
   - Replay Timing Randomized (replay injected during active cue/choice steps instead of rest)
   - Replay Content Randomized (Gaussian noise injected into replayed state)
   - Replay Wrong Episode (replaying traces from an unrelated conflicting task)
   - Replay Budget Ablation ($K_{\text{replay}} \in \{0, 1, 2, 4, 8, 16\}$)
3. **Integrated System Ablations:**
   - Remove Routing (keep Replay + Consolidation)
   - Remove Replay (keep Routing + Consolidation)
   - Remove Consolidation (keep Routing + Replay)
   - Remove Both Routing and Replay (Consolidation only)
   - Lesion Full System to Baseline C0

---

## 8. Statistical Standards & Acceptance Thresholds

- **Primary Statistical Metric:** Held-out frozen evaluation accuracy across 5 independent seeds (`42, 101, 2024, 777, 999`), reported as Mean $\pm$ Standard Deviation.
- **Hypothesis Classification Categories:**
  - `SUPPORTED`: Statistically significant improvement ($p < 0.01$, effect size Cohen's $d > 0.8$) surviving all causal ablations.
  - `PARTIALLY SUPPORTED`: Improvement observed on a subset of delays/tasks, or minor degradation under secondary ablations.
  - `NOT SUPPORTED`: Performance indistinguishable from baseline ($p \ge 0.05$).
  - `FALSIFIED`: Performance significantly below baseline or hypothesized claim contradicted.
  - `INCONCLUSIVE`: High seed variance ($\text{SD} > 25\%$) preventing statistically definitive inference.
