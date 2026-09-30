# 3D-Brain Phase 18 — Comprehensive Scientific Audit & Results Report

**Date:** September 30, 2026  
**Auditor:** Autonomous Scientific Audit Harness (Phase 18 Protocol)  
**Architecture Under Audit:** MIN_E (Minimal Sufficient Brain: 42 neurons, 851 synapses, 8 active mechanisms, 4 pruned)  
**Primary Reference Baseline:** Phase 17 Git Commit `3be96fc`  
**Numerical Data Artifact:** [`experiments/results/PHASE_18_AUDIT_RESULTS.json`](file:///Users/kalachanddash/Documents/3D-Brain/experiments/results/PHASE_18_AUDIT_RESULTS.json)

---

### Executive Scientific Verdict

Phase 18 was an adversarial scientific audit designed not to defend Phase 17, but to aggressively attempt to falsify its conclusions. Every surprising result was subjected to independent deconstruction, un-leaked evaluation, parameter accounting, and matched-budget control comparison.

**The decisive scientific conclusion of Phase 18 is:**
> While MIN_E's core biophysical coupling—**persistent NMDA conductances paired with slow action assembly integration**—is a genuine and robust biological mechanism for short-to-medium horizon retention (up to ~600 ms) without backpropagation, **the strongest Phase 17 headline claims were artifacts of evaluation leakage, benchmark accounting failures, and metric distortion.**

Specifically:
1. The **5,000 ms / 100% accuracy** result was caused by evaluation leakage (passing target labels during delay steps).
2. The **100K scaling benchmark (215.98B syn-ops/sec)** was a synthetic extrapolation artifact caused by clamping the simulated matrix to $120 \times 120$ while scaling operation counts linearly.
3. The **0% catastrophic forgetting** claim was metrically invalid (the task was never learned above 50% chance initially).
4. The **76.7% compositional generalization** claim relied on test-time plastic updates and an explicit shortcut channel ($x[4 + \text{comb\_id}] = 0.4$).
5. The **distractor robustness** claim described performance that was at or below the 50% chance baseline.

---

### 1. Master Classification Matrix of Phase 17 Claims

| # | Phase 17 Headline Claim | Phase 17 Reported Metric | Phase 18 Audited Metric | Pre-Registered Classification |
|:---:|:---|:---:|:---:|:---:|
| **1** | **Temporal Extrapolation (5,000 ms)** | 100.0% accuracy @ 5,000 ms | **42.7% ± 1.9%** (Chance) | **IMPLEMENTATION ARTIFACT** |
| **2** | **100K Scaling Throughput** | 107,989 steps/s, 215.98B syn-ops/s | **220.9 steps/s, 441.8M syn-ops/s** | **IMPLEMENTATION ARTIFACT** |
| **3** | **0% Catastrophic Forgetting** | 0.0% loss, retention = 1.00 | **Retention = 0.20** (when learned) | **METRICALLY INVALID** |
| **4** | **Compositional Novelty (XOR)** | 76.7% accuracy on novel (1,1) | **33.3%** (Frozen, no shortcut) | **NOT SUPPORTED** |
| **5** | **Distractor Robustness** | Graceful degradation (44.0%) | **36.0% – 44.0%** ($\le$ 50% Chance) | **METRICALLY INVALID** |
| **6** | **Parameter Compactness** | "80 parameters" | **82 plastic + 1,920 fixed + 282 state** | **PARTIALLY REPRODUCED** |
| **7** | **In-Sample Mutual Information** | Stage 6/7 MI $> 0.40$ bits | **50.0% CV Decodability** (Null) | **PARTIALLY REPRODUCED** |
| **8** | **NMDA Necessity** | Lesion collapses delayed memory | **Collapses to 36.0%** ($p < 0.01$) | **ROBUSTLY REPRODUCED** |
| **9** | **Dendritic Gating Necessity** | Gating required for local credit | **Collapses to 32.0%** across tasks | **ROBUSTLY REPRODUCED** |
| **10** | **Pruned Mechanisms Harmful** | STDP (D) and Retrograde (E) harmful | **Confirmed harmful** ($41.3\%$ acc) | **ROBUSTLY REPRODUCED** |
| **11** | **Task Generality** | Single substrate across 6 tasks | **50.7% Mean Zero-Shot** | **PARTIALLY SUPPORTED** |

---

### 2. Explicit Answers to Pre-Registered Questions

#### A. Is the 5000 ms / 100% result genuine?
**NO. It is an implementation artifact.**  
In the Phase 17 evaluation loop, `st.target_action` was passed into `model.step()` on every step of the trial, including the 125 delay steps. Because `G1_ACTION_TARGET_SCALAR` emitted $M_k(t) = [+1, -1]$ whenever reward was zero, the plastic readout weights were trained with the true answer throughout the delay interval. Under un-leaked evaluation where target actions are delivered only at the reward step, accuracy at 5,000 ms is **42.7% ± 1.9%** (chance level).

#### B. Is the 100K throughput result genuine under corrected accounting?
**NO. It is a benchmark accounting failure.**  
The benchmark script contained an explicit clamp: `dim = min(N, 120)`. It simulated only a 120-neuron matrix, and multiplied the measured loop speed by $(N \times 20)$. True execution of an actual sparse $100\text{K}$-neuron state vector in hardware yields **220.9 steps/sec** and **441.8 million synaptic operations/sec** (a 488× overstatement in operations and a 488× overstatement in steps/sec).

#### C. Is 0% catastrophic forgetting genuine?
**NO. It is metrically invalid.**  
Initial Task A accuracy was 50.0% (chance) and retest accuracy was 50.0% (chance). Calculating retention as $0.50 / 0.50 = 1.00$ obscured the fact that Task A was never learned in that sequence. When Task A is actually learned to high proficiency (83.3%), subsequent training on an inverted mapping (Task B) overwrites the plastic weights, causing retest performance to drop to **16.7%** (retention = 0.20, showing severe catastrophic forgetting).

#### D. Is compositional generalization genuine?
**NO. It is not supported.**  
Phase 17's reported 76.7% novel accuracy on $(1,1)$ required online plastic updates during test trials, and was aided by an explicit input shortcut channel ($x[4 + \text{comb\_id}] = 0.4$). Under frozen zero-shot testing without the shortcut channel, accuracy on the novel combination collapses to **33.3%** (substantially below chance). MIN_E cannot solve non-linear XOR composition zero-shot.

#### E. Is distractor robustness genuine?
**NO. It is metrically invalid.**  
All tested distractor conditions resulted in accuracies between 36.0% and 44.0%, which are at or below the 50% chance baseline for binary decisions. Labeling sub-chance performance as "robust" because it did not decline further is scientifically invalid.

#### F. Is MIN_E genuinely task-general?
**PARTIALLY.**  
The MIN_E architecture can run across diverse task environments without runtime crash or code restructuring, but its frozen zero-shot accuracy averages **50.7%** across the 6 task families. Without task-specific tuning or extensive adaptation, it does not generalize zero-shot.

#### G. Which mechanisms are universally necessary?
Under un-leaked multi-task evaluation:
- `C_DENDRITIC_ELIGIBILITY` and `B_DENDRITIC_TEACHING` are universally necessary for local credit assignment without global backpropagation.
- `K_PERSISTENT_LATENT_NMDA` is universally necessary for any delayed contextual retention beyond instantaneous sensory input.

#### H. Which mechanisms are task-dependent?
- `H_SLOW_ACTION_ASSEMBLY`: Beneficial for filtering noise on multi-hundred-millisecond horizons, but introduces inertia/hysteresis on fast-switching tasks.
- `I_MUTUAL_INHIBITION`: Essential when lateral competition is required to prevent dual-action co-firing, but redundant on simple single-action tasks.
- `A_TARGET_SPECIFIC_TEACHING` & `L_TARGET_ACTION_ROUTING`: Crucial for asymmetric credit tasks, but redundant on symmetric sequence prediction.

#### I. Which Phase 17 claims must be downgraded?
1. **Downgraded from 100% to Chance (~43%):** 5,000 ms temporal extrapolation. True memory retention horizon is bounded at $\approx 400\text{–}600\text{ ms}$.
2. **Downgraded from 215.98B to 441.8M syn-ops/s:** 100K scaling throughput.
3. **Downgraded from 0% to Severe Forgetting (Retention = 0.20):** Continual learning stability on a single linear readout.
4. **Downgraded from 76.7% to 33.3% (Below Chance):** Compositional generalization on XOR context rules.
5. **Downgraded from "Robust" to "Sub-Chance Performance":** Distractor robustness.
6. **Downgraded from "80 parameters" to "2,018 total parameters / 282 state variables":** Architectural complexity accounting.

#### J. What remains scientifically established after the audit?
1. **Persistent NMDA Attractors are Biophysically Real:** Slow conductances ($g_{\text{nmda}}$) maintain significant context information ($76.0\%$ held-out CV accuracy, $p < 0.01$ vs null) across hundreds of milliseconds without external input.
2. **Local Dendritic Credit Assignment Works:** The combination of dendritic eligibility traces ($C$) and compartmentalized teaching ($B$) solves delayed credit assignment on short-to-medium delays without backward error gradients.
3. **Pruned Mechanisms Remain Harmful:** Dendritic STDP ($D$) and diffuse retrograde signaling ($E$) actively degrade performance through synaptic drift and credit leakage.
4. **Deterministic Reproducibility is 100%:** All code, architectures, and random seed behaviors reproduce perfectly from a clean checkout.

---

### 3. Recommendations for Future Research (Phase 19+)

1. **Mandate Un-Leaked Evaluation Protocol:** Ground truth action targets must NEVER be passed to `model.step()` during observation or delay steps. Target actions may only be exposed during designated reward delivery steps.
2. **Mandate Frozen Test Regimes:** Generalization, transfer, and novel combination benchmarks must freeze synaptic plasticity during evaluation.
3. **Mandate Hardware-Measured Benchmarks:** Throughput must measure actual executed operations on allocated data structures, never theoretical extrapolations from clamped proxies (`min(N, 120)`).
4. **Mandate Matched-Budget Controls:** Every model claim must be compared against simple baselines (leaky integrators, finite-state memory, majority predictors, tiny RNNs) to isolate genuine neural contributions.
5. **Preserve MIN_E as Audited Reference:** Do not add mechanisms to artificially patch these weaknesses. The boundaries of MIN_E are now rigorously mapped and scientifically grounded.
