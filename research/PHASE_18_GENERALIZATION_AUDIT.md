# Phase 18 — Generalization & Transfer Audit Report

**Audit Target:** Generalization, Cross-Task Transfer, Extrapolation & Robustness  
**Date:** September 30, 2026  
**Status:** **AUDITED — FALSIFICATIONS, LEAKAGE IDENTIFIED & SURVIVING PROPERTIES ISOLATED**

---

### Executive Summary

Phase 17 tested the Minimal Sufficient Brain (MIN_E) across 6 task families and concluded:
1. Robust temporal extrapolation up to 5,000 ms with 100% accuracy.
2. 0% catastrophic forgetting (retention = 1.00) under sequential distribution shifts.
3. Genuine compositional generalization (76.7% accuracy on novel held-out XOR combination).
4. Distractor robustness under high sensory interference (44.0% accuracy).

Our Phase 18 independent audit subjected every generalization claim to rigorous falsification testing:
1. **5,000 ms Temporal Extrapolation:** **FALSIFIED.** The 100% accuracy was an evaluation leakage artifact where `st.target_action` was passed during delay steps. Under un-leaked evaluation, 5000 ms accuracy drops to **42.7% ± 1.9%** (chance level).
2. **0% Catastrophic Forgetting:** **METRICALLY INVALID.** In Phase 17, initial Task A accuracy was 50.0% and retest was 50.0% (neither was learned above chance). When Task A is actually learned (83.3%), subsequent training on Task B and C causes accuracy to collapse to **16.7%** (retention ratio = 0.20), revealing severe catastrophic forgetting in the plastic readout.
3. **Compositional Novelty:** **FALSIFIED.** In Phase 17, evaluation weights were not frozen and the task injected an explicit combination shortcut channel `x[4 + comb_id] = 0.4`. When tested under genuine frozen zero-shot conditions without the shortcut, accuracy on the held-out $(1,1)$ combination drops to **33.3%** (substantially below chance).
4. **Distractor Robustness:** **METRICALLY INVALID.** All evaluated noise conditions yielded accuracies between 36.0% and 44.0%, which are at or below the 50% chance baseline for binary choice.
5. **Architectural Generalization Without Tuning:** A single frozen MIN_E substrate averages **54.0% zero-shot accuracy** across the 6 task families. Without task-specific tuning, MIN_E operates near baseline chance.

---

### 1. Detailed Audit by Generalization Dimension

#### 1.1 Temporal Horizon Extrapolation (Audit #1)
- **Phase 17 Reported:** $0\text{ ms} \to 37.3\%$; $1000\text{ ms} \to 48.0\%$; $3000\text{ ms} \to 85.3\%$; $5000\text{ ms} \to \mathbf{100.0\%}$.
- **Mechanism of Failure (Leakage):**
  In `brain/phase17_suite.py` line 638, `st.target_action` was passed into `model.step()` on every step of the trial. In `brain/action_readout.py` line 364:
  `m *= abs(reward) if abs(reward) > 1e-6 else 1.0`
  When reward was 0.0 during delay steps, the teaching signal emitted $M = [+1, -1]$, training the readout weights continuously for 125 delay steps before the decision was made.
- **Un-Leaked Reality:**
  - $0\text{ ms}$: **46.7% ± 1.9%**
  - $1000\text{ ms}$: **37.3% ± 6.8%**
  - $3000\text{ ms}$: **44.0% ± 0.0%**
  - $5000\text{ ms}$: **42.7% ± 1.9%**
  - **Disjoint Delay Extrapolation (Train: [0, 400 ms], Test: [1000, 5000 ms]):** **48.0% ± 2.0%** (exact chance level).
- **Verdict:** **IMPLEMENTATION ARTIFACT / NOT SUPPORTED**.

---

#### 1.2 Continual Learning & Catastrophic Forgetting (Audit #4)
- **Phase 17 Reported:** Initial Task A = 50.0%, Retest Task A = 50.0%, Forgetting Index = 0.0%, Retention = 1.00.
- **Audit Analysis:**
  Reporting 100% retention on a task that was never learned above 50% chance is a textbook metric accounting artifact ($0.50 / 0.50 = 1.00$).
- **Independent Battery:**
  When MIN_E is trained with adequate exposure on Task A to reach high performance (**83.3%**), and subsequently trained on Task B (inverted action mapping) and Task C:
  - Initial Task A: **83.3%**
  - Retest Task A: **16.7%**
  - Retention Ratio: **0.20**
  - Readout Weight Drift ($\Delta W$ Frobenius Norm): **2.08**
- **Verdict:** **METRICALLY INVALID / SEVERE FORGETTING IDENTIFIED**.  
  Because MIN_E has a single linear readout layer ($W \in \mathbb{R}^{40 \times 2}$), training on an inverted mapping overwrites the weights. The recurrent NMDA substrate retains state dynamics, but the plastic action interface cannot retain conflicting mappings without compartmentalized or modular routing.

---

#### 1.3 Compositional Generalization (Audit #5)
- **Phase 17 Reported:** Seen combinations = 66.7%, Held-out novel combination $(1, 1) = \mathbf{76.7\%}$.
- **Audit Findings:**
  1. *Shortcut Injection:* `TaskFamilyE_CompositionalContext` line 442 contained:
     `x[4 + comb_id] = 0.4` where `comb_id = ctx_a * 2 + ctx_b`.
     This supplied an explicit unique channel identifying the combination.
  2. *Online Adaptation:* The test evaluation loop executed plastic updates on test trials.
- **Controlled Battery Results:**
  - Original Unfrozen (with shortcut): Seen = 66.7%, Novel = 76.7%
  - Genuine Frozen Zero-Shot (with shortcut): Seen = 65.0%, Novel = **0.0%**
  - Pure Composition (no shortcut, frozen): Seen = 55.0%, Novel = **33.3%**
- **Verdict:** **NOT SUPPORTED / IMPLEMENTATION ARTIFACT**.  
  MIN_E lacks multi-layer hierarchical binding; a linear readout on temporal population features cannot solve XOR non-linear compositional binding zero-shot.

---

#### 1.4 Distractor & Interference Robustness (Audit #6)
- **Phase 17 Reported:** Low = 42.7%, Medium = 37.3%, High = 44.0% with stable CSI and low chatter.
- **Audit Finding:**
  Binary classification chance level is 50.0%. Performance of 36.0%–44.0% is at or below chance. Labeling below-chance accuracy as "robust" because it did not decline further is mathematically fallacious.
- **Verdict:** **METRICALLY INVALID**.

---

#### 1.5 Architectural Generalization Without Task-Specific Tuning (Audit #12)
- Tested a single frozen MIN_E configuration across all 6 task families:
  - Task A (Delayed Choice): 53.3%
  - Task B (Sequence Prediction): 52.0%
  - Task C (Embodied Spatial): 52.0%
  - Task D (Continual Learning): 53.3%
  - Task E (Compositional Context): 53.3%
  - Task F (Distractor Interference): 40.0%
  - **Mean Zero-Shot Across 6 Families:** **50.7% ± 5.2%**
- **Verdict:** **PARTIALLY SUPPORTED**.  
  The substrate executes without runtime crash across all task families, but its zero-shot performance is indistinguishable from chance without task-specific optimization.
