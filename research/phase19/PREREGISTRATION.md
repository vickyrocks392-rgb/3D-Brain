# Phase 19 Preregistration: Biological Credit Horizon, Synaptic Consolidation & Fair Baseline Comparison

**Document ID:** PREREG-PHASE19-001  
**Date:** September 30, 2026  
**Auditor / Lead Investigator:** Autonomous Scientific Research Agent  
**Parent Foundation:** Phase 18 Scientific Audit & Falsification Report (`research/PHASE_18_RESULTS.md`)  
**Repository:** `~/Documents/3D-Brain`  
**GitHub:** `https://github.com/vickyrocks392-rgb/3D-Brain.git`

---

## 1. Context and Motivation

Phase 18 systematically audited and falsified several major headline claims from earlier phases:
1. **Temporal Extrapolation (5,000 ms / 100% accuracy):** Falsified as an implementation artifact caused by target label leakage into `model.step()` during delay steps. Under strict un-leaked evaluation, MIN_E achieved 42.7% ± 1.9% (chance).
2. **0% Catastrophic Forgetting (Retention = 1.00):** Falsified as metrically invalid because Task A was at chance initially. When Task A was learned to 83.3%, retraining on an inverted mapping caused Task A retest accuracy to drop to 16.7% (retention ratio = 0.20, severe catastrophic forgetting).
3. **100K Throughput (215.98B syn-ops/s):** Falsified as a benchmark accounting failure due to `min(N, 120)` matrix clamping. Real execution yielded 441.8M syn-ops/s.
4. **Compositional Generalization (76.7%):** Falsified because it relied on online test-time plastic updates and an explicit sensory shortcut channel.

Phase 19 starts from this verified audit. The purpose of Phase 19 is NOT to add complexity or make exaggerated claims, but to conduct rigorous, controlled scientific investigations across three core tracks:
- **Track A:** What is the *true* empirical temporal credit-assignment horizon of local dendritic eligibility + target-specific teaching when all continuous supervision and target leakage are removed?
- **Track B:** Can biologically plausible local synaptic consolidation mechanisms (dual-timescale synapses, synaptic tagging and capture, metaplasticity) protect previously learned knowledge from catastrophic forgetting while preserving new learning capacity?
- **Track C:** How does the minimal biological architecture (MIN_E) compare head-to-head against a causal Transformer, a BPTT recurrent control, and a simple recurrent baseline under strictly equalized parameter, compute, and memory budgets?

---

## 2. Non-Negotiable Methodological Rules

1. **No Target Supervision During Delay:** The agent must never receive the target action, cue identity, or teacher signal during delay intervals.
2. **No Target Signal on Reward == 0:** Teaching signals must not fire merely because reward is 0. Feedback occurs ONLY at causal reward delivery steps.
3. **Frozen Evaluation:** During zero-shot test evaluation, all synaptic plasticity, state consolidation, and statistics must be strictly frozen (`is_frozen = True`).
4. **Strict Causal Timing:** All sensory events, choices, rewards, and teaching updates must occur in causal order:
   $$\text{Cue} \to \text{Latent Hold} \to \text{Delay} \to \text{Choice Step} \to \text{Reward Delay} \to \text{Sparse Outcome/Teaching}$$
5. **No Shortcut Channels:** Inputs must strictly represent task stimuli without extraneous index markers or leaked labels.
6. **Finite Sample Controls & Multiple Seeds:** Every experiment must be evaluated across $\ge 5$ independent random seeds with reported mean, standard deviation, and statistical confidence intervals.
7. **Two-Sided Stability/Plasticity Criterion:** In continual learning, a mechanism is successful if and only if it preserves old knowledge AND permits new knowledge acquisition. Freezing the network is an explicit failure mode.

---

## 3. Track A: True Long-Horizon Credit Assignment

### 3.1 Research Question
What is the actual temporal credit-assignment horizon of local dendritic eligibility + target-specific teaching in MIN_E when continuous supervision is completely removed, and how does performance scale across delays from 0 ms to $\ge 5,000$ ms?

### 3.2 Preregistered Hypotheses
- **H_A1 (Finite Biological Credit Horizon):** The credit assignment horizon of MIN_E is finite and strictly bounded by the interaction of NMDA receptor conductance decay ($\tau_{\text{nmda}}$) and dendritic eligibility trace decay ($\tau_{\text{elig}}$). Beyond a critical delay $\tau_{\text{credit}} \approx 400\text{–}800\text{ ms}$, performance drops monotonically toward chance ($50\%$).
- **H_A2 (Necessity of Persistent NMDA Conductance - K):** Lesioning Mechanism K collapses retention at all delays $\ge 250\text{ ms}$, reducing task performance to chance ($p < 0.01$).
- **H_A3 (Necessity of Dendritic Eligibility - C):** Lesioning Mechanism C eliminates the trace required to bridge the delay between choice and reward, collapsing credit specificity to zero.
- **H_A4 (Necessity of Target-Specific Teaching - A):** Lesioning Mechanism A to a global scalar reward (G0) causes credit diffusion, impairing learning efficiency compared to target-specific routing (G1/G3).
- **H_A5 (Baseline Contrast):** BPTT and Causal Transformers, having explicit non-local temporal pathways, will maintain above-chance performance across longer horizons than unaugmented local biological traces, but will require higher memory buffers.

### 3.3 Independent & Dependent Variables
- **Independent Variables:**
  - Delay interval: $0, 25, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000\text{ ms}$ (steps $0, 1, 2, 3, 6, 12, 25, 38, 50, 75, 100, 125$ at $40\text{ ms/step}$).
  - Architectural condition:
    1. MIN_E (Full minimal architecture)
    2. MIN_E without dendritic eligibility (Ablate C)
    3. MIN_E without target-specific teaching (Ablate A, fallback to G0)
    4. MIN_E without persistent NMDA (Ablate K)
    5. BPTT reference (matched recurrent baseline)
    6. Simple recurrent baseline (DiscreteLeakyRNN)
    7. Causal Transformer reference (SmallCausalTransformer)
- **Dependent Variables:**
  - Task accuracy (% correct on evaluation set)
  - Held-out cross-validated accuracy
  - Credit Specificity Index (CSI)
  - Synaptic update magnitude ($|\Delta W|$)
  - Eligibility trace persistence at choice and reward
  - Empirical credit horizon ($\tau_{\text{horizon}}$, defined as the maximum delay where test accuracy $\ge 65.0\%$ with $p < 0.05$ vs chance).

---

## 4. Track B: Synaptic Consolidation & Catastrophic Forgetting

### 4.1 Research Question
Can a biologically plausible local synaptic consolidation mechanism prevent catastrophic forgetting in the minimal architecture while preserving the ability to learn new tasks?

### 4.2 Candidate Mechanisms
1. **M0 — Baseline MIN_E:** Single-timescale plastic readout weights with unconstrained decay/overwriting (Phase 18 baseline).
2. **M1 — Dual-Timescale Synapses ($W = W_{\text{fast}} + W_{\text{slow}}$):**
   - $W_{\text{fast}}$: Fast plastic component updated via local dendritic eligibility and teaching; subject to homeostatic decay toward zero.
   - $W_{\text{slow}}$: Slow consolidated component that integrates low-pass filtered updates from $W_{\text{fast}}$:
     $$\Delta W_{\text{slow}} = \frac{1}{\tau_{\text{cons}}} W_{\text{fast}}$$
   - Readout drive: $u_k = \sum_j (W_{\text{fast}, jk} + W_{\text{slow}, jk}) x_j + b_k$.
3. **M2 — Synaptic Tagging & Consolidation (STC):**
   - Synaptic tag $T_{jk}(t) = \text{decay} \cdot T_{jk}(t-1) + |e_{jk}(t)|$.
   - When a successful reward event occurs ($R > 0$), systemic protein-synthesis-like consolidation transfers tagged weights into a protected consolidated matrix $W_{\text{cons}}$, while protecting against subsequent anti-correlated updates.
4. **M3 — Metaplasticity (Synaptic Stability Cascade):**
   - Each synapse possesses an internal stability state $S_{jk} \ge 0$.
   - Concurrently reinforced updates increment $S_{jk}$.
   - Effective learning rate scales inversely with stability:
     $$\eta_{\text{eff}, jk} = \frac{\eta_0}{1 + \gamma S_{jk}}$$
   - High-stability synapses resist overwriting during subsequent tasks, while low-stability synapses remain plastic.

### 4.3 Preregistered Success / Failure Criteria
- **Success Criterion (Stability + Plasticity):**
  - Meaningful retention of Task A: Absolute post-retention $acc_{A,\text{post}} \ge 65.0\%$, relative retention ratio $acc_{A,\text{post}} / acc_{A,\text{pre}} \ge 0.75$.
  - AND Meaningful acquisition of Task B: $acc_{B} \ge 70.0\%$.
- **Failure Modes:**
  - Catastrophic Forgetting: $acc_{A,\text{post}} / acc_{A,\text{pre}} < 0.40$.
  - Plasticity Freeze: $acc_{B} < 60.0\%$ (the mechanism protects old knowledge simply by disabling learning).
  - Capacity Exhaustion: Failure to learn Task C after A and B ($acc_C < 60.0\%$).

### 4.4 Task Protocols
- Protocol 1: $A \to B$
- Protocol 2: $B \to A$
- Protocol 3: $A \to B \to A$ (Reversal / Catastrophic Forgetting test)
- Protocol 4: $A \to B \to C$ (Multi-task sequential adaptation)
- Protocol 5: $A \to B \to C \to A$ (Longitudinal retention loop)

---

## 5. Track C: Fair Causal Transformer & Baseline Comparison

### 5.1 Research Question
Under rigorously equalized parameter, memory, and compute budgets, how does MIN_E compare against standard non-biological baselines on delayed credit assignment and continual learning?

### 5.2 Equalization Methodology
To ensure scientific fairness, all models are constrained to a matched parameter budget ($\approx 2,000$ total parameters) and trained under identical trial schedules:
1. **MIN_E:**
   - 40 hidden units, 8 inputs, 2 outputs.
   - Fixed recurrent & input synapses: $8 \times 40 + 40 \times 40 = 1,920$.
   - Plastic readout synapses: $40 \times 2 = 80$ (+ 2 biases = 82 trainable).
   - Total parameters: $2,002$.
   - State variables: 282 (soma, dendrite, NMDA, assemblies, eligibilities).
2. **Small Causal Transformer:**
   - $d_{\text{model}} = 16$, $n_{\text{heads}} = 2$, $d_{\text{k}} = 8$, $d_{\text{ff}} = 32$, $L = 1$, context window $T = 150$.
   - Embedding: $W_{\text{in}} [8 \times 16] = 128$.
   - Attention: $W_q, W_k, W_v, W_o [4 \times (16 \times 16)] = 1,024$.
   - LayerNorms: $2 \times 2 \times 16 = 64$.
   - FeedForward: $W_{ff1} [16 \times 32] + b_{ff1} [32] + W_{ff2} [32 \times 16] + b_{ff2} [16] = 1,072$.
   - Readout: $W_{\text{out}} [16 \times 2] + b_{\text{out}} [2] = 34$.
   - Total trainable parameters: $\approx 2,322$ (matched scale $\sim 2\text{K}$).
3. **BPTT Recurrent Reference (DiscreteLeakyRNN):**
   - Hidden dimension $H = 40$, inputs $8$, outputs $2$.
   - $W_{xh} [8 \times 40] = 320$, $W_{hh} [40 \times 40] = 1600$, biases $40 + 2 = 42$, $W_{hy} [40 \times 2] = 80$.
   - Total trainable parameters: $2,042$.
4. **Simple Recurrent Baseline (Heuristic / Online Control):**
   - Same $H = 40$ leaky recurrent substrate with simple reward-modulated Hebbian / delta readout.

### 5.3 Comparison Metrics
- Trainable parameters
- Fixed parameters
- State variables
- Execution time per trial (training & inference)
- Sample efficiency (trials required to reach $75\%$ accuracy)
- Final held-out test accuracy
- Long-delay performance ($500\text{ ms}$, $1,000\text{ ms}$, $5,000\text{ ms}$)
- Continual learning retention ratio ($A \to B \to A$)

---

## 6. Preregistered Statistical Protocols

- All experiments executed across 5 independent random seeds: `[42, 101, 2024, 777, 999]`.
- Performance metrics reported as $\text{Mean} \pm \text{Standard Deviation}$ and $95\%$ bootstrap confidence intervals.
- Significance testing via paired two-tailed $t$-tests and Wilcoxon signed-rank tests against the null hypothesis (chance level $50\%$ or baseline performance).
- Threshold for scientific support: $p < 0.05$ with minimum effect size Cohen's $d > 0.80$ (large effect).

---

## 7. Preregistered Claims Classification Criteria

At the conclusion of Phase 19, all claims will be strictly classified into:
- **SUPPORTED:** Empirical evidence meets preregistered success criteria across all random seeds with $p < 0.05$ and no contradictory ablations.
- **PARTIALLY SUPPORTED:** Hypothesis confirmed in a bounded regime (e.g. short delays only) or with moderate effect size.
- **NOT SUPPORTED:** Empirical evidence fails to reach statistical significance ($p \ge 0.05$) or fails success criteria.
- **FALSIFIED:** Empirical evidence directly contradicts the hypothesis (e.g. proposed mechanism performs worse than baseline).
- **INCONCLUSIVE:** Insufficient statistical power or high variance preventing clear determination.
