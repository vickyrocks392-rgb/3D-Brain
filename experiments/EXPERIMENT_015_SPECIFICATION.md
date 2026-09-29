# Experiment 015 Specification: Target-Specific Dendritic Learning, Retrograde Signaling & Multi-Timescale Action Assemblies

## 1. Central Scientific Question
> **Can a target-specific, biologically constrained dendritic learning mechanism convert the temporally integrated contextual representation into reliable autonomous action without global backpropagation through time (BPTT)?**

Phase 14 resolved the representational bottleneck:
- Continuous contextual state: $1.000\text{ bit}$ ($100\%$ probe accuracy)
- Persistent action representation: $0.2358\text{ bit}$ ($77.8\%$ probe accuracy)
- Autonomous behavioral choice: $\sim 44\%$ accuracy (near chance)

Experiment 015 investigates why the brain can represent the answer in its neural state but cannot reliably turn that representation into the correct action. It isolates whether the failure is in **representation**, **action readout**, **learning / credit assignment**, or **action competition**.

---

## 2. Experimental Conditions Matrix (A through N)

All conditions are evaluated across deterministic seeds `42, 43, 44` on identical long-delay contextual choice tasks (25 trials per seed, 75 trials total per condition):

| Condition ID | Name | Core Architecture & Plasticity Mechanism |
|---|---|---|
| **COND_A** | `PHASE14_BASELINE` | Phase 14 point-process burst/coincidence baseline (S9/L) |
| **COND_B** | `FROZEN_READOUT` | Frozen Action Readout ($\Delta W = 0$, Case A control) |
| **COND_C** | `OFFLINE_LINEAR_READOUT` | Pure offline regularized ridge readout on temporal state (O0 control) |
| **COND_D** | `LOCAL_HEBBIAN` | Local correlation / Oja Hebbian learning (D2) |
| **COND_E** | `ELIGIBILITY_ONLY` | Unsupervised dendritic eligibility accumulation without teaching (D3) |
| **COND_F** | `GLOBAL_REWARD` | Three-factor global scalar reward: $M_k(t) = R(t)$ for all $k$ (D4, G0) |
| **COND_G** | `TARGET_SPECIFIC_REWARD` | Target-specific scalar reward: $M_k(t) \in \{+1, -1\}$ (D5, G1) |
| **COND_H** | `DENDRITIC_TEACHING` | Postsynaptic dendritic eligibility $\times$ target teaching signal (D7, G3) |
| **COND_I** | `RETROGRADE_SIGNAL` | Action-compartment specific retrograde feedback (D7 + R2) |
| **COND_J** | `DENDRITIC_STDP` | Active dendritic timing-dependent plasticity (S2/S3) |
| **COND_K** | `TARGET_SPECIFIC_DENDRITIC_STDP` | Neuromodulated target-specific dendritic STDP (S4) |
| **COND_L** | `MULTI_TIMESCALE_ASSEMBLY` | Multi-timescale action assemblies (Fast 25ms + Med 250ms + Slow 1500ms, A5) |
| **COND_M** | `TEMPORAL_POP_PLUS_TARGET_LEARNING`| Combined temporal population representation (I6) + target dendritic learning (D8) |
| **COND_N** | `INTEGRATED_PHASE15_CANDIDATE` | Integrated Phase 15 architecture (D8 + G3 + R2 + S4 + A5 + I6) |

---

## 3. Decision Tree: Bottleneck Localization (Cases A–E)
- **CASE A:** Freeze upstream representation ($\Delta W_{\text{rec}} = 0$), train only action readout.
- **CASE B:** Freeze action readout ($\Delta W_{\text{readout}} = 0$), change upstream representation.
- **CASE C:** Train both upstream substrate and action readout with local learning.
- **CASE D:** Analytical BPTT reference upper bound.
- **CASE E:** Continuous-state subthreshold oracle readout.

---

## 4. Pure Offline Readout Suite (O0–O3)
Extracts temporal representations from the frozen brain and trains offline classifiers without modifying synaptic weights:
- **O0:** Regularized Linear Classifier (Ridge regression)
- **O1:** Nonlinear Shallow Classifier (RBF kernel / Nearest Centroid)
- **O2:** Small Recurrent Readout (Leaky integrator RNN)
- **O3:** Population Action Readout (Multi-unit ensemble pooling)

---

## 5. Parametric Sweeps & Robustness Batteries
1. **Delayed Reward Horizon Sweep:** $\Delta t \in [0, 50, 100, 250, 500, 1000, 2000, 5000]\text{ ms}$
2. **Teaching Signal Structure:** G0 (Global), G1 (Target scalar), G2 (Local centered), G3 (Dendritic targeted), G4 (Shuffled), G5 (Sign-flipped)
3. **Retrograde Feedback Modes:** R0 (None), R1 (Scalar), R2 (Action-specific), R3 (Compartment-specific), R4 (Delayed)
4. **Dendritic STDP Regimes:** S0 (None), S1 (Somatic pair), S2 (Dendritic pair), S3 (Eligibility STDP), S4 (Target-specific STDP)
5. **Timescale Assembly Regimes:** A0 (Fast), A1 (Med), A2 (Slow), A3 (Fast+Med), A4 (Med+Slow), A5 (Fast+Med+Slow)
6. **Input Representation Regimes:** I0 (Spikes), I1 (Counts), I2 (Bursts), I3 (Temporal pop), I4 (Phase), I5 (Resonance), I6 (Combined)
7. **Action Competition Variations:** No inhibition, weak, baseline, strong, no recurrent excitation
8. **Computational Scaling Ladder:** $N \in [50, 100, 300, 1000, 10000, 50000, 100000]$

---

## 6. Causal Action Intervention Suite (C1–C12)
- `C1_CONTEXT_SCRAMBLE`: Scrambles upstream contextual cue
- `C2_CONTEXT_SWAP`: Inverts context cue ($A \leftrightarrow B$) while holding delay inputs constant
- `C3_READOUT_LESION`: Ablates $80\%$ of readout synapses
- `C4_IRRELEVANT_CONTEXT`: Injects orthogonal noise context into readout
- `C5_LABEL_PERMUTATION`: Permutes action-target assignment
- `C6_FREEZE_READOUT`: Disables readout synaptic updates
- `C7_ACTION_POP_SWAP`: Permutes action population drives
- `C8_TARGET_SIGNAL_SHUFFLE`: Shuffles teaching signals across actions
- `C9_TARGET_SIGNAL_SIGN_FLIP`: Inverts sign of teaching signals
- `C10_RETROGRADE_REMOVAL`: Completely silences retrograde feedback
- `C11_ELIGIBILITY_REMOVAL`: Clamps dendritic eligibility to zero
- `C12_TEMPORAL_REP_REMOVAL`: Strips temporal features, forcing instantaneous spike inputs (I0)

---

## 7. Metrics & Credit Telemetry
- **Behavioral Accuracy (%):** Fraction of correct delayed contextual choices
- **Action Margin:** Difference between winner and runner-up activation
- **Decision Confidence:** Normalized softmax margin
- **Action Chatter Ratio:** High-frequency toggling between action populations
- **Credit Specificity Index (CSI):** Fraction of weight updates allocated to target action synapses
- **Credit Precision:** Fraction of updates aligned with causal improvement
- **Credit Leakage:** $1.0 - \text{CSI}$
- **Gini Concentration:** Inequality of updates across synapses
- **Shannon Mutual Information (bits):** Context MI $I(X; C)$ and Action-Relevant MI $I(X; A^*)$
