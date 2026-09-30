# Phase 19 Experiment Specifications

**Document ID:** EXPSPEC-PHASE19-001  
**Date:** September 30, 2026  
**Status:** Approved & Frozen for Execution  
**Parent Document:** [`research/phase19/PREREGISTRATION.md`](file:///Users/kalachanddash/Documents/3D-Brain/research/phase19/PREREGISTRATION.md)

---

## Overview of Experiment Matrix

Phase 19 executes three primary empirical campaigns corresponding to Tracks A, B, and C, followed by full causal ablation sweeps and comprehensive parameter/compute accounting.

| Experiment ID | Track | Primary Investigation | Conditions / Sweeps | Seeds | Primary Metrics |
|:---|:---:|:---|:---|:---:|:---|
| **EXP-19-A1** | A | True Delayed-Credit Horizon Sweep | 12 Delays (0 to 5000 ms), 7 Models | 5 | Accuracy, CSI, Horizon Bound |
| **EXP-19-A2** | A | Biophysical Mechanism Necessity in Long Delays | Lesion K, C, A, H, I on delays | 5 | $\Delta\text{Accuracy}$, Causal Necessity |
| **EXP-19-A3** | A | Causal Attribution & Parameter Sweeps | Sweep $\tau_{\text{elig}}$, $\tau_{\text{nmda}}$, plateau | 5 | Sensitivity curves, Optimal regime |
| **EXP-19-B1** | B | Baseline Forgetting Reproduction | Reversals ($A \to B \to A$) on MIN_E | 5 | Absolute & Relative Retention |
| **EXP-19-B2** | B | Synaptic Consolidation Mechanisms | Baseline, Dual-Timescale, Tagging, Metaplasticity | 5 | Retention, Acquisition, Turnover |
| **EXP-19-B3** | B | Continual Learning Sequence Generalization | 5 Sequences ($A \to B$, $B \to A$, $A \to B \to A$, $A \to B \to C$, $A \to B \to C \to A$) | 5 | Forward/Backward Transfer, Capacity |
| **EXP-19-B4** | B | Causal Ablation of Consolidation | Tag disruption, decay variation, component knockout | 5 | Necessity of consolidation components |
| **EXP-19-C1** | C | Equalized Budget Benchmark | MIN_E, Causal Transformer, BPTT, Simple RNN | 5 | Param/FLOP/RSS/Time/Accuracy |
| **EXP-19-C2** | C | Long-Delay Stress Test Under Matched Budget | 0 ms to 5000 ms on all 4 baselines | 5 | Scaling with sequence length |
| **EXP-19-C3** | C | Continual Learning Stress Test Under Matched Budget | $A \to B \to A$ on all 4 baselines | 5 | Retention across architectures |

---

## Detailed Experiment Protocols

### 1. Track A: True Long-Horizon Credit Assignment

#### Protocol EXP-19-A1: Clean Delayed-Credit Sweep
- **Task Structure:**
  - `Cue Steps:` 5 steps ($200\text{ ms}$)
  - `Delay Steps:` $0, 1, 2, 3, 6, 12, 25, 38, 50, 75, 100, 125$ steps (corresponding to $0, 40, 80, 120, 240, 480, 1000, 1520, 2000, 3000, 4000, 5000\text{ ms}$).
  - `Choice Step:` 1 step. Model must output action probabilities and commit to a choice. No target or feedback is given.
  - `Reward Delay:` 2 steps ($80\text{ ms}$) between choice and reward delivery.
  - `Reward Step:` 1 step. Scalar reward $R \in \{+1.0, -0.5\}$ is delivered, along with local teaching signal.
  - `Total Trial Steps:` $5 + \text{delay} + 1 + 2 + 1 = 9 + \text{delay}$.
- **Training Set:** 30 training trials per condition.
- **Evaluation Set:** 30 held-out test trials per condition, with weights completely frozen (`is_frozen = True`).
- **Models Evaluated:**
  1. `MIN_E`
  2. `MIN_E_NO_ELIGIBILITY` (Mechanism C ablated)
  3. `MIN_E_NO_TARGET_TEACHING` (Mechanism A ablated, G0 global scalar)
  4. `MIN_E_NO_NMDA` (Mechanism K ablated)
  5. `BPTT_RECURRENT` (DiscreteLeakyRNN trained via BPTT)
  6. `SIMPLE_RECURRENT` (DiscreteLeakyRNN with reward-modulated Hebbian learning)
  7. `CAUSAL_TRANSFORMER` (SmallCausalTransformer)

#### Protocol EXP-19-A2: Causal Ablations & Credit Attributions
- Test causal attribution: Does the reward at step $T$ update synapses that were active during the cue at step $0\dots 4$?
- Measure cross-correlation between Cue representation and synaptic weight updates $\Delta W$:
  $$\rho_{\text{cue}} = \text{Corr}(x_{\text{cue}}, \Delta W_{\text{reward}})$$
- Perform lesion studies on K (NMDA), C (Eligibility), A (Target Teaching), H (Slow Assembly), I (Lateral Inhibition).

---

### 2. Track B: Synaptic Consolidation & Catastrophic Forgetting

#### Task Families for Continual Learning:
- **Task A (Delayed Contextual Choice - Standard):** Context 0 $\to$ Action 0; Context 1 $\to$ Action 1.
- **Task B (Delayed Contextual Choice - Inverted):** Context 0 $\to$ Action 1; Context 1 $\to$ Action 0.
- **Task C (Delayed Contextual Choice - Feature Shift):** Shifted input cues ($x[2], x[3]$) mapped to Actions 0 and 1.

#### Evaluated Consolidation Architectures:
1. **Baseline MIN_E (M0):** Standard plastic readout weights with unconstrained learning.
2. **Dual-Timescale Readout (M1):**
   - $W = W_{\text{fast}} + W_{\text{slow}}$.
   - $W_{\text{fast}}$ updates via eligibility and teaching: $\Delta W_{\text{fast}} = \eta_{\text{fast}} (e \cdot M) - \lambda_{\text{fast}} W_{\text{fast}}$.
   - $W_{\text{slow}}$ integrates fast weights slowly: $\Delta W_{\text{slow}} = \frac{1}{\tau_{\text{cons}}} W_{\text{fast}}$.
3. **Synaptic Tagging & Capture (M2):**
   - Tag accumulation: $T_{jk}(t) = \gamma_T T_{jk}(t-1) + |e_{jk}(t)|$.
   - Tagged capture: Upon positive reward ($R > 0$), consolidated weights $W_{\text{cons}, jk} += \eta_{\text{cons}} T_{jk} \cdot \text{sign}(W_{jk})$.
   - Effective weight: $W_{jk} = W_{\text{plastic}, jk} + W_{\text{cons}, jk}$.
4. **Metaplasticity Cascade (M3):**
   - Stability register $S_{jk} \ge 0$.
   - Concurrently reinforced updates increment $S_{jk} += \Delta S$.
   - Effective learning rate $\eta_{\text{eff}, jk} = \frac{\eta_0}{1 + \beta S_{jk}}$.

#### Evaluation Protocol:
- Train on Task A (40 trials). Evaluate Task A frozen ($acc_{A,\text{pre}}$).
- Train on Task B (40 trials). Evaluate Task B frozen ($acc_{B}$).
- Retest Task A frozen ($acc_{A,\text{post}}$).
- Compute:
  - Absolute Retention: $acc_{A,\text{post}}$
  - Relative Retention Ratio: $acc_{A,\text{post}} / acc_{A,\text{pre}}$
  - Backward Transfer: $acc_{A,\text{post}} - acc_{A,\text{pre}}$
  - Plasticity / Acquisition: $acc_{B}$
  - Stability-Plasticity Product: $acc_{A,\text{post}} \times acc_{B}$

---

### 3. Track C: Fair Causal Transformer Baseline Comparison

#### Parameter and Compute Accounting Methodology:
- Count trainable parameters, fixed parameters, and state variables strictly.
- Measure physical execution time (seconds per 1,000 steps).
- Measure peak memory (resident set size in MB).
- Measure FLOPs per step analytically:
  - MIN_E: Matrix-vector products for recurrent state ($40 \times 40 = 1,600$), input ($8 \times 40 = 320$), readout ($40 \times 2 = 80$), assemblies ($2 \times 3 \times 2 = 12$). Total $\approx 2,012$ MACs = $4,024$ FLOPs/step.
  - Transformer: Causal self-attention ($2 T D^2 + 2 T^2 D$), Feedforward ($4 T D D_{ff}$), Readout ($2 T D K$). Scales quadratically with $T$.
  - BPTT Leaky RNN: Matrix-vector products ($40 \times 40 + 8 \times 40 + 40 \times 2 = 2,000$ MACs) forward + backward unrolled over $T$ steps ($2\times$ forward compute).

---

## 4. Execution Plan & Artifact Outputs

1. Implementation of consolidation mechanisms in `brain/synaptic_consolidation.py` and integration into `ActionReadout` / `Phase16Network`.
2. Implementation of Track A benchmark suite in `benchmarks/phase19_credit_horizon.py`.
3. Implementation of Track B benchmark suite in `benchmarks/phase19_consolidation.py`.
4. Implementation of Track C benchmark suite in `benchmarks/phase19_transformer_comparison.py`.
5. Master Phase 19 runner in `experiments/run_phase19_experiments.py`.
6. Output raw data JSON artifacts in `experiments/results/PHASE_19_RESULTS.json`.
7. Analysis and documentation in `research/phase19/`.
