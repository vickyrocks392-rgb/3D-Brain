# Phase 18 — Baseline Reproduction Report

**Audit Phase:** Phase 18 — Scientific Audit, Reproducibility & Benchmark Integrity  
**Date:** September 30, 2026  
**Reference Commit:** `3be96fc` (`feat(phase-17): validate minimal brain generalization and transfer`)  
**Hardware & Architecture:** Apple Silicon (macOS-27.0-arm64-arm-64bit-Mach-O, Clang 21.0.0)  
**Python Runtime:** Python 3.14.6  
**Target Architecture:** MIN_E (Minimal Sufficient Brain: 8 active mechanisms, 4 pruned mechanisms)  
**Status:** **REPRODUCED WITHIN 0.0% STATISTICAL TOLERANCE (EXACT MATCH)**

---

### 1. Environmental Specification

| Property | Value |
|:---|:---|
| Git Commit Hash | `3be96fc6a7f7d1f5e8b4e75d0e2e9c15b9c03b1e` (HEAD -> main) |
| Operating System | macOS 27.0 (Darwin arm64) |
| Python Version | 3.14.6 |
| Compiler | Clang 21.0.0 (clang-2100.0.123.102) |
| Random Seeds | Evaluation Seeds: `42`, `43`, `44` |
| Primary Benchmark | `LongDelayContextualCreditTask`, `TaskFamilyA` through `TaskFamilyF` |
| Primary BrainLog | `experiments/results/brainlogs/EXP017_generalization.brainlog` (v3.3) |
| Primary Numerical Data | `experiments/results/EXP_017_RESULTS.json` |

---

### 2. Architecture & Parameter Specification (MIN_E)

- **Active Mechanisms (8):**
  1. `A_TARGET_SPECIFIC_TEACHING` (Action-specific scalar teaching signal $M_k(t)$)
  2. `B_DENDRITIC_TEACHING` (Dendritic compartment-gated teaching)
  3. `C_DENDRITIC_ELIGIBILITY` (Dendritic potential modulation of eligibility trace)
  4. `H_SLOW_ACTION_ASSEMBLY` ($\tau_{\text{slow}} = 1500\text{ ms}$ continuous integrator)
  5. `I_MUTUAL_INHIBITION` ($w_{\text{lat}} = 0.50$ lateral cross-inhibition)
  6. `J_TEMPORAL_POPULATION` (Combined temporal phase and rate representation $I_6$)
  7. `K_PERSISTENT_LATENT_NMDA` (Slow NMDA conductances $g_{\text{nmda}}$, $\tau_{\text{nmda}} = 100\text{ ms}$)
  8. `L_TARGET_ACTION_ROUTING` (Specific routing to action assemblies)

- **Pruned Mechanisms (4):**
  1. `D_DENDRITIC_STDP` ($S_0$, pruned)
  2. `E_RETROGRADE_SIGNALING` ($R_0$, pruned)
  3. `F_FAST_ACTION_ASSEMBLY` (pruned)
  4. `G_MEDIUM_ACTION_ASSEMBLY` (pruned)

- **Model Dimensions & Counts:**
  - Sensory input dimension: $8$
  - Hidden recurrent neurons: $40$ (two-compartment)
  - Output actions: $2$ (Action 0, Action 1)
  - Total physical neurons: $42$
  - Total synapses: $851$ ($8 \times 40 = 320$ input, $40 \times 40$ sparse recurrent $= 451$, $40 \times 2 = 80$ readout)
  - Readout trainable plastic weights: $80$
  - Readout biases: $2$
  - State variables: $282$

---

### 3. Quantitative Baseline Reproduction Battery

The original Phase 17 experiments were re-executed using the identical scripts, seeds, and evaluation harnesses to confirm whether the code reproduces the reported numbers.

#### 3.1 Stage A: Phase 16 Baseline Reproduction

| Metric | Phase 17 Reported | Phase 18 Re-executed | Tolerance ($\Delta$) | Status |
|:---|:---:|:---:|:---:|:---:|
| **Mean Accuracy** | 60.0% ± 5.7% SEM | 60.0% ± 5.7% SEM | 0.00% | **EXACT MATCH** |
| - Seed 42 | 48.0% | 48.0% | 0.00% | EXACT MATCH |
| - Seed 43 | 60.0% | 60.0% | 0.00% | EXACT MATCH |
| - Seed 44 | 72.0% | 72.0% | 0.00% | EXACT MATCH |
| **Mean Margin** | 2.3128 | 2.3128 | < 1e-4 | EXACT MATCH |
| **CSI** | 0.3339 | 0.3339 | < 1e-4 | EXACT MATCH |
| **Credit Leakage** | 0.6661 | 0.6661 | < 1e-4 | EXACT MATCH |
| **Decision Switches** | 0.9867 | 0.9867 | < 1e-4 | EXACT MATCH |

#### 3.2 Stage B: Temporal Horizon Extrapolation (Original Harness)

| Delay (ms) | Delay Steps | Phase 17 Reported | Phase 18 Re-executed | Match |
|:---:|:---:|:---:|:---:|:---:|
| 0 ms | 0 | 37.3% ± 18.0% | 37.3% ± 18.0% [0.52, 0.12, 0.48] | EXACT |
| 80 ms | 2 | 49.3% ± 3.8% | 49.3% ± 3.8% [0.52, 0.44, 0.52] | EXACT |
| 200 ms | 5 | 49.3% ± 1.9% | 49.3% ± 1.9% [0.52, 0.48, 0.48] | EXACT |
| 400 ms | 10 | 46.7% ± 5.0% | 46.7% ± 5.0% [0.52, 0.40, 0.48] | EXACT |
| 600 ms | 15 | 36.0% ± 0.0% | 36.0% ± 0.0% [0.36, 0.36, 0.36] | EXACT |
| 1000 ms | 25 | 48.0% ± 5.7% | 48.0% ± 5.7% [0.40, 0.52, 0.52] | EXACT |
| 1600 ms | 40 | 56.0% ± 5.7% | 56.0% ± 5.7% [0.52, 0.52, 0.64] | EXACT |
| 3000 ms | 75 | 85.3% ± 20.7% | 85.3% ± 20.7% [1.00, 0.56, 1.00] | EXACT |
| **5000 ms** | **125** | **100.0% ± 0.0%** | **100.0% ± 0.0% [1.00, 1.00, 1.00]** | **EXACT** |

#### 3.3 Stage D: Continual Learning & Forgetting

- **Reported Initial Task A:** 50.0%
- **Re-executed Initial Task A:** 50.0%
- **Reported Retest Task A:** 50.0%
- **Re-executed Retest Task A:** 50.0%
- **Reported Forgetting Index:** 0.0% (Retention: 1.00)
- **Re-executed Forgetting Index:** 0.0% (Retention: 1.00)

#### 3.4 Stage E: Compositional Generalization

- **Reported Seen Combinations:** 66.7% | **Re-executed:** 66.7%
- **Reported Novel Combination (1, 1):** 76.7% | **Re-executed:** 76.7%

#### 3.5 Stage F: Distractor Robustness

- **Reported LOW:** 42.7% | **Re-executed:** 42.7%
- **Reported MEDIUM:** 37.3% | **Re-executed:** 37.3%
- **Reported HIGH:** 44.0% | **Re-executed:** 44.0%

---

### 4. Reproduction Verdict

**BASELINE REPRODUCTION VERDICT: PASS**

The original Phase 17 implementation is 100% numerically reproducible from a clean checkout. Every number reported in `research/PHASE_17_RESULTS.md` reproduces exactly under the original code and seeds.

Having established absolute reproducibility of the original pipeline, Phase 18 now proceeds to execute the pre-registered scientific audit to determine whether these numbers reflect genuine biophysical computation or evaluation and benchmark artifacts.
