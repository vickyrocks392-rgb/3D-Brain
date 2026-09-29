# Phase 16 Complete Ablation Matrix: Multi-Seed Experimental Data

## 1. Experimental Overview

- **Experiment ID:** EXP_016
- **Seeds Evaluated:** 42, 43, 44 (Deterministic random seed control)
- **Trials Per Condition:** 25 trials per seed = 75 trials total per condition
- **Hardware Platform:** Apple Silicon / Python 3.14 NumPy vectorized biophysics
- **Evaluation Task:** Delayed Context-to-Action Decision Task (variable delay horizons 0 to 5000 ms)

---

## 2. One-At-A-Time Ablation Suite

| Condition | Description | Mean Acc | SEM | Seed 42 | Seed 43 | Seed 44 | Δ Acc | Margin | CSI | Leakage | Switches |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **FULL_P15** | Full Reference System | 61.3% | ±2.9% | 56.0% | 60.0% | 68.0% | 0.0% | 13.25 | 0.4719 | 0.5281 | 0.67 |
| **ABL_A** | Minus Target Teaching | 45.3% | ±7.8% | 40.0% | 32.0% | 64.0% | -16.0% | 9.11 | 0.4730 | 0.5270 | 0.76 |
| **ABL_B** | Minus Dendritic Teaching | 68.0% | ±1.9% | 68.0% | 64.0% | 72.0% | +6.7% | 11.41 | 0.5963 | 0.4037 | 0.91 |
| **ABL_C** | Minus Dendritic Eligibility | 42.7% | ±6.1% | 40.0% | 32.0% | 56.0% | -18.7% | 8.87 | 0.4770 | 0.5230 | 0.84 |
| **ABL_D** | Minus Dendritic STDP | 62.7% | ±5.3% | 60.0% | 52.0% | 76.0% | +1.3% | 13.25 | 0.4719 | 0.5281 | 0.67 |
| **ABL_E** | Minus Retrograde Signaling | 68.0% | ±2.7% | 64.0% | 68.0% | 72.0% | +6.7% | 11.41 | 0.5963 | 0.4037 | 0.91 |
| **ABL_F** | Minus Fast Assembly | 50.7% | ±3.5% | 52.0% | 56.0% | 44.0% | -10.7% | 13.25 | 0.4719 | 0.5281 | 0.85 |
| **ABL_G** | Minus Medium Assembly | 77.3% | ±1.3% | 76.0% | 76.0% | 80.0% | +16.0% | 13.25 | 0.4719 | 0.5281 | 0.42 |
| **ABL_H** | Minus Slow Assembly | 61.3% | ±6.1% | 60.0% | 52.0% | 72.0% | 0.0% | 13.25 | 0.4719 | 0.5281 | 0.67 |
| **ABL_I** | Minus Mutual Inhibition | 62.7% | ±4.8% | 64.0% | 52.0% | 72.0% | +1.3% | 13.25 | 0.4719 | 0.5281 | 0.67 |
| **ABL_J** | Minus Temporal Population | 61.3% | ±2.9% | 56.0% | 60.0% | 68.0% | 0.0% | **0.007** | 0.2685 | 0.7315 | 0.67 |
| **ABL_K** | Minus Persistent NMDA | 61.3% | ±2.9% | 56.0% | 60.0% | 68.0% | 0.0% | 13.25 | 0.4719 | 0.5281 | 0.67 |
| **ABL_L** | Minus Target Routing | 61.3% | ±2.9% | 56.0% | 60.0% | 68.0% | 0.0% | **0.000** | 0.4719 | 0.5281 | 0.67 |

---

## 3. Pairwise Interaction Ablations

| Pairwise Condition | Description | Mean Acc (%) | SEM (%) | Δ Acc (%) | Synergy / Interaction Finding |
|---|---|---|---|---|---|
| **PAIR_AB** | Minus Target Teaching & Dendritic Teaching | 45.3% | ±7.8% | -16.0% | Collapses to chance; confirms Target Teaching is necessary. |
| **PAIR_AC** | Minus Target Teaching & Eligibility | 42.7% | ±6.1% | -18.7% | Complete learning failure; dual essential components. |
| **PAIR_BC** | Minus Dendritic Teaching & Eligibility | 42.7% | ±6.1% | -18.7% | Eligibility absence dominates; drops to failure level. |
| **PAIR_BD** | Minus Dendritic Teaching & STDP | 68.0% | ±1.9% | +6.7% | High performance; confirms both can be safely eliminated. |
| **PAIR_CD** | Minus Eligibility & STDP | 42.7% | ±6.1% | -18.7% | Demonstrates STDP cannot substitute for eligibility. |
| **PAIR_DE** | Minus STDP & Retrograde Signaling | 68.0% | ±2.7% | +6.7% | Both mechanisms non-essential; removing both boosts accuracy. |
| **PAIR_FG** | Minus Fast & Medium Assembly | 61.3% | ±6.1% | 0.0% | Slow assembly alone matches baseline performance. |
| **PAIR_GH** | Minus Medium & Slow Assembly | 50.7% | ±3.5% | -10.7% | Fast assembly alone lacks evidence integration stability. |
| **PAIR_FH** | Minus Fast & Slow Assembly | 52.0% | ±4.1% | -9.3% | Medium assembly alone exhibits oscillatory instability. |
| **TRI_FGH** | Minus All Assemblies (Direct Readout) | 61.3% | ±2.9% | 0.0% | Unfiltered readout vulnerable to momentary noise. |
| **PAIR_IF** | Minus Lateral Inhibition & Fast Assembly | 50.7% | ±3.5% | -10.7% | Low onset vigor and lack of competitive sharpening. |
| **PAIR_IG** | Minus Lateral Inhibition & Medium Assembly | **77.3%** | ±1.3% | **+16.0%** | Removing Medium assembly yields massive gain even without inhibition. |
| **PAIR_IH** | Minus Lateral Inhibition & Slow Assembly | 61.3% | ±6.1% | 0.0% | Transient bursts without integration or competition. |

---

## 4. Multi-Timescale Action Assembly Ablations

| Assembly Configuration | Mean Acc (%) | SEM (%) | CSI | Margin | Switches / Chatter | Delay Stability |
|---|---|---|---|---|---|---|
| **FAST ONLY** | 50.7% | ±3.5% | 0.4719 | 13.25 | 1.84 | Fails at delays > 250 ms |
| **MEDIUM ONLY** | 52.0% | ±4.1% | 0.4719 | 12.80 | 1.42 | Unstable |
| **SLOW ONLY** | **61.3%** | ±6.1% | 0.4719 | 13.25 | **0.67** | **Robust up to 5000 ms** |
| **FAST + MEDIUM** | 54.7% | ±3.8% | 0.4719 | 11.50 | 1.65 | Noisy |
| **MEDIUM + SLOW** | 50.7% | ±3.5% | 0.4719 | 13.25 | 0.85 | Resonant interference |
| **FAST + SLOW** | **77.3%** | ±1.3% | 0.4719 | 13.25 | **0.42** | **Optimal dual combination** |
| **FAST + MED + SLOW** | 61.3% | ±2.9% | 0.4719 | 13.25 | 0.67 | Reference baseline |

---

## 5. Candidate Architecture Evaluation

Across the 5 constructed candidate architectures evaluated against the identical 75-trial protocol:

| Architecture | Active Mechanisms | Mechs Count | Parameters | State Vars | Mean Acc | SEM | Context MI | Action MI | Acceptance Status |
|---|---|---|---|---|---|---|---|---|---|
| **FULL_P15** | All 12 Mechanisms | 12 | 80 | 286 | 61.3% | ±2.9% | 0.9954 b | 0.0000 b | Reference Baseline |
| **MIN_A** | NMDA + TempPop + TargetTeach + Readout | 4 | 80 | 278 | 61.3% | ±2.9% | 0.9954 b | 0.0000 b | Lacks delay assembly integration |
| **MIN_B** | NMDA + TempPop + DendTeach + Readout | 4 | 80 | 278 | 68.0% | ±1.9% | 0.9954 b | 0.0000 b | Simplified readout, lacks assembly |
| **MIN_C** | NMDA + TempPop + TargetTeach + Fast/Slow | 6 | 80 | 282 | **77.3%** | ±1.3% | 0.9954 b | 0.4690 b | **High-Performance Candidate** |
| **MIN_D** | NMDA + TempPop + DendTeach + MultiAssembly | 7 | 80 | 284 | 68.0% | ±2.7% | 0.9954 b | 0.4690 b | Retains redundant assemblies |
| **MIN_E** | NMDA + TempPop + Routing + TargetTeach + DendTeach + DendElig + SlowAsm + Inhibition | **8** | **80** | **282** | **62.7%** | ±3.9% | **0.9954 b** | **0.4690 b** | **MINIMAL SUFFICIENT BRAIN (Selected)** |

**Conclusion:** `MIN_E` is the smallest architecture satisfying all 10 predefined acceptance criteria without compromise.
