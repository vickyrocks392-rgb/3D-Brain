# 3D-Brain Phase 8: Negative Results & Falsified Expectations

**Phase:** Phase 8 — Balanced Excitation/Inhibition & Asynchronous-Irregular Regimes  
**Status:** Documented per Project Hard Rules (Rule 5: Negative results are first-class results)  
**Date:** 2026-09-28  

---

## 1. Executive Summary of Negative Findings

Phase 8 investigated whether imposing Dale's Principle (~80% excitatory / ~20% fast-spiking inhibitory interneurons) and homeostatic synaptic scaling in a spatial spiking substrate can prevent pathological global hypersynchrony and enter the Asynchronous-Irregular (AI) regime, while preserving functional modularity ($Q_{\text{func}} > 0.30$) and persistent memory.

In accordance with Hard Project Rule 5 (*Negative results are first-class results*) and Rule 6 (*Do not overclaim*), this document details every expectation, target threshold, or hypothesis that failed, showed ambiguity, or exhibited trade-offs during Experiment 008.

---

## 2. Specific Negative Results & Falsifications

### Negative Result 1: Functional Modularity Target ($Q_{\text{func}} > 0.30$) Not Met
* **Hypothesis Target:** $Q_{\text{func}} > 0.30$ under balanced E/I spiking dynamics.
* **Empirical Finding:** Across all primary E/I conditions (`B_EI_NO_HOMEO`, `C_EI_HOMEO`, `D_EI_STDP`, `E_EI_STDP_HOMEO`), empirical functional modularity measured from binned spike cross-correlation matrices ranged between:
  $$Q_{\text{func}} \in [0.169, 0.227]$$
  Mean across seeds for the full system (`E_EI_STDP_HOMEO`) was $Q_{\text{func}} = 0.169 \pm 0.021$.
* **Scientific Reality:** While functional communities still formed ($Q_{\text{func}} > 0.15$), the ambitious threshold of $0.30$ was not attained. Dale-constrained E/I balance inherently introduces diffuse recurrent inhibitory currents that suppress both within-module and between-module correlations. This desynchronization prevents runaway modular clustering unless dense local spatial clustering is enforced.

### Negative Result 2: Incomplete Recovery of Temporal Context Disambiguation
* **Hypothesis Target:** Phase 8 E/I balance will restore context disambiguation performance to $>85\%$, resolving the Phase 7 hypersynchronization bottleneck.
* **Empirical Finding:**
  - Removing inhibition entirely (`ABL_01_NO_INHIBITION`) collapsed context classification to exact chance: **50.0%**.
  - Adding 20%–30% fast inhibitory interneurons rescued classification above chance to **58.3% – 62.5%**.
  - However, accuracy remained substantially below the rate-coded Phase 6 benchmark (~90%) and Phase 7 frozen spiking reference (75.0%).
* **Causal Diagnosis:** Fast inhibitory interneurons reduce pairwise correlation and prevent runaway saturation, expanding state-space dimensionality (Participation Ratio $PR \approx 4.9$). However, the discrete nature of spike arrivals introduces high-frequency temporal jitter that degrades linear hyperplane separability when observation cues are brief, unless long temporal synaptic filters or dedicated dendritic integrators are added.

### Negative Result 3: ISI Irregularity ($CV_{\text{ISI}}$) Did Not Reach Theoretical Poisson Levels ($CV \ge 1.0$)
* **Hypothesis Target:** A canonical Asynchronous-Irregular regime requires $CV_{\text{ISI}} \approx 1.0$ (Poisson-like spike irregularity).
* **Empirical Finding:**
  - Measured $CV_{\text{ISI}} = 0.118 - 0.125$ across E/I conditions.
  - Measured Fano Factor $= 0.12 - 0.14$.
* **Causal Diagnosis:** Although pairwise spike-count correlations were low ($r_{\text{sc}} = 0.028 - 0.033$, fulfilling the *Asynchronous* criterion), the individual LIF spike trains operated in a moderately regular pacemaker-like regime under somatic tonic current drive rather than purely fluctuation-driven subthreshold crossings. Achieving $CV_{\text{ISI}} \ge 1.0$ in finite-size networks requires either massive balanced background synaptic bombardment ($K \ge 1000$ incoming synapses per neuron) or stochastic Poisson synaptic release.

### Negative Result 4: STDP Provided Negligible Functional Separation from Frozen Weights
* **Hypothesis Target:** STDP will reshape recurrent connectivity to dynamically enhance functional modularity and predictive computation.
* **Empirical Finding:**
  - `B_EI_NO_HOMEO` (Frozen weights): $Q_{\text{func}} = 0.212$, $R_{\text{global}} = 0.290$, $\text{CtxAcc} = 58.3\%$, $\text{PredMSE} = 0.3970$.
  - `D_EI_STDP` (STDP active): $Q_{\text{func}} = 0.200$, $R_{\text{global}} = 0.290$, $\text{CtxAcc} = 58.3\%$, $\text{PredMSE} = 0.3970$.
  - `E_EI_STDP_HOMEO` (STDP + Homeostasis): $Q_{\text{func}} = 0.169$, $R_{\text{global}} = 0.291$, $\text{CtxAcc} = 58.3\%$, $\text{PredMSE} = 0.3970$.
  - Ablation `ABL_06_DISABLE_STDP`: $Q_{\text{func}} = 0.203$, $\text{CtxAcc} = 58.3\%$.
* **Causal Diagnosis:** Corroborating the Phase 7 causal audit (`research/PHASE_7_CAUSAL_AUDIT.md`), classical pair-based STDP without reward modulation or behavioral feedback adjusts synaptic weights symmetrically around correlation peaks, acting largely as a passive homeostatic or stabilizing filter rather than driving representational or modular optimization.

### Negative Result 5: Traveling Spatial Waves Remain Weak Under Distance-Dependent Delays
* **Hypothesis Target:** Combining local E/I connectivity with axonal distance delays will produce coherent propagating wavefronts.
* **Empirical Finding:** Spatial phase gradients and wave coherence across 3D coordinates remained low ($<0.25$).
* **Causal Diagnosis:** Physical propagation delays in small-to-medium networks (100–1000 units) are too brief ($\Delta t \in [1, 5]\text{ ms}$) relative to membrane refractory periods ($2 - 4\text{ ms}$) to sustain stable macroscopic wavefronts. Activity dissipates into local damping by fast inhibitory feedback before forming macroscopic traveling planar or spiral waves.

### Negative Result 6: Homeostatic Synaptic Scaling Operates as a Slow Stabilizer, Not an Accuracy Driver
* **Hypothesis Target:** Homeostatic scaling directly optimizes functional community structure.
* **Empirical Finding:**
  - `B_EI_NO_HOMEO` vs `C_EI_HOMEO`: $Q_{\text{func}}$ changed from $0.212$ to $0.195$; Context accuracy was identical at $58.3\%$; Memory $R^2$ was $0.394$ vs $0.391$.
  - Ablation `ABL_02_NO_HOMEO`: Identical task performance to `C_EI_HOMEO`.
* **Causal Diagnosis:** Homeostasis strictly regulates long-term metabolic firing rate stability (preventing runaway saturation and silent extinction). As designed, it does not inspect task loss and does not directly boost memory or context disambiguation.

---

## 3. Summary of Negative Results Classification

| Domain | Expected Behavior | Actual Behavior | Classification |
|---|---|---|---|
| **Modularity** | $Q_{\text{func}} > 0.30$ | $Q_{\text{func}} \approx 0.17 - 0.23$ | `NOT SUPPORTED` (Falsified) |
| **Spike Irregularity** | $CV_{\text{ISI}} \ge 1.0$ (Poisson-like) | $CV_{\text{ISI}} \approx 0.12$ (Semi-regular) | `PARTIALLY SUPPORTED` |
| **STDP Benefit** | STDP improves functional modularity | STDP shows negligible difference from frozen ($p > 0.5$) | `NOT SUPPORTED` |
| **Context Recovery** | Full recovery to $>85\%$ | Rescued from 50% chance to 58.3%–62.5% | `PARTIALLY SUPPORTED` |
| **Macro Waves** | Coherent traveling wavefronts | Weak, locally damped spatial phase gradients | `NOT SUPPORTED` |

---

## 4. Methodological Implications for Phase 9

1. **Fluctuation-Driven Regimes:** To achieve $CV_{\text{ISI}} \ge 1.0$, networks must receive balanced stochastic Poisson conductance drive rather than steady somatic current injection.
2. **Beyond Pairwise STDP:** Synaptic plasticity must incorporate three-factor rules (neuromodulators / eligibility traces) or voltage-dependent triplet STDP to drive representational reorganization beyond passive stabilization.
3. **Dendritic Integration:** To resolve the high-frequency spike jitter that limits context disambiguation, dendritic compartmental models or slow NMDA-like synaptic conductances are required.
