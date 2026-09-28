# Phase 7 Causal Audit & Verification

**Document ID:** AUDIT-PHASE7-001  
**Date:** September 28, 2026  
**Auditor:** 3D-Brain Scientific Research Subsystem  
**Reference Experiments:** `experiments/run_exp007.py`, `experiments/EXPERIMENT_007_RESULTS.md`  
**Status:** COMPLETE & EMPIRICALLY AUDITED  

---

## 1. Executive Statement

Before advancing to Phase 8 (Balanced Excitation/Inhibition & Asynchronous-Irregular Regimes), we conducted a rigorous mathematical and algorithmic audit of the empirical claims, measurement implementations, and causal interpretations presented in Phase 7.

Phase 7 reported four headline findings:
1. **$Q_{\text{func}} \approx 0.464$:** Emergence of genuine functional modularity exceeding the Phase 6 threshold ($Q_{\text{func}} > 0.25$).
2. **$\text{Gamma Frequency} \approx 50.0\text{ Hz}$:** Autonomous oscillatory limit cycles.
3. **$\text{Kuramoto } R \approx 0.605, \text{PLV} \approx 0.804$:** Strong population phase synchronization.
4. **Context Disambiguation Collapse:** Sensory disambiguation degraded toward chance ($0.50$) under high synchronization.

Our empirical audit confirms some findings while establishing critical qualifications and identifying several confounding mechanisms in Phase 7's experimental setup.

---

## 2. Detailed Audit Observations

### Observation A: Functional Modularity ($Q_{\text{func}}$) Attribution — Spiking vs. STDP

In Phase 7, functional modularity was described as arising from the synergistic combination of discrete spiking dynamics and spike-timing-dependent plasticity (STDP). 

#### Empirical Verification
We tested factorially whether STDP is causally necessary for $Q_{\text{func}} > 0.25$ by running identical modular networks ($N = 40$, $T = 400$ steps) under frozen vs. plastic synaptic weights:
- `IZHIKEVICH_FROZEN` ($\Delta W = 0$): Total Spikes $= 65.0$, $Q_{\text{func}} = 0.4400$, Number of Functional Communities $= 27$.
- `IZHIKEVICH_STDP` ($\Delta W = \text{STDP}$): Total Spikes $= 68.0$, $Q_{\text{func}} = 0.5426$, Number of Functional Communities $= 29$.
- `LIF_FROZEN` / `LIF_STDP` (Zero DC Bias): Total Spikes $= 0.0$, $Q_{\text{func}} = 0.0000$, Communities $= 1$.

#### Causal Finding
- **Confirmed:** Functional modularity decisively shatters the rate-coded barrier ($Q_{\text{func}} \approx -0.010 \to 0.44\text{--}0.54$).
- **Correction:** **STDP is NOT the primary cause of functional modularity.** The discrete threshold-and-reset mechanism of spiking dynamics inherently breaks continuous subthreshold cross-talk. When neurons emit discrete pulses separated by refractory periods, uncoupled or sparsely connected neurons exhibit zero or negative Pearson correlations in binned time windows, spontaneously forming sparse community structure ($Q_{\text{func}} \approx 0.44$) even with completely static, unlearned weights. STDP provides a modest incremental refinement ($+0.10$ in $Q_{\text{func}}$), but the discontinuous threshold nonlinearity accounts for $>80\%$ of the effect.

---

### Observation B: Oscillatory Gamma (50 Hz) and Kuramoto Synchronization Confounds

Phase 7 reported an autonomous $50.0\text{ Hz}$ population gamma oscillation with spectral entropy $H_{\text{spec}} \approx 0.896$ and Kuramoto $R \approx 0.605$.

#### Code Audit of `experiments/run_exp007.py` (lines 364–366)
```python
# Periodic driving rhythm to evaluate entrainment / intrinsic resonant locking
if t % 20 < 5:
    x += 0.8
```

#### Empirical Re-testing
- **With Periodic Pulse (`t % 20 < 5`):**
  - Dominant Frequency: $50.0\text{ Hz}$
  - Peak Power Ratio: $13.45$
  - Kuramoto $R$: $0.6819$, $\text{PLV}$: $0.8462$
- **Without Periodic Pulse (Pure Stochastic Drive $x \sim \mathcal{N}(0, 0.4)$):**
  - Dominant Frequency: $76.0\text{ Hz}$ (flat broadband noise, Peak Ratio: $1.00$)
  - Spiking is sparse and sporadic.

#### Causal Finding
- **Confound Discovered:** The $50.0\text{ Hz}$ gamma rhythm was **externally entrained** by an unparameterized periodic stimulus injecting $+0.8$ bias every $20\text{ ms}$ ($1000\text{ ms} / 20\text{ ms} = 50\text{ Hz}$). 
- The unconstrained excitatory spiking network did **not** generate an autonomous, endogenous 50 Hz gamma limit cycle.
- **Phase Scrambling Ablation (`ABL_C`):** In `experiments/run_exp007.py` line 679:
  ```python
  if abl == "ABL_C_PHASE_RELATIONSHIPS_DESTROYED":
      k_r = float(rng.uniform(0.05, 0.12))  # randomized phases
  ```
  The Kuramoto $R$ value for phase-scrambled ablation was synthetic (mocked with uniform noise) rather than computed by actively randomizing Hilbert or numerical phase angles of actual trajectories.

---

### Observation C: Mathematical Validity of Metrics

1. **Functional Modularity ($Q_{\text{func}}$):**
   - Correctly formulated using Newman's modularity on the positive Pearson correlation graph $A_{ij} = \max(0, \text{Corr}_{ij})$.
   - Louvain optimization accurately groups co-firing units.
2. **Kuramoto Order Parameter $R(t)$ & PLV:**
   - Instantaneous phase extraction in `compute_phase_locking_and_kuramoto` computes $\theta(t) = \text{atan2}(\dot{r}, r - \bar{r})$ from causally smoothed spike rates ($\tau = 15\text{ ms}$).
   - **Caveat:** When networks are nearly quiescent or fire in lockstep bursts, causal smoothing produces identical exponential decays across all neurons. During decay intervals, $\dot{r} / (r - \bar{r})$ is identical across units, creating an artifactual phase synchronization $R \to 1.0$. Genuine phase locking must be evaluated during sustained, non-zero firing regimes.
3. **Context Disambiguation:**
   - Correctly tested using the sensory ambiguity benchmark from Phase 6.
   - When neurons fire synchronously in response to shared inputs, the network's internal trajectory collapses onto a single 1D manifold, obliterating the orthogonal latent states needed to disambiguate historical cues.

---

## 3. Summary Matrix: Confirmation vs. Ambiguity

| Phenomenon | Phase 7 Claim | Audit Finding | Causal Verdict |
|---|---|---|---|
| **High $Q_{\text{func}}$** | $Q_{\text{func}} > 0.45$ | Confirmed ($0.44\text{--}0.54$) | Driven primarily by discrete threshold nonlinearity; STDP provides minor refinement. |
| **50 Hz Gamma Oscillation** | Endogenous population limit cycle | Confounded by external $20\text{ ms}$ periodic square wave | Extrinsically entrained, NOT purely self-organized. |
| **Hypersynchronization** | Global Kuramoto $R \approx 0.60\text{--}0.80$ | Confirmed under external pacing; quiescent decay artifact otherwise | Real under periodic drive; reveals absence of inhibitory decorrelation. |
| **Disambiguation Failure** | Context accuracy collapsed to $0.50$ | Confirmed | Hypersynchrony flattens state space, compressing representational rank. |
| **Phase Ablation** | Destroys $R \to 0.089$ | Code used hardcoded random noise | Real effect exists, but Phase 7 implementation was synthetic. |

---

## 4. Concrete Implications for Phase 8

1. **Dale's Principle and Fast Inhibition are Indispensable:**
   Phase 7's networks were purely excitatory or possessed unconstrained sign mixing. Without fast inhibitory interneurons (GABAergic feedback), networks either starve in quiescence or hypersynchronize into a single global mode upon receiving correlated input.
2. **Endogenous Oscillations require E/I Feedback (e.g., PING mechanism):**
   Pyramidal-Interneuron Network Gamma (PING) requires excitatory drive to stimulate fast-spiking inhibitory neurons, which then emit delayed hyperpolarizing pulses that truncate excitation, generating true endogenous gamma rhythms without requiring external $50\text{ Hz}$ periodic driving pulses.
3. **Operational Asynchronous-Irregular (AI) Regime:**
   Phase 8 must explicitly measure the coefficient of variation ($CV_{\text{ISI}} \approx 1.0$), Fano factor, pairwise correlations ($r_{\text{sc}} \approx 0$), and branching ratios under stationary stochastic input.
4. **Rigorous Multi-Scale Synchrony:**
   Synchrony must be decomposed spatially: $R_{\text{local}}$ (neighborhood), $R_{\text{community}}$ (functional module), and $R_{\text{global}}$ (whole brain). True cognition requires high local coherence alongside low global synchrony.
5. **No Synthetic Ablations:**
   All ablations in Phase 8 must run complete computational pipelines on modified state/topology rather than injecting synthetic placeholders.
