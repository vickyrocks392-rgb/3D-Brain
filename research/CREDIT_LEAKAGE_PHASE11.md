# Credit Flow & Synaptic Leakage Analysis (Phase 11)

**Substrate Component:** `TargetedSpatialFeedbackEngine`  
**Core Metrics:** Credit Specificity Index (CSI), Credit Precision, Credit Leakage, Gini Concentration  
**Comparison Baseline:** Phase 10 Two-Compartment Targeted Plasticity ($70.8\%$ Leakage, $29.2\%$ Precision)  

---

## 1. Credit Allocation Problem in Recurrent Networks

In a two-compartment recurrent network, action selection depends on a cascade:
$$\text{Sensory Input} \longrightarrow \text{Dendritic NMDA} \longrightarrow \text{Attractor Recurrence} \longrightarrow \text{Action Accumulators}$$

When a delayed scalar reward $R(T)$ arrives at the end of a trial, plasticity must selectively reinforce only the subset of synapses $\mathcal{S}_{\text{causal}}$ that actively contributed to generating the rewarded action. Any plasticity applied to non-causal synapses $\mathcal{S}_{\text{irrel}}$ represents **Credit Leakage**.

---

## 2. Quantitative Credit Metrics Across Conditions

| Condition ID & Name | Credit Specificity Index (CSI) | Credit Precision (%) | Credit Leakage (%) | Gini Concentration |
|---|---|---|---|---|
| **COND_01: Phase 10 Baseline** | -0.379 | 15.1% | 84.9% | 0.382 |
| **COND_06: Shuffled Gate Ablation** | -0.447 | 13.3% | 86.7% | 0.354 |
| **COND_11: No Recurrent Inhibition** | -0.556 | 13.3% | 86.7% | 0.312 |
| **COND_14: Global Scalar Reward** | -0.467 | 11.8% | 88.2% | 0.301 |
| **COND_15: Targeted Feedback (Phase 10)** | -0.379 | 15.1% | 84.9% | 0.382 |
| **COND_16: Targeted Feedback + Gate** | -0.379 | 15.1% | 84.9% | 0.381 |
| **COND_17: Full Architecture (Gate + Attractor)** | **-0.447** | **13.3%** | **86.7%** | **0.360** |

---

## 3. Findings & Mechanistic Insights

1. **Credit Leakage Did Not Decrease:** In Phase 11, credit leakage remained elevated ($86.7\%$ vs $70.8\%$ in Phase 10). Credit precision was $13.3\%$.
2. **Impact of Recurrent Collaterals on Trace Dispersion:** Why did adding recurrent attractors increase leakage?
   In Phase 10, the network was purely feedforward, so eligibility traces decayed monotonically during the delay. In Phase 11, recurrent excitation maintains persistent spiking in the attractor pool throughout the delay. Because these neurons continue spiking, their local eligibility traces:
   $$e_{ij}(t) = (1 - \alpha_e) e_{ij}(t-1) + \alpha_e \cdot v_{\text{dend},i}(t) \cdot s_{\text{soma},j}(t)$$
   remain active across the entire recurrent collateral matrix. When the reward vector arrives, it updates hundreds of collateral synapses that were active merely to maintain baseline spiking, not to encode context.
3. **Targeted Spatial Feedback vs Global Reward:** Targeted spatial feedback (Condition 15/17) outperformed global scalar reward (Condition 14: Precision $= 11.8\%$, Leakage $= 88.2\%$). However, spatial alignment alone cannot prevent temporal trace dispersion across recurrent collaterals.

---

## 4. Why BPTT Succeeds Where Local Credit Fails

| Characteristic | Local Three-Factor Plasticity (Phase 11) | Analytical BPTT Baseline (Ref 18) |
|---|---|---|
| **Gradient Flow** | Forward-only eligibility traces | Exact temporal adjoint backpropagation |
| **Synaptic Credit** | Spatially dispersed ($86.7\%$ leakage) | Exact Jacobian credit ($0\%$ leakage) |
| **Choice Accuracy** | 24.2% – 25.0% | **58.3% – 98.8%** |
| **Biological Plausibility** | High (local 3-factor, Dale-compliant) | Non-biological (symmetric backward weights) |

This comparison highlights the fundamental challenge of biological credit assignment: **recurrent self-excitation keeps synapses eligible for reward, but cannot distinguish sustaining background activity from causal task-relevant activity.**
