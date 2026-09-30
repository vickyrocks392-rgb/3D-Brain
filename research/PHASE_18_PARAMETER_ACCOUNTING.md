# Phase 18 — Parameter Accounting & Capacity Audit Report

**Audit Target:** Model Complexity, Trainable vs Fixed Parameters, State Variable Census  
**Date:** September 30, 2026  
**Status:** **AUDITED — COMPLETE CENSUS DOCUMENTED & FAIR BASELINES ESTABLISHED**

---

### Executive Summary

Phase 17 highlighted that **MIN_E requires only $\approx 80$ trainable parameters**, contrasting this favorably against classical baselines (BPTT: ~4,200 parameters; Transformer: ~18,500 parameters).

Our Phase 18 audit performed an exhaustive census of all parameters, fixed weights, architectural constants, and dynamic state variables. **We found that the "80 parameters" figure counts exclusively the plastic readout synapses connecting the recurrent substrate to the action outputs.** It completely omits $1,920$ fixed upstream synaptic connections (of which $851$ are active and non-zero), $16$ biophysical structural constants, and $282$ dynamic continuous state variables. When compared fairly against a Tiny RNN control (which has $96$ total parameters and $8$ state variables), MIN_E actually represents a substantially more complex dynamical system.

---

### 1. Exhaustive Parameter Census of MIN_E

#### 1.1 Trainable / Plastic Parameters
- Readout Synaptic Weights: $W_{\text{readout}} \in \mathbb{R}^{40 \times 2} = \mathbf{80}$ parameters.
- Readout Action Biases: $b_{\text{readout}} \in \mathbb{R}^2 = \mathbf{2}$ parameters.
- **Total Trainable Parameters:** **82** (reported as 80 in Phase 17).

#### 1.2 Fixed Synaptic Connections (Upstream Biophysical Substrate)
- Sensory Input Synapses: $W_{\text{in}} \in \mathbb{R}^{40 \times 8} = \mathbf{320}$ connections.
- Recurrent Collateral Synapses: $W_{\text{rec}} \in \mathbb{R}^{40 \times 40} = \mathbf{1,600}$ connections.
- **Total Fixed Synaptic Weights:** **1,920** (with 851 non-zero active connections under topology sparsity).

#### 1.3 Structural & Biophysical Hyper-Parameters (Configured Constants)
1. $\tau_{\text{soma}}$ (somatic membrane time constant: $20.0\text{ ms}$)
2. $\tau_{\text{dend}}$ (dendritic membrane time constant: $35.0\text{ ms}$)
3. $g_{\text{coupling}}$ (inter-compartmental electrotonic coupling: $0.40$)
4. $V_{\text{rest}}$ (resting membrane potential: $-70.0\text{ mV}$)
5. $V_{\text{thresh}}$ (somatic action potential threshold: $-50.0\text{ mV}$)
6. $V_{\text{reset}}$ (post-spike reset potential: $-75.0\text{ mV}$)
7. $\tau_{\text{nmda}}$ (NMDA unblock decay time constant: $100.0\text{ ms}$)
8. $\alpha_{\text{nmda}}$ (NMDA channel unblock sensitivity: $0.15$)
9. $\theta_{\text{nmda}}$ (dendritic NMDA activation threshold: $-45.0\text{ mV}$)
10. $\tau_{\text{calcium}}$ (dendritic calcium accumulation time constant: $50.0\text{ ms}$)
11. $\tau_{\text{slow\_assembly}}$ (slow action assembly timescale: $1500.0\text{ ms}$)
12. $w_{\text{self}}$ (self-excitation parameter: $0.20$)
13. $w_{\text{cross\_inh}}$ (mutual lateral inhibition weight: $0.50$)
14. $\eta_{\text{readout}}$ (dendritic learning rate: $0.015$)
15. $\tau_{\text{eligibility}}$ (synaptic eligibility trace decay: $300.0\text{ ms}$)
16. $w_{\text{min}}, w_{\text{max}}$ (synaptic saturation limits: $[-2.0, +2.0]$)
- **Total Structural Constants:** **16** configured constants.

#### 1.4 Active State Variables (Dynamic Memory Footprint per Step)
- Somatic membrane potentials $V_{\text{soma}}$: $40$
- Dendritic potentials $V_{\text{dend}}$: $40$
- NMDA conductances $g_{\text{nmda}}$: $40$
- Primary eligibility traces $e_{jk}$: $40 \times 2 = 80$
- Dendritic eligibility traces $e^{\text{dend}}_{jk}$: $40 \times 2 = 80$
- Assembly continuous activations (Fast, Med, Slow): $2$
- **Total Dynamic State Variables:** **282** real-valued variables.

---

### 2. Fair Comparison Across Architectures

| Model Architecture | Trainable Weights | Fixed Synaptic Weights | Structural Hyper-Params | Dynamic State Variables | Total Parameter Capacity |
|:---|:---:|:---:|:---:|:---:|:---:|
| **MIN_E (Full Substrate)** | **82** | **1,920** | **16** | **282** | **2,018** |
| **MIN_E (Readout Only)** | 82 | *(omitted)* | *(omitted)* | *(omitted)* | 82 (misleading) |
| **Matched Leaky Integrator** | 2 | 0 | 2 | 1 | **5** |
| **Tiny RNN Control (h=6)** | 96 | 0 | 2 | 6 | **104** |
| **BPTT Recurrent Baseline** | 4,200 | 0 | 4 | 100 | **4,304** |
| **Causal Transformer Control** | 18,500 | 0 | 6 | 256 | **18,762** |

---

### 3. Scientific Implications

1. **Capacity Attribution:**
   MIN_E's computational behavior cannot be attributed purely to its 82 trainable weights. The $1,920$ fixed upstream synapses (pre-configured with distance-dependent spatial delays and structured E/I connectivity) and the $282$ persistent state variables perform the heavy lifting of temporal integration and spatial separation.

2. **Fair Accounting Recommendation:**
   Scientific reports must distinguish between **plastic/trainable capacity** (82 parameters) and **effective architectural complexity** (2,018 parameters, 282 state variables). Comparing MIN_E's 82 plastic weights directly against an end-to-end trained RNN's total weights without acknowledging the fixed biophysical substrate constitutes an asymmetric comparison.
