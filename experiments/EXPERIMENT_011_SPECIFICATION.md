# Experiment 011 Specification: Predictive Dendritic Disinhibition, Recurrent Attractor Latching & Selective Action Gating

## 1. Scientific Objective
Determine whether combining biologically grounded **predictive dendritic disinhibition**, **recurrent attractor latching**, and **selective action routing** can:
1. Selectively route preserved slow NMDA contextual representations into action-selection circuits.
2. Form stable, distinguishable population attractors (Attractor Pool A vs. Attractor Pool B) during sensory deprivation with suppressed state drift.
3. Substantially reduce the Phase 10 credit-leakage bottleneck ($70.8\%$) and negative global reward drift.
4. Improve delayed contextual choice accuracy beyond the Phase 10 baseline ($43.8\%$).

---

## 2. Mathematical & Algorithmic Definitions

### 2.1 Predictive Dendritic Disinhibition
Dendritic inhibitory synaptic current is modulated by endogenous predictive state $G_j(t) \in [0, 1]$:
$$I_{\text{syn}, j}^{I, \text{gated}}(t) = I_{\text{syn}, j}^I(t) \cdot \left(1.0 - \gamma_{\text{disinhib}} \cdot G_j(t)\right)$$
Where:
* $I_{\text{syn}, j}^I \le 0.0$ (hyperpolarizing inhibitory current).
* $\gamma_{\text{disinhib}} = 0.85$ (85% disinhibition of target dendritic compartment).
* $G_j(t) = \sigma(W_{\text{pred}} \mathbf{s}_{\text{attractor}} + b_{\text{pred}})$.

**Gate Selectivity Metric:**
$$\text{Selectivity}_{\text{gate}} = \frac{\langle G \rangle_{\text{relevant}} - \langle G \rangle_{\text{irrelevant}}}{\langle G \rangle_{\text{relevant}} + \langle G \rangle_{\text{irrelevant}} + \epsilon}$$

### 2.2 Recurrent Attractor Latching
Recurrent connectivity $W_{\text{rec}}[\text{post}, \text{pre}]$ with Dale's constraints ($80\%$ E, $20\%$ I):
* Intra-pool excitation: $W_{AA}^{EE} > 0, W_{BB}^{EE} > 0$.
* Inter-pool cross-inhibition: $W_{AB}^{EI} \to W_{BI}^{IE} \le 0$.
* Homeostatic synaptic scaling maintains target firing rates without runaway excitation.

**Attractor Diagnostics:**
* Autonomous Drift: $\frac{1}{T-1} \sum_{t=1}^T \|\mathbf{s}(t) - \mathbf{s}(t-1)\|_2$
* Attractor Radius: $\sigma_p = \frac{1}{T} \sum_{t=1}^T \|\mathbf{s}_p(t) - \mathbf{\mu}_p\|_2$
* Centroid Distance: $D_{AB} = \|\mathbf{\mu}_A - \mathbf{\mu}_B\|_2$
* Attractor Separation Ratio: $S_{\text{attractor}} = \frac{D_{AB}}{\sigma_A + \sigma_B + \epsilon}$
* Spontaneous Switching Rate: rate of crossing between attractor basins during deprivation.

### 2.3 Action Routing & Causal Lesions
Action activations $\mathbf{v}_{\text{act}}$ evolve via leaky integration with mutual inhibition:
$$\tau_{\text{act}} \frac{d\mathbf{v}_{\text{act}}}{dt} = -\mathbf{v}_{\text{act}} + W_{\text{route}} \mathbf{s}_{\text{attractor}} - W_{\text{mutual}} \mathbf{v}_{\text{act}}$$
* **Action Selectivity:** $\frac{v_{\text{act}, \text{target}} - v_{\text{act}, \text{other}}}{|v_{\text{act}, \text{target}}| + |v_{\text{act}, \text{other}}| + \epsilon}$
* **Mutual Information:** $I(\text{Context}; \text{Action})$
* **Causal Lesion Test:**
  - Relevant Pathway Lesion: $W_{\text{route}}[\text{rel}, :] \leftarrow 0$
  - Matched Irrelevant Pathway Lesion: $W_{\text{route}}[\text{irrel}, :] \leftarrow 0$

---

## 3. Required 18 Model Conditions Matrix

### Controls & Baseline
1. `COND_01_PHASE10_TWO_COMP_BASELINE`: Phase 10 standard two-compartment model without gating or structured recurrence.
2. `COND_02_TWO_COMP_NMDA_ONLY`: Two-compartment model with slow NMDA conductance only.
3. `COND_03_TWO_COMP_RECURRENT_ATTRACTOR`: Two-compartment + recurrent attractor latching (no predictive gating).
4. `COND_04_TWO_COMP_PREDICTIVE_GATING`: Two-compartment + predictive gating (no recurrent attractor latching).
5. `COND_05_TWO_COMP_GATING_AND_ATTRACTOR`: Complete Phase 11 substrate with gating and attractor latching.

### Gating Causal Ablations
6. `ABL_06_GATE_SHUFFLED`: Gating vector randomly permuted across neurons at each step.
7. `ABL_07_GATE_DELAYED`: Gating vector delayed by 5 simulation steps.
8. `ABL_08_GATE_REVERSED`: Gating vector inverted ($1.0 - G$).
9. `ABL_09_NO_DISINHIBITION`: Predictive gating factor clamped to zero (passive full inhibition).

### Attractor Causal Ablations
10. `ABL_10_NO_RECURRENT_EXCITATION`: Recurrent excitatory weights clamped to zero.
11. `ABL_11_NO_RECURRENT_INHIBITION`: Recurrent cross-pool inhibition clamped to zero.
12. `ABL_12_REDUCED_RECURRENT_COUPLING`: Recurrent weights attenuated by 80%.
13. `ABL_13_EXCESSIVE_RECURRENT_COUPLING`: Recurrent weights amplified by 300%.

### Credit Assignment Variants
14. `COND_14_GLOBAL_REWARD`: Phase 9-style global reward broadcast.
15. `COND_15_PHASE10_TARGETED_FEEDBACK`: Phase 10 targeted feedback without gating or attractor latching.
16. `COND_16_TARGETED_FEEDBACK_AND_GATE`: Targeted feedback + predictive dendritic gating.
17. `COND_17_TARGETED_FEEDBACK_GATE_ATTRACTOR`: Targeted feedback + predictive gating + recurrent attractor latching.

### Reference Upper Bound
18. `REF_18_BPTT_GRADIENT_BASELINE`: Non-biological exact analytical BPTT baseline.

---

## 4. Evaluation Protocol
* Multiple independent seeds: `[42, 43, 44, 45]`
* Delays: $\Delta t \in \{10, 25, 50, 100, 250, 500, 1000, 2000\}\text{ ms}$
* Continual Learning: Stationary, Abrupt Reversal, Gradual Drift, Recurring Context
* Scale Ladder: $100, 1\text{K}, 10\text{K}, 50\text{K}, 100\text{K}$ neurons
* Output Artifacts:
  - `experiments/results/exp011_metrics.json`
  - `experiments/results/brainlogs/EXP011_predictive_gating_attractor.brainlog` (Schema v2.9)
