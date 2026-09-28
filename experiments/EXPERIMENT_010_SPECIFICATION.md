# Experiment 010 Specification: Two-Compartment Neurons, Slow Conductances & Targeted Credit Assignment

## 1. Scientific Context & Motivation

Phase 9 revealed fundamental computational boundaries in point-neuron spiking substrates:
1. **The Credit Diffusion Problem:** In recurrent networks, an action potential at step $t$ triggers a cascade of recurrent spikes at $t+1, t+2, \dots$. By the time reward arrives at delayed step $t + \Delta t$, eligibility traces have accumulated across hundreds of unrelated recurrent synapses. A global scalar teaching signal reinforces the entire cascade indiscriminately, collapsing credit assignment to chance ($~50\%$).
2. **Absence of Working-Memory Attractors:** Without slow conductances or active dendritic compartments, transient sensory context vanishes from membrane potentials during delay intervals.
3. **Modularity Ceiling:** Scalar neuromodulation broadcast identically to all synapses fails to carve functional modularity beyond $Q_{\text{func}} \approx 0.25$.

Phase 10 directly investigates whether biologically grounded **dendritic compartmentalization**, **slow NMDA conductances**, and **spatially targeted credit assignment** can resolve these limitations.

---

## 2. Central Scientific Question

> Can a neuron with separate dendritic and somatic computation preserve contextual information over long delays and assign delayed reward/prediction-error credit to the relevant synapses without requiring global backpropagation through time?

---

## 3. Core Hypotheses

| Hypothesis | Description | Quantitative Success Criteria |
|:---|:---|:---|
| **H1: Compartmentalized Computation** | Two-compartment neurons exhibit computational dynamics measurably distinct from point-neuron models. | Statistically significant difference in representation dimensionality, context retention ($p < 0.05$), and firing rate distribution. |
| **H2: Slow Conductance Memory Retention** | Slow NMDA conductances ($\tau_{\text{NMDA}} \in [50, 250\text{ ms}]$) sustain persistent contextual representations across long inter-stimulus delays. | Context decoding $R^2 \ge 0.50$ at $\Delta t = 100$ steps; significantly higher than point-neuron baseline ($\Delta R^2 > 0.20$). |
| **H3: Targeted Credit Specificity** | Spatially targeted feedback alignment selectively reinforces causally relevant synapses over irrelevant recurrent pathways. | Positive Credit Specificity Index ($\text{CSI} > +0.30$), Credit Precision $> 65\%$, Credit Leakage $< 35\%$. |
| **H4: Persistent Working-Memory Attractors** | Network sustains context-specific stable attractor states without global hypersynchrony. | Attractor persistence duration $> 50$ steps, spontaneous switching rate $< 5\%$, Kuramoto $R_{\text{global}} < 0.50$. |
| **H5: Functional Modularity & Synchrony Safety** | Targeted dendritic plasticity promotes modular organization while preserving asynchronous-irregular stability. | $Q_{\text{func}} > 0.25$, $CV_{\text{ISI}} > 0.3$, $R_{\text{global}} \le 0.45$, strict Dale compliance ($80\%$ E, $20\%$ I). |

---

## 4. Definition of Predefined Metrics

To prevent post-hoc metric tuning, all evaluation metrics are formally defined here prior to experimental execution:

### 4.1 Credit Specificity Index (CSI)
$$\text{CSI} = \frac{\overline{|\Delta w|}_{\text{relevant}} - \overline{|\Delta w|}_{\text{irrelevant}}}{\overline{|\Delta w|}_{\text{relevant}} + \overline{|\Delta w|}_{\text{irrelevant}} + \epsilon}$$
- Range: $[-1.0, +1.0]$.
- $\text{CSI} > 0$: Plasticity is selectively concentrated in causally relevant synapses.
- $\text{CSI} \approx 0$: Diffuse, non-specific plasticity (credit diffusion).
- $\text{CSI} < 0$: Plasticity erroneously dominates irrelevant synapses (credit leakage).

### 4.2 Credit Precision and Leakage
$$\text{Precision} = \frac{\sum_{(i,j) \in \text{relevant}} |\Delta w_{ij}|}{\sum_{\text{all } (i,j)} |\Delta w_{ij}| + \epsilon}, \quad \text{Leakage} = 1.0 - \text{Precision}$$

### 4.3 Eligibility Concentration
$$\text{Gini}(|e|) = \frac{2 \sum_{k=1}^P k \cdot |e|_{(k)}}{P \sum_{k=1}^P |e|_{(k)}} - \frac{P+1}{P}$$
Measures the inequality of synaptic eligibility traces across all active synapses.

---

## 5. Experimental Matrix: 16 Model Conditions

### Controls
1. `COND_01_POINT_SPIKING_BASELINE`: Phase 9 three-factor point-neuron baseline with scalar RPE.
2. `COND_02_FROZEN_POINT_SPIKING`: Point-neuron network with frozen recurrent synapses.
3. `COND_03_TWO_COMPARTMENT_FROZEN`: Two-compartment neuron network with frozen recurrent synapses.

### Plasticity Variants
4. `COND_04_POINT_ELIGIBILITY`: Point-neuron coincidence eligibility with constant modulation.
5. `COND_05_DENDRITIC_ELIGIBILITY`: Two-compartment voltage-gated dendritic eligibility with constant modulation.
6. `COND_06_DENDRITIC_GLOBAL_REWARD`: Dendritic eligibility with global scalar reward broadcast.
7. `COND_07_DENDRITIC_PRED_ERROR`: Dendritic eligibility modulated by sensory prediction error.
8. `COND_08_DENDRITIC_RPE`: Dendritic eligibility modulated by signed reward prediction error.
9. `COND_09_DENDRITIC_TARGETED_SPATIAL`: Dendritic eligibility with Spatially Targeted Feedback Alignment.

### Mechanistic Ablations
10. `ABL_10_NO_SLOW_CONDUCTANCE`: Two-compartment network without NMDA conductance ($\tau_{\text{NMDA}} = 1.0\text{ ms}, \alpha = 0$).
11. `ABL_11_NO_DENDRITIC_NONLINEARITY`: Two-compartment network with linear dendrites ($g_{\text{nl}} = 0$).
12. `ABL_12_NO_COMPARTMENT_COUPLING`: Disconnected soma and dendrite ($g_c = 0$).
13. `ABL_13_RANDOMIZED_FEEDBACK`: Spatial feedback alignment weights randomly shuffled across coordinates.
14. `ABL_14_SHUFFLED_REWARD`: Scrambled reward delivery decorrelated from behavior.
15. `ABL_15_NO_ELIGIBILITY`: Synaptic eligibility traces forced to zero ($e_{ij} = 0$).

### Gradient Reference
16. `REF_16_BPTT_GRADIENT_BASELINE`: Compact Leaky RNN trained via exact analytical Backpropagation Through Time.

---

## 6. Delay Matrix

Reward delivery delay post-choice:
$$\Delta t \in [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000]$$

---

## 7. Memory & Working Memory Experiments

1. **Delayed Cue Recall:** Linear decoding $R^2(\text{context}, \mathbf{s}(t))$ at delays $\Delta t \in [10, 25, 50, 100, 250, 500, 1000]$.
2. **Context Disambiguation:** Accuracy and Fisher Discriminant Ratio (FDR) on ambiguous probe following inter-stimulus delay.
3. **Attractor Persistence:** Stability of internal latent representation following stimulus offset in the absence of external drive.

---

## 8. Biological Constraints & Safety Protocols

1. **Dale's Principle:** $80\%$ Excitatory ($w_{ij} \in [0, w_{\text{max}}^E]$), $20\%$ Fast-Spiking Inhibitory ($w_{ij} \in [-w_{\text{max}}^I, 0]$).
2. **Homeostasis:** Multiplicative synaptic scaling maintaining target firing rate $\sim 10\text{ Hz}$.
3. **Synchrony Safety:** Kuramoto order parameter $R_{\text{global}} < 0.50$ to strictly prevent epileptic runaway hypersynchrony.

---

## 9. Computational Scale Ladder

Test across:
$$N \in [100, 1000, 10000, 50000, 100000]$$
Measuring steps/sec, spikes/sec, synaptic operations/sec, memory overhead, and computational fairness.
