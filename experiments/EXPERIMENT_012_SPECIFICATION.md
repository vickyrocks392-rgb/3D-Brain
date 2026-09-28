# Experiment 012 Specification: Learnable Context-to-Action Interface & Causal Action Readout

## 1. Scientific Objective
Phase 11 proved that slow NMDA-like conductances, predictive dendritic disinhibition, and Dale-constrained recurrent attractor pools stabilize latent memory representations across delays ($13.7\times$ drift reduction, attractor separation ratio $0.963$). However, behavioral accuracy remained at chance ($24.2\text{--}25.0\%$), and credit leakage reached $86.7\%$.

The primary goal of Experiment 012 is to determine:
> **Can a learnable downstream context-to-action interface convert a stable latent attractor representation into causally correct action selection without requiring global BPTT?**

---

## 2. Mathematical & Algorithmic Definitions

### 2.1 Frozen Upstream Substrate
The recurrent two-compartment substrate from Phase 11 is held strictly frozen during learning ($\Delta W_{\text{rec}} = 0, \Delta W_{\text{pred}} = 0$):
* Somatic potential: $C_s \frac{dV_s}{dt} = -g_L (V_s - E_L) + g_c (V_d - V_s) + I_{\text{ext}} + I_{\text{rec}}$
* Dendritic potential: $C_d \frac{dV_d}{dt} = -g_L (V_d - E_L) + g_c (V_s - V_d) + I_{\text{syn}}^{\text{gated}} + I_{\text{nmda}}$
* Slow NMDA conductances: $\tau_{\text{nmda}} \frac{dg_{\text{nmda}}}{dt} = -g_{\text{nmda}} + s_{\text{pre}}(t)$
* Predictive gating: $G_j(t) = \sigma(W_{\text{pred}} \mathbf{s}(t) + b_{\text{pred}})$

### 2.2 Context-to-Action Interface
Synaptic transmission to competitive action accumulators:
$$z_j(t) = \sum_{i=1}^{N_{\text{ctx}}} W_{ij}(t) s_i^{\text{ctx}}(t) + b_j(t)$$

Action accumulator dynamics with lateral mutual inhibition:
$$\tau_{\text{act}} \frac{da_j}{dt} = -a_j(t) + z_j(t) - w_{\text{inh}} \sum_{k \ne j} a_k(t) + w_{\text{rec}} a_j(t)$$

Discrete integration:
$$a_j(t) = (1 - \alpha) a_j(t-1) + \alpha \left[ z_j(t) - w_{\text{inh}} \sum_{k \ne j} a_k(t-1) + w_{\text{rec}} a_j(t-1) \right]$$
Where $\alpha = \Delta t / \tau_{\text{act}}$.

Winner selection:
$$a^*(t) = \arg\max_j a_j(t)$$
$$\text{Margin}(t) = a_{\text{winner}}(t) - a_{\text{runner-up}}(t)$$

### 2.3 Plasticity Rules
1. **R1 (Linear Supervised):**
   $$\Delta W_{ij}(t) = \eta (y_j(t) - \hat{y}_j(t)) s_i^{\text{ctx}}(t)$$
2. **R2 (Local Hebbian with Oja Normalization):**
   $$\Delta W_{ij}(t) = \eta \left( s_i^{\text{ctx}}(t) a_j(t) - W_{ij}(t) a_j(t)^2 \right)$$
3. **R3 (Three-Factor Interface):**
   $$e_{ij}(t) = \lambda_e e_{ij}(t-1) + s_i^{\text{ctx}}(t) a_j(t)$$
   $$\Delta W_{ij}(t) = \eta M_{\text{action}}(t) e_{ij}(t)$$
4. **R4 (Prediction-Error Gated):**
   $$M_{\text{action}}(t) = (r(t) - \bar{r}) - \hat{V}(t)$$
   $$\Delta W_{ij}(t) = \eta M_{\text{action}}(t) e_{ij}(t)$$
5. **R5 (Delayed Reward + Synapse-Isolated Eligibility):**
   Eligibility updated continuously at the interface; scalar reward applied upon trial completion.
6. **R6 (Analytical BPTT Readout):**
   Supervised gradient descent reference bound on $W_{\text{ctx}\to\text{act}}$.
7. **R7 (Phase 11 Baseline):**
   Original static routing with fixed cross-inhibition ($W_{\text{cross}} = -0.5$).

---

## 3. Experimental Conditions Matrix (18 Conditions)

### Controls
* `COND_01_PHASE11_ROUTING_BASELINE`: Phase 11 fixed routing mechanism unchanged.
* `COND_02_FIXED_RANDOM_READOUT`: Frozen Gaussian random projection.

### Local Learning
* `COND_03_LINEAR_SUPERVISED_READOUT`: Linear delta-rule learning on interface weights.
* `COND_04_LOCAL_HEBBIAN`: Pre $\times$ Post Hebbian correlation with Oja normalization.
* `COND_05_THREE_FACTOR_INTERFACE`: Eligibility traces gated by local action reward.
* `COND_06_PRED_ERROR_GATED`: Reward prediction error modulatory gating.
* `COND_07_DELAYED_REWARD_ELIGIBILITY`: Scalar reward delayed with isolated eligibility.

### Reference Bound
* `REF_08_ANALYTICAL_BPTT_READOUT`: Analytical gradient descent on interface weights.

### Action Competition Ablations
* `ABL_09_NO_ACTION_INHIBITION`: $w_{\text{inh}} = 0.0$
* `ABL_10_WEAK_ACTION_INHIBITION`: $w_{\text{inh}} = 0.1$
* `ABL_11_STRONG_ACTION_INHIBITION`: $w_{\text{inh}} = 0.8$
* `ABL_12_NO_ACTION_RECURRENCE`: $w_{\text{rec}} = 0.0$

### Representation Source Variations
* `REP_13_SOMATIC_STATE`: Pre-synaptic context from somatic voltage $V_{\text{soma}}$.
* `REP_14_DENDRITIC_STATE`: Pre-synaptic context from dendritic voltage $V_{\text{dend}}$.
* `REP_15_NMDA_STATE`: Pre-synaptic context from slow NMDA conductances $g_{\text{nmda}}$.
* `REP_16_CONCATENATED_STATE`: Pre-synaptic context from concatenated multi-compartment state.
* `REP_17_PCA_COMPRESSED`: Pre-synaptic context compressed to $k=8$ principal components.
* `REP_18_RANDOM_PROJECTION`: Random projection control ($k=8$).

---

## 4. Diagnostic Benchmarks & Suites

### 4.1 Diagnostic Pure Context Decode
Tests whether the Phase 11 representation is linearly or nonlinearly separable without reward delay or credit assignment.
* Inputs: 60 trials (30 Context A, 30 Context B).
* Metrics: Centroid Distance ($D_{AB}$), Separation Ratio ($S$), Linear Probe Accuracy, Mutual Information ($I(\text{ctx}; \text{act})$).

### 4.2 Delay Sweeps
Evaluates temporal retention at reward delays:
$$\Delta t \in [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000]\text{ ms}$$

### 4.3 Causal Interventions Suite (C1–C6)
* `C1_CONTEXT_SCRAMBLE`: Permuting context dimensions.
* `C2_CONTEXT_SWAP`: Feeding Context A when Context B is expected.
* `C3_READOUT_LESION`: Ablating 25%, 50%, 75%, 100% of interface synapses.
* `C5_LABEL_PERMUTATION`: Permuting context $\to$ action mapping.
* `C6_FREEZE_PLASTICITY`: Disabling plasticity post-training.

### 4.4 Phase 11 Lesion Paradox Suite
Testing whether the Phase 11 lesion paradox stems from disinhibition:
* `INTACT`: Standard baseline.
* `RELEVANT_LESION`: Zeroing relevant attractor pool connections.
* `IRRELEVANT_LESION`: Zeroing irrelevant attractor pool connections.
* `BOTH_LESION`: Zeroing both attractor pool connections.
* `CROSS_INH_DISABLED`: Lateral cross-inhibition set to 0.
* `CROSS_INH_REVERSED`: Lateral cross-inhibition inverted to excitation.

### 4.5 Empirical Scaling Ladder
Evaluating computational efficiency across network scales:
$$N \in [30, 50, 100, 300, 1000, 10000]$$

---

## 5. Telemetry & Observability (BrainLog v3.0)

BrainLog v3.0 records all computational events without fabricated telemetry:
* `CONTEXT_READOUT`: Features extracted from substrate.
* `ACTION_EVIDENCE_ACCUMULATED`: Accumulator activation and margin.
* `ACTION_SELECTED`: Winning action choice.
* `ACTION_ERROR_COMPUTED`: Prediction error or supervised delta.
* `ACTION_ELIGIBILITY_UPDATED`: Synapse-isolated eligibility traces.
* `ACTION_CREDIT_ASSIGNED`: Causal credit allocation.
* `ACTION_SYNAPSE_UPDATED`: Weight matrix updates.
* `ACTION_READOUT_LESION`: Synaptic lesion events.
* `CONTEXT_INTERVENTION`: Counterfactual context manipulations.
* `ACTION_INTERFACE_STATE`: Full interface telemetry snapshot.
