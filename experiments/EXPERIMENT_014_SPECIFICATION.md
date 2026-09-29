# Experiment 014 Specification: Temporal Neural Coding, Burst Dynamics, Dendritic Coincidence & Population Resonance

## 1. Central Scientific Question
> **Can temporally structured spike coding, burst dynamics, dendritic coincidence detection, and population-level resonance preserve substantially more contextual information during the continuous → spike → action transformation than Phase 13's point-process coding?**

Phase 13 quantitatively isolated the exact locus of information destruction:
- Continuous contextual state: $1.000\text{ bit}$ MI, $100\%$ linear probe accuracy
- Somatic subthreshold state: $1.000\text{ bit}$ MI, $100\%$ linear probe accuracy
- Instantaneous binary spike state: $0.000\text{ bit}$ MI, $50\%$ chance linear probe accuracy
- Downstream persistent action state: $0.0817\text{ bit}$ MI, $66.7\%$ linear probe accuracy

Experiment 014 investigates whether biologically grounded temporal coding, structured burst events, dendritic calcium coincidence, and population resonance can bridge this continuous-to-discrete gap and preserve contextual representation through downstream action execution.

---

## 2. Experimental Families (Conditions A–L)

Evaluated across deterministic seeds `42, 43, 44` on identical delayed contextual decision tasks:

| Family | Condition Name | Core Mechanism |
|---|---|---|
| **A** | `COND_A_PHASE13_BASELINE` | Point-process spike encoding baseline (S9) |
| **B** | `COND_B_BURST_CODING` | Plateau & calcium-triggered burst dynamics (B7) |
| **C** | `COND_C_CALCIUM_COINCIDENCE` | Dendritic Ca2+ coincidence detector ($\Delta t = 10\text{ ms}$) |
| **D** | `COND_D_TEMPORAL_CODING` | Multi-step temporal sequence and phase coding (T8) |
| **E** | `COND_E_POPULATION_CODING` | Ensemble population pooling with heterogeneous thresholds |
| **F** | `COND_F_BURST_PLUS_TEMPORAL` | Combined burst structure and relative latency coding |
| **G** | `COND_G_BURST_PLUS_POPULATION` | Bursting ensembles pooled across action populations |
| **H** | `COND_H_BURST_PLUS_CALCIUM` | Dendritic coincidence driving burst discharge |
| **I** | `COND_I_BURST_CALCIUM_POPULATION` | Coincidence-driven bursting with population pooling |
| **J** | `COND_J_LOCAL_RESONANCE` | Local community oscillatory assemblies ($R_{\text{local}}$) |
| **K** | `COND_K_GLOBAL_RESONANCE` | Global network-wide phase synchronization ($R_{\text{global}}$) |
| **L** | `COND_L_INTEGRATED_CANDIDATE` | Integrated Phase 14 architecture combining all verified mechanisms |

---

## 3. Parametric Sweeps & Robustness Batteries

### 3.1 Delay Retention Sweeps
Evaluates contextual choice retention across delays:
$$\Delta t \in [0, 50, 100, 250, 500, 1000, 2000]\text{ ms}$$

### 3.2 Temporal Integration Window Sweep
Measures information recovery as a function of the temporal decoding horizon:
$$W \in [1, 2, 5, 10, 20, 50, 100]\text{ steps}$$

### 3.3 Dendritic Coincidence Window Sweep
Evaluates the optimal integration window for spatio-temporal coincidence:
$$\Delta t \in [0, 1, 2, 5, 10, 20, 50]\text{ ms}$$

### 3.4 Temporal Jitter Robustness
Injects controlled temporal noise into spike rasters:
$$\text{Jitter} \in [\pm 1\%, \pm 5\%, \pm 10\%, \pm 20\%, \pm 50\%]$$

### 3.5 Spike Dropout Robustness
Randomly ablates action potentials to test resilience against stochastic channel failures:
$$\text{Dropout} \in [0\%, 5\%, 10\%, 20\%, 30\%, 50\%]$$

### 3.6 Causal Interventions Suite (C1–C12)
Selectively disrupts individual biological components to confirm causality:
- `C1_REMOVE_BURST`: Converts all bursts to single point-process spikes
- `C2_REMOVE_COINCIDENCE`: Disables dendritic supralinear coincidence gating
- `C3_REMOVE_NMDA`: Sets slow NMDA conductance $g_{\text{NMDA}} = 0$
- `C4_REMOVE_TEMPORAL`: Constrains temporal window to $W = 1$ step
- `C5_SHUFFLE_TIMING`: Shuffles spike timing while preserving spike counts
- `C6_SHUFFLE_BURSTS`: Scrambles burst structure (ISIs and lengths)
- `C7_SHUFFLE_COUNTS`: Permutes spike counts across action pools
- `C8_DESTROY_PHASE`: Randomizes population phase alignment
- `C9_DESTROY_POPULATION`: Randomly reassigns neuron pool affiliations
- `C10_DESTROY_SPATIAL`: Shuffles 3D spatial distance coordinates
- `C11_RANDOMIZE_RESONANCE`: Randomizes coupling weights and frequencies
- `C12_FREEZE_INTERFACE`: Disables interface synaptic plasticity

---

## 4. 11-Stage Information Bottleneck Audit
Measures mutual information $I(X; Y)$, linear probe accuracy, nonlinear probe accuracy, separation ratio, and effective dimensionality (Participation Ratio $PR$) across all 11 anatomical stages:
$$\mathbf{c} \to \mathbf{V}_{\text{dend}} \to \mathbf{g}_{\text{NMDA}} \to [\text{Ca}^{2+}] \to \mathbf{V}_{\text{soma}} \to (\boldsymbol{\theta}_{\text{eff}}, \mathbf{G}) \to \mathbf{s}_{\text{spike}} \to \mathbf{b}_{\text{burst}} \to \mathbf{r}_{\text{temporal}} \to \mathbf{a}_{\text{pop}} \to \mathbf{a}_{\text{persist}}$$

---

## 5. Reference Architecture Controls
- Phase 13 Point-Process Baseline
- Continuous Subthreshold Oracle (Information ceiling)
- Analytical BPTT Reference
- Small Causal Transformer Reference (2 layers, 4 heads)
- Shuffled Temporal Control
- Random Population Control
