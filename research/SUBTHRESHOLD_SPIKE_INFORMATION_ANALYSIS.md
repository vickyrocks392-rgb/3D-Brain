# Subthreshold-to-Spike Information Analysis: Quantifying Information Loss Along the Neural Pathway

**Phase:** Phase 13  
**Artifact:** Research Analysis  
**Central Focus:** Formal Information Theory Audit of the Continuous-to-Discrete Transition  

---

## 1. Problem Formulation & Multi-Stage Instrumentation

Phase 12 established that upstream recurrent dynamics contain highly preserved contextual information, but identified the "Subthreshold-to-Spike Rate Encoding Gap" as the primary bottleneck in autonomous action routing.

In Phase 13, we implemented the `BiophysicalPathwayAuditor` (`brain/biophysical_audit.py`) to quantify information loss at every step of the biophysical chain:

$$\text{Context Vector } \mathbf{c} \to \mathbf{V}_{\text{dend}} \to \mathbf{g}_{\text{NMDA}} \to \mathbf{V}_{\text{soma}} \to \boldsymbol{\theta}_{\text{adapt}} \to \mathbf{s}_{\text{spike}} \to \mathbf{r}_{\text{pop}} \to \mathbf{a}_{\text{persist}}$$

At each stage $k$, we measured:
1. **Geometric Separation Ratio:** $R_{\text{sep}} = \frac{\text{Between-Class Variance}}{\text{Within-Class Variance}}$
2. **Linear Probe Accuracy:** Accuracy of a linear Ridge classifier predicting context class $y \in \{0, 1\}$.
3. **Nonlinear Probe Accuracy:** Accuracy of an MLP probe with RBF/tanh kernels.
4. **Mutual Information:** $I(X_k; Y)$ in bits (where maximum binary entropy is $1.00\text{ bit}$).
5. **Incremental Information Loss:** $\Delta I_k = I(X_{k-1}; Y) - I(X_k; Y)$.

---

## 2. Quantitative Pathway Audit Results

The audit was executed across all trials of Experiment 013:

| Stage $k$ | Anatomical / Biophysical Stage | Dimension | Centroid Distance | Separation Ratio | Linear Probe Acc | Nonlinear Probe Acc | Mutual Info $I(X; Y)$ | Incremental Loss |
|---|---|---|---|---|---|---|---|---|
| **1** | Context Cue Vector | 40 | 19.23 | 1.432 | $100.0\%$ | $100.0\%$ | **$1.000\text{ bits}$** | $0.000\text{ b}$ |
| **2** | Dendritic Potential ($V_{\text{dend}}$) | 40 | 31.38 | 1.519 | $100.0\%$ | $100.0\%$ | **$1.000\text{ bits}$** | $0.000\text{ b}$ |
| **3** | Slow NMDA Conductance ($g_{\text{NMDA}}$) | 40 | 2.56 | 622.63 | $100.0\%$ | $100.0\%$ | **$1.000\text{ bits}$** | $0.000\text{ b}$ |
| **4** | Somatic Integration ($V_{\text{soma}}$) | 40 | 19.23 | 1.432 | $100.0\%$ | $100.0\%$ | **$1.000\text{ bits}$** | $0.000\text{ b}$ |
| **5** | Axon Initial Segment ($\theta_{\text{adapt}}$) | 32 | 28.91 | 0.010 | $50.0\%$ | $46.7\%$ | **$0.000\text{ bits}$** | **$1.000\text{ b}$** |
| **6** | Instantaneous Spike Raster ($\mathbf{s}$) | 32 | 0.44 | 0.006 | $50.0\%$ | $60.0\%$ | **$0.000\text{ bits}$** | $0.000\text{ b}$ |
| **7** | Action Population Rates ($\mathbf{r}_{\text{pop}}$) | 2 | 65.74 | 0.010 | $60.0\%$ | $60.0\%$ | **$0.029\text{ bits}$** | $-0.029\text{ b}$ |
| **8** | Persistent Action State ($\mathbf{a}$) | 2 | 0.06 | 0.008 | $66.7\%$ | $66.7\%$ | **$0.082\text{ bits}$** | $-0.053\text{ b}$ |

---

## 3. Key Theoretical & Empirical Findings

### 3.1 Where Is Information Lost?
- **Stages 1 through 4 (Continuous Domain):** Mutual information is **completely conserved** ($I = 1.000\text{ bits}$, Probe Accuracy $= 100.0\%$). Dendritic integration, NMDA non-linear conductances, and somatic membrane potentials preserve the context representation without degradation.
- **Stage 5 & 6 (The Discrete Discretization Cliff):** The instantaneous discretization of somatic potentials into binary spikes ($s_i(t) \in \{0, 1\}$) at a single simulation time step drops mutual information from $1.000\text{ bits}$ to $0.000\text{ bits}$, causing single-step linear probe accuracy to drop to chance ($50.0\%$).
- **Stages 7 & 8 (Downstream Re-aggregation):** When downstream action populations integrate these spikes over time and across neural pools, mutual information begins to recover:
  - Action population firing rates: $I = 0.029\text{ bits}$ ($60.0\%$ probe accuracy).
  - Persistent recurrent action state: $I = 0.082\text{ bits}$ ($66.7\%$ probe accuracy).

### 3.2 Information Recovery Mechanism
Why does downstream action state show higher mutual information than instantaneous spikes?
Because the action persistence attractor acts as an **integrator of spike volleys over a finite temporal horizon**. Single binary spikes are noisy point-process samples; the recurrent attractor filters high-frequency shot noise and reconstructs the subthreshold manifold evidence.

---

## 4. Scientific Implication for Spiking Architectures
The central lesson of Phase 13's information audit is:
**A spiking neural network cannot make reliable autonomous decisions based on single-step instantaneous spike readouts.**
Information preservation across the subthreshold-to-spike boundary requires:
1. **Temporal phase coding** rather than instantaneous point evaluation.
2. **Population ensembles** to average out stochastic discretization noise.
3. **Recurrent attractor persistence** to integrate evidence over time.
