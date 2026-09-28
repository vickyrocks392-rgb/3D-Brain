# Experiment 008 Specification: Balanced Excitation/Inhibition & Asynchronous-Irregular Regimes

**Project:** 3D-Brain Research Program  
**Phase:** Phase 8  
**Experiment ID:** EXP-008  
**Date:** September 2026  
**Status:** APPROVED FOR EXECUTION  
**Lead Investigators:** 3D-Brain Research Team  

---

## 1. Scientific Objective

The central objective of Experiment 008 is to resolve the fundamental problem discovered in Phase 7:
> **Can biologically constrained excitation/inhibition dynamics (Dale's principle: ~80% excitatory / ~20% fast-spiking inhibitory interneurons) and homeostatic synaptic scaling prevent pathological global hypersynchronization, entering the Asynchronous-Irregular (AI) regime, while simultaneously preserving functional modularity ($Q_{\text{func}} > 0.30$), persistent latent memory, multi-step prediction, and restoring fine sensory context disambiguation?**

In Phase 7, spiking dynamics crossed the functional modularity barrier ($Q_{\text{func}} \approx 0.464$), but exhibited excessive global phase synchronization ($R \approx 0.605$, $PLV \approx 0.804$). This compressed state-space representations and degraded temporal context disambiguation toward chance.

Experiment 008 tests whether:
$$\text{Local Coherence} + \text{Mesoscale Organization} + \text{Global Asynchronous-Irregular Activity}$$
can emerge and restore representational degrees of freedom without destroying persistent memory or prediction.

---

## 2. Hypotheses & Falsification Criteria

| Hypothesis | Description | Success Criterion | Falsification Condition |
|---|---|---|---|
| **H1: Dale's Principle** | Segregating neurons into fixed excitatory (~80%) and inhibitory (~20%) populations produces qualitatively different dynamics than unconstrained signed networks. | Distinct $I_E$ and $I_I$ current balance, stabilized firing rates, non-zero $CV_{\text{ISI}}$, stable population activity. | Indistinguishable from unconstrained signed networks or complete network silence/explosion. |
| **H2: Inhibition Prevents Hypersynchrony** | Fast-spiking inhibitory interneurons break pathological global phase locking while maintaining computational activity. | Global Kuramoto $R_{\text{global}} < 0.45$ and $PLV_{\text{global}} < 0.50$ while mean firing rate remains active ($2\text{--}30\text{ Hz}$). | Network remains hypersynchronized ($R_{\text{global}} \ge 0.60$) or inhibition silences network ($< 0.1\text{ Hz}$). |
| **H3: Asynchronous-Irregular (AI) Regime** | Balanced recurrent drive produces an operational AI state. | $CV_{\text{ISI}} \ge 0.70$, pairwise spike-count correlation $|r_{\text{sc}}| < 0.15$, Fano factor $FF \approx 1.0\pm 0.5$, AI score $\ge 0.60$. | Regular periodic firing ($CV_{\text{ISI}} < 0.30$), high correlation ($r_{\text{sc}} > 0.30$), or complete quiescence. |
| **H4: Local Synchrony vs Global Asynchrony** | Synchrony varies across spatial scales: structured local coordination coexisting with global decorrelation. | $R_{\text{local}} > R_{\text{global}} + 0.10$ and $PLV_{\text{local}} > PLV_{\text{global}} + 0.15$. | Spatial synchrony profile is flat ($R_{\text{local}} \approx R_{\text{global}}$). |
| **H5: Functional Modularity Preservation** | E/I balance maintains functional modularity without runaway coherence. | Functional modularity $Q_{\text{func}} \ge 0.30$ derived from binned spike counts. | $Q_{\text{func}} < 0.15$. |
| **H6: Context Disambiguation Recovery** | Breaking global hypersynchrony restores fine sensory/contextual representation. | Context classification accuracy $> 70\%$ (statistically superior to Phase 7 hypersynchronized control). | Context classification remains near chance ($\le 55\%$) or degrades further. |
| **H7: Persistent Memory Preservation** | E/I balance does not destroy persistent latent memory mechanisms. | Delayed cue recall $R^2 \ge 0.60$ for $\Delta t = 50, 100, 250, 500$ steps. | $R^2 < 0.20$ across delay horizons. |
| **H8: Predictive Computation Preservation** | Multi-step forward prediction is preserved or improved under E/I balance. | Rollout MSE $< 0.15$ across horizons $k \in \{1, 2, 5, 10, 25\}$. | Rollout MSE explodes ($> 1.0$) or errors grow catastrophically. |
| **H9: Homeostatic Synaptic Scaling** | Multiplicative synaptic scaling regulates firing rates toward target distributions. | Mean firing rate maintained within $\pm 30\%$ of target without direct task optimization; bounded weights. | Firing rates diverge into runaway excitation or quiescent extinction. |
| **H10: Dynamical Stability** | E/I balance improves long-term numerical and dynamical stability. | Zero weight explosion, bounded membrane potentials, no NaN/Inf, bounded population variance over $10^4$ steps. | Runaway excitation or numerical breakdown. |

---

## 3. Experimental Matrix

### 3.1 Primary Conditions (Section 14)
- **Condition A (REF_PHASE7)**: Phase 7 Spiking Reference (Hypersynchronous control without Dale E/I balance).
- **Condition B (EI_NO_HOMEO)**: Dale E/I (80% E, 20% I), static weights, no homeostasis.
- **Condition C (EI_HOMEO)**: Dale E/I + Homeostatic Synaptic Scaling.
- **Condition D (EI_STDP)**: Dale E/I + Dale-constrained STDP.
- **Condition E (EI_STDP_HOMEO)**: Dale E/I + Dale-constrained STDP + Homeostatic Synaptic Scaling (Full Phase 8 system).
- **Condition F1 (EI_FRAC_10)**: E/I with 10% inhibitory fraction.
- **Condition F2 (EI_FRAC_30)**: E/I with 30% inhibitory fraction.
- **Condition G1 (EI_STR_WEAK)**: E/I with weak inhibitory strength ($g = 1.0\times$).
- **Condition G2 (EI_STR_STRONG)**: E/I with strong inhibitory strength ($g = 4.0\times$).

### 3.2 Formal Controls (Section 15)
- **Control A**: Phase 7 Spiking Network.
- **Control B**: Signed unconstrained spiking network (weights unrestricted by Dale's law).
- **Control C**: Dale-constrained network without homeostasis.
- **Control D**: Dale + homeostasis.
- **Control E**: Dale + STDP + homeostasis.
- **Control F**: Dale constraint broken while keeping identical topology and total weight magnitude.

### 3.3 Causal Ablations (Section 16)
- `ABL_01_NO_INHIBITION`: Inhibitory population silenced (0% I / 100% E).
- `ABL_02_NO_HOMEOSTASIS`: Homeostatic scaling disabled ($\eta_{\text{homeo}} = 0$).
- `ABL_03_BREAK_DALE`: Sign constraints removed allowing synapses to change sign arbitrarily.
- `ABL_04_RANDOMIZE_SIGNS`: Synaptic signs assigned randomly independent of pre-synaptic neuron type.
- `ABL_05_SHUFFLE_IDENTITIES`: Inhibitory neuron identities permuted across spatial positions.
- `ABL_06_DISABLE_STDP`: STDP learning rate set to zero.
- `ABL_07_DISABLE_STRUCT_PLAST`: Structural plasticity / latency plasticity fixed.
- `ABL_08_GLOBAL_HOMEOSTASIS`: Local neuron-specific scaling replaced with population-averaged scalar scaling.
- `ABL_09_NO_DELAYS`: Axonal propagation delays eliminated (latency = 1 step for all connections).
- `ABL_10_DESTROY_COMMUNITY`: Modular community structure destroyed via degree-preserving edge swap.

### 3.4 Topologies Evaluated (Section 17)
- `MODULAR`: High intra-cluster density, sparse inter-cluster bridge connections.
- `SPATIAL_LOCAL`: Exponential distance decay in 3D metric Euclidean space.
- `SMALL_WORLD`: Watts-Strogatz spatial model with rewiring.
- `RANDOM`: Erdős-Rényi metric-free connectivity.

### 3.5 Benchmark Tasks (Section 18 & 19)
1. **Task A: Delayed Cue Recall** ($\Delta t \in \{50, 100, 250, 500\}$ steps).
2. **Task C: Temporal Context Disambiguation** (Identical sensory input at probe step $X$, divergent temporal history context).
3. **Task D: Multi-Step Forward Prediction** ($k \in \{1, 2, 5, 10, 25\}$ steps ahead).
4. **Task E: Sequence Reconstruction**.
5. **Task F: Continual Latent Memory Retention**.

### 3.6 Scale Ladder (Section 24 & 25)
- $N \in \{100, 1000, 10000, 50000, 100000\}$ neurons on Apple Silicon.
- Metrics: steps/sec, spikes/sec, synaptic events/sec, RAM footprint, runtime.

### 3.7 Reproducibility
- 3 independent deterministic seeds per condition: $\{42, 43, 44\}$.
- All metrics reported with mean, standard deviation, and statistical significance tests.
- BrainLog v2.6 traces generated and validated for replay.
