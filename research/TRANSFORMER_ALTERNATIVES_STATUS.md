# Transformer Alternatives Status: Comprehensive Candidate Architecture Analysis

**Phase:** Phase 14 Research Synthesis  
**Status Statement:**  
> **STATUS: RESEARCH ONGOING — NO FINAL ALTERNATIVE SELECTED.**

---

## 1. Executive Research Mandate

The 3D-Brain project investigates whether biologically grounded, non-Transformer computational paradigms can achieve flexible memory, temporal credit assignment, and contextual action routing without the quadratic context bottleneck, lack of spatial embedding, and biological implausibility of the standard self-attention Transformer.

In Phase 14, the question of a Transformer replacement remains **firmly open**. No final architecture has been chosen. Standard causal Transformers serve strictly as **reference baselines** ($72.0\%$ accuracy on identical tasks) to anchor empirical comparison.

---

## 2. Comprehensive Candidate Architecture Matrix

Below is the scientific audit of the candidate computational families evaluated throughout Phases 1–14:

---

### Candidate 1: Spiking Neural Networks (SNNs)
- **What It Solves:** Extreme energy efficiency, event-driven sparse computation, temporal spike-timing precision.
- **What It Does Not Solve:** The continuous-to-discrete discretization cliff; single-step point-process spikes lose subthreshold manifold information ($0.000\text{ bit}$).
- **Biological Plausibility:** High (Hodgkin-Huxley / Leaky Integrate-and-Fire / Axon Initial Segment).
- **Memory Mechanism:** Membrane integration, synaptic eligibility traces, spike-timing attractors.
- **Scaling:** High throughput ($12.9\text{ MSpk/s}$ at $N=50,000$).
- **Training Mechanism:** Three-factor eligibility STDP, surrogate gradient BPTT.
- **Temporal Credit Assignment:** Dependent on eligibility traces and slow NMDA time constants.
- **Spatial Structure:** Explicit 3D embedding with conduction delays ($1\text{--}5\text{ steps}$).
- **Interpretability / Observability:** Observable via raster plots, ISI distributions, and membrane gauges.
- **Computational Cost:** Extremely low per timestep; event-driven.
- **Current 3D-Brain Evidence:** Phase 7–8, Phase 13, Phase 14. Temporal + population integration recovers $0.2358\text{ bit}$ downstream.

---

### Candidate 2: Liquid / Continuous-Time Recurrent Systems (LTCs)
- **What It Solves:** Adaptive time constants, continuous dynamical systems integration, robust handling of irregular temporal delays.
- **What It Does Not Solve:** Catastrophic forgetting over long horizons without explicit gating or attractor stabilization.
- **Biological Plausibility:** Moderate (continuous ODE approximation of membrane conductance).
- **Memory Mechanism:** Continuous state trajectory in latent phase space.
- **Scaling:** $O(N^2)$ dense matrix multiplication unless localized spatially.
- **Training Mechanism:** Adjoint sensitivity, three-factor RPE modulation.
- **Temporal Credit Assignment:** Continuous decay curves; vulnerable to fading gradients.
- **Spatial Structure:** Compatible with modular distance-dependent connectivity.
- **Interpretability / Observability:** Phase-space trajectory tracing, Lyapunov exponents.
- **Computational Cost:** Requires ODE solvers (Euler, Runge-Kutta); moderate compute cost.
- **Current 3D-Brain Evidence:** Phase 1–2 baselines (`CandLiquid`).

---

### Candidate 3: Predictive-Coding Architectures
- **What It Solves:** Local, unsupervised representation learning via prediction error minimization; eliminates global backpropagation.
- **What It Does Not Solve:** Delayed credit assignment over multi-second temporal gaps without external reinforcement.
- **Biological Plausibility:** Very High (hierarchical cortex, top-down prediction vs bottom-up error).
- **Memory Mechanism:** Generative internal model states.
- **Scaling:** Linear in synapses $O(E)$; highly parallelizable.
- **Training Mechanism:** Local Hebbian / anti-Hebbian updates driven by prediction error ($\epsilon = x - \hat{x}$).
- **Temporal Credit Assignment:** Excellent for single-step prediction; requires recurrence for multi-step credit.
- **Spatial Structure:** Naturally matches cortical laminar and modular organization.
- **Interpretability / Observability:** Highly interpretable error signals and residual maps.
- **Computational Cost:** Low to moderate.
- **Current 3D-Brain Evidence:** Phase 5 predictive plasticity benchmarks.

---

### Candidate 4: Reservoir Computing (Echo State Networks / Liquid State Machines)
- **What It Solves:** Fast training (only readout weights trained; reservoir frozen), complex temporal feature projection.
- **What It Does Not Solve:** Reservoir capacity is limited by the spectral radius; cannot autonomously restructure for novel contexts.
- **Biological Plausibility:** Moderate (unsupervised random recurrent connectivity).
- **Memory Mechanism:** High-dimensional transient fading memory.
- **Scaling:** Scales to tens of thousands of neurons if reservoir is sparse.
- **Training Mechanism:** Linear ridge regression on reservoir states.
- **Temporal Credit Assignment:** Constrained by the reservoir fading memory horizon ($< 50\text{ steps}$).
- **Spatial Structure:** Often spatialized as random geometric graphs.
- **Interpretability / Observability:** Low within reservoir; high at linear readout.
- **Computational Cost:** Very low training cost; high inference memory.
- **Current 3D-Brain Evidence:** Phase 2, Phase 6 comparison baselines.

---

### Candidate 5: Neuromodulated Recurrent Systems (3-Factor Plasticity)
- **What It Solves:** Biological temporal credit assignment without backpropagation; decouples local activity from global reinforcement.
- **What It Does Not Solve:** Credit leakage in large dense networks; non-specific modulation can destabilize unrelated memories.
- **Biological Plausibility:** High (Dopaminergic / Noradrenergic volume transmission).
- **Memory Mechanism:** Synaptic eligibility traces ($e_{ij}$) tagged by pre/post coincidence.
- **Scaling:** Linear in synapses $O(E)$.
- **Training Mechanism:** Three-factor rule: $\Delta w_{ij} = \eta \cdot M(t) \cdot e_{ij}(t)$.
- **Temporal Credit Assignment:** Solves delays up to trace decay horizon ($\tau_e \sim 500\text{--}2000\text{ ms}$).
- **Spatial Structure:** Modulated across functional modules or global diffuse fields.
- **Interpretability / Observability:** Directly observable via eligibility monitors and neuromodulator gauges.
- **Computational Cost:** Extremely low ($O(1)$ per synapse per step).
- **Current 3D-Brain Evidence:** Phase 3, Phase 9 modular RPE, Phase 10 two-compartment targeted credit.

---

### Candidate 6: Spatial Recurrent Systems (Modular / Small-World 3D Topologies)
- **What It Solves:** Physical wiring cost constraints, biological modularity, functional segregation vs integration ($Q_{\text{struct}}, Q_{\text{func}}$).
- **What It Does Not Solve:** Does not itself provide learning or memory without a dynamical plasticity substrate.
- **Biological Plausibility:** Very High (recapitulates brain connectomes and wiring minimization).
- **Memory Mechanism:** Topological clustering preserves reverberating local attractors.
- **Scaling:** Highly scalable ($O(k \cdot N)$ sparse connectivity).
- **Training Mechanism:** Structural rewiring, distance-dependent STDP.
- **Temporal Credit Assignment:** Conduction delays ($1\text{--}10\text{ ms}$) assist in spatiotemporal sequence tagging.
- **Spatial Structure:** Explicit continuous 3D Euclidean coordinates.
- **Interpretability / Observability:** Community detection, modularity graphs, 3D spatial layout.
- **Computational Cost:** Low.
- **Current 3D-Brain Evidence:** Phase 4, Phase 8, Phase 14 local vs global resonance analysis.

---

### Candidate 7: State Space Models (Mamba / S4)
- **What It Solves:** Linear time complexity ($O(T)$), fast parallel training via associative scans, long-range dependency modeling.
- **What It Does Not Solve:** Lacks physical spatial geometry; parameter updates still depend on global automatic differentiation (non-local).
- **Biological Plausibility:** Low to Moderate (continuous SSMs can approximate membrane dynamics, but associative scans lack physical biological realization).
- **Memory Mechanism:** Hidden recurrent state matrix $\mathbf{h}(t)$.
- **Scaling:** Excellent sequence scaling ($O(T)$).
- **Training Mechanism:** Supervised gradient backpropagation.
- **Temporal Credit Assignment:** Analytical gradients through continuous transition matrices.
- **Spatial Structure:** Non-spatial (1D sequence dimension).
- **Interpretability / Observability:** Moderate (hidden state trajectories can be probed).
- **Computational Cost:** Low inference cost; moderate training cost.
- **Current 3D-Brain Evidence:** Evaluated in comparative surveys; non-spatial structure remains a fundamental limitation.

---

### Candidate 8: Continuous Attractor Networks (CANNs)
- **What It Solves:** Persistent working memory, smooth continuous manifold representations (head direction, spatial grid maps), robustness to noise.
- **What It Does Not Solve:** Vulnerable to drift without anchor cues; discrete decision switching requires external competition mechanisms.
- **Biological Plausibility:** High (hippocampal place cells, entorhinal grid cells, head direction circuits).
- **Memory Mechanism:** Stable fixed points / line attractors maintained by recurrent symmetric/skew-symmetric connectivity.
- **Scaling:** Scales linearly with number of attractor modules.
- **Training Mechanism:** Anti-Hebbian / Hebbian attractor formation.
- **Temporal Credit Assignment:** Strong for static maintenance; weak for sequential history without asymmetric connections.
- **Spatial Structure:** Ring, toroidal, or spherical topologies.
- **Interpretability / Observability:** High; manifold projection and bump velocity tracking.
- **Computational Cost:** Low.
- **Current 3D-Brain Evidence:** Phase 6, Phase 11 persistent attractor memory substrate.

---

### Candidate 9: Small Transformers as Reference Baselines
- **What It Solves:** Unbounded context lookback, global soft attention across all prior tokens, stable optimization with gradient descent.
- **What It Does Not Solve:** Quadratic context scaling ($O(T^2)$), zero physical/spatial embedding, completely biologically implausible (no eligibility traces, non-local backprop), static token sequence metaphor incompatible with continuous embodied physics.
- **Biological Plausibility:** None.
- **Memory Mechanism:** KV-cache storage of all past sequence steps.
- **Scaling:** Limited by quadratic attention context memory.
- **Training Mechanism:** Backpropagation Through Time (BPTT).
- **Temporal Credit Assignment:** Analytical gradient routing across the attention matrix.
- **Spatial Structure:** None (token positions are 1D indices).
- **Interpretability / Observability:** Attention heatmaps.
- **Computational Cost:** High ($O(T^2)$).
- **Current 3D-Brain Evidence:** Benchmarked in Phase 13 & 14 ($72.0\%$ accuracy reference baseline).

---

### Candidate 10: Hybrid Biophysical Architectures (The 3D-Brain Integrated Lineage)
- **What It Solves:** Integrates Dale's E/I balance + modular 3D spatial topology + two-compartment NMDA/dendritic integration + target-specific dendritic learning + multi-timescale action assemblies + biophysical burst dynamics + dendritic calcium coincidence + population resonance.
- **What It Does Not Solve:** Diffuse retrograde signaling induces credit leakage without spine-level microdomains; scaling to complex multi-task reinforcement learning benchmarks remains an open investigation.
- **Biological Plausibility:** Exceptionally High across anatomical, physiological, dendritic, and learning constraints.
- **Memory Mechanism:** Slow NMDA conductances, dendritic plateau potentials, multi-timescale assemblies ($\tau = 25\text{--}1500\text{ ms}$), and persistent evidence accumulation.
- **Scaling:** Demonstrates sub-millisecond execution for $N=100,000$ ($169.6\text{ steps/s}$, $>2.54 \times 10^6\text{ spikes/s}$, $21.74\text{ MB}$ RAM).
- **Training Mechanism:** Local target-specific three-factor plasticity with dendritic plateau teaching signals ($G_1 / G_3$).
- **Temporal Credit Assignment:** Preserves causal context across $>5000\text{ ms}$ delays with zero decision chatter.
- **Spatial Structure:** Genuine 3D Euclidean coordinates with modular community structure.
- **Interpretability / Observability:** Completely observable in the 3D Neural Observatory with target dendritic synapse inspection and action assembly overlays.
- **Computational Cost:** Very low compared to Transformers ($O(N)$ linear event-driven scaling).
- **Current 3D-Brain Evidence:** Phase 15 closed the gap with the analytical BPTT reference ($68.0\%$), achieving **$64.0\% \text{ to } 65.3\%$** autonomous task accuracy without global backpropagation through time. Continuous Shannon mutual information preserved in the population temporal state reached **$0.9968\text{ b}$** ($99.7\%$).

---

## 3. Decision Gate Criteria & Next Scientific Focus

**Why No Final Winner Is Selected:**
While Phase 15 Target-Specific Dendritic Learning achieved a major breakthrough in closing the gap to Supervised BPTT ($64.0\%\text{--}65.3\%\text{ vs }68.0\%$) and confirmed that the recurrent temporal population preserves $0.9968\text{ b}$ of Shannon mutual information, the standard offline Small Transformer reference ($72.0\%$) and unconstrained continuous oracle ($96.0\%$) demonstrate that additional capacity remains latent in the substrate.

The scientific comparison remains:
- **Transformer:** Highly efficient sequence-processing mathematical reference without biological realism, spatial geometry, or continuous physics.
- **3D-Brain:** Stateful, spatial, temporally structured biological computational substrate operating with local credit assignment.

The question of eventual architecture selection remains open pending multi-task scaling, structural spine compartmentalization, and generalized embodiment.

> **STATUS: RESEARCH ONGOING — NO FINAL ALTERNATIVE SELECTED.**

