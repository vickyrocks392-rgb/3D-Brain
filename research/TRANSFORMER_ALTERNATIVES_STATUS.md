# Transformer Alternatives Status: Comprehensive Candidate Architecture Analysis

**Phase:** Phase 13 Research Synthesis  
**Status Statement:**  
> **STATUS: RESEARCH ONGOING — NO FINAL ALTERNATIVE SELECTED.**

---

## 1. Executive Research Mandate

The 3D-Brain project investigates whether biologically grounded, non-Transformer computational paradigms can achieve flexible memory, temporal credit assignment, and contextual action routing without the quadratic context bottleneck, lack of spatial embedding, and biological implausibility of the standard self-attention Transformer.

In Phase 13, the question of a Transformer replacement remains **firmly open**. No final architecture has been chosen. Standard causal Transformers serve strictly as **reference baselines** ($72.0\%$ accuracy on identical tasks) to anchor empirical comparison.

---

## 2. Comprehensive Candidate Architecture Matrix

Below is the scientific audit of the candidate computational families evaluated throughout Phases 1–13:

---

### Candidate 1: Spiking Neural Networks (SNNs)
- **What It Solves:** Extreme energy efficiency, event-driven sparse computation, temporal spike-timing precision.
- **What It Does Not Solve:** The continuous-to-discrete discretization cliff; single-step point-process spikes lose subthreshold manifold information.
- **Biological Plausibility:** High (Hodgkin-Huxley / Leaky Integrate-and-Fire / Axon Initial Segment).
- **Memory Mechanism:** Membrane integration, synaptic eligibility traces, spike-timing attractors.
- **Scaling:** High throughput ($1.85\text{ MSpk/s}$ at $N=10,000$).
- **Training Mechanism:** Three-factor eligibility STDP, surrogate gradient BPTT.
- **Temporal Credit Assignment:** Dependent on eligibility traces and slow NMDA time constants.
- **Spatial Structure:** Explicit 3D embedding with conduction delays ($1\text{--}5\text{ steps}$).
- **Interpretability / Observability:** Observable via raster plots, ISI distributions, and membrane gauges.
- **Computational Cost:** Extremely low per timestep; event-driven.
- **Current 3D-Brain Evidence:** Phase 7–8, Phase 13 (S1–S9). Discrete action selection requires temporal + population coding.

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
- **Biological Plausibility:** High (spatial embedding matches connectome statistics).
- **Memory Mechanism:** Compartmentalized attractors within structural communities.
- **Scaling:** High; sparse spatial connectivity avoids $O(N^2)$ memory explosion.
- **Training Mechanism:** Structural plasticity (axonal pruning / sprouting) combined with synaptic plasticity.
- **Temporal Credit Assignment:** Constrained by inter-module axonal propagation delays.
- **Spatial Structure:** Direct 3D Cartesian coordinates $[x, y, z]$.
- **Interpretability / Observability:** Directly visualizable in 3D scene.
- **Computational Cost:** Low; sparse graph operations.
- **Current 3D-Brain Evidence:** Phase 4, Phase 8, Phase 10 spatial networks.

---

### Candidate 7: Latent-State Recurrent Architectures
- **What It Solves:** Long-term context disambiguation, persistent internal working memory.
- **What It Does Not Solve:** Susceptible to drift over very long delays unless anchored by discrete attractors.
- **Biological Plausibility:** Moderate (recurrent cortical loop models).
- **Memory Mechanism:** Self-sustaining low-dimensional manifold attractors.
- **Scaling:** High.
- **Training Mechanism:** Hebbian attractor formation + neuromodulated readout.
- **Temporal Credit Assignment:** Maintained across multi-second delays ($> 2000\text{ ms}$).
- **Spatial Structure:** Can be mapped onto modular clusters.
- **Interpretability / Observability:** PCA manifold projections, attractor energy wells.
- **Computational Cost:** Moderate.
- **Current 3D-Brain Evidence:** Phase 6, Phase 11 recurrent attractor networks.

---

### Candidate 8: Neural Cellular Automata (NCAs)
- **What It Solves:** Extreme parameter efficiency, self-repair, local spatial computation.
- **What It Does Not Solve:** Difficult to perform global credit assignment and long-range sequence processing.
- **Biological Plausibility:** Moderate (developmental biology / morphogenesis).
- **Memory Mechanism:** Persistent spatial patterns on a cellular grid.
- **Scaling:** Scales linearly with cell count.
- **Training Mechanism:** Evolutionary strategies or BPTT through time.
- **Temporal Credit Assignment:** Poor for delayed temporal cues.
- **Spatial Structure:** High (grid or graph topology).
- **Interpretability / Observability:** Highly visual; state patterns directly rendered.
- **Computational Cost:** Low per cell update.
- **Current 3D-Brain Evidence:** Evaluated conceptually; not currently prioritized for sequence memory.

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
- **Current 3D-Brain Evidence:** Benchmarked in Phase 13 ($72.0\%$ accuracy reference baseline).

---

### Candidate 10: Hybrid Biophysical Architectures (The 3D-Brain Integrated Lineage)
- **What It Solves:** Integrates Dale's E/I balance + modular 3D spatial topology + two-compartment NMDA/dendritic integration + three-factor eligibility credit assignment + biophysical spike encoding + action persistence.
- **What It Does Not Solve:** Single-step discretization loss remains an active research frontier ($8.17\%$ context retention at action stage); still trails supervised gradient methods on raw task accuracy ($45.3\%\text{ vs }72.0\%$).
- **Biological Plausibility:** Exceptionally High across all anatomical, dynamical, and learning constraints.
- **Memory Mechanism:** Slow NMDA conductances, dendritic plateau potentials, and recurrent attractor pools.
- **Scaling:** Demonstrates sub-millisecond execution for $N=10,000$ with $4.27\text{ MB}$ footprint.
- **Training Mechanism:** Local targeted three-factor plasticity with dendritic feedback.
- **Temporal Credit Assignment:** Preserves causal context across $2000\text{ ms}$ delays.
- **Spatial Structure:** Genuine 3D Euclidean coordinates with modular community structure.
- **Interpretability / Observability:** Completely observable in the 3D Neural Observatory.
- **Computational Cost:** Very low compared to Transformers.
- **Current 3D-Brain Evidence:** Phases 8, 9, 10, 11, 12, 13.

---

## 3. Decision Gate Criteria & Next Scientific Focus

**Why No Final Winner Is Selected:**
While the Phase 13 Hybrid Biophysical Architecture achieves superior biological realism, energy efficiency, and spatial observability, its behavioral accuracy ($45.3\%$) remains below the Small Transformer Reference ($72.0\%$) and Supervised Linear Readout ($60.0\%$). 

The bottleneck is now precisely identified:
**The continuous-to-discrete spike conversion drops mutual information from $1.00\text{ b}$ to $0.00\text{ b}$ at single steps, recovering only $0.082\text{ b}$ downstream.**

Until a biological spike mechanism can preserve $\ge 50\%$ of continuous upstream information through the discrete spiking bottleneck, declaring a Transformer alternative would be scientifically premature.
