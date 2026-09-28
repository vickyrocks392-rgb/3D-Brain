# 3D-Brain: Research Questions & Empirical Hypotheses

## 1. Primary Research Question

> **What is the smallest computational system we can build that produces interesting, persistent, adaptive cognitive dynamics — and can those dynamics be represented as an interactive, living 3D neural world?**

This breaks down into three interrelated scientific investigations:
1. **Representational Capacity**: What minimal dynamical degrees of freedom (e.g. membrane dynamics, recurrent state, synaptic traces) are necessary to maintain temporal context without external sequence buffers?
2. **Online Learning & Plasticity**: Can local, error-driven, or neuromodulated plasticity rules adapt to non-stationary environments continually without suffering catastrophic forgetting, avoiding full Backpropagation Through Time (BPTT)?
3. **Observability & Spatial Topology**: Does embedding computational units into a 3D metric space (with distance-dependent propagation latencies and localized receptive fields) improve both computational efficiency and human interpretability?

---

## 2. Progressive Capability Ladder

We structure our empirical investigations around an ascending ladder of cognitive capabilities. Each tier must be mastered before advancing:

```text
Level 12: Multi-step Reasoning & Causal Planning
Level 11: Dynamic Memory Retrieval & Associative Binding
Level 10: Disentangled Representation Formation
Level 09: Structural Generalization Across Environments
Level 08: Zero-shot & Few-shot Task Transfer
Level 07: Dynamic Strategy & Hypothesis Selection
Level 06: Rule Discovery Under Latent Shifts
Level 05: Continual Adaptation Without Catastrophic Forgetting
Level 04: Prediction-Error Driven Representation Updating
Level 03: Non-Markovian Sequence Prediction
Level 02: Delayed Comparison Over Variable Temporal Intervals
Level 01: Temporal Memory & Persistence
```

---

## 3. Explicit Research Hypotheses & Falsification Criteria

### Hypothesis H1: Continuous-Time Dynamics vs. Discrete Steps
* **Claim**: A recurrent system governed by continuous-time differential equations (e.g., Liquid Time-Constant / Leaky-Integrator networks with input-dependent time constants $\tau(x)$) requires significantly fewer parameters (fewer units) to solve multi-scale temporal retention tasks than fixed-step RNNs or discrete gated architectures.
* **Metric**: Parameter efficiency ratio $\frac{\text{Params}(\text{Discrete})}{\text{Params}(\text{Continuous})}$ required to achieve $>95\%$ accuracy on the Delayed Comparison task across varying delay intervals $T \in [10, 100, 500]$ steps.
* **Falsification Condition**: If an optimized discrete GRU or LSTM matches or outperforms the continuous-time network with equal or fewer parameters across all delay intervals $T$, reject H1.

### Hypothesis H2: Local Predictive Error vs. Global BPTT
* **Claim**: Local predictive coding updates (where each population minimizes local prediction error $\epsilon = x - \hat{x}$ via Hebbian/anti-Hebbian synaptic changes) can adapt to online distribution shifts in non-stationary streams without requiring global backpropagation through time.
* **Metric**: Retention accuracy on initial task $A$ after streaming adaptation on task $B$ (Catastrophic Forgetting Index: $F = Acc_{A,\text{initial}} - Acc_{A,\text{post-}B}$).
* **Falsification Condition**: If local predictive updates fail to achieve stability or if $F > 0.40$ on the continuous adaptation benchmark, reject H2 as a standalone mechanism.

### Hypothesis H3: Neuromodulation for Gated Plasticity
* **Claim**: A low-dimensional, diffuse neuromodulatory signal (representing novelty, reward, or surprise $\sigma$) acting as a dynamic gate on synaptic plasticity ($\Delta W_{ij} = \eta \cdot \sigma(t) \cdot f(a_i, a_j)$) prevents destructive weight updates during stable regimes and focuses plasticity exclusively when unexpected errors occur.
* **Metric**: Signal-to-noise ratio of synaptic weight updates and stability of memory traces over $10^5$ inference steps with random distractors.
* **Falsification Condition**: If unmodulated continuous plasticity achieves equal or superior stability and memory retention under distractor noise, reject H3.

### Hypothesis H4: Spatialized 3D Embedding Imparts Inductive Bias
* **Claim**: Restricting connectivity by Euclidean distance in 3D space ($\Pr(\text{edge}_{ij}) \propto \exp(-d_{ij}/\lambda)$) combined with finite propagation velocity naturally produces modular functional clustering, hierarchical frequency decomposition, and superior fault tolerance compared to all-to-all dense connectivity.
* **Metric**: Modularity metric $Q$ and performance degradation curve under random node/edge ablation.
* **Falsification Condition**: If spatially constrained topologies perform strictly worse on sequence benchmarks at equivalent parameter counts without exhibiting functional specialization, reject H4.

---

## 4. Open Questions

1. **Topological Growth vs. Static Matrix**: Should the 3D brain begin with a fixed number of neurons and prune/strengthen connections, or should it instantiate new computational units (neurogenesis / synaptogenesis) when predictive error remains stubbornly high?
2. **Hybrid Learning Schemes**: What is the optimal balance between offline meta-initialization (learning to learn) and real-time online adaptation (plasticity during execution)?
3. **Bandwidth of Observability**: For networks with $10^5$ to $10^6$ parameters, what is the maximum lossless telemetry stream that can be transported and rendered at 60 FPS on Apple Silicon unified memory? Where must aggregation occur?
