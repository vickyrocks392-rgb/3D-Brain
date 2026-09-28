# Comprehensive Survey of Existing Approaches

To establish a solid scientific foundation for **3D-Brain**, this document evaluates key computational paradigms from computational neuroscience, dynamic systems theory, and non-transformer machine learning.

For each approach, we explicitly delineate:
1. **Established Research**: Proven mathematical formulation, empirical strengths, and standard applications.
2. **Our Hypothesis**: How 3D-Brain can leverage, adapt, or integrate this paradigm.
3. **Open Question**: Known failure modes, theoretical boundaries, and unresolved questions.

---

## 1. Recurrent Neural Systems (RNNs, LSTMs, GRUs)

### Established Research
* **Formulation**: Discrete-time hidden state recurrence:
  $$\mathbf{h}_t = \sigma(W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + \mathbf{b}_h)$$
  Gated architectures (Hochreiter & Schmidhuber 1997; Cho et al. 2014) introduce additive memory cells and input/forget/output gates that mitigate vanishing and exploding gradients.
* **Strengths**: Standardized, well-understood gradient flow, high GPU batching efficiency, compact parameter footprint relative to transformers for streaming inputs.
* **Limitations**: Discrete uniform time steps $\Delta t$; difficulty handling irregularly sampled temporal signals; catastrophic forgetting under continual streaming learning; requires unrolling through time (BPTT), which has high memory overhead and lacks biological/physical plausibility.

### Our Hypothesis
* Standard gated RNNs (GRU/LSTM) serve as **mandatory, rigorous baseline models** for all our temporal benchmarks.
* We hypothesize that purely discrete gate mechanisms create rigid step-bound dynamics that perform poorly when delays are variable or when signals must be modeled as continuous differential flows.

### Open Question
* Can a continuous-time formulation achieve equal or better long-range temporal retention than a GRU with fewer parameters?

---

## 2. Liquid Neural Networks & Continuous-Time Models (LTCs, CfCs)

### Established Research
* **Formulation**: Liquid Time-Constant (LTC) networks (Hasani et al., 2021) and Closed-form Continuous-depth (CfC) networks (Hasani et al., 2022) model hidden states via continuous ordinary differential equations (ODEs):
  $$\frac{d\mathbf{x}(t)}{dt} = -\left[\frac{1}{\boldsymbol{\tau}} + f(\mathbf{x}(t), \mathbf{I}(t))\right]\mathbf{x}(t) + f(\mathbf{x}(t), \mathbf{I}(t)) A$$
  where the effective time constant $\boldsymbol{\tau}_{\text{eff}}(\mathbf{x}, \mathbf{I})$ varies dynamically based on input and state.
* **Strengths**: Extreme parameter parsimony (often orders of magnitude fewer units than deep feedforward networks for control/temporal tasks); handles irregular time intervals naturally; smooth state trajectories suitable for continuous physical modeling.
* **Limitations**: ODE numerical solvers (Runge-Kutta, Euler) can be computationally heavy per step if not using closed-form approximations (CfC); gradient flow through stiff ODEs can be delicate.

### Our Hypothesis
* Dynamic, input-dependent time constants $\tau(x, I)$ are ideally suited for our spatial 3D substrate, providing both fast reactive responses and slow, persistent attractor dynamics within the same compact network.

### Open Question
* Can LTC/CfC formulations be trained or adapted via local plasticity rules rather than end-to-end BPTT?

---

## 3. Predictive Coding (Rao & Ballard, Friston)

### Established Research
* **Formulation**: Hierarchical generative architecture where higher layers predict lower-layer activity ($\hat{\mathbf{r}}_{l-1} = U_l \mathbf{r}_l$), and lower layers send only the residual prediction error upward:
  $$\boldsymbol{\epsilon}_{l-1} = \mathbf{r}_{l-1} - \hat{\mathbf{r}}_{l-1}$$
  Neurons minimize free energy / prediction error locally via gradient updates on states and weights (Rao & Ballard 1999; Friston 2005; Bogacz 2017).
* **Strengths**: Highly biologically plausible; replaces global backpropagation with local error minimization; naturally implements bidirectional top-down context and bottom-up sensory verification.
* **Limitations**: Inference requires multi-step iterative convergence (settling) per time step, increasing latency on conventional serial hardware; scaling to deep, complex cognitive tasks has historically lagged behind backprop.

### Our Hypothesis
* Prediction error is the foundational currency of both computation and 3D visualization. Representing prediction errors as distinct spatial events allows both the network to adapt locally and the human observer to pinpoint where expectations fail.

### Open Question
* Can a hierarchical predictive coding network maintain stable temporal attractors over extended time horizons without settling into trivial null predictions?

---

## 4. Reservoir Computing & Echo State Networks (ESNs, LSMs)

### Established Research
* **Formulation**: A large, fixed, randomly connected recurrent reservoir with spectral radius $\rho < 1$ (the Echo State Property), projecting to a trainable linear readout:
  $$\mathbf{x}_t = (1 - \alpha)\mathbf{x}_{t-1} + \alpha \tanh(W_{\text{res}} \mathbf{x}_{t-1} + W_{\text{in}} \mathbf{u}_t)$$
  $$\mathbf{y}_t = W_{\text{out}} \mathbf{x}_t$$
  Only $W_{\text{out}}$ is trained (via simple ridge regression or LMS), while internal weights $W_{\text{res}}$ remain fixed (Jaeger 2001; Maass et al. 2002).
* **Strengths**: Extremely fast, zero backpropagation through the reservoir; avoids vanishing/exploding gradients entirely; rich non-linear spatiotemporal feature expansion.
* **Limitations**: High parameter overhead (large reservoirs needed for complex non-linear memory); reservoir dynamics are passive and cannot self-tune to task-specific structural requirements unless augmented with plasticity.

### Our Hypothesis
* We can use reservoir concepts for our initial sub-10K parameter baseline experiments to demonstrate how temporal information is separated across spatial dimensions before evaluating plastic reservoirs.

### Open Question
* Can targeted, error-modulated local plasticity convert an arbitrary random reservoir into a compact, task-specialized dynamical substrate?

---

## 5. Spiking Neural Networks (SNNs)

### Established Research
* **Formulation**: Neurons communicate via discrete, sparse action potentials (spikes) triggered when membrane potential $V(t)$ crosses threshold $V_{\text{th}}$ (Leaky Integrate-and-Fire, Izhikevich models):
  $$\tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + R I(t)$$
  Learning typically uses Spike-Timing-Dependent Plasticity (STDP) or surrogate gradient backpropagation (Neftci et al., 2019).
* **Strengths**: Extreme event-driven energy efficiency on neuromorphic hardware (e.g., Loihi); precise temporal spike-timing codes.
* **Limitations**: Non-differentiable threshold functions complicate gradient descent (requiring surrogate gradients); high clock-frequency simulation needed on standard CPUs/GPUs, creating compute bottlenecks on standard Mac hardware.

### Our Hypothesis
* Continuous-rate or event-driven bursting models are preferable to full microsecond-level biophysical spike integration for our initial stages, preserving smooth mathematical telemetry for 60 FPS visualization.

### Open Question
* Does binary spike timing provide computational advantages over continuous-rate activations for our targeted cognitive capabilities at the 10K–50K scale?

---

## 6. Plastic Neural Networks & Neuromodulated Learning

### Established Research
* **Formulation**: Synaptic weights possess both a static baseline $w_{ij}$ and a dynamic plastic trace that updates online:
  $$W_{ij}(t) = w_{ij} + \alpha_{ij} \cdot \text{Trace}_{ij}(t)$$
  Differentiable plasticity (Miconi et al., 2018; Soltoggio et al., 2018) enables gradient-based meta-learning of plasticity coefficients, while neuromodulatory signals $m(t)$ act as dopamine/acetylcholine analogues to gate when learning occurs:
  $$\Delta \text{Trace}_{ij}(t) = \eta \cdot m(t) \cdot [a_j(t) a_i(t) - a_i(t) \text{Trace}_{ij}(t)]$$
* **Strengths**: Enables fast online adaptation to changing reward contingencies or input rules within a single evaluation episode without re-training base weights.
* **Limitations**: Can be vulnerable to runaway positive feedback or catastrophic memory erasure if neuromodulatory gating is improperly tuned.

### Our Hypothesis
* Neuromodulated plasticity is a prime candidate for our adaptive substrate: it bridges offline architectural optimization with online, zero-BPTT streaming adaptation.

### Open Question
* What minimal mathematical rule for neuromodulatory release $m(t)$ achieves the optimal balance between plasticity (learning new rules) and stability (protecting existing memories)?

---

## 7. Local Learning Rules (Forward-Forward, Equilibrium Propagation)

### Established Research
* **Formulation**: Replaces global gradient backpropagation with layer-local or contrastive phases:
  * **Forward-Forward Algorithm** (Hinton, 2022): Uses two forward passes (positive data with high "goodness", negative data with low "goodness"), updating weights locally to maximize goodness on positive data.
  * **Equilibrium Propagation** (Scellier & Bengio, 2017): Uses recurrent energy-based relaxation; updates weights by comparing free equilibrium states with weakly clamped states.
* **Strengths**: No backward computational graph required; physically localized updates; hardware-friendly.
* **Limitations**: Convergence can be slow; complex sequence modeling with long temporal dependencies remains an active research challenge.

### Our Hypothesis
* Local learning principles are vital to keep the computational substrate inspectable: every weight update can be evaluated locally between adjacent nodes without mysterious global adjoint vectors.

### Open Question
* Can Forward-Forward or contrastive energy dynamics be adapted effectively to streaming temporal sequences with variable delays?

---

## 8. Neural Cellular Automata (NCAs)

### Established Research
* **Formulation**: Grid or spatial graph of identical discrete units updating according to local convolutional or neighborhood rules:
  $$\mathbf{s}_i(t+1) = \mathbf{s}_i(t) + \text{MLP}(\mathbf{s}_i(t), \mathcal{N}_i(t))$$
  (Mordvintsev et al., 2020). Demonstrated remarkable self-repair, morphogenetic pattern generation, and decentralized computation.
* **Strengths**: Fully decentralized; completely local communication; extraordinary robustness to damage and unit deletion.
* **Limitations**: Highly challenging to program or train for arbitrary symbolic or sequential memory tasks; prone to spatial limit cycles.

### Our Hypothesis
* Spatial neighborhood message passing from NCAs can guide our 3D topological connection schemes, promoting localized self-organizing clusters.

### Open Question
* Can NCA-style local message passing sustain persistent working memory vectors across distant spatial regions without signal dissipation?

---

## 9. Continual Learning & Synaptic Consolidation

### Established Research
* **Formulation**: Techniques to prevent catastrophic forgetting when learning sequential tasks $T_1, T_2, \dots, T_k$:
  * **Elastic Weight Consolidation (EWC)** (Kirkpatrick et al., 2017): Penalizes changes to weights critical for past tasks using the Fisher Information Matrix.
  * **Synaptic Intelligence** (Zenke et al., 2017): Computes an online path integral of parameter importance during trajectory execution.
* **Strengths**: Preserves past capabilities without storing raw training historical data.
* **Limitations**: Requires task boundary identifiers; struggles with continuously drifting, unsegmented non-stationary streams.

### Our Hypothesis
* Incorporating online synaptic consolidation traces directly into our 3D synaptic data structure ensures that important connections visually stiffen and resist perturbation during novel experiences.

### Open Question
* Can task boundaries be inferred autonomously from internal prediction error surges rather than external supervisor cues?

---

## Summary Comparison Matrix

| Paradigm | Temporal Memory | Continual Adaptation | Local Computation | Parameter Efficiency | 3D Visual Interpretability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gated RNN (GRU)** | High | Low (Forgetting) | None (Global BPTT) | Moderate | Moderate (Latent vector) |
| **Liquid Networks (LTC/CfC)**| High | Moderate | Low (Standard BPTT) | **Extremely High** | **High (Continuous flow)** |
| **Predictive Coding** | Moderate | High (Local error) | **High (Local errors)** | High | **Extremely High (Error bursts)**|
| **Reservoir Computing** | High (Short-term) | Low (Fixed res) | High (Readout only)| Low (Needs size) | High (Phase space) |
| **Spiking Networks (SNN)** | High | High (STDP) | **High (Spike local)** | High | High (Sparse spikes) |
| **Plastic / Neuromodulated** | High | **Extremely High** | **High (Gated local)**| High | **Extremely High (Live updates)**|
| **Neural Cellular Automata** | Low (Spatial focus)| Moderate | **High (Neighborhood)**| High | **High (Self-organizing)** |
