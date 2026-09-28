# Phase 5 Design Audit & Architectural Specification

**Project**: 3D-Brain  
**Phase**: Phase 5 — Self-Supervised Embodied World Model & Dynamic Brain Growth  
**Date**: September 2026  
**Status**: DESIGN AUDIT COMPLETE  
**Author**: 3D-Brain Research Group  

---

## 1. Executive Summary

This design audit establishes the formal technical requirements, existing substrate capabilities, architectural constraints, and scientific boundaries for Phase 5. In accordance with project governance, Phase 5 directly builds upon the validated findings of Phases 0–4 while explicitly incorporating the critical negative results from Phase 4:

1. **Geometry alone did not produce superior task performance.**
2. **Propagation delays generally dispersed signals and degraded discrete task memory.**
3. **Imposed structural modularity ($Q_{struct} \approx 0.63 - 0.74$) did not spontaneously yield functional modularity ($Q_{func} \approx -0.02$).**
4. **Rate-coded neural dynamics did not sustain autonomous oscillatory phase synchrony ($R < 0.20$).**

Therefore, Phase 5 does not assume that spatial embodiment, world models, cognitive modularity, or dynamic self-organization will spontaneously emerge. Instead, Phase 5 empirically tests whether **self-supervised next-state prediction, closed-loop sensorimotor interaction, and dynamic structural plasticity (synaptogenesis and activity-dependent pruning)** provide measurable computational advantages over frozen, passive, and static-topology baselines.

---

## 2. Audit of Existing System

### 2.1 Computational Substrates (Phases 1–4)
* **Base Architectures (`brain/`)**:
  * `DISC-LRNN` (`DiscreteLeakyRNN`): Discrete leaky recurrent unit with learnable timescales $\alpha_i = \sigma(\rho_i)$, candidate state updates $\tilde{h}_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$, and activation $a_t = \tanh(h_t)$.
  * `BASE-GRU-MemBias`: Gated recurrent unit with explicit memory retention bias.
  * `CandLiquid`: Continuous-time liquid state machine baseline with Euler integration.
* **Spatial Substrate (`brain/spatial.py`)**:
  * `SpatialNeuron`: situated at $\mathbf{x} \in \mathbb{R}^3$, module ID, activation, state.
  * `SpatialSynapse`: pre/post unit indices, weight $w_{ij}$, physical distance $d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|_2$, propagation delay $\tau_{ij} = \max(1, \lfloor d_{ij}/v \rceil)$.
  * `SignalPropagationQueue`: Ring-buffer queue managing in-transit signals with finite transmission velocities.
  * `SpatialPlasticRNN`: Integrates 3D metric geometry, ring-buffer signal arrivals $I_{\text{arrived}}(t)$, leaky integration, and online plasticity.

### 2.2 Learning Mechanisms (Phase 3)
* Modular local plasticity rules (`brain/plasticity.py`):
  * `FrozenPlasticity`: $\Delta W = 0$.
  * `HebbianPlasticity`: $\Delta w_{ij} = \eta \cdot (a_i a_j) - \lambda w_{ij}$.
  * `OjaPlasticity`: $\Delta w_{ij} = \eta \cdot a_j (a_i - a_j w_{ij})$.
  * `ModulatedPlasticity`: Three-factor rule $\Delta w_{ij} = M(t) \cdot e_{ij}(t)$ with neuromodulatory scalar $M(t)$ and eligibility trace $e_{ij}(t)$.
  * `HomeostaticPlasticity`: Activity-dependent target rate regulation.
* **Limitation**: Plasticity in Phase 3 was strictly synaptic weight modification over a fixed structural graph ($E = \text{constant}$). Connections could not be born or pruned.

### 2.3 Observability Architecture (Phase 2–4)
* **Core Components (`observability/`)**:
  * `StateCapture`: Non-invasive step capture logging node states, activations, signals in transit, dynamic variables ($\alpha, \text{dw}$), and events.
  * `ComputationalEvent`: Typed event model (`EventType`) marking discrete transitions (e.g. `TRIAL_STARTED`, `PLASTICITY_UPDATE`, `ENVIRONMENT_CHANGE`).
  * `BrainTopology`: Node and directed edge manifest including spatial positions, distances, and delays.
  * `.brainlog` Serializer: Deterministic JSON recording with periodic full-state snapshots (every 10 steps). Current schema version: `v2.2`.

### 2.4 Replay Engine (`observability/replay.py`)
* Standalone headless replay interface supporting:
  * Forward/backward stepping, arbitrary step seeking.
  * Node inspection (`NodeInspection`), edge inspection (`EdgeInspection`).
  * Plasticity event extraction, in-transit signal query, spatial dynamic metrics retrieval.

### 2.5 Visualization Console (`visualization/player.html`)
* WebGL / Three.js 3D viewport rendering:
  * Spatial neuron positions as colored spheres modulated by activation and state.
  * Synaptic connection lines colored by weight.
  * In-transit signal sprites traveling along edges.
  * Interactive HUD with timeline scrubber, model switcher, and node/edge inspector panels.

---

## 3. Architectural Limitations & Confounds to Address

1. **Static Graph Topology**: In Phases 1–4, the adjacency matrix $\mathbf{A} \in \{0, 1\}^{N \times N}$ was immutable during simulation. Synaptic weights could adjust, but connection topology was static. Phase 5 must introduce dynamic structural plasticity (synaptogenesis and pruning) while preserving deterministic reproducibility.
2. **Supervised Target Signal vs. Self-Supervised Prediction**: Previous tasks (Task A, Continual CDA, Task C) supplied external target labels $y^*(t)$. Phase 5 must implement true self-supervised next-step sensory prediction $\hat{s}_{t+1}$ where the ground truth is simply the future environment observation $s_{t+1}$.
3. **Passive Sequence Processing vs. Closed-Loop Embodiment**: In previous tasks, inputs were fed sequentially regardless of model output. Phase 5 requires a closed sensorimotor loop where the brain's action $a_t$ actually determines the next environmental state and sensory observation.
4. **Graph Explosion and Resource Leaks**: Dynamic synapse formation and neuron growth can cause unbounded memory and compute inflation. Phase 5 must enforce rigorous upper bounds on degree $\langle k \rangle \le k_{\max}$, connection count $E \le E_{\max}$, and population growth $N \le N_{\max}$.

---

## 4. Proposed Phase 5 Extensions

### 4.1 Minimal Embodied Environment (`GridWorld2D` / `DiscreteArena`)
* 2D continuous or discrete spatial grid containing:
  * Agent spatial coordinates $\mathbf{p}_t = (x_t, y_t)$, heading $\theta_t$.
  * Fixed landmarks / obstacles with characteristic sensory signatures.
  * Local sensory field: distance/ray sensors + landmark signature sensors (dimension $D_{sensory} = 8$).
  * Discrete actions: `{FORWARD, TURN_LEFT, TURN_RIGHT, NO_OP}`.
  * Deterministic transition dynamics: $\mathbf{p}_{t+1} = \mathcal{T}(\mathbf{p}_t, a_t)$.
  * Non-stationary environmental shifts: Environment A (standard layout) $\to$ Environment B (altered layout / dynamics) $\to$ Environment A.

### 4.2 Self-Supervised Predictive Substrate (`PredictiveSpatialRNN`)
* Recurrent hidden state $h_t$ integrated via distance-dependent propagation queue.
* Action-conditioned sensory prediction:
  $$\hat{s}_{t+1} = \mathcal{P}(h_t, a_t) = W_{\text{pred}} [a_t; h_t] + b_{\text{pred}}$$
* Real computational prediction error:
  $$e_t = s_{t+1} - \hat{s}_{t+1}, \quad \mathcal{E}_t = \frac{1}{2} \|e_t\|_2^2$$
* Exposed directly to telemetry without external supervisory labels.

### 4.3 Predictive Learning Rules (`PredictiveLearningRule`)
* **Rule A: Frozen Baseline** ($\Delta W = 0$).
* **Rule B: Local Predictive Plasticity**: Local three-factor update where presynaptic activity, postsynaptic candidate state, and back-projected local prediction error drive synaptic adjustments.
* **Rule C: Neuromodulated Predictive Plasticity**: Modulatory prediction error magnitude $M(t) = \tanh(\beta \mathcal{E}_t)$ gating local eligibility traces.
* **Rule D: Gradient-Based Baseline**: BPTT / online gradient descent benchmark strictly as a control condition.

### 4.4 Dynamic Structural Plasticity (`StructuralPlasticity`)
* **Synapse Pruning**: Periodically prune inactive, low-weight, or task-uninformative connections:
  $$\text{Prune if } |w_{ij}| < \theta_{\text{prune}} \text{ and usage frequency } < \nu_{\min}$$
* **Synapse Formation (Synaptogenesis)**: Preferentially form new synapses between co-active units under spatial distance constraints:
  $$P(\text{form } i \to j) \propto \exp\left(-\frac{d_{ij}^2}{2\sigma_{\text{struct}}^2}\right) \cdot \text{corr}(a_i, a_j)$$
* **Strict Conservation & Degree Bounds**:
  $$k_{\min} \le \text{deg}(i) \le k_{\max}, \quad E(t) \le E_{\max}$$
* **Dynamic Neuron Population (Dynamic-N)**: Controlled neuron recruitment under persistent high prediction error, strictly bounded by $N \le N_{\max}$.

### 4.5 Schema & Telemetry Evolution (`.brainlog` v2.3)
* Retain full backward compatibility with v1.0, v2.0, v2.1, and v2.2.
* Extend frame payload to include:
  * `environment`: agent position, heading, active landmarks.
  * `prediction`: predicted sensory vector $\hat{s}_{t+1}$, actual sensory vector $s_{t+1}$, prediction error $\mathcal{E}_t$.
  * `actions`: selected action $a_t$.
  * `structural_events`: dynamic synapse formation, synapse pruning, neuron birth/death events.

---

## 5. Explicit Scientific Boundaries

### What Phase 5 Will Claim (if supported by evidence):
* Quantitative comparison of next-state prediction error between active embodied agents and passive observers.
* Measured impact of dynamic structural plasticity on adaptation rate and catastrophic forgetting under environmental shifts.
* Quantitative comparison of $Q_{\text{struct}}$ vs. $Q_{\text{func}}$ under predictive self-supervised learning.
* Deterministic reproducibility and scaling efficiency of the dynamic spatial substrate.

### What Phase 5 Will NOT Claim:
* The system will NOT be described as conscious, intelligent, sentient, or cognitively self-aware.
* Negative results (e.g. failure of functional modularity to emerge, lack of active embodiment advantage) will be reported objectively as valid scientific findings.
* Decorative visual animations will not be substituted for verifiable computational dynamics.
