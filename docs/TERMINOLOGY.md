# 3D-Brain: Formal Terminology & Glossary

To maintain scientific precision across the research codebase, papers, benchmarks, and 3D visualization layers, all terms are formally defined below. Hand-waving, poetic metaphors, and anthropomorphic assertions are strictly avoided.

---

### 1. Computational Brain
The discrete or continuous-time dynamical system $(S, \Theta, \mathcal{T}, \mathcal{E})$ comprising a set of stateful computational units $S$, dynamic connection parameters $\Theta$, an integration time-step rule $\mathcal{T}: S_t \times X_t \to S_{t+1}$, and an observable telemetry emitter $\mathcal{E}: (S_t, \Theta_t) \to \mathcal{O}_t$. The Brain is stateful, persistent, and non-stateless.

### 2. Computational Neuron (Unit)
A stateful mathematical node $i$ characterized by:
* An internal state variable $h_i(t) \in \mathbb{R}$ (membrane potential, hidden continuous state, or reservoir variable).
* An output activation $a_i(t) = \phi(h_i(t)) \in \mathbb{R}$.
* An intrinsic time constant $\tau_i$ or state decay rate.
* A 3D spatial coordinate $\mathbf{p}_i = (x_i, y_i, z_i) \in \mathbb{R}^3$.
* Optional local prediction $\hat{x}_i(t)$ and prediction error $\epsilon_i(t)$ components.

A computational neuron is **not** an entire biological cell with biochemical organelles; it is an idealized dynamical computational unit situated in physical 3D space.

### 3. Connection (Synapse)
A directed transmission path from source unit $j$ to target unit $i$ characterized by:
* A transmission weight $w_{ij}(t) \in \mathbb{R}$.
* A transmission latency/delay $d_{ij} \in \mathbb{N}$ or continuous delay $\delta_{ij} \in \mathbb{R}^+$, functionally proportional to Euclidean distance $\|\mathbf{p}_i - \mathbf{p}_j\|$.
* A plasticity trace or eligibility trace $e_{ij}(t) \in \mathbb{R}$.
* A current signal payload $\sigma_{ij}(t) = w_{ij} \cdot a_j(t - d_{ij})$ in transit.

### 4. Activation
The instantaneous output $a_i(t)$ emitted by unit $i$ to its outgoing connections, typically computed by a non-linear mapping $\phi(\cdot)$ (e.g., tanh, sigmoid, GELU, or ReLU) operating on its internal state $h_i(t)$. In 3D space, activation maps to neuron luminance, core size, and emission intensity.

### 5. Memory
The preservation of past information over time, manifested through two distinct physical mechanisms:
1. **Dynamic (Working) Memory**: Energy reverberating in recurrent state vectors $\mathbf{h}(t)$, trajectory attractors, or decaying delay lines. Transient, fast, volatile.
2. **Structural (Synaptic) Memory**: Persistent alterations in connection weights $W$ and plastic eligibility traces $E$. Long-term, slowly changing, non-volatile across sessions.

### 6. Prediction
An internally generated estimate $\hat{\mathbf{y}}(t + \Delta t)$ or $\hat{\mathbf{x}}_i(t)$ of future sensory input, downstream neural activity, or state transitions, produced by forward or top-down computational pathways prior to the arrival of actual signals.

### 7. Prediction Error
The algebraic or vector difference between actual incoming signals and predicted signals:
$$\mathbf{\epsilon}(t) = \mathbf{y}(t) - \hat{\mathbf{y}}(t)$$
Prediction error is the fundamental currency of computation, driving state updates, directional attention, and synaptic adaptation.

### 8. Learning
The systematic optimization or reconfiguration of internal parameters (synaptic weights $\Theta$, time constants $\boldsymbol{\tau}$, structural connections) to improve future performance or reduce cumulative prediction error across time. Learning may occur offline (gradient descent) or online (streaming adaptation).

### 9. Plasticity
The capacity of individual connections to alter their transmission efficacy $w_{ij}$ in real time based on local activity, temporal correlation (e.g., Hebbian STDP), and global neuromodulatory signals:
$$\Delta w_{ij}(t) = f\big(a_i(t), a_j(t), w_{ij}(t), m(t)\big)$$
where $m(t)$ is a diffuse neuromodulatory factor. Plasticity is distinct from offline gradient updates because it operates continuously during live execution.

### 10. Cognitive Event
A discrete, detectable qualitative transition or threshold crossing in the brain's internal dynamical trajectory. Examples:
* A sudden surge in prediction error ($\|\boldsymbol{\epsilon}(t)\| > \theta_{\epsilon}$).
* Attractor basin switching (decision commit).
* Neuromodulatory burst release.
* Synaptic weight consolidation.

Cognitive events are formally defined by mathematical detector predicates, not arbitrary visual labels.

### 11. Telemetry Stream
The structured sequence of serialized state snapshots, delta changes, and cognitive event packets emitted by the computational substrate at each time step $t$, consumed by storage engines, replay buffers, and 3D visualizers.

### 12. Observability Layer
The rendering and inspection subsystem (macOS/Metal/WebGPU application) that translates the telemetry stream into an interactive 3D spatial world, rendering real state without fabrication or synthetic filler animations.

### 13. Eligibility Trace
A decaying synapse-local memory flag $e_{ij}(t)$ that marks recent pre/post coincidences or STDP pairings as eligible for subsequent reinforcement. Bounded strictly by anatomical connectivity ($A_{ij} = 1$).

### 14. Third-Factor Teaching Signal
A diffuse scalar or community-partitioned teaching signal $M(t)$ (reward, prediction error, signed RPE) that modulates the magnitude and sign of weight updates across eligible synapses.

### 15. Signed Reward Prediction Error (RPE)
The difference between received extrinsic reward and an expected moving baseline: $\delta(t) = R(t) - \bar{R}(t)$. Produces bidirectional reinforcement (potentiation vs depression).

### 16. Temporal Credit Horizon
The maximum temporal interval $\Delta t$ between neural activity and reward delivery over which synaptic credit assignment remains effective before recurrent network spikes overwrite and diffuse eligibility traces.

### 17. Functional Modularity ($Q_{\text{func}}$)
The Newman modularity metric calculated directly from post-synaptic spike correlation matrices using Louvain community detection, quantifying the degree of functional segregation in active neural dynamics.
