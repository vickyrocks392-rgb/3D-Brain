# 3D-Brain: Conceptual Model & Mathematical Mapping

## 1. System Mathematical Abstraction

The computational substrate of 3D-Brain is formulated as an embodied continuous-or-discrete dynamical graph embedded in 3D Euclidean space:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E}_{edges}, \mathcal{P}, \mathcal{M})$$

Where:
* $\mathcal{V} = \{v_1, v_2, \dots, v_N\}$ is the set of $N$ computational neurons.
* $\mathcal{P} = \{\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_N\} \subset \mathbb{R}^3$ denotes the immutable or slowly evolving spatial coordinates of each neuron.
* $\mathcal{E}_{edges} \subseteq \mathcal{V} \times \mathcal{V}$ is the directed connection graph.
* $\mathcal{M}$ represents the global neuromodulatory context $\mathbf{m}(t) \in \mathbb{R}^K$.

---

## 2. Core Concepts: Computational Definition to 3D Observability Mapping

The table below defines the strict 1-to-1 correspondence between computational state variables and their physical representation in the 3D world:

| Concept | Mathematical / Computational Formulation | 3D Representation in macOS App | Telemetry Field |
| :--- | :--- | :--- | :--- |
| **Neuron** | State vector $\mathbf{h}_i(t) \in \mathbb{R}^d$, position $\mathbf{p}_i \in \mathbb{R}^3$, intrinsic time constant $\tau_i$. | Point/Sphere mesh at position $(x, y, z)$. Base radius reflects population type; base color indicates resting potential. | `id`, `pos[3]`, `pop_id`, `state` |
| **Connection** | Directed edge $(j \to i)$ with weight $w_{ij}(t)$, propagation latency $d_{ij} = \lceil \|\mathbf{p}_i - \mathbf{p}_j\| / v_c \rceil$. | Directed 3D spline/line between $\mathbf{p}_j$ and $\mathbf{p}_i$. Thickness $\propto |w_{ij}|$, color hue indicates sign (excitatory = blue/cyan, inhibitory = red/orange). | `src`, `dst`, `weight`, `latency` |
| **Activation** | $a_i(t) = \phi(h_i(t)) \in [-1, 1]$ or $[0, 1]$. | Dynamic emissive bloom and surface brightness of neuron $i$. Higher activation increases glow radius and photometric luminance. | `activations[N]` |
| **Signal in Transit** | Discrete packet $\sigma_{ij}(t, \delta) = w_{ij} \cdot a_j(t - \delta)$ where $0 \le \delta \le d_{ij}$. | Luminous pulse travelling along the spline from $\mathbf{p}_j$ to $\mathbf{p}_i$ at speed $v_c$. Pulse intensity corresponds to signal magnitude. | `signals: [{src, dst, progress, val}]` |
| **Dynamic Memory** | Reverberating hidden state attractor $\mathbf{h}(t) = f(\mathbf{h}(t-1), \mathbf{x}(t))$. | Coordinated phase-locked rhythmic oscillation or sustained glow across a spatial sub-network. | `hidden_states[N]` |
| **Synaptic Memory** | Plastic trace matrix $W_{plastic}(t) = W_0 + \Delta W(t)$. | Static or slowly drifting spline thickness and opacity persistent across temporal trials. | `weight_deltas` (periodic keyframes) |
| **Prediction** | $\hat{\mathbf{y}}(t) = g(\mathbf{h}(t))$ (top-down or forward signal). | Forward projection beams or target population anticipation rings glowing before incoming stimulus arrives. | `predictions[M]` |
| **Prediction Error** | Residual error $\boldsymbol{\epsilon}(t) = \mathbf{y}(t) - \hat{\mathbf{y}}(t)$. | High-frequency chromatic pulse (e.g., magenta/amber flash) erupting at error-detecting nodes; radiating outward along feedback connections. | `errors[M]`, `total_error` |
| **Plasticity** | $\frac{d w_{ij}}{dt} = \eta \cdot m(t) \cdot H(a_i, a_j) - \gamma w_{ij}$. | Brief flash of light along the synapse edge during weight modification; color shift indicating potentiation vs. depression. | `plasticity_events: [{edge_id, dw}]` |
| **Neuromodulation** | Global scalar or low-dim vector $\mathbf{m}(t)$ (e.g., novelty, surprise, reward). | Atmospheric ambient volumetric illumination or global field color tinting the 3D space. | `neuromodulators[K]` |
| **Cognitive Event** | Qualitative state bifurcation (e.g., attractor switch, threshold crossing: $\|\boldsymbol{\epsilon}(t)\| > \theta$). | Spatial event ring ripple expanding through 3D space, timestamped in the timeline scrubber with a marker icon. | `events: [{type, t, node_id, mag}]` |

---

## 3. Propagation and Delay Physics

In artificial networks, propagation is usually instantaneous. In 3D-Brain, space matters:
1. Signal transmission between unit $j$ and unit $i$ takes finite time:
   $$\tau_{\text{transit}}(j, i) = \frac{\|\mathbf{p}_i - \mathbf{p}_j\|}{v_{\text{propagation}}}$$
2. This creates natural delay lines, phase shifts, and spatio-temporal interference patterns.
3. In the 3D visualizer, signals are observable physical entities moving through space, bridging the computational execution with intuitive human comprehension.

---

## 4. Telemetry and Event Stream Contract

The computational engine produces two synchronized streams:
1. **Dense Frame Telemetry (Sampled at $10\text{--}60\text{ Hz}$)**:
   * Compact array of activation floats: `float32[N]`.
   * Current global error scalars: `float32[E]`.
   * Active signals in transit: structured list of byte triplets `(edge_idx, progress_u8, amplitude_f16)`.
2. **Sparse Event Telemetry (Asynchronous / Event-Driven)**:
   * Significant plasticity updates ($\Delta w_{ij} > \delta_{\text{thresh}}$).
   * Predictive failure bursts ($\|\epsilon_k\| > \epsilon_{\text{thresh}}$).
   * Attractor basin transitions / Decision commits.

This separation guarantees that headless benchmarks run at maximum native compute speed without rendering bottlenecks, while the visualizer receives exactly the information needed to render a living world.

---

## 5. Dale's Principle, E/I Balance & Multi-Scale Synchrony (Phase 8)

Phase 8 introduces biological excitation/inhibition constraints:
1. **Dale's Principle**:
   - Each neuron is typed as either **Excitatory (E)** or **Fast-Spiking Inhibitory (I)** ($\sim 80\% \text{ E} / 20\% \text{ I}$).
   - Excitatory neurons emit strictly positive weights ($w_{ij} > 0$); inhibitory neurons emit strictly negative weights ($w_{ij} < 0$).
2. **Current Accounting**:
   - Excitatory current $I_E(t) \ge 0$ and inhibitory current $I_I(t) \le 0$ are separately integrated in ring buffers.
   - Net somatic current $I_{\text{net}}(t) = I_E(t) - |I_I(t)|$.
3. **Multi-Scale Synchrony**:
   - Kuramoto order parameter $R$ and Phase Locking Value ($PLV$) are evaluated across spatial radii:
     $$R_{\text{local}} \text{ (within neighborhood radius } r) \quad \text{vs} \quad R_{\text{global}} \text{ (whole brain)}$$
   - Operational Asynchronous-Irregular (AI) regime requires local functional coordination ($R_{\text{local}} > R_{\text{global}}$) alongside low pairwise spike-count correlation ($r_{\text{sc}} \approx 0.03$), preventing runaway global hypersynchrony and representational collapse.

---

## 6. Three-Factor Plasticity & Temporal Credit Assignment (Phase 9)

Phase 9 integrates local eligibility traces with extrinsic teaching signals:
1. **Local Synaptic Eligibility Traces ($e_{ij}$)**:
   $$e_{ij}(t) = \lambda_e e_{ij}(t-1) + S_i(t) S_j(t)$$
   Constrained to existing physical synapses $A_{ij} \in \{0, 1\}$.
2. **Third Factor Teaching Signals ($M(t)$)**:
   - Global Reward $R(t)$, Prediction Error $|y - \hat{y}|$, Signed RPE $R(t) - \bar{R}(t)$, or Local Module Feedback $R_c(t) - \bar{R}_c(t)$.
3. **Weight Update Rule**:
   $$\Delta w_{ij}(t) = \eta \cdot M(t) \cdot e_{ij}(t)$$
   Strictly bounded within Dale bounds $[0, w_{\text{max}}^E]$ or $[-w_{\text{max}}^I, 0]$.
4. **Credit Diffusion Constraint**:
   Recurrent spike cascades over delays $\Delta t \ge 10$ steps diffuse eligibility across unrelated synapses, establishing an operational horizon for scalar three-factor reinforcement learning.

---

## 7. Audited Biophysical Boundaries of the Minimal Substrate MIN_E (Phase 18)

Phase 18 established rigorous, empirical operational boundaries for the minimal sufficient brain architecture (`MIN_E`):

1. **Parameter & State Topology**:
   - **Plastic Parameters**: Exactly 82 (80 plastic readout weights $W_{\text{readout}} \in \mathbb{R}^{40 \times 2}$ plus 2 output biases $\mathbf{b} \in \mathbb{R}^2$).
   - **Upstream Substrate**: 1,920 fixed synaptic projections (320 sensory input weights $W_{\text{in}} \in \mathbb{R}^{40 \times 8}$, 1,600 recurrent reservoir weights $W_{\text{rec}} \in \mathbb{R}^{40 \times 40}$, of which 851 are non-zero).
   - **Dynamic State Variables**: Exactly 282 continuous state variables ($40\times v_{\text{soma}}$, $40\times v_{\text{dend}}$, $40\times g_{\text{nmda}}$, $40\times e_{\text{elig}}$, $40\times \theta_{\text{adapt}}$, $80\times e_{\text{readout}}$, $2\times a_{\text{slow}}$).
   - **Total Representation Capacity**: 2,018 parameters/variables.

2. **Genuine Surviving Mechanisms**:
   - **Persistent NMDA Conductance ($g_{\text{nmda}}$)**: Genuinely preserves context memory across intermediate horizons ($200\text{--}1600\text{ ms}$; linear probe decoding accuracy $76.0\%$, $p < 0.01$).
   - **Target-Specific Dendritic Learning & Eligibility ($B, C, A$)**: Solves credit assignment locally without global backpropagation through time.
   - **Mutual Inhibition ($I$)**: Enforces winner-take-all decisive action selection, preventing co-firing chatter.

3. **Audited Operational Limits**:
   - **5,000 ms Horizon**: Without continuous intermediate supervision leakage, autonomous accuracy decays to chance ($42.7\% \pm 1.9\%$).
   - **Continual Adaptation**: Readout plasticity lacks protected subspace consolidation; training conflicting tasks produces rapid catastrophic interference (retention ratio collapses to $0.20$).
   - **Zero-Shot Compositionality**: Pure temporal-dendritic mapping cannot perform zero-shot non-linear XOR without online adaptation (accuracy $33.3\%$).
   - **High-Interference Robustness**: Performance degrades below chance ($44.0\%$) under high distractor noise.

4. **1-to-1 Visual Observability in Audit Mode [A]**:
   - The 3D Observatory exposes actual un-leaked evaluation metrics, real measured sparse synaptic throughput ($441.8\text{M syn-ops/s}$ at $N=100\text{K}$), and real RSS memory delta ($43.2\text{ MB}$), maintaining zero cosmetic fabrications.

