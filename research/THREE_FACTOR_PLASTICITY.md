# Three-Factor Plasticity & Temporal Credit Assignment in 3D-Brain

## 1. Overview and Theoretical Formulation

Phase 9 investigates whether biological three-factor plasticity rules can enable a spatial, spiking, Dale-constrained recurrent neural network to solve temporal credit assignment problems and carve functional modularity ($Q_{\text{func}}$).

In classical two-factor plasticity (e.g., standard Hebbian learning or pair-based STDP), synaptic weight updates depend solely on local pre- and post-synaptic activity:
$$\Delta w_{ij}(t) = f(\text{pre}_i(t), \text{post}_j(t))$$
As demonstrated conclusively in Phase 8, two-factor STDP is functionally passive: when tested under balanced excitation/inhibition and homeostatic scaling, disabling STDP produced negligible changes in functional modularity ($Q_{\text{func}} \approx 0.203$ vs $0.203$, $p > 0.5$) and context disambiguation accuracy ($58.3\%$).

Three-factor plasticity introduces a third, extrinsic signal $M(t)$—representing neuromodulation, reward prediction error (RPE), or global task error—that gates and scales local eligibility traces $e_{ij}(t)$:
$$e_{ij}(t) = \lambda_e e_{ij}(t-1) + f(\text{pre}_i(t), \text{post}_j(t))$$
$$\Delta w_{ij}(t) = \eta \cdot M(t) \cdot e_{ij}(t)$$

Where:
* $e_{ij}(t)$ is the local synaptic eligibility trace at synapse $i \to j$.
* $\lambda_e = \exp(-\Delta t / \tau_{\text{elig}})$ is the exponential eligibility decay factor ($\tau_{\text{elig}} = 50.0\text{ ms}$).
* $f(\text{pre}_i(t), \text{post}_j(t))$ is the local coincidence or STDP pairing function.
* $M(t)$ is the neuromodulatory teaching signal.
* $\eta$ is the synaptic learning rate ($\eta = 0.005$).

---

## 2. Mathematical Specification

### 2.1 Local Synaptic Eligibility Traces

Eligibility traces are strictly localized to existing physical synapses defined by the structural adjacency mask $A_{ij} \in \{0, 1\}$. Disconnected pairs maintain $e_{ij}(t) \equiv 0$ identically to prevent non-local phantom credit assignment.

#### Mode A: Coincidence Eligibility
$$e_{ij}(t) = \lambda_e e_{ij}(t-1) + S_i(t) S_j(t) \quad \forall (i, j) \text{ where } A_{ij} = 1$$
where $S_i(t), S_j(t) \in \{0, 1\}$ represent instantaneous pre- and post-synaptic action potentials.

#### Mode B: Asynchronous STDP Eligibility
Pre- and post-synaptic spike histories are tracked via decaying exponential traces:
$$x_i(t) = \lambda_+ x_i(t-1) + S_i(t), \quad \lambda_+ = \exp(-\Delta t / \tau_+)$$
$$y_j(t) = \lambda_- y_j(t-1) + S_j(t), \quad \lambda_- = \exp(-\Delta t / \tau_-)$$
The eligibility trace accumulates asymmetric causal STDP increments:
$$e_{ij}(t) = \lambda_e e_{ij}(t-1) + \left[ A_+ x_i(t) S_j(t) - A_- S_i(t) y_j(t) \right]$$

---

### 2.2 Third-Factor Neuromodulatory Signals

Four teaching signal modes were implemented:

1. **Global Reward ($M_{\text{global}}$):**
   $$M(t) = R(t)$$
   Scalar extrinsic feedback delivered upon goal evaluation.

2. **Prediction Error ($M_{\text{error}}$):**
   $$M(t) = |y(t) - \hat{y}(t)|$$
   Instantaneous sensory mismatch or rollout prediction error.

3. **Signed Reward Prediction Error ($M_{\text{RPE}}$):**
   $$M(t) = \delta(t) = R(t) - \bar{R}(t)$$
   $$\bar{R}(t) \leftarrow (1 - \alpha) \bar{R}(t-1) + \alpha R(t)$$
   Where $\bar{R}$ is an exponential moving average baseline ($\alpha = 0.05$). Positive prediction errors reinforce eligible synapses, while negative prediction errors induce heterosynaptic depression.

4. **Localized Community Neuromodulation ($M_{\text{local}}$):**
   $$M_c(t) = R_c(t) - \bar{R}_c(t)$$
   Spatially partitioned teaching signals delivered specifically to distinct 3D topological modules.

---

### 2.3 Strict Dale's Principle and Weight Bounding

Plasticity cannot violate Dale's Principle. Excitatory neurons ($80\%$) strictly project non-negative weights, and inhibitory neurons ($20\%$) strictly project non-positive weights:
$$w_{ij}(t) = \begin{cases}
\text{Clip}\left(w_{ij}(t-1) + \Delta w_{ij}(t),\, 0,\, w_{\text{max}}^E\right) & \text{if neuron } i \text{ is Excitatory} \\
\text{Clip}\left(w_{ij}(t-1) + \Delta w_{ij}(t),\, -w_{\text{max}}^I,\, 0\right) & \text{if neuron } i \text{ is Inhibitory}
\end{cases}$$

---

### 2.4 Interaction with Homeostatic Synaptic Scaling

Homeostatic scaling operates concurrently on recurrent synapses to stabilize long-term firing rates:
$$w_{ij}(t) \leftarrow w_{ij}(t) \cdot \left(1 + \eta_{\text{homeo}} (r_{\text{target}} - \bar{r}_j(t))\right)$$
When three-factor plasticity potentiates recurrent pathways, homeostatic scaling acts as a negative-feedback damper, preventing epileptic runaway while preserving relative synaptic strengths.
