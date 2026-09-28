# 3D-Brain — Online Plasticity & Continual Adaptation
## Comprehensive Theoretical Review & Mathematical Formulations

**Document**: `research/ONLINE_PLASTICITY_REVIEW.md`  
**Phase**: Phase 3  
**Status**: APPROVED RESEARCH FOUNDATION  

---

## 1. Introduction: The Stability-Plasticity Dilemma

In artificial neural systems, parameters are typically adjusted offline via Backpropagation Through Time (BPTT) and subsequently frozen during evaluation:
$$\text{Data} \xrightarrow{\text{BPTT}} \theta^* \xrightarrow{\text{Freeze}} \text{Inference}$$

While this regime yields stable representations on stationary data, it fails in non-stationary environments. Sequential training on a new task distribution $D_B$ following an initial distribution $D_A$ causes **catastrophic forgetting**: gradient updates under $D_B$ overwrite the weight configurations supporting $D_A$:
$$\mathbb{E}_{(x, y) \sim D_A} [\mathcal{L}(f(x; \theta_B), y)] \gg \mathbb{E}_{(x, y) \sim D_A} [\mathcal{L}(f(x; \theta_A), y)]$$

Biological neural circuits resolve this through **continual online local plasticity**, governed by local pre- and post-synaptic activity, neuromodulatory gating, and homeostatic setpoints:
$$\text{Experience} \to \text{State Dynamics } x(t) \to \text{Local Synaptic Change } \Delta W(t) \to \text{Adaptive Behavior}$$

The core objective of Phase 3 is to determine whether small recurrent computational substrates ($N \le 3000$ parameters) can exploit local online plasticity rules to continuously adapt to changing environmental demands while retaining prior knowledge without global backpropagation.

---

## 2. Taxonomy of Local Synaptic Plasticity Rules

A synaptic plasticity rule is defined as **strictly local** if the update to connection $W_{ij}$ (from pre-synaptic unit $i$ to post-synaptic unit $j$) depends exclusively on quantities physically or computationally co-located at that synapse:
$$\Delta W_{ij}(t) = \mathcal{F}\left( a_i(t), a_j(t), W_{ij}(t), M(t), \mathbf{s}_j(t) \right)$$
where:
* $a_i(t)$ is pre-synaptic activation.
* $a_j(t)$ is post-synaptic activation.
* $W_{ij}(t)$ is the current synaptic efficacy.
* $M(t)$ is an optional scalar neuromodulatory/reward/prediction-error broadcast.
* $\mathbf{s}_j(t)$ is local post-synaptic state (e.g. running average activity).

Importantly, $\mathcal{F}$ does **not** depend on downstream synaptic weights $W_{jk}$ or global adjoint gradient signals $\frac{\partial \mathcal{L}}{\partial a_k}$ backpropagated across arbitrary graph paths.

---

## 3. Mathematical Formulations of Investigated Rules

### 3.1 Rule A: Frozen Baseline (Control)
$$\Delta W_{ij}(t) = 0, \quad \forall i, j, t$$
The frozen baseline represents the standard artificial intelligence paradigm: weights are fixed after initial training. It serves as the experimental control for measuring zero-plasticity retention, forward transfer, and adaptation failure.

---

### 3.2 Rule B: Plain Hebbian Plasticity
Formulated from Donald Hebb's 1949 postulate ("neurons that fire together wire together"):
$$\Delta W_{ij}(t) = \eta \cdot a_i(t) \cdot a_j(t)$$
where $\eta > 0$ is the learning rate.

#### Theoretical Properties:
* **Linear Positive Feedback**: If $W_{ij} > 0$, high activity in $a_i$ drives higher activity in $a_j$, which in turn increases $W_{ij}$.
* **Instability & Divergence**: Because $\Delta W_{ij} \ge 0$ whenever $a_i$ and $a_j$ share the same sign, unbounded correlation leads to exponential growth of synaptic weights:
  $$\|W(t)\| \to \infty \quad \text{as } t \to \infty$$
* **Saturation**: Hidden units saturate against nonlinear bounds ($\tanh \approx \pm 1.0$), destroying dynamic range and temporal selectivity.
* **Stabilization Mechanism**: Hard bounding $W_{ij} \in [W_{min}, W_{max}]$ and decay $-\lambda W_{ij}$.

---

### 3.3 Rule C: Oja's Normalized Plasticity
Erkki Oja (1982) derived a self-stabilizing rule that performs online principal component analysis (PCA) through a first-order Taylor expansion of normalized Hebbian learning:
$$\Delta W_{ij}(t) = \eta \cdot a_j(t) \cdot \left[ a_i(t) - a_j(t) W_{ij}(t) \right]$$

Expanding the terms:
$$\Delta W_{ij}(t) = \underbrace{\eta a_i(t) a_j(t)}_{\text{Hebbian Reinforcement}} - \underbrace{\eta a_j(t)^2 W_{ij}(t)}_{\text{Post-synaptic Weight Decay}}$$

#### Theoretical Properties:
1. **Asymptotic Convergence to Unit Norm**: For stationary inputs, the weight vector $\mathbf{w}_j$ converges to the principal eigenvector of the input correlation matrix:
   $$\lim_{t \to \infty} \|\mathbf{w}_j(t)\|_2 = 1$$
2. **Local Weight Decay**: Weight decay is proportional to the square of post-synaptic activity $a_j^2$, scaling automatically with firing rates without requiring external norm constraints.
3. **Selectivity**: Synapses aligned with dominant input fluctuations strengthen, while orthogonal synapses decay.

---

### 3.4 Rule D: Neuromodulated / Reward-Gated Three-Factor Plasticity
In biological circuits, diffuse ascending projection systems (dopamine, acetylcholine, noradrenaline) broadcast global behavioral significance signals that gate local Hebbian changes. This is formalized as **three-factor learning**:
$$\Delta W_{ij}(t) = M(t) \cdot \eta \cdot \mathcal{H}(a_i(t), a_j(t), W_{ij}(t))$$
where:
* Factor 1: Pre-synaptic activity $a_i(t)$.
* Factor 2: Post-synaptic activity $a_j(t)$.
* Factor 3: Neuromodulatory gating signal $M(t) \in \mathbb{R}$.

For our continual adaptation studies, the modulation signal is defined as an instantaneous performance or surprise metric:
$$M(t) = \begin{cases} 
\tanh\left( \delta(t) \right) & \text{if error / novelty present} \\
0 & \text{during stable quiescent processing}
\end{cases}$$
where $\delta(t) = |y(t) - \hat{y}(t)|$ is the local prediction error, or an environmental transition flag.

#### Theoretical Properties:
1. **Gated Learning**: When performance is optimal ($\delta(t) \approx 0$), $M(t) \approx 0$, freezing synaptic weights and protecting existing memories from drift.
2. **Targeted Adaptation**: When distribution shift occurs, $\delta(t)$ spikes, enabling rapid synaptic reconfiguration exclusively when needed.
3. **Catastrophic Forgetting Mitigation**: Prevents continuous overwriting during steady-state exploitation.

---

### 3.5 Rule E: Bounded Homeostatic Plasticity
Biological neurons maintain target firing rates over long timescales through **synaptic scaling** (Turrigiano et al., 1998): when a neuron is chronically hyperactive, all its incoming synapses are scaled down; when hypoactive, they are scaled up.

We formulate homeostatic plasticity as:
$$\Delta W_{ij}(t) = \eta \cdot \left[ a_j(t)(a_i(t) - a_j(t) W_{ij}(t)) \right] + \gamma \cdot \left( a^* - \bar{a}_j(t) \right) \cdot |W_{ij}(t)|$$
where:
* $a^*$ is the homeostatic target activity level (e.g. $a^* = 0.2$).
* $\bar{a}_j(t)$ is the low-pass filtered running average of post-synaptic activity:
  $$\bar{a}_j(t) = (1 - \beta) \bar{a}_j(t-1) + \beta |a_j(t)|, \quad \beta \in (0, 1)$$
* $\gamma$ is the homeostatic scaling rate ($\gamma \ll \eta$).

#### Theoretical Properties:
1. **Activity Setpoint**: Prevents quiescent collapse ($\bar{a}_j \to 0$) and chronic saturation ($\bar{a}_j \to 1$).
2. **Dynamic Range Preservation**: Ensures that all recurrent neurons remain receptive to temporal signals.
3. **Multi-Timescale Dynamics**: Fast Hebbian/Oja tracking combined with slow homeostatic stabilization.

---

## 4. Summary Comparison of Plasticity Formulations

| Rule | Mathematical Formulation | Locality | Stability Mechanism | Theoretical Risk |
| :--- | :--- | :--- | :--- | :--- |
| **Frozen** | $\Delta W_{ij} = 0$ | Perfect | Fixed by definition | Inability to adapt to non-stationary tasks |
| **Hebbian** | $\eta a_i a_j$ | Strict ($a_i, a_j$) | Hard clipping $[W_{min}, W_{max}]$ | Runaway weight explosion, early saturation |
| **Oja** | $\eta a_j (a_i - a_j W_{ij})$ | Strict ($a_i, a_j, W_{ij}$) | Intrinsic activity-squared decay | May over-specialize on highest-variance input |
| **Modulated** | $M(t) \cdot \eta \cdot \mathcal{H}_{oja}$ | Modulated ($a_i, a_j, W_{ij}, M$) | Gated by surprise / error | Requires meaningful broadcast signal $M(t)$ |
| **Homeostatic** | $\Delta W_{oja} + \gamma (a^* - \bar{a}_j) |W_{ij}|$ | Local + internal setpoint ($a^*, \bar{a}_j$) | Slow negative feedback synaptic scaling | Parametric tension between adaptation & setpoint |

---

## 5. Integration with Recurrent Dynamical Substrates

In a discrete leaky recurrent neural network (`DISC-LRNN`):
$$\tilde{h}(t) = \tanh(W_{in} u(t) + W_{rec}(t) h(t-1) + b_h)$$
$$h(t) = (1 - \alpha) \odot h(t-1) + \alpha \odot \tilde{h}(t)$$

Online plasticity operates directly on the recurrent transition matrix $W_{rec}(t)$:
$$W_{rec}(t+1) = W_{rec}(t) + \Delta W_{rec}(t)$$

Because $W_{rec}$ governs the internal state attractor landscape, local weight changes actively reshape:
1. The memory retention factor and fading memory horizon.
2. The recurrent amplification of transient input cues.
3. The functional specialization of individual sub-populations.

In Phase 3, we implement and measure this dynamical interaction directly across all 5 conditions.
