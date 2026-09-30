# Phase 20 Experiment Specifications: Context-Gated Routing & Sparse Replay

**Document ID:** EXPSPEC-PHASE20-001  
**Date:** September 30, 2026  
**Auditor / Lead Investigator:** Autonomous Scientific Research Agent  
**Parent Foundation:** Phase 20 Preregistration (`research/phase20/PREREGISTRATION.md`)  
**Repository:** `~/Documents/3D-Brain`  

---

## 1. Mathematical Architecture Specifications

### 1.1 Track A: Context-Gated Modular Routing

Let $x \in \mathbb{R}^{D_{\text{in}}}$ be the upstream sensory/temporal representation produced by Stage 1–3 of MIN_E ($D_{\text{in}} = 40$ in standard configuration). Let $K_{\text{act}} = 2$ be the number of discrete action outputs, and $M = 3$ be the number of modular readout branches.

#### 1.1.1 Condition A0: Shared Linear Readout (Baseline MIN_E)
A single weight matrix $W \in \mathbb{R}^{D_{\text{in}} \times K_{\text{act}}}$:
$$u = x^T W + b$$
Synaptic update:
$$\Delta W = \eta \cdot (e \odot M_k)$$
where $e$ is dendritic eligibility and $M_k$ is the target-specific teaching signal.

#### 1.1.2 Condition A1: Fixed Modular Branches
$M$ static branches $W^{(0)}, \dots, W^{(M-1)} \in \mathbb{R}^{D_{\text{in}} \times K_{\text{act}}}$.
Without dynamic context gating, forward drive is an unweighted average:
$$u = \frac{1}{M} \sum_{m=0}^{M-1} (x^T W^{(m)} + b^{(m)})$$
Plasticity updates all branches uniformly.

#### 1.1.3 Condition A2: Context-Gated Modular Branches (Soft Gating)
A gating linear projection $W_g \in \mathbb{R}^{D_{\text{in}} \times M}$ computes branch affinity logits:
$$z_g = x^T W_g + b_g$$
$$g = \text{softmax}(z_g) \in (0, 1)^M, \quad \sum_{m=0}^{M-1} g_m = 1$$
Forward drive:
$$u = \sum_{m=0}^{M-1} g_m (x^T W^{(m)} + b^{(m)})$$
Updates modulate branches by soft gate values: $\Delta W^{(m)} = g_m \cdot \eta \cdot (e \odot M_k)$.

#### 1.1.4 Condition A3: Learned Sparse Context Gate
The gate selects the single most active branch via hard Winner-Take-All (WTA):
$$m^* = \text{argmax}_{m} (x^T W_g + b_g)$$
$$g_m = \begin{cases} 1 & \text{if } m = m^* \\ 0 & \text{otherwise} \end{cases}$$
Forward drive is computed exclusively by the routed branch:
$$u = x^T W^{(m^*)} + b^{(m^*)}$$
The gate weights $W_g$ are updated via competitive local Hebbian learning with Oja normalization:
$$\Delta W_{g, \cdot m^*} = \eta_g (x - W_{g, \cdot m^*})$$
preventing weight explosion and encouraging orthogonal context specialization.

#### 1.1.5 Condition A4: Context-Gated Branch-Specific Plasticity
Extends Condition A3 by strictly restricting plastic updates to the active branch:
$$\Delta W^{(m)} = \begin{cases} \eta \cdot (e^{(m)} \odot M_k) & \text{if } m = m^* \\ 0 & \text{if } m \neq m^* \end{cases}$$
Eligibility traces $e^{(m)}$ are maintained independently per branch. This guarantees zero gradient/synaptic cross-talk between distinct task contexts.

#### 1.1.6 Condition A5: Oracle-Gated Control (Upper Bound)
The ground-truth task context ID $c \in \{0, \dots, M-1\}$ is provided directly to select branch $m^* = c$. Evaluated strictly as a theoretical upper bound.

---

### 1.2 Track B: Sparse Hippocampal-Style Replay

#### 1.2.1 Episodic Memory Buffer
The replay buffer $\mathcal{B}$ stores up to $N_{\text{mem}} = 64$ discrete experience records.
When an action choice is committed at step $t_{\text{choice}}$, a pre-outcome record is formed:
$$r_i = \langle \text{id}_i, x_i, a_i, \tau_i, e_i \rangle$$
where $x_i$ is the context representation, $a_i$ is the chosen action, $\tau_i$ is the choice timestamp, and $e_i$ is the snapshot of dendritic eligibility traces at choice time.

When the reward outcome occurs at step $t_{\text{reward}}$, the record is updated with:
$$r_i \leftarrow r_i \cup \langle R_i, \delta_i \rangle$$
where $R_i \in \{-0.5, 1.0\}$ is the observed reward, and $\delta_i = R_i - V(x_i)$ is the prediction error / surprise magnitude.

#### 1.2.2 Replay Phase Protocol (Quiescent Rest)
Following each trial, the system enters an explicit REST period of $T_{\text{rest}} = 10\text{ steps} = 400\text{ ms}$.
During REST:
- Sensory input $x_{\text{rest}} = \vec{0}$,
- External reward $R_{\text{rest}} = 0$,
- Target label is absent,
- Plasticity is inactive except during discrete replay firing events.

#### 1.2.3 Replay Conditions
- **B0 — No Replay:** Buffer disabled; network remains quiescent during rest.
- **B1 — Random Replay:** Uniformly sample $K_{\text{replay}} = 4$ episodes from $\mathcal{B}$.
- **B2 — Event-Triggered Replay:** Replay triggers only if the latest trial had high surprise ($|\delta_i| \ge \theta_{\text{surprise}} = 0.5$).
- **B3 — Prioritized Replay:** Sample $K_{\text{replay}} = 4$ episodes with probability:
  $$P(i) = \frac{|\delta_i|^\alpha + \epsilon}{\sum_j (|\delta_j|^\alpha + \epsilon)}, \quad \alpha = 1.0, \epsilon = 10^{-4}$$
- **B4 — Event-Triggered Replay + Consolidation:** Replayed traces induce dual-timescale weight updates into slow consolidated weights $W_{\text{slow}}$.
- **B5 — Prioritized Replay + Consolidation:** Prioritized sampling combined with dual-timescale consolidation.

#### 1.2.4 Computational Replay Operation
For each replayed trace $r_k$:
1. Re-instantiate the saved dendritic eligibility trace $e_k$ and context $x_k$.
2. Compute the counterfactual teaching signal $M(a_k, R_k)$ using the saved choice and outcome.
3. Apply local synaptic update to the active readout/branch weights:
   $$\Delta W = \eta_{\text{replay}} \cdot (e_k \odot M(a_k, R_k))$$
4. Compute analytical FLOPs and record the verifiable computational event `REPLAY_EVENT` in BrainLog telemetry.

---

## 2. Experimental Task Protocols

### 2.1 Task Generator Specifications

#### 2.1.1 Task A (Standard Linear Mapping)
- Context 0: Stimulus bits $[x_0=1.0, x_2..x_4=0.6] \to$ Target Action 0.
- Context 1: Stimulus bits $[x_1=1.0, x_5..x_7=0.6] \to$ Target Action 1.

#### 2.1.2 Task B (Conflicting Inverted Mapping)
Uses identical sensory stimuli to Task A, but reverses the required action mapping:
- Context 0: Stimulus bits $[x_0=1.0, x_2..x_4=0.6] \to$ Target Action 1.
- Context 1: Stimulus bits $[x_1=1.0, x_5..x_7=0.6] \to$ Target Action 0.

#### 2.1.3 Task C (Orthogonal Sensory Mapping)
Uses non-overlapping sensory channels:
- Context 0: Stimulus bits $[x_2=1.0, x_4=0.7] \to$ Target Action 0.
- Context 1: Stimulus bits $[x_3=1.0, x_5=0.7] \to$ Target Action 1.

#### 2.1.4 Task XOR (Nonlinear Context Integration)
Specifically designed to prove nonlinear separation capability.
Two context indicator bits $(c_1, c_2) \in \{0, 1\}^2$:
- $(0, 0) \to$ Action 0
- $(0, 1) \to$ Action 1
- $(1, 0) \to$ Action 1
- $(1, 1) \to$ Action 0
Requires modular routing or nonlinear combination; cannot be resolved by a single hyperplane without catastrophic interference.

---

### 2.2 Credit-Horizon Delay Grid
Delays tested:
1. **0 ms** (0 delay steps)
2. **40 ms** (1 delay step)
3. **80 ms** (2 delay steps)
4. **120 ms** (3 delay steps)
5. **240 ms** (6 delay steps)
6. **480 ms** (12 delay steps)
7. **500 ms** (13 delay steps)
8. **1,000 ms** (25 delay steps)
9. **2,000 ms** (50 delay steps)
10. **5,000 ms** (125 delay steps)
11. **10,000 ms** (250 delay steps)
12. **30,000 ms** (750 delay steps)
13. **60,000 ms** (1,500 delay steps)

---

## 3. Quantitative Metrics

1. **Task Accuracy ($acc$):** Fraction of evaluation trials where $\text{argmax}(u) = \text{target\_action}$.
2. **Relative Retention Ratio ($\rho_{\text{ret}}$):**
   $$\rho_{\text{ret}} = \frac{acc_{A,\text{post}}}{acc_{A,\text{pre}}}$$
3. **Cross-Task Interference Index ($\mathcal{I}_{AB}$):**
   $$\mathcal{I}_{AB} = \max(0.0, acc_{A,\text{pre}} - acc_{A,\text{post}})$$
4. **Branch Selectivity ($S_{\text{branch}}$):**
   $$S_{\text{branch}} = 1.0 - \frac{H(g)}{\log(M)}, \quad H(g) = -\sum_{m=0}^{M-1} g_m \log(g_m + 10^{-12})$$
   $S = 1.0$ indicates perfect specialization (one-hot routing); $S = 0.0$ indicates complete diffusion (uniform mixture).
5. **Gate Sparsity:** Fraction of branches with activation $g_m < 0.05$.
6. **Credit Specificity Index (CSI):**
   $$\text{CSI} = \frac{|\Delta W_{\text{target}}|}{|\Delta W_{\text{target}}| + \sum_{k \neq \text{target}} |\Delta W_k|}$$
7. **FLOPs Accounting:**
   - Feedforward step: $2 D_{\text{in}} K_{\text{act}} + 2 D_{\text{in}} M$ FLOPs.
   - Replay step: $K_{\text{replay}} \cdot (4 D_{\text{in}} K_{\text{act}} + 2 K_{\text{act}})$ FLOPs.
