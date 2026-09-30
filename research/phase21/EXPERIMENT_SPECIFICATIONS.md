# Phase 21 Experiment Specifications: Episodic Memory, Consolidation & Transfer

**Document ID:** EXPSPEC-PHASE21-001
**Date:** September 30, 2026
**Auditor / Lead Investigator:** Autonomous Scientific Research Agent
**Parent Foundation:** Phase 21 Preregistration (`research/phase21/PREREGISTRATION.md`)
**Repository:** `~/Documents/3D-Brain`

> Written before implementation. Defines the exact mathematics, data structures,
> and protocols used by all Phase 21 experiments.

---

## 1. Substrate and Notation

Phase 21 does **not** alter the MIN_E substrate established in Phase 16 and validated
in Phases 19-20. All new mechanisms are explicit, separately instantiated modules
with their own accounting.

| Symbol | Meaning | Value |
|:--|:--|:--|
| $D_{in}$ | Sensory input dimension | 8 |
| $D_{hid}$ | Reservoir population size | 40 |
| $D_{ro}$ | Readout representation dimension | 40 |
| $K$ | Action outputs | 2 |
| $\Delta t$ | Simulation timestep | 40 ms |
| $N_{ep}$ | Number of episodes | protocol-dependent |
| $k$ | Sparse retrieval arity (top-$k$) | 3 |
| $d_{key}$ | Episode key dimensionality | 12 |

---

## 2. Track A: Episodic Memory Specification

### 2.1 Episode Record (Legitimate Content Only)

An episode record stores **event-level and latent** content, never supervised input-to-target mappings:

$$
r_i = \langle \ id_i,\ c_i,\ E_i,\ \alpha_i,\ \pi_i,\ r_i,\ \kappa_i,\ \tau_i,\ \ell_i,\ \sigma_i\ \rangle
$$

| Field | Symbol | Description | Legitimacy |
|:--|:--|:--|:--|
| Episode ID | $id_i$ | Integer handle (internal bookkeeping only; **never** an input feature) | - |
| Context representation | $c_i \in \mathbb{R}^{D_{ro}}$ | Readout input vector at episode open | Available at time $t$ |
| Event sequence | $E_i = \langle z_{i,1},\dots,z_{i,T_i}\rangle$ | Latent states at each event step | Available at time $t$ |
| Action | $\alpha_i$ | Action actually emitted at the decision step | Available at time $t$ |
| Prediction error | $\pi_i$ | $\lvert \delta_i\rvert$, filled **only at outcome** | Available at outcome time |
| Reward | $r_i$ | Scalar outcome, filled **only at outcome** | Available at outcome time |
| Key | $\kappa_i \in \mathbb{R}^{d_{key}}$ | Sparse content address derived from $c_i, E_i$ | Derived from available state |
| Timestamp | $\tau_i$ | Event time in ms | Available at time $t$ |
| Sparse activity | $\ell_i \in \{0,1\}^{N_{act}}$ | Thresholded unit-activity bitmap | Available at time $t$ |
| Duration | $\sigma_i$ | Episode length in steps | Available at time $t$ |

**Prohibited fields (asserted absent by unit test):** future target action, future reward, target-derived features, episode ground-truth label, test-set identity.

The two-phase write protocol is mandatory:

1. `open_episode(...)` at the decision step stores $c_i, E_i, \alpha_i, \kappa_i, \tau_i, \ell_i, \sigma_i$ with $\pi_i = 0$, $r_i = 0$.
2. `close_episode(outcome)` writes $\pi_i$ and $r_i$ **only after** the reward step has been observed.

### 2.2 Sparse Content Address

The key is a **sparse projection** of the concatenated context and event summary. With a fixed sparse dictionary $D \in \mathbb{R}^{d_{key} \times D_{ro}}$ (orthonormal rows, fixed at construction, counted as parameters):

$$
\phi_i = \frac{E_i}{\lVert E_i\rVert_2}, \qquad \kappa_i = \mathrm{sign}(D\phi_i) \odot \big\lvert D\phi_i\big\rvert
$$

Keys are thresholded at $\theta_{sp}$ to yield an explicit sparsity level, and L2-normalized so that similarity is a bounded inner product:

$$
### 2.3 Retrieval

Retrieval returns the **top-$k$** episodes by similarity, not a dense broadcast:

$$
\mathcal{R}(q) = \operatorname{TopK}_k\big(\{\text{sim}(q, \kappa_i)\}\big)
$$

**Retrieval sparsity** is defined as

$$
S_{ret} = \frac{k}{N_{ep}}
$$

so $S_{ret} \ll 1$ constitutes sparse retrieval, and is reported directly.

### 2.4 Memory Conditions

| ID | Condition | Retrieval rule |
|:--|:--|:--|
| **A0** | No episodic memory | `retrieve()` always returns the empty set |
| **A1** | Passive buffer | Uniform random sample of $k$ traces |
| **A2** | Indexed | Exact-match lookup on a scalar context bucket |
| **A3** | Sparse content-addressable | Top-$k$ cosine similarity over sparse keys |
| **A4** | A3 + Phase 20 replay | A3 retrieval feeding the Phase 20 replay buffer |
| **A5** | A4 + consolidation | A4 plus dual-timescale + metaplasticity consolidation |

### 2.5 Episode Discrimination Protocol

Episodes share an **identical initial observation** and differ only in the temporal event sequence and resulting action:

| Episode | Initial observation | Event sequence | Correct action |
|:--|:--|:--|:--|
| A | $x_0$ | $e_1, e_2, e_3$ | $\alpha_A$ |
| B | $x_0$ | $e_3, e_1, e_2$ | $\alpha_B$ |

The retrieval query is built from the **observed event history**, never from an episode ID. Correct retrieval requires that the representation encode *order*, not merely event membership.

Measured quantities:

$$
\text{Acc}_{ret} = \frac{1}{N}\sum \mathbb{1}\big[\arg\max_i \text{sim}(q, \kappa_i) = id_{true}\big]
$$

$$
\text{Contextual confusion} = \Pr\big[\text{retrieved episode has different context, same action}\big]
$$
$$
\text{Temporal confusion} = \Pr\big[\text{retrieved episode has same context, different event order}\big]
$$

### 2.6 Causal Ablations for Memory

| Intervention | Implementation |
|:--|:--|
| `REMOVE_MEMORY` | `retrieve()` returns the empty set |
| `RANDOMIZE_RETRIEVAL` | Return a uniformly random top-$k$ set |
| `SHUFFLE_IDENTITY` | Permute stored episode IDs independently of content |
| `WRONG_EPISODE` | Return the top-$k$ set with indices rotated by one rank |
| `REDUCE_CAPACITY` | Shrink capacity to 10 percent |

### 2.7 Memory Accounting

Reported exactly, per episode:

$$
\text{Bytes}_{ep} = \underbrace{d_{key}\cdot 4}_{key} + \underbrace{D_{ro}\cdot 4}_{context} + \underbrace{T_i\cdot D_{hid}\cdot 4}_{events} + \underbrace{N_{act}/8}_{sparse bitmap} + \underbrace{C_{scalar}\cdot 8}_{scalars}
$$
## 3. Track B: Replay Consolidation Specification

### 3.1 Quiescent Rest Protocol

Rest is explicit and strictly silent:

- Sensory input $x_{rest} = \mathbf{0}$
- No new target, no target supervision
- No future reward, no hidden supervision
- No environment interaction

Only pre-outcome stored traces may be reactivated. Rest duration is a fixed protocol parameter.

### 3.2 Dual-Timescale Consolidation

Readout weights $W$ are decomposed into a fast pool $W_f$ and a slow pool $W_s$:

$$
W_{eff} = W_f + \gamma_s W_s
$$

With consolidation rate $\gamma_c$, transfer occurs only when a synapse carries an eligibility tag above threshold:

$$
\text{if } \varepsilon_j > \theta_{tag}: \quad
W_s \leftarrow (1-\gamma_c)\,W_s + \gamma_c\, W_f, \qquad W_f \leftarrow W_f
$$

**Tagging & capture:** a synaptic tag $\tau_j$ is set at reward delivery and decays with $\tau_{tag}$:

$$
\tau_j \leftarrow \tau_j e^{-\Delta t / \tau_{tag}}
$$

Reactivation during rest restores $\tau_j$ from stored eligibility, permitting capture into the slow pool. Tags are never set by target labels alone.

### 3.3 Metaplasticity

A global **learning-rate scale** $m$ (the BCM-like metaplasticity term) modulates plasticity strength according to recent activity:

$$
m \leftarrow m + \eta_m\big(\bar{\varepsilon} - m\big), \qquad \eta_{eff} = \eta \cdot \tanh(m)
$$

This is a slow-state variable, reported explicitly in the consolidation telemetry.

### 3.4 Track B Conditions

| ID | Replay | Consolidation |
|:--|:--|:--|
| **B0** | None | None |
| **B1** | Random | None |
| **B2** | Event-triggered | None |
| **B3** | Prioritized | None |
| **B4** | Prioritized | Dual-timescale |
| **B5** | Prioritized | Metaplasticity |
| **B6** | Prioritized | Dual-timescale + metaplasticity |

### 3.5 Replay Content Audit (7 protocols)

## 4. Track C: Abstract Transfer Specification

### 4.1 Compositional Task Structure

Three **component rules** are experienced individually during training. Each maps a feature subspace to an action contribution:

| Rule | Feature subspace | Input channel mapping |
|:--|:--|:--|
| **A** | Shape | channels 0-2 |
| **B** | Color | channels 2-4 |
| **C** | Temporal relation | channels 4-6 |

**Forbidden shortcut (rule R6):** the forbidden pattern $x[\text{fixed\_index}] = \text{task\_id}$ is **never** used. All tasks share an identical sensory schema; the only difference is which feature subspaces are **active** and which rules are **bound**.

### 4.2 Composition Rule

For a combination $S \subseteq \{A,B,C\}$, the target action is the parity of the component predictions:

$$
y_S(x) = \Big(\sum_{j \in S} f_j(x)\Big) \bmod 2
$$

Individual rules $f_j$ are trained alone; combinations are **never** trained. Test sets: $A{+}B$, $A{+}C$, $B{+}C$, $A{+}B{+}C$.

### 4.3 Zero-Shot Transfer Protocol (Strict)

For every combination test:

1. Train only on individual rules $A$, $B$, $C$ (TRAIN split).
2. **Freeze** the system: `set_frozen(True)`, disable replay, disable consolidation, disable all plasticity.
3. Evaluate combinations on the TEST split with **no weight update, no replay, no target access**.
4. Report accuracy on the frozen model.

An automated assertion verifies that no weight changed and no plasticity call occurred during evaluation.

### 4.4 Nonlinear Composition

An explicit XOR-like rule over two binary components:

$$
y_{xor} = f_A(x) \oplus f_B(x)
$$

Controls: **linear control** (additive rule $f_A + f_B$), **shuffled-label control** (rule labels permuted), **unseen combination** (evaluated frozen).

The claim used is **compositional transfer**. The phrase "abstract reasoning" is **not** used unless evidence clearly exceeds compositional recombination.

### 4.5 Nonlinearity Test (Expressivity Check)

To determine whether transfer is genuine composition rather than linear interpolation, a **linear separability probe** is fit on the combined representation:

$$
\text{Separability} = \text{accuracy of a linear probe on composed inputs}
$$

If a linear probe solves the novel combination while the system does not, the limitation is identified as **readout expressivity**, not as an absence of abstraction.

### 4.6 Retrieval Dependence

Transfer is re-run with episodic memory ablated (`REMOVE_MEMORY`) and with retrieval randomized. The delta tests whether transfer **depends** on episodic retrieval (H-C2). If transfer is unchanged, the dependency claim is falsified.
### 4.7 Representation Similarity

For each pair of rules, representational similarity is measured on frozen representations:

$$\text{RS}(i,j) = \text{linear-CKA}\big(\Phi_i, \Phi_j\big)$$

Reported for trained rules and for novel combinations.

---

## 5. Track D: Integrated System Specification

| ID | Condition |
|:--|:--|
| **D0** | Phase 20 baseline (MIN_E) |
| **D1** | D0 + episodic memory |
| **D2** | D0 + replay + consolidation |
| **D3** | D0 + episodic memory + replay |
| **D4** | D0 + episodic memory + replay + consolidation |
| **D5** | D4 + context-gated routing |

**D5 is not assumed to win.** The purpose is to determine whether mechanisms are complementary, redundant, antagonistic, or task-dependent.

---

## 6. Capacity, Forgetting and Control Protocols

### 6.1 Memory Capacity Experiment

Capacities: **10, 25, 50, 100, 250, 500, 1000 episodes.**

Measured per capacity: retrieval accuracy, memory footprint in bytes, interference, replay cost, retrieval cost, consolidation cost. Memory capacity is **never hidden inside neural parameter counts**; bytes are reported explicitly.

### 6.2 Forgetting Experiment

Sequences: **A->B->A**, **A->B->C->A**, **A->B->C->D->A**.

Reported per sequence: initial A accuracy, B accuracy, C accuracy, D accuracy, final A accuracy, retention ratio $\rho$, absolute forgetting $F_{abs}$, relative forgetting $F_{rel}$ (only alongside $F_{abs}$), forward transfer, backward transfer.

### 6.3 Transfer Without Replay (Critical Control)

Four protocols, all reported:

1. `learn -> test` (no replay)
2. `learn -> rest/replay -> test`
3. `learn A -> replay A -> learn B -> test A`
4. `learn A -> no replay A -> learn B -> test A`

If replay causes consolidation, protocols 2 and 3 must differ measurably from 1 and 4.

### 6.4 Causal Ablation Matrix

| Family | Interventions |
|:--|:--|
| Episodic memory | removed, retrieval randomized, identity shuffled, wrong episode, capacity reduced |
| Replay | removed, timing randomized, content randomized, reversed, wrong episode |
| Consolidation | removed, rate altered, tagging removed, metaplasticity removed |
| Transfer | context removed, temporal information removed, nonlinear component removed |
| Integrated | remove memory, remove replay, remove consolidation, remove routing, remove all three |

---

## 7. Parameter / Memory / Compute Accounting

Reported **separately** for every system:

1. **Neural parameters** - trainable and fixed
2. **Dynamic neural state** - state variables
3. **Episodic memory** - bytes
4. **Replay buffer** - bytes
5. **Structural hyperparameters** - counts
6. **Computational workspace** - peak allocation

FLOPs are counted analytically and separately for feedforward, retrieval, replay, and consolidation steps:

$$\text{FLOPs}_{ff} = 2 D_{ro} K, \quad \text{FLOPs}_{retrieval} = 2\, d_{key}\, N_{ep}, \quad \text{FLOPs}_{replay} = 4 D_{ro} K$$

It is forbidden to claim "fewer parameters" while silently granting an external-memory advantage. Every comparison states explicitly whether external memory is included.

---

## 8. Statistical Protocol

- Seeds: **[42, 101, 2024, 777, 999]** (exact N = 5 per condition)
- Report mean, standard deviation, 95% confidence interval, effect size (Cohen's d), exact N
- Paired tests where the same seed is run under both conditions
- Holm-Bonferroni correction across the pre-registered primary family (P1-P7)
- Chance-normalized metrics only as secondary annotations, never as sole endpoints

---

## 9. Telemetry and BrainLog Specification

Real computational state is emitted for every event. No decorative or fabricated telemetry.

**Episodic memory [E]:** episode creation, retrieval event, retrieval score, selected memory, competing memories, retrieval errors.

**Replay [R]:** replay source, replay target, replay timestamp, replay content, replay sequence.

**Consolidation [C]:** synaptic changes, tagging, slow-state changes, consolidation events.

**Transfer [T]:** learned components, novel combination, retrieval activity, action routing.

All events must be replayable through the BrainLog replay engine.

---

## 10. Leakage Audit Checklist

Before finalizing, inspect specifically for:

- hidden task IDs
- future reward leakage
- future target leakage
- replay contamination
- evaluation-time plasticity
- accidental memory access
- episode-label shortcuts
- compositional-test leakage
- parameter-count errors
- memory-accounting errors
- FLOP-count errors
- chance-normalized metric errors
- train/test contamination
- statistical cherry-picking

If any problem is discovered: **stop, document, fix, rerun.** Do not preserve an attractive result.



