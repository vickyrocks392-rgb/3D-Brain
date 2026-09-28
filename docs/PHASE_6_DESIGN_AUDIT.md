# Phase 6 Design Audit: Persistent Latent Memory & Predictive State

**Project:** 3D-Brain Research Program  
**Phase:** Phase 6  
**Date:** September 2026  
**Status:** COMPLETE AUDIT  
**Author:** 3D-Brain Research Team  

---

## 1. Executive Summary

This audit assesses the state of the 3D-Brain computational codebase at the conclusion of Phase 5 (commit `1bb412d`) and defines the exact architectural, benchmark, mathematical, and telemetry requirements for Phase 6: **Persistent Latent Memory & Predictive State**.

Phase 5 confirmed that:
1. Self-supervised sensory prediction operates effectively via local predictive plasticity and feedback alignment ($98.9\%$ error reduction vs. frozen control).
2. Active sensorimotor embodiment provides a $6.1\times$ causal predictive advantage over passive observation ($p = 4.2 \times 10^{-6}$).
3. Dynamic structural plasticity (distance-decay synaptogenesis and activity pruning) improves internal state representation decodability ($3.8\times$ improvement, $R^2 = 0.350$) and structural modularity ($Q_{\text{struct}} \to 0.620$).
4. Dynamic neuron recruitment (Dynamic-N) did not show a statistically significant advantage over fixed-capacity synaptic rewiring ($p = 0.58$).
5. Structural modularity does not imply functional modularity ($Q_{\text{func}} \approx -0.004$ to $-0.015$).
6. The substrate scales linearly to 100,000 neurons and 2,000,000 synapses on unified memory.

Phase 6 addresses the core scientific question:
> **Can the embodied substrate develop a persistent internal latent state that represents information across time, supports delayed prediction, preserves useful information after the originating sensory input disappears, and generalizes to previously unseen temporal situations?**

---

## 2. Audit of Existing Codebase

### 2.1 Current State Representation
In `brain/predictive_spatial.py` (`PredictiveSpatialRNN`):
- Internal neuronal state is single-layered: $h_t \in \mathbb{R}^H$, with activation $a_t = \tanh(h_t)$.
- Leaky integration update:
  $$h_t = (1 - \alpha) \odot h_{t-1} + \alpha \odot \tanh(W_{xh} s_t + I_{\text{arrived}} + b_h)$$
  where $\alpha = \sigma(\rho) \in (0, 1)^H$ represents learned per-unit leak rates, and $I_{\text{arrived}}$ denotes recurrent signals arriving from the spatial propagation queue.
- **Evaluation:** $h_t$ is overloaded: it must simultaneously process instantaneous incoming sensory drive $s_t$ while acting as the sole recurrent memory. Consequently, when $s_t$ changes or disappears, incoming inputs rapidly overwrite previous state trajectories. There is no decoupled or protected latent state $z_t$.

### 2.2 Current Sensory Prediction Mechanism
- Prediction head:
  $$\hat{s}_{t+1} = \text{clip}(W_{\text{pred}} [a_t; \text{onehot}(a_t)] + b_{\text{pred}}, 0, 1)$$
  where $W_{\text{pred}} \in \mathbb{R}^{(H + A) \times D_{\text{obs}}}$.
- **Evaluation:** Strictly a **one-step ahead** predictor ($k=1$). The system predicts only the immediate next observation $\hat{s}_{t+1}$. It cannot evaluate or predict multi-step horizons ($s_{t+2}, s_{t+5}, s_{t+10}, s_{t+25}$).

### 2.3 Current Temporal Memory Capability
- Memory operates solely via:
  1. Leaky integration retention timescale $\tau_i = 1 / \alpha_i$ (typically 5 to 15 timesteps).
  2. Spatial propagation delays $d_{uv} \in [1, 5]$ in the ring-buffer queue.
- **Evaluation:** Current memory cannot bridge delays of 25, 50, 100, or 500 steps. In Task A/B scenarios with intervening distractors, the state $h_t$ degrades to baseline within 20 timesteps.

### 2.4 Current Prediction Horizon
- **Horizon $k = 1$ step.** No multi-step forward projection or temporal abstraction exists.

### 2.5 Current Learning Signals
- One-step sensory prediction error:
  $$e_t = s_{t+1} - \hat{s}_{t+1}$$
- Plasticity rules:
  - `LocalPredictivePlasticity`: Feedback alignment $\Delta W_{\text{rec}} = \eta (B_{\text{FA}} e_t) \otimes h_{\text{pre}} - \lambda W$.
  - `NeuromodulatedPredictivePlasticity`: Eligibility trace $E_{uv}(t)$ gated by scalar novelty $\tanh(\|e_t\|^2)$.
  - `GradientPredictiveBaseline`: BPTT on $\|s_{t+1} - \hat{s}_{t+1}\|^2$.
- **Evaluation:** Signals are purely immediate and local in time. No multi-step predictive error, temporal difference, or latent consistency objectives exist.

### 2.6 Current Structural Plasticity
- In `brain/structural_plasticity.py`:
  - Prunes connections if $|W_{uv}| < \theta_{\text{prune}}$ with $k_{\min} \ge 1$.
  - Forms connections via Hebbian co-activation correlation $\langle h_u, h_v \rangle$ weighted by spatial distance decay $\exp(-d_{uv}/\lambda)$.
  - Bounds in/out degrees in $[k_{\min}, k_{\max}]$.
  - Recruits new neurons if recent error exceeds $\theta_{\text{grow}}$.
- **Evaluation:** Solid, deterministic, and modular. It will be retained in Phase 6 as a structural substrate mechanism.

### 2.7 Current Limitations Summary
| Dimension | Current State (Phase 5) | Limitation |
|---|---|---|
| **Latent State** | Only hidden activation $h_t$ | Overwritten by immediate sensory input |
| **Prediction Horizon** | 1 step ($s_{t+1}$) | Cannot anticipate extended temporal trajectories |
| **Memory Retention** | Leaky decay ($< 15$ steps) | Cannot perform delayed recall or sequence tracking across $50+$ steps |
| **Temporal Disambiguation** | Instantaneous mapping | Fails when current observations match but histories differ ($A \to B \to X$ vs $C \to D \to X$) |
| **Representational Analysis** | Basic linear decodability probe | Lacks RSA, state separability, history dependence, and timescale analysis |
| **Replay & UI** | 1-step prediction inspection | Does not show latent memory dynamics, multi-step traces, or history comparison |

---

## 3. Exact New Mechanisms Required for Phase 6

To address the limitations without building an LLM or using non-neural external tools, Phase 6 introduces:

1. **Persistent Latent State Architecture (`brain/latent_memory.py` & `PersistentLatentSpatialRNN`)**:
   - Explicit latent state vector $z_t \in \mathbb{R}^{D_z}$ decoupled from the raw recurrent activation $h_t$.
   - Multi-timescale persistent latent dynamics:
     $$z_t = \lambda_z \odot z_{t-1} + (1 - \lambda_z) \odot \tanh(W_{hz} h_t + W_{zz} z_{t-1} + W_{az} a_t + b_z)$$
     with heterogeneous retention vectors $\lambda_z \in [0.90, 0.999]^{D_z}$, enabling sustained attractor dynamics.
   - Decoupled feedback projection from $z_t$ back to the recurrent spatial substrate $h_t$:
     $$I_{\text{latent}} = W_{zh} z_t$$
   - Multi-step prediction head:
     $$\hat{s}_{t+k} = \mathcal{P}_k(z_t, a_t, \dots, a_{t+k-1})$$
     supporting direct or autoregressive rollouts for $k \in \{1, 2, 5, 10, 25\}$.

2. **Temporal Memory Task Suite (`benchmarks/temporal_memory.py`)**:
   - **Task A (Delayed Cue Recall):** Sensory cue presentation, variable distractor delay $\Delta t \in \{10, 25, 50, 100, 250, 500\}$, cue reconstruction.
   - **Task B (Sequence Reconstruction):** Presentation of sequences $A \to B \to C \to D$ with overlapping elements; query element order and sequence identity.
   - **Task C (Temporal Context Disambiguation):** Divergent histories producing identical current observation:
     $$\text{History 1: } A \to B \to X \to Y_1$$
     $$\text{History 2: } C \to D \to X \to Y_2$$
     Tests whether latent state encodes historical context unavailable in current observation $X$.
   - **Task D (Multi-Step Prediction):** Continuous state prediction across horizons $k \in \{1, 2, 5, 10, 25\}$.

3. **Representational Analysis Module (`brain/representation_analysis.py`)**:
   - Linear decoding probes across time.
   - Representational Similarity Analysis (RSA).
   - State separability (Fisher Discriminant Ratio / Silhouette between histories).
   - Temporal stability (autocorrelation decay).
   - History dependence quantification.
   - Effective dimensionality (Participation Ratio).
   - Spatial modularity and participation coefficient analysis.

4. **Observability & `.brainlog` v2.4**:
   - New event types: `LATENT_STATE_UPDATED`, `MEMORY_RETAINED`, `MEMORY_DECAYED`, `MULTISTEP_PREDICTION`, `CONTEXT_RETRIEVED`, `MEMORY_ABLATION`.
   - Frame telemetry capturing `latent_z`, `latent_norm`, `multistep_preds`, and `context_id`.

5. **Replay & 3D Visualization Upgrades**:
   - Causal replay methods for latent state inspection, multi-step prediction curves, and side-by-side history comparison.
   - Web console UI updates showing latent memory HUD, temporal trace charts, and history disambiguation tracks.

---

## 4. Preservation & Backward Compatibility Matrix

| Prior Artifact | Current Version | Phase 6 Status | Action |
|---|---|---|---|
| `brain/base.py` | v1.0 | Compatible | Preserved without changes |
| `brain/discrete_leaky_rnn.py` | v2.0 | Compatible | Preserved |
| `brain/gru.py` | v2.0 | Compatible | Preserved |
| `brain/plasticity.py` | v2.1 | Compatible | Preserved |
| `brain/spatial.py` | v2.2 | Compatible | Preserved |
| `brain/predictive_plasticity.py` | v2.3 | Compatible | Preserved; extended for latent updates |
| `brain/structural_plasticity.py` | v2.3 | Compatible | Preserved |
| `benchmarks/embodied_environment.py` | v2.3 | Compatible | Preserved |
| `observability/brainlog.py` | v2.3 | Validates v1.0–v2.3 | Upgraded to validate v2.4 while preserving v1.0–v2.3 |
| `observability/events.py` | v2.3 | Enum extended | New Phase 6 events added |
| `observability/replay.py` | v2.3 | Compatible | Extended with latent state inspection and history comparison |
| `visualization/player.html` | v2.3 | Compatible | Upgraded with Latent Memory HUD and History Comparison |

---

## 5. Audit Approval

The design audit is complete. Phase 6 implementation may proceed according to the formal specification.
