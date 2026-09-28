# Phase 1.1 Scientific Reproducibility & Claim Audit Report

## 1. Executive Summary

This document reports the formal scientific and implementation audit of **Phase 1** of 3D-Brain.

Phase 1 successfully engineered the first closed-loop pipeline from task simulation to neural dynamics, metrics, structured telemetry, `.brainlog` files, and interactive 3D visualization. 

However, in accordance with the core scientific principles of 3D-Brain ("Be ambitious in hypotheses. Be conservative in claims. Be ruthless about empirical validation"), this audit investigates whether the reported experimental findings and scientific conclusions are genuinely justified by the code and recorded evidence.

**Overall Audit Finding: PASS WITH CAVEATS.**
* The software execution, deterministic reproducibility, telemetry schemas, and 3D visual fidelity are verified to be 100% genuine and free of fabrication.
* However, the scientific attribution that `CAND-LI`'s performance represents an inherent superiority of "continuous-time dynamics over discrete steps" is **confounded**. The current model is integrated via discrete forward Euler with $\Delta t = 1.0$, making it mathematically equivalent to a discrete-time leaky RNN. Furthermore, a gate initialization choice in `BASE-GRU` ($b_z = +1.0$) inadvertently biased the GRU toward overwriting memory rather than preserving it.

---

## 2. Phase 0 / Phase 1 Consistency

All Phase 0 requirements and specifications were cross-checked against the Phase 1 implementation:

| Specification Document | Phase 1 Implementation Status | Audit Notes |
| :--- | :--- | :--- |
| [PROJECT_CHARTER.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/PROJECT_CHARTER.md) | **Consistent** | No LLMs, no chatbots, no multi-agent systems, no AutoForge subsystems. Parameter budgets strictly constrained. |
| [SYSTEM_PRINCIPLES.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/SYSTEM_PRINCIPLES.md) | **Consistent** | Decoupled execution (headless benchmarks run with zero visualization tax). Symmetric telemetry without fake thoughts. |
| [RESEARCH_QUESTIONS.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/RESEARCH_QUESTIONS.md) | **Partially Consistent** | Hypothesis H1 was tested, but claim strength exceeded empirical isolation (see Section 12). |
| [TERMINOLOGY.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/TERMINOLOGY.md) | **Consistent** | Strict mathematical naming maintained throughout code and telemetry. |
| [CONCEPTUAL_MODEL.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/CONCEPTUAL_MODEL.md) | **Consistent** | 3D spatial node coordinates, splines, and signal packets directly mirror mathematical variables. |
| [VISUALIZATION_CONCEPT_AND_TELEMETRY.md](file:///Users/kalachanddash/Documents/3D-Brain/visualization/VISUALIZATION_CONCEPT_AND_TELEMETRY.md) | **Consistent** | Exact JSON frame schemas, static topology schemas, and `.brainlog` files implemented and validated. |

---

## 3. Benchmark Integrity: Experiment 001 Audit

We audited [benchmarks/delayed_match.py](file:///Users/kalachanddash/Documents/3D-Brain/benchmarks/delayed_match.py) and [experiments/runner.py](file:///Users/kalachanddash/Documents/3D-Brain/experiments/runner.py) against [experiments/EXPERIMENT_001_SPECIFICATION.md](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_001_SPECIFICATION.md):

| Dimension | Specified in EXP-001 | Implemented in Code | Status | Deviation Description |
| :--- | :--- | :--- | :---: | :--- |
| **Stimulus Encoding** | $x_1(t) \in \{-1, +1\}$, $x_2(t) = 1.0$ | `X[0:5, 0] = s_cue`, `X[0:5, 1] = 1.0` | **VERIFIED** | None |
| **Cue Phase Length** | $T_{\text{cue}} = 5$ steps | `t_cue = 5` | **VERIFIED** | None |
| **Delay Phase Inputs** | $\mathcal{N}(0, \sigma^2)$, flag $0.0$ | `rng.normal(0, 0.1)`, flag $0.0$ | **VERIFIED** | None |
| **Probe Phase Inputs** | $s_{\text{probe}} \in \{-1, +1\}$, flag $2.0$ | `X[probe, 0] = s_probe`, flag $2.0$ | **VERIFIED** | None |
| **Target Output** | $0.0$ cue/delay, $\pm 1.0$ probe | $0.0$ cue/delay, $+1$ match, $-1$ mismatch | **VERIFIED** | None |
| **Distractor Noise** | $\sigma = 0.1$ | `distractor_sigma = 0.1` | **VERIFIED** | None |
| **Tested Delays** | $T \in \{10, 25, 50, 100, 200\}$ | $T \in \{10, 25, 50\}$ | **DEVIATION** | Delays 100 and 200 were omitted from the 54-run grid to keep initial execution time bounded. |
| **Parameter Budgets** | 300, 1000, 3000 params | 300 and 1000 tested in grid | **DEVIATION** | Budget 3000 was validated in unit tests but omitted from the 54-run grid. |
| **Test Sample Size** | $M = 500$ held-out trials | $M = 200$ test trials | **DEVIATION** | 200 test samples evaluated per condition instead of 500. |
| **Random Seeds** | 5 seeds: `[42, 137, 256, 1024, 2048]` | 3 seeds: `[42, 137, 256]` | **DEVIATION** | 3 seeds evaluated instead of 5. |
| **Learning Rate** | $\text{lr} = 10^{-3}$ ($0.001$) | $\text{lr} = 5 \times 10^{-3}$ ($0.005$) | **DEVIATION** | $\text{lr} = 0.005$ used to ensure convergence within 60 epochs. |
| **Loss & Masking** | Masked MSE on response | Full sequence MSE masked by probe flag | **VERIFIED** | None |

**Verdict**: The task mechanics, input/output flags, sensory noise, and evaluation criteria are faithful to the specification. The deviations are minor operational sample-size and grid-size adjustments to bound local CPU runtime, which must be clearly documented.

---

## 4. Model Comparability Audit

We audited [brain/rnn.py](file:///Users/kalachanddash/Documents/3D-Brain/brain/rnn.py), [brain/gru.py](file:///Users/kalachanddash/Documents/3D-Brain/brain/gru.py), and [brain/liquid.py](file:///Users/kalachanddash/Documents/3D-Brain/brain/liquid.py) to assess whether experimental conditions were fair and equivalent.

### Common Conditions Maintained
* Identical batch size ($B=32$), identical training batches per epoch (15), identical total trial presentations.
* Identical AdamW optimizer implementation, weight decay ($10^{-4}$), and gradient clipping threshold ($\|\mathbf{g}\| \le 1.0$).
* Identical evaluation protocol: linear readout layer $W_{hy} a_t + b_y$ with decision threshold at $0.0$.
* All parameters initialized via Xavier normal distributions.

### Critical Implementation Asymmetries (Confounds)
1. **Additive Leaky State Persistence**:
   * `CAND-LI`: Updates state via $h_t = (1 - \alpha) h_{t-1} + \alpha u_t$. When $\alpha \to 0$, $h_t \to h_{t-1}$ unconditionally. This provides an explicit autoregressive linear skip connection across time.
   * `BASE-RNN`: Updates state via $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$. It has **no** additive autoregressive skip. For an Elman RNN to preserve memory across 50 steps of noise, $W_{hh}$ must maintain an exact orthogonal or saturation attractor, while incoming distractor noise enters with full unattenuated gain ($1.0$).
2. **GRU Update Gate Bias Initialization**:
   * In `brain/gru.py`, `self.b_z = np.ones(hidden_dim)`.
   * In Cho et al. GRUs, $h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t$, where $z_t = \sigma(\dots + b_z)$.
   * Setting $b_z = +1.0$ sets the initial replacement factor to $z_t \approx \sigma(1) = 0.73$.
   * **Consequence**: The model was initialized to overwrite $73\%$ of its state with sensory noise at every time step, leaving only $27\%$ memory retention factor ($(0.27)^{25} \approx 10^{-14}$).
   * In standard LSTM networks, the *forget* gate $f_t$ is biased to $+1.0$ to preserve memory ($c_t = f_t c_{t-1} + i_t \tilde{c}_t$). In GRU, the corresponding preservation requires $z_t \to 0$, which corresponds to $b_z \ll 0$.
   * This initialization artifact heavily penalized `BASE-GRU` at $T \ge 25$.

---

## 5. Deterministic Reproducibility Audit

To verify that results are not stochastic flukes or unseeded runs, we re-ran a subset of the matrix using the saved seeds:
* `BASE-RNN` ($b=300, d=10, s=42$): Original Acc: `1.0`, Loss: `0.0019`, MSR: `1.4637`. Re-run Acc: `1.0`, Loss: `0.0019`, MSR: `1.4637`.
* `BASE-GRU` ($b=300, d=10, s=42$): Original Acc: `1.0`, Loss: `0.7420`, MSR: `5.7353`. Re-run Acc: `1.0`, Loss: `0.7420`, MSR: `5.7353`.
* `CAND-LI` ($b=300, d=10, s=42$): Original Acc: `1.0`, Loss: `0.0204`, MSR: `6.6593`. Re-run Acc: `1.0`, Loss: `0.0204`, MSR: `6.6593`.

**Verdict**: The implementation achieves **100% bit-exact deterministic reproducibility**.

---

## 6. Result Verification Audit

Raw JSON data in [exp001_results.json](file:///Users/kalachanddash/Documents/3D-Brain/experiments/results/exp001_results.json) was recomputed independently:
* Total runs in file: **54** (18 distinct parameter/model/delay conditions $\times$ 3 seeds).
* Mean accuracy, standard deviations, parameter counts, and MSR values reported in the Phase 1 report match the machine-readable database to the final decimal place.
* Zero data discrepancies were detected.

---

## 7. Audit of the MSR Metric

In [experiments/runner.py](file:///Users/kalachanddash/Documents/3D-Brain/experiments/runner.py), MSR is computed as:
$$\text{MSR} = \frac{\|\bar{\mathbf{h}}_{+} - \bar{\mathbf{h}}_{-}\|_2}{\operatorname{std}(\mathbf{H}_{\text{delay}}) + 10^{-6}}$$
where $\bar{\mathbf{h}}_+$ and $\bar{\mathbf{h}}_-$ are the mean hidden state vectors averaged across the delay interval for trials where $s_{\text{cue}} = +1$ versus $s_{\text{cue}} = -1$.

### Scientific Assessment of MSR
1. **What it actually measures**: It is a signal-to-noise ratio of attractor centroid separation relative to overall population variance during the silent delay.
2. **Validity**: It is a valid qualitative indicator that two distinct attractor clusters formed in state space. When MSR is close to zero ($0.04\text{--}0.22$, as seen in failed RNN runs), the trajectories for $+1$ and $-1$ overlap completely.
3. **Flaws & Limitations**:
   * **Dimension bias**: The Euclidean norm $\|\bar{\mathbf{h}}_+ - \bar{\mathbf{h}}_-\|_2$ scales as $\mathcal{O}(\sqrt{d_h})$ for independent neurons. Models with larger hidden dimensions ($d_h=16$ for RNN vs $d_h=9$ for GRU) have an unnormalized metric advantage.
   * **Variance aggregation**: Computing $\operatorname{std}(\mathbf{H})$ across all units simultaneously masks whether a single unit carries the entire signal while other units are dead.
   * **Recommendation**: For future phases, normalize distance by $\sqrt{d_h}$ or use Fisher's Linear Discriminant ratio.

---

## 8. Audit of Parameter Efficiency Score (PES)

$$\text{PES} = \frac{\text{Test Accuracy}}{N_{\text{parameters}}} \times 10^3$$

### Scientific Assessment of PES
* **Finding**: PES is a **derived descriptive ratio**, not an independent scientific metric.
* Because all models achieved either ~100% or ~50% accuracy, PES simply mirrors $\frac{1000}{N_{\text{parameters}}}$ or $\frac{500}{N_{\text{parameters}}}$.
* It provides no insight into computational FLOPs, memory traffic, or algorithmic depth.
* **Recommendation**: Do not cite PES as scientific evidence of cognitive efficiency. Report raw parameter count and task accuracy directly.

---

## 9. Telemetry Scientific Integrity Audit

We audited [visualization/telemetry.py](file:///Users/kalachanddash/Documents/3D-Brain/visualization/telemetry.py) and inspected [EXP001_default.brainlog](file:///Users/kalachanddash/Documents/3D-Brain/experiments/results/brainlogs/EXP001_default.brainlog) directly:
1. **Mathematical Correspondence**:
   * At step 0: `inputs = [-1.0, 1.0]`, `activations[0] = -1.0`, `activations[1] = 1.0`. Exactly matches trial presentation.
   * At step 5: `inputs = [0.0084, 0.0]`, sensory noise injected into unit 0 matches the seeded RNG sequence.
   * At step 19 (probe): `inputs = [-1.0, 2.0]`, target `= 1.0`, output `= 0.4919`, error `= 0.5081`. Exactly matches the model forward pass.
2. **Visualizer Consumption**:
   * [player.html](file:///Users/kalachanddash/Documents/3D-Brain/visualization/player.html) binds node mesh emissive intensity directly to `frame.activations[nodeId]` via WebGL shaders.
   * Click-to-inspect HUD retrieves values directly from `frame.activations` and `frame.states`.
3. **No Synthetic Artifacts**:
   * Verified that no idle random animations, particle noise, or fake cognitive strings exist in the rendering loop.

---

## 10. Replay Integrity Audit

The `.brainlog` trace format was audited for replay capability:
* **Supported Capabilities**:
  * Sequential playback at variable speeds ($0.5\times, 1.0\times, 2.0\times$).
  * Pause, resume, and arbitrary step scrubbing.
  * Raycasted neuron selection displaying live activation and connection counts.
  * Event markers for phase transitions (`CUE_START`, `DELAY_START`, `PROBE_START`).
* **Unsupported Capabilities (Future Phases)**:
  * Backward causal attribution path traversal (tracing upstream edges responsible for an error).
  * Dynamic structural plasticity deltas ($\Delta W_{ij}(t)$ per frame).
  * Branching counterfactual scrubbing.

---

## 11. Scientific Claim Classification Table

| Claim | Empirical Evidence | Scientific Status |
| :--- | :--- | :---: |
| **Experiment 001 is deterministic** | Seed reproducibility tests produced bit-exact identical losses, test accuracies, and MSR values. | **Supported** |
| **Models were compared under identical data conditions** | Same task generator, seeds, sequence length, loss function, and evaluation masks. | **Supported** |
| **Models were structurally equivalent** | `CAND-LI` had leaky persistence absent in `BASE-RNN`; `BASE-GRU` had update gate bias initialized to $b_z = +1.0$. | **Not Supported (Confounds Present)** |
| **CAND-LI solves tested delays ($T \le 50$)** | 100% accuracy on all 18 runs ($T=10, 25, 50$, budgets 300 & 1000, seeds 42, 137, 256). | **Supported** |
| **CAND-LI generalizes to arbitrary temporal memory** | Only $T \le 50$ tested with binary cue and Gaussian noise. Delays $T > 50$, continuous cues, or variable delays per trial were not tested. | **Unknown** |
| **Continuous-time dynamics cause the observed advantage** | `CAND-LI` is simulated via discrete forward Euler with $\Delta t = 1.0$, which is mathematically identical to a discrete-time Leaky RNN with learnable autoregressive parameter $\alpha$. | **Unknown / Confounded** |
| **CAND-LI is generally superior** | Superiority was demonstrated only on static delayed comparison under noise. Continual adaptation, sequence prediction, and multi-step reasoning have not been tested. | **Not Established (Premature)** |
| **Telemetry represents actual computation** | Trace audit confirmed frame arrays match model state and activation outputs exactly. Zero synthetic filler. | **Supported** |
| **3D visualization is derived from actual state** | Browser inspection verified visual node luminance, colors, and HUD values reflect `.brainlog` numerical data. | **Supported** |
| **Brain exhibits general cognition** | Only working memory persistence on binary match-to-sample was evaluated. | **Not Established** |

---

## 12. Confounding Factors Identified

1. **The Leaky Integration Equivalence Confound**:
   * Forward Euler integration of $\tau \frac{dh}{dt} = -h + u$ with $\Delta t = 1.0$ is:
     $$h_t = (1 - \alpha) h_{t-1} + \alpha u_t$$
   * This is mathematically a discrete-time Leaky RNN. Claiming that this demonstrates a "continuous-time" advantage over "discrete-time" systems conflates leaky autoregressive recurrence with continuous dynamics.
2. **The GRU Gate Initialization Confound**:
   * Setting $b_z = +1.0$ in `BASE-GRU` penalized it by setting initial retention to $27\%$. A memory-biased initialization ($b_z = -2.0 \implies 88\%$ retention) was not tested.
3. **The Static Delay Confound**:
   * Delays were fixed within each batch ($T=10, 25, 50$), allowing time constants $\tau_i$ to tune to a specific expected interval rather than handling dynamically variable delays per trial.

---

## 13. What Phase 1 Actually Demonstrates

1. **Software & Observability Feasibility**: A complete, headless, deterministic computational neural substrate can emit structured telemetry that drives an interactive, faithful 3D WebGL visualizer with zero synthetic animation.
2. **Autoregressive Leaky Integration Solves Delayed Comparison with 301 Parameters**: A compact 301-parameter recurrent network with learnable state decay factors $\alpha_i \in (0, 1)$ completely solves Delayed Match-to-Sample across $T \le 50$ under Gaussian distractor noise.
3. **Standard Elman RNNs Suffer Catastrophic Attractor Decay**: Without leaky state persistence, standard Elman RNNs fail to reliably maintain working memory attractors across $T \ge 25$ under sensory distractor noise.

---

## 14. What Phase 1 Does NOT Demonstrate

1. It does **not** prove that continuous-time neural ODEs are superior to discrete gated architectures when gates are appropriately initialized.
2. It does **not** prove that `CAND-LI` can perform continual learning, rule switching, or non-stationary adaptation (models were trained offline via BPTT).
3. It does **not** prove that 3D-Brain exhibits cognitive flexibility, associative memory, or multi-step reasoning.
4. It does **not** test spatial metric constraints (connections were dense rather than constrained by 3D Euclidean distance).

---

## 15. Minimum Next Experiment: Experiment 002

To resolve the key confounds identified in this audit, we specify [experiments/EXPERIMENT_002_SPECIFICATION.md](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_002_SPECIFICATION.md):
* **Objective**: Isolate leaky recurrence from continuous-time ODE dynamics, and isolate GRU gate initialization from architectural limitations.
* **Models**:
  1. `BASE-RNN` (Standard Elman, no leak)
  2. `DISC-LRNN` (Explicit discrete-time Leaky RNN with learnable leak $\alpha$)
  3. `CAND-LI` (Continuous-time ODE formulation with learnable $\tau$)
  4. `BASE-GRU-Default` ($b_z = +1.0$)
  5. `BASE-GRU-MemBias` ($b_z = -2.0$)
* **Delays**: $T \in \{10, 25, 50, 100\}$ steps.

---

## 16. Recommended Phase 2 Entry Conditions

Before committing to desktop rendering frameworks or complex multi-population architectures in Phase 2:
1. Accept the Phase 1.1 audit report and acknowledge the leaky equivalence and GRU initialization confounds.
2. Execute **Experiment 002** to empirically establish whether discrete leaky recurrence is sufficient or whether continuous-time formulations provide distinct mathematical advantages.
3. Maintain the decoupled telemetry contract so that headless scientific ablations continue to run with zero visualizer overhead.
