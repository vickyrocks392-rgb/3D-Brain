# Phase 18 — Scientific Audit & Pre-Registration Protocol

**Project:** 3D-Brain Autonomous Biophysical Intelligence  
**Phase:** 18 — Scientific Audit, Reproducibility & Benchmark Integrity  
**Date:** September 30, 2026  
**Status:** **PRE-REGISTERED BEFORE AUDIT EXECUTION**

---

### Central Scientific Question

> *Are the strongest Phase 17 results genuine properties of the Phase 16 minimal brain (MIN_E), or can they be explained by task-generation artifacts, evaluation leakage, hidden task-specific tuning, metric/probe leakage, benchmark accounting errors, insufficient controls, or implementation details?*

The purpose of this audit protocol is to establish rigorous, unalterable criteria for falsification. The goal is not to defend Phase 17, but to aggressively attempt to falsify its claims.

---

### 1. Pre-Registered Audit Criteria

#### Criterion A: What Counts as Reproduced?
A result is considered **Reproduced** if:
1. Re-running the original evaluation code with the original seeds produces metrics within defined numerical tolerance.
2. Re-running under an **independent evaluator** that eliminates all label leakage, online test adaptation, and shortcut channels produces statistically equivalent results ($p > 0.05$ difference or non-overlapping confidence intervals that maintain the headline scientific claim).

#### Criterion B: Acceptable Statistical Tolerance
- Deterministic seed reproduction under original harness: exact match ($\Delta = 0.00\%$) for identical random number generators.
- Stochastic multi-seed independent evaluation:
  - Accuracy: within $\pm 5.0\%$ of mean.
  - CSI / Credit leakage: within $\pm 0.05$.
  - Margins / switches: within $\pm 0.20$.

#### Criterion C: What Constitutes Evidence of Leakage?
Evaluation or task leakage is definitively established if:
1. **Target / Label Leakage:** The ground-truth target action $y^*$ or reward $R$ is passed to the network's plastic step during steps prior to the choice/reward trigger (e.g. during cue or delay intervals), allowing synaptic updates to occur before the decision is committed.
2. **Causal Invariance Failure (Leakage Indicator):** If scrambling, shuffling, or zeroing the sensory input during delay or cue preserves high performance, the model is extracting target information through unintended channels (such as label leakage, fixed bias, or state persistence).
3. **Shortcut Channels:** An input channel exists that uniquely identifies the ground truth label or combination without requiring the claimed latent computation (e.g. explicit combination index channels).
4. **Online Test Plasticity:** Readout weights or internal states are modified during evaluation episodes without an explicit frozen evaluation regime.

#### Criterion D: What Constitutes Benchmark Accounting Failure?
A benchmark accounting failure is established if:
1. **Theoretical Operations Counted as Executed:** Multiplying step count by a hypothetical number of neurons/synapses without actually simulating, allocating, or updating those neurons/synapses in hardware.
2. **Substrate Dimensionality Clamping:** Restricting internal matrix dimensions (e.g. `dim = min(N, 120)`) while reporting throughput for a larger $N$ (e.g. $N = 100,000$).
3. **Memory Omission:** Calculating theoretical weight memory ($N \times \text{synapses} \times 4\text{ bytes}$) while ignoring process RSS, state buffers, Python object overhead, and telemetry.
4. **Baseline Disparity:** Comparing trainable parameter counts (e.g. 80 weights) while omitting fixed upstream weights (1,920 parameters), structural parameters, and state variables when comparing against classical baselines (BPTT, Transformers).

#### Criterion E: What Constitutes a Failed Generalization Claim?
A generalization claim is classified as **Failed / Not Supported** if:
1. Performance on held-out distributions, novel compositions, or extrapolated horizons falls to or below chance level (50% on binary choice).
2. Performance relies upon test-time plastic adaptation or continuous supervised guidance rather than invariant latent representation.
3. Performance on "robustness" tests is at or below chance level while being characterized as robust.

#### Criterion F: What Constitutes Insufficient Evidence?
A claim is classified as **Inconclusive / Insufficient Evidence** if:
1. The sample size ($n$) or seed count is too small to distinguish true signal from stochastic noise ($p \ge 0.05$ against a majority or random baseline).
2. Effect size (Cohen's $d < 0.2$) is negligible despite nominal metric differences.

---

### 2. Pre-Registered Hypotheses and Classification Matrix

Each Phase 17 headline claim will be evaluated against independent tests and classified into one of the following scientific categories:
- **ROBUSTLY REPRODUCED:** Survives independent audit, un-leaked evaluation, and matched controls.
- **REPRODUCED:** Matches numerical baseline under original conditions, but exhibits documented constraints.
- **PARTIALLY REPRODUCED:** Valid in a restricted regime, but fails broader generalization.
- **NOT REPRODUCED:** Fails under independent evaluation or held-out seeds.
- **METRICALLY INVALID:** Metric definition was mathematically flawed (e.g. chance-level labeled as robust or 0% forgetting on unlearned task).
- **IMPLEMENTATION ARTIFACT:** Result was caused by leakage, shortcut channels, or synthetic benchmark extrapolation.

| Claim | Phase 17 Headline | Audit Test Condition | Downgrade / Falsification Trigger |
|:---|:---|:---|:---|
| **Claim 1: 5000 ms Horizon** | 100% extrapolation accuracy at 5000 ms | Un-leaked delay (target provided only at reward step); disjoint delay ranges | Accuracy drops significantly below 100% when delay label leakage is eliminated. |
| **Claim 2: Continual Learning Stability** | 0% catastrophic forgetting (retention = 1.00) | Baseline task learning validation; alternating & abrupt shifts; representation drift | Initial accuracy was at chance (50%), meaning retention = 1.00 is a division-by-chance artifact. |
| **Claim 3: Compositional Novelty** | 76.7% accuracy on novel held-out XOR combination | Pure composition (no `x[4+comb_id]` shortcut); frozen test evaluation | Accuracy drops to chance or below when frozen or shortcut removed. |
| **Claim 4: Distractor Robustness** | Robust to high noise (44.0% accuracy, CSI preserved) | Comparison to 50% chance baseline & majority predictor | Performance is at or below 50% chance, rendering "robustness" metrically invalid. |
| **Claim 5: 100K Throughput** | 107,989 steps/s, 215.98B syn-ops/s, 7.63 MB | True full-scale allocation ($N=100K$ state vectors); actual executed sparse ops | Benchmark simulated only $N=120$ and linearly multiplied operations. |
| **Claim 6: Parameter Count** | MIN_E = 80 trainable parameters | Full parameter census: fixed weights, structural parameters, state variables | Fixed upstream weights (1,920) and structural parameters hidden from comparison. |
| **Claim 7: Mechanism Universality** | All 8 retained mechanisms are universally necessary | Independent re-ablation across diverse tasks | Redundant or task-dependent mechanisms identified. |
| **Claim 8: Task Generality** | Single frozen MIN_E substrate generalizes across 6 families | Zero-shot & few-shot testing without task-specific hyperparameter retuning | Low zero-shot transfer (~50%) without retuning. |

---

### 3. Execution Mandate

The Phase 18 audit must execute all tests using strictly deterministic independent generators, verified held-out seeds, and separate evaluation code. The original Phase 17 code must remain untouched as the reference benchmark, while audit code is isolated in independent audit modules.
