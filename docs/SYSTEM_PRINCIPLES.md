# 3D-Brain: System Principles & Architecture Philosophy

## 1. Co-Design of Computation and Observability

The 3D neural world is not a post-hoc aesthetic dashboard glued onto an uninterpretable black box.

1. **Symmetric Evolution**: Any internal state variable in the computational substrate (membrane potential, prediction error vector, synaptic trace, neuromodulatory concentration) must have a defined mapping into the telemetry stream.
2. **Faithful Telemetry**: The visualization displays only what actually happens computationally. If a population is inactive, it is dark. If a synapse does not change, its visual attributes remain static. There are no "idle animation loops" simulating fake computational vitality.
3. **Spatial Semantics**: Spatial arrangements of neurons (layers, columns, clusters, reservoirs) reflect actual topological connectivity, propagation delays, and functional modularity.

```text
                 ┌───────────────────────────┐
                 │         3D-BRAIN          │
                 │     macOS Desktop App     │
                 └─────────────┬─────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
       COMPUTATIONAL BRAIN                 3D NEURAL WORLD
              │                                 │
       neural units                         neurons
       connections                           synapses
       activation                            signals
       memory                                activity
       prediction                            dynamics
       learning                              structure
       adaptation                            interaction
              │                                 │
              └────────────────┬────────────────┘
                               │
                         SHARED LIVE STATE
                               │
                         OBSERVABILITY API
```

---

## 2. Decoupled Operating Modes

The computational substrate and the visualization engine must be strictly decoupled across three execution modes:

```text
LIVE MODE
┌─────────────────┐       Telemetry Stream       ┌───────────────────┐
│ Computational   │ ───────────────────────────> │ Interactive 3D    │
│ Brain Engine    │ <─────────────────────────── │ Renderer (Metal/  │
└─────────────────┘       Interactive Input      │ WebGPU/Native)    │
                                                 └───────────────────┘

REPLAY MODE
┌─────────────────┐       Telemetry Log          ┌───────────────────┐
│ Serialized      │ ───────────────────────────> │ Replay Engine &   │
│ State Trace     │                              │ Time-Travel UI    │
└─────────────────┘                              └───────────────────┘

EXPERIMENT & BENCHMARK MODE (Headless)
┌─────────────────┐       Execution & Eval       ┌───────────────────┐
│ Computational   │ ───────────────────────────> │ Metrics, Tables,  │
│ Brain Substrate │                              │ Statistical Logs  │
└─────────────────┘                              └───────────────────┘
```

* **Zero Visualization Tax**: Running headless benchmarks on CPU or GPU compute must incur zero overhead from rendering subsystems.
* **Trace Portability**: A run recorded during headless experimentation can be loaded into the desktop visualizer for deep temporal inspection.

---

## 3. Cognitive Time Travel

Biological and artificial brains make decisions based on trajectories through phase space. Understanding *why* an event occurred requires traversing time.

* **Bidir-Stepping**: The system architecture must support scrubbable state history: stepping backward step-by-step ($t \to t-1$), jumping to critical divergence points, and inspecting causal pre-conditions.
* **Checkpoint & Delta Logging**: Store keyframe state snapshots combined with lightweight event delta logs to enable long-trajectory replay without unbounded memory consumption.
* **Causal Attribution**: The user should be able to click any prediction error burst or decision node and trace backward along the active incoming synapses to observe which precursor signals caused the transition.

---

## 4. Parameter Parsimony: "Earn Every Parameter"

* Scale is a scientific question, not a badge of honor.
* Target investigation tiers:
  * **Tier 1 (Sub-10K)**: Primitive dynamics, temporal memory, delay lines, simple oscillator loops.
  * **Tier 2 (10K – 50K)**: Sequence prediction, prediction-error driven adaptation, multi-step delayed comparison.
  * **Tier 3 (50K – 250K)**: Modular populations, context-switching, rule discovery, continuous task switching.
  * **Tier 4 (250K – 1M+)**: Hierarchical predictive structures, multi-modal integration, complex temporal planning.
* Every parameter added must be justified by an ablation study proving simpler topologies fail on the target benchmark.

---

## 5. The Scientific Development Loop

Every engineering cycle follows a closed scientific loop:

```text
Hypothesis Formulation
       ↓
Minimal Substrate Implementation
       ↓
Controlled Baseline Comparison
       ↓
Quantitative Metric Logging
       ↓
Telemetry Stream Generation & 3D Visual Inspection
       ↓
Ablation & Failure Analysis
       ↓
Model Refinement / Rejection
```

We do not write thousands of lines of speculative code in hopes that intelligence will emerge. We write the minimum code needed to test a precise hypothesis, evaluate it against baselines, and visually verify its internal dynamics.

---

## 6. Engineering Quality & Determinism

1. **Deterministic Execution**:
   * All pseudo-random number generators must accept explicit seeds (`seed=42`).
   * Computational updates given identical input sequences and identical initial weights must produce bit-exact identical telemetry across runs.
2. **Structured Telemetry Schemas**:
   * Telemetry packets use explicit, strictly typed data definitions (compact binary or structured JSON/FlatBuffers).
3. **Test-Driven Rigor**:
   * Unit tests for each mathematical update rule (activation functions, ODE integrators, plasticity updates, normalization).
   * Property-based tests verifying bounded activations, numerical stability, and energy/loss conservation.

---

## 7. Dale's Constraint & Homeostatic Invariants (Phase 8)

1. **Sign Preservation**:
   * Synaptic polarity must be an invariant property of the presynaptic neuron ($w_{ij} \ge 0$ for Excitatory, $w_{ij} \le 0$ for Inhibitory). Plasticity updates (STDP, scaling) must preserve this invariant.
2. **Current Mass Balance**:
   * Inhibitory and excitatory currents must be accounted for and integrated independently to avoid numerical cancellation artifacts.
3. **Homeostatic Autonomy**:
   * Homeostatic scaling operates purely on local activity feedback to regulate target metabolic rates. It must never inspect objective loss or external task supervision.

---

## 8. Three-Factor Credit Assignment Invariants (Phase 9)

1. **Strict Synaptic Adjacency of Eligibility**:
   * Eligibility traces $e_{ij}(t)$ must strictly exist only where a physical anatomical connection $A_{ij} = 1$ exists. Non-local phantom credit assignment is strictly prohibited.
2. **Third-Factor Decoupling**:
   * The neuromodulatory teaching signal $M(t)$ (whether scalar reward, prediction error, or signed RPE) provides gating and magnitude scaling; it does not replace local pre-post synaptic state tracking.
3. **Honest Reporting of Plasticity Efficacy**:
   * If three-factor plasticity fails to break modularity or context disambiguation ceilings, report failure directly. Never retrofit hyperparameters to simulate artificial success.

