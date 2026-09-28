# 3D-Brain: Strategic Research & Engineering Roadmap

## Phased Progression Overview

The development of 3D-Brain proceeds through empirical phases. Each phase gates the next through measurable acceptance criteria.

```text
Phase 0: Foundations & Conceptual Model (COMPLETE)
   ↓
Phase 1: Minimal Adaptive Substrate & Benchmark Engine (Exp-001) (COMPLETE)
   ↓
Phase 2: Observability Engine & Causal Ablation (Exp-002) (COMPLETE)
   ↓
Phase 3: Online Plasticity & Continual Adaptation (Exp-003) (COMPLETE)
   ↓
Phase 4: Spatialized 3D Topology & Parameter Scaling (COMPLETE)
   ↓
Phase 5: Cognitive Time Travel & Native Desktop Shell (NEXT)
   ↓
Phase 6: Causal Query Layer & Interactive Explanations
```

---

## Phase 0: Foundations, Survey & Specifications (Complete)
* **Goal**: Establish rigorous theoretical foundations, research taxonomy, and architectural specifications before writing speculative code.
* **Deliverables**:
  * [x] Project Charter ([PROJECT_CHARTER.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/PROJECT_CHARTER.md))
  * [x] System Principles ([SYSTEM_PRINCIPLES.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/SYSTEM_PRINCIPLES.md))
  * [x] Research Questions & Falsification Criteria ([RESEARCH_QUESTIONS.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/RESEARCH_QUESTIONS.md))
  * [x] Formal Terminology ([TERMINOLOGY.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/TERMINOLOGY.md))
  * [x] Conceptual Model ([CONCEPTUAL_MODEL.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/CONCEPTUAL_MODEL.md))
  * [x] Literature Survey of Existing Approaches ([SURVEY_EXISTING_APPROACHES.md](file:///Users/kalachanddash/Documents/3D-Brain/research/SURVEY_EXISTING_APPROACHES.md))
  * [x] Candidate Architectures Evaluation Matrix ([CANDIDATE_ARCHITECTURES_EVALUATION.md](file:///Users/kalachanddash/Documents/3D-Brain/research/CANDIDATE_ARCHITECTURES_EVALUATION.md))
  * [x] Experiment 001 Specification ([EXPERIMENT_001_SPECIFICATION.md](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_001_SPECIFICATION.md))
  * [x] 3D Telemetry & Visual Contract ([VISUALIZATION_CONCEPT_AND_TELEMETRY.md](file:///Users/kalachanddash/Documents/3D-Brain/visualization/VISUALIZATION_CONCEPT_AND_TELEMETRY.md))
* **Exit Gate**: Complete documentation approved and baseline experiment ready for implementation.

---

## Phase 1: Minimal Adaptive Substrate & Benchmark Engine (Complete)
* **Goal**: Implement the headless computational core and execute Experiment 001 (Delayed Comparison Benchmark).
* **Deliverables**:
  * [x] Three initial comparison systems (`BASE-RNN`, `BASE-GRU`, `CAND-LI`).
  * [x] Continuous Delayed Match-to-Sample benchmark engine.
  * [x] Telemetry pipeline generating `.brainlog` v1.0 traces.
  * [x] Phase 1 Implementation Report ([PHASE_1_IMPLEMENTATION_REPORT.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_1_IMPLEMENTATION_REPORT.md)).
  * [x] Phase 1.1 Scientific Reproducibility & Claim Audit ([PHASE_1_1_SCIENTIFIC_AUDIT.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_1_1_SCIENTIFIC_AUDIT.md)).
* **Exit Gate**: Quantitative baseline comparison showing which architecture solves delayed comparison with the fewest parameters; deterministic telemetry file generated (PASSED).

---

## Phase 2: Brain Observability Engine & Causal Ablation (Complete)
* **Goal**: Transform the telemetry prototype into a robust Brain Observability Engine, resolve Phase 1.1 scientific confounds via Experiment 002, and build bidirectional cognitive replay.
* **Deliverables**:
  * [x] Confound resolution: `DISC-LRNN` and `BASE-GRU-MemBias` substrates.
  * [x] Dedicated `observability/` engine package (`capture.py`, `events.py`, `topology.py`, `brainlog.py`, `replay.py`).
  * [x] `.brainlog` v2.0 schema with first-class computational events and periodic state snapshots.
  * [x] Standalone `ReplayEngine` with bidirectional scrubbing, causal stepping, and dynamic state inspection.
  * [x] Complete Experiment 002 matrix execution across 96 conditions with machine-readable results (`exp002_results.json`).
  * [x] Upgraded 3D WebGL Observability Console v2.0 with interactive timeline event track, neuron dynamic variable inspector, and synaptic connection inspector.
  * [x] Phase 2 Implementation & Scientific Report ([PHASE_2_IMPLEMENTATION_REPORT.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_2_IMPLEMENTATION_REPORT.md)).
* **Exit Gate**: Full bidirectional causal replay, dynamic state variable inspection, Experiment 002 matrix executed, and scientific claims audited (PASSED).

---

## Phase 3: Online Plasticity & Continual Adaptation (Complete)
* **Goal**: Investigate learning without catastrophic forgetting on non-stationary input streams through local plasticity rules.
* **Deliverables**:
  * [x] Modular local plasticity abstraction (`brain/plasticity.py`): `Frozen`, `Hebbian`, `Oja`, `Modulated`, `Homeostatic`.
  * [x] Continual Delayed Association (CDA) benchmark engine (`benchmarks/continual_adaptation.py`) with 5 non-stationary regimes (Stationary, Abrupt Shift, Recurring, Multiple, Gradual Drift).
  * [x] `.brainlog` v2.1 schema evolution with dynamic synaptic weight capture and plasticity event streams.
  * [x] ReplayEngine plasticity inspection methods (`get_plasticity_events()`, `get_weight_evolution()`).
  * [x] Full Experiment 003 matrix execution across 95 condition runs (`experiments/results/exp003_results.json`).
  * [x] Upgraded 3D Observability Console v2.1 with interactive Synaptic Edge Inspector HUD, event badges, and dynamic connection color shifts.
  * [x] Phase 3 Implementation Report ([PHASE_3_IMPLEMENTATION_REPORT.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_3_IMPLEMENTATION_REPORT.md)) & Literature Review ([ONLINE_PLASTICITY_REVIEW.md](file:///Users/kalachanddash/Documents/3D-Brain/research/ONLINE_PLASTICITY_REVIEW.md)).
* **Exit Gate**: Empirical evaluation demonstrating that three-factor neuromodulated plasticity reduces catastrophic forgetting by $>80\%$ over unconstrained Hebbian learning while preserving recurrent temporal memory attractors (PASSED).

---

## Phase 4: Spatialized 3D Topology & Parameter Scaling (Complete)
* **Goal**: Introduce physical metric space constraints (distance-dependent propagation delays, localized receptive fields) and parameter scaling.
* **Deliverables**:
  * [x] Explicit 3D spatial abstraction (`brain/spatial.py`): `SpatialNeuron`, `SpatialSynapse`, `SpatialGeometry`, `SpatialConnectivityBuilder`, `SignalPropagationQueue`, `SpatialPlasticRNN`.
  * [x] Four controlled topology families (`RANDOM`, `LOCAL`, `MODULAR`, `SMALL_WORLD`) with parameter-controlled budgets.
  * [x] Finite transmission delays and high-performance circular ring-buffer propagation queue.
  * [x] Parameter scaling from 100 to 100,000 units ($2 \times 10^6$ synapses) sustaining $>10,000$ steps/sec at 100K on Apple Silicon unified memory.
  * [x] Task C benchmark engine (`benchmarks/spatial_tasks.py`): Spatially Distributed Temporal Task.
  * [x] Scientific graph metrics (`brain/spatial_analysis.py`): Newman structural modularity $Q_{struct}$, functional modularity $Q_{func}$, Kuramoto phase synchrony $R$, participation coefficients.
  * [x] `.brainlog` v2.2 schema evolution with in-transit signals, distances, delays, and modular properties.
  * [x] Upgraded 3D Observability Console v2.2 with Structural vs. Functional Connectivity toggle, dynamic $Q_{func}$ and Sync $R$ tags, and edge spatial HUD.
  * [x] Full Experiment 004 execution across topologies, speeds, plasticity, and scale (`experiments/results/exp004_results.json`).
  * [x] Literature Review ([SPATIAL_TOPOLOGY_REVIEW.md](file:///Users/kalachanddash/Documents/3D-Brain/research/SPATIAL_TOPOLOGY_REVIEW.md)) & Implementation Report ([PHASE_4_IMPLEMENTATION_REPORT.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_4_IMPLEMENTATION_REPORT.md)).
* **Exit Gate Evaluation**:
  * Spatial Computation: PASSED.
  * Scaling (100K units): PASSED.
  * Reproducibility: PASSED.
  * Functional Modularity: NOT SUPPORTED ($Q_{func} \approx -0.02$ despite $Q_{struct} \approx 0.63 - 0.74$).
  * Latency-Based Synchronization: INCONCLUSIVE (phase coherence remained low $R < 0.20$; delays dispersed signals).

---

## Phase 5: Cognitive Time Travel & Causal Replay Engine
* **Goal**: Enable frictionless bidirectional scrubbing and causal debugging.
* **Workstreams**:
  * Implement keyframe checkpointing and delta-log playback.
  * Build UI timeline scrubber with bookmarkable cognitive event markers.
  * Backward causal attribution: click an error burst, trace backward to identify contributing upstream activations.
* **Exit Gate**: User can pause any run, scrub back 500 steps, and inspect the causal synaptic path that led to a decision.

---

## Phase 6: Human-Facing Explanatory Layer (JARVIS)
* **Goal**: Provide an interactive query interface that explains observable computational state transitions.
* **Workstreams**:
  * Implement cognitive query engine: translates natural language queries ("Why did unit 42 activate?") into graph traversal queries over telemetry history.
  * Visual camera directors: automatically fly the 3D camera to highlight the active neural sub-network responsible for an event.
* **Exit Gate**: Interactive natural language query highlights verified, observable computational history without fabricating ungrounded explanations.
