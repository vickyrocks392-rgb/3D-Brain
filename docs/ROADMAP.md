# 3D-Brain: Strategic Research & Engineering Roadmap

## Phased Progression Overview

The development of 3D-Brain proceeds through empirical phases. Each phase gates the next through measurable acceptance criteria.

```text
Phase 0: Foundations & Conceptual Model (CURRENT)
   ↓
Phase 1: Minimal Adaptive Substrate & Benchmark Engine (Exp-001)
   ↓
Phase 2: Observability Engine & macOS Desktop Shell
   ↓
Phase 3: Online Plasticity & Continual Adaptation
   ↓
Phase 4: Spatialized 3D Topology & Parameter Scaling
   ↓
Phase 5: Cognitive Time Travel & Causal Replay Engine
   ↓
Phase 6: Causal Query Layer & JARVIS Integration
```

---

## Phase 0: Foundations, Survey & Specifications (Current)
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

## Phase 1: Minimal Adaptive Substrate & First Benchmark
* **Goal**: Implement the headless computational core and execute Experiment 001 (Delayed Comparison Benchmark).
* **Workstreams**:
  * Implement baseline substrates (Elman RNN, GRU, Liquid Time-Constant leaky integrator).
  * Build a deterministic benchmark runner with seeded PRNGs.
  * Execute parameter scaling ablation (e.g. 500 params vs. 2K vs. 10K).
  * Export structured telemetry packets (`.brainlog` format) from actual runs.
* **Exit Gate**: Quantitative baseline comparison showing which architecture solves delayed comparison with the fewest parameters; deterministic telemetry file generated.

---

## Phase 2: Observability Engine & macOS Desktop Shell
* **Goal**: Render the telemetry generated in Phase 1 inside an interactive 3D spatial window on macOS.
* **Workstreams**:
  * Evaluate native macOS graphics options (Swift/Metal vs. WebGPU/Rust/Wasm desktop container).
  * Implement 3D spatial layout of the Phase 1 network nodes and edges.
  * Render activation glow, signal propagation along connection splines, and error flashes.
  * Implement interactive camera controls (pan, zoom, orbit, node selection).
* **Exit Gate**: 60 FPS rendering of a recorded Phase 1 run, with node click inspector displaying real state variables.

---

## 3. Phase 3: Online Plasticity & Continual Adaptation
* **Goal**: Investigate learning without catastrophic forgetting on non-stationary input streams.
* **Workstreams**:
  * Implement local plasticity rules (Hebbian STDP, predictive coding error minimization, neuromodulatory gating).
  * Benchmark continual adaptation: Task A $\to$ Task B $\to$ Task A memory retention.
  * Compare local online plasticity against standard backpropagation through time.
  * Telemetry updates: visualize synaptic weight changes ($\Delta w_{ij}$) in real time.
* **Exit Gate**: Substrate demonstrating lower catastrophic forgetting than standard RNN baselines under distribution shift.

---

## Phase 4: Spatialized 3D Topology & Parameter Scaling
* **Goal**: Introduce physical metric space constraints (distance-dependent propagation delays, localized receptive fields).
* **Workstreams**:
  * Scale from sub-10K to 50K and 100K computational units.
  * Test modular clustering and small-world topology advantages.
  * Implement spatial signal propagation queues with finite transmission speed.
  * Benchmark throughput and GPU compute scaling on Apple Silicon unified memory.
* **Exit Gate**: Validated spatial network displaying emergent functional modularity and latency-based phase synchronization.

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
