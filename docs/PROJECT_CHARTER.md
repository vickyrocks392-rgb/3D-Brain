# 3D-Brain: Project Charter

## 1. Executive Summary & Vision

**3D-Brain** is a standalone, empirical research initiative and software architecture whose objective is to investigate:

> *Can a relatively compact, adaptive computational substrate produce persistent, dynamic, learning-oriented cognitive behavior—while exposing that computation as a live, interactive, observable 3D neural world?*

The system is designed as a native-feeling macOS desktop application uniting a computational substrate and a real-time 3D observability environment into a single co-evolving system.

Unlike conventional machine learning pipelines that train static weight matrices offline and deploy black-box inference APIs, 3D-Brain models neural computation as continuous, stateful, dynamic, and spatially organized. The visualization is an unvarnished window into the actual mathematical and dynamical state of the machine: if a neuron flashes, an edge thickens, or a signal flows, it is because actual state variables changed, actual predictions failed or succeeded, or actual plasticity rules updated internal weights.

---

## 2. Boundaries & Anti-Goals

To maintain rigorous scientific focus and prevent scope creep, 3D-Brain explicitly defines what it is **not**:

| What 3D-Brain Is NOT | Architectural Rationale |
| :--- | :--- |
| **Not a chatbot or Transformer wrapper** | We are investigating whether non-transformer, persistent dynamical substrates can exhibit adaptive cognitive capabilities without trillions of tokens or frozen parameters. |
| **Not an LLM orchestration / multi-agent framework** | We do not chain external API calls or construct prompt DAGs. All cognitive dynamics originate within our computational substrate. |
| **Not an AutoForge subsystem** | 3D-Brain is an independent project with its own clean architecture, independent scientific track, and separate codebase. |
| **Not a decorative visualization** | We do not generate cosmetic particle effects or fake "AI thoughts." Every visual entity is an observability primitive bound to real telemetry. |
| **Not a conventional neural network with a 3D skin** | The computational substrate and the spatial neural world are co-designed. Topology, locality, signal propagation delays, and modularity have computational meaning. |
| **Not an AGI / Consciousness claim** | We make no assertions regarding consciousness, human brain equivalence, or artificial general intelligence. We operate strictly under testable, falsifiable scientific hypotheses. |

---

## 3. Core Objectives

1. **Investigate Minimal Adaptive Substrates**: Discover the smallest viable dynamical architecture (from 10K to 1M parameters) capable of temporal memory, delayed comparison, prediction-error processing, and continual adaptation without catastrophic forgetting.
2. **Co-Evolve Substrate and Observability**: Formulate telemetry protocols and data contracts that bridge computational state transitions with 3D spatial representations at interactive frame rates (60+ FPS).
3. **Enable Cognitive Time Travel**: Support lossless or checkpointed deterministic recording, stepping, scrubbing, replaying, and causal inspection of internal computational history.
4. **Empirical Grounding & Baseline Discipline**: Every proposed substrate must earn its parameters by competing against rigorous, standard baselines (e.g., MLPs, Elman RNNs, GRUs, Reservoir Computing/ESNs, Liquid Time-Constant networks).
5. **Native macOS Desktop Application**: Deliver the system as an ergonomic, performant desktop experience, decoupling computational execution from rendering so that headless benchmarks and interactive visualization share identical core code.

---

## 4. Stakeholders, Roles & Governance

* **Lead Research Engineer**: Owns architectural direction, mathematical formulations, baseline benchmarking, telemetry schema design, and research logging.
* **Engineering Standards**:
  * **Falsifiability**: Every architectural claim must be framed as a measurable hypothesis.
  * **Parameter Parsimony**: "Earn every parameter." Prefer a 10K parameter network that solves a temporal problem cleanly over a 1M parameter brute-force network.
  * **Modularity**: Headless execution must run with zero visualization overhead. Visualizers must run on saved telemetry without re-running compute.
  * **Determinism**: Seeded pseudo-randomness, reproducible initializations, and traceable execution traces are non-negotiable for scientific validity.
