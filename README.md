# 3D-Brain

> **Investigate whether a relatively small adaptive computational substrate can produce persistent, dynamic, learning-oriented cognitive behavior, while exposing that computation as a live interactive 3D neural world.**

---

## 1. Project Vision

**3D-Brain** is a standalone, empirical research initiative and software architecture building a computational system inspired by biological brain dynamics, running as a native-feeling macOS desktop application.

This is **not** a chatbot, a transformer wrapper, an LLM orchestration layer, or a decorative cosmetic visualization displaying fabricated thoughts. 

The core principle of 3D-Brain is the **co-design of computation and observability**:
* The 3D neural world is a live, high-fidelity window into the actual mathematical state of the system.
* If a neuron activates, a connection changes, or an error occurs, it directly represents real computational events.
* The system is engineered to enable **cognitive time travel**: scrubbing backward and forward through execution history to causally trace activations, predictions, errors, and adaptation.

---

## 2. Repository Structure

```text
3D-Brain/
├── brain/                  # Core computational substrate models, ODE integrators, and update rules
├── visualization/          # 3D spatial layout, telemetry contracts, shaders, and rendering engine
├── research/               # Literature surveys, candidate evaluations, and theoretical formulations
├── experiments/            # Controlled experiment specifications, runners, and telemetry generators
├── benchmarks/             # Cognitive capability benchmarks (delayed comparison, continual adaptation)
├── docs/                   # Project charter, conceptual model, principles, glossary, and roadmap
└── README.md
```

---

## 3. Documentation Index

The foundation of 3D-Brain is comprehensively documented in the following charters and specifications:

* **[Project Charter](file:///Users/kalachanddash/Documents/3D-Brain/docs/PROJECT_CHARTER.md)**: Vision, boundaries, anti-goals, and core objectives.
* **[System Principles](file:///Users/kalachanddash/Documents/3D-Brain/docs/SYSTEM_PRINCIPLES.md)**: Architectural philosophy, decoupled modes (Live, Replay, Experiment), cognitive time travel, and parameter parsimony ("earn every parameter").
* **[Research Questions & Hypotheses](file:///Users/kalachanddash/Documents/3D-Brain/docs/RESEARCH_QUESTIONS.md)**: 12-level progressive capability ladder and falsifiable hypotheses with quantitative metrics.
* **[Formal Terminology](file:///Users/kalachanddash/Documents/3D-Brain/docs/TERMINOLOGY.md)**: Precise definitions of Brain, neuron, connection, activation, memory, prediction, error, learning, and plasticity.
* **[Conceptual Model](file:///Users/kalachanddash/Documents/3D-Brain/docs/CONCEPTUAL_MODEL.md)**: Mathematical abstraction of the dynamical graph and 1-to-1 mapping to 3D visual observability.
* **[Strategic Roadmap](file:///Users/kalachanddash/Documents/3D-Brain/docs/ROADMAP.md)**: Phased milestones from foundations (Phase 0) to minimal substrate (Phase 1), 3D desktop shell (Phase 2), continual learning (Phase 3), and causal query layers.
* **[Survey of Existing Approaches](file:///Users/kalachanddash/Documents/3D-Brain/research/SURVEY_EXISTING_APPROACHES.md)**: Deep analysis of RNNs, Liquid Neural Networks (LTCs/CfCs), Predictive Coding, Reservoir Computing, SNNs, Neuromodulated Plasticity, and Local Learning.
* **[Candidate Architectures Evaluation](file:///Users/kalachanddash/Documents/3D-Brain/research/CANDIDATE_ARCHITECTURES_EVALUATION.md)**: Quantitative 10-criteria comparison matrix and phased selection strategy.
* **[Experiment 001 Specification](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_001_SPECIFICATION.md)**: Benchmark design for Delayed Match-to-Sample and temporal memory retention under distraction.
* **[Phase 1 Implementation Spec](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_1_IMPLEMENTATION_SPEC.md)**: Module layout, candidate interfaces, parameter budget allocations, and `.brainlog` data schemas.
* **[Phase 1 Scientific Evaluation Report](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_1_IMPLEMENTATION_REPORT.md)**: Measured empirical results across 54 conditions comparing `BASE-RNN`, `BASE-GRU`, and `CAND-LI`, telemetry audit, and visual verification findings.
* **[Phase 1.1 Scientific Audit Report](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_1_1_SCIENTIFIC_AUDIT.md)**: Rigorous reproducibility audit, causal attribution analysis, confound identification, and claim classifications.
* **[Experiment 002 Specification](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_002_SPECIFICATION.md)**: Tightly controlled ablation isolating leaky dynamics from continuous-time ODEs and investigating GRU gate initialization.
* **[Phase 2 Implementation Report](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_2_IMPLEMENTATION_REPORT.md)**: Observability Engine architecture, Experiment 002 matrix results across 96 conditions, and causal replay console documentation.

---

## 4. Running Benchmarks & 3D Visualization

### Execute Automated Test Suite (38 Tests)
```bash
python3 -m pytest tests -v
```

### Run Full Experiment 001 Matrix (54 Runs)
```bash
python3 experiments/run_exp001.py
```

### Run Full Experiment 002 Matrix (96 Runs)
```bash
python3 experiments/run_exp002.py
```

### Launch Interactive 3D Observability & Replay Console
```bash
python3 visualization/serve.py
# Open http://localhost:8080/visualization/player.html in any browser
```

---

## 5. Development Loop

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

---

## 6. Next Steps

With Phase 1 and Phase 2 successfully completed:
1. **Phase 3: Online Plasticity & Continual Adaptation**: Implement local plasticity rules (Hebbian STDP, predictive coding error minimization) and evaluate sequential learning without catastrophic forgetting under non-stationary task distributions.
2. Advance toward native desktop shell integrations and scalable spatial topologies.
