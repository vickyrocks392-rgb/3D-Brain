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
* **[Online Plasticity Literature Review](file:///Users/kalachanddash/Documents/3D-Brain/research/ONLINE_PLASTICITY_REVIEW.md)**: Deep analysis of Hebbian, Oja, Homeostatic, Predictive Coding, and Three-Factor Neuromodulated plasticity.
* **[Experiment 003 Specification](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_003_SPECIFICATION.md)**: Benchmark design for Continual Delayed Association (CDA) across 5 non-stationary environmental regimes.
* **[Experiment 003 Results Report](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_003_RESULTS.md)**: Empirical evaluation of 95 experimental condition runs comparing Frozen, Hebbian, Oja, Modulated, and Homeostatic plasticity.
* **[Phase 3 Implementation Report](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_3_IMPLEMENTATION_REPORT.md)**: Mathematical formulations, continual adaptation dynamics, catastrophic forgetting analysis, stability findings, and 3D visual observability of dynamic synaptic weights.
* **[Spatial Topology Literature Review](file:///Users/kalachanddash/Documents/3D-Brain/research/SPATIAL_TOPOLOGY_REVIEW.md)**: Deep analysis of spatial embedding, network neuroscience, small-world topologies, axonal propagation delays, Kuramoto phase synchrony, and parameter scaling.
* **[Experiment 004 Specification](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_004_SPECIFICATION.md)**: Formal protocol investigating spatial topology × propagation latency × scale across Delayed Match, Continual CDA, and Spatially Distributed Temporal tasks.
* **[Experiment 004 Results Report](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_004_RESULTS.md)**: Empirical findings across 16 primary conditions, ablations A-F, null models, and scaling benchmark to 100,000 units on Apple Silicon.
* **[Phase 4 Implementation Report](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_4_IMPLEMENTATION_REPORT.md)**: Comprehensive 24-section report evaluating spatial computation, modular compartmentalization, empirical falsification of spontaneous functional emergence, and exit-gate audit.

---

## 4. Running Benchmarks & 3D Visualization

### Execute Automated Test Suite (68 Tests)
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

### Run Full Experiment 003 Continual Plasticity Matrix (95 Runs)
```bash
python3 experiments/run_exp003.py
```

### Run Full Experiment 004 Spatial Topology & Scaling Matrix
```bash
python3 experiments/run_exp004.py
```

### Launch Interactive 3D Observability & Replay Console (v2.2)
```bash
python3 visualization/serve.py
# Open http://localhost:8080/visualization/player.html in any browser
```

---

## 5. Phase 4 Architecture & Key Findings

### 5.1 Spatialized Substrate Architecture
```text
3D Position (x_i) ──► Distance (d_ij = ||x_i - x_j||) ──► Latency (tau_ij = d_ij / v)
                                                                 │
                                                                 ▼
Neuron State (h_i) ◄── Arrived Recurrent Input ◄── Circular Delay Ring Buffer
        │                                                        ▲
        ▼                                                        │
Activation (a_i) ──────────────── Synapse (w_ij) ────────────────┘
        │
        ▼ (Online Local Plasticity masked by structural adjacency)
 Synaptic Weight Update (dw_ij)
```

### 5.2 Key Scientific Findings
1. **Physical Scaling to 100,000 Units**: Sparse contiguous array representations and circular ring-buffer delay queues scaled smoothly from $100$ to $100{,}000$ neurons and $2{,}000{,}000$ synapses, sustaining $>10{,}000$ steps/sec with only $41.96\text{ MB}$ memory footprint on Apple Silicon M1.
2. **Structural vs. Functional Modularity**: While strong structural modularity was engineered ($Q_{struct} = 0.63 - 0.74$), **spontaneous functional modularity was NOT observed** ($Q_{func} \approx -0.02$). Imposed structural clusters alone do not create functional segregation without heterogeneous input streams.
3. **Propagation Latency**: Transmission delays acted primarily as a temporal dispersion penalty rather than an information processing asset; collective phase coherence remained weak ($R < 0.20$), though Small-World networks exhibited higher synchrony ($R \approx 0.16$) than Modular or Local graphs ($R \approx 0.12$).
4. **Plasticity $\times$ Space**: Modular structural topology significantly protected against catastrophic forgetting under online neuromodulated plasticity ($0.033$ forgetting vs. $0.150$ in random networks), demonstrating that physical modular boundaries provide compartmentalization against interference.

---

## 6. Next Steps

With Phase 1, Phase 2, Phase 3, and Phase 4 successfully completed:
1. **Phase 5: Cognitive Time Travel & Native Desktop Shell**: Implement keyframe delta checkpoints, timeline scrubbers, and backward causal synaptic attribution in a native desktop window.
2. **Phase 6: Causal Query Layer & Interactive Explanations**: Grounded natural language query engine.
