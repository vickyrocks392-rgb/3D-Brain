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
* **[Phase 4 Implementation Report](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_4_IMPLEMENTATION_REPORT.md)**: Comprehensive report evaluating spatial computation, modular compartmentalization, empirical falsification of spontaneous functional emergence, and exit-gate audit.
* **[Phase 5 Literature Review](file:///Users/kalachanddash/Documents/3D-Brain/research/PHASE_5_LITERATURE_REVIEW.md)**: Predictive coding, active inference, feedback alignment, synaptogenesis, and embodied sensorimotor integration.
* **[Phase 5 Design Audit](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_5_DESIGN_AUDIT.md)**: Architectural specifications for closed-loop embodiment, dynamic structural plasticity, and `.brainlog` v2.3 telemetry.
* **[Experiment 005 Results Report](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_005_RESULTS.md)**: Empirical findings across 102 conditions evaluating self-supervised prediction error, active embodiment advantage, and dynamic brain growth.
* **[Phase 5 Implementation Report](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_5_IMPLEMENTATION_REPORT.md)**: Complete analysis of self-supervised sensory prediction, active vs passive embodiment, structural rewiring, and scaling.
* **[Phase 6 Literature Review](file:///Users/kalachanddash/Documents/3D-Brain/research/PHASE_6_LITERATURE_REVIEW.md)**: Foundations of persistent latent memory, multi-timescale temporal processing, reservoir dynamics, credit assignment without BPTT, and representational geometry.
* **[Phase 6 Design Audit](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_6_DESIGN_AUDIT.md)**: Architecture audit for persistent latent state $z_t$, multi-step prediction heads, feedback alignment, and `.brainlog` v2.4.
* **[Experiment 006 Specification](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_006_SPECIFICATION.md)**: Protocol investigating persistent latent memory, multi-step prediction, context disambiguation, and representational drift across delay intervals.
* **[Experiment 006 Results Report](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_006_RESULTS.md)**: Comprehensive empirical evaluation of Experiment 006 across 5 architectures, 4 benchmark tasks, 6 delay intervals, 4 environments, and ablations A-F.
* **[Phase 6 Implementation Report](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_6_IMPLEMENTATION_REPORT.md)**: Exhaustive 26-section synthesis covering mathematical formulations, empirical results, ablations, negative results, and exit-gate evaluation.
* **[Phase 7 Design Audit](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_7_DESIGN_AUDIT.md)**: Biophysical architecture audit for LIF/Izhikevich spiking substrates, axonal delay queues, STDP, and `.brainlog` v2.5.
* **[Phase 7 Literature Review](file:///Users/kalachanddash/Documents/3D-Brain/research/PHASE_7_LITERATURE_REVIEW.md)**: Scientific review of spiking dynamics, STDP rules, gamma oscillation mechanisms, and functional modularity metrics.
* **[Experiment 007 Specification](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_007_SPECIFICATION.md)**: Formal protocol investigating spiking dynamics, STDP, modularity emergence, oscillatory limit cycles, and synchronization-memory trade-offs.
* **[Experiment 007 Results Report](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_007_RESULTS.md)**: Empirical evaluation across 7 controls, 4 topologies, 3 tasks, 6 causal ablations, and $N=100\text{K}$ scaling ladder.
* **[Phase 7 Implementation Report](file:///Users/kalachanddash/Documents/3D-Brain/docs/PHASE_7_IMPLEMENTATION_REPORT.md)**: Complete 26-section report detailing biophysical equations, empirical verifications, negative results, and exit-gate audit.

---

## 4. Running Benchmarks & 3D Visualization

### Execute Automated Test Suite (113 Tests)
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

### Run Full Experiment 005 Embodied World Model Matrix
```bash
python3 experiments/run_exp005.py
```

### Run Full Experiment 006 Persistent Latent Memory Matrix
```bash
python3 experiments/run_exp006.py
```

### Run Full Experiment 007 Spiking Dynamics & STDP Matrix
```bash
python3 experiments/run_exp007.py
```

### Run Full Experiment 008 Balanced E/I & AI Regime Matrix
```bash
python3 experiments/run_experiment_008.py
python3 experiments/analyze_experiment_008.py
```

### Launch Interactive 3D Observability & Replay Console (v2.6)
```bash
python3 visualization/serve.py
# Open http://localhost:8080/visualization/player.html in any browser
```

---

## 5. Phase 8 Architecture & Key Findings

### 5.1 Dale E/I Balanced Spiking Substrate
```text
Sensory Input (x_t) ─────────────► Dale Spiking Population (~80% E / ~20% Fast-Spiking I)
                                          │                     ▲
                                   Spikes │                     │ Circular Delay Queue
                                          ▼                     │ (d_ij ~ dist_ij)
                                  Dale STDP Synapses ───────────┘
                                          │
                                 I_E(t) / I_I(t) Current Balance & Homeostatic Scaling
                                          │
                                          ▼
                             Persistent Latent Memory (z_t)
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
              1-Step Forecast      Multi-Step Forecast   Context Disambiguation
                (k = 1)             (k = 2, 5, 10, 25)    (Probes, RSA, FDR)
```

**Biophysical Principles & Invariants:**
* **Dale's Principle**: Neurons strictly emit excitatory ($w_{ij} \ge 0$) or fast inhibitory ($w_{ij} \le 0$) projections.
* **Current Mass Accounting**: Independent ring buffers for $I_E$ and $I_I$ currents; net drive $I_{\text{net}} = I_E - |I_I|$.
* **Multi-Scale Synchrony**: Spatial gradient analysis ($R_{\text{local}}$ vs $R_{\text{global}}$).
* **Asynchronous-Irregular Diagnostic**: Operational composite score evaluating firing stability, low pairwise correlation ($r_{\text{sc}}$), and non-pathological synchrony.

### 5.2 Empirical Benchmark Summary (Experiment 008)

| Condition / Substrate | $Q_{\text{func}}$ | $R_{\text{global}}$ | $R_{\text{local}}$ | $CV_{\text{ISI}}$ | Fano | $r_{\text{sc}}$ | Context Acc | Memory $R^2$ ($\Delta t=500$) | Pred MSE ($k=5$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A_REF_PHASE7** | $0.186$ | $0.215$ | $0.400$ | $0.158$ | $0.30$ | $0.017$ | $75.0\%$ | $0.417$ | $0.2821$ |
| **B_EI_NO_HOMEO** | $0.212$ | $0.290$ | $0.437$ | $0.122$ | $0.13$ | $0.032$ | $58.3\%$ | $0.394$ | $0.3970$ |
| **C_EI_HOMEO** | $0.195$ | $0.291$ | $0.437$ | $0.122$ | $0.13$ | $0.033$ | $58.3\%$ | $0.391$ | $0.3970$ |
| **D_EI_STDP** | $0.200$ | $0.290$ | $0.437$ | $0.122$ | $0.13$ | $0.032$ | $58.3\%$ | $0.382$ | $0.3970$ |
| **E_EI_STDP_HOMEO (Full)** | $0.169$ | $0.291$ | $0.437$ | $0.122$ | $0.13$ | $0.033$ | $58.3\%$ | $0.410$ | $0.3970$ |
| *Ablation 1 (No Inhibition)* | $0.214$ | $\mathbf{0.307}$ | $0.443$ | $0.125$ | $0.14$ | $\mathbf{0.048}$ | $\mathbf{50.0\%}$ | $0.415$ | $0.3970$ |
| *Ablation 6 (Disable STDP)* | $0.203$ | $0.291$ | $0.437$ | $0.122$ | $0.13$ | $0.033$ | $58.3\%$ | $0.391$ | $0.3970$ |

### 5.3 Key Scientific Findings
1. **Inhibition Rescues Context Disambiguation from Collapse (SUPPORTED)**: Completely removing inhibition (`ABL_01`) caused pairwise correlation to surge ($+45\%$) and context discrimination to collapse to pure chance ($50.0\%$). Restoring $20\%-30\%$ fast inhibition reduced correlation ($r_{\text{sc}} \approx 0.03$) and recovered discrimination to $58.3\% - 62.5\%$.
2. **Local Synchrony Coexists with Global Decorrelation (SUPPORTED)**: The multi-scale synchrony analyzer revealed a strong spatial gradient: $R_{\text{local}} = 0.437 > R_{\text{global}} = 0.291$ ($\Delta R = +0.147$) and $\text{PLV}_{\text{local}} > \text{PLV}_{\text{global}}$. Local neural assemblies coordinate coherent signal transmission while global population dynamics remain decorrelated.
3. **Modularity Target Falsification (NEGATIVE RESULT)**: The ambitious hypothesis $Q_{\text{func}} > 0.30$ was falsified. Measured functional modularity reached $Q_{\text{func}} = 0.169 - 0.227$. Recurrent inhibitory currents suppress cross-modular and within-modular co-activations equally, preventing runaway modular clustering.
4. **Passive Nature of Classical Two-Factor STDP (AUDIT CONFIRMED)**: Disabling STDP produced results statistically indistinguishable from active STDP ($p > 0.5$). Spiking dynamics and topology drive the computational state; unmodulated pair-based STDP acts as a passive stabilizing filter.
5. **Scale Ladder to 100,000 Units (PASSED)**: Vectorized sparse E/I queue sustained **$322.4\text{ steps/s}$**, **$1.61\text{M spikes/s}$**, and **$12.90\text{M synaptic events/s}$** with only **$13.83\text{ MB}$** RAM on Apple Silicon.

---

## 6. Next Steps

Phase 8 is **COMPLETE (PASS WITH CAVEATS)**.
* Next research phase: **Phase 9: Three-Factor Plasticity & Dendritic Compartments** (Pending user authorization).


