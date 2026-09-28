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

---

## 4. Running Benchmarks & 3D Visualization

### Execute Automated Test Suite (97 Tests)
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

### Launch Interactive 3D Observability & Replay Console (v2.4)
```bash
python3 visualization/serve.py
# Open http://localhost:8080/visualization/player.html in any browser
```

---

## 5. Phase 6 Architecture & Key Findings

### 5.1 Persistent Latent Substrate Architecture
```text
Sensory Input (x_t) ─────────────► Fast Recurrent Layer (h_t) ◄── 3D Spatial Delay Ring Buffer
                                         │            ▲
                                         │ W_rec      │ W_fb
                                         ▼            │
Time Retention (lambda_z) ─────► Persistent Latent Memory (z_t)
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
             1-Step Forecast      Multi-Step Forecast   Latent Decoding
               (k = 1)             (k = 2, 5, 10, 25)    (Probes, RSA)
```

**Latent Update Equations:**
$$z_t = \lambda_z \odot z_{t-1} + (1 - \lambda_z) \odot \tanh\left(W_{\text{lat}} z_{t-1} + W_{\text{rec}} h_t + b_z\right)$$
$$\hat{x}_{t+k} = W_{\text{pred}}^{(k)} z_t + b_{\text{pred}}^{(k)}, \quad k \in \{1, 2, 5, 10, 25\}$$

### 5.2 Empirical Benchmark Summary (Experiment 006)

| Architecture / Substrate | Task A Cue $R^2$ ($\Delta t=10$) | Task A Cue $R^2$ ($\Delta t=100$) | Task A Cue $R^2$ ($\Delta t=500$) | Task C Centroid Dist | Task C Action Acc | Task D Flat MSE ($k=25$) | Continual Forgetting |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BASE_PHASE5`** | $0.218$ | $-0.134$ | $-0.134$ | $0.000$ | $46.7\%$ | $0.052$ | $0.00226$ |
| **`BASE_LARGER_RECURRENT`** | $0.245$ | $-0.118$ | $-0.118$ | $0.000$ | $46.7\%$ | $0.052$ | $0.00215$ |
| **`PERSISTENT_LATENT_PASSIVE`** | $0.627$ | $0.518$ | $0.461$ | $0.054$ | $46.7\%$ | $0.052$ | $\mathbf{0.00000}$ |
| **`PERSISTENT_LATENT_ACTIVE`** | $\mathbf{0.630}$ | $\mathbf{0.518}$ | $\mathbf{0.459}$ | $\mathbf{0.057}$ | $\mathbf{56.7\%}$ | $0.052$ | $\mathbf{0.00000}$ |
| **`GRADIENT_PREDICTIVE_BPTT`** | $0.628$ | $0.518$ | $0.460$ | $0.056$ | $50.0\%$ | $\mathbf{0.033}$ | $\mathbf{0.00000}$ |
| *Causal Ablation (B: Step Reset)* | $0.218$ | $-0.343$ | $-0.343$ | $0.000$ | $50.0\%$ | $0.052$ | $0.00234$ |

### 5.3 Key Scientific Findings
1. **Persistent Memory Across Long Delays (SUPPORTED)**: `PERSISTENT_LATENT` models maintain high linear decodability of transient cue identity ($R^2 = 0.459 \pm 0.039$) across silent retention intervals up to $\Delta t = 500$ steps. Baselines without explicit latent persistence collapse completely by $\Delta t = 50$ ($R^2 < 0$).
2. **Causal Necessity of Latent Continuity (SUPPORTED)**: Ablating latent state persistence between steps (`ABL_B_STEP_RESET`) causes catastrophic memory failure ($R^2 = -0.343$ at $\Delta t = 100$), confirming that memory maintenance is causally driven by the persistent latent state.
3. **Temporal Context Disambiguation (PARTIALLY SUPPORTED)**: Identical sensory observations following distinct histories ($A \to B \to X$ vs $C \to D \to X$) are cleanly segregated in latent space ($\text{FDR} = 104.17$, centroid distance $= 0.057$). Downstream behavioral disambiguation achieved $56.7\% \pm 9.4\%$ accuracy.
4. **Multi-Step Rollout Stability (PARTIALLY SUPPORTED)**: Multi-step prediction heads sustain stable, non-exploding error across horizons $k \in [1, 25]$ ($\text{MSE} \approx 0.041 - 0.052$), although offline backpropagation-through-time achieves lower absolute error ($0.027 - 0.033$).
5. **Elimination of Catastrophic Forgetting (SUPPORTED)**: Persistent latent memory completely eliminates catastrophic forgetting under recurring environmental shifts ($0.0000$ vs $0.00226$ in Phase 5 baseline), with backward retention ratio $4.64 \pm 1.63$.
6. **Parameter Scaling to 100,000 Units (PASSED)**: The substrate scales with linear memory and runtime complexity to $N=100{,}000$ neurons and $1{,}500{,}000$ synapses, executing at $10{,}439$ steps/sec with $<23\text{ MB}$ RAM on Apple Silicon.
7. **Functional Modularity (NEGATIVE RESULT)**: Continuous rate-coded recurrent networks still do not produce high functional modularity ($Q_{\text{func}} \approx -0.010$).

---

## 6. Next Steps

With Phase 1 through Phase 6 completed:
1. **Phase 7: Autonomous Oscillatory Dynamics & Spiking Phase Processing**: Address the continuous-substrate functional modularity barrier ($Q_{\text{func}} \approx 0$) by investigating spiking neuron dynamics (LIF / Izhikevich substrates with STDP), phase-locking limit cycles, and Kuramoto oscillatory coherence while maintaining persistent latent predictive memory.
