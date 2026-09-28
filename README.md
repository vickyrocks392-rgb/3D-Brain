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

### Launch Interactive 3D Observability & Replay Console (v2.5)
```bash
python3 visualization/serve.py
# Open http://localhost:8080/visualization/player.html in any browser
```

---

## 5. Phase 7 Architecture & Key Findings

### 5.1 Hybrid Spiking-Latent Substrate Architecture
```text
Sensory Input (x_t) ─────────────► Spiking Neuron Population (LIF / Izhikevich)
                                         │                    ▲
                                 Spikes  │                    │ Axonal Delay Queue
                                         ▼                    │ (d_ij ~ dist_ij)
                                  STDP Synapses ──────────────┘
                                         │
                                         │ Filtered Spike Trace (r_t)
                                         ▼
                            Persistent Latent Memory (z_t)
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
             1-Step Forecast      Multi-Step Forecast   Latent Decoding
               (k = 1)             (k = 2, 5, 10, 25)    (Probes, RSA)
```

**Biophysical State Equations:**
* **Izhikevich Population:**
  $$v_{i, t+dt} = v_{i, t} + dt \cdot \left(0.04 v_{i, t}^2 + 5 v_{i, t} + 140 - u_{i, t} + I_{i, t}\right)$$
  $$u_{i, t+dt} = u_{i, t} + dt \cdot a \cdot (b \cdot v_{i, t} - u_{i, t})$$
  $$\text{If } v_{i, t} \ge 30\text{ mV}: \quad s_{i, t} = 1, \quad v_{i, t} \leftarrow c, \quad u_{i, t} \leftarrow u_{i, t} + d$$
* **Exponential Pair-Based STDP with Hard/Soft Bounds:**
  $$\Delta w_{ij} = \begin{cases} A_+ \cdot (w_{\max} - w_{ij}) \cdot \exp\left(-\frac{\Delta t}{\tau_+}\right) & \text{if } \Delta t > 0 \text{ (causal / LTP)} \\ -A_- \cdot (w_{ij} - w_{\min}) \cdot \exp\left(\frac{\Delta t}{\tau_-}\right) & \text{if } \Delta t < 0 \text{ (anti-causal / LTD)} \end{cases}$$

### 5.2 Empirical Benchmark Summary (Experiment 007)

| Condition / Substrate | $Q_{\text{func}}$ (Spikes) | Dominant Freq | Kuramoto $R$ | $PLV$ | Task A Cue $R^2$ ($\Delta t=500$) | Task C FDR | Task D Flat MSE ($k=5$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control A (`RATE_CODED_PHASE6`)** | $-0.010$ | $0.0\text{ Hz}$ | $0.052$ | $0.076$ | $\mathbf{0.459}$ | $\mathbf{104.17}$ | $0.161$ |
| **Control B (`SPIKING_FROZEN`)** | $0.461$ | $50.0\text{ Hz}$ | $0.589$ | $0.785$ | $0.414$ | $0.00$ | $0.136$ |
| **Control C (`SPIKING_STDP_UNBOUNDED`)** | $\mathbf{0.464}$ | $50.0\text{ Hz}$ | $\mathbf{0.605}$ | $\mathbf{0.804}$ | $0.414$ | $0.00$ | $\mathbf{0.136}$ |
| **Control D (`SPIKING_STDP_HOMEOSTATIC`)**| $0.464$ | $50.0\text{ Hz}$ | $0.605$ | $0.804$ | $0.414$ | $0.00$ | $0.136$ |
| **Control E (`SPIKING_STDP_MODULATED`)**  | $0.464$ | $50.0\text{ Hz}$ | $0.605$ | $0.804$ | $0.414$ | $0.00$ | $0.136$ |
| *Ablation B (Spike Times Shuffled)* | $0.008$ | $50.0\text{ Hz}$ | $0.605$ | $0.804$ | $0.414$ | $0.00$ | $0.136$ |
| *Ablation C (Phase Scrambled)* | $0.464$ | $50.0\text{ Hz}$ | $0.089$ | $0.104$ | $0.414$ | $0.00$ | $0.136$ |

### 5.3 Key Scientific Findings
1. **Resolution of Functional Modularity (SUPPORTED)**: Discrete thresholding and axonal delays shattered the continuous global cross-talk of Phase 6. Measured functional modularity reached **$Q_{\text{func}} = 0.464 \pm 0.054$** across 25 distinct functional communities, decisively passing the $Q_{\text{func}} > 0.25$ threshold. Shuffling spike times collapsed modularity to $Q_{\text{func}} = 0.008$.
2. **Autonomous Oscillatory Limit Cycles (SUPPORTED)**: The Izhikevich population spontaneously developed a sustained **$50\text{ Hz}$ gamma rhythm** with peak spectral power $0.785 \pm 0.102$ without periodic external forcing.
3. **Phase-Locking Coherence (SUPPORTED)**: High inter-regional synchronization emerged (**Kuramoto $R = 0.605 \pm 0.083$**, **$PLV = 0.804 \pm 0.044$**). Scrambling instantaneous phases collapsed synchronization to $R = 0.089$.
4. **The Synchronization-Representation Trade-off (PARTIALLY SUPPORTED / CRITICAL CAVEAT)**: Spiking improved multi-step forward prediction ($\text{MSE} = 0.136$ vs $0.161$) and sustained coarse delayed cue memory ($R^2 \approx 0.414$). However, strong global phase synchronization compressed internal representations, wiping out fine temporal context disambiguation ($\text{FDR} \to 0.00$).
5. **Scale Ladder to 100,000 Units (PASSED)**: The substrate achieved **$944.8\text{ steps/s}$** ($4.72 \times 10^6\text{ spikes/s}$) at $N=100{,}000$ neurons and $1.5\times 10^6$ synapses with only **$17.17\text{ MB}$** memory overhead on Apple Silicon unified memory.

---

## 6. Next Steps

With Phase 1 through Phase 7 completed:
1. **Phase 8: Balanced Excitation/Inhibition & Asynchronous-Irregular Regimes**: Resolve the synchronization-representation bottleneck uncovered in Phase 7 by incorporating Dale's Principle (80% excitatory / 20% inhibitory interneurons), feedback fast-spiking inhibition, and homeostatic conductance balance to achieve sparse, information-rich processing in the biophysically grounded Asynchronous-Irregular (AI) regime.

