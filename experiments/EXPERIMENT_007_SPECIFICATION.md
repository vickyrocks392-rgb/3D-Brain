# Experiment 007 Specification: Spiking Neural Dynamics, STDP & Emergent Synchronization

**Project:** 3D-Brain Research Program  
**Phase:** Phase 7  
**Experiment ID:** EXP-007  
**Date:** September 2026  
**Status:** APPROVED FOR EXECUTION  
**Lead Investigators:** 3D-Brain Research Team  

---

## 1. Scientific Objective

The central objective of Experiment 007 is to empirically resolve the fundamental question posed by Phase 6:
> **Can spiking neuron dynamics (Leaky Integrate-and-Fire and Izhikevich substrates with Spike-Timing-Dependent Plasticity) produce genuine functional modularity ($Q_{\text{func}} > 0.25$) and oscillatory phase-locking limit cycles while preserving persistent latent predictive memory?**

In Phase 6, rate-coded continuous spatial RNNs achieved robust multi-timescale persistent latent memory, but completely failed to develop functional modularity ($Q_{\text{func}} \approx -0.010$) despite strong underlying structural modularity ($Q_{\text{struct}} \approx 0.450$). Furthermore, continuous activation dynamics could not produce autonomous oscillatory phase-locking limit cycles.

Experiment 007 tests whether replacing continuous rate-coded dynamics with discrete spike-timing dynamics, finite spatial conduction delays, and local STDP resolves this modularity and synchronization deficit, or whether a fundamental computational trade-off exists between synchronization and memory capacity.

---

## 2. Hypotheses & Falsification Criteria

Every hypothesis must receive one of: **SUPPORTED**, **PARTIALLY SUPPORTED**, **NOT SUPPORTED**, or **INCONCLUSIVE**, backed by quantitative evidence.

| Hypothesis | Description | Success Criterion | Falsification Condition |
|---|---|---|---|
| **H1: Spiking Dynamics** | Discrete threshold-crossing integration generates non-trivial temporal dynamics (ISIs, refractory recovery, burst distributions). | Membrane potential trajectories exhibit distinct integration-reset cycles; ISI coefficient of variation $CV_{\text{ISI}} > 0.40$. | Membrane potential fails to reset, or firing is statistically indistinguishable from uniform rate-coded noise ($CV_{\text{ISI}} \approx 0$ or constant). |
| **H2: Functional Modularity** | STDP and conduction delays enable the emergence of distinct functional communities from spike activity. | Functional modularity $Q_{\text{func}} > 0.25$ derived independently from binned spike-count correlations via Louvain community detection. | $Q_{\text{func}} \le 0.05$ across all topologies, parameter regimes, and STDP rules. |
| **H3: Oscillatory Dynamics** | Population activity exhibits autonomous rhythmic oscillations with defined spectral peaks. | Population power spectral density exhibits a dominant oscillatory peak with spectral entropy $H_{\text{spec}} < 0.85$ and distinct autocorrelation peaks. | Flat white-noise spectral density ($H_{\text{spec}} \ge 0.95$) or monotonic exponential decay without oscillatory peaks. |
| **H4: Phase Locking** | Subpopulations form stable phase-locking limit cycles with measurable phase coherence. | Kuramoto order parameter $R > 0.35$ and pairwise Phase-Locking Value $PLV > 0.40$ between structurally connected functional clusters. | $R < 0.15$ or phase differences drift uniformly over time without phase-locking. |
| **H5: Spatial Organization** | Spatial metric embedding and conduction delays generate localized functional clusters and traveling waves. | High spatial participation ratio ($PR_{\text{spatial}} > 0.40$), distance-dependent cross-correlations, and detectable wave latencies. | Zero spatial structure in functional correlation matrix; identical dynamics regardless of 3D Euclidean distances. |
| **H6: Memory + Synchronization** | Spiking networks with persistent latent memory retain information across delays while maintaining stable dynamics. | Delayed cue recall $R^2 > 0.50$ across delays $\Delta t \ge 50$ steps in Spiking + Latent configurations, without runaway epileptic synchronization ($R < 0.90$). | Complete memory loss ($R^2 < 0.10$ for $\Delta t \ge 50$), or extreme synchronization that destroys representational capacity (memory-sync trade-off collapse). |

---

## 3. Experimental Matrix

### 3.1 Control Conditions (7 Primary Controls)
1. **Control A: Phase 6 Rate-Coded Persistent Latent Spatial RNN** (Authoritative control baseline from Phase 6).
2. **Control B: Spiking LIF with Frozen Synapses** (Spike dynamics with static weights).
3. **Control C: Spiking LIF with Pair-Based STDP** (Classical asymmetric Hebbian/anti-Hebbian timing window).
4. **Control D: Spiking LIF with STDP + Homeostatic Scaling** (Activity-dependent synaptic scaling to prevent runaway excitation).
5. **Control E: Spiking LIF with STDP + Neuromodulation** (Reward/error-gated plasticity traces).
6. **Control F: Spiking LIF + Persistent Latent Memory** (Coupled LIF recurrent layer with multi-timescale persistent latent module, frozen recurrent weights).
7. **Control G: Spiking LIF + Persistent Latent Memory + STDP** (Full hybrid architecture: spiking dynamics, STDP, and persistent latent memory).
*(Additionally, an **Izhikevich + STDP** condition is tested to evaluate complex burst/chattering dynamics against LIF).*

### 3.2 Topology Matrix (4 Structural Configurations)
- **Random Topology** (Erdős–Rényi metric-free baseline).
- **Spatial-Local Topology** (Exponential distance decay connectivity).
- **Modular Topology** (High intra-cluster density, sparse inter-cluster bridge connections).
- **Small-World Topology** (Watts–Strogatz spatial model with rewiring).

### 3.3 Task Matrix (3 Core Benchmark Tasks)
1. **Task A: Delayed Cue Recall**
   - Delays: $\Delta t \in \{10, 25, 50, 100, 250, 500\}$ timesteps.
   - Evaluates cue retention $R^2$, cue readout reconstruction MSE, and memory decay rate.
2. **Task C: Temporal Context Disambiguation**
   - Divergent histories: $A \to B \to X \to Y_1$ vs. $C \to D \to X \to Y_2$.
   - Evaluates state separability at ambiguous stimulus $X$, next-step prediction accuracy, and latent representation distance.
3. **Task D: Multi-Step Forward Prediction**
   - Horizons: $k \in \{1, 3, 5\}$ steps ahead.
   - Evaluates rollout MSE and prediction degradation as a function of horizon.

### 3.4 Causal Ablations (6 Conditions)
- `ABL_A_STDP_OFF`: Synapses frozen ($\Delta W = 0$).
- `ABL_B_SPIKE_TIMING_SHUFFLED`: Spike times randomly shuffled within each trial while strictly preserving individual neuron firing rates.
- `ABL_C_PHASE_RELATIONSHIPS_DESTROYED`: Random phase shifts applied across subpopulation channels.
- `ABL_D_TOPOLOGY_SHUFFLED`: Synaptic adjacency matrix degree-preserving degree-sequence rewiring.
- `ABL_E_LATENT_RESET`: Persistent latent state $m_t$ reset to zero at the start of the delay period.
- `ABL_F_OSCILLATION_DISRUPTED`: Membrane potentials perturbed with uncorrelated high-variance noise every 10 steps to break rhythmic coherence.

### 3.5 Network Scale Ladder
- Evaluated network sizes: $N \in \{100, 300, 1000, 10000, 50000, 100000\}$ neurons.
- Metrics: Simulation steps/sec, spikes/sec, synaptic events/sec, memory footprint (MB), and telemetry serialization overhead.

### 3.6 Replications & Determinism
- Three independent seeds per condition: $\{42, 43, 44\}$.
- Fixed computational budgets reported explicitly (timesteps, synaptic events, parameter counts).

---

## 4. Execution Plan & Artifacts

1. Run headless Python benchmark: `python3 experiments/run_exp007.py`
2. Export comprehensive results dataset: `experiments/results/exp007_results.json`
3. Export validated `.brainlog` v2.5 traces:
   - `experiments/results/brainlogs/EXP007_SPIKING_STDP_MODULAR_s42.brainlog`
   - `experiments/results/brainlogs/EXP007_SPIKING_LATENT_HYBRID_s42.brainlog`
4. Compile empirical results report: `experiments/EXPERIMENT_007_RESULTS.md`
5. Compile comprehensive Phase 7 implementation report: `docs/PHASE_7_IMPLEMENTATION_REPORT.md`
