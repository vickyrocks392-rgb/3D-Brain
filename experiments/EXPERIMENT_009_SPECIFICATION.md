# Experiment 009 Specification: Three-Factor Credit Assignment & Active Functional Reorganization

**Project:** 3D-Brain Research Program  
**Phase:** Phase 9  
**Experiment ID:** EXP-009  
**Date:** September 2026  
**Status:** APPROVED FOR RIGOROUS EMPIRICAL EXECUTION  
**Lead Investigators:** 3D-Brain Research Team  

---

## 1. Scientific Framing & Motivation

Experiment 009 addresses the critical empirical finding of Phase 8:
> **Pair-based STDP is computationally passive in recurrent spiking networks.** Disabling STDP produced virtually identical functional modularity ($Q_{\text{func}} \approx 0.203$ vs $0.198$), identical context disambiguation accuracy ($58.3\%$), and null causal impact ($p > 0.50$). In Phase 7 and Phase 8, functional modularity was driven by topology and spontaneous E/I dynamical regimes rather than task-driven learning.

The scientific question for Phase 9 is:
> **Can three-factor synaptic plasticity—combining local synaptic eligibility traces $e_{ij}(t)$ with global or community-specific neuromodulatory signals $M(t)$—actively carve functional modularity ($Q_{\text{func}} > 0.30$) and rescue fine sensory context disambiguation across temporal delays while preserving persistent latent memory, multi-step prediction, and stable E/I balance?**

We do not assume three-factor learning succeeds. Negative results are first-class scientific findings. If eligibility traces fail to assign credit across long delays or fail to elevate $Q_{\text{func}}$ above $0.30$, we document that failure explicitly.

---

## 2. Formal Hypotheses & Falsification Criteria

| Hypothesis | Description | Success Criterion | Falsification / Negative Result Criterion |
|---|---|---|---|
| **H1: Eligibility-Driven Credit Assignment** | Synaptic eligibility traces $e_{ij}(t) = \lambda_e e_{ij}(t-1) + f(\text{pre}_i, \text{post}_j)$ retain temporal memory of coincidence, allowing a delayed neuromodulator $M(t)$ to selectively adapt task-relevant synapses. | Statistically significant positive correlation ($r > 0.30, p < 0.01$) between delayed reward and weight update; learning sustained across delays $\Delta t \ge 10$ steps. | Learning rate collapses to frozen baseline level for all delays $\Delta t > 1$, or eligibility traces decay to 0 before reward arrives. |
| **H2: Task-Relevant Functional Reorganization** | Three-factor plasticity produces greater task-directed structural/functional divergence than passive STDP or frozen controls. | $\Delta Q_{\text{func}} > 0$ relative to STDP; significant community weight specialization ($p < 0.05$). | Indistinguishable from STDP ($p > 0.05$) across functional and behavioral metrics. |
| **H3: Modularity Ceiling Breakthrough** | Reward-modulated plasticity selectively prunes non-functional cross-cluster connections and strengthens within-cluster pathways. | Functional modularity $Q_{\text{func}} \ge 0.30$ on empirical binned spike counts without global hypersynchrony ($R_{\text{global}} < 0.40$). | $Q_{\text{func}} < 0.25$ across all modulatory conditions, failing to surpass Phase 8 baseline ($0.17 - 0.23$). |
| **H4: Context Disambiguation Restoration** | Targeted credit assignment recovers temporal context disambiguation toward rate-coded Phase 6 reference (~90%). | Context classification accuracy $\ge 75.0\%$ (significantly above Phase 8's $58.3\% - 62.5\%$). | Context accuracy remains $\le 65.0\%$, showing credit assignment is unable to resolve instantaneous ambiguity. |
| **H5: Temporal Credit Assignment Horizon** | Learning effectiveness decays systematically with reward delay $\Delta t \in \{0, 1, 5, 10, 25, 50, 100, 250\}$. | Monotonic or characteristic decay curve matching theoretical eligibility horizon $\tau_e = 50\text{ ms}$; positive task reward for $\Delta t \le 25$. | Immediate breakdown at $\Delta t = 1$, or flat noise response across all delays. |

---

## 3. Experimental Matrix

### 3.1 Plasticity Comparison Matrix (Section 9)
All conditions evaluated under identical Dale E/I balance (80% E, 20% I, fast-spiking interneurons, and homeostatic synaptic scaling):
- **Condition A (FROZEN)**: Static weights, zero plasticity ($M(t) = 0$).
- **Condition B (PAIR_STDP)**: Standard pair-based STDP (Phase 8 baseline).
- **Condition C (ELIGIBILITY_ONLY)**: Constant modulation $M(t) = 1.0$ (Hebbian accumulation with eligibility decay, no teaching signal).
- **Condition D (GLOBAL_REWARD)**: Global scalar reward $M(t) = R(t)$.
- **Condition E (PRED_ERROR)**: Prediction error teaching signal $M(t) = \text{MSE}_{\text{pred}}(t)$.
- **Condition F (SIGNED_RPE)**: Signed Reward Prediction Error $M(t) = R(t) - \bar{V}(t)$ with exponential running baseline.
- **Condition G (LOCAL_COMMUNITY)**: Community-specific neuromodulation routing $M_c(t)$ based on spatial cluster performance.
- **REF_PHASE8**: Phase 8 E/I STDP + Homeostasis historical baseline.
- **REF_PHASE7**: Phase 7 Spiking Reference (hypersynchronous control).

### 3.2 Temporal Credit Assignment Horizon Battery (Section 10)
Evaluate learning performance, final reward, context accuracy, and weight delta across reward delays:
$$\Delta t_{\text{reward}} \in \{0, 1, 5, 10, 25, 50, 100, 250\}\text{ steps}$$

### 3.3 Causal Ablation Matrix (14 Distinct Conditions, Section 19)
1. `ABL_01_NO_ELIGIBILITY`: Eligibility trace zeroed out ($\lambda_e = 0$, $e_{ij} = 0$).
2. `ABL_02_NO_THIRD_FACTOR`: Neuromodulator clamped to $0$ ($M(t) = 0$).
3. `ABL_03_SHUFFLE_REWARD_TIME`: Reward signal temporally permuted across trials.
4. `ABL_04_SHUFFLE_SYNAPTIC_ELIG`: Eligibility traces randomly permuted across synapses at each step.
5. `ABL_05_SHUFFLE_NEUROMOD_TIMING`: Neuromodulator pulse jittered by random offsets $\pm 10$ steps.
6. `ABL_06_SIGN_FLIPPED_REWARD`: Reward polarity inverted ($M(t) \to -M(t)$).
7. `ABL_07_NO_HOMEOSTASIS`: Multiplicative synaptic scaling disabled ($\eta_{\text{homeo}} = 0$).
8. `ABL_08_BREAK_DALE`: Neuron-specific sign constraints removed; weights can cross zero arbitrarily.
9. `ABL_09_NO_SPATIAL_TOPOLOGY`: Modular 3D spatial geometry flattened to Erdős-Rényi random graph.
10. `ABL_10_NO_STDP_ELIGIBILITY`: Compare simple coincidence eligibility $f(\text{pre}, \text{post}) = \text{pre} \cdot \text{post}$ vs asymmetric STDP trace eligibility.
11. `ABL_11_FREEZE_RECURRENT`: Only readout weights plastic; recurrent substrate frozen.
12. `ABL_12_FREEZE_INHIBITORY`: Excitatory synapses plastic; inhibitory synapses frozen.
13. `ABL_13_FREEZE_EXCITATORY`: Inhibitory synapses plastic; excitatory synapses frozen.
14. `ABL_14_GLOBAL_VS_LOCAL`: Global broadcast reward vs community-localized reward signals.

### 3.4 Spatial Credit Assignment Comparison (Section 16)
Compare identical Three-Factor RPE learning on four 3D geometric topologies:
- `MODULAR` (4 clustered spatial modules)
- `SPATIAL_LOCAL` (metric distance-decay connectivity)
- `SMALL_WORLD` (Watts-Strogatz spatial clustering with long-range bridges)
- `RANDOM` (metric-free Erdős-Rényi graph)

### 3.5 Benchmark Tasks
1. **Task A: Delayed Contextual Choice**:
   Context A + Cue X $\to$ Action 1 $\to$ Reward  
   Context B + Cue X $\to$ Action 2 $\to$ Reward  
   Evaluated with delays $\Delta t \in \{0, 1, 5, 10, 25, 50, 100, 250\}$.
2. **Task B: Temporal Context Disambiguation** (Probe step identical observation, divergent histories).
3. **Task C: Delayed Cue Recall** ($\Delta t \in \{50, 100, 250, 500\}$ steps).
4. **Task D: Multi-Step Forward Prediction** ($k \in \{1, 2, 5, 10, 25\}$ rollout horizons).
5. **Task E: Continual Learning**: Sequential non-stationary environments $A \to B \to C$ measuring forward transfer, backward transfer, and catastrophic forgetting.
6. **Task F: Transformer Reference Comparison**: Direct head-to-head evaluation against a small causal Transformer baseline at matched parameter budget.

### 3.6 Computational Scaling Ladder
Hardware benchmark on Apple Silicon (M-series CPU):
$$N \in \{100, 1000, 10000, 50000, 100000\}\text{ neurons}$$
Track steps/sec, spikes/sec, synaptic events/sec, RAM footprint, eligibility overhead, and replay logging overhead. Specifically verify and resolve the Phase 8 $N=1\text{K}$ scaling anomaly.

---

## 4. Statistical Rigor & Stopping Criteria

- **Seeds**: Three independent deterministic random seeds per condition: $\{42, 43, 44\}$.
- **Metrics Reported**: Mean, standard deviation ($\sigma$), 95% bootstrap confidence intervals, Cohen's $d$ effect sizes, and two-tailed Welch's $t$-test / Mann-Whitney $U$-test $p$-values.
- **Stopping Criteria**: All trials run for fixed, pre-specified durations. No optional stopping or cherry-picking.
- **Falsification Guardrails**: Treat target thresholds ($Q_{\text{func}} > 0.30$, Context Acc $> 75\%$) strictly as testable hypotheses. If an outcome fails, report it unambiguously as `NOT SUPPORTED`.
