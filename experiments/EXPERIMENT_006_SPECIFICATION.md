# Experiment 006 Specification: Persistent Latent Memory & Predictive State

**Project:** 3D-Brain Research Program  
**Phase:** Phase 6  
**Experiment ID:** EXP-006  
**Date:** September 2026  
**Status:** APPROVED FOR EXECUTION  
**Lead Investigators:** 3D-Brain Research Team  

---

## 1. Scientific Objective

The objective of Experiment 006 is to empirically determine:
> **Can the embodied substrate develop a persistent internal latent state that represents information across time, supports delayed prediction, preserves useful information after the originating sensory input disappears, and generalizes to previously unseen temporal situations?**

Specifically, we test whether introducing a dedicated, multi-timescale persistent latent state $z_t \in \mathbb{R}^{D_z}$ coupled with the spatial recurrent substrate $h_t \in \mathbb{R}^H$ overcomes the rapid information-loss limitation of single-layer recurrent networks without resorting to non-neural external tools (LLMs, vector DBs).

---

## 2. Hypotheses & Falsification Criteria

### Primary Hypothesis
> A predictive embodied neural substrate with persistent latent state can learn representations that retain task-relevant information beyond the immediate sensory window and use that information for multi-step prediction and delayed inference.

### Secondary Hypotheses & Falsification Conditions

| Hypothesis | Description | Success Criterion | Falsification Condition |
|---|---|---|---|
| **H1: Persistent Memory** | State $z_t$ retains information across delay periods after input disappears. | Linear cue decodability $R^2 > 0.50$ across delays $\Delta t \ge 50$ steps. | $R^2 \le 0.10$ for $\Delta t \ge 50$ or no advantage over Phase 5 baseline. |
| **H2: Multi-Step Prediction** | Latent state enables lower multi-step prediction error than reactive baselines. | Prediction MSE at horizon $k=5, 10$ is $\ge 2\times$ lower than 1-step baseline. | Error is statistically indistinguishable from naive 1-step persistence forecast ($p > 0.05$). |
| **H3: Context Disambiguation** | Latent state separates identical sensory observations conditioned on divergent histories. | Fisher Separability $\text{FDR} > 2.0$ at ambiguous step $X$; disambiguation accuracy $> 85\%$. | Latent states for $A \to B \to X$ and $C \to D \to X$ are identical or $\text{FDR} < 0.2$. |
| **H4: Memory Efficiency** | Persistent latent memory outperforms parameter-matched larger recurrent networks. | Performance / parameter ratio is significantly higher for latent architecture ($p < 0.01$). | Increasing raw recurrent size $H$ matches or beats latent memory at equal parameter budget. |
| **H5: Continual Learning** | Persistent memory reduces catastrophic forgetting across environmental switches. | Backward transfer $R_{\text{retention}} > 0.80$ in recurring regime $A \to B \to A$. | Prior task associations are completely overwritten ($R_{\text{retention}} < 0.20$). |

---

## 3. Experimental Matrix

### 3.1 Architectures (5 Conditions)
1. **Phase 5 Baseline:** Standard `PredictiveSpatialRNN` ($H=40$, no latent state).
2. **Larger Recurrent Baseline:** Parameter-matched `PredictiveSpatialRNN` ($H=64$, matched parameter budget, no latent state).
3. **Persistent Latent Architecture:** `PersistentLatentSpatialRNN` ($H=40, D_z=24$, multi-timescale persistence $\lambda_z \in [0.85, 0.98]$).
4. **Frozen Latent Control:** `PersistentLatentSpatialRNN` with fixed random latent projections and no latent plasticity.
5. **Gradient Control:** BPTT-optimized predictive latent model.

### 3.2 Task Matrix (4 Task Domains)
1. **Task A: Delayed Cue Recall**
   - Delays: $\Delta t \in \{10, 25, 50, 100, 250, 500\}$ timesteps.
   - Measures cue retention curve, reconstruction MSE, and linear decoding $R^2$.
2. **Task B: Sequence Reconstruction**
   - Overlapping 4-element sequences ($A \to B \to C \to D$, $E \to B \to C \to F$, etc.).
   - Measures sequence order accuracy, start element recall, and transition prediction.
3. **Task C: Temporal Context Disambiguation**
   - Divergent histories: $A \to B \to X \to Y_1$ vs. $C \to D \to X \to Y_2$.
   - Evaluates state separability at ambiguous step $X$, next-step prediction accuracy, and latent distance.
4. **Task D: Multi-Step Forward Prediction**
   - Continuous embodied trajectory rollouts.
   - Horizons: $k \in \{1, 2, 5, 10, 25\}$ steps.
   - Evaluates MSE growth rate as a function of horizon $k$.

### 3.3 Environmental Regimes (4 Conditions)
- `STATIONARY`: Constant landmark configuration.
- `ABRUPT_SHIFT`: Single abrupt shift at midpoint ($t=125$).
- `RECURRING`: Reversion schedule ($A \to B \to A$).
- `GRADUAL_DRIFT`: Continuous landmark drift ($0.02\text{ units/step}$).

### 3.4 Causal Ablations (7 Conditions)
- `ABL_A_NO_PERSISTENCE`: $\lambda_z = 0$.
- `ABL_B_STEP_RESET`: State reset $z_{t-1} = 0$ every timestep.
- `ABL_C_DESTROY_HISTORY`: Random noise replaces historical cues.
- `ABL_D_SHUFFLE_ORDER`: Scrambled temporal sequence order.
- `ABL_E_NO_PREDICTIVE_LEARNING`: Frozen synaptic weights.
- `ABL_F_NO_STRUCTURAL_PLASTICITY`: Fixed static connectivity.
- `ABL_G_PARAM_MATCHED`: Parameter-budget equivalence.

### 3.5 Parameter Scaling Ladder
- $N \in \{300, 1000, 10000, 50000, 100000\}$ neurons evaluated for steps/sec, memory footprint, and telemetry overhead.

### 3.6 Replications
- Deterministic seeds: $\{42, 43, 44\}$.

---

## 4. Execution Plan & Artifacts

1. Run headless Python script: `python3 -m experiments.run_exp006`
2. Generate full machine-readable dataset: `experiments/results/exp006_results.json`
3. Export representative `.brainlog` v2.4 traces:
   - `experiments/results/brainlogs/EXP006_PERSISTENT_LATENT_ACTIVE_s42.brainlog`
   - `experiments/results/brainlogs/EXP006_CONTEXT_DISAMBIGUATION_s42.brainlog`
4. Compile empirical results report: `experiments/EXPERIMENT_006_RESULTS.md`
5. Compile comprehensive Phase 6 implementation report: `docs/PHASE_6_IMPLEMENTATION_REPORT.md`
