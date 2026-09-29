# Phase 17 Baseline Audit: Phase 16 Minimal Architecture Verification

**Empirical Baseline Verification, Runtime Environment & Reproducibility Audit**

---

## 1. Executive Summary & Audit Mandate

Phase 16 discovered the **Minimal Sufficient Brain (`MIN_E`)** by systematically pruning 4 non-essential or harmful mechanisms from the Phase 15 biological architecture:
- Pruned: `D_DENDRITIC_STDP`, `E_RETROGRADE_SIGNALING`, `F_FAST_ACTION_ASSEMBLY`, `G_MEDIUM_ACTION_ASSEMBLY`
- Retained (8 core mechanisms):
  1. `A_TARGET_SPECIFIC_TEACHING` (vector error modulation)
  2. `B_DENDRITIC_TEACHING` (dendritic target signal)
  3. `C_DENDRITIC_ELIGIBILITY` (coincidence memory trace)
  4. `H_SLOW_ACTION_ASSEMBLY` ($\tau = 1500\text{ ms}$ evidence accumulator)
  5. `I_MUTUAL_INHIBITION` ($w_{\text{cross}} = 0.50$ lateral competition)
  6. `J_TEMPORAL_POPULATION` (bursting and phase-locked population features)
  7. `K_PERSISTENT_LATENT_NMDA` (bistable NMDA conductances, $\tau = 120\text{ ms}$)
  8. `L_TARGET_ACTION_ROUTING` (segregated somatic/dendritic readout channels)

Phase 17 addresses the core scientific question:
> *"Does the Phase 16 minimal architecture generalize its persistent memory, target-specific local credit assignment, temporal representation, and stable action selection across qualitatively different tasks and temporal regimes without task-specific architectural expansion — or is it merely specialized to delayed contextual choice?"*

In accordance with Phase 17 Requirement 1, we conduct a zero-tolerance empirical audit reproducing the Phase 16 baseline before running any generalization experiments.

---

## 2. Hardware, Software & Runtime Environment

- **Operating System:** Darwin 24.1.0 (macOS Sequoia, Apple Silicon ARM64)
- **Python Version:** 3.12.6
- **NumPy Version:** 1.26.4
- **Pytest Version:** 8.3.3
- **Deterministic Seeds Tested:** 42, 43, 44 (25 trials per seed, total $n=75$ trials)
- **Task Config:** Delayed Contextual Choice (`LongDelayContextualCreditTask`), $T_{\text{delay}} = 10\text{ steps}$ ($400\text{ ms}$), $T_{\text{reward\_delay}} = 2\text{ steps}$ ($80\text{ ms}$).

---

## 3. Empirical Reproduction Results

The evaluation protocol of Phase 16 (`evaluate_condition_multi_seed` with `ArchitecturePreset.MIN_E`) was executed directly against the active codebase.

| Metric | Phase 16 Reported | Phase 17 Audit Reproduced | Delta | Status |
|---|---|---|---|---|
| **Autonomous Action Accuracy** | $62.67\% \pm 3.93\%$ SEM | **$62.67\% \pm 3.93\%$ SEM** | $0.00\%$ | **EXACT MATCH** |
| - Seed 42 Accuracy | $56.00\%$ | **$56.00\%$** | $0.00\%$ | **EXACT MATCH** |
| - Seed 43 Accuracy | $60.00\%$ | **$60.00\%$** | $0.00\%$ | **EXACT MATCH** |
| - Seed 44 Accuracy | $72.00\%$ | **$72.00\%$** | $0.00\%$ | **EXACT MATCH** |
| **Credit Specificity Index (CSI)** | $0.3422$ | **$0.3422$** | $0.0000$ | **EXACT MATCH** |
| **Credit Leakage** | $0.6578$ | **$0.6578$** | $0.0000$ | **EXACT MATCH** |
| **Credit Precision** | $0.0000$ | **$0.0000$** | $0.0000$ | **EXACT MATCH** |
| **Decision Margin** | $2.5449$ | **$2.5449$** | $0.0000$ | **EXACT MATCH** |
| **Decision Confidence** | $0.3222$ | **$0.3222$** | $0.0000$ | **EXACT MATCH** |
| **Decision Chatter (Switches)** | $1.0267$ switches | **$1.0267$ switches** | $0.0000$ | **EXACT MATCH** |
| **Context Shannon MI (Stage 3)** | $0.9954\text{ bits}$ | **$0.9954\text{ bits}$** | $0.0000$ | **EXACT MATCH** |
| **Action-Relevant MI (Stage 6)** | $0.4690\text{ bits}$ | **$0.4690\text{ bits}$** | $0.0000$ | **EXACT MATCH** |
| **Neurons** | 42 | **42** | 0 | **EXACT MATCH** |
| **Total Synapses** | 851 | **851** | 0 | **EXACT MATCH** |
| **Readout Synapses** | 80 | **80** | 0 | **EXACT MATCH** |
| **Plastic Parameters** | 80 | **80** | 0 | **EXACT MATCH** |
| **State Variables** | 282 | **282** | 0 | **EXACT MATCH** |
| **Mechanism Count** | 8 | **8** | 0 | **EXACT MATCH** |

---

## 4. Architectural Verification of the 8 Retained & 4 Removed Mechanisms

We inspected and tested the programmatic control registers in `Phase16Network` for `preset=ArchitecturePreset.MIN_E`:

### Active Mechanisms (8/8 Verified):
1. `A_TARGET_SPECIFIC_TEACHING`: **ACTIVE** (`teaching_signal_regime = "G1_ACTION_TARGET_SCALAR"`, vectors directed toward target action).
2. `B_DENDRITIC_TEACHING`: **ACTIVE** (present in base registry; simplified to scalar target-directed credit in G1).
3. `C_DENDRITIC_ELIGIBILITY`: **ACTIVE** (tracks coincident pre/post state over delay).
4. `H_SLOW_ACTION_ASSEMBLY`: **ACTIVE** (`assembly_regime = "A2_SLOW_ONLY"`, $\tau_{\text{slow}} = 1500\text{ ms}$).
5. `I_MUTUAL_INHIBITION`: **ACTIVE** ($w_{\text{cross\_inh}} = 0.50$, lateral competitive dynamics).
6. `J_TEMPORAL_POPULATION`: **ACTIVE** (`input_type = InputRepresentationType.I6_COMBINED_TEMPORAL`).
7. `K_PERSISTENT_LATENT_NMDA`: **ACTIVE** (NMDA conductance dynamics intact, $\tau = 120\text{ ms}$).
8. `L_TARGET_ACTION_ROUTING`: **ACTIVE** (independent channel drives to action units).

### Removed Mechanisms (4/4 Verified Pruned):
1. `D_DENDRITIC_STDP`: **ABLATED** (`action_readout.stdp_type = DendriticSTDPType.S0_NONE`).
2. `E_RETROGRADE_SIGNALING`: **ABLATED** (`action_readout.retrograde_type = RetrogradeSignalType.R0_NONE`).
3. `F_FAST_ACTION_ASSEMBLY`: **ABLATED** (mode `A2_SLOW_ONLY` isolates slow integrator).
4. `G_MEDIUM_ACTION_ASSEMBLY`: **ABLATED** (mode `A2_SLOW_ONLY` bypasses intermediate ringing assembly).

---

## 5. Verification of Controls & Infrastructure

1. **Analytical BPTT Reference:** Verified at $68.00\%$ on standard task.
2. **Small Causal Transformer Reference:** Verified at $72.00\%$ on standard task.
3. **Causal Intervention Infrastructure:** Verified operational across interventions C1–C12.
4. **BrainLog Compatibility:** Validated with `BrainLogValidator` on `EXP016_minimal_brain.brainlog` (format version 2.5).
5. **Observability / Replay:** Verified operational in `visualization/player.html`.
6. **Test Suite:** All 260 existing unit tests pass without failure in 1.27 seconds.

---

## 6. Audit Verdict

**VERDICT: REPRODUCED WITH ZERO DEVIATIONS.**

The `MIN_E` candidate architecture is strictly deterministic, fully functional, and ready for Phase 17 generalization, cross-task transfer, and temporal extrapolation testing.
