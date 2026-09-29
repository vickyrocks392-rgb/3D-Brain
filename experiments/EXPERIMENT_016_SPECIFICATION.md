# Experiment 016 Specification: Minimal Sufficient Brain
**Causal Architecture Pruning, Mechanism Necessity & Minimality Verification**

---

## 1. Central Scientific Question & Core Hypotheses

### Central Scientific Question
> *What is the smallest set of computational mechanisms that is sufficient to preserve contextual information, achieve reliable autonomous action, maintain low credit leakage, and remain dynamically stable?*

Phase 15 demonstrated that target-specific dendritic learning achieves $64.0\%\text{--}65.3\%$ autonomous accuracy (approaching analytical BPTT at $68.0\%$). However, the integrated candidate system (**COND_N**) combined seven distinct mechanisms and reached only $62.7\%$ with high credit leakage ($0.5594$).

### Hypotheses

- **$H_1$ (Teaching Signal Necessity):**
  Target-specific teaching ($G_1 / G_3$) is strictly necessary for credit assignment. Removing target directionality ($G_1 \to G_0$) will collapse accuracy to chance ($44.0\%$) and inflate credit leakage ($>0.70$).

- **$H_2$ (Retrograde Signaling Redundancy & Hazard):**
  Retrograde feedback ($R_2 / R_1$) without spine-level microdomains is non-essential ($C_{10} = 0.0\%$ delta) or actively harmful ($R_1 = -8.0\%$ delta). Silencing retrograde signaling will preserve accuracy ($\ge 60\%$) while improving credit localization ($\text{CSI} \ge 0.50$).

- **$H_3$ (Dendritic STDP Redundancy):**
  Dendritic STDP adds history buffers and computational overhead without providing instructional credit beyond what is already captured by pre-synaptic eligibility traces and apical plateau potentials. Ablating STDP will produce no meaningful degradation ($|\Delta \text{Acc}| < 2.0\%$).

- **$H_4$ (Timescale Assembly Minimality):**
  The tri-timescale assembly ($A_5$: Fast $25\text{ ms}$, Med $250\text{ ms}$, Slow $1500\text{ ms}$) is redundant. A single slow assembly ($A_2$) or a dual fast/slow assembly ($A_3$) is sufficient to eliminate decision chatter and bridge multi-second delay horizons.

- **$H_5$ (Existence of a Minimal Sufficient Brain):**
  A reduced architecture containing $\le 8$ mechanisms will satisfy the predefined acceptance vector ($\text{Acc} \ge 60\%$, $\text{CSI} \ge 0.45$, $\text{Leakage} \le 0.55$, $\text{Chatter} = 0$, $\text{MI}_{\text{ctx}} \ge 0.90\text{ bit}$) while cutting state variables and complexity by $>30\%$.

---

## 2. Formal Minimality Principles

1. **NECESSARY:** Removing subsystem $\mathcal{M}$ causes statistically significant degradation ($p < 0.05$ or $\Delta \text{Acc} \le -5.0\%$) in one or more acceptance criteria.
2. **REDUNDANT:** Removing subsystem $\mathcal{M}$ causes no meaningful degradation across any acceptance criteria ($|\Delta \text{Acc}| < 2.0\%$).
3. **HARMFUL:** Removing subsystem $\mathcal{M}$ improves performance ($\Delta \text{Acc} \ge +2.0\%$ or $\Delta \text{CSI} \ge +0.05$) without violating stability.
4. **SYNERGISTIC:** $\text{Perf}(A+B) - \text{Perf}(\text{Base}) > \Delta_A + \Delta_B$, or the pair preserves a capability neither can achieve alone.

---

## 3. Formal Acceptance Vector

| Index | Metric | Variable | Predefined Threshold | Baseline Reference (P15) |
|---|---|---|---|---|
| **Crit-A** | Autonomous Action Accuracy | $\text{Acc}$ | $\ge 60.0\%$ | $62.67\%$ |
| **Crit-B** | Contextual Shannon Mutual Information | $I(X; S)$ | $\ge 0.900\text{ bit}$ | $0.9968\text{ bit}$ |
| **Crit-C** | Action-Relevant Mutual Information | $I(S; A^*)$ | $\ge 0.200\text{ bit}$ | $0.2701\text{ bit}$ |
| **Crit-D** | Credit Specificity Index | $\text{CSI}$ | $\ge 0.4500$ | $0.4406$ |
| **Crit-E** | Credit Leakage Rate | $\text{Leakage}$ | $\le 0.5500$ | $0.5594$ |
| **Crit-F** | Decision Chatter | $\text{Switches}$ | $\equiv 0$ | $0$ |
| **Crit-G** | Decision Margin | $\text{Margin}$ | $\ge 1.000$ | $4.788$ |
| **Crit-H** | Delay Robustness ($2000\text{ ms}$) | $\text{Acc}_{2000}$ | $\ge 50.0\%$ | $56.0\%$ |
| **Crit-I** | Dynamical Stability | $r_{\text{local}}$ | $< 0.850$ | $0.521$ |
| **Crit-J** | Complexity Reduction | $\Delta \text{Mech}$ | $\ge 1$ mechanism pruned | $0$ |

---

## 4. Experimental Architecture & Evaluation Matrix

### Phase 1: One-at-a-Time Systematic Ablations
Evaluate `FULL_P15` vs single-mechanism removals:
- `FULL_P15`: All 12 mechanisms active.
- `ABL_A`: Full minus Target-Specific Reward Teaching (falls back to global scalar $G_0$).
- `ABL_B`: Full minus Dendritic Teaching Signal (falls back to action-target scalar $G_1$).
- `ABL_C`: Full minus Dendritic Eligibility (falls back to flat synaptic eligibility).
- `ABL_D`: Full minus Dendritic STDP ($S_0$ None).
- `ABL_E`: Full minus Retrograde Signaling ($R_0$ None).
- `ABL_F`: Full minus Fast Action Assembly (Medium + Slow only).
- `ABL_G`: Full minus Medium Action Assembly (Fast + Slow only).
- `ABL_H`: Full minus Slow Action Assembly (Fast + Medium only).
- `ABL_I`: Full minus Mutual Action Competition ($w_{\text{cross\_inh}} = 0$).
- `ABL_J`: Full minus Temporal Population Representation ($I_0$ instantaneous spikes).
- `ABL_K`: Full minus Persistent Latent / NMDA State ($g_{\text{nmda}} \equiv 0$).
- `ABL_L`: Full minus Target-Specific Action Routing (uniform drive broadcast).

### Phase 2: Pairwise Ablation Matrix
- `ABL_DE`: STDP + Retrograde removed.
- `ABL_FG`: Fast + Medium assemblies removed (Slow only).
- `ABL_GH`: Medium + Slow assemblies removed (Fast only).
- `ABL_FH`: Fast + Slow assemblies removed (Medium only).
- `ABL_FGH`: All assemblies removed (Continuous direct readout).
- `ABL_IF`: Mutual inhibition + Fast assembly removed.
- `ABL_IH`: Mutual inhibition + Slow assembly removed.
- `ABL_BD`: Dendritic teaching + STDP removed.
- `ABL_BC`: Dendritic teaching + Dendritic eligibility removed.
- `ABL_CD`: Dendritic eligibility + STDP removed.

### Phase 3: Multi-Timescale Assembly Minimality Sweep
- `A0_FAST_ONLY` ($\tau = 25\text{ ms}$)
- `A1_MEDIUM_ONLY` ($\tau = 250\text{ ms}$)
- `A2_SLOW_ONLY` ($\tau = 1500\text{ ms}$)
- `A3_FAST_MEDIUM`
- `A4_MEDIUM_SLOW`
- `A5_FAST_MEDIUM_SLOW`
- `DIRECT_READOUT` (no assembly)

### Phase 4: Retrograde Signaling Final Causal Audit
- `R0_NONE`: Silenced retrograde.
- `R1_DIFFUSE`: Diffuse scalar broadcast.
- `R2_ACTION_SPECIFIC`: Confined strictly to chosen action.
- `R3_COMPARTMENT`: Gated by local dendritic voltage.
- `R4_DELAYED`: 25 ms propagation latency.

### Phase 5: Target-Specific Teaching Signal Necessity Sweep
- `G0_GLOBAL_SCALAR`: Undifferentiated reward broadcast.
- `G1_ACTION_TARGET_SCALAR`: Channel-directed scalar reward.
- `G2_POPULATION_LOCAL`: Locally averaged scalar.
- `G3_DENDRITIC_TARGET`: Apical dendritic plateau teaching vector.
- `G4_SHUFFLED_TARGET`: Permuted action target control.
- `G5_SIGN_FLIPPED`: Inverted gradient anti-learning control.

### Phase 6: Information Bottleneck Audit (7 Stages)
Measure continuous Shannon MI ($I(X; S)$) and action-relevant MI ($I(S; A^*)$) across:
1. `1_SENSORY_CONTEXT`
2. `2_PERSISTENT_LATENT`
3. `3_TEMPORAL_POPULATION`
4. `4_DENDRITIC_REPRESENTATION`
5. `5_ACTION_INTERFACE`
6. `6_ACTION_ASSEMBLY`
7. `7_FINAL_ACTION`

### Phase 7: Causal Action Interventions (C1 to C12)
Evaluate intact vs perturbed rollouts across all 12 interventions.

### Phase 8: Candidate Architecture Evaluation
Compare `FULL_P15` against candidate architectures `MIN_A`, `MIN_B`, `MIN_C`, `MIN_D`, `MIN_E`.

---

## 5. Statistical & Reproducibility Methodology

- **Seeds:** Deterministic seeds $42, 43, 44$ ($n=25$ trials per seed = $75$ trials per condition).
- **Task:** `LongDelayContextualCreditTask` with delay horizon $10\text{ steps} = 400\text{ ms}$ (and sweeps up to $5000\text{ ms}$).
- **Reporting:** Mean, standard deviation, Standard Error of the Mean (SEM), delta differences ($\Delta$), and paired differences.
- **Controls:** Analytical BPTT ($68.0\%$), Small Transformer ($72.0\%$), Continuous Oracle ($96.0\%$).
