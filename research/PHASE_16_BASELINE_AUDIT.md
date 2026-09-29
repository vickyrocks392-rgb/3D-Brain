# Phase 16 Baseline Audit: Minimal Sufficient Brain
**Empirical Baseline Verification, Formal Minimality Principles & Acceptance Vector**

---

## 1. Executive Summary & Audit Mandate

Phase 15 successfully resolved the temporal-representation-to-action readout bottleneck by demonstrating that biologically constrained target-specific dendritic learning can achieve **$64.0\% \text{ to } 65.3\%$** autonomous task accuracy (closing within $2.7\%$ of analytical BPTT at $68.0\%$). However, the integrated Phase 15 candidate system (**COND_N**) combined multiple complex mechanisms simultaneously:
1. Temporal population representation ($I_6$)
2. Target-specific dendritic teaching ($G_3$)
3. Action-specific retrograde signaling ($R_2$)
4. Target-specific dendritic STDP ($S_4$)
5. Tri-timescale action assembly ($A_5$: Fast $25\text{ ms}$, Med $250\text{ ms}$, Slow $1500\text{ ms}$)
6. Lateral mutual inhibition ($w_{\text{cross}} = 0.50$)
7. Two-compartment NMDA/dendritic plateau dynamics

Crucially, **COND_N** achieved only **$62.7\%$** accuracy, with a Credit Specificity Index (CSI) of $0.4406$ and credit leakage of $0.5594$. Meanwhile, isolated mechanisms (such as Target-Specific Reward alone, COND_G) reached $64.0\%$ accuracy with superior credit localization ($\text{CSI} = 0.5778$, $\text{Leakage} = 0.4222$).

Phase 16 therefore rejects the premise that adding more biological mechanisms inevitably improves performance. Instead, Phase 16 executes a systematic, rigorous **Causal Architecture Pruning Study** to discover the **Minimal Sufficient Brain**:
> *What is the smallest set of computational mechanisms that is sufficient to preserve contextual information, achieve reliable autonomous action, maintain low credit leakage, and remain dynamically stable?*

---

## 2. Comprehensive Audit of Phase 15 Research Reports

Before conducting pruning experiments, all core Phase 15 research documents were systematically reviewed:

1. **`research/PHASE_15_RESULTS.md`**:
   - Confirmed that target-specific dendritic teaching ($G_1, G_3$) and dendritic eligibility traces achieve $64.0\%\text{--}65.3\%$ accuracy.
   - Identified the **Retrograde Signaling Hazard**: diffuse retrograde feedback caused severe leakage ($\text{CSI} = 0.441$, accuracy dropped to $49.3\%$).
   - Established that fast assemblies ($25\text{ ms}$) react immediately while slow assemblies ($1500\text{ ms}$) eliminate decision chatter.
2. **`research/ACTION_READOUT_ANALYSIS.md`**:
   - Documented the modular `ActionReadout` architecture ($W_{kj} \in \mathbb{R}^{2 \times N_{\text{in}}}$).
   - Confirmed the lesion test ($C_3$): lesioning readout synapses collapsed accuracy to $20.0\%$ ($\Delta = -36.0\%$), proving causal necessity.
3. **`research/TARGET_SPECIFIC_CREDIT_ANALYSIS.md`**:
   - Analyzed the credit assignment dilemma: global scalar rewards broadcast non-specific feedback, driving accuracy to chance ($44.0\%$).
   - Verified that vector target teaching signals ($G_1, G_3$) localize plastic updates, whereas shuffled ($G_4$) and sign-flipped ($G_5$) controls collapse accuracy to $44.0\%$.
4. **`research/DENDRITIC_RETROGRADE_ANALYSIS.md`**:
   - Proved that diffuse retrograde feedback ($R_1$) degrades performance by $-8.0\%$ due to un-compartmentalized cross-talk across active upstream eligibility traces.
   - In causal intervention $C_{10}$, removing retrograde signaling resulted in a $0.0\%$ delta on performance, establishing its potential redundancy.
5. **`research/MULTI_TIMESCALE_ACTION_ANALYSIS.md`**:
   - Demonstrated zero action decay across $5000\text{ ms}$ delay horizons and zero switching chatter (0 switches across all evaluation trials).
   - Showed that the medium and slow assemblies provide behavioral hysteresis, but left open whether all three timescales are strictly necessary.
6. **`research/CAUSAL_ACTION_INFORMATION_ANALYSIS.md`**:
   - Tested causal interventions $C_1 \to C_{12}$, confirming the causal roles of readout synapses ($C_3$), target teaching ($C_8, C_9$), and action population identity ($C_7$).
7. **`research/PHASE_15_ABLATIONS.md`**:
   - Documented initial ablations ABL_01 to ABL_08, highlighting the catastrophic failure of ablating target specificity ($G_1 \to G_0$, $44.0\%$).
8. **`research/INFORMATION_MEASUREMENT_AUDIT_PHASE15.md`**:
   - Resolved the apparent DPI violation: the legacy 1-step instantaneous BSC proxy $C = 1 - H_2(P_e)$ discarded temporal evidence. Continuous Shannon MI demonstrated that the recurrent population temporal state preserves $0.9968\text{ b}$ ($99.7\%$).
9. **`research/TRANSFORMER_ALTERNATIVES_STATUS.md`**:
   - Maintained formal status: **RESEARCH ONGOING — NO FINAL ALTERNATIVE SELECTED**. Standard causal Transformer baseline serves as a $72.0\%$ control.

---

## 3. Experimental Verification & Reference Baselines

The Phase 15 reference values were experimentally verified using deterministic evaluation across identical seeds ($n=75$ trials; seeds 42, 43, 44; 25 trials each, delay horizon $10\text{ steps} = 400\text{ ms}$):

### Full Phase 15 Integrated Candidate (`FULL_P15` / `COND_N`)
- **Autonomous Action Accuracy:** $62.67\% \pm 2.88\%$ SEM (Seed 42: $56.0\%$, Seed 43: $64.0\%$, Seed 44: $68.0\%$; standard deviation: $4.99\%$).
- **Offline Linear Readout Accuracy:** $53.33\%$ (cross-validated split; $72.0\%$ unregularized train).
- **Analytical BPTT Reference:** $68.00\%$.
- **Small Transformer Baseline Reference:** $72.00\%$.
- **Continuous Oracle (Upper Bound):** $96.00\%$.
- **Credit Specificity Index (CSI):** $0.4406$ (Seed 42: $0.5511$, Seed 43: $0.5051$, Seed 44: $0.2655$).
- **Credit Precision:** $0.4800$.
- **Credit Leakage:** $0.5594$.
- **Credit Gini Concentration:** $0.4406$.
- **Contextual Mutual Information ($I(X; S)$):** $0.9968\text{ bit}$ ($99.7\%$ of 1-bit binary context).
- **Action-Relevant Mutual Information ($I(S; A^*)$):**
  - Action Population (Stage 10): $0.2701\text{ bit}$.
  - Persistent Action Assembly (Stage 11): $0.4793\text{ bit}$.
- **Decision Margin:** $4.7881$ (Seed 42: $14.0600$, Seed 43: $0.2193$, Seed 44: $0.0851$).
- **Decision Confidence:** $0.3733$.
- **Decision Switches / Chatter:** $0.00$ switches per trial.
- **Delayed-Action Horizon Performance:**
  - $0\text{ ms}$ delay: $56.0\%$
  - $250\text{ ms}$ delay: $56.0\%$
  - $1000\text{ ms}$ delay: $56.0\%$
  - $5000\text{ ms}$ delay: $56.0\%$
- **Memory Retention (Attractor Persistence):** $1.000$ (zero drift during delay).
- **Dynamical Synchronization:** $r_{\text{local}} = 0.521$, $r_{\text{global}} = 0.484$ (asynchronous-irregular regime preserved).
- **Functional Modularity:** $Q_{\text{func}} = 0.423$, $Q_{\text{struct}} = 0.481$.
- **Substrate Parameter Count:**
  - Upstream Recurrent Synapses: $40 \times 40 = 1,600$ (distance-pruned to $820$ active).
  - Sensory Afferents: $8 \times 40 = 320$.
  - Action Readout Synapses: $40 \times 2 = 80$ plastic parameters.
  - Readout Eligibility Traces: $80$ state variables.
  - Dendritic Eligibility Traces: $80$ state variables.
  - Assembly Variables: $6$ state variables (Fast, Med, Slow for 2 actions).
  - Total Plastic Parameters in Action Interface: $80$.
  - Total State Variables in Action Interface: $166$.
- **Compute Cost:** $161.3\text{ steps/sec}$, inference latency $\approx 0.082\text{ ms/step}$.

---

## 4. Formal Minimality Principles

To classify subsystems without subjective bias, Phase 16 defines four formal mathematical criteria:

Let $\mathcal{A}$ be an architecture configuration, and let $\mathcal{M}$ be a candidate subsystem. Let $\mathcal{A} \setminus \{\mathcal{M}\}$ denote the ablated architecture. Let $\mathbf{V}(\mathcal{A})$ denote the evaluation metric vector:
$$\mathbf{V}(\mathcal{A}) = \left[ \text{Acc}(\mathcal{A}), \text{CSI}(\mathcal{A}), \text{Leak}(\mathcal{A}), \text{MI}_{\text{ctx}}(\mathcal{A}), \text{MI}_{\text{act}}(\mathcal{A}), \text{Margin}(\mathcal{A}), \text{Chatter}(\mathcal{A}) \right]$$

### 4.1 Definition: Subsystem Necessity
A subsystem $\mathcal{M}$ is **NECESSARY** if and only if removing it produces a statistically significant degradation ($p < 0.05$ or $\Delta \text{Acc} \le -5.0\%$, or violation of predefined acceptance thresholds) in one or more required capabilities:
$$\mathcal{M} \in \text{NECESSARY} \iff \text{Acc}(\mathcal{A} \setminus \{\mathcal{M}\}) < \text{Acc}_{\text{thresh}} \quad \lor \quad \text{CSI}(\mathcal{A} \setminus \{\mathcal{M}\}) < \text{CSI}_{\text{thresh}} \quad \lor \quad \text{Chatter}(\mathcal{A} \setminus \{\mathcal{M}\}) > 0$$

### 4.2 Definition: Subsystem Redundancy
A subsystem $\mathcal{M}$ is **REDUNDANT** if and only if removing it produces no statistically significant degradation across all acceptance criteria:
$$\mathcal{M} \in \text{REDUNDANT} \iff |\Delta \text{Acc}| < 2.0\% \quad \land \quad \mathbf{V}(\mathcal{A} \setminus \{\mathcal{M}\}) \text{ satisfies all acceptance criteria}$$

### 4.3 Definition: Subsystem Harm
A subsystem $\mathcal{M}$ is **HARMFUL** if and only if removing it improves performance ($\Delta \text{Acc} \ge +2.0\%$ or $\Delta \text{CSI} \ge +0.05$) or reduces instability without violating any acceptance criteria:
$$\mathcal{M} \in \text{HARMFUL} \iff \text{Acc}(\mathcal{A} \setminus \{\mathcal{M}\}) > \text{Acc}(\mathcal{A}) \quad \lor \quad \text{CSI}(\mathcal{A} \setminus \{\mathcal{M}\}) > \text{CSI}(\mathcal{A}) + 0.05$$

### 4.4 Definition: Subsystem Synergy
Two subsystems $\mathcal{M}_A$ and $\mathcal{M}_B$ are **SYNERGISTIC** if and only if the joint benefit exceeds the sum of individual benefits relative to the common baseline $\mathcal{A}_{\text{base}}$:
$$\text{Performance}(\mathcal{A}_{\text{base}} \cup \{\mathcal{M}_A, \mathcal{M}_B\}) - \text{Performance}(\mathcal{A}_{\text{base}}) > \Delta_{\mathcal{M}_A} + \Delta_{\mathcal{M}_B}$$
or if the combination uniquely preserves a required capability that neither subsystem can preserve in isolation.

---

## 5. Formal Phase 16 Acceptance Vector

The minimal sufficient architecture candidate must satisfy the following predefined thresholds:

| Criterion Index | Metric Description | Variable | Acceptance Threshold | Baseline (P15) |
|---|---|---|---|---|
| **Crit-A** | Autonomous Action Accuracy | $\text{Acc}$ | $\ge 60.0\%$ | $62.67\%$ |
| **Crit-B** | Contextual Shannon Mutual Information | $I(X; S)$ | $\ge 0.900\text{ bit}$ | $0.9968\text{ bit}$ |
| **Crit-C** | Action-Relevant Mutual Information | $I(S; A^*)$ | $\ge 0.200\text{ bit}$ | $0.2701\text{ bit}$ |
| **Crit-D** | Credit Specificity Index | $\text{CSI}$ | $\ge 0.4500$ | $0.4406$ |
| **Crit-E** | Credit Leakage Rate | $\text{Leakage}$ | $\le 0.5500$ | $0.5594$ |
| **Crit-F** | Decision Stability / Chatter | $\text{Switches}$ | $\equiv 0$ switches | $0$ |
| **Crit-G** | Decision Margin | $\text{Margin}$ | $\ge 1.000$ | $4.788$ |
| **Crit-H** | Delayed Action Robustness ($2000\text{ ms}$) | $\text{Acc}_{2000}$ | $\ge 50.0\%$ | $56.0\%$ |
| **Crit-I** | Dynamical Stability (Sync Index) | $r_{\text{local}}$ | $< 0.850$ (no runaway sync) | $0.521$ |
| **Crit-J** | Complexity Reduction | $\Delta \text{Mech}$ | $\ge 1$ mechanism removed | $0$ |

These thresholds are established **PRIOR** to experimental execution and must not be altered post-hoc.
