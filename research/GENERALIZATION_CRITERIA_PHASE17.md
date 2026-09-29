# Phase 17: Formal Generalization & Transfer Evaluation Criteria

**Pre-Experimental Criteria Specification for the Minimal Sufficient Brain (`MIN_E`)**

*Date: 2026-09-30*  
*Status: Locked Prior to Experimentation*

---

## 1. Scientific Objective

Phase 16 identified the minimal architecture `MIN_E` consisting of eight computational mechanisms:
`A_TARGET_SPECIFIC_TEACHING`, `B_DENDRITIC_TEACHING`, `C_DENDRITIC_ELIGIBILITY`, `H_SLOW_ACTION_ASSEMBLY`, `I_MUTUAL_INHIBITION`, `J_TEMPORAL_POPULATION`, `K_PERSISTENT_LATENT_NMDA`, `L_TARGET_ACTION_ROUTING`.

This document establishes the pre-registered, quantitative criteria used to evaluate whether `MIN_E` is a genuinely general computational architecture or an artifact of overfitting to the canonical delayed contextual choice task.

---

## 2. Pre-Registered Generalization Criteria (A through K)

### Criterion A: Within-Task Generalization
- **Definition:** Performance on unseen test trials drawn from the same generative distribution as training trials.
- **Success Threshold:**
  - Mean accuracy $\ge 60.0\%$ ($p < 0.05$ vs chance $50.0\%$).
  - Generalization gap (Train Accuracy - Test Accuracy) $\le 10.0\%$.

### Criterion B: Temporal Extrapolation
- **Definition:** Retention of performance when delay intervals $\Delta t$ exceed the maximum training horizon by a factor of 2 to 10 ($\Delta t \in [750\text{ ms}, 5000\text{ ms}]$ when trained on $\le 500\text{ ms}$).
- **Success Threshold:**
  - Accuracy at $\Delta t = 1500\text{ ms} \ge 55.0\%$.
  - Accuracy at $\Delta t = 5000\text{ ms} \ge 50.0\%$.
  - Temporal degradation slope $\frac{d\text{Acc}}{d(\log \Delta t)} \ge -0.15\text{ / decade}$.
  - Latent memory correlation $r(\text{state}_0, \text{state}_t) \ge 0.50$ across the delay.

### Criterion C: Context Extrapolation
- **Definition:** Correct action selection when presented with novel context combinations, reordered context mappings, or unfamiliar cue vectors.
- **Success Threshold:**
  - Accuracy on novel context vectors $\ge 58.0\%$.
  - Context mutual information $I(X; S) \ge 0.70\text{ bits}$ in the recurrent reservoir.

### Criterion D: Cross-Task Transfer
- **Definition:** Zero-shot transfer from a source task to a qualitatively distinct target task (e.g. contextual choice $\to$ sequence prediction or spatial navigation).
- **Success Threshold:**
  - Positive transfer ratio $\frac{\text{Acc}_{\text{transfer}} - \text{Acc}_{\text{random}}}{\text{Acc}_{\text{trained}} - \text{Acc}_{\text{random}}} \ge 0.20$.
  - Zero-shot accuracy above random chance baseline ($p < 0.05$).

### Criterion E: Few-Shot Adaptation
- **Definition:** Rate of performance recovery when learning a new task or rule mapping with $k \in \{1, 5, 10, 25\}$ exemplars.
- **Success Threshold:**
  - $\text{Acc}_{k=5} \ge 55.0\%$.
  - $\text{Acc}_{k=25} \ge 65.0\%$.
  - Adaptation sample efficiency: achieves $>80\%$ of asymptotic performance within $\le 25$ trials.

### Criterion F: Continual Learning & Catastrophic Forgetting
- **Definition:** Retention of Task A performance after sequential learning of Task B and Task C (A $\to$ B $\to$ C $\to$ A).
- **Success Threshold:**
  - Backward Transfer / Retention: Retained Task A accuracy $\ge 50.0\%$ (avoiding complete catastrophic forgetting).
  - Forgetting Index $F = \text{Acc}_{\text{initial}} - \text{Acc}_{\text{retest}} \le 20.0\%$.
  - Forward Transfer: Adaptation to Task C faster than from scratch.

### Criterion G: Distractor Robustness
- **Definition:** Resilience to irrelevant sensory channels, distractor temporal events, and ambient sensory noise ($\sigma = 0.10\text{--}0.50$).
- **Success Threshold:**
  - Performance degradation with distractors $\Delta\text{Acc} \le 12.0\%$.
  - Preserved context Shannon MI $I(X; S) \ge 0.80\text{ bits}$.
  - Preserved CSI $\ge 0.30$.

### Criterion H: Interference Resistance
- **Definition:** Preservation of committed action states in the presence of competing intermediate action triggers or misleading reward signals.
- **Success Threshold:**
  - Action switching chatter $\le 1.50$ switches per trial under interference.
  - Action-relevant MI $I(S; A^*) \ge 0.25\text{ bits}$.

### Criterion I: Representation Preservation
- **Definition:** Information-theoretic transmission of task-relevant signals through all 7 anatomical stages (Input $\to$ Latent $\to$ Temporal $\to$ Dendritic $\to$ Readout $\to$ Assembly $\to$ Action).
- **Success Threshold:**
  - Stage 1–4 Context MI $\ge 0.75\text{ bits}$.
  - Stage 5–7 Action MI $\ge 0.25\text{ bits}$.
  - No information bottleneck collapse ($I > 0$ across all stages).

### Criterion J: Credit Localization
- **Definition:** Specificity of dendritic learning updates directed toward the causally relevant action channel without diffuse cross-talk.
- **Success Threshold:**
  - Credit Specificity Index (CSI) $\ge 0.30$ across all evaluated tasks.
  - Credit Leakage $\le 0.70$.
  - Credit precision $> 0.0$ on structured tasks.

### Criterion K: Dynamical Stability
- **Definition:** Prevention of runaway excitation, epileptic synchronization, or premature attractor collapse across extended horizons ($T > 10{,}000\text{ ms}$).
- **Success Threshold:**
  - Mean firing rate remains in healthy physiological regime: $1.0\text{ Hz} \le \bar{f} \le 15.0\text{ Hz}$.
  - Global synchrony index $r_{\text{global}} \le 0.75$ (asynchronous-irregular / weakly clustered regime preserved).
  - Numerical stability: zero NaN / Inf across all trials and parameter sweeps.

---

## 3. Statistical Protocol

1. **Repetition:** All primary conditions evaluated across at least 3 distinct deterministic seeds ($s \in \{42, 43, 44\}$) with $n \ge 25$ trials per seed (total $N \ge 75$ trials per condition).
2. **Reporting:** Report mean, standard deviation (SD), and standard error of the mean (SEM).
3. **No Retroactive Threshold Modification:** The success and failure classifications are strictly tied to these thresholds.
