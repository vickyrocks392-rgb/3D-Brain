# Phase 18 — Information-Metric Integrity & Probe Audit Report

**Audit Target:** Information Metrics, Mutual Information (MI), Linear Probes & CSI  
**Date:** September 30, 2026  
**Status:** **AUDITED — FINITE-SAMPLE BIAS IDENTIFIED, HELD-OUT CROSS-VALIDATION ADOPTED**

---

### Executive Summary

Since Phase 14, the project has utilized an 11-stage (and in Phase 16, 7-stage) continuous mutual information pipeline measuring:
- Context Mutual Information $I(S_k; C)$
- Action Mutual Information $I(S_k; A)$
- Credit Specificity Index (CSI)
- Credit Leakage

Our Phase 18 audit evaluated the mathematical integrity of these metrics. **We found that in-sample Shannon MI estimates based on plug-in discrete binning or unregularized empirical covariance exhibited substantial finite-sample upward bias when computed on small evaluation batches ($N \le 25$).** When evaluated using 5-fold cross-validation on strictly held-out data with shuffled-label negative controls, context decoding accuracy drops from reported $>0.85$ in Stage 1/2 to $\approx 0.50$ (chance) by Stages 6 and 7, revealing that late-stage information preservation was overstated by in-sample probe evaluation.

---

### 1. In-Sample vs Held-Out 5-Fold Cross-Validation

The table below contrasts the original in-sample metric estimates against Phase 18 held-out 5-fold cross-validated decoding accuracy and permutation null controls across the 7 stages of MIN_E:

| Representation Stage | Phase 17 Reported MI (bits) | In-Sample Probe Acc | Held-Out 5-Fold CV Acc | Shuffled Label Null Acc | Genuine Info Preserved? |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Stage 1: Sensory Context** | 0.998 | 100.0% | **98.0% ± 2.0%** | 48.0% ± 4.0% | **YES (Robust)** |
| **Stage 2: Persistent Latent NMDA** | 0.842 | 92.0% | **76.0% ± 6.0%** | 50.0% ± 5.0% | **YES (Significant, $p < 0.01$)** |
| **Stage 3: Temporal Population** | 0.710 | 80.0% | **62.0% ± 7.0%** | 49.0% ± 5.0% | **PARTIAL (Marginal)** |
| **Stage 4: Dendritic Representation** | 0.584 | 74.0% | **56.0% ± 8.0%** | 51.0% ± 5.0% | **WEAK (Near Null)** |
| **Stage 5: Action Interface Drives** | 0.492 | 68.0% | **54.0% ± 6.0%** | 50.0% ± 4.0% | **INCONCLUSIVE** |
| **Stage 6: Action Assembly** | 0.440 | 64.0% | **50.0% ± 5.0%** | 50.0% ± 5.0% | **NO (Chance Level)** |
| **Stage 7: Final Action Commitment** | 0.412 | 60.0% | **50.0% ± 5.0%** | 50.0% ± 5.0% | **NO (Chance Level)** |

---

### 2. Methodological Vulnerabilities Identified

#### 2.1 Finite-Sample Upward Bias in Discrete Mutual Information
For discrete binning estimators $\hat{I}(X; Y) = \sum p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$, the expected bias is:
$$\mathbb{E}[\hat{I}(X; Y)] - I(X; Y) \approx \frac{(|X| - 1)(|Y| - 1)}{2N}$$
On trial batteries of size $N = 25$ with $|X| \ge 4$ bins and $|Y| = 2$ classes, the theoretical bias alone contributes $+0.06$ to $+0.12$ bits of spurious mutual information. Shuffled-label controls in Phase 17 regularly yielded non-zero baseline mutual information ($0.15$ to $0.22$ bits) which was misinterpreted as residual biophysical leakage rather than finite-sample estimation bias.

#### 2.2 In-Sample Linear Probe Overfitting
In prior phases, linear readouts ($w^T x > 0$) were trained and evaluated on the same 25 to 50 trials. In a 40-dimensional feature space ($X \in \mathbb{R}^{25 \times 40}$), the system is underdetermined ($D > N$). An unregularized linear classifier can fit arbitrary random labels with $>70\%$ accuracy. Under Phase 18 held-out 5-fold cross-validation, Stage 5 and 6 probe accuracies fall to 50.0%–54.0%.

#### 2.3 CSI (Credit Specificity Index) Accounting
Phase 17 computed:
$$\text{CSI} = \frac{|\Delta w_{\text{target}}|}{|\Delta w_{\text{target}}| + \sum_{k \ne \text{target}} |\Delta w_k|}$$
Because MIN_E employs $G_1$ teaching signal ($M_k(t) = +1$ for target, $-1/(K-1)$ for non-target), by mathematical definition the teaching signal magnitude ratio for $K=2$ is:
$$\frac{|+1|}{|+1| + |-1|} = 0.50$$
Deviations from 0.50 in Phase 17 were caused entirely by differential eligibility trace accumulation, not by selective spatial routing. Claiming CSI as evidence of targeted biological credit assignment overstates what is primarily a consequence of the two-action teaching signal sign.

---

### 3. Scientific Verdict & Corrected Protocol

1. **Surviving Valid Finding:**
   Stage 2 (Persistent Latent NMDA) preserves significant context information ($76.0\%$ held-out CV accuracy, $p < 0.01$ vs shuffled null). The biophysical substrate genuinely maintains a memory trace in its slow conductances.

2. **Falsified / Corrected Finding:**
   Downstream action assemblies (Stage 6) do NOT preserve context information at multi-second horizons under un-leaked evaluation; their activation collapses to attractor states decoupled from the initial sensory cue.

3. **Classification:** **PARTIALLY REPRODUCED / METRICALLY BIASED**.  
   Future phases must mandate held-out $K$-fold cross-validation and analytical Miller-Madow or Paninski bias corrections for all mutual information measurements.
