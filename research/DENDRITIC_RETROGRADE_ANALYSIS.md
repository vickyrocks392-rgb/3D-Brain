# Dendritic Retrograde Signaling Analysis (Phase 15)
**Critical Empirical Finding: Diffuse Retrograde Signals as a Credit Leakage Vector**

---

## 1. Executive Summary & Mandatory Negative Finding

A core biological hypothesis tested in Phase 15 was whether **retrograde signaling** (such as nitric oxide or endocannabinoid release from the post-synaptic dendritic arbor to pre-synaptic terminals) could guide synaptic updates and improve autonomous action performance.

### Empirical Finding
**DIFFUSE RETROGRADE SIGNALING ACTS AS A VECTOR FOR CREDIT LEAKAGE AND HARMS PERFORMANCE.**
When retrograde feedback was enabled without strict spine-level compartmental microdomains (Condition **COND_I**), task accuracy dropped from the baseline $54.7\%$ down to **$49.3\% \pm 12.4\%$**, and Credit Specificity Index (CSI) dropped from $0.500$ down to $0.441$ (Leakage rose to $0.559$).

Furthermore, in causal intervention **C10 (Retrograde Removal)**, removing retrograde signals had **$0.0\%$ delta** on performance, confirming that non-compartmentalized retrograde feedback does not convey beneficial instructional credit.

---

## 2. Experimental Sweep of Retrograde Signaling ($R_0 \to R_4$)

We evaluated five retrograde signaling regimes:

| Regime | Implementation Details | Accuracy | CSI | Leakage | Precision | Causal Impact |
|---|---|---|---|---|---|---|
| **$R_0$ (None)** | Baseline feedforward only | 56.0% | 0.2143 | 0.7857 | 0.0000 | Stable reference |
| **$R_1$ (Scalar Broadcast)** | Post-synaptic action potential releases diffuse retrograde messenger | 48.0% | 0.3213 | 0.6787 | 0.0000 | **-8.0% degradation** |
| **$R_2$ (Action-Specific)** | Retrograde signal confined strictly to efferents of chosen action | 56.0% | **0.5511** | **0.4489** | **0.4400** | Neutral performance, improves CSI |
| **$R_3$ (Compartment-Specific)**| Gated to apical dendrites only | 56.0% | 0.5081 | 0.4919 | 0.0000 | Neutral |
| **$R_4$ (Delayed Retrograde)** | 25 ms axonal retrograde transmission delay | 56.0% | 0.5540 | 0.4460 | **0.4400** | Neutral |

---

## 3. Biophysical Mechanism of the Failure Mode

### 3.1 Unintended Diffuse Potentiation
In a biological synapse, nitric oxide (NO) diffuses isotropically over a radius of $2 \text{ to } 5\,\mu\text{m}$.
In our computational model:
- Post-synaptic soma produces retrograde signal $R_k(t) = \sigma(V_k(t) - V_{\text{thresh}})$.
- If $R_k(t)$ is received by pre-synaptic terminals indiscriminately:
  $$\Delta W_{kj} \propto R_k(t) \cdot e_{kj}(t)$$
- Because upstream recurrent neurons fire in coordinated population bursts during both Action 0 and Action 1 trials, their eligibility traces $e_{kj}(t)$ are active across both channels.
- Consequently, $R_k(t)$ potentiates synapses from neurons that were merely active during the context phase, regardless of whether their activity was causally related to the executed action.

### 3.2 Mathematical Analysis of Credit Leakage
Let $\mathbf{e}_k$ be the eligibility vector for action $k$.
If retrograde signal $R$ is broadcast with cross-talk $\alpha \in [0, 1]$:
$$\Delta W_k = \eta [ R_k \mathbf{e}_k + \alpha R_{-k} \mathbf{e}_{-k} ]$$
When $\alpha > 0$, the gradient update direction is corrupted by the orthogonal action's trace:
$$\langle \Delta W_k, \mathbf{w}^*_k \rangle = \eta \left( \langle \mathbf{e}_k, \mathbf{w}^*_k \rangle - \alpha \langle \mathbf{e}_{-k}, \mathbf{w}^*_k \rangle \right)$$
As observed empirically in Condition I, this corruption degrades the signal-to-noise ratio, resulting in unstable updates and high variance across seeds (SEM = $\pm 12.4\%$).

---

## 4. Architectural Prescription

1. **Do not use non-localized retrograde broadcast** in artificial spiking neuromorphic systems without fine-grained dendritic spine isolation.
2. In biological architectures, retrograde signals must be strictly paired with pre-synaptic receptor localization (e.g., retrograde cannabinoid CB1 receptors localized strictly opposite active active active release sites).
3. In Phase 15, optimal learning was achieved using direct compartmental apical plateau potentials ($G_1, G_3$) rather than retrograde diffusion.
