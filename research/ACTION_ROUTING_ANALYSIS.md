# Selective Action Routing & Causal Lesion Analysis (Phase 11)

**Substrate Component:** `ActionSelectionCircuit`  
**Circuit Function:** Feedforward Projection from Context Attractor Pools to Downstream Action Accumulators with Lateral Cross-Inhibition  
**Downstream Units:** Action A ($y_0$) and Action B ($y_1$) Accumulators  

---

## 1. Action Routing Design

To test Hypothesis H3 (Selective Action Routing), Phase 11 implemented a dedicated downstream decision-making circuit:

$$\mathbf{y}(t) = (1 - \alpha_y) \mathbf{y}(t-1) + \alpha_y \left[ \mathbf{W}_{\text{act}} \mathbf{s}(t) + \mathbf{W}_{\text{cross}} \mathbf{y}(t-1) \right]$$

- **Feedforward Routing:** Units in Attractor Pool A project excitatorily to Action 0 ($W_{\text{act}}[0, \text{Pool}_A] > 0$), while units in Attractor Pool B project to Action 1 ($W_{\text{act}}[1, \text{Pool}_B] > 0$).
- **Reciprocal Competition:** Mutual lateral cross-inhibition ($W_{\text{cross}} = -0.50$) forces a winner-take-all choice between action alternatives.
- **Decision Rule:** $\text{Action} = \arg\max_k y_k(T_{\text{choice}})$.

---

## 2. Information Transmission Bottleneck

We measured the mutual information between the presented contextual cue ($C \in \{0, 1\}$) and the continuous action accumulator states ($\mathbf{y} \in \mathbb{R}^2$):

$$\text{MI}(C; \mathbf{y}) = 0.012\text{ bits}$$
$$\text{Action Selectivity} = 0.041\text{ a.u.}$$
$$\text{Winner Margin} = 0.038\text{ a.u.}$$

Despite high context retention within the attractor pools ($D_{\text{sep}} / R = 0.963$), the **information transfer across the attractor-to-action boundary is almost zero**.

### Why did routing fail?
1. **Asynchronous Spike Timing:** Downstream accumulators integrate continuous somatic voltage rather than synchronized burst coincidences.
2. **Untrained Feedforward Synapses:** Three-factor targeted feedback modulated recurrent synapses within the attractor network, but did not strongly adapt the feedforward projections from attractor $\to$ action accumulators because reward was delivered at the final step, when attractor activity had already ceased or drifted.

---

## 3. Causal Pathway Lesioning Experiment

To rigorously evaluate causal dependency, we performed targeted in silico lesions:

| Experimental Condition | Description | Accuracy (%) | Change from Intact |
|---|---|---|---|
| **Intact Network** | Fully intact routing pathways | 33.3% | 0.0% (Baseline) |
| **Relevant Pathway Lesion** | Projection from current context pool set to zero | **70.0%** | **+36.7% (Inversion!)** |
| **Irrelevant Pathway Lesion** | Matched projection from opposing pool set to zero | **0.0%** | **-33.3% (Collapse)** |

### Causal Specificity Test: FAILED
In a properly wired feedforward circuit, lesioning the relevant pathway must degrade performance ($\Delta \text{Acc} < 0$), while lesioning the irrelevant pathway should have minimal effect ($\Delta \text{Acc} \approx 0$).
Here, the exact reverse occurred!

### Mechanistic Explanation: Reciprocal Disinhibition Inversion
1. In the intact network, the two action accumulators receive weak, noisy inputs and suppress each other via reciprocal inhibition ($W_{\text{cross}} = -0.5$).
2. When the "relevant" pathway is severed, the active attractor pool can no longer drive Action 0.
3. However, Action 0's silence completely **removes lateral inhibition** from Action 1.
4. Action 1 is released from tonic inhibition, firing strongly and winning the trial.
5. In testing sessions where target action distributions are structured, this disinhibition release artifact accidentally matches the target, producing a false surge in accuracy to $70.0\%$.

**Takeaway for Neural Architecture:** Competitive winner-take-all circuits with lateral inhibition cannot be debugged with simple feedforward credit assignment; inhibitory disinhibition loops invert the causal relationship between synaptic strength and behavioral output.
