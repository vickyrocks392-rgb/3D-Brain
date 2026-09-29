# Experiment 013 Specification: Biophysical Spike Encoding, Action Persistence & Information Preservation

## 1. Central Scientific Question
> **Can biophysically plausible spike encoding, adaptive thresholds/gain, population coding, and action-persistence mechanisms convert the preserved context representation into stable, causally correct, self-sustaining action spike patterns?**

Phase 12 demonstrated that continuous subthreshold states (somatic, dendritic, NMDA) preserve 100% linear decoding accuracy and 1.0 bit mutual information across extended delays. However, converting these analog potentials into sparse, discrete, autonomous spike trains created an information bottleneck: binary point-process representations suffered from high variance, rapid action chatter, and premature switching.

Experiment 013 systematically tests whether biophysical spike-generation interfaces can close this subthreshold-to-spike gap without resorting to continuous artificial readouts or non-biological backpropagation.

---

## 2. Experimental Conditions Matrix

### Primary Spike-Encoding Conditions (S0–S9)
* `COND_S0_PHASE12_BASELINE`: Analog accumulator / simple Poisson rate generation.
* `COND_S1_ADAPTIVE_THRESHOLD`: Axon initial segment dynamic threshold $\theta_i(t) = \theta_0 + \Delta\theta_i(t)$.
* `COND_S2_ADAPTIVE_GAIN`: Activity-dependent membrane and synaptic gain normalization $g_i(t)$.
* `COND_S3_THRESHOLD_PLUS_GAIN`: Combined threshold adaptation and divisive gain control.
* `COND_S4_RATE_POPULATION`: Heterogeneous population rate coding ($M=16$ neurons/action pool).
* `COND_S5_TEMPORAL_CODING`: Latency-to-first-spike and relative phase spike timing.
* `COND_S6_POPULATION_TEMPORAL`: Combined population activity and spatio-temporal burst coincidence.
* `COND_S7_WTA_COMPETITION`: Dale-constrained E/I lateral competition across action pools.
* `COND_S8_PERSISTENT_ATTRACTOR`: Recurrent self-excitation attractor latching with hysteresis lockout.
* `COND_S9_FULL_BIOPHYSICAL`: Integrated biophysical interface combining S1–S8.

### Delay Retention Sweeps
Evaluates contextual choice retention across delays:
$$\Delta t \in [0, 50, 100, 250, 500, 1000, 2000]\text{ ms}$$
Under deterministic seeds: `42, 43, 44`.

### Causal Interventions Suite (C1–C12)
* `C1_CONTEXT_SCRAMBLE`: Permuting context feature dimensions.
* `C2_CONTEXT_SWAP`: Feeding Context A when Context B is expected (tests causal reversal).
* `C3_READOUT_LESION`: Ablating 25%, 50%, and 100% of context-to-action synapses.
* `C4_IRRELEVANT_CONTEXT`: Injecting uncorrelated distractor noise into the interface.
* `C5_LABEL_PERMUTATION`: Permuting reward contingency mapping.
* `C6_FREEZE_PLASTICITY`: Freezing synaptic plasticity post-convergence.
* `C7_SPIKE_TIMING_SHUFFLE`: Shuffling spike times while preserving total spike counts.
* `C8_SPIKE_COUNT_SHUFFLE`: Shuffling spike counts across action pools while preserving timing.
* `C9_THRESHOLD_RANDOMIZATION`: Randomizing adaptive thresholds to disrupt refractory dynamics.
* `C10_GAIN_RANDOMIZATION`: Randomizing gains to disrupt dynamic range normalization.
* `C11_COMPETITION_DISABLED`: Disabling lateral cross-inhibition ($w_{\text{inh}} = 0$).
* `C12_PERSISTENCE_DISABLED`: Disabling recurrent self-excitation and lockout ($w_{\text{exc}} = 0, \Delta = 0$).

### Coding Scheme Direct Comparison
Directly compares:
1. Rate Coding
2. Temporal Coding
3. Population Coding
4. Population + Temporal Coding

### Action Competition Ablations
* Lateral Inhibition: $w_{\text{inh}} \in [0.0, 0.20, 0.55, 1.00]$
* Recurrent Self-Excitation: $w_{\text{exc}} \in [0.0, 0.20, 0.40, 0.80]$
* Symmetric vs. Asymmetric Competition

### Homeostatic Regulation (H0–H4)
* `H0_NONE`: Baseline without homeostatic regulation.
* `H1_FIRING_RATE`: Target rate homeostasis on baseline threshold.
* `H2_THRESHOLD`: Intrinsic threshold adaptation.
* `H3_GAIN`: Divisive gain scaling.
* `H4_COMBINED`: Dual threshold and gain homeostasis.

### Multi-Stage Information Audit
Instruments the full biophysical transmission chain:
$$\text{Context} \to V_{\text{dend}} \to g_{\text{nmda}} \to V_{\text{soma}} \to \theta_{\text{eff}} \to \text{Spikes} \to \text{Action Rates} \to \text{Persistent Action}$$
Measures linear probe accuracy, nonlinear probe accuracy, mutual information (bits), and information loss at each transition.

### Computational Scaling Ladder
Evaluates computational throughput and parameter scaling at $N \in [30, 50, 100, 300, 1000, 10000]$.

---

## 3. Success Criteria & Decision Gates
1. **Information Preservation:** Spike encoding must preserve significantly more contextual mutual information ($> 0.60\text{ bits}$) than Phase 12 binary point-process baseline ($0.00\text{ bits}$).
2. **Causal Control:** C2 Context Swap must invert action selection ($> 80\%$), proving causal dependency on upstream context.
3. **Action Persistence:** Persistence controller must achieve stability $> 0.80$, chatter $< 0.10$, and low unwanted switching under delays up to $2000\text{ ms}$.
4. **Zero Upstream Leakage:** Strict $0.0\%$ credit leakage into the frozen recurrent substrate.
