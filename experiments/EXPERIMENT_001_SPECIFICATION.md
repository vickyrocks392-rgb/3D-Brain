# Experiment 001 Specification: Delayed Comparison & Temporal Memory Retention

## 1. Scientific Motivation & Cognitive Property

To begin our empirical investigation without premature complexity, **Experiment 001** evaluates the most fundamental building block of cognition: **temporal persistence under distraction**.

A purely feedforward or memoryless network cannot solve tasks where input information at time $t_1$ must be retained across an arbitrary temporal interval $\Delta t$ and compared against an input at time $t_2$.

This experiment tests:
1. **Working Memory Persistence**: Can the dynamical substrate maintain an input value in recurrent state without decaying to zero or exploding?
2. **Distractor Robustness**: Can the memory trace survive intervening distractor noise during the delay interval?
3. **Parameter Parsimony**: What is the minimal number of parameters ($N_{\text{params}} \in [200, 1000, 5000]$) required to maintain $>95\%$ comparison accuracy as the delay interval $T_{\text{delay}}$ scales from $10$ to $200$ time steps?

---

## 2. Formal Task Definition: Continuous Delayed Match-to-Sample

Each trial consists of an input stream $\mathbf{x}(t) \in \mathbb{R}^2$ presented over $T_{\text{total}} = T_{\text{cue}} + T_{\text{delay}} + T_{\text{probe}} + T_{\text{response}}$ steps:

1. **Cue Phase ($t \in [1, T_{\text{cue}}]$, $T_{\text{cue}} = 5$)**:
   * Input $x_1(t) = s_{\text{cue}} \in \{-1.0, +1.0\}$ (binary feature or continuous scalar).
   * Input $x_2(t) = 1.0$ (cue flag indicating presentation).
   * Target output $y(t) = 0.0$ (hold response).
2. **Delay Phase ($t \in [T_{\text{cue}}+1, T_{\text{cue}} + T_{\text{delay}}]$)**:
   * Input $x_1(t) \sim \mathcal{N}(0, \sigma_{\text{distractor}}^2)$ with $\sigma = 0.1$ (zero-mean sensory noise/distractors).
   * Input $x_2(t) = 0.0$ (no cue flag).
   * Target output $y(t) = 0.0$ (hold response).
   * **Variable Delay Intervals**: $T_{\text{delay}} \in \{10, 25, 50, 100, 200\}$ steps.
3. **Probe Phase ($t \in [T_{\text{cue}} + T_{\text{delay}} + 1, T_{\text{cue}} + T_{\text{delay}} + T_{\text{probe}}]$, $T_{\text{probe}} = 5$)**:
   * Input $x_1(t) = s_{\text{probe}} \in \{-1.0, +1.0\}$.
   * Input $x_2(t) = 2.0$ (probe flag indicating decision prompt).
   * Target output:
     $$y^*(t) = \begin{cases} +1.0 & \text{if } s_{\text{probe}} == s_{\text{cue}} \quad (\text{Match}) \\ -1.0 & \text{if } s_{\text{probe}} \ne s_{\text{cue}} \quad (\text{Mismatch}) \end{cases}$$

---

## 3. Substrates Compared

We compare three architectures with strictly matched parameter budgets:

| Model ID | Substrate Name | Core Dynamics Formulation | Parameter Budgets Tested |
| :--- | :--- | :--- | :--- |
| **BASE-RNN** | Standard Elman RNN | $\mathbf{h}_t = \tanh(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b})$ | 300, 1000, 3000 params |
| **BASE-GRU** | Gated Recurrent Unit | $\mathbf{z}_t = \sigma(W_z \mathbf{x}_t + U_z \mathbf{h}_{t-1})$; $\mathbf{h}_t = (1-\mathbf{z}_t)\odot\mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$ | 300, 1000, 3000 params |
| **CAND-LI** | Leaky-Integrator / Liquid Continuous-Time Unit | $\tau \frac{d\mathbf{h}}{dt} = -\mathbf{h}(t) + \tanh(W_{hh}\mathbf{h}(t) + W_{xh}\mathbf{x}(t) + \mathbf{b})$, with learnable $\tau_i \in [\Delta t, 100\Delta t]$ | 300, 1000, 3000 params |

---

## 4. Quantitative Metrics

1. **Decision Accuracy ($Acc$)**:
   $$\text{Accuracy} = \frac{1}{M}\sum_{k=1}^M \mathbb{I}\left[\operatorname{sign}(\hat{y}(T_{\text{probe}})) == y^*(T_{\text{probe}})\right]$$
   Evaluated over $M = 500$ held-out test trials.
2. **Mean Squared Error on Response Phase**:
   $$\text{MSE}_{\text{response}} = \frac{1}{T_{\text{probe}}} \sum_{t \in \text{probe}} (\hat{y}(t) - y^*(t))^2$$
3. **Memory Trace Stability Ratio ($MSR$)**:
   Ratio of state variance during delay vs. baseline noise:
   $$\text{MSR} = \frac{\operatorname{Var}_{t \in \text{delay}}(\mathbf{h}(t) \mid s_{\text{cue}}=+1) - \operatorname{Var}_{t \in \text{delay}}(\mathbf{h}(t) \mid s_{\text{cue}}=-1)}{\|\mathbf{h}_{\text{cue}}\|}$$
   Measures whether distinct attractor states remain well-separated in state space throughout the delay.
4. **Parameter Efficiency Score ($\text{PES}$)**:
   $$\text{PES} = \frac{\text{Accuracy}}{N_{\text{parameters}}} \times 10^3$$

---

## 5. Experimental Protocol & Reproducibility

* **Deterministic Seeds**: Runs evaluated across 5 fixed random seeds: `seeds = [42, 137, 256, 1024, 2048]`.
* **Hardware Target**: Local Apple Silicon (Mac M-series), headless execution time $< 60$ seconds per training run.
* **Loss Function**: Masked Mean Squared Error applied only during probe phase (or full-sequence MSE with zero target during delay).
* **Optimization**: AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), gradient clipping at $\|\mathbf{g}\| \le 1.0$.

---

## 6. Telemetry Output & Artifacts

Each experiment run must automatically generate:
1. `metrics.json`: Final accuracy, loss curves, parameter counts, and timing statistics.
2. `trajectory_sample.json`: Complete time-series trace of inputs $\mathbf{x}(t)$, hidden unit states $\mathbf{h}(t)$, activations $\mathbf{a}(t)$, and output predictions $\hat{y}(t)$ for 3 representative test trials (Match, Mismatch, Distractor test).
3. Telemetry format compliant with `visualization/VISUALIZATION_CONCEPT_AND_TELEMETRY.md` for immediate replay in the 3D visualizer.
