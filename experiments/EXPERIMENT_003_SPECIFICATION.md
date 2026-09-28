# 3D-Brain — Experiment 003 Specification
## Continual Adaptation, Online Plasticity, and Catastrophic Forgetting

**Experiment ID**: `EXP003`  
**Phase**: Phase 3  
**Status**: APPROVED SPECIFICATION  

---

## 1. Objective and Research Questions

Experiment 003 investigates whether small recurrent computational substrates can continuously adapt to non-stationary task distributions through online local synaptic plasticity without catastrophic forgetting:

1. **Adaptation Efficiency**: Does local online plasticity allow faster convergence to new task demands than a frozen model?
2. **Catastrophic Forgetting**: Does learning a new task $B$ inevitably destroy previously consolidated representations of task $A$?
3. **Comparative Rule Dynamics**: Which local rule (Plain Hebbian, Oja, Modulated Three-Factor, Homeostatic) provides the optimal trade-off between plasticity (rapid adaptation) and stability (memory preservation)?
4. **Plasticity $\times$ Memory Interaction**: Does online weight adaptation degrade or enhance temporal working memory persistence over delay intervals?

---

## 2. Benchmark Task: Continual Delayed Association (CDA)

The task extends the continuous delayed comparison paradigm from Experiments 001 and 002 into a multi-phase, non-stationary sequential environment.

### 2.1 Environmental Mappings
Each trial presents a cue $s \in \{-1.0, +1.0\}$ for $T_{cue} = 5$ steps, a delay period of $T_{delay}$ steps with zero/distractor noise, and a probe stimulus for $T_{probe} = 5$ steps. The target output during probe is determined by the active environment rule:

* **Environment A (Identity Association)**:
  $$y^*(t) = +1.0 \text{ if } s_{probe} == s_{cue} \text{ else } -1.0$$
* **Environment B (Inverted Association)**:
  $$y^*(t) = -1.0 \text{ if } s_{probe} == s_{cue} \text{ else } +1.0$$
* **Environment C (Attenuated / Shifted Association)**:
  $$y^*(t) = +0.5 \text{ if } s_{probe} == s_{cue} \text{ else } -0.5$$
* **Drift Environment (Continuous Rotation)**:
  $$y^*(t) = \cos(\theta_k) \cdot (+1.0 \text{ if match else } -1.0), \quad \theta_k = k \cdot \Delta \theta$$

---

## 3. Experimental Conditions

The benchmark defines 5 rigorous environmental sequence conditions:

### Condition 1 — Stationary
* **Sequence**: Environment A (100 trials)
* **Purpose**: Measure baseline stability, steady-state accuracy, and confirm that plasticity does not induce spontaneous drift in stationary regimes.

### Condition 2 — Abrupt Distribution Shift
* **Sequence**: Environment A (50 trials) $\to$ Environment B (50 trials)
* **Purpose**: Measure adaptation speed ($T_{adapt}$), transient loss spikes, and post-shift convergence.

### Condition 3 — Recurring Environment
* **Sequence**: Environment A (40 trials) $\to$ Environment B (40 trials) $\to$ Environment A (40 trials)
* **Purpose**: Quantify **Catastrophic Forgetting** of Task A after mastering Task B, and measure Recovery Time ($T_{recover}$).

### Condition 4 — Multiple Sequential Environments
* **Sequence**: Environment A (30 trials) $\to$ Environment B (30 trials) $\to$ Environment C (30 trials) $\to$ Environment A (30 trials) $\to$ Environment B (30 trials)
* **Purpose**: Measure long-term retention, sequential interference, and plasticity decay across multiple distribution changes.

### Condition 5 — Gradual Drift
* **Sequence**: 100 trials where target mapping continuously rotates by $\Delta \theta = \frac{\pi}{50}$ per trial.
* **Purpose**: Measure tracking error and parameter drift in non-abrupt continuous non-stationarity.

---

## 4. Plasticity Mechanisms Under Comparison

All rules are evaluated under identical parameter budgets and random seeds:

1. **`Frozen`**: Baseline control. Recurrent weights $W_{rec}$ remain fixed throughout: $\Delta W_{rec} = 0$.
2. **`Hebbian`**: Pure pairwise correlation:
   $$\Delta W_{ij} = \eta \cdot a_i \cdot a_j$$
   with hard bounds $W_{ij} \in [W_{min}, W_{max}]$.
3. **`Oja`**: Self-stabilizing normalized Hebbian rule:
   $$\Delta W_{ij} = \eta \cdot a_j \cdot (a_i - a_j W_{ij})$$
4. **`Modulated`**: Three-factor prediction-error gated rule:
   $$\Delta W_{ij} = M(t) \cdot \eta \cdot a_j \cdot (a_i - a_j W_{ij}), \quad M(t) = \tanh(|y(t) - \hat{y}(t)|)$$
5. **`Homeostatic`**: Oja rule combined with slow synaptic scaling toward setpoint $a^* = 0.2$:
   $$\Delta W_{ij} = \eta \cdot a_j (a_i - a_j W_{ij}) + \gamma \cdot (a^* - \bar{a}_j) |W_{ij}|$$

---

## 5. Mathematical Metrics

### 5.1 Behavioral Metrics
* **Task Accuracy ($Acc$)**:
  $$Acc = \frac{1}{N_{trials}} \sum_{k=1}^{N_{trials}} \mathbb{I}(\text{sgn}(\hat{y}_k) == \text{sgn}(y^*_k))$$
* **Adaptation Time ($T_{adapt}$)**:
  Number of trials after an environment switch until the model achieves $\ge 85\%$ accuracy over a 5-trial window.
* **Recovery Time ($T_{recover}$)**:
  Number of trials upon re-entering Environment A until accuracy reaches $\ge 85\%$ of its pre-switch level.

### 5.2 Continual Learning Metrics
* **Catastrophic Forgetting ($\mathcal{F}_A$)**:
  $$\mathcal{F}_A = Acc(A)_{\text{before } B} - Acc(A)_{\text{after } B}$$
  $\mathcal{F}_A = 0$ indicates perfect preservation; $\mathcal{F}_A > 0.5$ indicates catastrophic forgetting.
* **Backward Transfer ($BWT$)**:
  $$BWT = Acc(A)_{\text{after } B} - Acc(A)_{\text{initial}}$$
* **Forward Transfer ($FWT$)**:
  $$FWT = Acc(B)_{\text{plastic}} - Acc(B)_{\text{from-scratch}}$$

### 5.3 Synaptic Plasticity Metrics
* **Mean Absolute Weight Change ($\langle |\Delta W| \rangle$)**:
  $$\langle |\Delta W| \rangle = \frac{1}{|E|} \sum_{(i, j) \in E} |W_{ij}(t) - W_{ij}(0)|$$
* **Max Weight Change ($\max |\Delta W|$)**:
  $$\max |\Delta W| = \max_{(i, j)} |W_{ij}(t) - W_{ij}(0)|$$
* **Fraction of Active Plastic Synapses ($F_{plastic}$)**:
  Fraction of connections with $|\Delta W_{ij}(t)| > 10^{-4}$.
* **Synaptic Weight Norm ($\|W\|_F$)**:
  Frobenius norm of the recurrent weight matrix over time.

### 5.4 Stability Metrics
* **Saturation Rate ($R_{sat}$)**: Percentage of hidden units with $|a_j(t)| > 0.95$.
* **Vanishing Rate ($R_{van}$)**: Percentage of hidden units with $|a_j(t)| < 0.05$.
* **Instability Flag ($\mathcal{U}$)**: True if $\|W(t)\|_F > 3 \cdot \|W(0)\|_F$ or if output generates NaNs/Infs.

### 5.5 Plasticity $\times$ Memory Interaction
* **Pre- vs. Post-Adaptation MSR**:
  Measure Mean Signal Retention (MSR) across $T_{delay} \in \{10, 25, 50\}$ before exposure to Environment B vs. after exposure to Environment B:
  $$\Delta MSR = MSR_{post} - MSR_{pre}$$

---

## 6. Experimental Matrix (135 Runs)

$$\text{5 Plasticity Rules} \times \text{5 Environmental Conditions} \times \text{1 Parameter Budget (300)} \times \text{3 Seeds (42, 137, 256)} = 75 \text{ core runs}$$
$$\text{Plus Scaling Ablations: Budgets } \{1000, 3000\} \times \text{Condition 3 (Recurring)} \times \text{5 Rules} \times \text{2 Seeds} = 20 \text{ runs}$$
$$\text{Plus Learning Rate \& Homeostatic Parameter Ablations} = 40 \text{ runs}$$
$$\text{Total Matrix: 135 controlled evaluation runs.}$$

All configurations are seeded deterministically, machine-logged to `experiments/results/exp003_results.json`, and accompanied by representative `.brainlog` v2.1 traces.
