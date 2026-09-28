# Experiment 002 Specification: Leaky Dynamics & Gate Initialization Ablation

## 1. Scientific Objective

Phase 1 established that `CAND-LI` achieved 100% test accuracy on Delayed Match-to-Sample across $T \in [10, 50]$ with distractor noise, whereas `BASE-GRU` collapsed at $T \ge 25$ and `BASE-RNN` showed high seed instability.

However, the **Phase 1.1 Scientific Audit** identified two major confounding factors:
1. **The Leaky Equivalence Confound**: `CAND-LI` integrated via forward Euler ($\Delta t = 1.0$) is mathematically identical to a discrete-time Leaky RNN with state update $h_t = (1 - \alpha) h_{t-1} + \alpha \tanh(\dots)$. Was the advantage caused by "continuous-time" principles, or simply by the additive autoregressive persistence $(1 - \alpha) h_{t-1}$ acting as a low-pass filter?
2. **The GRU Gate Initialization Confound**: In `BASE-GRU`, the update gate bias was initialized to $b_z = +1.0$, which sets initial update probability $z_t \approx 0.73$. In Cho et al. GRUs, $h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t$. This caused the model to overwrite ~73% of its hidden state with sensory noise at every step. Does a memory-biased GRU ($b_z = -2.0 \implies z_t \approx 0.12$, retaining 88% of memory) solve the task?

**Experiment 002 isolates these causal mechanisms.**

---

## 2. Model Ablation Set

All models tested at strictly matched parameter budgets ($N_{\text{params}} \approx 300$ and $N_{\text{params}} \approx 1000$):

| Model ID | Architecture | State Update Rule | Gate / Leak Initialization | Causal Question Tested |
| :--- | :--- | :--- | :--- | :--- |
| **M1: BASE-RNN** | Standard Elman RNN | $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$ | No leak term | Control baseline (no persistence) |
| **M2: DISC-LRNN** | Discrete-time Leaky RNN | $h_t = (1 - \alpha) h_{t-1} + \alpha \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$ | Learnable $\alpha = \sigma(\rho)$, $\rho_0 = -1.0$ | Tests if discrete leak equals continuous LI |
| **M3: CAND-LI** | Continuous Leaky-Integrator | $\tau \frac{dh}{dt} = -h + \tanh(W_{xh} x_t + W_{hh} a_{t-1} + b)$ | Learnable $\tau = 1/\sigma(\rho)$, $\rho_0 = -1.0$ | Tests ODE formulation with activation recurrence $a_{t-1}$ |
| **M4: GRU-Default** | Standard GRU | $h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t$ | $b_z = +1.0$ (replacement bias) | Replicates Phase 1 collapse |
| **M5: GRU-MemBias** | Memory-Biased GRU | $h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t$ | $b_z = -2.0$ (retention bias) | Tests if gate bias was the sole cause of GRU failure |

---

## 3. Benchmark Task & Delays

Identical to Experiment 001 with an extended delay ladder:
* **Delays Evaluated**: $T_{\text{delay}} \in \{10, 25, 50, 100\}$ steps
* **Distractor Noise**: $\sigma = 0.1$ Gaussian noise injected during delay
* **Seeds**: 5 seeds ($42, 137, 256, 512, 1024$)
* **Evaluation Trials**: $M = 500$ held-out test trials

---

## 4. Quantitative Hypotheses & Falsification Criteria

* **Hypothesis 2A (Leak Sufficiency)**: `DISC-LRNN` matches `CAND-LI` within $\pm 2\%$ accuracy across all delays $T \le 100$.
  * *If True*: The advantage is fully explained by autoregressive leaky integration; the "continuous-time" formulation provides no distinct computational advantage under fixed-step simulation.
* **Hypothesis 2B (GRU Initialization Artifact)**: `GRU-MemBias` achieves $>95\%$ accuracy on $T=25$ and $T=50$, outperforming `GRU-Default`.
  * *If True*: Phase 1 GRU failure was an initialization artifact rather than an architectural limitation.

---

## 5. Execution Status

**Specification only.** Implementation and execution deferred until Phase 1.1 audit is formally accepted.
