# Phase 6 Literature Review: Persistent Latent Memory, Multi-Timescale Dynamics, and Predictive State Representations

**Project:** 3D-Brain  
**Phase:** Phase 6 — Persistent Latent Memory & Predictive State  
**Date:** September 2026  
**Status:** COMPLETE  
**Author:** 3D-Brain Research Group  

---

## 1. Executive Summary

This literature review synthesizes theoretical and empirical research across neuroscience, cognitive science, and computational learning theory to establish the rigorous scientific foundation for Phase 6. We examine:

1. **Persistent Neural Activity and Attractor Dynamics**: How biological circuits maintain task-relevant representations after sensory cues disappear (Goldman-Rakic, 1995; Wang, 2001).
2. **Multi-Timescale Recurrent Dynamics**: How cortical hierarchies exhibit diverse temporal receptive windows without artificial time-delays (Chaudhuri et al., 2015; Bernacchia et al., 2011).
3. **Predictive State Representations (PSRs) and Multi-Step World Models**: How internal representations grounded in future predictions summarize past experience (Littman & Sutton, 2002; Silver et al., 2017; Hafner et al., 2020).
4. **Temporal Context Disambiguation**: How neural circuits disambiguate identical instantaneous observations that have divergent temporal histories (Wood et al., 2000; Frank et al., 2000).
5. **Continual Learning & Complementary Memory Systems**: How persistent representations avoid catastrophic interference during environmental shifts (McClelland et al., 1995; Fusi et al., 2005).

---

## 2. Persistent Neural Activity & Working Memory

### 2.1 Classical Attractor Models
In seminal electrophysiological studies of primate prefrontal cortex (PFC), Goldman-Rakic (1995) and Fuster (1997) observed sustained elevated spiking activity during delay periods of delayed-response tasks. When a visual cue is presented and subsequently extinguished, prefrontal pyramidal neurons continue firing across delays of several seconds, bridging the temporal gap until action execution.

Theoretical models explain this via **recurrent attractor dynamics** (Amit & Brunel, 1997; Wang, 2001):
$$\tau \frac{dh}{dt} = -h + W_{\text{rec}} \sigma(h) + W_{\text{in}} s + b$$
Strong recurrent excitatory connections generate stable fixed points or line attractors. However, pure recurrent attractor networks are vulnerable to:
1. **Drift and instability**: Small perturbations can cause states to diffuse along line attractors (Burak & Fiete, 2012).
2. **Interference from new inputs**: Strong incoming sensory inputs can easily displace or overwrite the attractor state unless gating or decoupling mechanisms exist.

### 2.2 Synaptic Facilitation and Hidden States
Mongillo, Barak, and Tsodyks (2008) demonstrated that working memory does not require continuous active firing; it can be maintained via short-term synaptic plasticity (calcium-mediated facilitation). The memory resides in a latent synaptic state that consumes minimal metabolic energy and can be reactivated by a non-specific cue.

In Phase 6, we draw upon this computational principle by formalizing an explicit **persistent latent state $z_t$** that is distinct from the immediate somatic activation $h_t$.

---

## 3. Multi-Timescale Recurrent Dynamics

### 3.1 Heterogeneous Cortical Decay
Empirical recordings across cortex reveal a hierarchy of intrinsic timescales: sensory areas (e.g., primary visual cortex V1) process inputs with short autocorrelation decay constants ($\tau \approx 20\text{–}50\text{ ms}$), whereas association areas (e.g., posterior parietal cortex, anterior PFC) exhibit long decay constants ($\tau \approx 200\text{–}1000\text{ ms}$) (Murray et al., 2014; Chaudhuri et al., 2015; Siegle et al., 2021).

In discrete recurrent formulations, this corresponds to heterogeneous per-unit leak rates:
$$h_{i, t} = (1 - \alpha_i) h_{i, t-1} + \alpha_i f(I_{i, t})$$
where $\alpha_i = \sigma(\rho_i) \in (0, 1)$ sets the effective time horizon $\tau_i = 1 / \alpha_i$. Units with $\alpha_i \to 0$ function as slow integrators, while units with $\alpha_i \to 1$ reflect instantaneous inputs.

### 3.2 Timescales in Continuous Latent State
Phase 6 generalizes this to a two-tier architecture:
- Fast tier: Spatial recurrent units $h_t$ interacting through finite-delay metric queues.
- Slow tier: Latent state $z_t$ governed by persistence factors $\lambda_z \in [0.90, 0.999]$, preserving history across hundreds of steps.

---

## 4. Predictive State Representations & Multi-Step World Models

### 4.1 Predictive State Representations (PSRs)
Littman and Sutton (2002) introduced PSRs as a representation of state defined entirely in terms of predictions of future observations conditioned on future actions:
$$P(s_{t+1}, s_{t+2}, \dots, s_{t+k} \mid a_t, a_{t+1}, \dots, a_{t+k-1})$$
Unlike hidden Markov models or arbitrary POMDP belief states, a predictive state representation has grounded physical meaning: two historical trajectories are functionally equivalent if and only if they make identical predictions about all future tests.

### 4.2 Multi-Step Prediction Horizons
In artificial world models (Ha & Schmidhuber, 2018; Hafner et al., 2020 [Dreamer]; Silver et al., 2017 [Predictron]), learning is evaluated not merely on one-step prediction $s_{t+1}$, but on multi-step rollouts:
$$\hat{s}_{t+k} = \mathcal{P}(z_t, a_{t:t+k-1})$$
A purely reactive system that lacks persistent state will suffer exponential error compounding over multi-step rollouts, whereas a system with an informative latent state can predict long-range dynamics accurately.

---

## 5. Temporal Context Disambiguation

### 5.1 The Equifinality Problem in Perception
In POMDP environments, identical sensory observations frequently arise from different underlying world states. For example:
- Route navigation: Traversing intersection $X$ from the North vs. traversing intersection $X$ from the West yields identical immediate visual features, but requires opposite turns to reach the goal.
- Sequence disambiguation: $A \to B \to X \to Y_1$ versus $C \to D \to X \to Y_2$.

Electrophysiological studies in rodents navigating T-mazes show that hippocampal CA1 and CA3 place cells exhibit **splitter cell** behavior (Wood et al., 2000; Frank et al., 2000): firing rates at the common central stem differ significantly depending on whether the animal originated from the left or right arm.

Phase 6 implements this directly in **Task C (Temporal Context Disambiguation)** to test whether the latent state $z_t$ encodes historical trajectory information when sensory observation $s_t = X$ is identical.

---

## 6. Continual Learning & Complementary Systems

### 6.1 Complementary Learning Systems (CLS)
McClelland, McNaughton, and O'Reilly (1995) proposed that the brain avoids catastrophic forgetting through two interacting systems:
1. Fast, episodic memory in the hippocampus that stores distinct experience traces without cross-interference.
2. Slow, structured semantic representations in the neocortex that gradually extract statistical regularities.

When an artificial neural network with a single distributed state learns non-stationary tasks, new associations overwrite past synaptic weights (catastrophic forgetting; French, 1999).

In Phase 6, we test whether a persistent latent state helps or hurts continual learning across task switches ($A \to B \to A$) and environmental transitions. Persistent memory could either protect prior representations (positive backward transfer) or produce stubborn interference (negative transfer).

---

## 7. Mathematical Synthesis for Phase 6

Based on this synthesis, Phase 6 specifies:
1. **Coupled Substrate-Latent Equations**:
   $$h_t = (1 - \alpha) \odot h_{t-1} + \alpha \odot \tanh(W_{xh} s_t + I_{\text{arrived}} + W_{zh} z_{t-1} + b_h)$$
   $$z_t = \lambda_z \odot z_{t-1} + (1 - \lambda_z) \odot \tanh(W_{hz} h_t + W_{zz} z_{t-1} + W_{az} a_t + b_z)$$
2. **Multi-Step Prediction Interface**:
   $$\hat{s}_{t+k} = \text{clip}(W_{\text{pred}}^{(k)} [z_t; a_t] + b_{\text{pred}}^{(k)}, 0, 1)$$
3. **Controlled Falsification Matrix**:
   Ablating persistence ($\lambda_z \to 0$ or resetting $z_t$ per step) must specifically eliminate performance on delayed recall and context disambiguation while leaving 1-step reactive predictions relatively intact.

This provides the exact theoretical and mathematical foundation for our empirical experiments.
