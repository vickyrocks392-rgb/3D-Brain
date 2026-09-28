# Phase 5 Literature Review: Predictive Processing, Self-Supervised World Models, and Structural Plasticity

**Project**: 3D-Brain  
**Phase**: Phase 5 — Self-Supervised Embodied World Model & Dynamic Brain Growth  
**Date**: September 2026  
**Status**: COMPLETE  
**Author**: 3D-Brain Research Group  

---

## 1. Executive Summary

This review synthesizes foundational literature across four interconnected domains governing Phase 5:
1. **Predictive Processing and Hierarchical Predictive Coding**: The computational hypothesis that brains function as self-supervised prediction machines that minimize prediction errors across temporal hierarchies.
2. **Self-Supervised World Models and Sensorimotor Contingencies**: How biological and artificial agents construct internal forward dynamics models through physical interaction.
3. **Local Predictive Plasticity and Neuromodulation**: Biologically plausible mechanisms that approximate error backpropagation through local three-factor rules and predictive feedback.
4. **Dynamic Structural Plasticity**: Synaptogenesis, activity-dependent pruning, and homeostatic rewiring in neural circuits.

Crucially, this survey serves as a theoretical and hypothesis-generating foundation, **not as proof of empirical efficacy in our specific discrete recurrent substrate**.

---

## 2. Predictive Processing & Predictive Coding

### 2.1 Classical Formulations
Predictive coding posits that perception is not passive feature extraction, but active inference: higher cortical areas generate top-down predictions of lower-level sensory states, while lower areas transmit only the residual prediction errors upstream (Rao & Ballard, 1999; Friston, 2005, 2010; Clark, 2013).

Mathematically, let $s_t$ denote the sensory state at time $t$, and let $h_t$ represent the internal latent state of the recurrent substrate. A forward predictive model parameterizes:
$$\hat{s}_{t+1} = \mathcal{P}(h_t, a_t)$$
where $a_t$ is the agent's motor command or action. The prediction error is defined as:
$$e_t = s_{t+1} - \hat{s}_{t+1}$$
In predictive coding networks, state updates follow gradient descent on the prediction error energy $\mathcal{E}_t = \frac{1}{2} \|e_t\|_{\Sigma^{-1}}^2$.

### 2.2 Temporal Predictive Hierarchies
Timescales of prediction vary across cortical hierarchies: primary sensory cortices predict rapid, fine-grained transitions, while associative areas predict slowly fluctuating environmental states (Kiebel et al., 2008). In our discrete leaky recurrent architecture (`DISC-LRNN`), this is directly mapped to the distribution of learnable leak rates $\alpha_i = \sigma(\rho_i)$, enabling heterogeneous internal retention timescales.

---

## 3. Sensorimotor World Models & Active Inference

### 3.1 Sensorimotor Contingencies
O'Regan & Noë (2001) proposed that spatial perception arises from the mastery of *sensorimotor contingencies*—the lawful relations between motor commands and sensory changes. Passive observation yields correlational statistics, but active control allows the agent to systematically query the causal structure of its environment:
$$\Delta s = f(s_t, a_t) \quad \text{vs.} \quad P(s_{t+1} \mid s_t)$$

### 3.2 Artificial World Models
In machine learning, world models (Ha & Schmidhuber, 2018; Hafner et al., 2020 [Dreamer]; LeCun, 2022 [JEPA]) demonstrate that compact recurrent latent state representations $h_t$ can anticipate environmental transitions, evaluate counterfactual actions, and support credit assignment without task-specific external rewards.

In Phase 5, we test whether a small spatial recurrent network can learn action-conditioned forward predictions in a closed sensorimotor loop without relying on large-scale transformers or backpropagation through time.

---

## 4. Local Predictive Learning & Neuromodulation

### 4.1 Biological Credit Assignment
Global error backpropagation (BPTT) requires non-local weight transports, synchronous backward phases, and exact transposed matrices, which biological neural tissue lacks (Lillicrap et al., 2020; Whittington & Bogacz, 2017).

Alternative biologically plausible mechanisms include:
* **Feedback Alignment** (Lillicrap et al., 2016): Random fixed backward weights can convey error gradients.
* **Predictive Coding Learning Rules** (Millidge et al., 2021): Updating synaptic weights strictly using pre-synaptic activations and local post-synaptic prediction errors:
  $$\Delta w_{ij} \propto a_i e_j$$
* **Three-Factor Neuromodulated Plasticity** (Frémaux & Gerstner, 2016; Kuśmierz et al., 2017): Modulating local Hebbian eligibility traces with global scalar signals:
  $$\Delta w_{ij}(t) = M(t) \cdot e_{ij}(t)$$
  where $M(t) = g(\|e_t\|)$ represents a global neuromodulatory broadcast of overall prediction error (e.g. noradrenergic or cholinergic novelty signal).

---

## 5. Dynamic Structural Plasticity

### 5.1 Biological Synaptogenesis and Pruning
In the mammalian brain, synaptic connectivity is not fixed; dendritic spines constantly form, stabilize, or retract based on experience and homeostatic pressure (Holtmaat & Svoboda, 2009; Butz et al., 2009). During developmental critical periods and adult learning:
1. **Activity-Dependent Synaptogenesis**: New synapses preferentially sprout between neurons exhibiting correlated activity within physical proximity.
2. **Competitive Pruning**: Synapses whose transmission fails to drive post-synaptic firing, or whose weights decay below a threshold, are enzymatically eliminated.
3. **Homeostatic Target Degree**: Neurons maintain an equilibrium firing rate and synaptic budget (Gallinaro & Rotter, 2018).

### 5.2 Algorithmic Structural Plasticity in Artificial Networks
Previous artificial models of structural plasticity (Mocanu et al., 2018 [SET]; Bellec et al., 2018 [Deep R]; Kappel et al., 2015) demonstrate that dynamic rewiring under strict connection budgets:
* Promotes small-world and scale-free topology emergence.
* Prevents catastrophic forgetting by reallocating dormant synaptic capacity to novel tasks.
* Alleviates representational bottlenecks without scaling parameter counts.

---

## 6. Hypotheses for Phase 5 Investigation

Based on the surveyed literature, we formulate the following concrete, testable hypotheses for Experiment 005:

1. **Hypothesis 1 (Self-Supervised Prediction)**: A recurrent spatial substrate trained via prediction error minimization will maintain lower next-state prediction error than a frozen baseline across stationary and non-stationary trajectories.
2. **Hypothesis 2 (Embodiment / Active vs. Passive)**: An active agent capable of selecting actions will learn a more accurate predictive model of action-contingent environmental transitions than a passive observer exposed to recorded trajectories.
3. **Hypothesis 3 (Structural Rewiring vs. Static Graph)**: Dynamic synaptic rewiring (formation + pruning under a fixed edge budget $E$) will accelerate recovery following environmental switches (Environment A $\to$ Environment B $\to$ Environment A) compared to static topology networks.
4. **Hypothesis 4 (Functional Modularity)**: In contrast to the negative result of Phase 4 (where static structural modularity failed to induce functional modularity), self-supervised predictive training in an embodied environment with dynamic rewiring may foster non-zero functional modularity ($Q_{\text{func}} > 0$).

---

## 7. References

1. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79-87.
2. Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B: Biological Sciences*, 360(1456), 815-836.
3. Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36(3), 181-204.
4. O'Regan, J. K., & Noë, A. (2001). A sensorimotor account of vision and visual consciousness. *Behavioral and Brain Sciences*, 24(5), 939-973.
5. Ha, D., & Schmidhuber, J. (2018). Recurrent world models facilitate policy evolution. *Advances in Neural Information Processing Systems*, 31.
6. Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to control: Learning behaviors by latent imagination. *ICLR 2020*.
7. Whittington, J. C., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local hebbian synaptic plasticity. *Neural Computation*, 29(5), 1229-1262.
8. Millidge, B., Tschantz, A., & Buckley, C. L. (2021). Predictive coding: a theoretical and experimental review. *arXiv:2107.12979*.
9. Frémaux, N., & Gerstner, W. (2016). Neuromodulated Hebbian learning: Neuromodulation, reward prediction, and credit assignment. *Frontiers in Neural Circuits*, 9, 85.
10. Holtmaat, A., & Svoboda, K. (2009). Experience-dependent structural synaptic plasticity in the mammalian brain. *Nature Reviews Neuroscience*, 10(9), 647-658.
11. Butz, M., Wörgötter, F., & van Ooyen, A. (2009). Activity-dependent structural plasticity. *Brain Research Reviews*, 60(2), 287-305.
12. Gallinaro, J. V., & Rotter, S. (2018). Integrating networks with dynamic structural plasticity. *PLOS Computational Biology*, 14(1), e1005939.
13. Bellec, G., Kappel, D., Maass, W., & Legenstein, R. (2018). Deep rewiring: Training very sparse neural networks. *ICLR 2018*.
