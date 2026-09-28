# 3D-Brain — Experiment 004 Specification
## Spatialized 3D Topology, Propagation Latency, and Parameter Scaling

**Experiment ID**: `EXP004`  
**Phase**: Phase 4  
**Status**: APPROVED SPECIFICATION  
**Author**: 3D-Brain Research Group  
**Date**: September 2026  

---

## 1. Executive Summary & Research Questions

Experiment 004 investigates whether embedding a recurrent neural substrate into an explicit physical 3D metric space produces measurable computational consequences, or whether spatial structure is merely an aesthetic/visualization artifact.

The primary research question is:
> **Does embedding the computational substrate into an explicit physical 3D metric space create measurable computational advantages that cannot be explained simply by parameter count or random connectivity?**

### Secondary Research Questions:
1. **Topological Efficiency**: Do modular and small-world topologies outperform random metric-free baselines on delayed and distributed tasks under strictly matched parameter budgets and average degree?
2. **Propagation Latency**: Does finite signal propagation velocity ($v < \infty$) create functional phase segregation or does it uniformly degrade performance relative to zero-delay controls?
3. **Emergent vs. Imposed Modularity**: Does functional modularity ($Q_{func}$) exceed structural modularity ($Q_{struct}$), and does online neuromodulated plasticity amplify this functional segregation?
4. **Long-Range Shortcuts**: Does adding sparse long-range projections ($\beta = 0.10$) to localized spatial networks bridge distal regions without destroying local modularity?
5. **Scale Dynamics**: At what substrate scale ($1\text{K} \to 100\text{K}$) do spatial metric constraints and propagation delays dominate computation and memory throughput?

---

## 2. Experimental Task Ladder

To ensure experimental validity and prevent task confounding, Experiment 004 uses a three-tier task ladder:

### Task A — Continuous Delayed Match-to-Sample (DMS)
* **Substrate**: Headless recurrent comparison with cue period ($T_{cue} = 5$), delay interval ($T_{delay} = 10$), and probe period ($T_{probe} = 5$).
* **Input**: Stimulus $s \in \{-1.0, +1.0\}$ on channel 0, phase gate on channel 1.
* **Target**: $y^*(t) = +1.0$ if match else $-1.0$.
* **Purpose**: Direct continuity with Phase 1 & 2 baselines.

### Task B — Continual Delayed Association (CDA)
* **Substrate**: Multi-phase sequential environment (Environment A identity $\to$ Environment B inverted $\to$ Environment A recovery).
* **Purpose**: Test interaction of 3D spatial metric layout with Phase 3 online plasticity rules (Frozen vs. Modulated).

### Task C — Spatially Distributed Temporal Task (SDTT)
* **Substrate**: Information enters at two physically distal spatial poles:
  * Channel 0 (Cue 1) enters at Module 0 (Left Hemisphere / $X < -5.0$).
  * Channel 1 (Cue 2) enters at Module 1 (Right Hemisphere / $X > +5.0$) after an inter-cue delay.
  * Channel 2 (Probe trigger) enters globally.
* **Target**: $y^*(t) = +1.0$ if $\text{sign}(\text{Cue 1}) == \text{sign}(\text{Cue 2})$ else $-1.0$.
* **Purpose**: Information *must* physically propagate across the metric diameter of the substrate to be integrated before the motor readout at the anterior pole ($Z = +8.0$).

---

## 3. Independent Variables & Experimental Dimensions

### 3.1 Topology Families (Controlled Parameter Budget)
All models share identical unit count $N$, identical connection count $E$, and identical average degree $\langle k \rangle = E / N$:
1. **Random Baseline (`RANDOM`)**: Directed connections formed independently of metric distance.
2. **Spatial Local (`LOCAL`)**: Distance-dependent connectivity governed by Gaussian spatial kernel:
   $$P(i \to j) \propto \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)$$
3. **Modular Clustering (`MODULAR`)**: $M=4$ distinct spatial modules with high intra-module connection density ($88\%$) and sparse inter-module bridge links ($12\%$).
4. **Small-World (`SMALL_WORLD`)**: Spatially local connectivity ($90\%$) augmented with sparse long-range shortcut projections ($\beta = 10\%$).

### 3.2 Propagation Latency & Speed Conditions
Physical connection distance $d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|_2$. Latency is discretized as:
$$\tau_{ij} = \max\left(1, \left\lfloor \frac{d_{ij}}{v} \right\rceil\right)$$
Conditions:
- **Instantaneous / Near-Zero (`DELAY_ZERO`)**: $v \to \infty \implies \tau_{ij} = 1$ for all edges.
- **High Speed (`SPEED_HIGH`)**: $v = 15.0$ distance units/step.
- **Medium Speed (`SPEED_MED`)**: $v = 5.0$ distance units/step.
- **Low Speed (`SPEED_LOW`)**: $v = 2.0$ distance units/step (producing delays up to $15$ steps).

### 3.3 Plasticity Regimes
- **Frozen**: Static weights after initial pretraining.
- **Modulated**: Online three-factor neuromodulated plasticity operating locally across active structural synapses.

### 3.4 Scale Ladder
- $N \in \{100, 300, 1000, 3000, 10000, 25000, 50000, 100000\}$.
- Evaluated for memory footprint, queue overhead, initialization time, and step throughput.

---

## 4. Controlled Ablations and Null Models

To isolate whether observed computational differences originate from geometry, topology, delays, or plasticity, six mandatory ablations and four null models are executed:

### Ablation Matrix:
* **Ablation A (Distance-Free Connectivity)**: Keep 3D neuron positions, but wire topology with uniform random probability.
* **Ablation B (Delay-Free Spatial)**: Retain spatial distance-dependent connectivity, but set all propagation delays to 1.
* **Ablation C (Position Shuffling)**: Preserve the exact adjacency matrix and delays, but randomly permute spatial coordinates $\mathbf{x}_i$.
* **Ablation D (Topology Shuffling)**: Preserve 3D neuron positions, but rewire adjacency matrix preserving in/out degree distribution.
* **Ablation E (No Long-Range Shortcuts)**: Remove all cross-module and long-range connections from Small-World/Modular networks.
* **Ablation F (Plasticity Disabled)**: Freeze weights in non-stationary CDA task.

### Null Models:
1. **Degree-Preserving Maslov-Sneppen Rewired Graph**: Assesses topological structure beyond degree sequence.
2. **Coordinate Shuffled Null**: Tests whether physical distance has computational significance.
3. **Module Label Permuted Null**: Tests whether functional modularity reflects imposed structural modules.
4. **Randomized Delay Null**: Assigns delays randomly independent of physical Euclidean distance.

---

## 5. Quantitative Measurement Metrics

1. **Task Performance**: Mean Squared Error ($MSE$), Binary Decision Accuracy ($\%$) during probe phase.
2. **Structural Modularity ($Q_{struct}$)**: Newman's modularity coefficient on synaptic adjacency graph.
3. **Functional Modularity ($Q_{func}$)**: Modularity of the activation cross-correlation matrix $C_{ij} = \text{corr}(a_i, a_j)$.
4. **Within / Cross Correlation**: Mean correlation among units in the same module vs. different modules.
5. **Phase Synchronization ($R$)**: Kuramoto phase order parameter $R(t) = \frac{1}{N} |\sum_j \exp(i \theta_j(t))|$ averaged over time.
6. **Characteristic Path Length ($L$) & Clustering ($C$)**: Graph topological efficiency metrics.
7. **Computational Throughput**: Steps per second, simulation latency, propagation queue cost, and memory footprint.

---

## 6. Success & Falsification Criteria

* **Falsification of Spatial Advantage**: If Modular and Small-World networks show no statistically significant advantage over degree-matched Random networks on Task C, or if removing propagation delays improves accuracy on all tasks, we falsify the claim that spatial metric constraints provide an intrinsic computational benefit.
* **Falsification of Emergent Modularity**: If $Q_{func} \le Q_{struct}$ across all learning conditions and cannot be dissociated from structural initialization, we conclude that modularity is strictly imposed, not emergent.
