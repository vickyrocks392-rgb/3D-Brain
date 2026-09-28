# Phase 7 Design Audit: Spiking Neural Dynamics, STDP & Emergent Synchronization

**Project:** 3D-Brain Research Program  
**Phase:** Phase 7  
**Date:** September 2026  
**Status:** COMPLETE AUDIT  
**Author:** 3D-Brain Research Group  

---

## 1. Executive Summary & Research Question

This design audit defines the architectural, mathematical, experimental, and telemetry foundations for Phase 7: **Spiking Neural Dynamics, STDP & Emergent Synchronization**.

At the conclusion of Phase 6 (authoritative commit `724f109`), the 3D-Brain project achieved:
1. **Persistent Latent Memory (H1):** Sustained linear decodability ($R^2 = 0.459 \pm 0.039$) across distractor delays up to $\Delta t = 500$ steps.
2. **Context Disambiguation (H3):** Linear separation of historical contexts with identical immediate sensory inputs ($\text{FDR} = 104.2$).
3. **Continual Memory Retention (H5):** Zero catastrophic forgetting ($0.0000$) on recurring environmental shifts.
4. **Predictive Scaling:** Linear scalability exceeding $10,000$ steps/sec at $100,000$ neurons and $1,500,000$ synapses.

However, a fundamental scientific phenomenon repeatedly failed across Phases 4, 5, and 6:
> **Functional modularity did not emerge:** $Q_{\text{func}} \approx -0.010$ to $-0.020$ despite strong structural modularity ($Q_{\text{struct}} \approx 0.62 - 0.74$). Furthermore, continuous rate-coded activations failed to form autonomous oscillatory phase-locking limit cycles ($R < 0.20$).

The central scientific question of Phase 7 is:
> **Can spiking neuron dynamics (LIF / Izhikevich substrates with STDP) produce genuine functional modularity ($Q_{\text{func}} > 0.25$) and oscillatory phase-locking limit cycles while preserving persistent latent predictive memory?**

### The Scientific Principle: Rigor Over Predetermination
As required by the scientific protocol, **we do NOT assume that spiking neurons will automatically solve the modularity dilemma**. Negative results will be rigorously measured, documented, and preserved. Every hypothesis will receive one of:
- `SUPPORTED`
- `PARTIALLY SUPPORTED`
- `NOT SUPPORTED`
- `INCONCLUSIVE`
backed by quantitative empirical data across multiple seeds.

---

## 2. Audit of Existing Codebase & Reusability Matrix

### 2.1 Reusable Components from Phases 1–6
The 3D-Brain codebase was designed with strict modularity. The following subsystems are directly reusable or extensible:

| Subsystem | Source File | Status | Phase 7 Reuse Plan |
|---|---|---|---|
| **3D Metric Geometry** | `brain/spatial.py` (`SpatialGeometry`) | Directly Reusable | Generate 3D coordinates (uniform sphere, clustered Gaussian, modular spatial) for spiking populations. |
| **Spatial Connectivity Builder** | `brain/spatial.py` (`SpatialConnectivityBuilder`) | Directly Reusable | Build Random, Local, Modular, and Small-World adjacency matrices, pairwise Euclidean distances, and discrete delays $\tau_{ij} = \max(1, \text{round}(d_{ij}/v))$. |
| **Persistent Latent Memory** | `brain/latent_memory.py` (`PersistentLatentMemory`) | Directly Reusable | Reused as the high-tier slow persistent memory module ($z_t$) coupled with spiking spatial dynamics. |
| **Temporal Benchmark Suite** | `benchmarks/temporal_memory.py` | Directly Reusable | Task A (Delayed Cue Recall), Task B (Sequence Reconstruction), Task C (Context Disambiguation), Task D (Multi-Step Horizon). |
| **Representation Analysis** | `brain/representation_analysis.py` | Reusable with extension | Linear decodability probes, RSA, state separability (FDR), autocorrelation decay, participation ratio. |
| **Graph Modularity Engine** | `brain/spatial_analysis.py` | Directly Reusable | Newman's structural modularity $Q_{\text{struct}}$, participation coefficient $P_i$, path lengths. |
| **Event Logging & Capture** | `observability/events.py`, `observability/capture.py` | Extensible | Upgrade to support discrete spike emission/arrival events, membrane potentials, and STDP updates. |
| **`.brainlog` Trace Format** | `observability/brainlog.py` | Backward Compatible Upgrade | Advance schema to `v2.5` while validating `v1.0` through `v2.4`. |
| **Replay Engine** | `observability/replay.py` | Extensible | Add spike event queries, membrane trace inspection, and phase coherence analysis. |
| **3D Web Visualizer** | `visualization/player.html` | Extensible | Add live traveling spike pulses, membrane potential oscilloscope, spike raster, oscillation monitor, and structural/functional network toggle. |

---

### 2.2 Assumptions Specific to Continuous Rate-Coded Dynamics
Previous phases (Phases 1–6) operated under rate-coded approximations:
1. **Activation State:** Neurons possessed continuous scalar activations $a_i(t) = \tanh(h_i(t)) \in [-1, 1]$.
2. **Continuous Synaptic Transmission:** The transmission queue enqueued continuous magnitudes $a_i(t) \cdot W_{ij}$ arriving at $t + \tau_{ij}$.
3. **Continuous Pearson Functional Modularity:** $Q_{\text{func}}$ was computed from Pearson cross-correlations of continuous activation histories $a_i(t)$, which smeared discrete timing differences.
4. **Instantaneous Velocity Phase:** Kuramoto order parameter $R(t)$ was estimated via $\theta_i(t) = \text{atan2}(da_i/dt, a_i - \bar{a}_i)$, which lacked genuine discrete limit-cycle phase definitions.
5. **Rate-Based Plasticity:** Plasticity rules updated weights based on continuous product terms $h_{\text{pre}} \otimes e_t$ or co-activation correlation.

---

## 3. Mathematical Specifications for Phase 7

### 3.1 Neuron Model A: Leaky Integrate-and-Fire (LIF)
The LIF neuron integrates synaptic input current with an exponential passive leak toward resting potential $V_{\text{rest}}$, firing a discrete spike when reaching threshold $V_{\text{th}}$:

$$\tau_m \frac{dV_i(t)}{dt} = -(V_i(t) - V_{\text{rest}}) + R_m I_i(t)$$

Discretized with simulation timestep $\Delta t$:
$$V_i(t + \Delta t) = V_i(t) + \frac{\Delta t}{\tau_m} (V_{\text{rest}} - V_i(t)) + \frac{\Delta t}{\tau_m} R_m I_i(t)$$

**Spike Condition and Reset:**
If $V_i(t + \Delta t) \ge V_{\text{th}}$ and $t_{\text{ref\_remaining}, i} \le 0$:
1. Emit discrete spike event: $S_i(t) = 1$
2. Reset membrane potential: $V_i(t + \Delta t) \leftarrow V_{\text{reset}}$
3. Enter refractory state: $t_{\text{ref\_remaining}, i} \leftarrow t_{\text{ref}}$

During the refractory period ($t_{\text{ref\_remaining}} > 0$):
$$V_i(t) = V_{\text{reset}}, \quad S_i(t) = 0$$

Canonical Parameters:
- Resting potential: $V_{\text{rest}} = -70.0\text{ mV}$
- Reset potential: $V_{\text{reset}} = -75.0\text{ mV}$
- Threshold: $V_{\text{th}} = -50.0\text{ mV}$
- Membrane time constant: $\tau_m = 20.0\text{ ms}$
- Refractory period: $t_{\text{ref}} = 3.0\text{ ms}$
- Simulation timestep: $\Delta t = 1.0\text{ ms}$

---

### 3.2 Neuron Model B: Izhikevich (IZH)
The two-variable Izhikevich dynamical system (Izhikevich, 2003) reproduces a wide spectrum of biological spiking and bursting behaviors:

$$\frac{dv}{dt} = 0.04 v^2 + 5 v + 140 - u + I$$
$$\frac{du}{dt} = a (b v - u)$$

**Spike Condition and Reset:**
If $v(t) \ge 30.0\text{ mV}$:
1. Emit spike: $S(t) = 1$
2. Reset membrane potential: $v \leftarrow c$
3. Increment recovery variable: $u \leftarrow u + d$

Canonical Parameter Regimes:
- **Regular Spiking (RS):** $a = 0.02, b = 0.20, c = -65.0, d = 8.0$ (standard cortical excitatory pyramidal).
- **Fast Spiking (FS):** $a = 0.10, b = 0.20, c = -65.0, d = 2.0$ (cortical inhibitory interneuron).
- **Intrinsically Bursting (IB):** $a = 0.02, b = 0.20, c = -55.0, d = 4.0$.

---

### 3.3 Spike Propagation Queue & Delayed Synaptic Transmission
When neuron $i$ emits a spike at time $t$, each outgoing synapse $(i \to j)$ with propagation delay $\tau_{ij} = \max(1, \text{round}(d_{ij} / v))$ schedules a post-synaptic current pulse at arrival time:
$$t_{\text{arrive}} = t + \tau_{ij}$$

The discrete propagation ring-buffer accumulates incoming synaptic weights:
$$I_{\text{syn}, j}(t) = \sum_{i \in \text{pre}(j)} W_{ij} \cdot S_i(t - \tau_{ij})$$

Total input current to neuron $j$:
$$I_j(t) = I_{\text{ext}, j}(t) + I_{\text{syn}, j}(t) + I_{\text{latent}, j}(t)$$

---

### 3.4 Spike-Timing-Dependent Plasticity (STDP)
Pair-based STDP modifies synaptic weight $W_{ij}$ based on the temporal disparity $\Delta t = t_{\text{post}} - t_{\text{pre}}$:

$$\Delta W_{ij} = \begin{cases} A_+ \exp\left(-\frac{\Delta t}{\tau_+}\right) & \text{if } \Delta t > 0 \quad (\text{pre before post, LTP}) \\ -A_- \exp\left(\frac{\Delta t}{\tau_-}\right) & \text{if } \Delta t < 0 \quad (\text{post before pre, LTD}) \end{cases}$$

Implemented via exponential online pre- and post-synaptic eligibility traces:
$$\frac{dx_i}{dt} = -\frac{x_i}{\tau_+} + S_i(t)$$
$$\frac{dy_j}{dt} = -\frac{y_j}{\tau_-} + S_j(t)$$

Updates:
- On pre-synaptic spike $S_i(t) = 1$: $\forall j \in \text{post}(i), \quad W_{ij} \leftarrow W_{ij} - A_- y_j(t) \cdot (W_{ij} - W_{\min})$
- On post-synaptic spike $S_j(t) = 1$: $\forall i \in \text{pre}(j), \quad W_{ij} \leftarrow W_{ij} + A_+ x_i(t) \cdot (W_{\max} - W_{ij})$

**Homeostatic Synaptic Normalization:**
To prevent runaway excitation and maintain firing stability:
1. Target rate homeostatic scaling: $\Delta W_{i, \text{homeo}} = \eta_{\text{homeo}} (r_{\text{target}} - r_i)$
2. Synaptic weight sum normalization: $\sum_i W_{ij} \le W_{\text{sum\_max}}$

**Neuromodulated STDP:**
$$\Delta W_{ij}^{\text{mod}} = M(t) \cdot \Delta W_{ij}^{\text{STDP}}$$
where $M(t) = \tanh(\|e_t\|^2)$ or external novelty/reward.

---

### 3.5 Functional Modularity on Spike-Train Dynamics ($Q_{\text{func}}$)
Functional modularity is evaluated **independently** of structural topology:
1. Spike trains are binned with temporal window $\Delta W_{\text{bin}} = 10\text{ ms}$ to produce binned count vectors $C_i[k] = \sum_{t \in \text{bin } k} S_i(t)$.
2. Pairwise spike-count Pearson correlation matrix is computed:
   $$R_{ij} = \frac{\text{Cov}(C_i, C_j)}{\sigma_{C_i} \sigma_{C_j}}$$
3. Functional communities $c_i^{\text{func}}$ are discovered using Newman modularity maximization / Louvain partitioning on the positive correlation graph $A_{ij}^{\text{func}} = \max(0, R_{ij})$.
4. Functional modularity is computed:
   $$Q_{\text{func}} = \frac{1}{2m} \sum_{i,j} \left[ A_{ij}^{\text{func}} - \frac{k_i k_j}{2m} \right] \delta(c_i^{\text{func}}, c_j^{\text{func}})$$
5. This is compared against $Q_{\text{func\_struct}}$ (evaluating spike correlation on structural modules) and $Q_{\text{struct}}$.

---

### 3.6 Oscillatory Dynamics & Phase Locking
1. **Population Rate Spectrum:** $R_{\text{pop}}(t) = \frac{1}{N} \sum_{i=1}^N S_i(t)$. Power spectral density $P(f) = |\mathcal{F}\{R_{\text{pop}}\}(f)|^2$ identifies dominant frequencies $f_{\text{dom}}$ and spectral entropy.
2. **Instantaneous Phase:** Derived via Hilbert transform on filtered population rate or spike convolution: $\theta_i(t) \in [-\pi, \pi]$.
3. **Kuramoto Order Parameter:**
   $$R(t) = \frac{1}{N} \left| \sum_{j=1}^N e^{i \theta_j(t)} \right| \in [0, 1]$$
   $R(t) \to 1$ indicates total phase synchronization; $R(t) \approx 0$ indicates incoherent asynchronous firing.
4. **Phase-Locking Value (PLV):**
   $$\text{PLV}_{ij} = \frac{1}{T} \left| \sum_{t=1}^T e^{i (\theta_i(t) - \theta_j(t))} \right|$$

---

### 3.7 Coupling Spiking Substrate with Persistent Latent Memory
To test whether synchronization coexists with persistent predictive memory, the two-tier architecture is preserved:
- **Spiking Spatial Recurrent Substrate:** Fast, discrete event processing with metric delays and STDP.
- **Continuous Post-Synaptic Filter:** $\tau_s \frac{d a_i}{dt} = -a_i + S_i(t)$ converts discrete spikes into smooth continuous activations $a_t \in \mathbb{R}^N$ for the latent interface.
- **Persistent Latent Memory Module ($z_t$):**
  $$z_t = \lambda_z \odot z_{t-1} + (1 - \lambda_z) \odot \tanh(W_{hz} a_t + W_{zz} z_{t-1} + W_{az} a_{\text{onehot}} + b_z)$$
- **Top-Down Modulation Feedback:**
  $$I_{\text{latent}, j}(t) = [W_{zh} z_t]_j$$
  injects subthreshold somatic current into the spiking neurons.

---

## 4. Architectural Comparison: Phase 6 vs Phase 7

| Dimension | Phase 6 (Rate-Coded Latent) | Phase 7 (Spiking STDP Latent) |
|---|---|---|
| **Neuron Dynamics** | Leaky continuous activation: $h_t = (1-\alpha) h_{t-1} + \alpha \tanh(\dots)$ | Explicit membrane potential integration (LIF/IZH) with threshold and refractory periods |
| **State Events** | Continuous vector updates | Discrete threshold crossing spikes: `SPIKE_EMITTED`, `SPIKE_ARRIVED` |
| **Transmission** | Weighted continuous activation in ring buffer | Discrete spike weight pulses delivered via metric delays |
| **Plasticity** | Local predictive feedback alignment $\Delta W = \eta (B_{\text{FA}} e) \otimes h_{\text{pre}}$ | Pair-based STDP ($\Delta t = t_{\text{post}} - t_{\text{pre}}$) + Homeostasis + Neuromodulation |
| **Modularity ($Q_{\text{func}}$)** | Pearson correlation on continuous $\tanh(h_t)$; $Q \approx -0.010$ | Binned spike-count correlation with independent community detection; Target $Q > 0.25$ |
| **Synchronization** | Derivative velocity phase: $R < 0.20$ | Explicit Kuramoto order parameter $R(t)$ and Pairwise Phase-Locking Value (PLV) |
| **Persistent Memory** | Tested in continuous substrate | Re-tested in spiking substrate to determine synchronization trade-off |
| **Observability** | `.brainlog` v2.4 (latent vectors, multi-step predictions) | `.brainlog` v2.5 (adds discrete spike raster, membrane traces, STDP updates, Kuramoto $R$) |

---

## 5. Audit Approval & Verification Signoff

The design audit is fully documented and mathematically specified. Implementation of Phase 7 proceeds under strict scientific controls.
