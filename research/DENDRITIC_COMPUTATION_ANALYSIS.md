# Two-Compartment Dendritic Computation Analysis (Phase 10)

## 1. Overview and Mathematical Formulation

In Phase 10, point-neuron approximations were replaced by an explicit two-compartment biophysical model (`TwoCompartmentPopulation`) consisting of:
* A **somatic compartment** responsible for action potential generation, absolute refractory reset, and leak dynamics.
* A **dendritic compartment** responsible for synaptic integration (AMPA, NMDA, GABA), slow state conductance integration, plateau current generation, and backward action potential (bAP) reception.

### Governing Differential Equations

#### Dendritic Compartment
$$C_d \frac{dV_d}{dt} = -g_{L,d}(V_d - V_{\text{rest}}) + I_{\text{syn}}^E + I_{\text{syn}}^I + I_{\text{nmda}} + I_{\text{plateau}} - g_c(V_d - V_s)$$

Where:
* $C_d = 1.0\text{ nF}$, $g_{L,d} = 0.05\text{ \mu S}$, $V_{\text{rest}} = -70.0\text{ mV}$.
* $I_{\text{nmda}} = g_{\text{nmda}} \cdot \text{Mg\_block}(V_d) \cdot (E_{\text{nmda}} - V_d)$ with $E_{\text{nmda}} = 0.0\text{ mV}$.
* $\text{Mg\_block}(V_d) = \frac{1}{1 + 0.28 \exp(-0.062 V_d)}$.
* Inter-compartment coupling conductance: $g_c = 0.7\text{ \mu S}$.

#### Somatic Compartment
$$C_s \frac{dV_s}{dt} = -g_{L,s}(V_s - V_{\text{rest}}) + g_c(V_d - V_s) + I_{\text{ext}}$$

Where:
* $C_s = 1.0\text{ nF}$, $g_{L,s} = 0.1\text{ \mu S}$.
* Somatic spike generation occurs if $V_s \ge V_{\text{thresh}} = -50.0\text{ mV}$ (excitatory) or $-52.0\text{ mV}$ (inhibitory).
* Upon threshold crossing:
  $$S(t) = 1, \quad V_s \leftarrow V_{\text{reset}} = -75.0\text{ mV}$$
  $$V_d \leftarrow V_d + A_{\text{bAP}} \quad (A_{\text{bAP}} = 15.0\text{ mV})$$

---

## 2. Experimental Verification of Biophysical Dynamics

Unit tests in [`tests/test_two_compartment.py`](file:///Users/kalachanddash/Documents/3D-Brain/tests/test_two_compartment.py) verify the five core biophysical features:
1. **Slow NMDA integration and exponential decay:** Verified across $\tau_{\text{NMDA}} \in [20, 250]\text{ ms}$.
2. **Coupling current balance:** Depolarization of the dendrite drives somatic depolarization ($I_c = g_c(V_d - V_s) > 0$).
3. **Dendritic load sink:** Resting dendrites act as a current sink, requiring sufficient somatic drive or dendritic pre-depolarization to trigger spiking.
4. **Nonlinear dendritic plateau events:** Depolarization beyond $\theta_{\text{dend}} = -45.0\text{ mV}$ triggers sustained plateau currents ($g_{\text{nl}} = 0.3$) for $10\text{ ms}$.
5. **Backpropagating action potentials (bAP):** Somatic spikes inject depolarizing currents back into the dendritic compartment.

---

## 3. Computational Impact: Point vs. Two-Compartment

| Metric | Point Neuron (Phase 9) | Two-Compartment (Phase 10) | $\Delta$ Change |
|:---|:---:|:---:|:---:|
| State Variables per Neuron | 1 ($V_m$) | 4 ($V_s, V_d, g_{\text{nmda}}, \text{plateau}$) | $+300\%$ |
| Context Memory Retention ($R^2$ at $100\text{ ms}$) | $0.009$ | $0.645$ | **$+7066\%$** |
| Context Memory Retention ($R^2$ at $1000\text{ ms}$) | $0.008$ | $0.577$ | **$+7112\%$** |
| Task Accuracy on Delayed Credit | $46.2\%$ | $43.8\%$ | $-2.4\%$ ($p > 0.05$) |
| Eligibility Concentration (Gini) | $0.000$ | $0.264$ | $+0.264$ |
| Simulation Latency ($1\text{K}$ neurons) | $0.045\text{ ms}$ | $0.108\text{ ms}$ | $+0.063\text{ ms}$ |

### Key Insight
Dendritic compartmentalization provides an order-of-magnitude increase in continuous latent state retention ($R^2 > 0.57$ across $1000\text{ ms}$) without inducing runaway epileptiform hypersynchrony. However, the conversion of this preserved dendritic trace into correct somatic action selection at the choice step requires targeted readout gating, which pure local feedback alignment does not fully resolve.
