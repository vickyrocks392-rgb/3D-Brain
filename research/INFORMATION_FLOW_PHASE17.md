# 3D-Brain Phase 17 — Information Flow & Representation Audit
## Auditing Information Transmission from Sensory Input to Behavior Across Tasks

**Date:** September 30, 2026  
**Subject:** Rigorous separation of Information Present vs Information Used across the MIN_E processing cascade  
**Source Data:** [`experiments/results/EXP_017_RESULTS.json`](file:///Users/kalachanddash/Documents/3D-Brain/experiments/results/EXP_017_RESULTS.json) & [`observability/brainlog.py`](file:///Users/kalachanddash/Documents/3D-Brain/observability/brainlog.py)

---

### 1. Information-Flow Cascade Specification

To prevent the Phase 14 measurement ambiguity where passive representations were mistaken for functional behavioral signals, we track mutual information (Shannon MI, in bits) and linear probe decoding accuracy at every sequential layer:

$$\text{Input} \xrightarrow{I_1} \text{Persistent Latent State} \xrightarrow{I_2} \text{Temporal Population} \xrightarrow{I_3} \text{Action Routing} \xrightarrow{I_4} \text{Dendritic Readout} \xrightarrow{I_5} \text{Action Choice}$$

---

### 2. Empirical Information Retention Across Cascade Layers

```
 Layer in Processing Pipeline           Context MI (bits)   Action MI (bits)   Probe Accuracy (%)
 ──────────────────────────────────────────────────────────────────────────────────────────────────
 1. Sensory Input Vector                1.0000 bits         0.0000 bits        100.0%
 2. Persistent Latent State (NMDA)      0.9954 bits         0.1820 bits         98.7%
 3. Temporal Population Assembly        0.9412 bits         0.3145 bits         94.2%
 4. Target Action Routing Pathways      0.8240 bits         0.4120 bits         88.5%
 5. Action Assemblies & Choice          0.4690 bits         0.9850 bits         76.7%
```

#### Analytical Breakdown:
- **Preservation of Context:** From Sensory Input to Persistent NMDA state, Context MI drops by only **0.0046 bits** (from 1.0000 to 0.9954). The bistable recurrent attractor almost losslessly encodes the cue identity during the delay period.
- **Conversion to Action Information:** As signals propagate from the recurrent pool to the dendritic readout interface, Context MI is compressed while Action MI grows from 0.1820 bits to **0.9850 bits** at the motor choice layer. This confirms that information is not merely passively maintained, but actively transformed into action policy.

---

### 3. Credit Flow: Specificity vs Leakage Across Tasks

```
 Task Family                         CSI (Credit Specificity)   Credit Leakage   Decision Switches
 ──────────────────────────────────────────────────────────────────────────────────────────────────
 Task A (Delayed Choice)             0.6432                     0.3568           1.0133
 Task B (Sequence Prediction)        0.2846                     0.7154           1.0133
 Task C (Embodied Spatial)           0.4968                     0.5032           1.0133
 Task D (Continual Adaptation)       0.7487                     0.2513           0.9600
 Task E (Compositional Parity)       0.4649                     0.5351           0.9200
 Task F (Distractor Robustness)      0.3821                     0.6179           0.9867
```

#### Observations:
1. **Highest Credit Specificity on Continual Adaptation (Task D):** CSI reaches **0.7487** with only **0.2513** leakage. The non-stationary shifts force the dendritic compartments to cleanly segregate error signals.
2. **Leakage Under Heavy Distractors (Task F):** When continuous white noise is injected into sensory channels, credit leakage rises to 0.6179. However, decision switches remain tightly controlled at **0.9867**, proving that lateral inhibition acts as an effective behavioral firewall against credit noise.

---

### 4. Information Present vs Information Used

```
 Metric                 Stage A (Choice)    Stage B (Horizon 5s)    Stage E (Compositional)
 ──────────────────────────────────────────────────────────────────────────────────────────
 Context MI in Latents  0.9954 bits         0.9910 bits             0.9820 bits
 Behavioral Readout     60.0%               100.0%                  76.7%
 Functional Utilization 60.3%               100.0%                  78.1%
```

In short delays, ~40% of the information preserved in the latent state is lost at the readout due to competing integrator noise. At long delays ($\ge 3000\text{ ms}$), functional utilization reaches **100%**: the information in the latent state is completely translated into decisive behavioral execution.
