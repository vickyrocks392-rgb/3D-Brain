# Phase 16 Information Bottleneck & Mutual Information Audit

## 1. Information Theoretic Methodology

Following the methodological correction established in Phase 15 (`research/INFORMATION_MEASUREMENT_AUDIT_PHASE15.md`), Phase 16 evaluated continuous Shannon Mutual Information via non-parametric Gaussian Kernel Density Estimation (KDE) and Kraskov entropy estimators:

$$I(X; Y) = H(X) + H(Y) - H(X, Y)$$

Where:
- **$I(X; \text{Context})$:** Quantifies how much sensory cue information is preserved in internal neural representation $X$.
- **$I(X; \text{Action})$:** Quantifies how much action-relevant choice information is encoded in representation $X$.

Instantaneous Binary Symmetric Channel (BSC) proxies are rejected as primary metrics, as established in Phase 15.

---

## 2. Seven-Stage Information Flow Audit

Every trial recorded continuous trajectory vectors across all seven functional computational processing stages:

1. **Stage 1: Sensory / Context Input** ($d = 40$)
2. **Stage 2: Persistent Latent State** ($d = 40$, NMDA and recurrent reservoir currents)
3. **Stage 3: Temporal Population Representation** ($d = 128$, bursting and spiking community vectors)
4. **Stage 4: Dendritic Representation** ($d = 40$, apical and basal dendritic potentials)
5. **Stage 5: Action Interface Representation** ($d = 2$, pooled readout synaptic currents)
6. **Stage 6: Action Assembly State** ($d = 2$, temporal evidence integrators)
7. **Stage 7: Final Action Choice** ($d = 2$, winner-take-all discrete commitment)

---

## 3. Comparative Information Profile: FULL_P15 vs. MIN_E

| Stage | Processing Level | FULL_P15 Context MI | FULL_P15 Action MI | MIN_E Context MI | MIN_E Action MI | Information Transformation |
|---|---|---|---|---|---|---|
| **Stage 1** | Sensory Context Input | 0.9954 b | 0.0000 b | 0.9954 b | 0.3481 b | Perfect external cue entropy ($1\text{ bit}$) |
| **Stage 2** | Persistent Latent (NMDA) | 0.9954 b | 0.0000 b | 0.9954 b | 0.2692 b | 100% cue preservation across delay epoch |
| **Stage 3** | Temporal Population | 0.9954 b | 0.0000 b | 0.9954 b | **0.4690 b** | High-dimensional orthogonal temporal basis |
| **Stage 4** | Dendritic Representation | 0.9954 b | 0.0000 b | 0.9954 b | 0.1587 b | Compartmental filtering of recurrent drive |
| **Stage 5** | Action Interface | 0.4669 b | 0.0000 b | 0.4882 b | 0.3013 b | Dimensionality reduction ($128 \to 2$ channels) |
| **Stage 6** | Action Assembly | 0.9954 b | 0.0000 b | 0.1562 b | **0.4690 b** | Conversion of context into sustained motor drive |
| **Stage 7** | Final Action Selection | 0.0000 b | 0.0000 b | 0.1903 b | **0.4690 b** | Discrete motor decision execution |

---

## 4. Key Scientific Insights

### 4.1 Resolving the Information Bottleneck
A fundamental question of Phase 16 was:
> Which mechanism is actually responsible for converting preserved contextual information into action?

The 7-stage information audit provides a definitive answer:
1. **Preservation:** Stages 1, 2, and 3 maintain near-maximal continuous Shannon mutual information ($0.9954\text{ bits}$ out of a theoretical maximum of $1.0\text{ bit}$). The combination of **Persistent Latent NMDA dynamics (`K`)** and **Temporal Population bursting (`J`)** ensures zero contextual information loss during the delay horizon.
2. **Transformation:** The critical bottleneck occurs at **Stage 5 (Action Interface)** and **Stage 6 (Action Assembly)**. In the un-pruned FULL_P15 system, action-relevant mutual information was 0.0000 b because competing timescales and diffuse retrograde signals prevented convergence.
3. In `MIN_E`, removing the interfering Medium assembly and diffuse retrograde feedback allows the Slow Assembly (Stage 6) to convert preserved context into a massive **$0.4690\text{ bits}$ of action-relevant mutual information**.

### 4.2 The Role of Target-Specific Routing
When Target-Specific Routing (`L`) is active, Stage 5 compresses the 128-dimensional temporal population into 2 mutually inhibitory action channels without erasing the decision signal ($I(\text{Interface}; \text{Context}) = 0.4882\text{ b}$, $I(\text{Interface}; \text{Action}) = 0.3013\text{ b}$). The Slow Assembly then amplifies this to $0.4690\text{ b}$, resulting in clean, decisive action commitment.
