# Phase 18 — Benchmark Integrity & Scaling Audit Report

**Audit Target:** Phase 17 Ultra-High Scaling Benchmark ($N=100\text{K}$)  
**Date:** September 30, 2026  
**Status:** **AUDITED — BENCHMARK ACCOUNTING FAILURE IDENTIFIED & CORRECTED**

---

### Executive Summary

Phase 17 reported the following headline benchmark results for MIN_E at $N=100\text{K}$:
- **Throughput:** $107,989\text{ steps/sec}$
- **Latency:** $\approx 0.0093\text{ ms/step}$
- **Memory:** $7.63\text{ MB}$
- **Synaptic Throughput:** $215.98\text{ billion synaptic operations/sec}$

Our Phase 18 audit subjected this benchmark harness to independent deconstruction and instrumentation. **We found that the Phase 17 scaling benchmark did not execute a 100,000-neuron model.** Instead, the benchmark code contained an explicit dimension clamp (`dim = min(N, 120)`) that simulated only a $120 \times 120$ matrix, and then multiplied the resulting loop frequency by a hypothetical $(N \times 20)$ to report theoretical FLOPs as executed hardware throughput.

---

### 1. Detailed Audit Inquiries & Findings

| Audit Question | Audit Finding | Classification |
|:---|:---|:---:|
| **1. What exactly is one "step"?** | In Phase 17, one step was defined as a single synthetic forward loop iteration: `s = np.tanh(W @ s + 0.1)`. It did not execute two-compartment dynamics, NMDA integration, burst detection, assembly competition, or plasticity. | **Synthetic Proxy Only** |
| **2. What is a "synaptic operation"?** | Theoretical multiplication count: `steps_per_sec * (N * 20)`. No individual synaptic updates or sparse graph traversals were executed. | **Theoretical Projection** |
| **3. Were operations actually executed?** | **NO.** Code inspection revealed: `dim = min(N, 120)`. For all $N \ge 120$, only 120 neurons ($120 \times 120 = 14,400$ multiply-adds) were executed. At $N=100\text{K}$, 99.88% of the claimed neurons were never computed. | **Critical Failure** |
| **4. Were zero-valued/sparse connections counted?** | Yes, the theoretical density of 20 synapses per neuron was assumed and multiplied directly. | **Unverified Assumption** |
| **5. Were vectorized operations counted as individual ops?** | Yes, SIMD BLAS matrix-vector product cycles were converted into scalar synaptic operation equivalents. | **Accounting Distortion** |
| **6. Was memory measured before or after allocation?** | Memory was **NOT measured from the operating system**. It was calculated using a theoretical formula: `memory_mb = (N * 20 * 4) / (1024 * 1024)`. | **Theoretical Calculation** |
| **7. Was Python object overhead included?** | No. Process RSS, NumPy array wrappers, garbage collection overhead, and telemetry structures were completely omitted. | **Omitted** |
| **8. Was model state included?** | No. State variables for soma, dendrite, NMDA conductances, eligibility traces, and assembly integrators were not allocated in the benchmark loop. | **Omitted** |
| **9. Was BrainLog generation included?** | No. Logging was bypassed during the benchmark loop. | **Omitted** |
| **10. Was telemetry included?** | No. Telemetry frame capture was omitted. | **Omitted** |
| **11. Was warm-up excluded?** | No explicit warm-up discard was implemented, though the effect was minor. | **Minor** |
| **12. Is compilation/caching involved?** | NumPy uses Apple Accelerate BLAS (caching vector operations). | **Observed** |
| **13. Is the benchmark CPU-only?** | Yes, single-thread CPU execution on Apple Silicon M-series. | **Confirmed** |
| **14. Were all 100K neurons dynamically updated?** | **NO.** Only 120 neurons were updated. | **False Claim** |
| **15. Were all claimed synapses represented?** | **NO.** No $100\text{K} \times 20$ sparse adjacency list or matrix was allocated. | **False Claim** |
| **16. Was operation count theoretical or measured?** | **THEORETICAL.** | **Correction Required** |

---

### 2. Independent Benchmark Harness & Measured Reality

Phase 18 implemented an independent benchmark harness allocating actual sparse neural representations across $N \in [50, 100, 300, 1\text{K}, 10\text{K}, 50\text{K}, 100\text{K}]$ with $k=20$ incoming synapses per neuron, executing actual sparse graph lookups, non-linear somatic integration, and measuring OS-level resident set size (`ru_maxrss`).

#### Measured vs Reported Comparison Matrix

| $N$ (Neurons) | Real Executed Steps/s | Real Executed Syn-Ops/s | Real $\Delta$RSS (MB) | Phase 17 Reported Syn-Ops/s | Overstatement Ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **50** | 41,200 | 41.2 M | 0.05 MB | 228.3 M | 5.5× |
| **100** | 35,400 | 70.8 M | 0.08 MB | 362.1 M | 5.1× |
| **300** | 28,100 | 168.6 M | 0.22 MB | 612.4 M | 3.6× |
| **1,000** | 23,237 | 464.7 M | 0.65 MB | 1,464.0 M | **3.2×** |
| **10,000** | 2,535 | 507.0 M | 4.80 MB | 14,290.6 M | **28.2×** |
| **50,000** | 460 | 460.0 M | 21.50 MB | 72,100.0 M | **156.7×** |
| **100,000** | **220.9** | **441.8 M** | **43.20 MB** | **215,980.0 M** | **488.8× to 2000×** |

---

### 3. Scientific Verdict & Corrected Benchmark Headline

1. **Root Cause:**
   The Phase 17 scaling law code line:
   ```python
   dim = min(N, 120)
   W_sim = np.random.randn(dim, dim).astype(np.float32) * 0.05
   ...
   syn_ops_per_sec = float(steps_per_sec * (N * 20))
   ```
   artificially evaluated a constant $120$-dimensional problem while scaling the multiplier $(N \times 20)$ linearly to $2,000,000$.

2. **Corrected $N=100\text{K}$ Throughput:**
   - **Actual Executed Throughput:** **220.9 steps/sec** (not 107,989 steps/sec).
   - **Actual Executed Synaptic Updates:** **441.8 million syn-ops/sec** (not 215.98 billion syn-ops/sec).
   - **Actual Process Memory Footprint:** **43.2 MB** delta RSS (not 7.63 MB static weight calculation).

3. **Classification:** **IMPLEMENTATION ARTIFACT / BENCHMARK ACCOUNTING FAILURE**.  
   The claimed 215.98B syn-ops/sec is hereby retracted and replaced with the empirically measured 441.8M syn-ops/sec.
