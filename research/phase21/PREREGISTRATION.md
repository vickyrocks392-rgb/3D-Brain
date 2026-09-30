# Phase 21 Preregistration: Episodic Memory, Replay Consolidation & Abstract Task Transfer

**Document ID:** PREREG-PHASE21-001
**Date:** September 30, 2026
**Auditor / Lead Investigator:** Autonomous Scientific Research Agent
**Parent Foundation:** Phase 20 Scientific Report (`research/phase20/PHASE_20_FINAL_REPORT.md`), Phase 19 Audit, Phase 18 Falsification Report
**Repository:** `~/Documents/3D-Brain`
**GitHub:** `https://github.com/vickyrocks392-rgb/3D-Brain.git`

> **PREREGISTRATION INTEGRITY NOTICE**
> This document was written **before** any Phase 21 implementation or data collection.
> No Phase 21 result influenced its contents. Any deviation discovered during execution
> must be logged in Section 12 (Deviation Log) with justification, and affected
> experiments must be rerun. Retroactive hypothesis changes are forbidden.

---

## 1. Context and Scientific Motivation

Phase 20 established, under its tested protocol:

- **Context-gated modular routing** reduced measured cross-task interference and required only a modest parameter increase.
- **Branch-specific plasticity** compartmentalized competing mappings.
- **Sparse replay during quiescent rest** extended the measured effective credit horizon from ~240 ms to 60,000 ms.
- **Replay effects survived causal interventions** (content randomization, timing randomization, wrong-episode injection).
- The integrated system showed strong continual-retention performance in the tested benchmark.
- 303 tests passed; BrainLog v3.6 validated.

### 1.1 The Gap Phase 21 Addresses

Phase 20 demonstrated that *replay can bridge long temporal gaps*. It did **not** demonstrate that the system possesses episodic memory.

The critical logical gap is:

> A system could obtain strong benchmark performance by **replaying specific, previously learned episodes** — i.e. memorization — without any capacity to **store distinct experiences**, **consolidate them offline**, **retrieve them selectively**, or **abstract structure across episodes**.

Memorization, episodic retrieval, and abstraction are three distinct capabilities. Phase 20 conflated the first with the second. Phase 21 separates all three.

### 1.2 Central Scientific Question

Can the system:

1. **Store** distinct episodic experiences?
2. **Consolidate** them during quiescent periods?
3. **Retrieve** them selectively?
4. **Abstract** information across episodes?
5. **Apply** the abstracted result to a novel task or combination?

### 1.3 Biological Interpretation Rule (Non-Negotiable)

"Episodic memory" and "consolidation" in this document refer to **computational mechanisms inspired by** hippocampal episodic indexing and systems consolidation. This phase does **not** establish biological equivalence, does not model CA1/CA3 circuitry, and does not prove that biological hippocampus implements these rules. Computational results and biological interpretation are reported in separate sections. Claims are restricted to **"compositional transfer"** unless evidence supports something stronger.

---

## 2. Non-Negotiable Scientific Rules

These are binding. Violation invalidates the affected experiment.

| # | Rule | Enforcement Mechanism |
|:--|:--|:--|
| R1 | **No data leakage** | Trial generators are pure functions of `(trial_idx, task, seed)`; test sets generated from disjoint seed streams |
| R2 | **No future target access** | `target_action` is delivered only at reward steps; asserted in unit tests |
| R3 | **No continuous target supervision** | Teaching signal fires only on `is_reward_step` |
| R4 | **No evaluation-time learning** | `net.set_frozen(True)` before every test evaluation; verified by weight-hash test |
| R5 | **No test-time plasticity** | Zero `replay`/`consolidation` calls during zero-shot test; asserted |
| R6 | **No hidden task labels** | No `x[fixed_index] = task_id` shortcuts; sensory schema identical across tasks |
| R7 | **No oracle context** except explicit upper-bound controls | Oracle conditions labelled `ORACLE` and excluded from primary claims |
| R8 | **No random test-set selection** | Test splits are deterministic functions of seed; enumerating all splits is rejected |
| R9 | **No architecture creep** | MIN_E substrate is unchanged; additions are explicit modules, counted and reported |
| R10 | **No unjustified parameter scaling** | Every added parameter reported separately in Section 11 accounting |
| R11 | **No retroactive hypothesis changes** | Section 12 deviation log; rerun required |
| R12 | **No metric cherry-picking** | Primary endpoints pre-registered in Section 6.4; others labelled exploratory |

### 2.1 Train / Validation / Test Separation (Mandatory)

Every experiment has three disjoint splits:

- **TRAIN**: learning permitted (plasticity, replay, consolidation allowed).
- **VALIDATION**: used for capacity/retention decisions only; never reported as test performance.
- **TEST**: frozen model, zero plasticity, zero replay, zero target access. Reported numbers come only from TEST.

Splits are generated from **disjoint seed streams** (`train`, `valid`, `test` namespaces), guaranteeing no trial appears in more than one split.

### 2.2 Zero-Shot Definition (Mandatory, Non-Negotiable)

A result is called **zero-shot** if and only if, during the entire evaluation episode:
## 3. Track Definitions and Core Hypotheses

Tracks A, B, and C are completed **independently** before Track D. Track E reuses Phase 19/20 baselines.

### 3.1 Track A — Episodic Memory

**Question:** Does the system possess an explicit episodic memory mechanism that stores event-level/latent experience (not supervised input→target mappings), discriminates similar episodes without hidden IDs, and retrieves selectively?

| ID | Hypothesis | Primary Endpoint |
|:--|:--|:--|
| **H-A1** | An explicit episodic memory improves selective retrieval accuracy over no memory (A0) | Test retrieval accuracy, A3 vs A0 |
| **H-A2** | Indexed/sparse content-addressable memory discriminates temporally confusable episodes better than a passive buffer | Episode discrimination accuracy |
| **H-A3** | Retrieval gains survive causal interventions (removal, randomization, identity shuffling, wrong-episode, capacity reduction) | Δaccuracy under intervention |
| **H-A4** | Episodic memory footprint is measurable in bytes and reported separately from neural parameters | Memory bytes (explicit, not hidden) |
| **H-A5** | Retrieval is sparse (few episodes retrieved per query), not a dense broadcast | Retrieval sparsity |

### 3.2 Track B — Replay Consolidation

**Question:** Does replay *cause* consolidation during quiescent rest, or merely re-present stored answers?

| ID | Hypothesis | Primary Endpoint |
|:--|:--|:--|
| **H-B1** | Replay during explicit rest produces measurable post-rest retention gain vs no replay | Post-rest retention, B6 vs B0 |
| **H-B2** | Consolidation changes (synaptic + slow-state) occur during rest and are measurable | Δsynaptic norm before/after rest |
| **H-B3** | Consolidation survives intervening tasks | Retention after intervening task |
| **H-B4** | Replay benefits are **content-specific**: exact replay > shuffled/reversed/wrong/synthetic | Content-audit delta |
| **H-B5** | Dual-timescale and metaplasticity are **complementary or redundant** (explicitly resolved, not assumed) | B4/B5/B6 comparison |

**Falsification rule (mandatory):** If random-synthetic replay performs comparably to exact replay, episodic specificity is **NOT** claimed and H-B4 is classified `NOT SUPPORTED`.

### 3.3 Track C — Abstract Transfer

**Question:** Can previously learned components be recombined into combinations never trained, under frozen zero-shot evaluation?

| ID | Hypothesis | Primary Endpoint |
|:--|:--|:--|
| **H-C1** | Zero-shot accuracy on unseen combinations (A+B, A+C, B+C, A+B+C) exceeds chance | Zero-shot test accuracy |
| **H-C2** | Transfer depends on episodic retrieval (removing memory collapses transfer) | Δaccuracy with memory ablated |
| **H-C3** | Transfer is compositional, not linear interpolation | Nonlinear composition accuracy |
| **H-C4** | Transfer survives frozen evaluation with zero plasticity/replay/targets | Frozen-evaluation assertion |
| **H-C5** | Abstraction exceeds verbatim episode replay (i.e. system generalizes beyond memorized episodes) | Abstraction vs replay delta |

**Falsification rule (mandatory):** If zero-shot accuracy is at chance, compositional transfer is reported as `NOT SUPPORTED` and no additional mechanism may be added to force a positive result.

### 3.4 Track D — Integrated System

Run only after Tracks A–C complete. Compares D0–D5. **D5 is not assumed to win.** The purpose is to determine whether mechanisms are **complementary**, **redundant**, **antagonistic**, or **task-dependent**.

### 3.5 Track E — Fair Baselines

Retains Phase 20/19 baselines under identical accounting. Reports exact benchmark measurements. **Does not claim "3D-Brain beats Transformers."**

## 4. Conditions (Pre-Registered)

### 4.1 Track A Conditions

| Condition | Description |
|:--|:--|
| **A0** | No episodic memory |
| **A1** | Passive replay buffer (store-all, uniform retrieval) |
| **A2** | Indexed episodic memory (key-indexed lookup) |
| **A3** | Sparse content-addressable episodic memory (sparse top-k over latent keys) |
| **A4** | Episodic memory + existing replay |
| **A5** | Episodic memory + replay + consolidation |

### 4.2 Track B Conditions

| Condition | Description |
|:--|:--|
| **B0** | No replay |
| **B1** | Random replay |
| **B2** | Event-triggered replay |
| **B3** | Prioritized replay |
| **B4** | Replay + dual-timescale consolidation |
| **B5** | Replay + metaplasticity |
| **B6** | Replay + both consolidation mechanisms |

### 4.3 Replay Content Audit (7 protocols)

1. Exact episode replay
2. Partial episode replay
3. Latent-state replay
4. Shuffled replay
5. Temporally reversed replay
6. Wrong-episode replay
7. Random synthetic replay

### 4.4 Track C Task Structure

**Training components (individually experienced):**
- Rule A: shape → action
- Rule B: color → action
- Rule C: temporal relation → action

**Test combinations (never trained):** A+B, A+C, B+C, A+B+C

**Nonlinear composition:** explicit XOR-like rule with a linear control and a shuffled-label control.

### 4.5 Track D Conditions

## 5. Experimental Design

### 5.1 Episode Discrimination Paradigm

Episodes are constructed with **deliberately similar observations but different temporal/contextual consequences**:

```
Episode A: same initial observation -> event sequence 1 -> action 1
Episode B: same initial observation -> event sequence 2 -> action 2
```

The system **must** use temporal/contextual information. **No hidden episode ID is provided as an input feature.** The query is constructed from legitimately available state (latent context + recent event sequence), never from a ground-truth episode label.

Measured: correct retrieval, incorrect retrieval, contextual confusion, temporal confusion, plus causal episode-swap experiments.

### 5.2 Memory Capacity Experiment

Capacities: **10, 25, 50, 100, 250, 500, 1000 episodes.**

Measure: retrieval accuracy, memory footprint (bytes), interference, replay cost, retrieval cost, consolidation cost.

### 5.3 Forgetting Experiment

Sequences: **A→B→A**, **A→B→C→A**, **A→B→C→D→A.**

Measure: per-task initial and final accuracy, retention ratio, absolute forgetting, relative forgetting, forward transfer, backward transfer.

**Metric definitions (see Section 6.3):** chance-normalized quantities are used **only** where the denominator is explicitly justified, and the raw (un-normalized) value is always reported alongside.

### 5.4 Transfer Without Replay (Critical Control)

- `learn -> test` (no replay)
- `learn -> rest/replay -> test`
- `learn A -> replay A -> learn B -> test A`
- `learn A -> no replay A -> learn B -> test A`

If replay is responsible for consolidation, the difference must be measurable.

### 5.5 Causal Ablation Matrix

**Episodic memory:** removed · retrieval randomized · episode identity shuffled · wrong episode retrieved · capacity reduced.
## 6. Statistical Protocol

### 6.1 Seeds

Mandatory seed set: **`[42, 101, 2024, 777, 999]`** — exact N = 5 seeds per condition, reported explicitly for every statistic.

### 6.2 Reported Statistics

For every primary endpoint report: **mean, standard deviation, 95% confidence interval, effect size (Cohen's d), exact N, and paired tests where the same seed is run under both conditions** (paired t-test; Wilcoxon signed-rank additionally reported when N is small and differences are non-normal).

### 6.3 Metric Definitions (Well-Defined, Non-Divergent)

| Metric | Definition |
|:--|:--|
| Accuracy $acc$ | Fraction of TEST trials with $\arg\max(u) = target$ |
| Retention ratio $\rho$ | $acc_{A,\text{post}} / acc_{A,\text{pre}}$, **defined only when $acc_{A,\text{pre}} > 0$**; otherwise reported as `UNDEFINED` with raw values shown |
| Absolute forgetting $F_{abs}$ | $\max(0,\ acc_{A,\text{pre}} - acc_{A,\text{post}})$ — always well-defined |
| Relative forgetting $F_{rel}$ | $F_{abs} / acc_{A,\text{pre}}$, reported **only** alongside $F_{abs}$; never used as the sole endpoint |
| Interference $I$ | $\max(0,\ acc_{A,\text{pre}} - acc_{A,\text{post}})$ for a task $A$ after intervening task $B$ |
| Forward transfer $FWD$ | $acc_{B,\text{init}} - acc_{B,\text{chance}}$ (raw accuracy always reported alongside) |
| Backward transfer $BWT$ | $acc_{A,\text{post}} - acc_{A,\text{pre}}$ |

**Anti-division rule (mandatory):** We never divide by chance accuracy unless the denominator is justified and stated. Chance-normalized quantities appear **only** as secondary annotations.

### 6.4 Multiple Comparisons

Primary endpoints are pre-registered (Section 6.5). **The family of primary tests is fixed in advance.** p-values are reported with **Holm–Bonferroni** correction across the primary family. Exploratory analyses are explicitly labelled `EXPLORATORY` and cannot be promoted to primary.

### 6.5 Pre-Registered Primary Endpoints (Fixed Before Data Collection)

| ID | Endpoint | Comparison |
|:--|:--|:--|
| **P1** | Test retrieval accuracy | A3 vs A0 |
| **P2** | Episode discrimination accuracy | A3 vs A1 |
| **P3** | Post-rest retention | B6 vs B0 |
| **P4** | Exact vs shuffled replay delta | content audit |
| **P5** | Zero-shot novel-combination accuracy | Track C frozen test |
| **P6** | Transfer delta with memory ablated | H-C2 |
| **P7** | Integrated retention | D4 vs D0 |

### 6.6 Hypothesis Classification Categories

| Category | Definition |
|:--|:--|
| **SUPPORTED** | Primary endpoint significant at corrected $p < 0.05$, effect size $d > 0.8$, surviving all causal ablations |
| **PARTIALLY SUPPORTED** | Improvement on a subset of conditions/datasets, or degrades under secondary ablations |
| **NOT SUPPORTED** | Indistinguishable from baseline at corrected $p \ge 0.05$ |
| **FALSIFIED** | Significantly **below** baseline, or the claim is contradicted by an explicit control |
| **INCONCLUSIVE** | Seed variance too high (relative SD > 25%) to support inference |

**A negative result is an acceptable and publishable outcome.** No mechanism may be added, tuned, or extended to convert a negative result into a positive one.

## 7. Stopping Rule

This Phase 21 campaign is **fixed in scope** at Tracks A–E plus the capacity, forgetting, ablation, and audit experiments. No additional mechanism, task, or hyperparameter sweep will be added in response to results. If a hypothesis fails, it is reported as failed and Phase 21 concludes.

---

## 8. Falsification Triggers (Pre-Registered)

The following results **automatically falsify** the corresponding claims, and the phase must report them as such:

1. **Random synthetic replay ≈ exact replay** ⇒ episodic specificity falsified; replay benefit is generic compute, not memory.
2. **Zero-shot novel-combination accuracy ≤ chance** ⇒ compositional transfer falsified.
3. **Transfer survives memory ablation equally well** ⇒ transfer does not depend on episodic memory.
4. **Consolidation gains vanish when rest is replaced by active experience** ⇒ consolidation is not quiescence-specific.
5. **Episode discrimination succeeds only with oracle context** ⇒ discrimination is not genuine.

---

## 9. Reporting Obligations

Every major hypothesis must be classified as `SUPPORTED` / `PARTIALLY SUPPORTED` / `NOT SUPPORTED` / `FALSIFIED` / `INCONCLUSIVE`.

The final report must contain all 18 required sections, the leakage audit, and explicit separation of supported vs rejected claims.

---

## 10. Reproducibility Requirements

- `research/phase21/` with all preregistration, track, ablation, and final-report documents.
- `experiments/results/PHASE_21_RESULTS.json` (complete raw numerical results).
- `experiments/results/brainlogs/EXP021_episodic_transfer.brainlog` (validated BrainLog, replayable).
- Deterministic re-execution from seeds alone.

---

## 11. Resource and Accounting Constraints

Phase 21 introduces an **accounting distinction** that must be reported separately and never conflated:

1. **Neural parameters** (trainable and fixed)
2. **Dynamic neural state**
3. **Episodic memory** (bytes)
4. **Replay buffer** (bytes)
5. **Structural hyperparameters**
6. **Computational workspace**

It is forbidden to claim a system has "few parameters" while silently granting it an unbounded external-memory advantage. Every comparison states explicitly whether external memory is included.

---

## 12. Deviation Log

To be populated during execution. Any deviation from this preregistration must be recorded with: date, section affected, description, justification, and whether affected experiments were rerun.

| Date | Section | Deviation | Justification | Rerun? |
|:--|:--|:--|:--|:--|
| — | — | *(no deviations recorded)* | — | — |

---

## 13. Phase 20 Protection

Phase 20 artifacts (`research/phase20/`, `experiments/results/PHASE_20_RESULTS.json`, `experiments/results/brainlogs/EXP020_routing_replay.brainlog`) are **read-only** for this phase. No Phase 20 result may be modified, regenerated, or deleted.

---

## 14. Approval

This preregistration constitutes the binding scientific contract for Phase 21. Implementation begins only after this document and `EXPERIMENT_SPECIFICATIONS.md` exist.

---


**Replay:** removed · timing randomized · content randomized · reversed · wrong episode replayed.

**Consolidation:** removed · rate altered · tagging removed · metaplasticity removed.

**Transfer:** context removed · temporal information removed · nonlinear component removed.

**Integrated:** remove episodic memory · remove replay · remove consolidation · remove routing · remove all three memory mechanisms.

---

| Condition | Description |
|:--|:--|
| **D0** | Phase-20 baseline (MIN_E, no episodic memory) |
| **D1** | + episodic memory |
| **D2** | + replay + consolidation |
| **D3** | + episodic memory + replay |
| **D4** | + episodic memory + replay + consolidation |
| **D5** | full system + context-gated routing |

---

---


- **No weight update** (all plasticity coefficients are zero or the model is frozen)
- **No plasticity mechanisms invoked** (no replay, no consolidation, no tagging, no metaplasticity)
- **No replay during test**
- **No target access** (targets not passed into `step()` on any evaluation step)
- The model is **frozen before** the evaluation begins

Any violation invalidates the "zero-shot" label. This is enforced by an automated assertion and a dedicated test.

---
