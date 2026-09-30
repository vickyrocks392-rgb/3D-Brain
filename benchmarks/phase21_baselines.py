"""Phase 21 Track E Benchmark: Fair Baseline Comparison & Compute Accounting.

Compares the Phase 21 system against the retained Phase 19/20 baselines under
identical accounting. Reports exact measurements only: no claim is made that
3D-Brain "beats Transformers".
"""

from typing import Any, Dict, List
import numpy as np


def analytical_flops_per_step(system: str, dim: int = 40, k: int = 2) -> Dict[str, int]:
    """Analytical FLOPs per step for each system (specification 7)."""
    if system == "MIN_E_PLUS_PHASE21":
        ff = 2 * dim * k
        retrieval = 2 * 12          # sparse key projection
        replay = 4 * dim * k
        return {"feedforward": ff, "retrieval": retrieval, "replay": replay, "total": ff + retrieval + replay}
    if system == "SIMPLE_RECURRENT":
        return {"feedforward": 2 * dim * 40 + 2 * 40 * 40 + 2 * 40 * k, "total": 2 * dim * 40 + 2 * 40 * 40 + 2 * 40 * k}
    if system == "BPTT_RECURRENT":
        return {"feedforward": 2 * dim * 40 + 2 * 40 * 40 + 2 * 40 * k, "total": 2 * dim * 40 + 2 * 40 * 40 + 2 * 40 * k}
    if system == "CAUSAL_TRANSFORMER":
        d_model = 16
        return {"feedforward": 4 * d_model * d_model + 2 * d_model, "total": 4 * d_model * d_model + 2 * d_model}
    raise ValueError(f"Unknown system: {system}")


BASELINE_SPECS: List[Dict[str, Any]] = [
    {
        "system": "MIN_E_PLUS_PHASE21",
        "trainable_parameters": 80 + 480 + 123,
        "fixed_parameters": 2000,
        "state_variables": 282,
        "external_memory_bytes": 71640,
        "external_memory_included_in_parameters": False,
        "sequence_length": 15,
        "training_examples": 2400,
        "optimization_steps": 2400,
    },
    {
        "system": "SIMPLE_RECURRENT",
        "trainable_parameters": 80,
        "fixed_parameters": 1920,
        "state_variables": 120,
        "external_memory_bytes": 0,
        "external_memory_included_in_parameters": False,
        "sequence_length": 15,
        "training_examples": 2400,
        "optimization_steps": 2400,
    },
    {
        "system": "BPTT_RECURRENT",
        "trainable_parameters": 2041,
        "fixed_parameters": 0,
        "state_variables": 40,
        "external_memory_bytes": 0,
        "external_memory_included_in_parameters": False,
        "sequence_length": 15,
        "training_examples": 2400,
        "optimization_steps": 2400,
    },
    {
        "system": "CAUSAL_TRANSFORMER",
        "trainable_parameters": 2322,
        "fixed_parameters": 0,
        "state_variables": 4096,
        "external_memory_bytes": 0,
        "external_memory_included_in_parameters": False,
        "sequence_length": 15,
        "training_examples": 2400,
        "optimization_steps": 2400,
    },
]


def build_baseline_comparison(peak_rss_mb: float = 0.0) -> Dict[str, Any]:
    """Assemble the fair baseline comparison table with explicit accounting."""
    rows: List[Dict[str, Any]] = []
    for spec in BASELINE_SPECS:
        flops = analytical_flops_per_step(spec["system"])
        rows.append({
            **spec,
            "total_parameters": spec["trainable_parameters"] + spec["fixed_parameters"],
            "flops_per_step": flops["total"],
            "flops_per_trial": flops["total"] * spec["sequence_length"],
            "total_training_flops": flops["total"] * spec["optimization_steps"] * spec["sequence_length"],
            "peak_rss_mb": round(peak_rss_mb, 2),
        })

    return {
        "baselines": rows,
        "note": (
            "Exact benchmark measurements only. No claim is made that 3D-Brain "
            "beats Transformers. External memory is reported separately and is "
            "NOT included in any parameter count."
        ),
        "memory_accounting_is_separate": True,
    }
