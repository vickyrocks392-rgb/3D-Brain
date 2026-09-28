"""Integration test for experiment runner and .brainlog generation."""

from pathlib import Path
from experiments.runner import ExperimentConfig, run_experiment
from visualization.telemetry import validate_brainlog


def test_experiment_runner_end_to_end(tmp_path):
    """Run a micro experiment and verify results format and brainlog validity."""
    config = ExperimentConfig(
        experiment_id="TEST_RUNNER",
        model_type="CAND-LI",
        target_budget=300,
        t_delay=10,
        distractor_sigma=0.05,
        seed=42,
        epochs=5,  # Short run for test
        batch_size=16,
        num_train_batches=4,
        num_val_samples=20,
        num_test_samples=30,
        lr=0.01,
    )

    result, brainlog_path = run_experiment(config)

    assert result["experiment_id"] == "TEST_RUNNER"
    assert result["model_id"] == "CAND-LI"
    assert "test_accuracy" in result
    assert "pes" in result
    assert "msr" in result
    assert result["parameter_count"] == 301

    assert brainlog_path is not None
    assert brainlog_path.exists()

    valid, reason = validate_brainlog(brainlog_path)
    assert valid, f"Brainlog validation failed: {reason}"
