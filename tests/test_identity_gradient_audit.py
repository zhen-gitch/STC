import csv

import pytest

from src.config import load_experiment_config
from src.diagnostics.identity_gradient_audit import (
    summarize_identity_gradient_audit,
    summarize_identity_gradient_run,
)


def _load(*overrides):
    return load_experiment_config(
        overrides=list(overrides),
        require_local_paths=False,
    )


def _write_metrics(path):
    fieldnames = [
        "epoch",
        "val_RMSE_epoch",
        "train_grad_bdi_norm_epoch",
        "train_grad_identity_reversed_norm_epoch",
        "train_grad_identity_to_bdi_ratio_epoch",
        "train_grad_cosine_epoch",
        "train_grad_conflict_epoch",
        "train_grad_cancellation_epoch",
        "train_identity_accuracy_epoch",
    ]
    rows = [
        {
            "epoch": 0,
            "val_RMSE_epoch": 12.0,
            "train_grad_bdi_norm_epoch": 2.0,
            "train_grad_identity_reversed_norm_epoch": 0.5,
            "train_grad_identity_to_bdi_ratio_epoch": 0.25,
            "train_grad_cosine_epoch": -0.4,
            "train_grad_conflict_epoch": 1.0,
            "train_grad_cancellation_epoch": 0.2,
            "train_identity_accuracy_epoch": 0.4,
        },
        {
            "epoch": 1,
            "val_RMSE_epoch": 13.0,
            "train_grad_bdi_norm_epoch": 1.0,
            "train_grad_identity_reversed_norm_epoch": 0.5,
            "train_grad_identity_to_bdi_ratio_epoch": 0.5,
            "train_grad_cosine_epoch": -0.8,
            "train_grad_conflict_epoch": 1.0,
            "train_grad_cancellation_epoch": 0.4,
            "train_identity_accuracy_epoch": 0.5,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_identity_gradient_configs_are_paired_and_validation_only():
    common = "configs/identity_gradient_audit/common_full40.yaml"
    reference = _load(
        "configs/stage_b/base_regression_only.yaml",
        common,
        "configs/identity_gradient_audit/reference.yaml",
    )
    adversarial = _load(
        "configs/stage_b/base_regression_only.yaml",
        common,
        "configs/identity_gradient_audit/bdi_identity_adversarial.yaml",
    )

    assert reference.EXPERIMENT_GROUP == adversarial.EXPERIMENT_GROUP
    assert reference.SEED == adversarial.SEED == 42
    assert reference.RUN_TEST_AFTER_FIT is adversarial.RUN_TEST_AFTER_FIT is False
    assert reference.PROCESS_TEMPORAL.MAX_EPOCHS == 40
    assert adversarial.PROCESS_TEMPORAL.MAX_EPOCHS == 40
    assert reference.EARLY_STOPPING.ENABLE is False
    assert adversarial.EARLY_STOPPING.ENABLE is False
    assert reference.MODEL.IDENTITY_ADVERSARIAL.ENABLE is False
    assert adversarial.MODEL.IDENTITY_ADVERSARIAL.ENABLE is True
    assert adversarial.MODEL.IDENTITY_ADVERSARIAL.LAMBDA_ID == 0.05
    assert adversarial.MODEL.IDENTITY_ADVERSARIAL.GRADIENT_AUDIT_ENABLE is True
    assert reference.MODEL.SEVERITY_BALANCED_REGRESSION.ENABLE is False
    assert adversarial.MODEL.SEVERITY_BALANCED_REGRESSION.ENABLE is False
    assert reference.MODEL.TASK_NUISANCE.ENABLE is False
    assert adversarial.MODEL.TASK_NUISANCE.ENABLE is False


def test_identity_gradient_summary_reports_conflict_and_writes_outputs(tmp_path):
    metrics = _write_metrics(tmp_path / "metrics.csv")
    summary, rows = summarize_identity_gradient_run("adv", metrics)

    assert len(rows) == 2
    assert summary["mean_identity_to_bdi_ratio"] == pytest.approx(0.375)
    assert summary["mean_cosine"] == pytest.approx(-0.6)
    assert summary["mean_conflict_rate"] == 1.0
    assert summary["corr_cosine_vs_val_rmse"] < 0.0

    generated = summarize_identity_gradient_audit(
        [("adv", metrics)],
        output_dir=tmp_path / "summary",
    )
    assert len(generated) == 3
    assert all(path.exists() for path in generated)
