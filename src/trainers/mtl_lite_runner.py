import math
import os
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.config import resolve_experiment_save_dir, resolve_next_experiment_version
from src.datasets.dataset import AVECDataModule, load_data_list
from src.models.mtl_lite import MTLLiteDepressionModel


def _get_config_value(configs, key, default):
    return getattr(configs, key, default)


def resolve_run_test_after_fit(cfgs):
    """Resolve whether a completed fit may open the test split."""
    value = _get_config_value(cfgs, "RUN_TEST_AFTER_FIT", True)
    if not isinstance(value, bool):
        raise ValueError(
            "RUN_TEST_AFTER_FIT must be a boolean, "
            f"got: {value!r}"
        )
    return value


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enable: bool = False
    monitor: str = "val_RMSE_epoch"
    mode: str = "min"
    patience: int = 8
    min_delta: float = 0.0
    strict: bool = True
    check_finite: bool = True


def _get_early_stopping_value(section, key, default):
    if section is None:
        return default
    return getattr(section, key, default)


def resolve_early_stopping_config(cfgs):
    """Resolve and validate the shared checkpoint/early-stopping policy."""
    section = getattr(cfgs, "EARLY_STOPPING", None)
    allowed_keys = {
        "ENABLE",
        "MONITOR",
        "MODE",
        "PATIENCE",
        "MIN_DELTA",
        "STRICT",
        "CHECK_FINITE",
    }
    if section is not None:
        if not hasattr(section, "keys"):
            raise ValueError("EARLY_STOPPING must be a mapping of policy fields")
        unknown_keys = set(section.keys()) - allowed_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unknown EARLY_STOPPING config field(s): {unknown}")

    defaults = EarlyStoppingConfig()
    values = {
        "enable": _get_early_stopping_value(section, "ENABLE", defaults.enable),
        "monitor": _get_early_stopping_value(section, "MONITOR", defaults.monitor),
        "mode": _get_early_stopping_value(section, "MODE", defaults.mode),
        "patience": _get_early_stopping_value(section, "PATIENCE", defaults.patience),
        "min_delta": _get_early_stopping_value(section, "MIN_DELTA", defaults.min_delta),
        "strict": _get_early_stopping_value(section, "STRICT", defaults.strict),
        "check_finite": _get_early_stopping_value(
            section, "CHECK_FINITE", defaults.check_finite
        ),
    }

    for key in ("enable", "strict", "check_finite"):
        if not isinstance(values[key], bool):
            raise ValueError(
                f"EARLY_STOPPING.{key.upper()} must be a boolean, "
                f"got: {values[key]!r}"
            )

    if not isinstance(values["monitor"], str) or not values["monitor"].strip():
        raise ValueError("EARLY_STOPPING.MONITOR must be a non-empty string")
    values["monitor"] = values["monitor"].strip()

    if not isinstance(values["mode"], str) or values["mode"] not in {"min", "max"}:
        raise ValueError(
            "EARLY_STOPPING.MODE must be either 'min' or 'max', "
            f"got: {values['mode']!r}"
        )

    patience = values["patience"]
    if isinstance(patience, bool) or not isinstance(patience, Integral) or patience < 0:
        raise ValueError(
            "EARLY_STOPPING.PATIENCE must be a non-negative integer, "
            f"got: {patience!r}"
        )
    values["patience"] = int(patience)

    min_delta = values["min_delta"]
    if (
        isinstance(min_delta, bool)
        or not isinstance(min_delta, Real)
        or not math.isfinite(float(min_delta))
        or min_delta < 0
    ):
        raise ValueError(
            "EARLY_STOPPING.MIN_DELTA must be a finite non-negative number, "
            f"got: {min_delta!r}"
        )
    values["min_delta"] = float(min_delta)

    return EarlyStoppingConfig(**values)


def build_mtl_lite_callbacks(cfgs, calibration_only=False):
    """Build callbacks with one validated monitor policy for model selection."""
    if calibration_only:
        return [
            RichProgressBar(),
            ModelCheckpoint(
                monitor=None,
                save_top_k=0,
                save_last=True,
                filename="mtl_lite-calibration-{step:04d}",
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

    early_stopping = resolve_early_stopping_config(cfgs)
    checkpoint_filename = (
        f"mtl_lite-{{epoch:03d}}-{{{early_stopping.monitor}:.4f}}"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=early_stopping.monitor,
        mode=early_stopping.mode,
        save_top_k=1,
        save_last=True,
        filename=checkpoint_filename,
    )
    callbacks = [
        RichProgressBar(),
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
    ]
    if early_stopping.enable:
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping.monitor,
                mode=early_stopping.mode,
                patience=early_stopping.patience,
                min_delta=early_stopping.min_delta,
                strict=early_stopping.strict,
                check_finite=early_stopping.check_finite,
            )
        )
    return callbacks


def build_train_subject_index(cfgs):
    """Build a train-only ``{subject_id: class_index}`` table.

    Subject ids are the first 5 characters of each video folder name, matching
    :meth:`AVECDataset._label_value_for_video_id` (``video_id[0:5]``).  Only the
    ``train`` split is read; val/test subjects are deliberately absent so that
    batches containing unseen subjects produce no identity loss.  Indices are
    assigned in first-seen order over the split file, so the table is
    deterministic for a fixed split.
    """
    train_folders = load_data_list(
        cfgs.DATASET_SPLIT_FILE, cfgs.IMAGE_DIR, "train"
    )
    subject_index = {}
    for folder in train_folders:
        video_id = Path(folder).name
        subject_id = video_id[0:5]
        if subject_id not in subject_index:
            subject_index[subject_id] = len(subject_index)
    return subject_index


def _severity_bin_for_score(score, edges):
    """Return the bin name for a raw BDI score, matching io.severity_group."""
    s = float(score)
    if s <= edges[0]:
        return "minimal"
    if s <= edges[1]:
        return "mild"
    if s <= edges[2]:
        return "moderate"
    return "severe"


def build_train_severity_bin_counts(cfgs, edges=(13, 19, 28)):
    """Count train-only samples in each severity bin.

    Reads each train video's ``<subject>_Depression.csv`` label directly
    (no image loading), so this is cheap and runs before ``trainer.fit``.
    Bin edges default to the diagnostics-standard ``[13, 19, 28]`` to match
    :func:`src.diagnostics.io.severity_group` and the model's own
    ``severity_bin_edges``; val/test labels are never read.
    """
    label_dir = Path(cfgs.LABEL_DIR).expanduser()
    train_folders = load_data_list(
        cfgs.DATASET_SPLIT_FILE, cfgs.IMAGE_DIR, "train"
    )
    counts = {"minimal": 0, "mild": 0, "moderate": 0, "severe": 0}
    for folder in train_folders:
        video_id = Path(folder).name
        subject_id = video_id[0:5]
        label_path = label_dir / f"{subject_id}_Depression.csv"
        if not label_path.exists():
            continue
        label = int(label_path.read_text().strip())
        counts[_severity_bin_for_score(label, edges)] += 1
    return counts


def build_train_severity_label_histogram(cfgs, max_score=63):
    """Build a train-only integer BDI histogram for continuous weighting."""
    label_dir = Path(cfgs.LABEL_DIR).expanduser()
    train_folders = load_data_list(
        cfgs.DATASET_SPLIT_FILE, cfgs.IMAGE_DIR, "train"
    )
    histogram = [0] * (int(max_score) + 1)
    for folder in train_folders:
        video_id = Path(folder).name
        subject_id = video_id[0:5]
        label_path = label_dir / f"{subject_id}_Depression.csv"
        if not label_path.exists():
            continue
        label = int(label_path.read_text().strip())
        if 0 <= label <= int(max_score):
            histogram[label] += 1
    return histogram


def build_mtl_lite_trainer(
    cfgs, calibration_only=False, calibration_steps=100
):
    """Build the Lightning trainer for the MTL-Lite mainline."""
    save_dir = resolve_experiment_save_dir(cfgs)
    version = resolve_next_experiment_version(save_dir)
    calibration_kwargs = {}
    if calibration_only:
        calibration_kwargs = {
            "max_steps": int(calibration_steps),
            "limit_val_batches": 0,
            "num_sanity_val_steps": 0,
        }
    return pl.Trainer(
        accelerator=cfgs.ACCELERATOR,
        devices=cfgs.DEVICES,
        strategy=_get_config_value(cfgs, "STRATEGY", "auto"),
        precision=cfgs.PRECISION,
        max_epochs=cfgs.PROCESS_TEMPORAL.MAX_EPOCHS,
        callbacks=build_mtl_lite_callbacks(
            cfgs, calibration_only=calibration_only
        ),
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        logger=[
            CSVLogger(save_dir=save_dir, name="", version=version),
            TensorBoardLogger(save_dir=save_dir, name="", version=version),
        ],
        **calibration_kwargs,
    )


def save_resolved_config(cfgs, trainer):
    """Save the merged run config next to the CSV logger output."""
    if not trainer.loggers:
        return

    log_dir = trainer.loggers[0].log_dir
    os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(config=cfgs, f=os.path.join(log_dir, "resolved_config.yaml"))


def run_mtl_lite(cfgs):
    """Train the lightweight model and optionally test the best checkpoint."""
    run_test_after_fit = resolve_run_test_after_fit(cfgs)
    model = MTLLiteDepressionModel(cfgs)
    data_module = AVECDataModule(cfgs)

    # Stage B1: inject a train-only subject index when identity-adversarial is
    # enabled.  The model is the single source of truth for its own switch
    # (read in __init__); the runner only builds the table and injects it.
    # Built before trainer.fit so the lazily-constructed subject_id_head is
    # registered before the optimizer is configured.
    if model.identity_adversarial:
        subject_index = build_train_subject_index(cfgs)
        model.set_subject_index(subject_index)
        print(
            f"[RUNNER] identity-adversarial ON: {len(subject_index)} train "
            f"subjects, lambda_id={model.lambda_id}"
        )

    # Stage B2: inject train-only severity bin counts when severity-balanced
    # regression is enabled.  Bin edges are taken from the model so the
    # runner-side counting and model-side bin indexing share one source of
    # truth.  Val/test labels are never read for weighting.
    if (
        model.severity_balanced_regression
        and model.severity_weighting_mode == "bin"
    ):
        bin_counts = build_train_severity_bin_counts(
            cfgs, edges=tuple(model.severity_bin_edges)
        )
        model.set_severity_bin_counts(bin_counts)
        print(
            f"[RUNNER] severity-balanced regression ON: train bin counts "
            f"{bin_counts}, power={model.severity_power}"
        )
    elif (
        model.severity_balanced_regression
        and model.severity_weighting_mode == "continuous"
    ):
        histogram = build_train_severity_label_histogram(
            cfgs, max_score=model.max_score
        )
        model.set_severity_label_histogram(histogram)
        print(
            "[RUNNER] continuous severity weighting ON: train histogram "
            f"observed_labels={sum(count > 0 for count in histogram)}, "
            f"samples={sum(histogram)}, sigma={model.severity_smoothing_sigma}, "
            f"power={model.severity_power}"
        )

    calibration_only = model.task_nuisance_config.calibration_only
    trainer = build_mtl_lite_trainer(
        cfgs,
        calibration_only=calibration_only,
        calibration_steps=model.task_nuisance_config.calibration_steps,
    )
    save_resolved_config(cfgs, trainer)

    print("\n[RUNNER] 正在启动 MTL-Lite 训练引擎...")
    trainer.fit(model, data_module)

    if calibration_only:
        print(
            "\n[RUNNER] Train-only auxiliary calibration finished at "
            f"{model.task_nuisance_config.calibration_steps} steps; "
            "validation and test were not opened."
        )
        return

    if not run_test_after_fit:
        print(
            "\n[RUNNER] Validation-only run finished; test split remains closed."
        )
        return

    print("\n[RUNNER] 训练结束，正在使用验证集最优 checkpoint 进行 Test 集评估...")
    trainer.test(model, datamodule=data_module, ckpt_path="best")
