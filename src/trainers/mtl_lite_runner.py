import os
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.config import resolve_experiment_save_dir, resolve_next_experiment_version
from src.datasets.dataset import AVECDataModule, load_data_list
from src.models.mtl_lite import MTLLiteDepressionModel


def _get_config_value(configs, key, default):
    return getattr(configs, key, default)


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


def build_mtl_lite_trainer(cfgs):
    """Build the Lightning trainer for the MTL-Lite mainline."""
    checkpoint_callback = ModelCheckpoint(
        monitor="val_RMSE_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="mtl_lite-{epoch:03d}-{val_RMSE_epoch:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    save_dir = resolve_experiment_save_dir(cfgs)
    version = resolve_next_experiment_version(save_dir)
    return pl.Trainer(
        accelerator=cfgs.ACCELERATOR,
        devices=cfgs.DEVICES,
        strategy=_get_config_value(cfgs, "STRATEGY", "auto"),
        precision=cfgs.PRECISION,
        max_epochs=cfgs.PROCESS_TEMPORAL.MAX_EPOCHS,
        callbacks=[RichProgressBar(), checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        logger=[
            CSVLogger(save_dir=save_dir, name="", version=version),
            TensorBoardLogger(save_dir=save_dir, name="", version=version),
        ],
    )


def save_resolved_config(cfgs, trainer):
    """Save the merged run config next to the CSV logger output."""
    if not trainer.loggers:
        return

    log_dir = trainer.loggers[0].log_dir
    os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(config=cfgs, f=os.path.join(log_dir, "resolved_config.yaml"))


def run_mtl_lite(cfgs):
    """Train and test the lightweight multi-task BDI model."""
    data_module = AVECDataModule(cfgs)
    model = MTLLiteDepressionModel(cfgs)

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
    if model.severity_balanced_regression:
        bin_counts = build_train_severity_bin_counts(
            cfgs, edges=tuple(model.severity_bin_edges)
        )
        model.set_severity_bin_counts(bin_counts)
        print(
            f"[RUNNER] severity-balanced regression ON: train bin counts "
            f"{bin_counts}, power={model.severity_power}"
        )

    trainer = build_mtl_lite_trainer(cfgs)
    save_resolved_config(cfgs, trainer)

    print("\n[RUNNER] 正在启动 MTL-Lite 训练引擎...")
    trainer.fit(model, data_module)

    print("\n[RUNNER] 训练结束，正在使用验证集最优 checkpoint 进行 Test 集评估...")
    trainer.test(model, datamodule=data_module, ckpt_path="best")
