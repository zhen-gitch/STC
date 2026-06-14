import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.datasets.openface_features import OpenFaceFeatureDataModule
from src.diagnostics.io import ensure_dir, write_prediction_table
from src.models.behavior_baseline import BehaviorBaselineModel


def _get_config_value(configs, key, default):
    return getattr(configs, key, default)


def build_behavior_baseline_trainer(cfgs):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_RMSE_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="behavior_baseline-{epoch:03d}-{val_RMSE_epoch:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

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
            CSVLogger(save_dir=cfgs.LOG_DIR, name="behavior_baseline_csv"),
            TensorBoardLogger(save_dir=cfgs.LOG_DIR, name="behavior_baseline_tensorboard"),
        ],
    )


def save_resolved_config(cfgs, trainer):
    if not trainer.loggers:
        return

    log_dir = trainer.loggers[0].log_dir
    os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(config=cfgs, f=os.path.join(log_dir, "resolved_config.yaml"))


def _label_items(labels, key, fallback_key=None):
    values = labels.get(key)
    if values is None and fallback_key is not None:
        values = labels.get(fallback_key)
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if hasattr(values, "detach"):
        return values.detach().cpu().tolist()
    return [str(item) for item in values]


def collect_behavior_predictions(model, data_loader, device):
    model.eval()
    model.to(device)

    all_video_ids = []
    all_subject_ids = []
    all_task_names = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for features, mask, labels in data_loader:
            features = features.to(device)
            mask = mask.to(device)
            outputs = model(features, mask)
            preds = model.prediction_for_metrics(outputs.bdi_pred).detach().cpu().numpy()
            targets = labels["bdi_score"].detach().cpu().numpy()

            all_video_ids.extend(_label_items(labels, "video_id", fallback_key="subject_id"))
            all_subject_ids.extend(_label_items(labels, "subject_id"))
            all_task_names.extend(_label_items(labels, "task_name"))
            all_targets.extend(targets.tolist())
            all_preds.extend(preds.tolist())

    return all_video_ids, all_subject_ids, all_task_names, all_targets, all_preds


def _resolve_export_device(cfgs):
    accelerator = str(getattr(cfgs, "ACCELERATOR", "cpu")).lower()
    if accelerator in {"gpu", "cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_best_weights_if_available(model, checkpoint_path):
    if not checkpoint_path:
        return False
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return True


def export_behavior_predictions(cfgs, model, data_module, trainer):
    if not trainer.loggers:
        return []

    log_dir = Path(trainer.loggers[0].log_dir)
    output_dir = ensure_dir(log_dir / "diagnostics" / "behavior")
    device = _resolve_export_device(cfgs)
    best_checkpoint = getattr(trainer.checkpoint_callback, "best_model_path", "")
    loaded_best = _load_best_weights_if_available(model, best_checkpoint)
    if loaded_best:
        print(f"[RUNNER] Loaded best checkpoint for behavior prediction export: {best_checkpoint}")
    else:
        print("[RUNNER] Best checkpoint was not available; exporting behavior predictions from current model weights.")

    generated = []
    for split, loader in (
        ("val", data_module.val_dataloader()),
        ("test", data_module.test_dataloader()),
    ):
        video_ids, subject_ids, task_names, targets, preds = collect_behavior_predictions(model, loader, device)
        prediction_path = output_dir / f"{split}_predictions.csv"
        write_prediction_table(
            prediction_path,
            subject_ids=subject_ids,
            targets=targets,
            preds=preds,
            video_ids=video_ids,
            task_names=task_names,
        )
        generated.append(prediction_path)
        print(f"[RUNNER] Behavior {split} predictions saved to: {prediction_path}")
    return generated


def run_behavior_baseline(cfgs):
    data_module = OpenFaceFeatureDataModule(cfgs)
    data_module.setup()
    if data_module.feature_dim is None:
        raise ValueError("OpenFaceFeatureDataModule did not infer feature_dim.")

    model = BehaviorBaselineModel(cfgs, input_dim=data_module.feature_dim)
    trainer = build_behavior_baseline_trainer(cfgs)
    save_resolved_config(cfgs, trainer)

    print("\n[RUNNER] 正在启动 Behavior-only baseline 训练引擎...")
    print(f"[RUNNER] OpenFace behavior feature dim: {data_module.feature_dim}")
    trainer.fit(model, data_module)

    print("\n[RUNNER] 训练结束，正在使用验证集最优 checkpoint 进行 Test 集评估...")
    trainer.test(model, datamodule=data_module, ckpt_path="best")
    export_behavior_predictions(cfgs, model, data_module, trainer)
