import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.datasets.openface_features import OpenFaceFeatureDataModule
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
