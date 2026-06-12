import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.datasets.dataset import AVECDataModule
from src.models.mtl_lite import MTLLiteDepressionModel


def _get_config_value(configs, key, default):
    return getattr(configs, key, default)


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
            CSVLogger(save_dir=cfgs.LOG_DIR, name="mtl_lite_csv"),
            TensorBoardLogger(save_dir=cfgs.LOG_DIR, name="mtl_lite_tensorboard"),
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
    trainer = build_mtl_lite_trainer(cfgs)
    save_resolved_config(cfgs, trainer)

    print("\n[RUNNER] 正在启动 MTL-Lite 训练引擎...")
    trainer.fit(model, data_module)

    print("\n[RUNNER] 训练结束，正在使用验证集最优 checkpoint 进行 Test 集评估...")
    trainer.test(model, datamodule=data_module, ckpt_path="best")
