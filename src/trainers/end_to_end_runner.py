import datetime
import os
import re
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.datasets.dataset import AVECDataModule
from src.models.end_to_end import EndToEndDepressionModel
from src.utils.visualize import plot_training_curves_from_csv


class DualLogger(object):
    """
    同步写入终端和实验日志文件的轻量 stdout/stderr 代理。
    """

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a", encoding='utf-8')
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        self.terminal.write(message)

        clean_msg = self.ansi_escape.sub('', message)
        if '\r' in clean_msg and '\n' not in clean_msg:
            return

        clean_msg = clean_msg.replace('\r', '')
        if clean_msg:
            self.log_file.write(clean_msg)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return True

    @property
    def encoding(self):
        return getattr(self.terminal, 'encoding', 'utf-8')

    def fileno(self):
        return self.terminal.fileno()


def build_end_to_end_trainer(cfgs):
    """Build the Lightning trainer used by the end-to-end experiment."""
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return pl.Trainer(
        accelerator=cfgs.ACCELERATOR,
        devices=cfgs.DEVICES,
        precision=cfgs.PRECISION,
        max_epochs=cfgs.PROCESS_TEMPORAL.MAX_EPOCHS,
        callbacks=[RichProgressBar(), lr_monitor],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        logger=[
            CSVLogger(save_dir=cfgs.LOG_DIR, name='csv_log'),
            TensorBoardLogger(save_dir=cfgs.LOG_DIR, name='tensorboard_log')
        ],
    )


def attach_console_logger(trainer):
    """Mirror console output to the current CSVLogger version directory."""
    if not trainer.logger:
        return

    current_version_dir = trainer.loggers[0].log_dir
    os.makedirs(current_version_dir, exist_ok=True)

    console_log_path = os.path.join(current_version_dir, "console_output.log")
    sys.stdout = DualLogger(console_log_path)
    sys.stderr = sys.stdout

    print("[SYSTEM] 动态日志跟踪已开启！")
    print(f"[SYSTEM] 本次实验的所有控制台输出将自动同步至: {console_log_path}")
    print(f"[SYSTEM] 启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 60)


def run_end2end(cfgs):
    """Train the end-to-end model, reload the best weights, and run test."""
    data_module = AVECDataModule(cfgs)
    model = EndToEndDepressionModel(cfgs)
    trainer = build_end_to_end_trainer(cfgs)
    attach_console_logger(trainer)

    print("\n[RUNNER] 正在启动端到端 (End-to-End) 训练引擎...")
    trainer.fit(model, data_module)

    print("\n[RUNNER] 训练结束，正在从磁盘加载验证集表现最优的权重...")
    best_weights_dir = os.path.join(trainer.loggers[0].log_dir, "weights")

    load_success = model.load_segmented_weights(weights_dir=best_weights_dir, tag="best")
    if load_success:
        print("\n[RUNNER] 最优权重加载完毕，正在进行 Test 集最终评估...")
    else:
        print("\n[WARNING] 权重加载失败，将使用最后的模型参数进行测试...")

    trainer.test(model, datamodule=data_module)
    if trainer.loggers:
        plot_training_curves_from_csv(
            log_dir=trainer.loggers[0].log_dir,
            save_dir=os.path.join(trainer.loggers[0].log_dir, "plots")
        )
