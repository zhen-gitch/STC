"""
# Author:
# Date:
# File:     train.py
"""
import torch
import sys
import traceback

# 提示：配置 Tensor Core 加速
torch.set_float32_matmul_precision('high')

from src.trainers.end_to_end_runner import run_end2end
from src.paths import *
from omegaconf import OmegaConf
import pytorch_lightning as pl


def load_config(config_path):
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在，请检查路径: {config_path}")
    with open(config_path, 'r') as f:
        return OmegaConf.load(f)


def run_from_config(cfgs):
    mode = str(cfgs.MODE)
    if mode != "full":
        raise ValueError(f"当前重构版本只保留端到端训练入口，MODE 必须为 'full'，当前为: {mode}")

    print("[INFO] START RUNNING END-TO-END STREAM LINE...")
    run_end2end(cfgs)


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    try:
        print("[INFO] LOADING CONFIG FILE default_config.yaml...")
        config_path = CONFIG_DIR / 'default_config.yaml'
        cfgs = load_config(config_path)
        run_from_config(cfgs)

    except torch.cuda.OutOfMemoryError as e:
        print("\n❌ [FATAL ERROR] 显卡内存爆表 (OOM)！")
        print("💡 建议：请去 default_config.yaml 中调小 BATCH_SIZE 或 MAX_SEQ_LEN。")
        sys.exit(1)

    except ValueError as e:
        print(f"\n❌ [VALUE ERROR] 参数或配置错误: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ [UNKNOWN ERROR] 发生未捕获的致命错误，程序被迫中断！")
        print(f"👉 详细错误信息: {e}")
        traceback.print_exc()
        sys.exit(1)
