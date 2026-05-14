"""
# Author:
# Date:
# File:     train.py
"""
import torch
import os
import sys

# 提示：配置 Tensor Core 加速
torch.set_float32_matmul_precision('medium')

from src.models.extract_features import run_features_extractor
from src.models.temporal import run_temporal_module
from src.paths import *
from omegaconf import OmegaConf

if __name__ == "__main__":

    try:
        print("[INFO] LOADING CONFIG FILE default_config.yaml...")
        config_path = CONFIG_DIR / 'default_config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在，请检查路径: {config_path}")

        with open(CONFIG_DIR / 'default_config.yaml', 'r') as f:
            cfgs = OmegaConf.load(f)

        # stage1 特征提取
        # print("[INFO] START RUNNING FEATURES EXTRACTION STREAM LINE...")
        # run_features_extractor(cfgs)
        # print("✅ Stage 1 ALL WORK DONE WITHOUT ERROR！")

        # stage2 时序处理与任务预测
        print("[INFO] START RUNNING TEMPORAL PROCESS AND TARGET PREDICT...")
        run_temporal_module(cfgs)
        print("✅ Stage 2 ALL WORK DONE WITHOUT ERROR！")

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
        sys.exit(1)