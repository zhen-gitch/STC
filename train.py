"""
# Author:
# Date:
# File:     train.py
"""
import torch
torch.set_float32_matmul_precision('medium')

from src.models.extract_features import run_features_extractor
from src.paths import *
from omegaconf import OmegaConf

# # 设置 huggingface_hub 镜像地址，防止 timm 库网络访问出错
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("==================================================")
print("🚀 Spatiotemporal CNN (STC) - Stage 1 START")
print("==================================================")
print("[INFO] LOADING CONFIG FILE default_config.yaml...")

with open(CONFIG_DIR / 'default_config.yaml', 'r') as f:
    cfgs = OmegaConf.load(f)

# stage1 特征提取
print("[INFO] START RUNNING FEATURES EXTRACTION STREAM LINE...")
run_features_extractor(cfgs)
print("✅ Stage 1 ALL WORK DONE WITHOUT ERROR！")
