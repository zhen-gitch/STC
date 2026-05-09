"""
# name: src/models/stage1.py
# author:
# date:
# description:
"""

import os
# 设置 huggingface_hub 镜像地址，防止 timm 库网络访问出错
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import pytorch_lightning as pl
import timm
import torch
from src.datasets.dataset import AVECDataModule
from pytorch_lightning.callbacks import RichProgressBar


def feature_extractor_model(model_name:str):
    """
    基于 timm 的特征提取器工厂
    :param model_name:
    :return:
    """
    try:
        # num_classes=0 ！会自动剥离所有的分类头（fc层），直接返回特征池化层的结果
        model = timm.create_model(
            model_name=model_name,
            pretrained=True,
            num_classes=0) # 自动剥离所有的分类头（fc层），直接返回特征池化层的结果
        return model
    except RuntimeError as e:
        # 网络或者底层的报错直接抛出
        raise e
    except Exception as e:
        # 捕捉所有的异常并清晰打印
        print(f"❌ [Error] timm 创建模型失败，对于 timm 非法的模型名称{model_name}！")
        raise e

class FeatureExtractor(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.cfgs = configs
        self.model_name = configs.MODEL_NAME
        self.target_embed_dim = configs.TARGET_EMBED_DIM
        self.chunk_size = configs.CHUNK_SIZE
        self.save_dir = Path(configs.FEATURES_DIR).expanduser().resolve()
        os.makedirs(self.save_dir, exist_ok=True)
        # 实例化模型
        self.model = feature_extractor_model(self.model_name)
        # 不论模型原生特征维度是多少都保存，在特征加载时进行映射处理操作


    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        该方法专用于离线推理
        lightning 会自动关闭梯度计算、切换 eval 模式、将数据搬到 GPU
        :param batch:
        :param batch_idx:
        :param dataloader_idx:
        :return:
        """
        # batch 来自 AVECDataset，batch_size = 1
        video_tensor, video_id = batch

        # 维度变换 [batch_size, seq_len, C, H, W] -> [seq_len, C, H, W]
        video_tensor = video_tensor.squeeze(0)

        video_id = video_id[0]

        dataset_names = ['train', 'val', 'test']
        current_dataset_name = dataset_names[dataloader_idx]

        if batch_idx == 0:
            print(f"\n✨ [INFO CHANGE] GETTING【{current_dataset_name.upper()}】DATASET FEATURES，PLEASE WAIT...")

        seq_len = video_tensor.size(0)
        feature_list = []

        # 防止在特征提取时出现 OOM，进行微批次切分，将 batch 划分为 chunk
        for start_idx in range(0, seq_len, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, seq_len)
            chunk = video_tensor[start_idx:end_idx] # [self.chunk_size, 3, 224, 224]

            # 特征提取
            features = self(chunk)

            # 放回 CPU，防止 GPU 显存占用
            feature_list.append(features.cpu())

        # 将所有属于一个样本的特征进行拼接
        sample_feature = torch.cat(feature_list, dim=0)

        # 保存特征
        save_path = Path(self.save_dir).joinpath(self.model_name, current_dataset_name)
        os.makedirs(save_path, exist_ok=True)
        save_path = save_path.joinpath(f"{video_id}.pt")

        torch.save(sample_feature, save_path)

        # 返回地址，用于日志打印
        return save_path


def run_features_extractor(configs):
    data_module = AVECDataModule(configs)
    feature_extractor = FeatureExtractor(configs)

    trainer = pl.Trainer(
        accelerator=configs.ACCELERATOR,
        devices=configs.DEVICES,
        precision=configs.PRECISION,
        callbacks=[RichProgressBar()]
    )

    trainer.predict(feature_extractor, datamodule=data_module)
