"""
# name: /src/datasets/dataset.py
# author:
# function:
# time:
# date:
"""


import json
import os
import torch
import bisect
from PIL import Image
import pytorch_lightning as pl
from omegaconf import OmegaConf  # omegaconf 可以直接使用 . 对 yaml 文件中的元素进行访问
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2  # 视频级的图像增强
from src.paths import *


def load_data_list(dataset_split_file_path:str, images_folder_path:str, dataset):
    """
    # 从包含 train, val, test 三个数据集中所有面部图片数据的文件夹中完成 train, val, test 三个数据集各自文件地址的拼接和存储
    :param dataset_split_file_path:
    :param images_folder_path:
    :param dataset: dataset name
    :return: <class 'list'>
    """
    data = []
    # 加载数据集划分文件
    file_path = Path(dataset_split_file_path).expanduser().resolve()  # 因为本项目在 linux 上运行，使用 expanduser() 对地址中的 ~ 进行正确解析
    if dataset_split_file_path is None:
        pass
    elif not file_path.exists():
        raise FileNotFoundError(f'Dataset split file not found: {file_path}')
    else:
        with open(file_path, 'r') as f:
            full_data_list = json.load(f)
    # 提取数据集划分内容
    target_data_list = full_data_list[dataset]
    for folder in target_data_list:
        full_folder_path = images_folder_path + '/' + folder    # 传递地址字符串，具体解析由后续的 Path() 函数负责
        data.append(full_folder_path)
    return data


class AVECDataset(Dataset):
    """
    用于原始图像数据集的加载、增强等操作
    """
    def __init__(self, configs, dataset:str):
        super(AVECDataset, self).__init__()
        self.IMAGE_DIR = configs.IMAGE_DIR
        self.DATASET_SPLIT_FILE = configs.DATASET_SPLIT_FILE
        self.data = load_data_list(self.DATASET_SPLIT_FILE, self.IMAGE_DIR, dataset)

        self.transform = v2.Compose([
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                         )
        ])


    def __len__(self):
        length = len(self.data)
        return length

    def __getitem__(self, idx):
        video_dir = Path(self.data[idx]).expanduser().resolve()
        video_id = video_dir.name
        # 使用 glob('*.jpg') 直接找到所有 jpg 文件
        # 它返回的直接是完整的绝对路径，不需要再去拼斜杠
        frame_paths = sorted(video_dir.glob('*.jpg'))   # 过滤非图片文件

        frames = []
        for frame_path in frame_paths:
            img = Image.open(Path(frame_path).expanduser()).convert('RGB')
            frames.append(img)

        # 将视频的图片转为张量（不进行截断）
        # 可以直接堆叠，因为 transform 中进行的是相同的操作，不具有随机性，逐张处理和一起处理结果相同
        tensor_frames = [self.transform(img) for img in frames]
        video_tensor = torch.stack(tensor_frames, dim=0)    # 形状：[实际帧数, 3, 224, 224]
        # 返回张量和名字
        return video_tensor, video_id

class AVECDataModule(pl.LightningDataModule):
    def __init__(self, configs):
        super(AVECDataModule, self).__init__()
        self.cfgs = configs
        self.num_workers = configs.NUM_WORKERS
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = AVECDataset(self.cfgs, dataset='train')
        self.val_dataset = AVECDataset(self.cfgs, dataset='val')
        self.test_dataset = AVECDataset(self.cfgs, dataset='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=False)

    def predict_dataloader(self):
        # 将三个数据集打包成列表， Lightning 按顺序自动提取
        return [
            self.train_dataloader(),
            self.val_dataloader(),
            self.test_dataloader()
        ]

class FeatureDataset(Dataset):
    """
    用于所提取特征的加载
    """
    def __init__(self, configs, dataset):
        super(FeatureDataset, self).__init__()
        self.cfgs = configs
        self.MAX_SEQ_LEN = self.cfgs.MAX_SEQ_LEN
        self.FEATURES_DIR = self.cfgs.FEATURES_DIR
        self.LABELS_DIR = self.cfgs.LABEL_DIR
        self.FEATURES_FOLDER_PATH = Path(self.FEATURES_DIR).joinpath(dataset).expanduser().resolve()
        self.data_list = os.listdir(self.FEATURES_FOLDER_PATH)

    def __len__(self):
        length = len(self.data_list)
        return length

    def __getitem__(self, idx):
        feature_path = Path(self.FEATURES_FOLDER_PATH).joinpath(self.data_list[idx]).resolve()
        video_id = feature_path.name
        match_id = video_id[0:5]    # 通过编号查找匹配样本编号

        # 提取 bdi-ii 分数
        with open(Path(self.LABELS_PATH + '/' + match_id + '_Depression.csv').expanduser().resolve(), 'r') as f:
            label = f.read()
            label = int(label)

        # feature 的形状是 [实际帧数，512]
        feature = torch.load(feature_path)

        actual_len = feature.shape[0]
        embed_dim = feature.shape[1]

        # 序列长度对齐与掩码生成 (Padding & Masking)
        if actual_len > self.MAX_SEQ_LEN:
            # 序列过长，截取最大长度
            padded_features = feature[:self.MAX_SEQ_LEN, :]
            # 全为有效数据，Mask 全为 1 (True)
            attention_mask = torch.ones(self.MAX_SEQ_LEN, dtype=torch.bool)
        else:
            # 序列太短，需要补 0
            padding_len = self.MAX_SEQ_LEN - actual_len
            padding_tensor = torch.zeros((padding_len, embed_dim), dtype=torch.float32)

            # 拼接到 MAX_SEQ_LEN
            padded_features = torch.cat([feature, padding_tensor], dim=0)
            # 生成 Mask：前面的实际帧为 1，后面补零的为 0
            attention_mask = torch.cat([
                torch.ones(actual_len, dtype=torch.bool),
                torch.zeros(padding_len, dtype=torch.bool)
            ])

        # 分类标签生成
        # 定义每个区间的右边界（不包含该值本身）
        # 对应: 14以下->0, 20以下->1, 29以下->2, 64以下->3, 64及以上->4
        breakpoints = [14, 20, 29, 64]

        # 实现分类映射
        classes_label = bisect.bisect_right(breakpoints, label)

        # 把两个标签打包成字典返回，方便后续算 Loss
        labels = {
            # 分类任务必须是 torch.long (即 int64)
            "class_label": torch.tensor(classes_label, dtype=torch.long),
            # 回归任务必须是 torch.float32
            "bdi_score": torch.tensor(label, dtype=torch.float32)
        }

        # 返回：对齐后的特征张量, 注意力掩码, 标签字典
        return padded_features, attention_mask, labels


class FeatureDataModule(pl.LightningDataModule):
    def __init__(self, configs):
        super(FeatureDataModule, self).__init__()
        self.cfgs = configs
        self.batch_size = configs.BATCH_SIZE
        self.num_workers = configs.NUM_WORKERS
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = FeatureDataset(self.cfgs, 'train')
        self.val_dataset = FeatureDataset(self.cfgs, 'val')
        self.test_dataset = FeatureDataset(self.cfgs, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)