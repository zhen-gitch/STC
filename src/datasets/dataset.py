"""
# name:     /src/datasets/dataset.py
# author:
# function:
# time:
# date:
"""


import bisect
import json
import random
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2  # 视频级的图像增强


def _get_config_value(configs, section_name, key, default):
    section = getattr(configs, section_name, None)
    if section is None:
        return default
    return getattr(section, key, default)


def load_data_list(dataset_split_file_path:str, images_folder_path:str, dataset):
    """Return absolute video-folder paths for the requested split."""
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

def generate_soft_spatial_mask(h=112, w=112, center_y=0.45, center_x=0.5, sigma_y=0.35, sigma_x=0.35, device='cpu'):
    """
    自适应生成一幅 2D 软性椭圆空间掩码矩阵 [H, W]
    center_y, center_x: 椭圆中心点位置（基于归一化比率，0.45 刚好对准眼睛和中庭）
    sigma_y, sigma_x: 控制横向和纵向的扩散边缘平滑度（值越大，保留范围越广；值越小，越聚焦中央）
    """
    # 1. 建立 2D 栅格坐标系
    y = torch.linspace(-1.0, 1.0, steps=h, device=device)
    x = torch.linspace(-1.0, 1.0, steps=w, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    # 2. 将中心点转换到 [-1, 1] 空间
    c_y = (center_y * 2.0) - 1.0
    c_x = (center_x * 2.0) - 1.0

    # 3. 计算二维高斯流形
    exponent = - (((grid_y - c_y) ** 2) / (2 * (sigma_y ** 2)) +
                  ((grid_x - c_x) ** 2) / (2 * (sigma_x ** 2)))
    spatial_mask = torch.exp(exponent)

    # 4. 归一化，确保中心核心区域最大引力为 1.0
    spatial_mask = spatial_mask / (spatial_mask.max() + 1e-8)

    return spatial_mask  # 返回形状: [112, 112]
# src/datasets/dataset.py

class AVECDataset(Dataset):
    """AVEC video dataset for end-to-end training.

    The training split returns three synchronized views of each video:
    an unaugmented anchor view and two independently augmented views for
    contrastive learning. Validation and test splits return a single clean view.
    """

    def __init__(self, configs, dataset: str):
        super(AVECDataset, self).__init__()
        self.LABEL_DIR = configs.LABEL_DIR
        self.IMAGE_DIR = configs.IMAGE_DIR
        self.DATASET_SPLIT_FILE = configs.DATASET_SPLIT_FILE
        self.data = load_data_list(self.DATASET_SPLIT_FILE, self.IMAGE_DIR, dataset)
        self.class_step = configs.PROCESS_TEMPORAL.CLASS_STEP
        self.sample_step = configs.PROCESS_TEMPORAL.SAMPLE_STEP
        self.max_len = int(configs.PROCESS_TEMPORAL.MAX_SEQ_LEN // self.sample_step)
        self.dataset_name = dataset
        self.return_multi_view_train = bool(
            _get_config_value(configs, "DATASET", "RETURN_MULTI_VIEW_TRAIN", True)
        )

        # Base transform must stay deterministic; random augmentations are applied
        # only in _apply_video_augmentation for the training views.
        self.transform = v2.Compose([
            v2.Resize((112, 112), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _apply_video_augmentation(self, raw_frames_tensor):
        """Apply one spatial augmentation consistently to every frame in a video."""
        v_tensor = self.transform(raw_frames_tensor)

        # Apply the same horizontal flip to all frames to avoid temporal flicker.
        if random.random() > 0.5:
            v_tensor = torch.flip(v_tensor, dims=[-1])

        # Apply a mild shared affine transform to reduce static identity cues.
        if random.random() > 0.5:
            angle = random.uniform(-4, 4)
            translations = (random.randint(-4, 4), random.randint(-4, 4))
            v_tensor = v2.functional.affine(
                v_tensor, angle=angle, translate=translations,
                scale=1.0, shear=0.0, interpolation=v2.functional.InterpolationMode.BILINEAR
            )
        return v_tensor

    def _select_frame_paths(self, video_dir):
        frame_paths = sorted(video_dir.glob('*.jpg'))
        frame_paths = frame_paths[::self.sample_step]
        actual_len = min(len(frame_paths), self.max_len)
        frame_paths = frame_paths[:self.max_len]

        mask = torch.cat([
            torch.ones(actual_len, dtype=torch.bool),
            torch.zeros(self.max_len - actual_len, dtype=torch.bool)
        ])
        return frame_paths, mask

    def _read_video_frames(self, frame_paths):
        tensor_frames = []
        for frame_path in frame_paths:
            img_tensor = read_image(str(frame_path), mode=ImageReadMode.RGB)
            tensor_frames.append(img_tensor)
        return torch.stack(tensor_frames, dim=0)

    def _pad_video(self, video_tensor):
        pad_len = self.max_len - video_tensor.size(0)
        if pad_len <= 0:
            return video_tensor

        _, C, H, W = video_tensor.shape
        padding_frames = torch.zeros((pad_len, C, H, W), dtype=video_tensor.dtype)
        return torch.cat((video_tensor, padding_frames), dim=0)

    def _build_video_output(self, raw_video_tensor):
        if self.dataset_name == "train" and self.return_multi_view_train:
            return {
                "orig": self._pad_video(self.transform(raw_video_tensor.clone())),
                "v1": self._pad_video(self._apply_video_augmentation(raw_video_tensor.clone())),
                "v2": self._pad_video(self._apply_video_augmentation(raw_video_tensor)),
            }

        return self._pad_video(self.transform(raw_video_tensor))

    def _label_value_for_video_id(self, video_id):
        match_id = video_id[0:5]
        label_path = Path(self.LABEL_DIR).joinpath(f'{match_id}_Depression.csv').expanduser()
        with open(label_path, 'r') as f:
            return match_id, int(f.read().strip())

    def _load_labels(self, video_id):
        match_id, label = self._label_value_for_video_id(video_id)

        breakpoints = [score for score in range(self.class_step, 64, self.class_step)]
        classes_label = bisect.bisect_right(breakpoints, label)

        return {
            "class_label": torch.tensor(classes_label, dtype=torch.long),
            "bdi_score": torch.tensor(label, dtype=torch.float32),
            "subject_id": match_id
        }

    def iter_bdi_scores(self):
        """Yield BDI labels without loading video frames.

        This is used by LDS label weighting and avoids a full image pass before
        training starts.
        """
        for video_dir in self.data:
            video_id = Path(video_dir).expanduser().resolve().name
            _, label = self._label_value_for_video_id(video_id)
            yield label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_dir = Path(self.data[idx]).expanduser().resolve()
        video_id = video_dir.name
        frame_paths, mask = self._select_frame_paths(video_dir)

        raw_video_tensor = self._read_video_frames(frame_paths)  # [S, 3, H, W]
        video_output = self._build_video_output(raw_video_tensor)

        labels = self._load_labels(video_id)
        return video_output, mask, labels


class AVECDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for the end-to-end AVEC pipeline."""
    def __init__(self, configs):
        super(AVECDataModule, self).__init__()
        self.cfgs = configs
        self.num_workers = configs.EXTRACT_FEATURE.NUM_WORKERS
        self.batch_size = configs.EXTRACT_FEATURE.BATCH_SIZE
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = AVECDataset(self.cfgs, dataset='train')
        self.val_dataset = AVECDataset(self.cfgs, dataset='val')
        self.test_dataset = AVECDataset(self.cfgs, dataset='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          prefetch_factor=4,
                          persistent_workers=True,
                          drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          prefetch_factor=4,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=False,
                          prefetch_factor=2)

