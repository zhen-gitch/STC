import torch
import numpy as np
import json
from omegaconf import OmegaConf
from src.paths import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class AVECDataset(Dataset):
    def __init__(self, cfgs, transform=None):
        super(AVECDataset, self).__init__()
        self.cfgs = cfgs
        self.Image_path = self.cfgs.Image_path
        self.Label_path = self.cfgs.Label_path
        self.Dataset_split_file = self.cfgs.Dataset_split_file

        # 查看样本属于哪个数据集，train, val or test
        dataset_split = self.load_split_file()
        train_set = dataset_split['train']
        val_set = dataset_split['val']
        test_set = dataset_split['test']

        # 对图像进行增强操作
        if transform is None:
            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        else:
            self.transform = transform

    def load_split_file(self):
        if self.Dataset_split_file is None:
            return None
        else:
            split_file = Path(self.Dataset_split_file).expanduser().resolve()   # python 中的 open 不认识 linux 中的 ~，使用 expanduser 进行展开
            with open(split_file, 'r') as f:
                return json.load(f)

    def image_load(self):
        pass

    def label_load(self):
        pass

    def transform(self):
        pass


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    with open(CONFIG_DIR / 'default_config.yaml', 'r') as f:
        cfgs = OmegaConf.load(f)
    dataset = AVECDataset(cfgs)