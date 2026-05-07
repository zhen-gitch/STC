import json
import torch
from PIL import Image
from omegaconf import OmegaConf  # omegaconf 可以直接使用 . 对 yaml 文件中的元素进行访问
from torch.utils.data import Dataset
from torchvision.transforms import v2  # 视频级的图像增强
from src.paths import *


def load_split_file(dataset_split_file):
    file_path = Path(dataset_split_file).expanduser().resolve()
    if dataset_split_file is None:
        return None
    elif not file_path.exists():
        raise FileNotFoundError(f'Dataset split file not found: {file_path}')
    else:
        with open(file_path, 'r') as f:
            return json.load(f)


def label_load(label_path):
    """
    从指定地址返回 label
    :param label_path:
    :return: <class 'int'>
    """
    with open(Path(label_path).expanduser(), 'r') as f:
        label = f.read()
        label = int(label)
    return label

def data_list(dataset_split_file_path, images_folder_path, dataset):
    """
    组合指定数据集的地址
    :param dataset_split_file_path:
    :param images_folder_path:
    :param dataset:
    :return: <class 'list'>
    """
    # 完成train, val, test数据集地址的拼接和存储
    data = []
    full_data = load_split_file(dataset_split_file_path)
    target_data = full_data[dataset]
    for folder in target_data:
        full_folder_path = images_folder_path + '/' + folder
        data.append(full_folder_path)
    return data

def transform():
    """"
    
    :return: transform
    """""
    return  v2.Compose([
        v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1),
        # transforms.RandomRotation(degrees=10),  # 输入图像是对齐后的，此处不应该再进行翻转？但翻转有利于增加模型的鲁棒性
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])



class AVECDataset(Dataset):
    """
    用于原始图像数据集的加载、增强等操作
    """
    def __init__(self, configs, data_set_name, mode='train'):
        super(AVECDataset, self).__init__()
        self.cfgs = configs
        self.IMAGE_PATH = self.cfgs.IMAGE_PATH
        self.DATASET_SPLIT_FILE = self.cfgs.DATASET_SPLIT_FILE
        self.data = data_list(self.DATASET_SPLIT_FILE, self.IMAGE_PATH, data_set_name)
        # 对图像进行增强操作
        if mode == 'train':
            self.transform = v2.Compose([
        v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1),
        # transforms.RandomRotation(degrees=10),  # 输入图像是对齐后的，此处不应该再进行翻转？但翻转有利于增加模型的鲁棒性
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
        else:
            self.transform = v2.Compose([
                v2.Resize((224, 224), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


    def __len__(self):
        length = len(self.data)
        return length

    def __getitem__(self, idx):
        video_id = self.data[idx]
        video_dir = Path(self.data[idx]).expanduser()
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

class FeatureDataset(Dataset):
    """
    用于所提取特征的加载
    """
    def __init__(self, configs):
        super(FeatureDataset, self).__init__()
        self.cfgs = configs



if __name__ == '__main__':
    with open(CONFIG_DIR / 'default_config.yaml', 'r') as f:
        cfgs = OmegaConf.load(f)
    train_set = AVECDataset(cfgs, "train", mode="train")
    print(type(train_set))
