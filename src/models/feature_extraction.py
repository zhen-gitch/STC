import os
import torch
from omegaconf import OmegaConf
from src.paths import *
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm  # 进度条神器
from src.datasets.dataset import AVECDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_and_save_features(configs, dataset_name, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备模型 (只提特征，去掉分类头)
    resnet = models.resnet18(pretrained=True)
    cnn = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    cnn.eval()  # ⚠️ 极其重要：必须设置为评估模式，否则 BatchNorm 会捣乱！

    # 2. 准备数据 (Batch Size 必须设为 1)
    # 因为每个视频的帧数(Seq_Len)不一样，不能把不同视频打包进同一个 Batch
    dataset = AVECDataset(configs, dataset_name, mode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    # 特征保存目录
    save_dir = f'~/dataset/depression/avec/2014/features/{mode}'
    os.makedirs(save_dir, exist_ok=True)

    # 防 OOM 的微批次大小 (如果显存大可以设为 128，显存小设为 32)
    chunk_size = 32

    print("🚀 开始提取特征...")
    with torch.no_grad():  # ⚠️ 极其重要：不计算梯度，省下 70% 显存
        for video_tensor, video_id in tqdm(dataloader):
            # video_tensor 形状: [1, Seq_Len, 3, 224, 224] (因为 batch_size=1, 多了第一维)
            # 把第一维去掉，变成 [Seq_Len, 3, 224, 224]
            video_tensor = video_tensor.squeeze(0).to(device)
            video_id = video_id[0]  # 取出字符串

            seq_len = video_tensor.shape[0]
            feature_list = []

            # 3. ⭐️ 核心：在时间序列上切块送入模型
            for start_idx in range(0, seq_len, chunk_size):
                end_idx = min(start_idx + chunk_size, seq_len)

                # 截取一小块：[64, 3, 224, 224]
                chunk = video_tensor[start_idx:end_idx]

                # 提取特征
                features = cnn(chunk)  # 输出: [64, 512, 1, 1]
                features = features.view(features.size(0), -1)  # 展平为 [64, 512]

                # 放回 CPU，防止 GPU 显存被屯满
                feature_list.append(features.cpu())

                # 4. 把切块提出来的特征重新拼成完整的视频特征
            # 最终形状恢复为: [Seq_Len, 512]
            final_video_feature = torch.cat(feature_list, dim=0)

            # 5. 保存为 .pt 文件
            video_full_path = video_id.split('/')
            video_id = video_full_path[-1]
            save_path = os.path.join(save_dir, f"{video_id}.pt")
            save_path = Path(save_path).expanduser()
            torch.save(final_video_feature, save_path)

    print("✅ 所有视频特征提取完毕！")



    """
    记得检查是不是所有样本的所有特征都在
    """


if __name__ == "__main__":
    with open(CONFIG_DIR / 'default_config.yaml', 'r') as f:
        cfgs = OmegaConf.load(f)
    train_set = AVECDataset(cfgs, "test", mode="test")

    extract_and_save_features(cfgs, "test", "test")