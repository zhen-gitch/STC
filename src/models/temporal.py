import torch
import math
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from src.models.temporal_blocks.bilstm_encoder import BiLSTMEncoder
from src.models.temporal_blocks.transformer_encoder import TransformerEncoderBlock
from src.models.temporal_blocks.mtl_blocks import CGC, SupervisedContrastiveLoss
from src.datasets.dataset import FeatureDataModule
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger


def build_temporal_encoder(model_name: str, input_dim: int, hidden_dim: int):
    """
    时序模型工厂：根据名字自动分配模型积木
    """
    print(f"🧩 [BUILDER] 正在拼装时序编码器: {model_name.upper()} (输入维度: {input_dim})")
    try:
        if model_name.lower() == 'lstm':
            return BiLSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        elif model_name.lower() == 'transformer':
            return TransformerEncoderBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=4,
                num_heads=8)
        else:
            raise ValueError(f"未知的时序模型名称: '{model_name}'")
    except Exception as e:
        print(f"❌ [BUILD ERROR] 模型构建失败！请检查 default_config.yaml 中的 TEMPORAL_MODEL 拼写。")
        raise e


def build_regression_task_head(input_dim: int, hidden_dim: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.Sigmoid(),   # 强制将输出压缩到 0.0 ~ 1.0 之间
    )


def build_classification_task_head(input_dim: int, hidden_dim: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.GELU(),
    )


class TemporalModule(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.cfgs = configs
        self.lr = configs.LEARNING_RATE
        self.weight_decay = configs.WEIGHT_DECAY
        self.step = configs.SAMPLE_STEP

        self.input_dim = configs.TARGET_DIM
        self.hidden_dim = configs.HIDDEN_DIM
        self.expert_dim = configs.EXPERT_DIM    # # CGC 专家的输出维度
        self.contrastive_dim = configs.CONTRASTIVE_DIM  # # 对比学习的嵌入维度

        # self.num_classes = configs.NUM_CLASSES
        self.num_classes = math.ceil(63 / self.step)


        # 特征降维对齐
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        #-------------- 时序网络 ---------------#
        temporal_name = configs.TEMPORAL_MODEL
        self.temporal_encoder = build_temporal_encoder(
            model_name=temporal_name,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim)

        # ============================================
        # ============   CGC + 对比学习   ==============
        # ============================================
        self.cgc_layer = CGC(
            input_dim = self.hidden_dim,
            expert_dim = self.expert_dim,
            num_shared=3,
            num_specific=3,
            dropout=0.2
        )

        #--------------- 任务头 ----------------#
        self.regression_task_head = build_regression_task_head(self.expert_dim, 256, 1)

        self.classification_task_head = build_classification_task_head(self.expert_dim, 256, self.num_classes)

        self.contrastive_task_head = nn.Sequential(
            nn.Linear(self.expert_dim, self.expert_dim),
            nn.GELU(),
            nn.Linear(self.expert_dim, self.contrastive_dim)
        )

        # 损失函数
        self.reg_weight = configs.REG_WEIGHT
        self.cls_weight = configs.CLS_WEIGHT
        self.con_weight = configs.CON_WEIGHT

        self.reg_loss = nn.MSELoss()
        # self.reg_loss = nn.SmoothL1Loss() # 抵抗震荡
        self.cla_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=0.07) # 对比学习损失

        self.pred_list = [[], [], []]
        self.bdi_list = [[], [], []]


    def forward(self, features, mask):
        try:
            # 安全检查，将可能存在的 NaN 和 inf 替换为 0
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            # 强制对特征进行 L2 归一化
            features = F.normalize(features, p=2, dim=-1)

            x = self.proj(features)

            # 统一降采样：创建一个专门的downsampled变量,降采样操作，防止爆显存
            x_down = x[:, ::self.step, :]  # [Batch, 400, 256]
            mask_down = mask[:, ::self.step]  # [Batch, 400]

            # 编码器使用相同的调用方式
            temporal_output = self.temporal_encoder(x_down, mask_down)

            # 掩码平均池化
            # mask_expanded = mask.unsqueeze(-1).float()    # 未降采样的掩码
            mask_expanded = mask_down.unsqueeze(-1).float() # 经过降采样的掩码

            masked_output = temporal_output * mask_expanded
            sum_pooled = torch.sum(masked_output, 1)
            valid_length = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pool_x = sum_pooled / valid_length  # [Batch, hidden_dim]

            # CGC 分流
            reg_features, cls_features, shared_features = self.cgc_layer(pool_x)

            bdi_pred = self.regression_task_head(reg_features).squeeze(-1)
            cls_pred = self.classification_task_head(cls_features)
            con_embeds = self.contrastive_task_head(shared_features)

            # 训练时需要 con_embeds 计算损失，推理时只需要预测值
            return bdi_pred, cls_pred, con_embeds

        except RuntimeError as e:
            print("\n❌ [FORWARD ERROR] 前向传播时发生张量维度不匹配！")
            print(f"👉 当前输入的特征形状: {features.shape}")
            print(f"👉 当前输入的 Mask 形状: {mask.shape}")
            raise e

    def training_step(self, batch, batch_idx):
        try:
            features, masks, labels = batch

            # 前向传播计算预测值
            bdi_preds, cls_preds, con_embeds = self(features, masks)

            # 提取真实标签
            true_bdi = labels['bdi_score']
            # 标签缩放
            true_bdi_scaled = true_bdi / 63.0
            true_cls = labels['class_label']

            self.pred_list[0].append(bdi_preds.detach().cpu())
            self.bdi_list[0].append(true_bdi.detach().cpu())

            # 多任务损失
            # 回归头损失 MSE
            loss_reg = self.reg_loss(bdi_preds, true_bdi_scaled)
            # 分类头损失 Cross Entropy
            loss_cls = self.cla_loss(cls_preds, true_cls)
            # 对比学习损失，使用分类标签进行对比学习，将同类抑郁程度的人脸特征拉近
            loss_con = self.contrastive_loss(con_embeds, true_cls)

            # 计算总损失
            total_loss = self.reg_weight * loss_reg + self.cls_weight * loss_cls + self.con_weight * loss_con

            # 核心防线：检查 Loss 是否崩溃变为 NaN
            if torch.isnan(total_loss):
                print(f"\n⚠️ [WARNING] 第 {batch_idx} 个 Batch 出现 NaN 损失！可能发生了梯度爆炸。")
                return None  # 让 Lightning 跳过这个坏掉的 batch


            self.log("train_loss", total_loss, on_epoch=True)
            self.log("train_reg_loss", loss_reg, on_epoch=True, on_step=False)
            self.log("train_cls_loss", loss_cls, on_epoch=True, on_step=False)
            self.log("train_con_loss", loss_con, on_epoch=True, on_step=False)

            return total_loss

        except Exception as e:
            print(f"\n❌ [TRAIN STEP ERROR] 在训练第 {batch_idx} 个 Batch 时发生严重崩溃！")
            raise e

    def on_train_epoch_end(self):
        if len(self.pred_list[0]) > 0:
            # 拼接所有 batch，并还原到 0~63 的真实分数区间
            preds = torch.cat(self.pred_list[0]) * 63.0
            targets = torch.cat(self.bdi_list[0])

            # 精确计算全局 RMSE 和 MAE
            rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
            mae = torch.mean(torch.abs(preds - targets))

            self.log("train_RMSE_epoch", rmse, prog_bar=True)
            self.log("train_MAE_epoch", mae, prog_bar=True)

            # 清空列表，迎接下一个 Epoch
            self.pred_list[0].clear()
            self.bdi_list[0].clear()

    def validation_step(self, batch, batch_idx):
        features, masks, labels = batch

        # 前向传播计算预测值
        bdi_preds, cls_preds, con_embeds = self(features, masks)

        # 提取真实标签
        true_bdi = labels['bdi_score']
        true_bdi_scaled = true_bdi / 63.0
        true_cls = labels['class_label']

        self.pred_list[1].append(bdi_preds.detach().cpu())
        self.bdi_list[1].append(true_bdi.detach().cpu())

        # 多任务损失
        # 回归头损失 MSE
        loss_reg = self.reg_loss(bdi_preds, true_bdi_scaled)
        # 分类头损失 Cross Entropy
        loss_cls = self.cla_loss(cls_preds, true_cls)
        # 对比学习任务头损失
        loss_con = self.contrastive_loss(con_embeds, true_cls)

        self.log("val_reg_loss", loss_reg)
        self.log("val_cls_loss", loss_cls)
        self.log("val_con_loss", loss_con)

        return None

    def on_validation_epoch_end(self):
        if len(self.pred_list[1]) > 0:
            preds = torch.cat(self.pred_list[1]) * 63.0
            targets = torch.cat(self.bdi_list[1])

            rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
            mae = torch.mean(torch.abs(preds - targets))

            # 这才是你真正能写进论文里的 Validation RMSE / MAE
            self.log("val_RMSE_epoch", rmse, prog_bar=True)
            self.log("val_MAE_epoch", mae, prog_bar=True)

            self.pred_list[1].clear()
            self.bdi_list[1].clear()

    def test_step(self, batch, batch_idx):
        features, masks, labels = batch

        # 前向传播计算预测值
        bdi_preds, cls_preds, con_embeds = self(features, masks)

        true_bdi = labels['bdi_score']
        # ⭐️ 收集测试集数据
        self.pred_list[2].append(bdi_preds.detach().cpu())
        self.bdi_list[2].append(true_bdi.detach().cpu())

        return None

    def on_test_epoch_end(self):
        if len(self.pred_list[2]) > 0:
            preds = torch.cat(self.pred_list[2]) * 63.0
            targets = torch.cat(self.bdi_list[2])

            rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
            mae = torch.mean(torch.abs(preds - targets))

            print(f'\n🎯 [TEST END] 精确 Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}')

            self.pred_list[2].clear()
            self.bdi_list[2].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # OneCycleLR, 目前深度学习收敛最快的调度器
        # 从极小值爬升到 self.lr, 再平滑下降到接近 0
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 5,  # 允许在 Warmup 顶点冲到一个较大的学习率加速收敛
            epochs=self.trainer.max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,  # 前 10% 的时间用于 Warmup 爬升
            anneal_strategy='cos'
        )
        # 余弦退火学习率，让学习率平滑下降
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfgs.MAX_EPOCHS)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # OneCycleLR 必须按 step 更新，而不是 epoch
            }
        }


def run_temporal_module(configs):
    data_module = FeatureDataModule(configs)
    temporal_model = TemporalModule(configs)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator=configs.ACCELERATOR,
        devices=configs.DEVICES,
        precision=configs.PRECISION,
        max_epochs=configs.MAX_EPOCHS,
        callbacks=[RichProgressBar(), lr_monitor],
        logger=[CSVLogger(configs.LOG_DIR + "/" + 'csv_log', name=''), TensorBoardLogger(configs.LOG_DIR + "/" + 'tensorboard_log', name='')],
    )

    print("[BUILDER] 🚀 开始执行时空网络端到端训练！")

    trainer.fit(temporal_model, datamodule=data_module)