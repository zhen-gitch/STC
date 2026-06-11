"""
file:       /src/models/end_to_end.py
author:
"""
import os
import warnings

# 在导入深度学习框架前限制 TensorFlow 日志，减少控制台噪声。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # 只显示 Fatal 级别错误 (1=INFO, 2=WARNING, 3=ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     # 关闭 oneDNN 的浮点舍入差异提示

import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from omegaconf import OmegaConf
from src.datasets.dataset import generate_soft_spatial_mask
from src.models.backbone_factory import build_feature_backbone
from src.models.task_heads import build_regression_task_head, build_classification_task_head, build_contrastive_task_head
from src.models.task_heads import get_coral_levels, coral_loss
from src.utils.adaptive_mask import FineGrainedChannelAdaptiveMask
from src.utils.decomposition import DeepDecompositionEncoder, TemporalGaussianFilter
from src.models.mtl_blocks import CGC
from src.utils.PCGrad import pcgrad_backward
from src.utils.label_distribution import compute_bdi_loss_weights
from src.losses.losses import CrossSubjectContinuousContrastiveLoss
from src.metrics.metrics import ConcordanceCorrCoefMetric, concordance_ccc_loss
from src.utils.visualize import (
    plot_embedding_diagnostics,
    plot_regression_diagnostics,
    plot_temporal_gating_summary,
    plot_temporal_gating_weights,
)
from torch.utils.checkpoint import checkpoint  # 官方检查点引擎

warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

class EndToEndDepressionModel(pl.LightningModule):
    """End-to-end depression severity model for AVEC-style face videos.

    Pipeline:
        video frames -> visual backbone -> temporal filtering/decomposition ->
        masked pooling -> CGC multi-task experts -> regression/classification/
        contrastive heads.

    The model uses manual optimization because PCGrad needs to compute and
    project gradients from multiple task losses before the optimizer step.
    """

    def __init__(self, configs):
        super().__init__()
        self.cfgs = configs
        self.ef_lr = configs.EXTRACT_FEATURE.LEARNING_RATE
        self.tmp_lr = configs.PROCESS_TEMPORAL.LEARNING_RATE
        self.ef_weight_decay = configs.EXTRACT_FEATURE.WEIGHT_DECAY
        self.ef_model_name = configs.EXTRACT_FEATURE.MODEL_NAME
        self.ef_weight_path = configs.EXTRACT_FEATURE.MODEL_WEIGHT_PATH

        self.backbone_base_lr = self.ef_lr

        self.best_val_rmse = float('inf')  # 追踪历史最优 RMSE
        self.max_epochs = configs.PROCESS_TEMPORAL.MAX_EPOCHS
        self.freeze_epochs = configs.PROCESS_TEMPORAL.FREEZE_EPOCHS
        self.warmup_epochs = configs.PROCESS_TEMPORAL.WARMUP_EPOCHS
        self.adversarial_mask_epochs = configs.PROCESS_TEMPORAL.ADVERSARIAL_MASK

        # 核心优化：用原生的 torchmetrics 代替不稳健的 Python 列表追加，覆盖所有运行生命周期
        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.train_ccc = ConcordanceCorrCoefMetric()
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_ccc = ConcordanceCorrCoefMetric()
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_ccc = ConcordanceCorrCoefMetric()

        # =================================================
        #  Backbone 特征提取阶段
        # =================================================
        self.backbone = build_feature_backbone(self.ef_model_name, self.ef_weight_path)

        # =================================================
        # 时序与多任务头
        # =================================================
        self.hidden_dim = configs.PROCESS_TEMPORAL.HIDDEN_DIM
        self.input_dim = configs.BACKBONE_OUT_DIMS.get(self.ef_model_name)
        # 特征对齐
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        self.kernel_size = configs.PROCESS_TEMPORAL.KERNEL_SIZE
        self.sigma = configs.PROCESS_TEMPORAL.SIGMA
        # 初始化滤波器（低通）
        # kernel_size=5 代表考虑前后 2 帧的上下文(前后各 2 帧加自身共 5 帧)，sigma 越大平滑力度越强
        # kernel_size为 0 时不使用
        self.temporal_filter = TemporalGaussianFilter(channels=self.hidden_dim, kernel_size=self.kernel_size, sigma=self.sigma)

        # 时序网络
        self.moving_avg_kernels = configs.PROCESS_TEMPORAL.MOVING_AVG_KERNELS
        self.gate_temperature = float(getattr(configs.PROCESS_TEMPORAL, "GATE_TEMPERATURE", 2.0))
        self.temporal_encoder = DeepDecompositionEncoder(
            in_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            moving_avg_kernels=self.moving_avg_kernels,
            gate_temperature=self.gate_temperature
        )

        # CGC 多任务层
        self.num_shared_gen = configs.PROCESS_TEMPORAL.NUM_SHARED_GEN
        self.num_shared_con = configs.PROCESS_TEMPORAL.NUM_SHARED_CON
        self.num_specific = configs.PROCESS_TEMPORAL.NUM_SPECIFIC
        self.expert_dim = configs.PROCESS_TEMPORAL.EXPERT_DIM
        self.dropout = configs.PROCESS_TEMPORAL.DROPOUT
        self.cgc_layer = CGC(
            input_dim=self.hidden_dim,
            expert_dim=self.expert_dim,
            num_shared_gen=self.num_shared_gen,
            num_shared_con=self.num_shared_con,
            num_specific=self.num_specific,
            dropout=self.dropout
        )

        # 特征增强，增强泛化性，对抗过拟合
        self.base_mask_prob = float(configs.PROCESS_TEMPORAL.BASE_MASK_PROB)
        self.max_mask_prob = float(configs.PROCESS_TEMPORAL.MAX_MASK_PROB)
        self.fine_grained_mask = FineGrainedChannelAdaptiveMask(
            base_mask_prob=self.base_mask_prob,
            max_mask_prob=self.max_mask_prob,
            share_feature_mask=True,  # 推荐 True，更稳定
            use_temporal_mask=False
        )
        # ====================================================
        # 对抗Mask缓存：避免每个batch都做一次额外反向传播
        # ====================================================
        self.mask_update_interval = configs.PROCESS_TEMPORAL.MASK_UPDATE_INTERVAL
        self._cached_f_mask = None
        self._cached_f_scale = None
        self._spatial_gate_cache = {}

        # 初始化任务头
        self.max_score = configs.EXTRACT_FEATURE.MAX_SCORE
        self.class_step = configs.PROCESS_TEMPORAL.CLASS_STEP
        self.num_classes = math.ceil(self.max_score / self.class_step)
        self.contrastive_dim = configs.PROCESS_TEMPORAL.CONTRASTIVE_DIM

        self.reg_task_head = build_regression_task_head(self.expert_dim, self.hidden_dim, 1)
        self.cls_task_head = build_classification_task_head(self.expert_dim, self.hidden_dim, self.num_classes)
        self.con_task_head = build_contrastive_task_head(self.expert_dim, self.hidden_dim, self.contrastive_dim)

        self.automatic_optimization = False  # 必须关闭自动优化，以便使用 PCGrad 进行手动 step

        # 初始化 UW（不确定性权重）相关的开关和参数
        self.use_uw = configs.PROCESS_TEMPORAL.USE_UNCERTAINTY_WEIGHT
        self.log_vars = nn.Parameter(torch.zeros(3))  # 不确定性权重参数, 必须用 nn.Parameter 包裹，让优化器自动更新

        # 是否使用PCGrad
        self.use_pcgrad = configs.PROCESS_TEMPORAL.USE_PCGRAD

        # 损失函数
        self.temperature = configs.PROCESS_TEMPORAL.TEMPERATURE
        # self.reg_loss_beta = 3.0 / float(self.max_score)
        # self.reg_loss = nn.SmoothL1Loss(beta=self.reg_loss_beta)
        self.reg_loss = nn.MSELoss()
        self.ccc_loss_weight = float(getattr(configs.PROCESS_TEMPORAL, "CCC_LOSS_WEIGHT", 0.0))
        self.pred_mean_loss_weight = float(getattr(configs.PROCESS_TEMPORAL, "PRED_MEAN_LOSS_WEIGHT", 0.0))
        self.pred_std_loss_weight = float(getattr(configs.PROCESS_TEMPORAL, "PRED_STD_LOSS_WEIGHT", 0.0))
        self.gate_entropy_weight = float(getattr(configs.PROCESS_TEMPORAL, "GATE_ENTROPY_WEIGHT", 0.01))
        self.gate_balance_weight = float(getattr(configs.PROCESS_TEMPORAL, "GATE_BALANCE_WEIGHT", 0.0))
        self.lds_sigma = float(getattr(configs.PROCESS_TEMPORAL, "LDS_SIGMA", 2.0))
        self.lds_severity_alpha = float(getattr(configs.PROCESS_TEMPORAL, "LDS_SEVERITY_ALPHA", 0.8))

        # 引入跨受试者连续损失
        self.base_bdi_sigma = configs.PROCESS_TEMPORAL.BDI_SIGMA
        self.bdi_sigma = self.base_bdi_sigma

        # 全局表征体检缓存容器
        self.val_features_storage = []
        self.val_subjects_storage = []
        self.val_scores_storage = []
        self.val_pred_storage = []
        self.val_target_storage = []
        self.test_pred_storage = []
        self.test_target_storage = []
        self.test_subjects_storage = []
        self.val_trend_weights_storage = []
        self.val_seasonal_weights_storage = []

        viz_cfg = getattr(configs, "VISUALIZATION", {})
        self.enable_visualizations = bool(getattr(viz_cfg, "ENABLE", True))
        self.regression_visualization_interval = int(getattr(viz_cfg, "REGRESSION_INTERVAL", 1))
        self.embedding_visualization_interval = int(getattr(viz_cfg, "EMBEDDING_INTERVAL", 5))
        self.gating_visualization_interval = int(getattr(viz_cfg, "GATING_INTERVAL", 5))

        # ====================================================
        # 初始 LDS 权重为全 1；on_train_start 中会根据训练集 BDI 分布更新为 LDS 权重
        # 确保全网在任何开局和推理阶段（如 diagnose.py）都不会报属性缺失错误
        # ====================================================
        self.register_buffer("bdi_loss_weights", torch.ones(self.max_score + 1).float())

    def _get_spatial_gate(self, height, width, device, dtype):
        """Return a cached soft face-region gate for the current frame shape."""
        cache_key = (height, width, str(device), str(dtype))
        if cache_key in self._spatial_gate_cache:
            return self._spatial_gate_cache[cache_key]

        spatial_gate = generate_soft_spatial_mask(
            h=height,
            w=width,
            center_y=0.45,
            center_x=0.5,
            sigma_y=0.38,
            sigma_x=0.35,
            device=device
        ).to(dtype=dtype).view(1, 1, 1, height, width)
        self._spatial_gate_cache[cache_key] = spatial_gate
        return spatial_gate

    def _should_visualize(self, interval):
        """Return whether the current validation epoch should emit a plot."""
        if not self.enable_visualizations:
            return False
        if interval <= 0:
            return False
        is_last_epoch = self.current_epoch == self.trainer.max_epochs - 1
        return self.current_epoch % interval == 0 or is_last_epoch

    def extract_visual_features(self, video_tensor, masks):
        """Extract backbone features from valid frames only.

        Args:
            video_tensor: Tensor with shape ``[B, S, C, H, W]``.
            masks: Boolean tensor with shape ``[B, S]`` where True marks a real
                frame and False marks padding.

        Returns:
            Feature tensor with shape ``[B, S, backbone_dim]``. Padding frames
            are filled with zeros.
        """
        B, S, C, H, W = video_tensor.shape

        spatial_gate = self._get_spatial_gate(H, W, video_tensor.device, video_tensor.dtype)
        video_tensor_gated = video_tensor * spatial_gate

        x_flat = video_tensor_gated.reshape(B * S, C, H, W)

        valid_indices = masks.view(-1).bool()
        valid_x = x_flat[valid_indices]

        chunk_size = self.cfgs.EXTRACT_FEATURE.CHUNK_SIZE
        valid_features_list = []
        n_frames = valid_x.size(0)

        if n_frames == 0:
            return torch.zeros(B, S, self.input_dim, device=video_tensor.device, dtype=video_tensor.dtype)

        is_grad_enabled = torch.is_grad_enabled()
        if self.training and self.current_epoch >= self.freeze_epochs:
            self.backbone.train()
        else:
            self.backbone.eval()

        backbone_requires_grad = any(p.requires_grad for p in self.backbone.parameters())

        i = 0
        while i < n_frames:
            end_idx = i + chunk_size
            if n_frames - end_idx == 1:
                end_idx = n_frames

            chunk = valid_x[i:end_idx].clone()
            bn_states = []
            if chunk.size(0) == 1:
                for m in self.backbone.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        bn_states.append((m, m.training))
                        m.eval()

            # Checkpoint only when the backbone participates in gradient updates.
            if is_grad_enabled and backbone_requires_grad:
                chunk_feat = checkpoint(self.backbone, chunk, use_reentrant=False)
            else:
                with torch.no_grad():
                    chunk_feat = self.backbone(chunk)

            valid_features_list.append(chunk_feat)

            for m, state in bn_states:
                if state:
                    m.train()
            i = end_idx

        valid_features_flat = torch.cat(valid_features_list, dim=0)
        features_flat = torch.zeros(B * S, self.input_dim, device=video_tensor.device, dtype=valid_features_flat.dtype)
        features_flat[valid_indices] = valid_features_flat
        return features_flat.reshape(B, S, -1)

    def _forward_features_to_cgc_tracks(
            self,
            single_video_tensor,
            masks,
            true_bdi_norm=None,
            forced_f_mask=None,
            batch_idx=None
    ):
        """
        Convert a video view into task-specific CGC feature pools.
        """
        B = single_video_tensor.size(0)

        # Empty-mask batches can appear in edge cases; return zero features
        # instead of running invalid pooling.
        if masks.sum() == 0:
            zero_pool = torch.zeros(B, self.expert_dim, device=single_video_tensor.device,
                                    dtype=single_video_tensor.dtype)
            return zero_pool, zero_pool, zero_pool, None

        features = self.extract_visual_features(single_video_tensor, masks)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        aligned_features = self.proj(features)
        if self.kernel_size != 0:
            aligned_features = self.temporal_filter(aligned_features)

        f_mask_out = None

        use_adv_mask = (
            self.training
            and self.current_epoch >= self.adversarial_mask_epochs
            and true_bdi_norm is not None
        )

        if use_adv_mask:
            if forced_f_mask is not None:
                aligned_final = aligned_features * forced_f_mask
                f_mask_out = forced_f_mask

            else:
                should_update_mask = (
                    self._cached_f_mask is None
                    or batch_idx is None
                    or batch_idx % self.mask_update_interval == 0
                )

                if should_update_mask:
                    features_for_grad = aligned_features.detach().clone().requires_grad_(True)

                    def aux_loss_fn(bdi_pred):
                        return self.reg_loss(bdi_pred.float(), true_bdi_norm.float())

                    _, _, _, prob_f = self.fine_grained_mask.compute_masks(
                        features=features_for_grad,
                        auxiliary_loss_fn=aux_loss_fn,
                        model=self,
                        mask=masks
                    )

                    warmup_denominator = max(1, self.warmup_epochs)
                    warmup_ratio = min(
                        1.0,
                        float(self.current_epoch - self.adversarial_mask_epochs + 1)
                        / float(warmup_denominator)
                    )

                    prob_f = torch.clamp(
                        prob_f * warmup_ratio,
                        min=0.0,
                        max=self.max_mask_prob
                    )

                    f_mask = (
                        torch.rand_like(prob_f) > prob_f
                    ).float().detach()

                    scale_factor = torch.clamp(
                        1.0 / ((1.0 - prob_f) + 1e-8),
                        max=3.0
                    ).detach()

                    self._cached_f_mask = f_mask
                    self._cached_f_scale = scale_factor

                aligned_final = aligned_features * self._cached_f_mask * self._cached_f_scale
                f_mask_out = self._cached_f_mask * self._cached_f_scale

        else:
            aligned_final = aligned_features

        time_seq_feature = self.temporal_encoder(aligned_final, masks)
        time_feature_pool = self.encode_and_pool(time_seq_feature, masks)
        reg_pool, cls_pool, con_pool = self.cgc_layer(time_feature_pool)

        return reg_pool, cls_pool, con_pool, f_mask_out

    def encode_and_pool(self, seq_features, mask):
        """Pool temporal features with both level and variation statistics."""
        mask_expanded = mask.unsqueeze(-1).float()
        masked_output = seq_features * mask_expanded

        mean_pool = torch.sum(masked_output, 1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        variance = torch.sum(((masked_output - mean_pool.unsqueeze(1)) * mask_expanded) ** 2, dim=1) / torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9)
        std_pool = torch.sqrt(variance + 1e-6)

        return mean_pool + std_pool

    def forward(self, video_tensor, mask, need_all_heads=True, true_bdi_norm=None, forced_f_mask=None):
        """Run the shared end-to-end inference path.

        Set ``need_all_heads=False`` during final test inference when only the
        BDI regression output is needed.
        """
        reg_pool, cls_pool, con_pool, _ = self._forward_features_to_cgc_tracks(
            video_tensor, mask, true_bdi_norm=true_bdi_norm, forced_f_mask=forced_f_mask
        )

        bdi_pred = self.reg_task_head(reg_pool).squeeze(-1)

        cls_pred = None
        con_embeds = None

        if need_all_heads:
            if hasattr(self, "cls_task_head") and self.cls_task_head is not None:
                cls_pred = self.cls_task_head(cls_pool)
            if hasattr(self, "con_task_head") and self.con_task_head is not None:
                con_embeds = self.con_task_head(con_pool)

        return bdi_pred, cls_pred, con_embeds

    def pcgrad_shared_parameters(self):
        """
        PCGrad只应该处理共享层参数。
        任务专属Head不参与梯度投影，否则会削弱各任务自己的学习能力。
        """
        params = []

        params += list(self.backbone.parameters())
        params += list(self.proj.parameters())
        params += list(self.temporal_filter.parameters())
        params += list(self.temporal_encoder.parameters())
        params += list(self.cgc_layer.parameters())

        return [p for p in params if p.requires_grad]

    def save_segmented_weights(self, tag="best"):
        if not self.trainer.is_global_zero or not self.trainer.logger:
            return

        # 动态获取当前实验的版本文件夹 (如 logs/csv_log/version_0)
        log_dir = self.trainer.loggers[0].log_dir
        save_path = os.path.join(log_dir, "weights")
        os.makedirs(save_path, exist_ok=True)

        # 保存 Backbone (特征提取阶段)
        torch.save(self.backbone.state_dict(), os.path.join(save_path, f"{tag}_backbone.pth"))

        other_weights = {
            'proj': self.proj.state_dict(),
            'cgc_layer': self.cgc_layer.state_dict(),
            'temporal_encoder': self.temporal_encoder.state_dict(),
            'task_heads': {
                'reg': self.reg_task_head.state_dict(),
                'cls': self.cls_task_head.state_dict(),
                'con': self.con_task_head.state_dict()
            },
            'log_vars': self.log_vars if self.use_uw else None
        }
        torch.save(other_weights, os.path.join(save_path, f"{tag}_temporal_heads.pth"))

    def on_train_start(self):
        """训练开始前，将当前配置备份到日志文件夹"""

        if self.trainer.loggers:
            log_dir = self.trainer.loggers[0].log_dir
        else:
            log_dir = self.cfgs.LOG_DIR

        os.makedirs(str(log_dir), exist_ok=True)

        config_path = os.path.join(str(log_dir), "model_config.yaml")
        OmegaConf.save(self.cfgs, config_path)
        print(f"[CONFIG] 实验配置已备份至: {config_path}")

        # ====================================================
        # 利用 PL 运行时上下文，全自动安全计算 LDS 连续标签权重
        # ====================================================
        try:
            # 从 PyTorch Lightning 运行时上下文读取已绑定的数据模块。
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                dm = self.trainer.datamodule

                if hasattr(dm, 'train_dataset') and dm.train_dataset is not None:
                    print("\n[LDS ENGINE] 已读取 PyTorch Lightning 数据模块，正在估计训练集连续标签分布...")

                    train_bdi_scores = list(dm.train_dataset.iter_bdi_scores())

                    lds_weights = compute_bdi_loss_weights(
                        train_bdi_scores,
                        self.max_score,
                        sigma=self.lds_sigma,
                        severity_alpha=self.lds_severity_alpha
                    )
                    self.bdi_loss_weights.copy_(lds_weights.to(self.bdi_loss_weights.device))

                    print(
                        f"[LDS SUCCESS] 连续标签权重已更新。最高权重比: {self.bdi_loss_weights.max().item():.2f}\n")
        except Exception as e:
            print(f"[LDS WARNING] 运行时提取标签分布失败，保持标准无加权 MSE 模式。异常原因: {str(e)}\n")

    def on_train_epoch_start(self):
        torch.cuda.empty_cache()  # 每一轮开始前清理显存碎片
        # 预热阶段冻结 Backbone，只训练时序模块和任务头。
        if self.current_epoch < self.freeze_epochs:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 可以在日志中打印状态
            if self.current_epoch == 0:
                print("[INFO] Backbone 已冻结，开始预热时序网络和多任务头。")

        # 第 freeze_epochs 个 Epoch 之后：解冻 Backbone 的深层特征（比如 stage4），配合极小学习率微调
        # 将特定离散点判定(==)升级为严格的区间闭环控制(>=)，确保解冻状态贯穿整个中后期生命周期
        elif self.current_epoch >= self.freeze_epochs:
            if self.current_epoch == self.freeze_epochs:
                print("[INFO] 满足设定阈值，正式解冻 Backbone，进入端到端微调阶段。")
            elif self.current_epoch == self.warmup_epochs:
                print("[INFO] WARMUP 结束，开始进行正常训练。")

            # 解冻后持续保持 Backbone 参数可训练，避免跨 Epoch 状态不一致。
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif self.current_epoch == self.warmup_epochs:
            print("[INFO] WARMUP 结束，开始进行正常训练。")
            for param in self.backbone.parameters():
                param.requires_grad = True

        if self.current_epoch % 10 == 0:
            self.bdi_sigma = max(1.5, self.base_bdi_sigma * ( 1 - (self.current_epoch / self.max_epochs)))
        else:
            self.bdi_sigma = self.base_bdi_sigma

        # 动态更新损失函数中的核方差参数
        self.cross_subject_contrastive_loss = CrossSubjectContinuousContrastiveLoss(
            temperature=self.temperature,
            bdi_sigma=self.bdi_sigma
        )

    def _split_video_views(self, video_tensor):
        if isinstance(video_tensor, dict):
            return video_tensor["orig"], video_tensor["v1"], video_tensor["v2"], True

        return video_tensor, video_tensor, video_tensor, False

    def _prepare_labels(self, labels):
        true_bdi = labels['bdi_score']
        true_bdi_norm = true_bdi / float(self.max_score)
        true_cls_levels = get_coral_levels(labels['class_label'], self.num_classes)
        return true_bdi, true_bdi_norm, true_cls_levels

    def _predict_main_view(self, video_tensor, masks, true_bdi_norm, batch_idx):
        reg_pool, cls_pool, con_pool, f_mask = self._forward_features_to_cgc_tracks(
            video_tensor,
            masks,
            true_bdi_norm=true_bdi_norm,
            forced_f_mask=None,
            batch_idx=batch_idx
        )

        bdi_preds = self.reg_task_head(reg_pool).squeeze(-1)
        cls_preds = self.cls_task_head(cls_pool)
        con_embeds = self.con_task_head(con_pool)

        return bdi_preds, cls_preds, con_embeds, f_mask

    def _compute_multiview_contrastive_loss(
            self,
            con_embed_orig,
            video_tensor_v1,
            video_tensor_v2,
            is_multiview,
            masks,
            true_bdi_norm,
            true_bdi,
            f_mask_shared,
            batch_idx
    ):
        if not is_multiview:
            return self.cross_subject_contrastive_loss(
                embeds_orig=con_embed_orig,
                embeds_v1=con_embed_orig,
                embeds_v2=con_embed_orig,
                bdi_scores=true_bdi
            )

        _, _, con_pool_v1, _ = self._forward_features_to_cgc_tracks(
            video_tensor_v1,
            masks,
            true_bdi_norm=true_bdi_norm,
            forced_f_mask=f_mask_shared,
            batch_idx=batch_idx
        )
        _, _, con_pool_v2, _ = self._forward_features_to_cgc_tracks(
            video_tensor_v2,
            masks,
            true_bdi_norm=true_bdi_norm,
            forced_f_mask=f_mask_shared,
            batch_idx=batch_idx
        )

        return self.cross_subject_contrastive_loss(
            embeds_orig=con_embed_orig,
            embeds_v1=self.con_task_head(con_pool_v1),
            embeds_v2=self.con_task_head(con_pool_v2),
            bdi_scores=true_bdi
        )

    def _compute_multitask_losses(self, bdi_preds, cls_preds, true_bdi, true_bdi_norm, true_cls_levels, loss_con):
        weight_indices = torch.clamp(true_bdi.long(), 0, self.max_score)
        batch_weights = self.bdi_loss_weights[weight_indices]

        raw_score_elements = F.mse_loss(
            bdi_preds.float(),
            true_bdi_norm.float(),
            reduction="none",
        )

        loss_mse = (raw_score_elements * batch_weights).mean()
        loss_ccc = concordance_ccc_loss(bdi_preds, true_bdi_norm)
        pred_mean = bdi_preds.float().mean()
        target_mean = true_bdi_norm.float().mean().detach()
        pred_std = bdi_preds.float().std(unbiased=False)
        target_std = true_bdi_norm.float().std(unbiased=False).detach()
        loss_pred_mean = (pred_mean - target_mean).pow(2)
        loss_pred_std = F.relu(target_std - pred_std).pow(2)
        loss_dist = (
            self.pred_mean_loss_weight * loss_pred_mean
            + self.pred_std_loss_weight * loss_pred_std
        )
        loss_reg = loss_mse + self.ccc_loss_weight * loss_ccc + loss_dist
        loss_cls = coral_loss(cls_preds, true_cls_levels)

        if self.use_uw:
            w_loss_reg = loss_reg * torch.exp(-self.log_vars[0]) + 0.5 * self.log_vars[0]
            w_loss_cls = loss_cls * torch.exp(-self.log_vars[1]) + 0.5 * self.log_vars[1]
            con_weight = torch.clamp(torch.exp(-self.log_vars[2]), min=0.2)
            w_loss_con = loss_con * con_weight + 0.5 * self.log_vars[2]
            losses = [w_loss_reg, w_loss_cls, w_loss_con]
        else:
            losses = [loss_reg, loss_cls, loss_con]

        return loss_reg, loss_cls, losses, loss_mse, loss_ccc, loss_dist


    def _prediction_for_metrics(self, bdi_preds):
        """Restore normalized predictions to the real BDI scale for metrics."""
        return bdi_preds.detach().float().clamp(0.0, 1.0) * float(self.max_score)

    def _compute_gating_penalty(self):
        t_weights = self.temporal_encoder.last_trend_weights
        s_weights = self.temporal_encoder.last_seasonal_weights

        t_entropy = - (t_weights * torch.log(t_weights + 1e-8)).sum(dim=-1).mean()
        s_entropy = - (s_weights * torch.log(s_weights + 1e-8)).sum(dim=-1).mean()
        num_scales = t_weights.size(-1)
        target = torch.full((num_scales,), 1.0 / num_scales, device=t_weights.device, dtype=t_weights.dtype)
        t_balance = F.mse_loss(t_weights.mean(dim=(0, 1)), target)
        s_balance = F.mse_loss(s_weights.mean(dim=(0, 1)), target)
        entropy_term = -self.gate_entropy_weight * (t_entropy + s_entropy)
        balance_term = self.gate_balance_weight * (t_balance + s_balance)
        return entropy_term + balance_term

    def _backward_multitask_losses(self, losses, gating_penalty, opt):
        if self.use_pcgrad:
            pcgrad_backward(
                losses,
                opt,
                self,
                param_list=self.pcgrad_shared_parameters()
            )
            if hasattr(gating_penalty, 'grad_fn') and gating_penalty.grad_fn is not None:
                gating_penalty.backward()
            return

        total_loss = sum(losses) + gating_penalty
        if hasattr(total_loss, 'grad_fn') and total_loss.grad_fn is not None:
            total_loss.backward()
        else:
            print("[WARNING] 检测到当前计算图意外常数化，跳过本次无意义的 Backward。")

    def _step_optimizer(self, opt, sch):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        opt.step()
        opt.zero_grad()

        if sch is not None:
            sch.step()

        if self.current_epoch < self.freeze_epochs:
            lr_scale = 0.0
        elif self.freeze_epochs <= self.current_epoch < self.warmup_epochs:
            total_warmup_epochs = max(1, self.warmup_epochs - self.freeze_epochs)
            lr_scale = (self.current_epoch - self.freeze_epochs + 1) / total_warmup_epochs
        else:
            lr_scale = 1.0

        opt.param_groups[0]['lr'] = self.backbone_base_lr * lr_scale

    def training_step(self, batch, batch_idx):
        try:
            # ====================================================
            # 阶段一：手动优化环境获取与数据流解包
            # ====================================================
            # 由于设置了 self.automatic_optimization = False，必须手动提取优化器和学习率调度器
            opt = self.optimizers()
            sch = self.lr_schedulers()

            # 从数据管道中提取当前 Batch 的视频张量、全零 Padding 遮罩以及标签字典
            video_tensor, masks, labels = batch

            video_tensor_orig, video_tensor_v1, video_tensor_v2, is_multiview = self._split_video_views(video_tensor)
            current_bs = video_tensor_orig.size(0)
            true_bdi, true_bdi_norm, true_cls_levels = self._prepare_labels(labels)
            bdi_preds, cls_preds, con_embed_orig, f_mask_shared = self._predict_main_view(
                video_tensor_orig,
                masks,
                true_bdi_norm,
                batch_idx
            )

            # 更新训练集的 TorchMetrics 指标计数器
            bdi_preds_for_metrics = self._prediction_for_metrics(bdi_preds)
            self.train_rmse(bdi_preds_for_metrics, true_bdi)
            self.train_mae(bdi_preds_for_metrics, true_bdi)
            self.train_ccc(bdi_preds_for_metrics, true_bdi)
            self.log("train_pred_norm_mean", bdi_preds.detach().float().mean(), on_epoch=True, on_step=False, batch_size=current_bs)
            self.log("train_pred_norm_min", bdi_preds.detach().float().min(), on_epoch=True, on_step=False, batch_size=current_bs)
            self.log("train_pred_mean", bdi_preds_for_metrics.mean(), on_epoch=True, on_step=False, batch_size=current_bs)
            self.log("train_pred_std", bdi_preds_for_metrics.std(unbiased=False), on_epoch=True, on_step=False, batch_size=current_bs)

            del video_tensor_orig

            loss_con = self._compute_multiview_contrastive_loss(
                con_embed_orig=con_embed_orig,
                video_tensor_v1=video_tensor_v1,
                video_tensor_v2=video_tensor_v2,
                is_multiview=is_multiview,
                masks=masks,
                true_bdi_norm=true_bdi_norm,
                true_bdi=true_bdi,
                f_mask_shared=f_mask_shared,
                batch_idx=batch_idx
            )

            # ====================================================
            # 损失加权汇总
            # ====================================================
            loss_reg, loss_cls, losses, loss_mse, loss_ccc, loss_dist = self._compute_multitask_losses(
                bdi_preds=bdi_preds,
                cls_preds=cls_preds,
                true_bdi=true_bdi,
                true_bdi_norm=true_bdi_norm,
                true_cls_levels=true_cls_levels,
                loss_con=loss_con
            )

            # ====================================================
            # 参数优化步进
            # ====================================================
            # 避免门控极化导致只关注某条特定的时序轨道
            # 在计算完总损失前，动态提取时序编码器当前的门控权重
            gating_penalty = self._compute_gating_penalty()
            self._backward_multitask_losses(losses, gating_penalty, opt)

            self._step_optimizer(opt, sch)

            self.log("train_reg_loss", loss_reg.detach(), on_epoch=True, on_step=False, batch_size=current_bs)
            self.log("train_mse_loss", loss_mse.detach(), on_epoch=True, on_step=False, batch_size=current_bs)
            self.log("train_ccc_loss", loss_ccc.detach(), on_epoch=True, on_step=False, batch_size=current_bs)
            self.log("train_dist_loss", loss_dist.detach(), on_epoch=True, on_step=False, batch_size=current_bs)
            self.log("train_cls_loss", loss_cls.detach(), on_epoch=True, on_step=False, batch_size=current_bs)
            self.log("train_con_loss", loss_con.detach(), on_epoch=True, on_step=False, batch_size=current_bs)

            if self.use_uw:
                self.log("uw_weight_reg", torch.exp(-self.log_vars[0]), on_epoch=True, on_step=False,
                         batch_size=current_bs)
                self.log("uw_weight_cls", torch.exp(-self.log_vars[1]), on_epoch=True, on_step=False,
                         batch_size=current_bs)
                self.log("uw_weight_con", torch.exp(-self.log_vars[2]), on_epoch=True, on_step=False,
                         batch_size=current_bs)

            return None

        except Exception as e:
            print(f"\n❌ [TRAIN STEP ERROR] 在训练第 {batch_idx} 个 Batch 时发生严重崩溃！")
            raise e

    def on_train_epoch_end(self):
        # 提取并同步 Epoch 训练集的度量结果
        self.log("train_RMSE_epoch", self.train_rmse.compute(), prog_bar=True)
        self.log("train_MAE_epoch", self.train_mae.compute(), prog_bar=True)
        self.log("train_CCC_epoch", self.train_ccc.compute(), prog_bar=True)
        self.train_rmse.reset()
        self.train_mae.reset()
        self.train_ccc.reset()

    def on_validation_epoch_start(self):
        """Reset validation caches before collecting epoch-level diagnostics."""
        self.val_features_storage.clear()
        self.val_scores_storage.clear()
        self.val_subjects_storage.clear()
        self.val_pred_storage.clear()
        self.val_target_storage.clear()
        self.val_trend_weights_storage.clear()
        self.val_seasonal_weights_storage.clear()
        torch.cuda.empty_cache()  # 清理上轮的显存碎片


    def validation_step(self, batch, batch_idx):
        # 如果是启动前的预检，直接掐断，不参与任何指标累加与特征缓存
        if self.trainer.sanity_checking:
            return None

        # 确保 BatchNorm 永远不会被验证集数据偷偷更新污染
        with torch.no_grad():
            self.eval()

        video_tensor, masks, labels = batch

        # 提取验证集当前 Batch 的真实大小。
        if isinstance(video_tensor, dict):
            current_bs = video_tensor["orig"].size(0)
        else:
            current_bs = video_tensor.size(0)

        is_embedding_epoch = self._should_visualize(self.embedding_visualization_interval)
        is_gating_epoch = self._should_visualize(self.gating_visualization_interval)

        bdi_preds, cls_preds, con_embeds = self(video_tensor, masks, need_all_heads=is_embedding_epoch)

        if is_embedding_epoch and con_embeds is not None:
            normalized_embeds = F.normalize(con_embeds, p=2, dim=-1)
            self.val_features_storage.append(normalized_embeds.detach().cpu())
            self.val_scores_storage.append(labels['bdi_score'].detach().cpu())
            self.val_subjects_storage.extend(labels['subject_id'])

        if is_gating_epoch and hasattr(self.temporal_encoder, "last_trend_weights"):
            self.val_trend_weights_storage.append(self.temporal_encoder.last_trend_weights.detach().cpu())
            self.val_seasonal_weights_storage.append(self.temporal_encoder.last_seasonal_weights.detach().cpu())

        if is_gating_epoch and batch_idx == 0 and self.trainer.is_global_zero:
            try:
                t_w = self.temporal_encoder.last_trend_weights[0]
                s_w = self.temporal_encoder.last_seasonal_weights[0]
                log_dir = self.trainer.loggers[0].log_dir
                os.makedirs(f"{log_dir}/plots/", exist_ok=True)
                plot_temporal_gating_weights(
                    trend_weights=t_w,
                    seasonal_weights=s_w,
                    moving_avg_kernels=self.moving_avg_kernels,
                    save_path=f"{log_dir}/plots/temporal_gating_epoch_{self.current_epoch}.png",
                    video_id=labels['subject_id'][0]
                )
            except AttributeError:
                pass

        # 计算基础监控指标
        true_bdi = labels['bdi_score']
        true_bdi_norm = true_bdi / float(self.max_score)
        loss_mse = F.mse_loss(
            bdi_preds.float(),
            true_bdi_norm.float()
        )
        loss_ccc = concordance_ccc_loss(bdi_preds, true_bdi_norm)
        loss_reg = loss_mse + self.ccc_loss_weight * loss_ccc
        self.log("val_reg_loss", loss_reg, batch_size=current_bs)
        self.log("val_mse_loss", loss_mse, batch_size=current_bs)
        self.log("val_ccc_loss", loss_ccc, batch_size=current_bs)

        bdi_preds_for_metrics = self._prediction_for_metrics(bdi_preds)
        self.val_rmse(bdi_preds_for_metrics, true_bdi)
        self.val_mae(bdi_preds_for_metrics, true_bdi)
        self.val_ccc(bdi_preds_for_metrics, true_bdi)
        self.log("val_pred_norm_mean", bdi_preds.detach().float().mean(), batch_size=current_bs)
        self.log("val_pred_norm_min", bdi_preds.detach().float().min(), batch_size=current_bs)
        self.log("val_pred_mean", bdi_preds_for_metrics.mean(), batch_size=current_bs)
        self.log("val_pred_std", bdi_preds_for_metrics.std(unbiased=False), batch_size=current_bs)
        self.val_pred_storage.append(bdi_preds_for_metrics.cpu())
        self.val_target_storage.append(true_bdi.detach().cpu())

        # 只有存在分类预测时才记录分类损耗
        if cls_preds is not None:
            true_cls_levels = get_coral_levels(labels['class_label'], self.num_classes)
            self.log("val_cls_loss", coral_loss(cls_preds, true_cls_levels), batch_size=current_bs)

        return None

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        global_rmse = self.val_rmse.compute()
        global_mae = self.val_mae.compute()
        global_ccc = self.val_ccc.compute()
        # 由 TorchMetrics 自行跨卡同步，self.log 内部不再强制显式指定 sync_dist=True，杜绝多卡死锁
        self.log("val_RMSE_epoch", global_rmse, prog_bar=True)
        self.log("val_MAE_epoch", global_mae, prog_bar=True)
        self.log("val_CCC_epoch", global_ccc, prog_bar=True)

        if len(self.val_features_storage) > 0 and self.trainer.is_global_zero and self.trainer.loggers:
            all_features = torch.cat(self.val_features_storage, dim=0).float()
            all_scores = torch.cat(self.val_scores_storage, dim=0).float()
            log_dir = self.trainer.loggers[0].log_dir
            plot_embedding_diagnostics(
                features=all_features, subject_ids=self.val_subjects_storage, bdi_scores=all_scores,
                save_dir=os.path.join(log_dir, "plots"), epoch=self.current_epoch
            )

        if (
                len(self.val_pred_storage) > 0
                and self._should_visualize(self.regression_visualization_interval)
                and self.trainer.is_global_zero
                and self.trainer.loggers
        ):
            log_dir = self.trainer.loggers[0].log_dir
            plot_regression_diagnostics(
                preds=torch.cat(self.val_pred_storage, dim=0),
                targets=torch.cat(self.val_target_storage, dim=0),
                subject_ids=self.val_subjects_storage,
                save_dir=os.path.join(log_dir, "plots"),
                stage="val",
                epoch=self.current_epoch,
                max_score=self.max_score,
                class_step=self.class_step
            )

        if len(self.val_trend_weights_storage) > 0 and self.trainer.is_global_zero and self.trainer.loggers:
            log_dir = self.trainer.loggers[0].log_dir
            plot_temporal_gating_summary(
                trend_weights=torch.cat(self.val_trend_weights_storage, dim=0),
                seasonal_weights=torch.cat(self.val_seasonal_weights_storage, dim=0),
                moving_avg_kernels=self.moving_avg_kernels,
                save_path=os.path.join(log_dir, "plots", f"temporal_gating_summary_epoch_{self.current_epoch}.png"),
                epoch=self.current_epoch
            )

        self.val_features_storage.clear()
        self.val_scores_storage.clear()
        self.val_subjects_storage.clear()
        self.val_pred_storage.clear()
        self.val_target_storage.clear()
        self.val_trend_weights_storage.clear()
        self.val_seasonal_weights_storage.clear()

        if global_rmse < self.best_val_rmse:
            self.best_val_rmse = global_rmse
            self.save_segmented_weights(tag="best")

        self.val_rmse.reset()
        self.val_mae.reset()
        self.val_ccc.reset()
        torch.cuda.empty_cache()

    def load_segmented_weights(self, weights_dir, tag="best"):
        """
        从指定目录加载分段权重
        """
        backbone_path = os.path.join(weights_dir, f"{tag}_backbone.pth")
        heads_path = os.path.join(weights_dir, f"{tag}_temporal_heads.pth")

        if not os.path.exists(backbone_path) or not os.path.exists(heads_path):
            print(f"❌ [ERROR] 找不到权重文件: {backbone_path} 或 {heads_path}")
            return False

        # 添加 map_location 确保设备一致
        device = self.device
        # 1. 加载 Backbone
        self.backbone.load_state_dict(torch.load(backbone_path, map_location=device))

        heads_dict = torch.load(heads_path, map_location=device)

        self.proj.load_state_dict(heads_dict['proj'])
        self.cgc_layer.load_state_dict(heads_dict['cgc_layer'])
        self.temporal_encoder.load_state_dict(heads_dict['temporal_encoder'])  #

        self.reg_task_head.load_state_dict(heads_dict['task_heads']['reg'])
        self.cls_task_head.load_state_dict(heads_dict['task_heads']['cls'])
        self.con_task_head.load_state_dict(heads_dict['task_heads']['con'])

        # 加载不确定性权重 (UW)
        if self.use_uw and heads_dict.get('log_vars') is not None:
            # 对于 nn.Parameter，直接赋值 data
            self.log_vars.data = heads_dict['log_vars'].data

        print(f"[SUCCESS] 成功从 {weights_dir} 加载最优分段权重！")
        return True

    def on_test_epoch_start(self):
        self.test_pred_storage.clear()
        self.test_target_storage.clear()
        self.test_subjects_storage.clear()

    def test_step(self, batch, batch_idx):
        video_tensor, masks, labels = batch
        # 测试阶段只计算主回归头，避免额外任务头带来的无关开销。
        bdi_preds, _, _ = self(video_tensor, masks, need_all_heads=False)

        bdi_preds_for_metrics = self._prediction_for_metrics(bdi_preds)
        self.test_rmse(bdi_preds_for_metrics, labels['bdi_score'])
        self.test_mae(bdi_preds_for_metrics, labels['bdi_score'])
        self.test_ccc(bdi_preds_for_metrics, labels['bdi_score'])
        self.test_pred_storage.append(bdi_preds_for_metrics.cpu())
        self.test_target_storage.append(labels['bdi_score'].detach().cpu())
        self.test_subjects_storage.extend(labels['subject_id'])
        return None

    def on_test_epoch_end(self):
        # 指标由 torchmetrics 在各 step 中累积，epoch 末统一规约输出。
        print(f'\n[TEST END] 精确 Test RMSE: {self.test_rmse.compute():.4f}, Test MAE: {self.test_mae.compute():.4f}')
        print(f'[TEST END] Test CCC: {self.test_ccc.compute():.4f}')
        self.test_rmse.reset()
        self.test_mae.reset()
        self.test_ccc.reset()
        if len(self.test_pred_storage) > 0 and self.trainer.is_global_zero and self.trainer.loggers:
            log_dir = self.trainer.loggers[0].log_dir
            plot_regression_diagnostics(
                preds=torch.cat(self.test_pred_storage, dim=0),
                targets=torch.cat(self.test_target_storage, dim=0),
                subject_ids=self.test_subjects_storage,
                save_dir=os.path.join(log_dir, "plots"),
                stage="test",
                epoch=None,
                max_score=self.max_score,
                class_step=self.class_step
            )
        self.test_pred_storage.clear()
        self.test_target_storage.clear()
        self.test_subjects_storage.clear()

    def configure_optimizers(self):
        max_lrs = [self.ef_lr] + [self.tmp_lr] * 7

        optimizer_grouped_parameters = [
            {"params": self.backbone.parameters(), "lr": self.ef_lr},
            {"params": self.proj.parameters(), "lr": self.tmp_lr},
            {"params": self.cgc_layer.parameters(), "lr": self.tmp_lr},
            # 单一的 temporal_encoder
            {"params": self.temporal_encoder.parameters(), "lr": self.tmp_lr},
            {"params": self.reg_task_head.parameters(), "lr": self.tmp_lr},
            {"params": self.cls_task_head.parameters(), "lr": self.tmp_lr},
            {"params": self.con_task_head.parameters(), "lr": self.tmp_lr},
            {"params": [self.log_vars], "lr": self.tmp_lr * 0.1},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=self.ef_weight_decay)

        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
