import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from src.metrics.metrics import ConcordanceCorrCoefMetric, concordance_ccc_loss
from src.models.backbone_factory import build_feature_backbone
from src.models.outputs import MTLLiteLosses, MTLLiteOutput
from src.models.task_heads import (
    build_classification_task_head,
    build_regression_task_head,
    coral_loss,
    get_coral_levels,
)
from src.models.temporal.pooling import masked_mean_pool


def _get_config_value(configs, section_name, key, default):
    section = getattr(configs, section_name, None)
    if section is None:
        return default
    return getattr(section, key, default)


def _get_nested_config_value(configs, section_name, nested_section_name, key, default):
    section = getattr(configs, section_name, None)
    if section is None:
        return default
    nested_section = getattr(section, nested_section_name, None)
    if nested_section is None:
        return default
    return getattr(nested_section, key, default)


class MTLLiteDepressionModel(pl.LightningModule):
    """Lightweight multi-task BDI prediction model.

    The model intentionally avoids legacy full-model components such as CGC,
    contrastive learning, adaptive masks, PCGrad, LDS, and distribution loss.
    """

    def __init__(self, configs):
        super().__init__()
        self.cfgs = configs

        self.max_score = int(_get_config_value(configs, "EXTRACT_FEATURE", "MAX_SCORE", 63))
        self.model_name = str(_get_config_value(configs, "EXTRACT_FEATURE", "MODEL_NAME", "deit_tiny_patch16_224"))
        self.weight_path = _get_config_value(configs, "EXTRACT_FEATURE", "MODEL_WEIGHT_PATH", None)
        self.timm_pretrained = bool(_get_config_value(configs, "EXTRACT_FEATURE", "TIMM_PRETRAINED", False))
        self.chunk_size = int(_get_config_value(configs, "EXTRACT_FEATURE", "CHUNK_SIZE", 64))
        self.freeze_backbone = bool(_get_config_value(configs, "EXTRACT_FEATURE", "FREEZE_BACKBONE", False))
        self.finetune_last_n_blocks = int(
            _get_config_value(configs, "EXTRACT_FEATURE", "FINETUNE_LAST_N_BLOCKS", 0)
        )

        self.hidden_dim = int(_get_config_value(configs, "PROCESS_TEMPORAL", "HIDDEN_DIM", 192))
        self.class_step = int(_get_config_value(configs, "PROCESS_TEMPORAL", "CLASS_STEP", 2))
        self.num_classes = math.ceil(self.max_score / self.class_step)
        self.dropout = float(_get_config_value(configs, "PROCESS_TEMPORAL", "DROPOUT", 0.0))
        self.learning_rate = float(_get_config_value(configs, "PROCESS_TEMPORAL", "LEARNING_RATE", 1e-4))
        self.weight_decay = float(_get_config_value(configs, "PROCESS_TEMPORAL", "WEIGHT_DECAY", 5e-4))

        self.ordinal_weight = float(_get_config_value(configs, "LOSSES", "ORDINAL_WEIGHT", 1.0))
        self.ccc_loss_weight = float(
            _get_config_value(
                configs,
                "LOSSES",
                "CCC_WEIGHT",
                _get_config_value(configs, "PROCESS_TEMPORAL", "CCC_LOSS_WEIGHT", 0.0),
            )
        )
        self.use_ordinal_task = bool(
            _get_nested_config_value(
                configs,
                "MODEL",
                "AUXILIARY_TASKS",
                "ORDINAL_CLASSIFICATION",
                self.ordinal_weight > 0.0,
            )
        )

        self.input_dim = int(configs.BACKBONE_OUT_DIMS.get(self.model_name))
        self.backbone = build_feature_backbone(
            model_name=self.model_name,
            weight_path=self.weight_path,
            timm_pretrained=self.timm_pretrained,
        )
        self.configure_backbone_trainability()

        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.temporal_encoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.reg_task_head = build_regression_task_head(self.hidden_dim, self.hidden_dim, 1)
        self.ordinal_task_head = None
        if self.use_ordinal_task:
            self.ordinal_task_head = build_classification_task_head(self.hidden_dim, self.hidden_dim, self.num_classes)

        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.train_ccc = ConcordanceCorrCoefMetric()
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_ccc = ConcordanceCorrCoefMetric()
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_ccc = ConcordanceCorrCoefMetric()

    @staticmethod
    def _set_module_trainable(module, trainable):
        for param in module.parameters():
            param.requires_grad = trainable

    def _count_backbone_parameters(self):
        total = sum(param.numel() for param in self.backbone.parameters())
        trainable = sum(param.numel() for param in self.backbone.parameters() if param.requires_grad)
        return total, trainable

    def configure_backbone_trainability(self):
        """Apply backbone freeze / high-layer finetuning config."""
        if not self.freeze_backbone:
            total, trainable = self._count_backbone_parameters()
            print(f"[BACKBONE] Full backbone trainable: {trainable}/{total} parameters.")
            return

        self._set_module_trainable(self.backbone, False)
        unfrozen_blocks = 0

        if self.finetune_last_n_blocks > 0 and hasattr(self.backbone, "blocks"):
            blocks = list(self.backbone.blocks)
            unfrozen_blocks = min(self.finetune_last_n_blocks, len(blocks))
            for block in blocks[-unfrozen_blocks:]:
                self._set_module_trainable(block, True)

            for norm_name in ("norm", "fc_norm"):
                norm_layer = getattr(self.backbone, norm_name, None)
                if norm_layer is not None:
                    self._set_module_trainable(norm_layer, True)
        elif self.finetune_last_n_blocks > 0:
            print(
                "[BACKBONE] FINETUNE_LAST_N_BLOCKS was set, but this backbone "
                "does not expose transformer-style `.blocks`; keeping backbone frozen."
            )

        total, trainable = self._count_backbone_parameters()
        print(
            "[BACKBONE] Frozen backbone with "
            f"{unfrozen_blocks} high-level blocks trainable: {trainable}/{total} parameters."
        )

    def extract_frame_features(self, video_tensor, mask):
        """Extract per-frame visual features while preserving padded positions."""
        batch_size, seq_len, channels, height, width = video_tensor.shape
        flat_video = video_tensor.reshape(batch_size * seq_len, channels, height, width)
        valid_indices = mask.reshape(-1).bool()
        valid_frames = flat_video[valid_indices]

        if valid_frames.numel() == 0:
            return torch.zeros(
                batch_size,
                seq_len,
                self.input_dim,
                device=video_tensor.device,
                dtype=video_tensor.dtype,
            )

        feature_chunks = []
        for start in range(0, valid_frames.size(0), self.chunk_size):
            feature_chunks.append(self.backbone(valid_frames[start:start + self.chunk_size]))

        valid_features = torch.cat(feature_chunks, dim=0)
        flat_features = torch.zeros(
            batch_size * seq_len,
            self.input_dim,
            device=video_tensor.device,
            dtype=valid_features.dtype,
        )
        flat_features[valid_indices] = valid_features
        return flat_features.reshape(batch_size, seq_len, self.input_dim)

    def encode_temporal_features(self, frame_features, mask):
        projected = self.proj(frame_features)
        encoded, _ = self.temporal_encoder(projected)
        return encoded * mask.to(device=encoded.device, dtype=encoded.dtype).unsqueeze(-1)

    def pool_video_features(self, temporal_features, mask):
        pooled = masked_mean_pool(temporal_features, mask)
        return self.dropout_layer(pooled)

    def forward(self, video_tensor, mask, return_features=False):
        frame_features = self.extract_frame_features(video_tensor, mask)
        temporal_features = self.encode_temporal_features(frame_features, mask)
        shared_features = self.pool_video_features(temporal_features, mask)
        bdi_pred = self.reg_task_head(shared_features).squeeze(-1)
        ordinal_logits = None
        if self.ordinal_task_head is not None:
            ordinal_logits = self.ordinal_task_head(shared_features)
        return MTLLiteOutput(
            bdi_pred=bdi_pred,
            ordinal_logits=ordinal_logits,
            shared_features=shared_features if return_features else None,
        )

    def prepare_labels(self, labels):
        true_bdi = labels["bdi_score"].float()
        true_bdi_norm = true_bdi / float(self.max_score)
        ordinal_levels = get_coral_levels(labels["class_label"].long(), self.num_classes)
        return true_bdi, true_bdi_norm, ordinal_levels

    def compute_losses(self, outputs, labels):
        _, true_bdi_norm, ordinal_levels = self.prepare_labels(labels)
        loss_reg = F.mse_loss(outputs.bdi_pred.float(), true_bdi_norm.float())
        loss_ccc = concordance_ccc_loss(outputs.bdi_pred, true_bdi_norm)
        loss_ord = None
        if outputs.ordinal_logits is not None and self.ordinal_weight > 0.0:
            loss_ord = coral_loss(outputs.ordinal_logits, ordinal_levels)

        total = loss_reg + self.ccc_loss_weight * loss_ccc
        if loss_ord is not None:
            total = total + self.ordinal_weight * loss_ord
        return MTLLiteLosses(
            total=total,
            regression=loss_reg,
            ordinal=loss_ord,
            ccc=loss_ccc,
        )

    def prediction_for_metrics(self, bdi_preds):
        return bdi_preds.detach().float().clamp(0.0, 1.0) * float(self.max_score)

    def _update_metrics(self, stage, bdi_preds, true_bdi):
        metric_preds = self.prediction_for_metrics(bdi_preds)
        getattr(self, f"{stage}_rmse")(metric_preds, true_bdi)
        getattr(self, f"{stage}_mae")(metric_preds, true_bdi)
        getattr(self, f"{stage}_ccc")(metric_preds, true_bdi)
        return metric_preds

    def _shared_step(self, batch, stage):
        video_tensor, mask, labels = batch
        outputs = self(video_tensor, mask)
        losses = self.compute_losses(outputs, labels)
        true_bdi = labels["bdi_score"].float()
        current_bs = video_tensor.size(0)

        self._update_metrics(stage, outputs.bdi_pred, true_bdi)
        self.log(f"{stage}_loss", losses.total, on_epoch=True, on_step=False, batch_size=current_bs)
        self.log(f"{stage}_reg_loss", losses.regression, on_epoch=True, on_step=False, batch_size=current_bs)
        if losses.ordinal is not None:
            self.log(f"{stage}_ordinal_loss", losses.ordinal, on_epoch=True, on_step=False, batch_size=current_bs)
        if losses.ccc is not None:
            self.log(f"{stage}_ccc_loss", losses.ccc, on_epoch=True, on_step=False, batch_size=current_bs)
        return losses.total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def _log_epoch_metrics(self, stage):
        rmse = getattr(self, f"{stage}_rmse")
        mae = getattr(self, f"{stage}_mae")
        ccc = getattr(self, f"{stage}_ccc")
        self.log(f"{stage}_RMSE_epoch", rmse.compute(), prog_bar=(stage == "val"))
        self.log(f"{stage}_MAE_epoch", mae.compute(), prog_bar=(stage == "val"))
        self.log(f"{stage}_CCC_epoch", ccc.compute(), prog_bar=(stage == "val"))
        rmse.reset()
        mae.reset()
        ccc.reset()

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
