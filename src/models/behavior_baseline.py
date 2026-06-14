import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from src.metrics.metrics import ConcordanceCorrCoefMetric, concordance_ccc_loss
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


class BehaviorBaselineModel(pl.LightningModule):
    """Behavior-only temporal baseline using OpenFace sequence features."""

    def __init__(self, configs, input_dim=None):
        super().__init__()
        self.cfgs = configs
        self.max_score = int(_get_config_value(configs, "EXTRACT_FEATURE", "MAX_SCORE", 63))
        self.input_dim = int(input_dim or _get_config_value(configs, "BEHAVIOR_MODEL", "INPUT_DIM", 0))
        if self.input_dim <= 0:
            raise ValueError("BehaviorBaselineModel requires a positive input_dim.")

        self.hidden_dim = int(_get_config_value(configs, "BEHAVIOR_MODEL", "HIDDEN_DIM", 128))
        self.num_layers = int(_get_config_value(configs, "BEHAVIOR_MODEL", "NUM_LAYERS", 1))
        self.bidirectional = bool(_get_config_value(configs, "BEHAVIOR_MODEL", "BIDIRECTIONAL", True))
        self.dropout = float(_get_config_value(configs, "BEHAVIOR_MODEL", "DROPOUT", 0.2))
        self.class_step = int(_get_config_value(configs, "PROCESS_TEMPORAL", "CLASS_STEP", 2))
        self.num_classes = math.ceil(self.max_score / self.class_step)
        self.learning_rate = float(_get_config_value(configs, "PROCESS_TEMPORAL", "LEARNING_RATE", 1e-4))
        self.weight_decay = float(_get_config_value(configs, "PROCESS_TEMPORAL", "WEIGHT_DECAY", 5e-4))

        self.ordinal_weight = float(_get_config_value(configs, "LOSSES", "ORDINAL_WEIGHT", 0.0))
        self.ccc_loss_weight = float(_get_config_value(configs, "LOSSES", "CCC_WEIGHT", 0.0))
        self.use_ordinal_task = bool(
            _get_nested_config_value(
                configs,
                "MODEL",
                "AUXILIARY_TASKS",
                "ORDINAL_CLASSIFICATION",
                self.ordinal_weight > 0.0,
            )
        )

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        recurrent_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.temporal_encoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=recurrent_dropout,
        )
        representation_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.reg_task_head = build_regression_task_head(representation_dim, self.hidden_dim, 1)
        self.ordinal_task_head = None
        if self.use_ordinal_task:
            self.ordinal_task_head = build_classification_task_head(
                representation_dim,
                self.hidden_dim,
                self.num_classes,
            )

        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.train_ccc = ConcordanceCorrCoefMetric()
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_ccc = ConcordanceCorrCoefMetric()
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_ccc = ConcordanceCorrCoefMetric()

    def forward(self, features, mask, return_features=False):
        projected = self.input_proj(features.float())
        encoded, _ = self.temporal_encoder(projected)
        encoded = encoded * mask.to(device=encoded.device, dtype=encoded.dtype).unsqueeze(-1)
        shared_features = self.dropout_layer(masked_mean_pool(encoded, mask))
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
        features, mask, labels = batch
        outputs = self(features, mask)
        losses = self.compute_losses(outputs, labels)
        true_bdi = labels["bdi_score"].float()
        batch_size = features.size(0)

        self._update_metrics(stage, outputs.bdi_pred, true_bdi)
        self.log(f"{stage}_loss", losses.total, on_epoch=True, on_step=False, batch_size=batch_size)
        self.log(f"{stage}_reg_loss", losses.regression, on_epoch=True, on_step=False, batch_size=batch_size)
        if losses.ordinal is not None:
            self.log(f"{stage}_ordinal_loss", losses.ordinal, on_epoch=True, on_step=False, batch_size=batch_size)
        if losses.ccc is not None:
            self.log(f"{stage}_ccc_loss", losses.ccc, on_epoch=True, on_step=False, batch_size=batch_size)
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
