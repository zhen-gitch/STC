import math
from typing import Dict, List, Optional

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


# RPDF Stage A1: suggested layer-wise probe targets.
#
# These are attribute paths resolved against ``self.backbone`` (for the
# ``layer_*`` backbone layers) and against the model itself (for
# ``layer_temporal`` / ``layer_shared``).  ``resolve_layer_module`` returns
# ``None`` for any path that does not exist on the actual backbone, so the
# probe degrades gracefully across DeiT / ViT / ResNet / iresnet backbones
# rather than hard-coding a single architecture.
LAYERWISE_PROBE_LAYERS = (
    "layer_stem",          # backbone.patch_embed output (patch + pos + cls token)
    "layer_block_0",       # backbone.blocks[0]
    "layer_block_3",       # backbone.blocks[3]
    "layer_block_6",       # backbone.blocks[6]
    "layer_block_9",       # backbone.blocks[9]
    "layer_block_11",      # backbone.blocks[11] (deit_tiny has 12 blocks)
    "layer_backbone_out",  # backbone final output (== extract_frame_features)
    "layer_temporal",      # GRU encoded + masked, before pooling
    "layer_shared",        # MTL shared representation (pooled), the P0-D baseline
)


def _resolve_backbone_block(backbone, index):
    """Return ``backbone.blocks[index]`` if available, else ``None``."""
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return None
    try:
        block_list = list(blocks)
    except TypeError:
        return None
    if 0 <= index < len(block_list):
        return block_list[index]
    return None


def resolve_layer_module(model, layer_name):
    """Resolve a probe layer name to a concrete ``nn.Module`` or ``None``.

    Backbone layers are resolved against ``model.backbone``; ``layer_temporal``
    and ``layer_shared`` have no single module (they are intermediate forward
    results) and return ``None`` here -- they are captured directly in
    :meth:`MTLLiteDepressionModel.forward` rather than via hooks.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None

    if layer_name == "layer_stem":
        return getattr(backbone, "patch_embed", None)
    if layer_name == "layer_backbone_out":
        # The backbone module itself; its forward output is the final feature.
        return backbone
    if layer_name.startswith("layer_block_"):
        try:
            index = int(layer_name[len("layer_block_"):])
        except ValueError:
            return None
        return _resolve_backbone_block(backbone, index)
    # layer_temporal / layer_shared are handled inline in forward(), not hooked.
    return None


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

        # RPDF Stage A1: layer-wise probe state.  Hooks are only registered by
        # an explicit diagnostic call; the training path never touches these,
        # so forward()/training_step behaviour is unchanged when inactive.
        self._layer_probe_handles: List = []
        self._layer_probe_cache: Dict[str, torch.Tensor] = {}
        self._layer_probe_active: bool = False

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

    # ------------------------------------------------------------------
    # RPDF Stage A1: layer-wise identity probe hooks.
    #
    # These methods are diagnostic-only.  They register forward hooks on
    # backbone sub-modules so that a subsequent ``forward(...,
    # return_layer_features=True)`` can collect per-layer embeddings for the
    # layer-wise identity retrieval audit.  The training path never calls
    # ``register_layer_hooks`` and ``_layer_probe_active`` stays False, so
    # ``forward`` and ``training_step`` behaviour is identical to before.
    # ------------------------------------------------------------------

    def clear_layer_hooks(self):
        """Remove all previously registered layer-probe hooks."""
        for handle in self._layer_probe_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._layer_probe_handles = []
        self._layer_probe_cache = {}
        self._layer_probe_active = False

    def register_layer_hooks(self, layer_names=None):
        """Register forward hooks to capture intermediate backbone activations.

        Args:
            layer_names: Iterable of layer names (see ``LAYERWISE_PROBE_LAYERS``).
                If ``None``, all names in ``LAYERWISE_PROBE_LAYERS`` are
                attempted.  Names that cannot be resolved to a module on the
                current backbone are skipped with a warning, so the probe works
                across DeiT / ViT / ResNet / iresnet backbones.

        ``layer_temporal`` and ``layer_shared`` are captured inline in
        :meth:`forward`, not via hooks; they are still accepted here so callers
        can pass the full ``LAYERWISE_PROBE_LAYERS`` list without filtering.
        """
        if layer_names is None:
            layer_names = LAYERWISE_PROBE_LAYERS

        self.clear_layer_hooks()

        registered = []
        skipped = []
        for layer_name in layer_names:
            # layer_temporal / layer_shared are handled in forward(), not hooked.
            if layer_name in ("layer_temporal", "layer_shared"):
                registered.append(layer_name)
                continue
            module = resolve_layer_module(self, layer_name)
            if module is None:
                skipped.append(layer_name)
                continue
            handle = module.register_forward_hook(self._make_layer_hook(layer_name))
            self._layer_probe_handles.append(handle)
            registered.append(layer_name)

        self._layer_probe_active = bool(registered)
        if skipped:
            print(
                f"[LAYER-PROBE] Skipped unresolved layers on this backbone: {skipped}"
            )
        if registered:
            print(f"[LAYER-PROBE] Registered hooks for layers: {registered}")
        else:
            print("[LAYER-PROBE] No layers registered; backbone has no hookable targets.")
        return registered

    def _make_layer_hook(self, layer_name):
        """Create a forward hook that *accumulates* the module output.

        ``extract_frame_features`` calls the backbone in chunks (and the
        backbone's internal sub-modules are therefore invoked once per chunk),
        so a single assignment would be overwritten by the last chunk.  Instead
        we concatenate each chunk's output along the batch dimension, preserving
        the valid-frame order, and slice back to per-video vectors in forward().
        """

        def hook(_module, _inputs, output):
            tensor = output
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            if tensor is None:
                return
            tensor = tensor.detach()
            existing = self._layer_probe_cache.get(layer_name)
            if existing is None:
                self._layer_probe_cache[layer_name] = tensor
            else:
                # Concatenate along the frame (batch) dimension.  Both tensors
                # share the same trailing shape from the same module.
                self._layer_probe_cache[layer_name] = torch.cat([existing, tensor], dim=0)

        return hook

    @staticmethod
    def _pool_layer_frame_features(frame_features, mask, eps=1e-8):
        """Pool per-frame layer activations to one video-level vector.

        Args:
            frame_features: Tensor of shape (T, ...) for the valid frames of one
                video, where T is the number of valid frames.  Trailing dims are
                flattened so transformer token outputs and CNN feature maps are
                both reduced by mean over the spatial/token dimension first.
            mask: Unused here (valid frames are already selected by the caller),
                kept for API symmetry with :func:`masked_mean_pool`.
        """
        flat = frame_features.reshape(frame_features.size(0), -1)
        return flat.mean(dim=0)

    def extract_frame_features(self, video_tensor, mask):
        """Extract per-frame visual features while preserving padded positions."""
        batch_size, seq_len, channels, height, width = video_tensor.shape
        flat_video = video_tensor.reshape(batch_size * seq_len, channels, height, width)
        valid_indices = mask.reshape(-1).bool()
        valid_frames = flat_video[valid_indices]

        # Reset the layer-probe cache so accumulated hook outputs start fresh
        # for this batch.  No-op when the probe is inactive (cache stays empty).
        if self._layer_probe_active:
            self._layer_probe_cache = {}

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

    def forward(self, video_tensor, mask, return_features=False, return_layer_features=False):
        frame_features = self.extract_frame_features(video_tensor, mask)
        temporal_features = self.encode_temporal_features(frame_features, mask)
        shared_features = self.pool_video_features(temporal_features, mask)
        bdi_pred = self.reg_task_head(shared_features).squeeze(-1)
        ordinal_logits = None
        if self.ordinal_task_head is not None:
            ordinal_logits = self.ordinal_task_head(shared_features)

        layer_features = None
        if return_layer_features:
            layer_features = self._collect_layer_features(
                frame_features=frame_features,
                temporal_features=temporal_features,
                shared_features=shared_features,
                mask=mask,
            )

        return MTLLiteOutput(
            bdi_pred=bdi_pred,
            ordinal_logits=ordinal_logits,
            shared_features=shared_features if return_features else None,
            layer_features=layer_features,
        )

    def _collect_layer_features(self, frame_features, temporal_features, shared_features, mask):
        """Build per-layer video-level embeddings for the layer-wise probe.

        Hook-captured backbone activations are accumulated in valid-frame order
        (see :meth:`extract_frame_features`), so they are sliced back to
        per-video vectors using each video's valid-frame count and pooled.

        ``layer_temporal`` and ``layer_shared`` are taken from the inline
        forward results (no hook), and ``layer_backbone_out`` falls back to the
        already-computed ``frame_features`` when the backbone hook captured
        nothing (e.g. when the backbone was called in a code path the hook did
        not cover).  Layers with no usable data are omitted from the result.
        """
        layer_features: Dict[str, torch.Tensor] = {}

        # Always available inline results.
        layer_features["layer_temporal"] = shared_features.detach()
        layer_features["layer_shared"] = shared_features.detach()

        # layer_backbone_out: prefer the frame_features already pooled under the
        # mask (reliable, identical to extract_frame_features output).
        layer_features["layer_backbone_out"] = masked_mean_pool(
            frame_features, mask
        ).detach()

        # Hooked backbone sub-modules (layer_stem, layer_block_*).
        if self._layer_probe_active and self._layer_probe_cache:
            valid_counts = mask.reshape(mask.size(0), -1).sum(dim=1).long().tolist()
            for layer_name, cached in self._layer_probe_cache.items():
                if layer_name in ("layer_temporal", "layer_shared", "layer_backbone_out"):
                    continue
                pooled = self._pool_cached_layer(cached, valid_counts, shared_features)
                if pooled is not None:
                    layer_features[layer_name] = pooled

        return layer_features

    def _pool_cached_layer(self, cached, valid_counts, reference):
        """Slice an accumulated hook cache into per-video pooled vectors.

        Args:
            cached: Tensor of shape (total_valid_frames, ...) accumulated by the
                hook in valid-frame order.
            valid_counts: List of per-video valid frame counts, summing to
                ``cached.size(0)``.
            reference: A tensor whose device/dtype the output should match.
        """
        if cached is None or cached.size(0) == 0:
            return None
        total_cached = cached.size(0)
        total_expected = int(sum(valid_counts))
        if total_cached != total_expected:
            # Frame-count mismatch means the hook did not capture every chunk
            # (e.g. an exception mid-loop).  Skip this layer rather than emit
            # misaligned embeddings.
            print(
                f"[LAYER-PROBE] Frame count mismatch for a hooked layer: "
                f"cached={total_cached} expected={total_expected}; skipping."
            )
            return None

        vectors = []
        cursor = 0
        feature_dim = cached.reshape(cached.size(0), -1).size(-1)
        for count in valid_counts:
            count = int(count)
            if count <= 0:
                # No valid frames: emit a zero vector matching this layer's dim.
                vectors.append(torch.zeros(feature_dim, device=reference.device, dtype=reference.dtype))
                continue
            segment = cached[cursor:cursor + count]
            cursor += count
            vectors.append(self._pool_layer_frame_features(segment, None))
        return torch.stack(vectors, dim=0).to(device=reference.device, dtype=reference.dtype)

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
