import math
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from src.metrics.metrics import ConcordanceCorrCoefMetric, concordance_ccc_loss
from src.models.backbone_factory import build_feature_backbone
from src.models.gradient_reversal import GradientReversalLayer
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

        # Stage B1: identity-adversarial state.  All switches default off so
        # the default RGB baseline (E0) is bit-identical when the config does
        # not enable ``MODEL.IDENTITY_ADVERSARIAL``.  The subject-id head is
        # built lazily by :meth:`set_subject_index` once the runner injects a
        # train-only subject table; until then ``forward`` emits no
        # ``identity_logits`` and ``compute_losses`` emits no identity loss.
        self.identity_adversarial = bool(
            _get_nested_config_value(
                configs, "MODEL", "IDENTITY_ADVERSARIAL", "ENABLE", False
            )
        )
        self.lambda_id = float(
            _get_nested_config_value(
                configs, "MODEL", "IDENTITY_ADVERSARIAL", "LAMBDA_ID", 0.05
            )
        )
        self.grl: Optional[GradientReversalLayer] = (
            GradientReversalLayer(self.lambda_id) if self.identity_adversarial else None
        )
        self.subject_id_head: Optional[nn.Module] = None
        self.subject_id_to_index: Dict[str, int] = {}

        # Stage B2: severity-balanced regression.  Default off so the default
        # baseline (E0) uses plain MSE.  When on, only the regression MSE term
        # is reweighted by per-bin weights derived from the train split; CCC
        # and ordinal auxiliary losses are deliberately left unweighted (see
        # docs/MTL_LITE_DESIGN.md section 13.5).  Bin edges reuse
        # ``src/diagnostics/io.py:severity_group`` (<=13 / <=19 / <=28 / >28)
        # so training-time reweighting matches the A4/severity diagnostics.
        self.severity_balanced_regression = bool(
            _get_nested_config_value(
                configs, "MODEL", "SEVERITY_BALANCED_REGRESSION", "ENABLE", False
            )
        )
        self.severity_power = float(
            _get_nested_config_value(
                configs, "MODEL", "SEVERITY_BALANCED_REGRESSION", "POWER", 0.5
            )
        )
        self.severity_min_weight = float(
            _get_nested_config_value(
                configs, "MODEL", "SEVERITY_BALANCED_REGRESSION", "MIN_WEIGHT", 0.5
            )
        )
        self.severity_max_weight = float(
            _get_nested_config_value(
                configs, "MODEL", "SEVERITY_BALANCED_REGRESSION", "MAX_WEIGHT", 4.0
            )
        )
        # Edges configurable but default to the diagnostics-standard boundaries.
        cfg_edges = _get_nested_config_value(
            configs, "MODEL", "SEVERITY_BALANCED_REGRESSION", "EDGES", [13, 19, 28]
        )
        self.severity_bin_edges = [int(e) for e in list(cfg_edges)]
        self.severity_bin_names = ("minimal", "mild", "moderate", "severe")
        # Injected by the runner from the train split; empty dict => disabled
        # even if the switch is on (no train stats yet).
        self.severity_bin_weights: Dict[str, float] = {}

    def set_subject_index(self, subject_id_to_index):
        """Inject the train-only subject table and build the attacker head.

        Called by the runner after constructing the model, using a subject
        table built exclusively from the train split.  Val/test unseen
        subjects are deliberately absent from the table so that batches
        containing them produce no identity loss (only diagnostic use).

        No-op when ``identity_adversarial`` is disabled, so the default
        baseline path is unaffected.
        """
        self.subject_id_to_index = dict(subject_id_to_index or {})
        if not self.identity_adversarial or not self.subject_id_to_index:
            return
        num_subject_classes = len(self.subject_id_to_index)
        # Mirror the regression head structure (Linear -> LayerNorm -> GELU ->
        # Linear) but emit ``num_subject_classes`` logits for plain CE.  Built
        # lazily so the optimizer (configured during ``trainer.fit``) sees it.
        self.subject_id_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, num_subject_classes),
        )

    def _map_subjects_to_index(self, subject_ids):
        """Map a batch of subject ids to a LongTensor of class indices.

        Returns ``None`` when any subject in the batch is absent from the
        train-only table (e.g. val/test unseen subjects, or the table was
        never injected).  Callers treat ``None`` as "skip identity loss for
        this batch" rather than an error.
        """
        if not self.subject_id_to_index:
            return None
        indices = []
        for sid in subject_ids:
            idx = self.subject_id_to_index.get(str(sid))
            if idx is None:
                return None
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)

    # ------------------------------------------------------------------
    # Stage B2: severity-bin reweighting for the regression MSE term.
    # ------------------------------------------------------------------

    def set_severity_bin_counts(self, bin_counts):
        """Inject train-only severity bin counts and compute the weight table.

        Args:
            bin_counts: Mapping from bin name (``minimal``/``mild``/``moderate``
                /``severe``) to train-split sample count.  Built by the runner
                from train labels only.

        Computes ``weight = (total / (num_bins * count_bin)) ** power`` per bin,
        clips to ``[MIN_WEIGHT, MAX_WEIGHT]``, then mean-normalizes so the
        average per-sample weight is 1.0 (keeping the regression loss on the
        same scale as plain MSE).  The full table (count, raw, clipped,
        normalized) is printed for log traceability.  No-op when the switch is
        off.
        """
        if not self.severity_balanced_regression:
            self.severity_bin_weights = {}
            return
        counts = {name: int(bin_counts.get(name, 0)) for name in self.severity_bin_names}
        total = sum(counts.values())
        num_bins = len(self.severity_bin_names)
        if total <= 0 or any(c <= 0 for c in counts.values()):
            # Empty train split or a degenerate bin would produce inf/nan
            # weights; refuse to weight rather than corrupt the gradient.
            print(
                f"[SEVERITY-BALANCE] Incomplete train bin counts {counts}; "
                f"falling back to plain MSE (weights disabled)."
            )
            self.severity_bin_weights = {}
            return

        raw = {
            name: (total / (num_bins * counts[name])) ** self.severity_power
            for name in self.severity_bin_names
        }
        clipped = {
            name: min(max(raw[name], self.severity_min_weight), self.severity_max_weight)
            for name in self.severity_bin_names
        }
        mean_clipped = sum(clipped.values()) / num_bins
        normalized = {name: clipped[name] / mean_clipped for name in self.severity_bin_names}
        self.severity_bin_weights = normalized
        print("[SEVERITY-BALANCE] Train-only severity-bin weight table:")
        for name in self.severity_bin_names:
            print(
                f"  {name:>9s}: count={counts[name]:4d}  raw={raw[name]:.4f}  "
                f"clipped={clipped[name]:.4f}  normalized={normalized[name]:.4f}"
            )

    def _severity_bin_index(self, true_bdi):
        """Map a tensor of raw BDI scores to per-sample bin indices.

        Bin edges come from ``self.severity_bin_edges`` (default ``[13, 19,
        28]``), matching :func:`src.diagnostics.io.severity_group`.  Returns a
        ``LongTensor`` of bin indices in ``[0, num_bins)``.
        """
        scores = true_bdi.float()
        # torch.bucketize with right=False assigns score -> index of first edge
        # strictly greater than it: score<=edges[0] -> 0, score==edges[0] -> 0,
        # score in (edges[0], edges[1]] -> 1, etc.  This matches the ``<=``
        # semantics of src.diagnostics.io.severity_group.
        return torch.bucketize(
            scores,
            torch.tensor(self.severity_bin_edges, device=scores.device, dtype=scores.dtype),
            right=False,
        )

    def _severity_weighted_mse(self, bdi_pred, true_bdi_norm, true_bdi):
        """Severity-bin reweighted MSE over the batch.

        ``loss = mean( weight(bin(y_i)) * (pred_i - y_i_norm)^2 )``.  When the
        weight table is empty (switch off or no train stats), this falls back
        to plain ``F.mse_loss`` so the result is bit-identical to E0.
        """
        if not self.severity_bin_weights:
            return F.mse_loss(bdi_pred.float(), true_bdi_norm.float())
        bin_idx = self._severity_bin_index(true_bdi).to(bdi_pred.device)
        weight_table = torch.tensor(
            [self.severity_bin_weights[name] for name in self.severity_bin_names],
            device=bdi_pred.device,
            dtype=bdi_pred.dtype,
        )
        per_sample_w = weight_table[bin_idx]
        sq_err = (bdi_pred.float() - true_bdi_norm.float()) ** 2
        return (per_sample_w * sq_err).mean()

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

        # Stage B1: identity-adversarial branch.  Only active when the switch
        # is on AND a train-only subject index has been injected (head built).
        # The GRL sits between ``shared_features`` and the attacker head so the
        # head learns to predict subject id while the reversed gradient pushes
        # ``z_dep`` away from identity.  Disabled path leaves identity_logits
        # as None and the baseline forward unchanged.
        identity_logits = None
        if self.identity_adversarial and self.subject_id_head is not None:
            identity_logits = self.subject_id_head(self.grl(shared_features))

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
            identity_logits=identity_logits,
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

    def compute_losses(self, outputs, labels, stage="train"):
        true_bdi, true_bdi_norm, ordinal_levels = self.prepare_labels(labels)
        # Stage B2: regression MSE is severity-bin reweighted when the switch
        # is on and a train-only weight table has been injected; otherwise
        # falls back to plain MSE (bit-identical to E0).  CCC and ordinal are
        # deliberately NOT reweighted (docs/MTL_LITE_DESIGN.md section 13.5).
        loss_reg = self._severity_weighted_mse(outputs.bdi_pred, true_bdi_norm, true_bdi)
        loss_ccc = concordance_ccc_loss(outputs.bdi_pred, true_bdi_norm)
        loss_ord = None
        if outputs.ordinal_logits is not None and self.ordinal_weight > 0.0:
            loss_ord = coral_loss(outputs.ordinal_logits, ordinal_levels)

        # Stage B1: identity-adversarial CE.  Gated by (a) the head having
        # produced logits, (b) train stage only -- val/test BDI loss must stay
        # comparable to E0, and (c) every subject in the batch mapping to the
        # train-only table.  ``lambda_id`` is already applied inside the GRL,
        # so the CE term is added without an extra weight.  When any gate
        # fails, ``loss_id`` stays None and ``total`` is bit-identical to E0.
        loss_id = None
        if outputs.identity_logits is not None and stage == "train":
            subject_index = self._map_subjects_to_index(labels["subject_id"])
            if subject_index is not None:
                subject_index = subject_index.to(outputs.identity_logits.device)
                loss_id = F.cross_entropy(outputs.identity_logits, subject_index)

        total = loss_reg + self.ccc_loss_weight * loss_ccc
        if loss_ord is not None:
            total = total + self.ordinal_weight * loss_ord
        if loss_id is not None:
            total = total + loss_id
        return MTLLiteLosses(
            total=total,
            regression=loss_reg,
            ordinal=loss_ord,
            ccc=loss_ccc,
            identity=loss_id,
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
        losses = self.compute_losses(outputs, labels, stage=stage)
        true_bdi = labels["bdi_score"].float()
        current_bs = video_tensor.size(0)

        self._update_metrics(stage, outputs.bdi_pred, true_bdi)
        self.log(f"{stage}_loss", losses.total, on_epoch=True, on_step=False, batch_size=current_bs)
        self.log(f"{stage}_reg_loss", losses.regression, on_epoch=True, on_step=False, batch_size=current_bs)
        if losses.ordinal is not None:
            self.log(f"{stage}_ordinal_loss", losses.ordinal, on_epoch=True, on_step=False, batch_size=current_bs)
        if losses.ccc is not None:
            self.log(f"{stage}_ccc_loss", losses.ccc, on_epoch=True, on_step=False, batch_size=current_bs)
        if losses.identity is not None:
            self.log(f"{stage}_identity_loss", losses.identity, on_epoch=True, on_step=False, batch_size=current_bs)
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
