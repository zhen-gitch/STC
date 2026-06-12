"""
Backbone construction utilities shared by end-to-end training and diagnostics.
"""

from pathlib import Path

import timm
import torch

from src.models.iresnet import iresnet50


TRANSFORMER_BACKBONE_PREFIXES = (
    "beit",
    "deit",
    "eva",
    "levit",
    "pit",
    "swin",
    "twins",
    "vit",
)


def _clean_state_dict_keys(state_dict):
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key
        for prefix in ("module.", "model.", "backbone."):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix):]
        cleaned_state_dict[cleaned_key] = value
    return cleaned_state_dict


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "backbone"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _load_optional_weights(model, model_name, weight_path):
    if not weight_path:
        print(f"[BACKBONE] {model_name}: no external weight path; using initialized model weights.")
        return model

    resolved_weight_path = Path(str(weight_path)).expanduser()
    if not resolved_weight_path.exists():
        print(f"[WARNING] Backbone weight file not found: {resolved_weight_path}")
        return model

    checkpoint = torch.load(resolved_weight_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported backbone checkpoint format: {resolved_weight_path}")

    cleaned_state_dict = _clean_state_dict_keys(state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"[BACKBONE] Loaded external weights for {model_name}: {resolved_weight_path}")
    if missing_keys:
        print(f"[BACKBONE] Missing keys while loading {model_name}: {len(missing_keys)}")
    if unexpected_keys:
        print(f"[BACKBONE] Unexpected keys while loading {model_name}: {len(unexpected_keys)}")
    return model


def _is_transformer_like_backbone(model_name):
    normalized_name = model_name.lower()
    return normalized_name.startswith(TRANSFORMER_BACKBONE_PREFIXES)


def _create_timm_backbone(model_name, timm_pretrained=False, img_size=112):
    create_kwargs = {
        "model_name": model_name,
        "pretrained": bool(timm_pretrained),
        "num_classes": 0,
    }

    if _is_transformer_like_backbone(model_name):
        try:
            return timm.create_model(**create_kwargs, img_size=img_size)
        except TypeError as exc:
            if "img_size" not in str(exc):
                raise
            print(f"[BACKBONE] {model_name} does not accept img_size; falling back to timm defaults.")

    return timm.create_model(**create_kwargs)


def build_feature_backbone(model_name, weight_path=None, timm_pretrained=True, img_size=112):
    """
    Create a visual backbone that returns feature embeddings instead of logits.

    `iresnet50` uses the local implementation. Other names are delegated to
    `timm.create_model(..., num_classes=0)` so modern timm backbones can be
    tested by config.
    """
    try:
        if model_name == "iresnet50":
            model = iresnet50()
        else:
            model = _create_timm_backbone(
                model_name=model_name,
                timm_pretrained=timm_pretrained,
                img_size=img_size,
            )

        return _load_optional_weights(model, model_name, weight_path)

    except RuntimeError:
        raise
    except Exception as exc:
        print(f"[Error] Failed to create backbone '{model_name}' with timm/local factory.")
        raise exc
