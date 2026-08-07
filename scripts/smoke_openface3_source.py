#!/usr/bin/env python3
"""Run a bounded OpenFace3 RGB Dataset and model-contract smoke."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.datasets.dataset import AVECDataset


class DummyBackbone(nn.Module):
    def __init__(self, output_dim: int = 8) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        pooled = tensor.mean(dim=(2, 3))
        repeat_count = (self.output_dim + pooled.size(1) - 1) // pooled.size(1)
        return pooled.repeat(1, repeat_count)[:, : self.output_dim]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video-id", default="206_1_Freeform_video")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _model_config(dataset_config):
    return OmegaConf.create(
        {
            "EXTRACT_FEATURE": {
                "MAX_SCORE": 63,
                "MODEL_NAME": "dummy_backbone",
                "MODEL_WEIGHT_PATH": None,
                "TIMM_PRETRAINED": False,
                "CHUNK_SIZE": 4,
                "FREEZE_BACKBONE": False,
                "FINETUNE_LAST_N_BLOCKS": 0,
            },
            "BACKBONE_OUT_DIMS": {"dummy_backbone": 8},
            "PROCESS_TEMPORAL": {
                "HIDDEN_DIM": 8,
                "CLASS_STEP": 2,
                "DROPOUT": 0.0,
                "LEARNING_RATE": 1e-4,
                "WEIGHT_DECAY": 0.0,
            },
            "LOSSES": {
                "ORDINAL_WEIGHT": 0.0,
                "CCC_WEIGHT": 0.0,
                "L1_WEIGHT": 0.0,
                "L2_WEIGHT": 0.0,
            },
            "MODEL": {"AUXILIARY_TASKS": {"ORDINAL_CLASSIFICATION": False}},
            "DATASET": dataset_config,
        }
    )


def main() -> int:
    args = parse_args()
    pilot_config = OmegaConf.load(args.config)
    root = pilot_config.DATASET.OPENFACE3_ROOT
    if not root:
        raise ValueError("Pilot config must define DATASET.OPENFACE3_ROOT")

    with tempfile.TemporaryDirectory(prefix="openface3-source-smoke-") as temp:
        temp_root = Path(temp)
        labels = temp_root / "labels"
        labels.mkdir()
        subject_id = args.video_id[:5]
        (labels / f"{subject_id}_Depression.csv").write_text(
            "12", encoding="utf-8"
        )
        split = temp_root / "split.json"
        split.write_text(
            json.dumps(
                {"train": [args.video_id], "val": [], "test": []}
            ),
            encoding="utf-8",
        )
        dataset_config = OmegaConf.create(
            {
                "IMAGE_SOURCE": "openface3",
                "OPENFACE3_ROOT": root,
                "OPENFACE3_VERIFY_HASHES": bool(
                    pilot_config.DATASET.get("OPENFACE3_VERIFY_HASHES", True)
                ),
                "OPENFACE3_MIN_CONTIGUOUS_FRAMES": int(
                    pilot_config.DATASET.get("OPENFACE3_MIN_CONTIGUOUS_FRAMES", 2)
                ),
                "OPENFACE3_MIN_RETAINED_VALID_RATIO": float(
                    pilot_config.DATASET.get(
                        "OPENFACE3_MIN_RETAINED_VALID_RATIO", 0.95
                    )
                ),
                "RETURN_MULTI_VIEW_TRAIN": False,
                "INPUT_VARIANT": "rgb",
                "PHOTOMETRIC_NORMALIZATION": {"MODE": "none"},
            }
        )
        cfg = _model_config(dataset_config)
        cfg.LABEL_DIR = str(labels)
        cfg.IMAGE_DIR = str(temp_root / "legacy-unused")
        cfg.DATASET_SPLIT_FILE = str(split)
        cfg.PROCESS_TEMPORAL.SAMPLE_STEP = 1
        cfg.PROCESS_TEMPORAL.MAX_SEQ_LEN = 4

        dataset = AVECDataset(cfg, "train")
        video, mask, metadata = dataset[0]
        if video.ndim != 4 or tuple(video.shape[1:]) != (3, 112, 112):
            raise AssertionError(f"Unexpected Dataset tensor shape: {tuple(video.shape)}")

        import src.models.mtl_lite as mtl_lite

        mtl_lite.build_feature_backbone = lambda **_: DummyBackbone(8)
        model = mtl_lite.MTLLiteDepressionModel(cfg)
        output = model(video.unsqueeze(0), mask.unsqueeze(0))
        loss = output.bdi_pred.square().mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError("OpenFace3 smoke produced invalid gradients")

        result = {
            "accepted": True,
            "source": "openface3",
            "video_id": metadata["video_id"],
            "tensor_shape": list(video.shape),
            "mask_count": int(mask.sum().item()),
            "prediction_shape": list(output.bdi_pred.shape),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "gradient_count": len(gradients),
        }

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
