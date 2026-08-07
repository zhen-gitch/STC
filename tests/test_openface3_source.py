import csv
import hashlib
import json

import pytest
import torch
from omegaconf import OmegaConf
from torchvision.io import write_jpeg

from src.datasets.dataset import AVECDataset
from src.datasets.openface3_source import OpenFace3Source


def _write_image(path, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = torch.full((3, 16, 16), value, dtype=torch.uint8)
    write_jpeg(image, str(path), quality=90)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(root, video_id="001_1_Freeform_video"):
    sample = root / "samples" / video_id
    rows = []
    for frame in (1, 2, 3, 5):
        rel = f"samples/{video_id}/aligned/frame_{frame:06d}.jpg"
        digest = _write_image(root / rel, frame)
        rows.append(
            {
                "sample_id": video_id,
                "frame_index_one": str(frame),
                "aligned_image_path": rel,
                "aligned_image_sha256": digest,
                "image_valid": "True",
            }
        )
    sample.mkdir(parents=True, exist_ok=True)
    with (sample / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_openface3_source_chooses_longest_contiguous_run(tmp_path):
    root = tmp_path / "openface3"
    _write_source(root)
    source = OpenFace3Source(root, min_retained_valid_ratio=0.75)
    entries = source.frame_entries("001_1_Freeform_video")
    assert [entry.frame_index_one for entry in entries] == [1, 2, 3]
    assert all(entry.path.is_file() for entry in entries)


def test_openface3_source_rejects_hash_mismatch(tmp_path):
    root = tmp_path / "openface3"
    _write_source(root)
    source = OpenFace3Source(root, min_retained_valid_ratio=0.75)
    entry = source.frame_entries("001_1_Freeform_video")[0]
    entry.path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        source.verify_frame(entry)


def test_dataset_openface3_preserves_rgb_contract(tmp_path):
    root = tmp_path / "openface3"
    _write_source(root)
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "001_1_Depression.csv").write_text("12", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"train": ["001_1_Freeform_video"], "val": [], "test": []}),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "LABEL_DIR": str(labels),
            "IMAGE_DIR": str(tmp_path / "legacy-unused"),
            "DATASET_SPLIT_FILE": str(split),
            "PROCESS_TEMPORAL": {
                "CLASS_STEP": 2,
                "SAMPLE_STEP": 1,
                "MAX_SEQ_LEN": 4,
            },
            "DATASET": {
                "IMAGE_SOURCE": "openface3",
                "OPENFACE3_ROOT": str(root),
                "OPENFACE3_VERIFY_HASHES": True,
                "OPENFACE3_MIN_CONTIGUOUS_FRAMES": 2,
                "OPENFACE3_MIN_RETAINED_VALID_RATIO": 0.75,
                "RETURN_MULTI_VIEW_TRAIN": False,
                "INPUT_VARIANT": "rgb",
                "PHOTOMETRIC_NORMALIZATION": {"MODE": "none"},
            },
        }
    )
    dataset = AVECDataset(cfg, "train")
    video, mask, metadata = dataset[0]
    assert video.shape == (4, 3, 112, 112)
    assert mask.tolist() == [True, True, True, False]
    assert metadata["video_id"] == "001_1_Freeform_video"
    assert torch.isfinite(video).all()


def test_dataset_defaults_to_legacy_source(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"train": [], "val": [], "test": []}), encoding="utf-8"
    )
    cfg = OmegaConf.create(
        {
            "LABEL_DIR": str(tmp_path / "labels"),
            "IMAGE_DIR": str(tmp_path / "images"),
            "DATASET_SPLIT_FILE": str(split),
            "PROCESS_TEMPORAL": {
                "CLASS_STEP": 2,
                "SAMPLE_STEP": 1,
                "MAX_SEQ_LEN": 4,
            },
            "DATASET": {
                "RETURN_MULTI_VIEW_TRAIN": False,
                "INPUT_VARIANT": "rgb",
                "PHOTOMETRIC_NORMALIZATION": {"MODE": "none"},
            },
        }
    )
    dataset = AVECDataset(cfg, "train")
    assert dataset.image_source == "legacy"
    assert dataset.openface3_source is None


def test_openface3_split_accepts_legacy_aligned_suffix(tmp_path):
    root = tmp_path / "openface3"
    _write_source(root)
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {"train": ["001_1_Freeform_video_aligned"], "val": [], "test": []}
        ),
        encoding="utf-8",
    )
    from src.datasets.dataset import load_video_ids

    assert load_video_ids(split, "train") == ["001_1_Freeform_video"]
