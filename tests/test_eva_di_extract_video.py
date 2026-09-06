"""extract_video: exact-zero black guard, feature remap, boolean-mask dtype.

Pins the E2E-caught P0: in the pinned env ``uint8_tensor.any(dim=1)`` returns
uint8; as a mask it degrades into integer fancy-indexing and silently
collapses features onto one row.  The amax()>0 guard must produce a bool
frame_valid with a correct per-frame feature remap.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.eva_di.contracts import EvaDiSchemaError
from src.eva_di.extract.runner import extract_video
from src.eva_di.of3_registry import load_video_record
from src.eva_di.paths import PathSet

from tests._eva_di_synth import build_tree


def _paths(tree, tmp_path):
    return PathSet.build(overrides={
        "of3_root": tree.of3_root, "avec_root": tree.avec_root,
        "split_file": tree.split_file, "label_dir": tree.label_dir,
        "cache_root": tmp_path / "fresh_cache", "weight_path": tree.weight_path,
        "behavior_norm_root": tree.norm_root,
        "log_root": tmp_path / "logs", "output_root": tmp_path / "outputs",
    })


def _write_aligned(tree, sample_id, black_rows):
    """Write real 112x112 jpgs (PIL round-trip) for every row of the video."""
    import csv as _csv
    csv_path = tree.of3_root / "samples" / sample_id / "features.csv"
    rows = list(_csv.DictReader(csv_path.open(encoding="utf-8")))
    for i, row in enumerate(rows):
        if not row["aligned_image_path"]:
            continue
        value = 0 if i in black_rows else 200
        path = tree.of3_root / row["aligned_image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((112, 112, 3), value, dtype=np.uint8)).save(path)


class _FakeEncoder:
    """encode(N kept frames) -> cls=row+1, gap=row+10 (identity fingerprint)."""

    def encode(self, frames):
        n = int(frames.shape[0])
        rows = torch.arange(n, dtype=torch.float32)
        cls = (rows + 1).unsqueeze(1).repeat(1, 384)
        gap = (rows + 10).unsqueeze(1).repeat(1, 384)
        return cls, gap


SID = "203_1_Freeform_video"


def test_clean_video_maps_features_identity_and_keeps_bool_mask(tmp_path):
    tree = build_tree(tmp_path / "synth", frames_per_video=6)
    paths = _paths(tree, tmp_path)
    _write_aligned(tree, SID, black_rows=set())
    sel, info = extract_video(paths, SID, _FakeEncoder(), verify_image_sha=False)
    assert sel.frame_valid.dtype == np.dtype(bool)
    assert sel.frame_valid.tolist() == [True] * 6
    assert info["n_black"] == 0
    assert np.allclose(sel.cls, np.arange(6, dtype=np.float32)[:, None] + 1)
    assert np.allclose(sel.gap, np.arange(6, dtype=np.float32)[:, None] + 10)
    record = load_video_record(tree.of3_root, SID)
    assert sel.frame_index_zero.tolist() == record.frame_index_zero.tolist()


def test_black_frames_trip_the_per_video_ratio_guard(tmp_path):
    tree = build_tree(tmp_path / "synth", frames_per_video=6)
    paths = _paths(tree, tmp_path)
    _write_aligned(tree, SID, black_rows={4})  # 1/6 >> MAX_INVALID_IMAGE_RATIO=0.01
    with pytest.raises(EvaDiSchemaError, match="invalid/black frame ratio"):
        extract_video(paths, SID, _FakeEncoder(), verify_image_sha=False)


def test_alignment_failed_blank_artifact_rows_are_masked_not_fatal(tmp_path):
    tree = build_tree(tmp_path / "synth", frames_per_video=6,
                      blank_schema_rows={SID: {3, 4}})
    record = load_video_record(tree.of3_root, SID)
    assert record.n_rows == 6
    assert record.image_valid.tolist() == [True, True, True, False, False, True]
    assert record.behavior_valid.tolist() == [True, True, True, False, False, True]
    assert record.pair_valid[3] is np.False_ or record.pair_valid[3] == False
    assert np.isfinite(record.behavior).all()  # masked rows are zero-filled
    assert record.aligned_image_path[3] == "" and record.aligned_image_sha256[3] == ""


def test_empty_schema_with_image_valid_true_is_fatal(tmp_path):
    import csv as _csv
    tree = build_tree(tmp_path / "synth", frames_per_video=6)
    csv_path = tree.of3_root / "samples" / SID / "features.csv"
    rows = list(_csv.DictReader(csv_path.open(encoding="utf-8")))
    for name in rows[0].keys():
        if name == "aligned_mask_schema":
            rows[0][name] = ""  # image_valid stays True -> partial artifacts
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    from src.eva_di.contracts import EvaDiSchemaError as _E
    with pytest.raises(_E, match="aligned_mask_schema"):
        load_video_record(tree.of3_root, SID)
