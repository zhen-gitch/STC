"""behavior: pinned-stats loading, masking semantics, train-only consistency."""

import json

import numpy as np
import pytest

from src.eva_di.behavior import assert_stats_consistency, load_pinned_stats, recompute_train_stats
from src.eva_di.contracts import EvaDiFingerprintError, EvaDiSchemaError
from src.eva_di.of3_registry import load_video_record

from tests._eva_di_synth import build_tree


@pytest.fixture()
def tree(tmp_path):
    return build_tree(tmp_path / "synth", frames_per_video=6)


def _train_records(tree):
    return [load_video_record(tree.of3_root, s) for s in tree.splits["train"]]


def test_pinned_stats_load_and_axis_lock(tree):
    stats = load_pinned_stats(tree.norm_root)
    assert stats.mean.shape == stats.std.shape == (206,)
    assert stats.count > 0
    assert stats.feature_names[0] == "landmark_00_x_norm"


def test_normalize_masks_invalid_rows(tree):
    stats = load_pinned_stats(tree.norm_root)
    record = load_video_record(tree.of3_root, "203_1_Freeform_video")
    valid = record.behavior_valid.copy()
    valid[0] = False
    out = stats.normalize(record.behavior, valid)
    assert np.all(out[0] == 0.0)
    assert np.isfinite(out[valid]).all()


def test_landmark_axes_ship_in_pinned_norm_space(tree):
    """E2E-caught P0: registry must emit 2*v/111-1, never raw 112-px pixels."""
    import csv as _csv
    csv_path = tree.of3_root / "samples" / "203_1_Freeform_video" / "features.csv"
    row = next(iter(_csv.DictReader(csv_path.open(encoding="utf-8"))))
    raw = np.asarray(json.loads(row["landmarks_98_xy_aligned"]), dtype=np.float32)
    record = load_video_record(tree.of3_root, "203_1_Freeform_video")
    # float32 ARRAY ufunc path (per-step f32 rounding), identical to the
    # generator's op order in trans clips.py:851
    expected = (2.0 * raw / 111.0 - 1.0).reshape(-1)
    assert np.array_equal(record.behavior[0, :196], expected)
    assert np.abs(record.behavior[:, :196]).max() < 3.0  # normalized magnitudes
    assert np.isclose(record.behavior[0, 196], np.float32(float(row["gaze_angle_x"])))  # gaze raw


def test_consistency_passes_for_train_only_recompute(tree):
    stats = load_pinned_stats(tree.norm_root)
    assert_stats_consistency(stats, _train_records(tree))  # no raise


def test_val_frames_never_shift_stats(tree):
    """Leakage guard: including val rows in the fit must trip the assertion."""
    stats = load_pinned_stats(tree.norm_root)
    polluted = _train_records(tree) + [load_video_record(tree.of3_root, "209_1_Freeform_video")]
    with pytest.raises(EvaDiFingerprintError):
        assert_stats_consistency(stats, polluted)


def test_tampered_mean_rejected(tree):
    stats = load_pinned_stats(tree.norm_root)
    tampered_mean = stats.mean.copy()
    tampered_mean[5] += 0.5
    tampered = type(stats)(mean=tampered_mean, std=stats.std, count=stats.count,
                           source_sha256=stats.source_sha256,
                           valid_frame_count=stats.valid_frame_count,
                           feature_names=stats.feature_names)
    with pytest.raises(EvaDiFingerprintError):
        assert_stats_consistency(tampered, _train_records(tree))


def test_wrong_schema_json_rejected(tmp_path, tree):
    payload = json.loads((tree.norm_root / "technical.json").read_text(encoding="utf-8"))
    payload["split"] = "all"
    (tree.norm_root / "technical.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaDiFingerprintError):
        load_pinned_stats(tree.norm_root)


def test_recompute_nonfinite_rejected(tree):
    stats = load_pinned_stats(tree.norm_root)
    records = _train_records(tree)
    kwargs = {k: getattr(records[0], k) for k in
              ("sample_id", "split", "n_rows", "frame_index_zero", "image_valid",
               "behavior_valid", "pair_valid", "aligned_image_path",
               "aligned_image_sha256", "features_sha256")}
    kwargs["behavior"] = records[0].behavior.copy()
    kwargs["behavior_valid"] = records[0].behavior_valid.copy()
    broken = type(records[0])(**kwargs)
    broken.behavior[0, 0] = np.nan  # non-finite sneaking through a broken mask
    broken.behavior_valid[0] = True
    with pytest.raises(EvaDiSchemaError):
        recompute_train_stats([broken])
