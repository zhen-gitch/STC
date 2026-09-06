"""dataset: split/label contracts, four-source index, prefix-run rule, train-only stats."""

import json

import numpy as np
import pytest

from src.eva_di.behavior import load_pinned_stats
from src.eva_di.contracts import SPLIT_ID_SUFFIX, EvaDiPathError, EvaDiSchemaError
from src.eva_di.dataset import (
    build_recording_index,
    load_bdi_score,
    load_split_ids,
)
from src.eva_di.paths import PathSet

from tests._eva_di_synth import build_cache_for_tree, build_tree

BAD_HOLE = "205_2_Freeform_video"  # train video with an interior behavior-invalid row


def _paths(tree, tmp_path, **extra):
    overrides = {
        "of3_root": tree.of3_root, "avec_root": tree.avec_root,
        "split_file": tree.split_file, "label_dir": tree.label_dir,
        "cache_root": tree.cache_root, "weight_path": tree.weight_path,
        "behavior_norm_root": tree.norm_root,
        "log_root": tmp_path / "logs", "output_root": tmp_path / "outputs",
    }
    overrides.update(extra)
    return PathSet.build(overrides=overrides)


@pytest.fixture()
def holed_tree(tmp_path):
    return build_tree(
        tmp_path / "synth", frames_per_video=7,
        bad_au_rows={BAD_HOLE: {3}},
        invalid_image_rows={"203_1_Freeform_video": {5}},
    )


def test_load_split_ids_strips_suffix_and_sorts(tree_splits):
    out = load_split_ids(tree_splits)
    assert out["train"] == sorted(["203_1_Freeform_video", "205_2_Freeform_video",
                                   "207_2_Northwind_video"])
    assert all(not i.endswith(SPLIT_ID_SUFFIX) for ids in out.values() for i in ids)


def _write_split(path, splits):
    path.write_text(json.dumps({s: list(v) for s, v in splits.items()}), encoding="utf-8")
    return path


def test_split_file_contract_violations(tmp_path):
    good = {"train": ["a_aligned"], "val": ["b_aligned"], "test": ["c_aligned"]}
    base = {"train": ["a"], "val": ["b_aligned"], "test": ["c_aligned"]}
    dup = {"train": ["a_aligned", "a_aligned"], "val": ["b_aligned"], "test": ["c_aligned"]}
    over = {"train": ["a_aligned"], "val": ["a_aligned"], "test": ["c_aligned"]}
    for payload, match in ((base, "suffix"), (dup, "duplicate"), (over, "overlap")):
        with pytest.raises(EvaDiSchemaError, match=match):
            load_split_ids(_write_split(tmp_path / f"s{match}.json", payload))


def test_bdi_score_label_contract(tmp_path):
    (tmp_path / "203_1_Depression.csv").write_text("33\n", encoding="utf-8")
    assert load_bdi_score(tmp_path, "203_1_Freeform_video") == 33.0
    (tmp_path / "204_1_Depression.csv").write_text("severe", encoding="utf-8")
    with pytest.raises(EvaDiSchemaError):
        load_bdi_score(tmp_path, "204_1_Freeform_video")
    with pytest.raises(EvaDiPathError):
        load_bdi_score(tmp_path, "999_9_Freeform_video")


def test_recording_index_end_to_end(holed_tree, tmp_path):
    paths = _paths(holed_tree, tmp_path)
    stats = load_pinned_stats(paths.behavior_norm_root)
    index = build_recording_index(paths, splits=("train", "val"), behavior_stats=stats)

    train_scores = [holed_tree.subject_scores[s[:5]] for s in holed_tree.splits["train"]]
    assert index.target_stats.mean == pytest.approx(float(np.mean(train_scores)))
    assert index.target_stats.std == pytest.approx(float(np.std(train_scores)))

    hole = index.recordings[BAD_HOLE]
    assert hole.n_selected == 7 and hole.n_dropped_tail == 4  # prefix cut at row 3
    assert hole.frame_mask.all() and hole.frame_mask.shape == (3,)
    assert np.isfinite(hole.behavior_norm).all()
    assert index.diagnostics["behavior_invalid_frames"] == 1

    for rec in index.by_split("train"):
        assert 0 <= rec.subject < len(index.subject_table)
        assert rec.bdi_norm == pytest.approx((rec.bdi_score - index.target_stats.mean)
                                             / index.target_stats.std)
    for rec in index.by_split("val"):
        assert rec.subject == -1  # train-only subject table
    with pytest.raises(EvaDiSchemaError):
        index.by_split("test")


def test_train_split_always_required(holed_tree, tmp_path):
    paths = _paths(holed_tree, tmp_path)
    stats = load_pinned_stats(paths.behavior_norm_root)
    with pytest.raises(EvaDiSchemaError, match="train"):
        build_recording_index(paths, splits=("val",), behavior_stats=stats)


def test_split_registry_mismatch_is_fatal(holed_tree, tmp_path):
    bad_split = tmp_path / "bad_split.json"
    payload = json.loads(holed_tree.split_file.read_text(encoding="utf-8"))
    payload["train"].append(f"999_9_Freeform_video{SPLIT_ID_SUFFIX}")
    bad_split.write_text(json.dumps(payload), encoding="utf-8")
    paths = _paths(holed_tree, tmp_path, split_file=bad_split)
    stats = load_pinned_stats(paths.behavior_norm_root)
    with pytest.raises(EvaDiSchemaError, match="registry mismatch"):
        build_recording_index(paths, splits=("train", "val"), behavior_stats=stats)


def test_cache_gaps_are_fatal(holed_tree, tmp_path):
    partial = tmp_path / "cache_partial"
    ids = sorted(set(holed_tree.splits["train"]) | set(holed_tree.splits["val"]))[:-1]
    build_cache_for_tree(partial, holed_tree.of3_root, ids,
                         rng=np.random.default_rng(3))
    paths = _paths(holed_tree, tmp_path, cache_root=partial)
    stats = load_pinned_stats(paths.behavior_norm_root)
    with pytest.raises(EvaDiSchemaError, match="cache lacks"):
        build_recording_index(paths, splits=("train", "val"), behavior_stats=stats)


@pytest.fixture()
def tree_splits(tmp_path):
    tree = build_tree(tmp_path / "splitsynth", frames_per_video=4)
    return tree.split_file
