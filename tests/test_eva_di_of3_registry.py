"""of3_registry: strict CSV schema, masking semantics, monotonicity."""

import numpy as np
import pytest

from src.eva_di.contracts import EvaDiPathError, EvaDiSchemaError
from src.eva_di.of3_registry import list_sample_ids, load_video_record

from tests._eva_di_synth import build_tree


@pytest.fixture()
def tree(tmp_path):
    return build_tree(
        tmp_path / "synth",
        frames_per_video=6,
        bad_landmark_rows={"209_1_Freeform_video": {2}},
        bad_au_rows={"211_1_Northwind_video": {1}},
        invalid_image_rows={"203_1_Freeform_video": {5}},
    )


def test_happy_path_shapes_and_dtypes(tree):
    record = load_video_record(tree.of3_root, "203_1_Freeform_video")
    assert record.n_rows == 6
    assert record.frame_index_zero.shape == (6,)
    assert record.behavior.shape == (6, 206) and record.behavior.dtype == np.float32
    assert record.image_valid.dtype == np.bool_ and record.image_valid.sum() == 5
    assert record.features_sha256 and len(record.features_sha256) == 64
    # masked image row keeps a row entry (all rows preserved), flag carries validity
    assert bool(record.image_valid[5]) is False


def test_nan_behavior_masked_never_imputed_into_valid(tree):
    record = load_video_record(tree.of3_root, "209_1_Freeform_video")
    assert bool(record.behavior_valid[2]) is False
    # contract: NaN *positions* become 0.0 so the raw array is finite; the row is
    # masked invalid so it can never enter stats/normalize (which zero the whole row).
    assert np.all(record.behavior[2, :2] == 0.0)  # the two NaN landmark coords -> 0
    assert np.isfinite(record.behavior).all()
    good = load_video_record(tree.of3_root, "203_1_Freeform_video")
    assert good.behavior_valid.all()


def test_au_nan_masked(tree):
    record = load_video_record(tree.of3_root, "211_1_Northwind_video")
    assert bool(record.behavior_valid[1]) is False


def test_missing_csv_raises_path_error(tree):
    with pytest.raises(EvaDiPathError):
        load_video_record(tree.of3_root, "999_9_Freeform_video")


def test_column_order_change_rejected(tree):
    csv_path = tree.of3_root / "samples/203_1_Freeform_video/features.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    header[1], header[2] = header[2], header[1]  # swap split/task_id
    csv_path.write_text(",".join(header) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(EvaDiSchemaError, match="schema violation"):
        load_video_record(tree.of3_root, "203_1_Freeform_video")


def test_nonmonotone_frame_index_rejected(tmp_path):
    from tests._eva_di_synth import write_features_csv

    root = tmp_path / "of3"
    rng = np.random.default_rng(1)
    write_features_csv(root / "samples/100_1_Freeform_video", "100_1_Freeform_video",
                       "train", n_frames=5, rng=rng, nonmonotone=True)
    with pytest.raises(EvaDiSchemaError, match="strictly increasing"):
        load_video_record(root, "100_1_Freeform_video")


def test_malformed_landmark_json_rejected(tmp_path):
    from tests._eva_di_synth import write_features_csv

    root = tmp_path / "of3"
    rng = np.random.default_rng(2)
    video = root / "samples/101_1_Freeform_video"
    write_features_csv(video, "101_1_Freeform_video", "train", n_frames=4, rng=rng)
    text = (video / "features.csv").read_text(encoding="utf-8")
    lines = text.splitlines()
    # break the landmark JSON of the data rows
    broken = [ln.replace("[[", "[1,]", 1) for ln in lines]
    (video / "features.csv").write_text("\n".join(broken) + "\n", encoding="utf-8")
    with pytest.raises(EvaDiSchemaError):
        load_video_record(root, "101_1_Freeform_video")


def test_list_sample_ids_sorted(tree):
    ids = list_sample_ids(tree.of3_root)
    assert ids == tuple(sorted(ids))
    assert len(ids) == 6
