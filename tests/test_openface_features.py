import csv
import json

from omegaconf import OmegaConf

from src.datasets.openface_features import (
    OpenFaceFeatureDataModule,
    build_openface_records,
    select_openface_feature_columns,
    select_openface_feature_columns_and_options,
)


def _write_openface_csv(path):
    fieldnames = [
        "frame",
        "timestamp",
        "confidence",
        "success",
        "pose_Rx",
        "pose_Ry",
        "pose_Rz",
        "gaze_angle_x",
        "gaze_angle_y",
        "x_0",
        "x_1",
        "y_0",
        "y_1",
        "AU01_r",
        "AU01_c",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(4):
            writer.writerow(
                {
                    "frame": idx + 1,
                    "timestamp": f"{idx * 0.04:.3f}",
                    "confidence": 0.95 - 0.01 * idx,
                    "success": 1,
                    "pose_Rx": 0.1 * idx,
                    "pose_Ry": 0.2 * idx,
                    "pose_Rz": -0.1 * idx,
                    "gaze_angle_x": 0.01 * idx,
                    "gaze_angle_y": -0.02 * idx,
                    "x_0": 10 + idx,
                    "x_1": 12 + idx,
                    "y_0": 20 + idx,
                    "y_1": 23 + idx,
                    "AU01_r": 0.2 * idx,
                    "AU01_c": idx % 2,
                }
            )


def _write_fixture(tmp_path):
    openface = tmp_path / "openface"
    labels = tmp_path / "labels"
    openface.mkdir()
    labels.mkdir()

    split = {"train": [], "val": [], "test": []}
    for subject_id, split_name, label in [
        ("001_1", "train", 8),
        ("002_1", "train", 16),
        ("003_1", "val", 24),
        ("004_1", "test", 32),
    ]:
        video_id = f"{subject_id}_Freeform_video"
        _write_openface_csv(openface / f"{video_id}.csv")
        (labels / f"{subject_id}_Depression.csv").write_text(str(label), encoding="utf-8")
        split[split_name].append(f"{video_id}_aligned")

    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return openface, labels, split_path


def _config(openface, labels, split_path):
    return OmegaConf.create(
        {
            "LABEL_DIR": str(labels),
            "DATASET_SPLIT_FILE": str(split_path),
            "EXTRACT_FEATURE": {
                "BATCH_SIZE": 2,
                "NUM_WORKERS": 0,
            },
            "PROCESS_TEMPORAL": {
                "CLASS_STEP": 2,
                "SAMPLE_STEP": 1,
                "MAX_SEQ_LEN": 4,
            },
            "DATASET": {
                "OPENFACE_ROOT": str(openface),
                "ALLOW_MISSING_OPENFACE": False,
            },
            "BEHAVIOR_FEATURES": {
                "FEATURE_SET": "custom",
                "INCLUDE_QUALITY": True,
                "INCLUDE_POSE": True,
                "INCLUDE_GAZE": True,
                "INCLUDE_AU": True,
                "INCLUDE_LANDMARKS": True,
                "INCLUDE_DELTA": True,
                "INCLUDE_ACCELERATION": False,
            },
        }
    )


def test_select_openface_feature_columns_respects_groups():
    fieldnames = [
        "frame",
        "confidence",
        "success",
        "pose_Rx",
        "gaze_angle_x",
        "x_0",
        "y_0",
        "AU01_r",
        "AU01_c",
    ]
    cfg = OmegaConf.create(
        {
            "BEHAVIOR_FEATURES": {
                "FEATURE_SET": "custom",
                "INCLUDE_QUALITY": True,
                "INCLUDE_POSE": False,
                "INCLUDE_GAZE": True,
                "INCLUDE_AU": True,
                "INCLUDE_LANDMARKS": False,
            }
        }
    )

    columns = select_openface_feature_columns(fieldnames, cfg)

    assert columns == ["confidence", "success", "gaze_angle_x", "AU01_r", "AU01_c"]


def test_named_feature_sets_select_expected_columns_and_options():
    fieldnames = [
        "frame",
        "confidence",
        "success",
        "pose_Rx",
        "pose_Ry",
        "gaze_angle_x",
        "x_0",
        "y_0",
        "AU01_r",
        "AU01_c",
    ]

    landmark_cfg = OmegaConf.create({"BEHAVIOR_FEATURES": {"FEATURE_SET": "landmark_delta_only"}})
    columns, options = select_openface_feature_columns_and_options(fieldnames, landmark_cfg)
    assert columns == ["x_0", "y_0"]
    assert options["include_raw"] is False
    assert options["include_delta"] is True

    mixed_cfg = OmegaConf.create({"BEHAVIOR_FEATURES": {"FEATURE_SET": "au_landmark_delta"}})
    columns, options = select_openface_feature_columns_and_options(fieldnames, mixed_cfg)
    assert columns == ["AU01_r", "AU01_c", "x_0", "y_0"]
    assert options["raw_indices"] == [0, 1]
    assert options["delta_indices"] == [2, 3]


def test_openface_feature_datamodule_builds_padded_batches(tmp_path):
    openface, labels, split_path = _write_fixture(tmp_path)
    cfg = _config(openface, labels, split_path)

    records = build_openface_records(openface, split_path, "train")
    data_module = OpenFaceFeatureDataModule(cfg)
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))
    features, mask, labels_batch = batch

    assert len(records) == 2
    assert data_module.feature_dim == len(data_module.feature_columns) * 2
    assert features.shape == (2, 4, data_module.feature_dim)
    assert mask.shape == (2, 4)
    assert labels_batch["bdi_score"].shape == (2,)


def test_feature_set_changes_derived_feature_dim(tmp_path):
    openface, labels, split_path = _write_fixture(tmp_path)
    cfg = _config(openface, labels, split_path)
    cfg.BEHAVIOR_FEATURES.FEATURE_SET = "all_without_raw_landmarks"

    data_module = OpenFaceFeatureDataModule(cfg)
    data_module.setup()

    non_landmark_raw_dim = 2 + 3 + 2 + 2
    all_delta_dim = 2 + 3 + 2 + 4 + 2
    assert data_module.feature_dim == non_landmark_raw_dim + all_delta_dim
