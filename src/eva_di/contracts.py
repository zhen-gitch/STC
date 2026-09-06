"""Frozen constants and exception hierarchy for ``eva_di`` (single source of truth).

Every schema id, hash pin, dimension and default path used anywhere in the
package is defined here and imported elsewhere; nothing may copy these values.
Pinned from docs/EVA_DI_VALIDATION_PLAN.md section 1 (read-only verified
2026-09-04); any SHA drift on the read-only asset roots is a hard stop --
constants are never silently updated.
"""

from __future__ import annotations

PLAN_ID = "PLAN-EVA-DI-VALIDATE-v1"

SCHEMA_FRAME_CACHE = "eva_di_frame_feature_cache_v1"
SCHEMA_CONFIG = "eva_di_config_v1"
SCHEMA_RUNNER_LOG = "eva_di_runner_log_v1"
SCHEMA_PROVENANCE = "eva_di_run_provenance_v1"
SCHEMA_RECORDING = "eva_di_recording_v1"

SELECTION_POLICY = "uniform512_v1"
FRAME_BUDGET = 512

# --- EVA02 backbone (docs/EVA_DI_VALIDATION_PLAN.md section 1.1) ------------
EVA_BACKBONE = "eva02_small_patch14_224.mim_in22k"
EVA_WEIGHT_SHA256 = "d3d632352efbd0a0a8269dce114ab8a214833f85ce6f738860221f11ab0f0c2f"
EVA_WEIGHT_SIZE_BYTES = 86_499_188
EVA_WEIGHT_REVISION = "79c7d4274f6dbf202549d8f976ae24eeaf97e5ad"
EVA_INPUT_SIZE = 224
EVA_SOURCE_SIZE = 112
EVA_PATCH_GRID = (16, 16)
EVA_TOKEN_COUNT = 257  # 1 CLS + 16*16 patch tokens
EVA_FEATURE_DIM = 384

# --- OpenFace3 formal outputs (docs/EVA_DI_VALIDATION_PLAN.md section 1.2) ---
OF3_ROOT_DEFAULT = (
    "/home/zhen/code/trans/outputs/openface3/"
    "20260727-avec2014-formal-features-masked-v2"
)
SCHEMA_ALIGNMENT = "retinaface_5pt_arcface112_masked_v1"
SCHEMA_BEHAVIOR = "openface3_wflw98xy_gaze2_anonymous_mtl8_v1"
SCHEMA_ALIGNED_MASK = "wflw98_dynamic_face_oval_black_v1"
BEHAVIOR_NORM_SCHEMA = "ribformer_behavior_normalization_v1"
BEHAVIOR_NORM_POLICY = "technical"

# Verbatim 42-column lock for per-video features.csv (order-sensitive).
FEATURES_CSV_COLUMNS: tuple[str, ...] = (
    "sample_id", "split", "task_id", "frame_index_zero", "frame_index_one",
    "timestamp", "source_video", "source_video_sha256", "frame_quality_sha256",
    "decoded", "detection_count", "confidence", "quality_status",
    "quality_geometry_status", "bbox_xyxy", "retinaface_5pt_xy_original",
    "technical_valid", "alignment_matrix_2x3", "landmarks_98_xy_original",
    "landmarks_98_xy_aligned", "aligned_image_path", "aligned_image_sha256",
    "aligned_mask_schema", "gaze_angle_x", "gaze_angle_y",
    "au_score_0", "au_score_1", "au_score_2", "au_score_3",
    "au_score_4", "au_score_5", "au_score_6", "au_score_7",
    "emotion_index", "alignment_status", "landmark_status", "behavior_status",
    "image_valid", "behavior_valid", "pair_valid", "inference_seconds", "issues",
)
FEATURES_CSV_FILE = "features.csv"

BEHAVIOR_DIM = 206  # landmarks 98*2=196 + gaze 2 + au 8
BEHAVIOR_LANDMARKS = 196  # leading landmark axes; rest is gaze(2) + au(8) raw
# The ``*_norm`` landmark axes are shipped normalized to [-1,1] over the
# 111-px aligned canvas -- NOT raw pixels.  Pinned by the generator itself:
# ``2.0 * landmarks / 111.0 - 1.0`` (OF3 feature definition; authoritative
# implementation: trans ``ribformer/data/clips.py:851``; stats in
# ``behavior_normalization_train_v1/technical.json`` carry this space).
BEHAVIOR_LANDMARK_CANVAS = 111.0


def expected_behavior_axis_names() -> tuple[str, ...]:
    """Canonical 206-d axis order, verified against the pinned train stats
    ``feature_names`` (landmark_00_x_norm..landmark_97_y_norm, gaze, au0..7)."""
    axes = [f"landmark_{i:02d}_{axis}_norm" for i in range(98) for axis in ("x", "y")]
    axes += ["gaze_angle_x", "gaze_angle_y"]
    axes += [f"au_score_{i}" for i in range(8)]
    if len(axes) != BEHAVIOR_DIM:  # never `assert`: stripped by `python -O`
        raise EvaDiSchemaError(f"axis drift: built {len(axes)} != {BEHAVIOR_DIM}")
    return tuple(axes)


# --- Model geometry ----------------------------------------------------------
PROJ_DIM = 192
SUBJECT_PREFIX_LEN = 5  # LABEL key := video_id[:5] ("203_1"), mirrors
                        # src/datasets/dataset.py:276-280 (files are session-level:
                        # depression_labels/203_1_Depression.csv, 150 files)
IDENTITY_PREFIX_LEN = 3  # identity class := video_id[:3] ("<person>") -- user
                         # decision 2026-09-05; persons overlap across splits by
                         # design of AVEC2014 (train∩val 26), which is exactly
                         # what the external identity attacker must measure
SEVERITY_BOUNDS = (13, 19, 28)
SEVERITY_GROUP_NAMES = ("mild", "moderate", "severe", "very_severe", "unknown")

# --- Splits ------------------------------------------------------------------
SPLIT_KEYS = ("train", "val", "test")
DECISION_SPLITS = ("train", "val")  # any entry point accepting splits uses this
SPLIT_ID_SUFFIX = "_aligned"  # dataset_split.json ids carry this suffix
# Pinned OF3 dialect (measured on the real csv files 2026-09-05): the per-video
# features.csv "split" column labels the json "val" split as "dev".
CSV_SPLIT_ALIASES = {"train": ("train",), "val": ("val", "dev"), "test": ("test",)}

# --- Defaults / paths ---------------------------------------------------------
AVEC_ROOT_DEFAULT = "/home/zhen/dataset/depression/avec/2014"
CACHE_ROOT_SUBDIR = "eva02_features_v1"
FEATURES_SUBDIR = "samples"
BEHAVIOR_NORM_SUBDIR = "behavior_normalization_train_v1"
BEHAVIOR_NORM_FILE = "technical.json"
LABEL_SUFFIX = "_Depression.csv"

# Cache on-disk file names per sample.
CACHE_CLS_FILE = "cls.npy"
CACHE_GAP_FILE = "gap.npy"
CACHE_INDEX_FILE = "frame_index_zero.npy"
CACHE_VALID_FILE = "frame_valid.npy"
CACHE_META_FILE = "meta.json"
CACHE_MANIFEST_FILE = "cache_manifest.json"
CACHE_EQUIV_DIR = "_equivalence"
CACHE_FEATURE_DTYPE = "float32"
CACHE_VALID_DTYPE = "bool"

# Design section 7: manifest provenance fields beyond the per-sample entries.
CACHE_MANIFEST_OF3_ROOT_KEY = "of3_root"
CACHE_MANIFEST_CSV_SHA_KEY = "features_csv_sha256_by_sample"

# Cache manifest structure (shared by writer/reader; audit R2: the reader kept
# a private copy the writer never enforced, so malformed manifests could be
# produced by any writer).
REQUIRED_MANIFEST_FIELDS = (
    "schema_version", "selection_policy", "frame_budget", "backbone",
    "weight_sha256", "weight_revision", "timm_version", "input_size",
    "feature_dim", "feature_dtype", "valid_dtype", "sample_count",
    "n_rows_total", "n_selected_total", "n_valid_total",
    "n_exact_zero_masked", "samples",
    CACHE_MANIFEST_OF3_ROOT_KEY, CACHE_MANIFEST_CSV_SHA_KEY,
)
REQUIRED_ENTRY_FIELDS = (
    "cls_sha256", "gap_sha256", "index_sha256", "valid_sha256",
    "n_frames", "n_valid",
)
REQUIRED_META_FIELDS = (
    "sample_id", "n_rows", "n_image_valid", "n_selected",
    "n_exact_zero_masked", "selection_policy", "features_sha256",
)

MAX_INVALID_IMAGE_RATIO = 0.01  # per-video hard stop during extraction


class EvaDiError(RuntimeError):
    """Base error for every eva_di contract violation (fail-fast, no fallback)."""


class EvaDiSchemaError(EvaDiError):
    """On-disk or in-memory structure deviates from a pinned schema."""


class EvaDiPathError(EvaDiError):
    """A required path/file is missing or unreadable."""


class EvaDiFingerprintError(EvaDiError):
    """A pinned hash / manifest fingerprint / recomputed-stat mismatch."""


class EvaDiConfigError(EvaDiError):
    """Configuration tree violates the strict schema."""


def parse_bool(value: str, *, field: str = "", row: int = -1) -> bool:
    """Strict 'True'/'False' parse; anything else is a schema error."""
    text = str(value).strip()
    if text == "True":
        return True
    if text == "False":
        return False
    raise EvaDiSchemaError(f"non-boolean {field!r} value {value!r} at row {row}")
