from omegaconf import OmegaConf

from src.config import CONFIG_DIR, load_experiment_config
from src.datasets.photometric_normalization import (
    resolve_photometric_normalization_config,
)


CONFIG_ROOT = CONFIG_DIR / "photometric_normalization"


def _load(experiment, debug=False):
    overrides = [CONFIG_ROOT / "common.yaml", CONFIG_ROOT / experiment]
    if debug:
        overrides.append(CONFIG_ROOT / "debug_smoke.yaml")
    return load_experiment_config(overrides=overrides, require_local_paths=False)


def _without_experiment_identity_and_mode(config):
    values = OmegaConf.to_container(config, resolve=True)
    values.pop("EXPERIMENT_NAME")
    values["DATASET"]["PHOTOMETRIC_NORMALIZATION"].pop("MODE")
    return values


def test_p0_p1_p2_differ_only_in_name_and_photometric_mode():
    p0 = _load("p0_rgb.yaml")
    p1 = _load("p1_luma_center.yaml")
    p2 = _load("p2_luma_center_contrast.yaml")

    assert _without_experiment_identity_and_mode(p1) == _without_experiment_identity_and_mode(p0)
    assert _without_experiment_identity_and_mode(p2) == _without_experiment_identity_and_mode(p0)

    assert resolve_photometric_normalization_config(p0).mode == "none"
    assert resolve_photometric_normalization_config(p1).mode == "center"
    assert resolve_photometric_normalization_config(p2).mode == "center_contrast"


def test_photometric_common_uses_frozen_validation_only_reference_policy():
    cfg = _load("p0_rgb.yaml")

    assert cfg.MODE == "mtl_lite"
    assert cfg.EXPERIMENT_GROUP == "photometric_normalization"
    assert cfg.EXPERIMENT_NAME == "p0_rgb"
    assert cfg.SEED == 42
    assert cfg.RUN_TEST_AFTER_FIT is False
    assert cfg.EARLY_STOPPING.ENABLE is True
    assert cfg.EARLY_STOPPING.MONITOR == "val_RMSE_epoch"
    assert cfg.EARLY_STOPPING.MODE == "min"
    assert cfg.EXTRACT_FEATURE.FREEZE_BACKBONE is True
    assert cfg.EXTRACT_FEATURE.FINETUNE_LAST_N_BLOCKS == 2
    assert cfg.DATASET.INPUT_VARIANT == "rgb"
    assert cfg.MODEL.IDENTITY_ADVERSARIAL.ENABLE is False
    assert cfg.MODEL.SEVERITY_BALANCED_REGRESSION.ENABLE is True
    assert cfg.MODEL.SEVERITY_BALANCED_REGRESSION.POWER == 0.5
    assert cfg.MODEL.TASK_NUISANCE.ENABLE is False


def test_canonical_targets_and_bounds_are_explicit_in_resolved_config():
    cfg = _load("p2_luma_center_contrast.yaml")
    section = cfg.DATASET.PHOTOMETRIC_NORMALIZATION

    assert section.TARGET_MEDIAN == 0.50
    assert section.TARGET_SPAN == 0.50
    assert section.LOW_QUANTILE == 0.10
    assert section.HIGH_QUANTILE == 0.90
    assert section.MAX_LUMA_SHIFT == 0.20
    assert section.MIN_CONTRAST_SCALE == 0.75
    assert section.MAX_CONTRAST_SCALE == 1.33
    assert section.BLACK_THRESHOLD == 8
    assert section.STATS_MAX_FRAMES == 32
    assert section.STATS_SPATIAL_STRIDE == 4
    assert section.MIN_VALID_STAT_PIXELS == 256
    assert section.MIN_VALID_STAT_FRACTION == 0.05


def test_photometric_debug_override_preserves_run_identity():
    cfg = _load("p1_luma_center.yaml", debug=True)

    assert cfg.EXPERIMENT_GROUP == "photometric_normalization_debug"
    assert cfg.EXPERIMENT_NAME == "p1_luma_center"
    assert cfg.PROCESS_TEMPORAL.MAX_EPOCHS == 2
    assert cfg.EARLY_STOPPING.PATIENCE == 1
    assert resolve_photometric_normalization_config(cfg).mode == "center"
