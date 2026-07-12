import pytest
from omegaconf import OmegaConf

from scripts import train_mtl_lite
from src.config import CONFIG_DIR, load_experiment_config


CONFIG_ROOT = CONFIG_DIR / "stage_c"


def _load(*overrides):
    return load_experiment_config(
        overrides=[CONFIG_ROOT / override for override in overrides],
        require_local_paths=False,
    )


def test_stage_c_reference_resolves_to_runnable_e2_policy():
    cfg = _load("common.yaml", "c_ref_e2.yaml", "seeds/seed_42.yaml")

    assert cfg.MODE == "mtl_lite"
    assert cfg.EXPERIMENT_GROUP == "stage_c"
    assert cfg.EXPERIMENT_NAME == "c_ref_e2"
    assert cfg.SEED == 42
    assert cfg.PROCESS_TEMPORAL.MAX_EPOCHS == 40
    assert cfg.EARLY_STOPPING.ENABLE is True
    assert cfg.EARLY_STOPPING.MONITOR == "val_RMSE_epoch"
    assert cfg.EARLY_STOPPING.MODE == "min"
    assert cfg.MODEL.SEVERITY_BALANCED_REGRESSION.ENABLE is True
    assert cfg.MODEL.SEVERITY_BALANCED_REGRESSION.POWER == 0.5
    assert cfg.MODEL.TASK_NUISANCE.ENABLE is False


@pytest.mark.parametrize(
    "experiment",
    [
        "c_bn_bottleneck.yaml",
        "c_rec_split_recon.yaml",
        "c_full_split_full.yaml",
        "calibration_train_only.yaml",
    ],
)
def test_unimplemented_stage_c_candidates_are_spec_only(experiment):
    cfg = _load("common.yaml", experiment, "seeds/seed_42.yaml")

    assert cfg.MODE == "stage_c_spec_only"
    assert cfg.MODEL.TASK_NUISANCE.ENABLE is True


def test_spec_only_candidate_is_rejected_by_training_entry():
    cfg = _load("common.yaml", "c_bn_bottleneck.yaml", "seeds/seed_42.yaml")

    with pytest.raises(ValueError, match="stage_c_spec_only"):
        train_mtl_lite.run_from_config(cfg)


def test_split_candidate_weights_remain_unfrozen_before_calibration():
    rec = _load("common.yaml", "c_rec_split_recon.yaml")
    full = _load("common.yaml", "c_full_split_full.yaml")

    assert rec.LOSSES.RECONSTRUCTION_WEIGHT is None
    assert rec.LOSSES.CROSS_CORRELATION_WEIGHT == 0.0
    assert full.LOSSES.RECONSTRUCTION_WEIGHT is None
    assert full.LOSSES.CROSS_CORRELATION_WEIGHT is None


@pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
def test_seed_override_files_change_only_the_resolved_seed(seed):
    reference = _load("common.yaml", "c_ref_e2.yaml", "seeds/seed_42.yaml")
    cfg = _load("common.yaml", "c_ref_e2.yaml", f"seeds/seed_{seed}.yaml")

    assert cfg.SEED == seed
    reference_without_seed = OmegaConf.to_container(reference, resolve=True)
    candidate_without_seed = OmegaConf.to_container(cfg, resolve=True)
    reference_without_seed.pop("SEED")
    candidate_without_seed.pop("SEED")
    assert candidate_without_seed == reference_without_seed


def test_debug_override_is_last_and_does_not_remove_experiment_identity():
    cfg = _load(
        "common.yaml",
        "c_ref_e2.yaml",
        "seeds/seed_43.yaml",
        "debug_smoke.yaml",
    )

    assert cfg.MODE == "mtl_lite"
    assert cfg.EXPERIMENT_GROUP == "stage_c_debug"
    assert cfg.EXPERIMENT_NAME == "c_ref_e2"
    assert cfg.SEED == 43
    assert cfg.EXTRACT_FEATURE.BATCH_SIZE == 2
    assert cfg.PROCESS_TEMPORAL.MAX_EPOCHS == 2
    assert cfg.EARLY_STOPPING.PATIENCE == 1
