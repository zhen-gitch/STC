import pytest
from omegaconf import OmegaConf

from src.config import CONFIG_DIR, load_experiment_config
from src.models.task_nuisance import resolve_task_nuisance_config


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
    ("experiment", "reconstruction_weight", "cross_correlation_weight"),
    [
        ("c_rec_split_recon.yaml", 0.001, 0.0),
        ("c_full_split_full.yaml", 0.001, 0.01),
    ],
)
def test_calibrated_stage_c_candidates_are_runnable(
    experiment,
    reconstruction_weight,
    cross_correlation_weight,
):
    cfg = _load("common.yaml", experiment, "seeds/seed_42.yaml")

    assert cfg.MODE == "mtl_lite"
    policy = resolve_task_nuisance_config(cfg, cfg.PROCESS_TEMPORAL.HIDDEN_DIM)
    assert policy.enabled is True
    assert policy.variant == "split"
    assert policy.reconstruction_enabled is True
    assert policy.reconstruction_weight == reconstruction_weight
    assert policy.cross_correlation_weight == cross_correlation_weight
    assert policy.cross_correlation_enabled is (cross_correlation_weight > 0.0)


@pytest.mark.parametrize(
    "experiment",
    ["c_bn_bottleneck.yaml", "calibration_train_only.yaml"],
)
def test_implemented_stage_c_support_configs_are_runnable(experiment):
    cfg = _load("common.yaml", experiment, "seeds/seed_42.yaml")

    assert cfg.MODE == "mtl_lite"
    policy = resolve_task_nuisance_config(cfg, cfg.PROCESS_TEMPORAL.HIDDEN_DIM)
    assert policy.enabled is True


def test_calibration_config_excludes_auxiliary_losses_from_objective():
    cfg = _load("common.yaml", "calibration_train_only.yaml")
    policy = resolve_task_nuisance_config(cfg, cfg.PROCESS_TEMPORAL.HIDDEN_DIM)

    assert policy.variant == "split"
    assert policy.calibration_only is True
    assert policy.reconstruction_enabled is True
    assert policy.cross_correlation_enabled is True
    assert policy.reconstruction_weight == 0.0
    assert policy.cross_correlation_weight == 0.0
    assert policy.calibration_steps == 100
    assert cfg.EARLY_STOPPING.ENABLE is False


def test_split_candidate_weights_match_seed_42_calibration():
    rec = _load("common.yaml", "c_rec_split_recon.yaml")
    full = _load("common.yaml", "c_full_split_full.yaml")

    assert rec.LOSSES.RECONSTRUCTION_WEIGHT == 0.001
    assert rec.LOSSES.CROSS_CORRELATION_WEIGHT == 0.0
    assert full.LOSSES.RECONSTRUCTION_WEIGHT == 0.001
    assert full.LOSSES.CROSS_CORRELATION_WEIGHT == 0.01


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
