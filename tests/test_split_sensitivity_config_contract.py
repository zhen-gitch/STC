from omegaconf import OmegaConf

from src.config import CONFIG_DIR, load_experiment_config


CONFIG_ROOT = CONFIG_DIR / "split_sensitivity"
PHOTO_ROOT = CONFIG_DIR / "photometric_normalization"


def _load(experiment):
    return load_experiment_config(
        overrides=[
            PHOTO_ROOT / "common.yaml",
            PHOTO_ROOT / "p0_rgb.yaml",
            CONFIG_ROOT / experiment,
        ],
        require_local_paths=False,
    )


def _without_role_experiment_fields(config):
    values = OmegaConf.to_container(config, resolve=True)
    values.pop("EXPERIMENT_GROUP")
    values.pop("EXPERIMENT_NAME")
    values.pop("RUN_TEST_AFTER_FIT")
    values["DATASET"].pop("SWAP_VAL_TEST")
    return values


def test_swap_condition_uses_test_for_validation_and_val_for_test():
    reference = _load("reference.yaml")
    swapped = _load("val_test_swapped.yaml")

    assert reference.DATASET.SWAP_VAL_TEST is False
    assert swapped.DATASET.SWAP_VAL_TEST is True
    assert reference.RUN_TEST_AFTER_FIT is True
    assert swapped.RUN_TEST_AFTER_FIT is True
    assert reference.EARLY_STOPPING.MONITOR == "val_RMSE_epoch"
    assert swapped.EARLY_STOPPING.MONITOR == "val_RMSE_epoch"
    assert _without_role_experiment_fields(reference) == _without_role_experiment_fields(
        swapped
    )


def test_swap_config_does_not_change_train_only_e2_policy():
    config = _load("val_test_swapped.yaml")

    assert config.EXPERIMENT_GROUP == "split_sensitivity"
    assert config.EXPERIMENT_NAME == "e2_p0_val_test_swapped"
    assert config.SEED == 42
    assert config.EARLY_STOPPING.ENABLE is True
    assert config.EXTRACT_FEATURE.FREEZE_BACKBONE is True
    assert config.EXTRACT_FEATURE.FINETUNE_LAST_N_BLOCKS == 2
    assert config.MODEL.SEVERITY_BALANCED_REGRESSION.ENABLE is True
    assert config.DATASET.INPUT_VARIANT == "rgb"
