from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from scripts import train_mtl_lite
from src.config import DEFAULT_BASE_CONFIG
from src.trainers import mtl_lite_runner
from src.trainers.mtl_lite_runner import (
    build_mtl_lite_callbacks,
    build_mtl_lite_trainer,
    resolve_early_stopping_config,
    resolve_run_test_after_fit,
)


def _config(early_stopping=None, seed=None):
    values = {"MODE": "mtl_lite"}
    if early_stopping is not None:
        values["EARLY_STOPPING"] = early_stopping
    if seed is not None:
        values["SEED"] = seed
    return OmegaConf.create(values)


def _callback(callbacks, callback_type):
    return next(item for item in callbacks if isinstance(item, callback_type))


def test_early_stopping_defaults_to_disabled_and_preserves_checkpoint_policy():
    policy = resolve_early_stopping_config(_config())
    callbacks = build_mtl_lite_callbacks(_config())

    assert policy.enable is False
    assert policy.monitor == "val_RMSE_epoch"
    assert policy.mode == "min"
    assert not any(isinstance(item, EarlyStopping) for item in callbacks)

    checkpoint = _callback(callbacks, ModelCheckpoint)
    assert checkpoint.monitor == policy.monitor
    assert checkpoint.mode == policy.mode
    assert checkpoint.save_top_k == 1
    assert checkpoint.save_last is True


def test_base_config_freezes_p0_defaults_and_seed_can_be_overridden():
    base = OmegaConf.load(DEFAULT_BASE_CONFIG)

    assert train_mtl_lite.resolve_seed(base) == 42
    assert base.EARLY_STOPPING == {
        "ENABLE": False,
        "MONITOR": "val_RMSE_epoch",
        "MODE": "min",
        "PATIENCE": 8,
        "MIN_DELTA": 0.0,
        "STRICT": True,
        "CHECK_FINITE": True,
    }
    assert resolve_run_test_after_fit(base) is True

    overridden = OmegaConf.merge(base, {"SEED": 1234})
    assert train_mtl_lite.resolve_seed(overridden) == 1234


def test_enabled_early_stopping_and_checkpoint_share_configured_policy():
    cfg = _config(
        {
            "ENABLE": True,
            "MONITOR": "val_CCC_epoch",
            "MODE": "max",
            "PATIENCE": 5,
            "MIN_DELTA": 0.01,
            "STRICT": False,
            "CHECK_FINITE": False,
        }
    )

    callbacks = build_mtl_lite_callbacks(cfg)
    checkpoint = _callback(callbacks, ModelCheckpoint)
    early_stopping = _callback(callbacks, EarlyStopping)

    assert checkpoint.monitor == early_stopping.monitor == "val_CCC_epoch"
    assert checkpoint.mode == early_stopping.mode == "max"
    assert "{val_CCC_epoch:.4f}" in checkpoint.filename
    assert early_stopping.patience == 5
    assert early_stopping.min_delta == 0.01
    assert early_stopping.strict is False
    assert early_stopping.check_finite is False


def test_calibration_callbacks_do_not_monitor_validation_or_early_stop():
    callbacks = build_mtl_lite_callbacks(_config(), calibration_only=True)
    checkpoint = _callback(callbacks, ModelCheckpoint)

    assert checkpoint.monitor is None
    assert checkpoint.save_top_k == 0
    assert checkpoint.save_last is True
    assert not any(isinstance(item, EarlyStopping) for item in callbacks)


def test_calibration_trainer_uses_train_only_step_limit(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "ACCELERATOR": "cpu",
            "DEVICES": 1,
            "STRATEGY": "auto",
            "PRECISION": "32-true",
            "LOG_DIR": str(tmp_path),
            "EXPERIMENT_GROUP": "stage_c_debug",
            "EXPERIMENT_NAME": "calibration",
            "PROCESS_TEMPORAL": {"MAX_EPOCHS": 40},
        }
    )
    monkeypatch.setattr(
        mtl_lite_runner, "CSVLogger", lambda **kwargs: ("csv", kwargs)
    )
    monkeypatch.setattr(
        mtl_lite_runner, "TensorBoardLogger", lambda **kwargs: ("tb", kwargs)
    )
    monkeypatch.setattr(
        mtl_lite_runner.pl, "Trainer", lambda **kwargs: kwargs
    )

    trainer_kwargs = build_mtl_lite_trainer(
        cfg, calibration_only=True, calibration_steps=17
    )

    assert trainer_kwargs["max_steps"] == 17
    assert trainer_kwargs["limit_val_batches"] == 0
    assert trainer_kwargs["num_sanity_val_steps"] == 0
    assert trainer_kwargs["max_epochs"] == 40


def test_calibration_run_skips_validation_selected_test(monkeypatch):
    calls = []

    class FakeModel:
        identity_adversarial = False
        severity_balanced_regression = False
        task_nuisance_config = SimpleNamespace(
            calibration_only=True,
            calibration_steps=11,
        )

    class FakeTrainer:
        def fit(self, model, data_module):
            calls.append(("fit", model, data_module))

        def test(self, *args, **kwargs):
            calls.append(("test", args, kwargs))

    data_module = object()
    trainer = FakeTrainer()
    monkeypatch.setattr(mtl_lite_runner, "MTLLiteDepressionModel", lambda cfg: FakeModel())
    monkeypatch.setattr(mtl_lite_runner, "AVECDataModule", lambda cfg: data_module)
    monkeypatch.setattr(
        mtl_lite_runner,
        "build_mtl_lite_trainer",
        lambda cfg, calibration_only, calibration_steps: trainer,
    )
    monkeypatch.setattr(mtl_lite_runner, "save_resolved_config", lambda *args: None)

    mtl_lite_runner.run_mtl_lite(OmegaConf.create({}))

    assert [call[0] for call in calls] == ["fit"]


@pytest.mark.parametrize(
    ("configured_value", "expected_calls"),
    [
        (None, ["fit", "test"]),
        (False, ["fit"]),
    ],
)
def test_run_test_after_fit_policy(monkeypatch, configured_value, expected_calls):
    calls = []

    class FakeModel:
        identity_adversarial = False
        severity_balanced_regression = False
        task_nuisance_config = SimpleNamespace(
            calibration_only=False,
            calibration_steps=100,
        )

    class FakeTrainer:
        def fit(self, model, data_module):
            calls.append(("fit", model, data_module))

        def test(self, *args, **kwargs):
            calls.append(("test", args, kwargs))

    data_module = object()
    trainer = FakeTrainer()
    monkeypatch.setattr(mtl_lite_runner, "MTLLiteDepressionModel", lambda cfg: FakeModel())
    monkeypatch.setattr(mtl_lite_runner, "AVECDataModule", lambda cfg: data_module)
    monkeypatch.setattr(
        mtl_lite_runner,
        "build_mtl_lite_trainer",
        lambda cfg, calibration_only, calibration_steps: trainer,
    )
    monkeypatch.setattr(mtl_lite_runner, "save_resolved_config", lambda *args: None)

    cfg = {}
    if configured_value is not None:
        cfg["RUN_TEST_AFTER_FIT"] = configured_value
    mtl_lite_runner.run_mtl_lite(OmegaConf.create(cfg))

    assert [call[0] for call in calls] == expected_calls


@pytest.mark.parametrize("value", [1, "false", None])
def test_invalid_run_test_after_fit_is_rejected(value):
    with pytest.raises(ValueError, match="RUN_TEST_AFTER_FIT must be a boolean"):
        resolve_run_test_after_fit(OmegaConf.create({"RUN_TEST_AFTER_FIT": value}))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ENABLE", "yes", "ENABLE must be a boolean"),
        ("MONITOR", " ", "MONITOR must be a non-empty string"),
        ("MODE", "auto", "MODE must be either"),
        ("MODE", ["min"], "MODE must be either"),
        ("PATIENCE", -1, "PATIENCE must be a non-negative integer"),
        ("PATIENCE", True, "PATIENCE must be a non-negative integer"),
        ("MIN_DELTA", float("inf"), "MIN_DELTA must be a finite"),
        ("MIN_DELTA", -0.1, "MIN_DELTA must be a finite"),
        ("STRICT", 1, "STRICT must be a boolean"),
        ("CHECK_FINITE", None, "CHECK_FINITE must be a boolean"),
    ],
)
def test_invalid_early_stopping_config_is_rejected(field, value, message):
    cfg = _config({field: value})

    with pytest.raises(ValueError, match=message):
        resolve_early_stopping_config(cfg)


def test_unknown_early_stopping_field_is_rejected():
    with pytest.raises(ValueError, match="Unknown EARLY_STOPPING.*MONITER"):
        resolve_early_stopping_config(_config({"MONITER": "val_RMSE_epoch"}))


def test_early_stopping_section_must_be_a_mapping():
    with pytest.raises(ValueError, match="EARLY_STOPPING must be a mapping"):
        resolve_early_stopping_config(OmegaConf.create({"EARLY_STOPPING": False}))


def test_seed_defaults_to_42_and_accepts_explicit_override():
    assert train_mtl_lite.resolve_seed(_config()) == 42
    assert train_mtl_lite.resolve_seed(_config(seed=1234)) == 1234


@pytest.mark.parametrize("seed", [True, 1.5, "42", -1, 2**32])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="SEED must"):
        train_mtl_lite.resolve_seed(_config(seed=seed))


def test_run_from_config_seeds_after_config_resolution(monkeypatch):
    calls = []

    monkeypatch.setattr(
        train_mtl_lite.pl,
        "seed_everything",
        lambda seed, workers: calls.append((seed, workers)),
    )
    monkeypatch.setattr(
        "src.trainers.mtl_lite_runner.run_mtl_lite",
        lambda cfgs: calls.append(("run", cfgs.SEED)),
    )

    train_mtl_lite.run_from_config(_config(seed=77))

    assert calls == [(77, True), ("run", 77)]
