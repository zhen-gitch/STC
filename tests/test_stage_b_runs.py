"""Tests for version-aware Stage B run discovery and sweep labeling.

Covers the pure labeling/discovery logic in ``src.diagnostics.stage_b_runs``:
sweep-signature -> label formatting, resolved_config parsing, and version
enumeration with signature de-duplication.  The config-merge path
(``_experiment_base_dir``) is not tested here -- it is a thin wrapper over the
real config stack and is exercised end-to-end by the shell scripts.
"""

from pathlib import Path

from omegaconf import OmegaConf

from src.diagnostics.stage_b_runs import (
    BASE_LAMBDA,
    BASE_POWER,
    discover_versions,
    label_for_sweep,
    read_sweep_from_config,
)


def _write_resolved_config(run_dir, identity=False, lambda_id=None, severity=False, power=None):
    """Write a minimal resolved_config.yaml with the given sweep switches."""
    model = {}
    if identity:
        model["IDENTITY_ADVERSARIAL"] = {"ENABLE": True, "LAMBDA_ID": lambda_id}
    if severity:
        model["SEVERITY_BALANCED_REGRESSION"] = {"ENABLE": True, "POWER": power}
    cfg = OmegaConf.create({"MODEL": model, "LOG_DIR": str(run_dir)})
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(run_dir) / "resolved_config.yaml")


def test_label_for_sweep_base_and_sweeps():
    assert label_for_sweep("e1", BASE_LAMBDA, BASE_POWER) == "e1"
    assert label_for_sweep("e1", 0.02, BASE_POWER) == "e1_lambda0.02"
    assert label_for_sweep("e1", 0.10, BASE_POWER) == "e1_lambda0.1"
    assert label_for_sweep("e2", BASE_LAMBDA, 1.0) == "e2_power1.0"
    assert label_for_sweep("e3", 0.02, 1.0) == "e3_lambda0.02_power1.0"
    # String values from YAML are coerced to float.
    assert label_for_sweep("e1", "0.20", "0.5") == "e1_lambda0.2"


def test_read_sweep_returns_base_when_switches_off():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "version_0"
        _write_resolved_config(d)  # no switches -> base signature
        assert read_sweep_from_config(d) == (BASE_LAMBDA, BASE_POWER)


def test_read_sweep_returns_none_without_resolved_config():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "version_0"
        d.mkdir()
        assert read_sweep_from_config(d) is None


def test_read_sweep_identity_and_severity():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "version_0"
        _write_resolved_config(d, identity=True, lambda_id=0.02)
        assert read_sweep_from_config(d) == (0.02, BASE_POWER)

        d2 = Path(tmp) / "version_1"
        _write_resolved_config(d2, severity=True, power=1.0)
        assert read_sweep_from_config(d2) == (BASE_LAMBDA, 1.0)

        d3 = Path(tmp) / "version_2"
        _write_resolved_config(d3, identity=True, lambda_id=0.10, severity=True, power=1.0)
        assert read_sweep_from_config(d3) == (0.10, 1.0)


def _make_version(base, idx, identity=False, lambda_id=None, severity=False, power=None):
    v = base / f"version_{idx}"
    _write_resolved_config(v, identity=identity, lambda_id=lambda_id, severity=severity, power=power)
    return v


def test_discover_versions_dedups_to_latest_per_signature(tmp_path):
    base = tmp_path / "e1_identity_adversarial"
    # v0: base (older base run), v1: lambda0.02, v2: lambda0.10, v3: base (rerun).
    _make_version(base, 0, identity=True, lambda_id=BASE_LAMBDA)
    _make_version(base, 1, identity=True, lambda_id=0.02)
    _make_version(base, 2, identity=True, lambda_id=0.10)
    _make_version(base, 3, identity=True, lambda_id=BASE_LAMBDA)

    specs = discover_versions(base, "e1")
    labels = {label: path for label, path in specs}
    # Two base versions collapse to the latest (v3), labeled e1.
    assert labels["e1"].name == "version_3"
    assert labels["e1_lambda0.02"].name == "version_1"
    assert labels["e1_lambda0.1"].name == "version_2"
    assert set(labels) == {"e1", "e1_lambda0.02", "e1_lambda0.1"}


def test_discover_versions_latest_only(tmp_path):
    base = tmp_path / "e2_severity_balanced"
    _make_version(base, 0, severity=True, power=BASE_POWER)
    _make_version(base, 1, severity=True, power=1.0)

    specs = discover_versions(base, "e2", latest_only=True)
    assert specs == [("e2", base / "version_1")]


def test_discover_versions_skips_missing_resolved_config(tmp_path):
    base = tmp_path / "e1_identity_adversarial"
    _make_version(base, 0, identity=True, lambda_id=BASE_LAMBDA)
    (base / "version_1").mkdir()  # no resolved_config.yaml -> skipped
    _make_version(base, 2, identity=True, lambda_id=0.02)

    specs = discover_versions(base, "e1")
    labels = [label for label, _ in specs]
    assert labels == ["e1", "e1_lambda0.02"]


def test_discover_versions_empty_when_no_versions(tmp_path):
    base = tmp_path / "e0_rgb_mtl_lite"
    base.mkdir()
    assert discover_versions(base, "e0") == []
    assert discover_versions(tmp_path / "does_not_exist", "e0") == []


def test_discover_versions_e3_independent_sweeps(tmp_path):
    """E3 sweeps lambda and power independently (not cartesian): each version
    has at most one non-base value, so labels stay single-suffix."""
    base = tmp_path / "e3_identity_adversarial_severity_balanced"
    _make_version(base, 0, identity=True, lambda_id=BASE_LAMBDA, severity=True, power=BASE_POWER)
    _make_version(base, 1, identity=True, lambda_id=0.02, severity=True, power=BASE_POWER)
    _make_version(base, 2, identity=True, lambda_id=BASE_LAMBDA, severity=True, power=1.0)

    specs = discover_versions(base, "e3")
    labels = [label for label, _ in specs]
    assert labels == ["e3", "e3_lambda0.02", "e3_power1.0"]
