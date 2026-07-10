"""Tests for the ``scripts/audit_openface_quality.py`` thin wrapper.

The core summarization (``summarize_openface_root`` /
``write_openface_quality_summary``) is already covered by
``tests/test_shortcut_audit.py`` and ``tests/test_artifact_weaklabels.py``.
These tests cover the only *new* logic in the wrapper: the OpenFace-root
resolution (``--openface-root`` vs ``--run-dir`` resolved_config.yaml) and the
graceful-degradation path (missing root -> header-only empty summary so the
downstream weak-label join still finds the file).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402

# Import the wrapper as a module (it lives under scripts/, not a package).
import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "audit_openface_quality", PROJECT_ROOT / "scripts" / "audit_openface_quality.py"
)
audit_openface_quality = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_openface_quality)
_resolve_openface_root = audit_openface_quality._resolve_openface_root


def _ns(**kwargs):
    defaults = dict(
        openface_root=None, run_dir=None, output_dir=".", low_confidence_threshold=0.5
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_openface_root_flag_wins_over_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # resolved_config.yaml carries a different root, but --openface-root wins.
    (run_dir / "resolved_config.yaml").write_text("DATASET:\n  OPENFACE_ROOT: /from/config\n")
    explicit = tmp_path / "explicit_openface"
    explicit.mkdir()

    root = _resolve_openface_root(_ns(openface_root=str(explicit), run_dir=str(run_dir)))
    assert root == explicit


def test_resolve_from_run_dir_resolved_config_absolute(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    of_root = tmp_path / "of_csvs"
    of_root.mkdir()
    (run_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml({"DATASET": {"OPENFACE_ROOT": str(of_root)}})
    )

    root = _resolve_openface_root(_ns(run_dir=str(run_dir)))
    assert root == of_root


def test_resolve_relative_openface_root_is_anchored_to_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml({"DATASET": {"OPENFACE_ROOT": "openface_csvs"}})
    )

    root = _resolve_openface_root(_ns(run_dir=str(run_dir)))
    # Relative path resolves against the run dir (resolved), not CWD.
    assert root == (run_dir.resolve() / "openface_csvs")


def test_returns_none_when_openface_root_unset(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml({"DATASET": {"OPENFACE_ROOT": None}})
    )
    assert _resolve_openface_root(_ns(run_dir=str(run_dir))) is None


def test_returns_none_when_resolved_config_missing(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # No resolved_config.yaml at all.
    assert _resolve_openface_root(_ns(run_dir=str(run_dir))) is None


def test_returns_none_when_no_args():
    assert _resolve_openface_root(_ns()) is None


def test_returns_none_when_dataset_key_absent(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # No DATASET block -> AttributeError inside, swallowed -> None.
    (run_dir / "resolved_config.yaml").write_text("EXPERIMENT_NAME: e1\n")
    assert _resolve_openface_root(_ns(run_dir=str(run_dir))) is None


def test_graceful_degradation_writes_empty_summary(tmp_path, monkeypatch):
    """Missing root -> header-only empty summary so weaklabel join finds the file.

    Drives the wrapper's main() (which reads sys.argv) with no resolvable root.
    """
    out_dir = tmp_path / "ofq"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_openface_quality.py",
            "--run-dir",
            str(tmp_path / "nonexistent_run"),
            "--output-dir",
            str(out_dir),
        ],
    )
    rc = audit_openface_quality.main()
    assert rc is None  # main returns None on the degrade path
    summary = out_dir / "tables" / "openface_quality_summary.csv"
    assert summary.exists()
    # Header-only (a single header line, no data rows) -- the file exists so the
    # weaklabel join can open it and skip openface fields cleanly.
    lines = summary.read_text().strip().splitlines()
    assert len(lines) == 1
