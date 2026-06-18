"""Tests for scripts/diagnose_mtl_lite.py split handling and output layout.

These tests import the diagnostic script directly and exercise its helper
functions.  Full model-forward tests are intentionally not included here because
those are covered by the script-level smoke runs in the real training
environment.
"""

import sys
from pathlib import Path

import pytest

# The diagnostic script lives in ``scripts/`` rather than ``src/``; make it
# importable as a module for testing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Importing the script pulls in torch / pytorch_lightning; skip the whole module
# if the runtime does not have them.
mtl_lite_diagnose = pytest.importorskip("diagnose_mtl_lite")


def _make_args(splits):
    """Build a minimal argparse Namespace for split resolution tests."""
    return type("Args", (object,), {"split": splits})()


def test_resolve_splits_defaults_to_test():
    args = _make_args(None)
    assert mtl_lite_diagnose.resolve_splits(args) == ["test"]


def test_resolve_splits_accepts_single_val():
    args = _make_args(["val"])
    assert mtl_lite_diagnose.resolve_splits(args) == ["val"]


def test_resolve_splits_accepts_multiple_splits():
    args = _make_args(["val", "test"])
    assert mtl_lite_diagnose.resolve_splits(args) == ["val", "test"]


def test_resolve_splits_removes_duplicates_preserves_order():
    args = _make_args(["test", "val", "test"])
    assert mtl_lite_diagnose.resolve_splits(args) == ["test", "val"]


def test_resolve_splits_rejects_invalid_split():
    args = _make_args(["train", "bad_split"])
    with pytest.raises(ValueError, match="Invalid split"):
        mtl_lite_diagnose.resolve_splits(args)


def test_split_output_root_keeps_test_default(tmp_path):
    output_root = tmp_path / "diagnostics"
    result = mtl_lite_diagnose.split_output_root(output_root, "test", ["test"])
    assert result == output_root


def test_split_output_root_creates_split_subdir_for_val(tmp_path):
    output_root = tmp_path / "diagnostics"
    result = mtl_lite_diagnose.split_output_root(output_root, "val", ["val"])
    assert result == output_root / "val"
    assert result.exists()


def test_split_output_root_creates_split_subdirs_for_multiple_splits(tmp_path):
    output_root = tmp_path / "diagnostics"
    result = mtl_lite_diagnose.split_output_root(output_root, "test", ["val", "test"])
    assert result == output_root / "test"
    assert result.exists()


def test_build_parser_accepts_multiple_splits():
    parser = mtl_lite_diagnose.build_parser()
    args = parser.parse_args(["--run-dir", ".", "--split", "val", "test"])
    assert args.split == ["val", "test"]


def test_build_parser_default_split_is_test():
    parser = mtl_lite_diagnose.build_parser()
    args = parser.parse_args(["--run-dir", "."])
    assert args.split == ["test"]
