"""Tests for scripts/diagnose_mtl_lite.py split handling and output layout.

These tests import the diagnostic script directly and exercise its helper
functions.  Full model-forward tests are intentionally not included here because
those are covered by the script-level smoke runs in the real training
environment.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.outputs import MTLLiteOutput

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


class _DiagnosticModel:
    def __init__(self, split=False):
        self.split = split

    def __call__(self, video_tensor, mask, return_features=False):
        batch_size = video_tensor.size(0)
        shared = torch.arange(batch_size * 4, dtype=torch.float32).reshape(
            batch_size, 4
        )
        return MTLLiteOutput(
            bdi_pred=torch.full((batch_size,), 0.5),
            shared_features=shared,
            h0_features=torch.ones(batch_size, 8) if self.split else None,
            z_dep_features=shared if self.split else None,
            z_nuisance_features=torch.full((batch_size, 4), 2.0)
            if self.split
            else None,
        )

    @staticmethod
    def prediction_for_metrics(predictions):
        return predictions * 63.0


def _diagnostic_loader():
    return [
        (
            torch.randn(2, 3, 3, 8, 8),
            torch.ones(2, 3, dtype=torch.bool),
            {
                "video_id": ["001_Freeform", "002_Northwind"],
                "subject_id": ["001", "002"],
                "task_name": ["Freeform", "Northwind"],
                "bdi_score": torch.tensor([10.0, 20.0]),
            },
        )
    ]


def test_reference_diagnostic_bundle_aliases_h0_and_z_dep_to_prediction_features():
    collected = mtl_lite_diagnose.collect_predictions_and_features(
        _DiagnosticModel(split=False), _diagnostic_loader(), torch.device("cpu")
    )
    representations = collected[-1]

    np.testing.assert_allclose(representations["features_h0"], collected[-2])
    np.testing.assert_allclose(representations["features_z_dep"], collected[-2])
    assert representations["features_z_nuisance"] is None


def test_split_diagnostic_bundle_exports_all_representations_from_one_forward():
    collected = mtl_lite_diagnose.collect_predictions_and_features(
        _DiagnosticModel(split=True), _diagnostic_loader(), torch.device("cpu")
    )
    representations = collected[-1]

    assert representations["features_h0"].shape == (2, 8)
    assert representations["features_z_dep"].shape == (2, 4)
    assert representations["features_z_nuisance"].shape == (2, 4)
