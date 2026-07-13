import csv

import numpy as np
import pytest

from src.diagnostics.io import save_features_npz
from src.diagnostics.representation_leakage import (
    analyze_run_representations,
    binary_auroc,
    evaluate_bdi_leakage_gate,
    identity_pair_verifier,
    load_representation_bundle,
    run_representation_leakage_analysis,
)


def _bundle_arrays(subject_start, count=8):
    rng = np.random.default_rng(subject_start)
    subject_ids = []
    video_ids = []
    task_names = []
    targets = []
    z_dep = []
    z_nuisance = []
    h0 = []
    for offset in range(count):
        subject = f"S{subject_start + offset:03d}"
        target = float(5 + 4 * offset)
        identity = np.zeros(count)
        identity[offset] = 1.0
        for task_index, task in enumerate(("Freeform", "Northwind")):
            task_signal = -1.0 if task_index == 0 else 1.0
            dep = np.asarray([target / 40.0, task_signal * 0.05, 1.0, offset / count])
            nuisance = np.asarray([target / 40.0, task_signal, 1.0, offset / count])
            subject_ids.append(subject)
            video_ids.append(f"{subject}_{task}_video")
            task_names.append(task)
            targets.append(target)
            z_dep.append(dep + rng.normal(0, 0.002, len(dep)))
            z_nuisance.append(nuisance + rng.normal(0, 0.002, len(nuisance)))
            h0.append(np.concatenate([dep, nuisance, 10.0 * identity]))
    return {
        "subject_ids": subject_ids,
        "video_ids": video_ids,
        "task_names": task_names,
        "targets": np.asarray(targets),
        "preds": np.asarray(targets) + 0.5,
        "h0": np.asarray(h0),
        "z_dep": np.asarray(z_dep),
        "z_nuisance": np.asarray(z_nuisance),
    }


def _write_bundle(path, arrays, nuisance=True):
    representations = {
        "features_h0": arrays["h0"],
        "features_z_dep": arrays["z_dep"],
        "features_z_nuisance": arrays["z_nuisance"] if nuisance else None,
    }
    save_features_npz(
        path,
        features=arrays["z_dep"],
        subject_ids=arrays["subject_ids"],
        targets=arrays["targets"],
        preds=arrays["preds"],
        video_ids=arrays["video_ids"],
        task_names=arrays["task_names"],
        representation_features=representations,
    )
    return path


def test_load_representation_bundle_preserves_not_applicable(tmp_path):
    arrays = _bundle_arrays(0)
    path = _write_bundle(tmp_path / "features.npz", arrays, nuisance=False)

    bundle = load_representation_bundle(path)

    assert bundle["representations"]["h0"].shape[0] == len(arrays["targets"])
    assert bundle["representations"]["z_nuisance"] is None
    assert bundle["statuses"]["z_nuisance"] == "not_applicable"
    assert set(bundle["task_names"]) == {"Freeform", "Northwind"}


def test_binary_auroc_handles_perfect_and_tied_scores():
    assert binary_auroc([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0]) == 1.0
    assert binary_auroc([0, 1], [0.5, 0.5]) == 0.5


def test_identity_pair_verifier_uses_train_threshold_and_val_pairs():
    train = _bundle_arrays(0)
    val = _bundle_arrays(100)
    result = identity_pair_verifier(
        train["h0"],
        train["subject_ids"],
        val["h0"],
        val["subject_ids"],
    )

    assert result["threshold"] is not None
    assert result["auroc"] >= 0.9
    assert result["balanced_accuracy"] >= 0.8


def test_analyze_run_representations_emits_available_and_na_rows(tmp_path):
    train_path = _write_bundle(tmp_path / "train.npz", _bundle_arrays(0), nuisance=False)
    val_path = _write_bundle(tmp_path / "val.npz", _bundle_arrays(100), nuisance=False)

    rows, predictions = analyze_run_representations(
        "C-BN",
        load_representation_bundle(train_path),
        load_representation_bundle(val_path),
    )

    by_name = {row["representation"]: row for row in rows}
    assert by_name["h0"]["status"] == "available"
    assert by_name["z_dep"]["bdi_mae"] < 1.0
    assert by_name["z_nuisance"]["status"] == "not_applicable"
    assert len(predictions) == 2 * len(_bundle_arrays(100)["targets"])


def test_bdi_leakage_gate_triggers_when_nuisance_matches_dep():
    rows = [
        {
            "run_name": "C-FULL",
            "representation": "z_dep",
            "status": "available",
            "bdi_mae": 4.0,
            "bdi_ccc": 0.60,
        },
        {
            "run_name": "C-FULL",
            "representation": "z_nuisance",
            "status": "available",
            "bdi_mae": 4.3,
            "bdi_ccc": 0.57,
        },
    ]

    gate = evaluate_bdi_leakage_gate(rows)[0]

    assert gate["bdi_leakage_triggered"] is True
    assert "ccc_gap<=0.05" in gate["reason"]
    assert "nuisance_mae<=dep_mae+0.50" in gate["reason"]


def test_run_representation_leakage_analysis_is_read_only(tmp_path):
    train_path = _write_bundle(tmp_path / "train.npz", _bundle_arrays(0))
    val_path = _write_bundle(tmp_path / "val.npz", _bundle_arrays(100))
    before = (train_path.read_bytes(), val_path.read_bytes())

    generated = run_representation_leakage_analysis(
        [("C-FULL", train_path, val_path)],
        tmp_path / "output",
    )

    assert (train_path.read_bytes(), val_path.read_bytes()) == before
    assert len(generated) == 4
    summary_path = tmp_path / "output" / "tables" / "representation_leakage_summary.csv"
    gate_path = tmp_path / "output" / "tables" / "bdi_leakage_gate.csv"
    assert summary_path.exists()
    assert gate_path.exists()
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["representation"] for row in rows} == {"h0", "z_dep", "z_nuisance"}
    assert (tmp_path / "output" / "reports" / "representation_leakage_report.md").exists()


def test_load_representation_bundle_rejects_nonfinite_features(tmp_path):
    arrays = _bundle_arrays(0)
    arrays["z_dep"][0, 0] = np.nan
    path = _write_bundle(tmp_path / "bad.npz", arrays)

    with pytest.raises(ValueError, match="non-finite"):
        load_representation_bundle(path)


def test_multi_run_analysis_rejects_mismatched_metadata(tmp_path):
    train_a = _write_bundle(tmp_path / "train_a.npz", _bundle_arrays(0))
    val_a = _write_bundle(tmp_path / "val_a.npz", _bundle_arrays(100))
    train_b = _write_bundle(tmp_path / "train_b.npz", _bundle_arrays(0))
    val_b = _write_bundle(tmp_path / "val_b.npz", _bundle_arrays(200))

    with pytest.raises(ValueError, match="metadata differs"):
        run_representation_leakage_analysis(
            [("C-REF", train_a, val_a), ("C-BN", train_b, val_b)],
            tmp_path / "output",
        )
