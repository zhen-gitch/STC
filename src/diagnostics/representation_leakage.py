"""Read-only Stage C representation leakage analysis.

The analysis consumes frozen train/validation feature archives produced by
``scripts/diagnose_mtl_lite.py``. Probe parameters and decision thresholds are
fit on train only and then applied to validation.
"""

import math
from pathlib import Path

import numpy as np

from src.diagnostics.identity_retrieval import compute_identity_retrieval_metrics
from src.diagnostics.io import (
    REPRESENTATION_FEATURE_KEYS,
    ensure_dir,
    severity_group,
    write_csv_rows,
)


REPRESENTATION_NAMES = {
    "features_h0": "h0",
    "features_z_dep": "z_dep",
    "features_z_nuisance": "z_nuisance",
}

SUMMARY_COLUMNS = [
    "run_name",
    "representation",
    "status",
    "train_count",
    "val_count",
    "feature_dim",
    "bdi_mae",
    "bdi_rmse",
    "bdi_pearson",
    "bdi_ccc",
    "task_accuracy",
    "task_auroc",
    "identity_pair_balanced_accuracy",
    "identity_pair_auroc",
    "identity_pair_threshold",
    "same_subject_top1_rate",
    "same_subject_top5_rate",
    "paired_task_rank_mean",
]

BDI_PREDICTION_COLUMNS = [
    "run_name",
    "representation",
    "video_id",
    "subject_id",
    "task_name",
    "true_bdi",
    "probe_pred_bdi",
]

GATE_COLUMNS = [
    "run_name",
    "z_dep_mae",
    "z_nuisance_mae",
    "mae_gap_nuisance_minus_dep",
    "z_dep_ccc",
    "z_nuisance_ccc",
    "ccc_gap_dep_minus_nuisance",
    "bdi_leakage_triggered",
    "reason",
]


def _task_from_video_id(video_id):
    text = str(video_id)
    if "Freeform" in text:
        return "Freeform"
    if "Northwind" in text:
        return "Northwind"
    return ""


def _string_array(data, key, count, fallback=None):
    if key in data.files:
        values = np.asarray(data[key]).astype(str)
    elif fallback is not None:
        values = np.asarray(fallback).astype(str)
    else:
        values = np.asarray([""] * count)
    if values.shape != (count,):
        raise ValueError(f"{key} must have shape ({count},), got {values.shape}")
    return values


def load_representation_bundle(npz_path):
    """Load and validate one Stage C feature archive without modifying it."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Representation archive not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        required = {"subject_ids", "true_bdi", "pred_bdi"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Missing NPZ field(s): {', '.join(sorted(missing))}")

        subject_ids = np.asarray(data["subject_ids"]).astype(str)
        count = len(subject_ids)
        true_bdi = np.asarray(data["true_bdi"], dtype=float).reshape(-1)
        pred_bdi = np.asarray(data["pred_bdi"], dtype=float).reshape(-1)
        if len(true_bdi) != count or len(pred_bdi) != count:
            raise ValueError("NPZ metadata arrays must have the same row count")
        if not np.isfinite(true_bdi).all() or not np.isfinite(pred_bdi).all():
            raise ValueError("NPZ targets and predictions must be finite")

        video_ids = _string_array(data, "video_ids", count, fallback=subject_ids)
        task_names = _string_array(data, "task_names", count)
        task_names = np.asarray(
            [name or _task_from_video_id(video_id) for name, video_id in zip(task_names, video_ids)]
        )

        representations = {}
        statuses = {}
        for key in REPRESENTATION_FEATURE_KEYS:
            name = REPRESENTATION_NAMES[key]
            if key not in data.files:
                status_key = f"{key}_status"
                status = (
                    str(np.asarray(data[status_key]).item())
                    if status_key in data.files
                    else "missing"
                )
                representations[name] = None
                statuses[name] = status
                continue
            values = np.asarray(data[key], dtype=float)
            if values.ndim != 2 or values.shape[0] != count:
                raise ValueError(
                    f"{key} must have shape ({count}, D), got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{key} contains non-finite values")
            representations[name] = values
            statuses[name] = "available"

    return {
        "path": npz_path,
        "subject_ids": subject_ids,
        "video_ids": video_ids,
        "task_names": task_names,
        "true_bdi": true_bdi,
        "pred_bdi": pred_bdi,
        "representations": representations,
        "statuses": statuses,
    }


def _standardize_train_val(train_x, val_x):
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (train_x - mean) / scale, (val_x - mean) / scale


def fit_ridge_predict(train_x, train_y, val_x, alpha=1.0):
    """Fit a centered ridge regressor on train and predict validation."""
    train_x = np.asarray(train_x, dtype=float)
    val_x = np.asarray(val_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float).reshape(-1)
    if train_x.ndim != 2 or val_x.ndim != 2:
        raise ValueError("Ridge features must be 2D")
    if train_x.shape[0] != len(train_y) or train_x.shape[1] != val_x.shape[1]:
        raise ValueError("Ridge train/validation shapes are incompatible")
    if alpha <= 0 or not math.isfinite(float(alpha)):
        raise ValueError("Ridge alpha must be finite and positive")

    train_scaled, val_scaled = _standardize_train_val(train_x, val_x)
    y_mean = float(train_y.mean())
    gram = train_scaled.T @ train_scaled
    rhs = train_scaled.T @ (train_y - y_mean)
    weights = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)
    return y_mean + val_scaled @ weights


def _pearson(targets, preds):
    targets = np.asarray(targets, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if len(targets) < 2 or np.std(targets) < 1e-12 or np.std(preds) < 1e-12:
        return None
    return float(np.corrcoef(targets, preds)[0, 1])


def concordance_ccc(targets, preds):
    targets = np.asarray(targets, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if len(targets) < 2:
        return None
    covariance = float(np.mean((targets - targets.mean()) * (preds - preds.mean())))
    denominator = float(targets.var() + preds.var() + (targets.mean() - preds.mean()) ** 2)
    return 0.0 if denominator < 1e-12 else 2.0 * covariance / denominator


def regression_metrics(targets, preds):
    targets = np.asarray(targets, dtype=float)
    preds = np.asarray(preds, dtype=float)
    errors = preds - targets
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "pearson": _pearson(targets, preds),
        "ccc": concordance_ccc(targets, preds),
    }


def binary_auroc(labels, scores):
    """Compute binary AUROC by pairwise score ordering, including ties."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _balanced_accuracy(labels, predicted):
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    recalls = []
    for value in (0, 1):
        mask = labels == value
        if mask.any():
            recalls.append(float(np.mean(predicted[mask] == value)))
    return float(np.mean(recalls)) if recalls else None


def tune_binary_threshold(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    unique = np.unique(scores)
    if len(unique) == 1:
        return float(unique[0])
    candidates = np.concatenate(
        ([unique[0] - 1e-12], (unique[:-1] + unique[1:]) / 2.0, [unique[-1] + 1e-12])
    )
    ranked = [
        (_balanced_accuracy(labels, scores >= threshold), -abs(float(threshold)), float(threshold))
        for threshold in candidates
    ]
    return max(ranked)[2]


def _binary_probe(train_x, train_labels, val_x, val_labels, alpha):
    classes = sorted(set(str(value) for value in train_labels))
    if len(classes) != 2 or not set(str(value) for value in val_labels).issubset(classes):
        return {"accuracy": None, "auroc": None}
    positive = classes[1]
    train_y = np.asarray([int(str(value) == positive) for value in train_labels])
    val_y = np.asarray([int(str(value) == positive) for value in val_labels])
    train_scores = fit_ridge_predict(train_x, train_y, train_x, alpha=alpha)
    val_scores = fit_ridge_predict(train_x, train_y, val_x, alpha=alpha)
    threshold = tune_binary_threshold(train_y, train_scores)
    return {
        "accuracy": float(np.mean((val_scores >= threshold) == val_y)),
        "auroc": binary_auroc(val_y, val_scores),
    }


def _pair_scores(features, subject_ids):
    features = np.asarray(features, dtype=float)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = np.divide(features, norms, out=np.zeros_like(features), where=norms > 1e-12)
    labels = []
    scores = []
    for left in range(len(features)):
        for right in range(left + 1, len(features)):
            labels.append(int(str(subject_ids[left]) == str(subject_ids[right])))
            scores.append(float(normalized[left] @ normalized[right]))
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def identity_pair_verifier(train_x, train_subjects, val_x, val_subjects):
    train_labels, train_scores = _pair_scores(train_x, train_subjects)
    val_labels, val_scores = _pair_scores(val_x, val_subjects)
    if len(train_labels) == 0 or len(val_labels) == 0:
        return {"balanced_accuracy": None, "auroc": None, "threshold": None}
    if len(np.unique(train_labels)) < 2 or len(np.unique(val_labels)) < 2:
        return {"balanced_accuracy": None, "auroc": None, "threshold": None}
    threshold = tune_binary_threshold(train_labels, train_scores)
    return {
        "balanced_accuracy": _balanced_accuracy(val_labels, val_scores >= threshold),
        "auroc": binary_auroc(val_labels, val_scores),
        "threshold": threshold,
    }


def _records_from_bundle(bundle):
    records = []
    for index in range(len(bundle["subject_ids"])):
        target = float(bundle["true_bdi"][index])
        records.append(
            {
                "video_id": str(bundle["video_ids"][index]),
                "subject_id": str(bundle["subject_ids"][index]),
                "task_name": str(bundle["task_names"][index]),
                "true_bdi": target,
                "pred_bdi": float(bundle["pred_bdi"][index]),
                "severity_group": severity_group(target),
            }
        )
    return records


def _bundle_signature(bundle):
    signature = {}
    for video_id, subject_id, task_name, target in zip(
        bundle["video_ids"],
        bundle["subject_ids"],
        bundle["task_names"],
        bundle["true_bdi"],
    ):
        key = str(video_id)
        if key in signature:
            raise ValueError(f"Duplicate video_id in representation bundle: {key}")
        signature[key] = (str(subject_id), str(task_name), float(target))
    return signature


def validate_bundle_protocol(run_name, train_bundle, val_bundle):
    train_subjects = set(str(value) for value in train_bundle["subject_ids"])
    val_subjects = set(str(value) for value in val_bundle["subject_ids"])
    if train_subjects & val_subjects:
        raise ValueError(f"{run_name} train/val subject overlap detected")
    train_videos = set(str(value) for value in train_bundle["video_ids"])
    val_videos = set(str(value) for value in val_bundle["video_ids"])
    if train_videos & val_videos:
        raise ValueError(f"{run_name} train/val video overlap detected")


def analyze_run_representations(run_name, train_bundle, val_bundle, ridge_alpha=1.0):
    """Compute the read-only leakage matrix for one run."""
    rows = []
    prediction_rows = []
    val_records = _records_from_bundle(val_bundle)
    for representation in ("h0", "z_dep", "z_nuisance"):
        train_x = train_bundle["representations"].get(representation)
        val_x = val_bundle["representations"].get(representation)
        if train_x is None or val_x is None:
            status = train_bundle["statuses"].get(representation, "missing")
            if val_bundle["statuses"].get(representation) != status:
                status = "inconsistent_availability"
            rows.append(
                {
                    "run_name": run_name,
                    "representation": representation,
                    "status": status,
                    "train_count": len(train_bundle["subject_ids"]),
                    "val_count": len(val_bundle["subject_ids"]),
                    "feature_dim": None,
                }
            )
            continue
        if train_x.shape[1] != val_x.shape[1]:
            raise ValueError(
                f"{run_name}/{representation} train/val dimensions differ: "
                f"{train_x.shape[1]} vs {val_x.shape[1]}"
            )

        bdi_preds = fit_ridge_predict(
            train_x,
            train_bundle["true_bdi"],
            val_x,
            alpha=ridge_alpha,
        )
        bdi = regression_metrics(val_bundle["true_bdi"], bdi_preds)
        task = _binary_probe(
            train_x,
            train_bundle["task_names"],
            val_x,
            val_bundle["task_names"],
            alpha=ridge_alpha,
        )
        verifier = identity_pair_verifier(
            train_x,
            train_bundle["subject_ids"],
            val_x,
            val_bundle["subject_ids"],
        )
        _, retrieval = compute_identity_retrieval_metrics(val_records, val_x)
        rows.append(
            {
                "run_name": run_name,
                "representation": representation,
                "status": "available",
                "train_count": train_x.shape[0],
                "val_count": val_x.shape[0],
                "feature_dim": train_x.shape[1],
                "bdi_mae": bdi["mae"],
                "bdi_rmse": bdi["rmse"],
                "bdi_pearson": bdi["pearson"],
                "bdi_ccc": bdi["ccc"],
                "task_accuracy": task["accuracy"],
                "task_auroc": task["auroc"],
                "identity_pair_balanced_accuracy": verifier["balanced_accuracy"],
                "identity_pair_auroc": verifier["auroc"],
                "identity_pair_threshold": verifier["threshold"],
                "same_subject_top1_rate": retrieval["same_subject_top1_rate"],
                "same_subject_top5_rate": retrieval["same_subject_top5_rate"],
                "paired_task_rank_mean": retrieval["paired_task_rank_mean"],
            }
        )
        for index, pred in enumerate(bdi_preds):
            prediction_rows.append(
                {
                    "run_name": run_name,
                    "representation": representation,
                    "video_id": str(val_bundle["video_ids"][index]),
                    "subject_id": str(val_bundle["subject_ids"][index]),
                    "task_name": str(val_bundle["task_names"][index]),
                    "true_bdi": float(val_bundle["true_bdi"][index]),
                    "probe_pred_bdi": float(pred),
                }
            )
    return rows, prediction_rows


def evaluate_bdi_leakage_gate(summary_rows):
    """Apply the frozen z_nuisance BDI leakage rule per run."""
    by_run = {}
    for row in summary_rows:
        by_run.setdefault(row["run_name"], {})[row["representation"]] = row
    results = []
    for run_name, representations in sorted(by_run.items()):
        dep = representations.get("z_dep")
        nuisance = representations.get("z_nuisance")
        if (
            dep is None
            or nuisance is None
            or dep.get("status") != "available"
            or nuisance.get("status") != "available"
        ):
            results.append(
                {
                    "run_name": run_name,
                    "bdi_leakage_triggered": None,
                    "reason": "not_applicable",
                }
            )
            continue
        ccc_gap = dep["bdi_ccc"] - nuisance["bdi_ccc"]
        mae_gap = nuisance["bdi_mae"] - dep["bdi_mae"]
        ccc_trigger = ccc_gap <= 0.05
        mae_trigger = mae_gap <= 0.50
        reasons = []
        if ccc_trigger:
            reasons.append("ccc_gap<=0.05")
        if mae_trigger:
            reasons.append("nuisance_mae<=dep_mae+0.50")
        results.append(
            {
                "run_name": run_name,
                "z_dep_mae": dep["bdi_mae"],
                "z_nuisance_mae": nuisance["bdi_mae"],
                "mae_gap_nuisance_minus_dep": mae_gap,
                "z_dep_ccc": dep["bdi_ccc"],
                "z_nuisance_ccc": nuisance["bdi_ccc"],
                "ccc_gap_dep_minus_nuisance": ccc_gap,
                "bdi_leakage_triggered": ccc_trigger or mae_trigger,
                "reason": ";".join(reasons) if reasons else "clear",
            }
        )
    return results


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.6f}"
    return value


def _write_rows(path, rows, columns):
    formatted = [
        {column: _format_value(row.get(column)) for column in columns}
        for row in rows
    ]
    write_csv_rows(path, formatted, columns)
    return Path(path)


def _write_report(report_path, summary_rows, gate_rows, generated):
    lines = [
        "# Stage C Representation Leakage Report",
        "",
        "All probes are fit/tuned on train and evaluated on validation. No checkpoint,",
        "feature archive, prediction table, split file, or test data is modified or opened.",
        "",
        "## Leakage Matrix",
        "",
        "| Run | Representation | BDI MAE | BDI CCC | Task AUROC | Identity pair AUROC | Same-subject top1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['run_name']} | {row['representation']} ({row['status']}) | "
            f"{_format_value(row.get('bdi_mae'))} | {_format_value(row.get('bdi_ccc'))} | "
            f"{_format_value(row.get('task_auroc'))} | "
            f"{_format_value(row.get('identity_pair_auroc'))} | "
            f"{_format_value(row.get('same_subject_top1_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## Frozen BDI Leakage Gate",
            "",
            "A split run triggers when CCC(z_dep)-CCC(z_nuisance) <= 0.05 or",
            "MAE(z_nuisance) <= MAE(z_dep)+0.50.",
            "",
        ]
    )
    for row in gate_rows:
        lines.append(
            f"- {row['run_name']}: triggered={row.get('bdi_leakage_triggered')} "
            f"reason={row.get('reason', '')}"
        )
    lines.extend(["", "## Generated Files", ""])
    lines.extend(f"- `{path}`" for path in generated)
    report_path = Path(report_path)
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_representation_leakage_analysis(run_specs, output_dir, ridge_alpha=1.0):
    """Run the multi-run train-to-validation representation audit."""
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    summary_rows = []
    prediction_rows = []
    reference_signatures = None
    for run_name, train_npz, val_npz in run_specs:
        train_bundle = load_representation_bundle(train_npz)
        val_bundle = load_representation_bundle(val_npz)
        validate_bundle_protocol(run_name, train_bundle, val_bundle)
        signatures = (_bundle_signature(train_bundle), _bundle_signature(val_bundle))
        if reference_signatures is None:
            reference_signatures = signatures
        elif signatures != reference_signatures:
            raise ValueError(
                f"{run_name} train/val metadata differs from the reference run"
            )
        rows, preds = analyze_run_representations(
            run_name,
            train_bundle,
            val_bundle,
            ridge_alpha=ridge_alpha,
        )
        summary_rows.extend(rows)
        prediction_rows.extend(preds)

    gate_rows = evaluate_bdi_leakage_gate(summary_rows)
    generated = []
    generated.append(
        _write_rows(
            tables_dir / "representation_leakage_summary.csv",
            summary_rows,
            SUMMARY_COLUMNS,
        )
    )
    generated.append(
        _write_rows(
            tables_dir / "bdi_probe_predictions.csv",
            prediction_rows,
            BDI_PREDICTION_COLUMNS,
        )
    )
    generated.append(
        _write_rows(
            tables_dir / "bdi_leakage_gate.csv",
            gate_rows,
            GATE_COLUMNS,
        )
    )
    report = _write_report(
        reports_dir / "representation_leakage_report.md",
        summary_rows,
        gate_rows,
        generated,
    )
    generated.append(report)
    return generated
