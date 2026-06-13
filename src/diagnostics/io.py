import csv
from pathlib import Path

import numpy as np


PREDICTION_FIELDS = [
    "subject_id",
    "true_bdi",
    "pred_bdi",
    "residual",
    "abs_error",
    "severity_group",
]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(csv_path, rows, fieldnames):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def severity_group(score):
    score = float(score)
    if score <= 13:
        return "minimal"
    if score <= 19:
        return "mild"
    if score <= 28:
        return "moderate"
    return "severe"


def build_prediction_records(subject_ids, targets, preds):
    targets = np.asarray(targets, dtype=float).reshape(-1)
    preds = np.asarray(preds, dtype=float).reshape(-1)
    records = []
    for subject_id, target, pred in zip(subject_ids, targets, preds):
        residual = float(pred - target)
        records.append(
            {
                "subject_id": str(subject_id),
                "true_bdi": f"{float(target):.6f}",
                "pred_bdi": f"{float(pred):.6f}",
                "residual": f"{residual:.6f}",
                "abs_error": f"{abs(residual):.6f}",
                "severity_group": severity_group(target),
            }
        )
    return records


def write_prediction_table(csv_path, subject_ids, targets, preds):
    records = build_prediction_records(subject_ids, targets, preds)
    write_csv_rows(csv_path, records, PREDICTION_FIELDS)
    return records


def read_prediction_table(csv_path):
    rows = read_csv_rows(csv_path)
    records = []
    for row in rows:
        try:
            records.append(
                {
                    "subject_id": row["subject_id"],
                    "true_bdi": float(row["true_bdi"]),
                    "pred_bdi": float(row["pred_bdi"]),
                    "residual": float(row["residual"]),
                    "abs_error": float(row["abs_error"]),
                    "severity_group": row.get("severity_group", severity_group(row["true_bdi"])),
                }
            )
        except (KeyError, ValueError):
            continue
    return records


def save_features_npz(save_path, features, subject_ids, targets, preds):
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    np.savez_compressed(
        save_path,
        features=np.asarray(features, dtype=float),
        subject_ids=np.asarray([str(item) for item in subject_ids]),
        true_bdi=np.asarray(targets, dtype=float),
        pred_bdi=np.asarray(preds, dtype=float),
    )


def numeric_columns_from_rows(rows, excluded=()):
    if not rows:
        return {}, []

    excluded = set(excluded)
    columns = {}
    for key in rows[0]:
        if key in excluded:
            continue
        values = []
        valid = True
        for row in rows:
            raw_value = row.get(key, "")
            if raw_value == "":
                valid = False
                break
            try:
                values.append(float(raw_value))
            except ValueError:
                valid = False
                break
        if valid and values:
            columns[key] = np.asarray(values, dtype=float)
    return columns, list(columns.keys())


def select_records(records, strategy="high_error", num_samples=10):
    if not records or num_samples <= 0:
        return []

    records = list(records)
    if strategy == "low_error":
        records.sort(key=lambda row: row["abs_error"])
        return records[:num_samples]

    if strategy == "severity_balanced":
        selected = []
        groups = {}
        for row in records:
            groups.setdefault(row["severity_group"], []).append(row)
        per_group = max(1, int(np.ceil(num_samples / max(1, len(groups)))))
        for group_name in sorted(groups):
            group_rows = sorted(groups[group_name], key=lambda row: row["abs_error"], reverse=True)
            selected.extend(group_rows[:per_group])
        return selected[:num_samples]

    records.sort(key=lambda row: row["abs_error"], reverse=True)
    return records[:num_samples]


def find_checkpoint(run_dir, ckpt="best"):
    run_dir = Path(run_dir)
    if ckpt and ckpt not in {"best", "last"}:
        ckpt_path = Path(ckpt)
        if ckpt_path.exists():
            return ckpt_path
        candidate = run_dir / ckpt
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if ckpt == "last":
        last_path = checkpoint_dir / "last.ckpt"
        if last_path.exists():
            return last_path

    candidates = sorted(checkpoint_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found under: {checkpoint_dir}")

    if ckpt == "last":
        return candidates[-1]

    non_last = [path for path in candidates if path.name != "last.ckpt"]
    return non_last[0] if non_last else candidates[0]
