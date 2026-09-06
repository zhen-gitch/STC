"""Cross-run matrix report: utility + external identity metrics in one table.

Collects, per run under a log root: ``summary.json`` (best CCC, epochs,
provenance flags), ``val_metrics.json`` (best-epoch utility) and
``identity/identity_metrics.json`` (LOVO attacker / pair-AUROC / A1 per exit,
written by ``src.eva_di.identity_metrics``).  Missing identity evidence is
reported as blank cells -- never as zeros.  Output: CSV + markdown table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.eva_di.contracts import EvaDiError

COLUMNS = ["run_id", "best_epoch", "best_val_ccc", "best_defined", "epochs_run",
           "stopped_early", "val_n", "n_subjects", "max_recordings",
           "p_attacker_top1", "p_attacker_top3", "p_pair_auroc",
           "p_a1_top1", "p_a1_top3",
           "v_attacker_top1", "v_attacker_top3", "v_pair_auroc",
           "val_rmse", "val_pcc",
           "v_a1_top1", "v_a1_top3", "checkpoint"]


def _exit_row(identity_report: list[dict], exit_name: str, key: str):
    for item in identity_report:
        if item.get("exit") != exit_name:
            continue
        if key == "attacker_top1":
            return item["attacker"]["top1_accuracy"]
        if key == "attacker_top3":
            return item["attacker"]["top3_accuracy"]
        if key == "pair_auroc":
            return item["pair_auroc"]["auc"]
        if key == "a1_top1":
            return item["a1_retrieval"]["top1_same_subject"]
        if key == "a1_top3":
            return item["a1_retrieval"]["top3_same_subject"]
    return None


def collect_rows(runs_root: Path) -> list[dict]:
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise EvaDiError(f"runs root missing: {runs_root}")
    rows: list[dict] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue  # not a completed run dir
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        identity_path = run_dir / "identity" / "identity_metrics.json"
        identity = (json.loads(identity_path.read_text(encoding="utf-8"))
                    if identity_path.is_file() else [])
        row = {"run_id": run_dir.name,
               "best_epoch": summary.get("best_epoch"),
               "best_val_ccc": summary.get("best_val_ccc"),
               "best_defined": summary.get("best_defined"),
               "epochs_run": summary.get("epochs_run"),
               "stopped_early": summary.get("stopped_early"),
               "val_n": summary.get("val_n"),
               "n_subjects": summary.get("n_subjects"),
               "max_recordings": summary.get("max_recordings"),
               "val_rmse": (summary.get("final") or {}).get("val_rmse"),
               "val_pcc": (summary.get("final") or {}).get("val_pcc"),
               "checkpoint": summary.get("checkpoint")}
        for exit_name, prefix in (("p_mean", "p_"), ("v_h0", "v_")):
            for key in ("attacker_top1", "attacker_top3", "pair_auroc",
                        "a1_top1", "a1_top3"):
                row[prefix + key] = _exit_row(identity, exit_name, key)
        rows.append(row)
    return rows


def write_reports(rows: list[dict], output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "matrix_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k))
                             for k in COLUMNS})
    md = ["| " + " | ".join(COLUMNS) + " |",
          "|" + "---|" * len(COLUMNS)]
    for row in rows:
        cells = ["" if row.get(k) is None else
                 (f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k]))
                 for k in COLUMNS]
        md.append("| " + " | ".join(cells) + " |")
    md_path = output_dir / "matrix_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="EVA-DI matrix report")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    rows = collect_rows(Path(args.runs_root))
    if not rows:
        raise EvaDiError(f"no completed runs (summary.json) under {args.runs_root}")
    for path in write_reports(rows, Path(args.output_dir)):
        print(str(path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
