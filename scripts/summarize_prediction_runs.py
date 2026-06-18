import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.prediction_runs import summarize_prediction_runs


def _resolve_prediction_csv(path, split="test"):
    path = Path(path)
    if path.is_file():
        return path
    split_file = f"{split}_predictions.csv"
    candidates = [
        path / split_file,
        path / f"{path.name}_{split_file}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(path.glob(f"*{split_file}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find {split} prediction CSV under: {path}"
    )


def _parse_run_spec(spec, split="test"):
    if "=" not in spec:
        raise ValueError(f"Run spec must be NAME=PATH, got: {spec}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Run spec has empty name: {spec}")
    return name, _resolve_prediction_csv(raw_path.strip(), split=split)


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize and compare multiple prediction CSV runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Prediction run as NAME=PATH. PATH may be a CSV or a run directory.",
    )
    parser.add_argument("--baseline", default=None, help="Optional baseline run name for pairwise improvement.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split prediction CSV to look for when PATH is a directory (default: test).",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for summary tables and report.")
    return parser


def main():
    args = build_parser().parse_args()
    run_specs = [_parse_run_spec(spec, split=args.split) for spec in args.run]
    generated = summarize_prediction_runs(
        run_specs,
        output_dir=Path(args.output_dir),
        baseline_name=args.baseline,
    )
    print("[PREDICTION_RUNS] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
