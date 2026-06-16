import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.training_overfit import summarize_training_overfit


def _resolve_metrics_csv(path):
    path = Path(path)
    if path.is_file():
        return path
    candidates = [
        path / "metrics.csv",
        path / "csv" / "metrics.csv",
    ]
    candidates.extend(sorted(path.glob("**/metrics.csv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find metrics.csv under: {path}")


def _parse_run_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Run spec must be NAME=PATH, got: {spec}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Run spec has empty name: {spec}")
    return name, _resolve_metrics_csv(raw_path.strip())


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize train/validation overfit gaps across metrics.csv runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Training run as NAME=PATH. PATH may be a metrics.csv file or a run directory.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for summary tables and report.")
    return parser


def main():
    args = build_parser().parse_args()
    run_specs = [_parse_run_spec(spec) for spec in args.run]
    generated = summarize_training_overfit(run_specs, output_dir=Path(args.output_dir))
    print("[TRAINING_OVERFIT] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
