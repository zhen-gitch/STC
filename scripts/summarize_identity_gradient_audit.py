#!/usr/bin/env python
import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.identity_gradient_audit import (
    summarize_identity_gradient_audit,
)


def _resolve_metrics_csv(path):
    path = Path(path)
    if path.is_file():
        return path
    candidates = [path / "metrics.csv"]
    candidates.extend(sorted(path.glob("**/metrics.csv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find metrics.csv under: {path}")


def _parse_run(spec):
    if "=" not in spec:
        raise ValueError(f"Run spec must be NAME=PATH, got: {spec}")
    name, path = spec.split("=", 1)
    if not name.strip():
        raise ValueError(f"Run spec has an empty name: {spec}")
    return name.strip(), _resolve_metrics_csv(path.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Summarize BDI versus reversed-identity gradient interaction."
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    generated = summarize_identity_gradient_audit(
        [_parse_run(spec) for spec in args.run],
        output_dir=args.output_dir,
    )
    print("[IDENTITY-GRADIENT-AUDIT] generated:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
