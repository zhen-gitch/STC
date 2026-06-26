#!/usr/bin/env python
"""RPDF Stage A2: prediction error x identity similarity coupling.

Consumes a layer-wise embedding NPZ (A1 output) plus a prediction CSV and
reports whether prediction error co-varies with subject identity signals across
layers.  See ``docs/SHORTCUT_AUDIT_DESIGN.md`` section 13.2.

Example:

    python scripts/audit_error_identity_coupling.py \\
        --features-npz <RUN>/diagnostics/layerwise_identity/test_layerwise_features.npz \\
        --predictions <RUN>/diagnostics/regression/test_predictions.csv \\
        --per-query-csv <RUN>/diagnostics/layerwise_identity/tables/layerwise_identity_per_query.csv \\
        --output-dir <RUN>/diagnostics/error_identity_coupling
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description="RPDF Stage A2: error x identity coupling audit."
    )
    parser.add_argument(
        "--features-npz",
        required=True,
        help="Layer-wise NPZ from A1 (audit_layerwise_identity_probe.py).",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction CSV with residual/abs_error (test_predictions.csv).",
    )
    parser.add_argument(
        "--per-query-csv",
        default=None,
        help="Optional A1 layerwise_identity_per_query.csv for rank/agreement signals.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer names to include (defaults to all in NPZ).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=20,
        help="Max high-coupling cases retained per layer.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    from src.diagnostics.error_identity_coupling import run_error_identity_coupling_audit

    layer_order = None
    if args.layers:
        layer_order = [name.strip() for name in args.layers.split(",") if name.strip()]

    generated = run_error_identity_coupling_audit(
        features_npz=Path(args.features_npz),
        predictions_csv=Path(args.predictions),
        output_dir=Path(args.output_dir),
        per_query_csv=Path(args.per_query_csv) if args.per_query_csv else None,
        layer_order=layer_order,
        max_cases=args.max_cases,
    )
    print("[ERROR-IDENTITY] Generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"[ERROR-IDENTITY] Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
