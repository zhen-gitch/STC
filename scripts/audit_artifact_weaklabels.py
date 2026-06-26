#!/usr/bin/env python
"""RPDF Stage A3: artifact weak-label audit for the z_art factor.

Joins the existing P0 diagnostic summaries (black-artifacts, alignment-geometry,
OpenFace-quality, temporal-sampling) with a prediction CSV and correlates each
weak label with true_bdi / pred_bdi / residual / abs_error.  The result decides
whether z_art enters the first RPDF-lite version.  See
``docs/SHORTCUT_AUDIT_DESIGN.md`` section 13.3.

Example:

    python scripts/audit_artifact_weaklabels.py \\
        --predictions <RUN>/diagnostics/regression/test_predictions.csv \\
        --black-artifacts <RUN>/diagnostics/black_artifacts/tables/black_artifact_summary.csv \\
        --alignment-geometry <RUN>/diagnostics/alignment_geometry/tables/alignment_geometry_summary.csv \\
        --openface-quality <RUN>/diagnostics/openface_quality/tables/openface_quality_summary.csv \\
        --temporal-sampling <RUN>/diagnostics/temporal_sampling/tables/temporal_sampling_summary.csv \\
        --output-dir <RUN>/diagnostics/artifact_weaklabels
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description="RPDF Stage A3: artifact weak-label audit for z_art."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction CSV with residual/abs_error (test_predictions.csv).",
    )
    parser.add_argument("--black-artifacts", default=None, help="black_artifact_summary.csv")
    parser.add_argument("--alignment-geometry", default=None, help="alignment_geometry_summary.csv")
    parser.add_argument("--openface-quality", default=None, help="openface_quality_summary.csv")
    parser.add_argument("--temporal-sampling", default=None, help="temporal_sampling_summary.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--coupling-threshold",
        type=float,
        default=0.2,
        help="|corr with abs_error| above which a weak label is error-coupled (default 0.2).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    from src.diagnostics.artifact_weaklabels import run_artifact_weaklabel_audit

    generated = run_artifact_weaklabel_audit(
        predictions_csv=Path(args.predictions),
        output_dir=Path(args.output_dir),
        black_artifacts_csv=Path(args.black_artifacts) if args.black_artifacts else None,
        alignment_geometry_csv=Path(args.alignment_geometry) if args.alignment_geometry else None,
        openface_quality_csv=Path(args.openface_quality) if args.openface_quality else None,
        temporal_sampling_csv=Path(args.temporal_sampling) if args.temporal_sampling else None,
        coupling_threshold=args.coupling_threshold,
    )
    print("[ARTIFACT-WEAKLABELS] Generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"[ARTIFACT-WEAKLABELS] Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
