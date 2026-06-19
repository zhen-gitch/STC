#!/usr/bin/env python
"""Offline entry point for the embedding identity retrieval audit.

This script reads a video-level embedding archive produced by
``scripts/diagnose_mtl_lite.py`` and runs paired-task retrieval diagnostics
to check whether the RGB model embeddings encode subject identity / static
appearance.  See ``docs/RGB_OVERFITTING_AUDIT_PLAN.md`` (P0-E) for the
research context.

Example:

    python scripts/audit_identity_retrieval.py \
        --features-npz <LOG_DIR>/default/rgb/version_0/diagnostics/embeddings/test_features.npz \
        --predictions <LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
        --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/identity_retrieval

When ``diagnose_mtl_lite.py`` is run with multiple splits, the artifacts move to
``<run_dir>/diagnostics/<split>/embeddings/<split>_features.npz`` and
``<run_dir>/diagnostics/<split>/regression/<split>_predictions.csv``.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.identity_retrieval import run_embedding_identity_retrieval_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit RGB/MTL-Lite embeddings for subject-identity shortcut."
    )
    parser.add_argument(
        "--features-npz",
        required=True,
        help="Path to the *_features.npz file produced by diagnose_mtl_lite.py.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional test_predictions.csv for video_id / task_name enrichment.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where tables, figures, and reports are written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Neighborhood size for severity / task agreement metrics (default 5).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_embedding_identity_retrieval_audit(
        features_npz=args.features_npz,
        output_dir=args.output_dir,
        predictions_csv=args.predictions,
        top_k=args.top_k,
    )
    print("[IDENTITY RETRIEVAL] Generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"[IDENTITY RETRIEVAL] Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
