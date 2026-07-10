#!/usr/bin/env python
"""RPDF Stage A3: OpenFace quality summary audit (thin wrapper).

Runs the existing OpenFace-quality summarization
(:func:`src.diagnostics.openface_quality.summarize_openface_root`) over an
OpenFace CSV directory and writes ``tables/openface_quality_summary.csv`` --
the upstream summary consumed by
:mod:`src.diagnostics.artifact_weaklabels` (the A3 weak-label join).

This is the OpenFace-quality counterpart of ``audit_black_artifacts.py`` /
``audit_alignment_geometry.py``: a thin CLI over an already-implemented
``src.diagnostics`` module.  It exists as a standalone script so the Stage B
diagnose chain (and ``scripts/stage_b/run_a3_artifacts.sh``) can produce the
openface summary without invoking the larger ``run_shortcut_audit`` entrypoint.

Example::

    python scripts/audit_openface_quality.py \\
        --openface-root <AVEC2014>/openface \\
        --output-dir <RUN>/diagnostics/test/openface_quality

    # or resolve the root from a run's resolved_config.yaml:
    python scripts/audit_openface_quality.py \\
        --run-dir <RUN> --output-dir <RUN>/diagnostics/test/openface_quality
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.openface_quality import (  # noqa: E402
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    summarize_openface_root,
    write_openface_quality_summary,
)


def _resolve_openface_root(args):
    """Resolve the OpenFace CSV root from --openface-root or a run dir.

    ``--openface-root`` wins.  Otherwise read ``DATASET.OPENFACE_ROOT`` from the
    run's ``resolved_config.yaml`` (the same path the Stage B trainer saves).
    Returns ``None`` when unset, so the caller can degrade gracefully.
    """
    if args.openface_root:
        return Path(args.openface_root)
    if not args.run_dir:
        return None
    cfg_path = Path(args.run_dir) / "resolved_config.yaml"
    if not cfg_path.exists():
        return None
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_path)
    root = None
    try:
        root = cfg.DATASET.OPENFACE_ROOT
    except (AttributeError, KeyError):
        pass
    if not root:
        return None
    p = Path(str(root))
    return p if p.is_absolute() else Path(args.run_dir).resolve() / p


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize OpenFace per-video quality (confidence/success/pose/gaze)."
    )
    parser.add_argument(
        "--openface-root",
        default=None,
        help="Root directory containing OpenFace CSV files. Overrides --run-dir resolution.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory whose resolved_config.yaml carries DATASET.OPENFACE_ROOT.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for the quality summary.")
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        help=f"Frame is low-confidence when its confidence < this (default {DEFAULT_LOW_CONFIDENCE_THRESHOLD}).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    openface_root = _resolve_openface_root(args)
    if openface_root is None or not Path(openface_root).exists():
        # Graceful degradation: emit an empty summary (header-only) so downstream
        # weaklabel joins still find the file and skip openface fields cleanly.
        out = Path(args.output_dir)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        write_openface_quality_summary([], out / "tables" / "openface_quality_summary.csv")
        print(
            "[OPENFACE_QUALITY] openface root not configured/missing; wrote empty summary: "
            f"{out / 'tables' / 'openface_quality_summary.csv'}"
        )
        return

    summaries = summarize_openface_root(
        openface_root,
        low_confidence_threshold=args.low_confidence_threshold,
    )
    out = Path(args.output_dir)
    quality_path = write_openface_quality_summary(
        summaries, out / "tables" / "openface_quality_summary.csv"
    )
    print("[OPENFACE_QUALITY] generated files:")
    print(f"  - {quality_path}")
    print(f"[OPENFACE_QUALITY] output directory: {out}")


if __name__ == "__main__":
    main()
