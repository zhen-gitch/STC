import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.au_coordinate_contract import (
    MAPPING_METHODS,
    run_coordinate_contract_audit,
    write_run_manifest,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the read-only AU-T0b detection-to-aligned coordinate contract audit. "
            "The command never uses independent x/y scaling."
        )
    )
    parser.add_argument("--image-root", required=True, help="Root containing aligned frame directories.")
    parser.add_argument("--openface-root", required=True, help="Detection-space per-video OpenFace CSV root.")
    parser.add_argument("--frame-contract-summary", required=True, help="AU-T0a frame_contract_summary.csv.")
    parser.add_argument("--selected-frame-mapping", required=True, help="AU-T0a selected_frame_mapping.csv.")
    parser.add_argument(
        "--dataset-split-file",
        required=True,
        help="Frozen train/val/test dataset_split.json.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for AU-T0b artifacts.")
    parser.add_argument(
        "--mapping-method",
        choices=sorted(MAPPING_METHODS),
        default="auto",
        help="auto priority: explicit affine, aligned re-detection, canonical similarity.",
    )
    parser.add_argument(
        "--transform-root",
        default=None,
        help="Optional per-video CSV root with frame,m00,m01,m02,m10,m11,m12.",
    )
    parser.add_argument(
        "--aligned-openface-root",
        default=None,
        help="Optional OpenFace CSV root produced by re-detecting landmarks on aligned JPGs.",
    )
    parser.add_argument(
        "--canonical-template",
        default=None,
        help="Optional aligned-space CSV with landmark_id,x,y used for similarity fitting.",
    )
    parser.add_argument("--min-mapping-valid-ratio", type=float, default=0.995)
    parser.add_argument("--min-in-bounds-ratio", type=float, default=0.80)
    parser.add_argument("--max-overlays", type=int, default=120, help="Maximum deterministic train-only overlays.")
    parser.add_argument("--max-videos", type=int, default=None, help="Optional deterministic debug limit.")
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    generated = run_coordinate_contract_audit(
        image_root=Path(args.image_root),
        openface_root=Path(args.openface_root),
        frame_contract_summary=Path(args.frame_contract_summary),
        selected_frame_mapping=Path(args.selected_frame_mapping),
        dataset_split_file=Path(args.dataset_split_file),
        output_dir=output_dir,
        mapping_method=args.mapping_method,
        transform_root=Path(args.transform_root) if args.transform_root else None,
        aligned_openface_root=Path(args.aligned_openface_root) if args.aligned_openface_root else None,
        canonical_template_path=Path(args.canonical_template) if args.canonical_template else None,
        min_mapping_valid_ratio=args.min_mapping_valid_ratio,
        min_in_bounds_ratio=args.min_in_bounds_ratio,
        max_overlays=args.max_overlays,
        max_videos=args.max_videos,
    )
    run_manifest = write_run_manifest(
        output_dir / "run_manifest.json",
        PROJECT_ROOT,
        vars(args),
        {
            "frame_contract_summary": Path(args.frame_contract_summary),
            "selected_frame_mapping": Path(args.selected_frame_mapping),
            "dataset_split_file": Path(args.dataset_split_file),
            "canonical_template": Path(args.canonical_template) if args.canonical_template else None,
        },
    )
    print("[AU-T0b] generated files:")
    for path in [*generated, run_manifest]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
