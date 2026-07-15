import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.image_integrity import (
    compare_image_manifests,
    run_image_inventory,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit aligned JPG decoding and byte/pixel identity across machines."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Create a deterministic image manifest.")
    inventory.add_argument("--image-root", required=True)
    inventory.add_argument("--output-dir", required=True)
    inventory.add_argument("--label", required=True, help="Machine/dataset label, e.g. server or local_wsl.")
    inventory.add_argument("--workers", type=int, default=None)
    inventory.add_argument("--expected-width", type=int, default=112)
    inventory.add_argument("--expected-height", type=int, default=112)
    inventory.add_argument("--max-videos", type=int, default=None, help="Deterministic debug limit.")
    inventory.add_argument("--progress-every", type=int, default=10000)
    inventory.add_argument(
        "--pixel-hash",
        action="store_true",
        help="Also hash decoded RGB pixels; slower but identifies harmless JPEG re-encoding.",
    )
    inventory.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip Pillow verification. Not recommended for the final integrity gate.",
    )

    compare = subparsers.add_parser("compare", help="Compare two sorted image manifests.")
    compare.add_argument("--reference-manifest", required=True, help="Usually the server manifest.")
    compare.add_argument("--candidate-manifest", required=True, help="Usually the local manifest.")
    compare.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "inventory":
        generated = run_image_inventory(
            image_root=Path(args.image_root),
            output_dir=Path(args.output_dir),
            label=args.label,
            workers=args.workers,
            pixel_hash=args.pixel_hash,
            decode_check=not args.skip_decode,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
            max_videos=args.max_videos,
            progress_every=args.progress_every,
            project_root=PROJECT_ROOT,
        )
        prefix = "IMAGE_INTEGRITY_INVENTORY"
    else:
        generated = compare_image_manifests(
            reference_manifest=Path(args.reference_manifest),
            candidate_manifest=Path(args.candidate_manifest),
            output_dir=Path(args.output_dir),
        )
        prefix = "IMAGE_INTEGRITY_COMPARE"
    print(f"[{prefix}] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
