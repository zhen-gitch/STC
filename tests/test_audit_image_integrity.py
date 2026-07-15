import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_image_integrity",
    PROJECT_ROOT / "scripts" / "audit_image_integrity.py",
)
audit_image_integrity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_image_integrity)


def test_inventory_parser_defaults_enable_decode_and_disable_pixel_hash():
    args = audit_image_integrity.build_parser().parse_args(
        [
            "inventory",
            "--image-root",
            "/images",
            "--output-dir",
            "/audit/server",
            "--label",
            "server",
        ]
    )

    assert args.command == "inventory"
    assert args.pixel_hash is False
    assert args.skip_decode is False
    assert args.workers is None
    assert args.expected_width == 112
    assert args.expected_height == 112
    assert args.max_videos is None
    assert args.progress_every == 10000


def test_compare_parser_requires_both_manifests():
    args = audit_image_integrity.build_parser().parse_args(
        [
            "compare",
            "--reference-manifest",
            "/audit/server/image_manifest.csv",
            "--candidate-manifest",
            "/audit/local/image_manifest.csv",
            "--output-dir",
            "/audit/comparison",
        ]
    )

    assert args.command == "compare"
    assert args.reference_manifest.endswith("image_manifest.csv")
    assert args.candidate_manifest.endswith("image_manifest.csv")
