import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_au_coordinate_contract",
    PROJECT_ROOT / "scripts" / "audit_au_coordinate_contract.py",
)
audit_au_coordinate_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_au_coordinate_contract)


def test_parser_defaults_keep_t0b_read_only_and_review_gated():
    args = audit_au_coordinate_contract.build_parser().parse_args(
        [
            "--image-root",
            "/images",
            "--openface-root",
            "/openface",
            "--frame-contract-summary",
            "/audit/tables/frame_contract_summary.csv",
            "--selected-frame-mapping",
            "/audit/tables/selected_frame_mapping.csv",
            "--dataset-split-file",
            "/data/dataset_split.json",
            "--output-dir",
            "/audit/t0b",
        ]
    )

    assert args.mapping_method == "auto"
    assert args.min_mapping_valid_ratio == 0.995
    assert args.min_in_bounds_ratio == 0.80
    assert args.max_overlays == 120
    assert args.transform_root is None
    assert args.aligned_openface_root is None
    assert args.canonical_template is None
