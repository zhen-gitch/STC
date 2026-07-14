import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_au_region_tracking",
    PROJECT_ROOT / "scripts" / "audit_au_region_tracking.py",
)
audit_au_region_tracking = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_au_region_tracking)


def test_parser_defaults_match_regression_only_protocol():
    args = audit_au_region_tracking.build_parser().parse_args(
        [
            "--image-root",
            "/images",
            "--openface-root",
            "/openface",
            "--output-dir",
            "/output",
        ]
    )

    assert args.sample_step == 10
    assert args.max_seq_len == 2000
    assert args.sampling_strategy == "stride_head"
    assert args.join_threshold == 0.995


def test_parser_accepts_frame_regex_and_debug_limit():
    args = audit_au_region_tracking.build_parser().parse_args(
        [
            "--image-root",
            "/images",
            "--openface-root",
            "/openface",
            "--output-dir",
            "/output",
            "--frame-id-regex",
            r"frame_(\d+)",
            "--max-videos",
            "3",
        ]
    )

    assert args.frame_id_regex == r"frame_(\d+)"
    assert args.max_videos == 3
