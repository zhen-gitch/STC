import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_openface_raw_au_fidelity_reference.ps1"
POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "behavior_alignment"
    / "privileged_behavior_au_fidelity_policy_v1.json"
)


def _script():
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_raw_reference_uses_frozen_openface_profile_and_input_mode():
    text = _script()
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    assert '$featureProfile = "quality_2d_pose_au_no_gaze_no_hog_v1"' in text
    assert '$featureArguments = @("-2Dfp", "-pose", "-aus")' in text
    assert '@("-f", $rawVideoPath, "-out_dir", $resolvedOutputRoot)' in text
    assert '"-fdir"' not in text
    assert '"-gaze"' not in text
    assert '"-hogalign"' not in text
    assert policy["source_contract"]["feature_arguments"] == ["-2Dfp", "-pose", "-aus"]


def test_script_reads_raw_video_path_but_never_historical_raw_openface_path():
    text = _script()

    assert "$row.raw_video_path" in text
    assert "$row.raw_openface_csv" not in text
    assert "raw_openface_csv_path_field_access_count = 0" in text
    assert "historical_raw_openface_feature_file_open_count = 0" in text
    assert 'selected_value_columns = $coreAuColumns' in text
    assert 'pose_value_access_count = 0' in text
    assert 'gaze_value_access_count = 0' in text
    assert 'extension_au_value_access_count = 0' in text


def test_full_run_is_clean_complete_fresh_and_non_resumable():
    text = _script()

    assert '$expectedVideoCount = 300' in text
    assert '$expectedFrameCount = 493141L' in text
    assert 'if ($MaxVideos -eq 0)' in text
    assert 'Full extraction requires a clean git checkout.' in text
    assert 'OutputRoot already exists; refusing overwrite or resume' in text
    assert 'OutputRoot cannot replace RawVideoRoot' in text
    assert 'OutputRoot must not overlap a source-video directory' in text
    assert 'OutputRoot must be outside the git repository' in text
    assert 'output_root_must_be_new = $true' in text
    assert 'run_scope = $(if ($MaxVideos -gt 0) { "debug_subset" } else { "full_dataset" })' in text
    assert 'debug_subset_must_not_be_treated_as_full_extraction = $MaxVideos -gt 0' in text
    assert "Resume" not in text


def test_script_and_policy_bind_same_immutable_sources():
    text = _script()
    source = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["source_contract"]

    bindings = {
        "ExpectedSourceVideoContractSha256": source[
            "required_source_video_contract_sha256"
        ],
        "ExpectedFeatureExtractionSha256": source[
            "required_feature_extraction_sha256"
        ],
        "ExpectedModelSha256": source["required_model_sha256"],
        "ExpectedReadmeSha256": source["required_openface_readme_sha256"],
        "ExpectedAuPredictorManifestSha256": source[
            "required_au_predictor_manifest_sha256"
        ],
    }
    for parameter, sha256 in bindings.items():
        assert f'[string]${parameter} = "{sha256}"' in text


def test_extractor_writes_every_manifest_consumed_by_p0c():
    text = _script()

    for filename in (
        "input_video_contract.csv",
        "binary_manifest.csv",
        "model_manifest.csv",
        "video_run_summary.csv",
        "csv_schema_manifest.csv",
        "csv_content_manifest.csv",
        "extraction_summary.json",
        "run_manifest.json",
    ):
        assert f'"{filename}"' in text
    assert 'csv_content_manifest_sha256' in text
    assert 'extraction_summary_sha256' in text
    assert 'status = "RUNNING"' in text
    assert '$provenance["status"] = $finalStatus' in text
    assert "Test-FileManifestUnchanged" in text
    assert "provenance_stable = $provenanceStable" in text
    assert '$provenance["binary_files_unchanged"] = $binaryFilesUnchanged' in text
    assert '$provenance["model_files_unchanged"] = $modelFilesUnchanged' in text


def test_binary_and_model_manifests_match_the_aligned_extractor_schema():
    text = _script()

    assert "relative_path = $file.Name\n        size = $file.Length" in text
    assert "relative_path = $relativePath\n            size = $file.Length" in text
    assert "size_bytes = $file.Length" not in text
