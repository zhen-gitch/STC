import csv
import math
from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_openface_aligned_behavior_features.ps1"


def _script_text():
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_behavior_feature_script_freezes_the_requested_no_gaze_profile():
    text = _script_text()

    assert '$openFaceFeatureArguments = @("-2Dfp", "-pose", "-aus")' in text
    assert 'feature_profile = $featureProfile' in text
    assert 'gaze_requested = $false' in text
    assert 'hog_enabled = $false' in text
    assert 'tracked_video_enabled = $false' in text
    assert 'aligned_image_generation_enabled = $false' in text
    assert 'pose_translation_output_retained = $true' in text
    assert 'pose_translation_authorized_for_behavior_targets = $false' in text
    assert 'input_space = "original aligned JPG used by the RGB pipeline"' in text
    assert 'behavior_source_columns = $downstreamBehaviorColumns' in text
    assert 'gaze_training_access_count = 0' in text
    assert 'gaze_normalizer_access_count = 0' in text
    assert 'gaze_loss_access_count = 0' in text
    assert 'csv_timestamp_semantics = "constant_zero_for_image_directory"' in text
    assert 'csv_timestamp_authorized_for_head_velocity = $false' in text
    assert "derive d_pose/dt only from an external source-video FPS/frame-time contract" in text
    assert 'source_frame_time_formula = "t=(source_frame_id-1)/30"' in text
    assert "wrapped shortest signed angular difference between adjacent valid rows" in text
    assert "unwrap rotation and difference" not in text
    for column in ("AU12_r", "AU14_r", "AU15_r", "pose_Rx", "pose_Ry", "pose_Rz"):
        assert column in text
    for forbidden_flag in (
        "-gaze",
        "-3Dfp",
        "-pdmparams",
        "-hogalign",
        "-simalign",
        "-tracked",
        "-multi_view",
    ):
        assert forbidden_flag not in text


def test_behavior_feature_script_freezes_openface_220_and_all_required_models():
    text = _script_text()

    assert '[string]$OpenFaceRoot = "D:\\Tools\\Openface_2.2.0_win_x64"' in text
    assert '$expectedPackageDirectory = "OpenFace_2.2.0_win_x64"' in text
    assert '$releaseRoot = Join-Path $OpenFaceRoot $expectedPackageDirectory' in text
    assert '$featureExtraction = Join-Path $releaseRoot "FeatureExtraction.exe"' in text
    assert '$modelPath = Join-Path $releaseRoot "model\\main_ceclm_general.txt"' in text
    assert '$auPredictorManifestPath = Join-Path $releaseRoot "AU_predictors\\AU_all_best.txt"' in text
    assert "a29ba49cfc59039bfe5e2f141898b2a110da420f6f520d6a923a86ac78cd96ae" in text
    assert "7efbef33dbc3e54197960300827657f9fe7a42c0953ef52c2af054a6fdbc3598" in text
    assert "4ccdd65f992124db8127688a545a9344b537bdeb2d97371cfddfd65fc68a1d93" in text
    assert "12f50c2311b9d89dde81e27fa83891226ca5f447ed7d3d56fe38cf673a1c5a31" in text
    assert "OpenFace AU predictor listed by AU_all_best.txt is missing or empty" in text
    assert "OpenFace CEN model dependency is missing or empty" in text
    assert 'binary_manifest_sha256 = $binaryManifestSha256' in text
    assert 'model_manifest_sha256 = $modelManifestSha256' in text
    assert 'au_predictor_manifest_sha256 = ' in text


def test_behavior_feature_script_requires_complete_aligned_input_and_a_new_output_root():
    text = _script_text()

    assert "[string]$IntegrityComparisonSummary" in text
    assert '$integrity.status -ne "EXACT_PASS"' in text
    assert 'status_counts.PSObject.Properties["EXACT_MATCH"]' in text
    assert "Aligned JPG count differs from the EXACT_PASS candidate count" in text
    assert "Aligned ImageRoot contains JPG files outside *_video_aligned directories" in text
    assert 'Where-Object { $_.Name -like "*_video_aligned" }' in text
    assert 'non_sequential_source_frames' in text
    assert 'frame_sequence_sha256 = Get-TextSha256' in text
    assert 'input_frame_contract.csv' in text
    assert "Source-video frame-time contract is not PASS/30 FPS/frame-count-matched" in text
    assert 'source_video_contract_sha256 = $sourceVideoContractSha256' in text
    assert "OutputRoot already exists; refusing to overwrite or resume" in text
    assert "OutputRoot must be outside the aligned-image and OpenFace package trees" in text
    assert "[switch]$Resume" not in text
    assert 'run_scope = $(if ($MaxVideos -gt 0) { "debug_subset" } else { "full_dataset" })' in text


def test_behavior_feature_script_enforces_exact_frame_and_schema_contracts():
    text = _script_text()

    for column in (
        '"frame", "face_id", "timestamp", "confidence", "success"',
        '"pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"',
        'columns.Add("$prefix`_$index")',
        'columns.Add("AU$($au)_r")',
        'columns.Add("AU$($au)_c")',
    ):
        assert column in text
    assert 'duplicate_column:$column' in text
    assert 'missing_column:$column' in text
    assert 'unexpected_column:$column' in text
    assert 'non_sequential_frames:$frameSequenceIssueCount' in text
    assert 'unexpected_nonzero_timestamps:$nonZeroTimestampCount' in text
    assert '$values = $line.Split([char]",")' in text
    assert '$values.Count -ne $columns.Count' in text
    assert 'row_column_count_errors:$rowColumnCountIssueCount' in text
    assert 'invalid_success_values:$successDomainIssueCount' in text
    assert 'invalid_confidence_values:$confidenceDomainIssueCount' in text
    assert 'invalid_behavior_values:$column`:$issueCount' in text
    assert 'out_of_range_behavior_values:$column`:$rangeIssueCount' in text
    assert '"AU12_r" = @(0.0, 5.0)' in text
    assert '"pose_Rx" = @(-[Math]::PI, [Math]::PI)' in text
    assert '-RequiredBehaviorColumns $downstreamBehaviorColumns' in text
    assert 'schema_sha256 = $contract.SchemaSha256' in text
    assert 'schema_consistent = $schemaConsistent' in text
    assert 'hog_file_count = $hogFiles.Count' in text
    assert 'tracked_video_file_count = $trackedVideoFiles.Count' in text
    assert 'regenerated_aligned_image_count = $regeneratedAlignedImages.Count' in text


def test_behavior_feature_script_records_stable_csv_content_provenance():
    text = _script_text()

    assert '$csvSizeBytes = [long]$csvItem.Length' in text
    assert '$csvSha256 = (Get-FileHash -LiteralPath $expectedCsv -Algorithm SHA256).Hash.ToLowerInvariant()' in text
    assert 'missing_csv_content_provenance' in text
    assert 'Sort-Object -Property video_id' in text
    assert '$csvContentManifestPath = Join-Path $auditRoot "csv_content_manifest.csv"' in text
    for field in (
        'video_id = $_.video_id',
        'csv_rows = $_.csv_rows',
        'schema_sha256 = $_.schema_sha256',
        'csv_size_bytes = $_.csv_size_bytes',
        'csv_sha256 = $_.csv_sha256',
        'status = $_.status',
    ):
        assert field in text
    assert 'csv_content_manifest = $csvContentManifestPath' in text
    assert 'csv_content_manifest_sha256 = $csvContentManifestSha256' in text
    assert '$provenance["extraction_summary"] = $extractionSummaryPath' in text
    assert '$provenance["extraction_summary_sha256"] = $extractionSummarySha256' in text
    assert 'run_manifest_sha256' not in text


def _function_block(text, start_marker, end_marker):
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end].rstrip()


def _required_csv_columns():
    columns = ["frame", "face_id", "timestamp", "confidence", "success"]
    columns.extend(["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"])
    for prefix in ("x", "y"):
        columns.extend(f"{prefix}_{index}" for index in range(68))
    columns.extend(
        f"AU{au}_r"
        for au in ("01", "02", "04", "05", "06", "07", "09", "10", "12", "14", "15", "17", "20", "23", "25", "26", "45")
    )
    columns.extend(
        f"AU{au}_c"
        for au in ("01", "02", "04", "05", "06", "07", "09", "10", "12", "14", "15", "17", "20", "23", "25", "26", "28", "45")
    )
    return columns


def _write_contract_csv(path, columns, row):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerow(row)


def _windows_path(path):
    result = subprocess.run(
        ["wslpath", "-w", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_behavior_feature_csv_contract_with_real_windows_powershell5(tmp_path):
    if shutil.which("powershell.exe") is None or shutil.which("wslpath") is None:
        pytest.skip("Windows PowerShell through WSL is unavailable")

    text = _script_text()
    function_text = "\n\n".join(
        (
            _function_block(text, "function Get-TextSha256 {", "function Get-GitOutput {"),
            _function_block(text, "function Get-RequiredCsvColumns {", "function Get-CsvContract {"),
            _function_block(text, "function Get-CsvContract {", "\n$requiredCsvColumns = Get-RequiredCsvColumns"),
        )
    )
    assert "FeatureExtraction" not in function_text

    harness_path = tmp_path / "csv_contract_harness.ps1"
    harness_path.write_text(
        "\n".join(
            (
                'Set-StrictMode -Version Latest',
                '$ErrorActionPreference = "Stop"',
                function_text,
                '$requiredCsvColumns = Get-RequiredCsvColumns',
                '$behaviorColumns = @("AU12_r", "AU14_r", "AU15_r", "pose_Rx", "pose_Ry", "pose_Rz")',
                'Write-Output ("PS_VERSION`t" + $PSVersionTable.PSVersion.ToString())',
                'foreach ($csvPath in $args) {',
                '    $contract = Get-CsvContract -CsvPath $csvPath -ExpectedRows 1 -RequiredSuccessRatio 0.0 -RequiredColumns $requiredCsvColumns -RequiredBehaviorColumns $behaviorColumns',
                '    Write-Output ([System.IO.Path]::GetFileName($csvPath) + "`t" + $contract.Status + "`t" + $contract.Issues)',
                '}',
                '',
            )
        ),
        encoding="utf-8",
    )

    columns = _required_csv_columns()
    base_row = ["0"] * len(columns)
    values = {
        "frame": "1",
        "face_id": "0",
        "timestamp": "0",
        "confidence": "0.98",
        "success": "1",
        "AU12_r": "1.2",
        "AU14_r": "0.4",
        "AU15_r": "0.1",
        "pose_Rx": "0.01",
        "pose_Ry": "-0.02",
        "pose_Rz": "0.03",
    }
    for column, value in values.items():
        base_row[columns.index(column)] = value

    cases = {
        "valid.csv": list(base_row),
        "truncated.csv": list(base_row[:-1]),
        "extra_column.csv": list(base_row) + ["0"],
        "success_nan.csv": list(base_row),
        "success_two.csv": list(base_row),
        "confidence_out_of_range.csv": list(base_row),
        "target_nonfinite.csv": list(base_row),
        "au_out_of_range.csv": list(base_row),
        "pose_out_of_range.csv": list(base_row),
    }
    cases["success_nan.csv"][columns.index("success")] = "NaN"
    cases["success_two.csv"][columns.index("success")] = "2"
    cases["confidence_out_of_range.csv"][columns.index("confidence")] = "1.5"
    cases["target_nonfinite.csv"][columns.index("AU12_r")] = "Infinity"
    cases["au_out_of_range.csv"][columns.index("AU12_r")] = "5.1"
    cases["pose_out_of_range.csv"][columns.index("pose_Rx")] = str(math.pi + 1e-6)

    csv_paths = []
    for name, row in cases.items():
        path = tmp_path / name
        _write_contract_csv(path, columns, row)
        csv_paths.append(path)

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            _windows_path(harness_path),
            *(_windows_path(path) for path in csv_paths),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line]
    assert lines[0].startswith("PS_VERSION\t5.")
    contracts = {}
    for line in lines[1:]:
        parts = line.split("\t", 2)
        name, status = parts[:2]
        issues = parts[2] if len(parts) == 3 else ""
        contracts[name] = (status, issues)

    assert contracts["valid.csv"] == ("PASS", "")
    assert contracts["truncated.csv"][0] == "FAIL"
    assert "row_column_count_errors:1" in contracts["truncated.csv"][1]
    assert contracts["extra_column.csv"][0] == "FAIL"
    assert "row_column_count_errors:1" in contracts["extra_column.csv"][1]
    assert contracts["success_nan.csv"][0] == "FAIL"
    assert any(
        marker in contracts["success_nan.csv"][1]
        for marker in ("success_parse_errors:1", "invalid_success_values:1")
    )
    assert contracts["success_two.csv"][0] == "FAIL"
    assert "invalid_success_values:1" in contracts["success_two.csv"][1]
    assert contracts["confidence_out_of_range.csv"][0] == "FAIL"
    assert "invalid_confidence_values:1" in contracts["confidence_out_of_range.csv"][1]
    assert contracts["target_nonfinite.csv"][0] == "FAIL"
    assert "invalid_behavior_values:AU12_r:1" in contracts["target_nonfinite.csv"][1]
    assert contracts["au_out_of_range.csv"][0] == "FAIL"
    assert "out_of_range_behavior_values:AU12_r:1" in contracts["au_out_of_range.csv"][1]
    assert contracts["pose_out_of_range.csv"][0] == "FAIL"
    assert "out_of_range_behavior_values:pose_Rx:1" in contracts["pose_out_of_range.csv"][1]


def test_get_git_output_distinguishes_clean_empty_from_command_failure(tmp_path):
    if shutil.which("powershell.exe") is None or shutil.which("wslpath") is None:
        pytest.skip("Windows PowerShell through WSL is unavailable")

    text = _script_text()
    function_text = _function_block(text, "function Get-GitOutput {", "function Get-SourceFrameId {")
    harness_path = tmp_path / "git_output_harness.ps1"
    harness_path.write_text(
        "\n".join(
            (
                'Set-StrictMode -Version Latest',
                '$ErrorActionPreference = "Stop"',
                'function git {',
                '    param(',
                '        [string]$C,',
                '        [Parameter(ValueFromRemainingArguments = $true)]',
                '        [string[]]$RemainingArguments',
                '    )',
                '    $mode = $RemainingArguments[$RemainingArguments.Count - 1]',
                '    if ($mode -eq "clean") {',
                '        $global:LASTEXITCODE = 0',
                '        return',
                '    }',
                '    if ($mode -eq "dirty") {',
                '        $global:LASTEXITCODE = 0',
                '        Write-Output " M tracked.txt"',
                '        return',
                '    }',
                '    if ($mode -eq "fail") {',
                '        $global:LASTEXITCODE = 17',
                '        return',
                '    }',
                '    throw "unexpected fake git mode: $mode"',
                '}',
                function_text,
                'foreach ($mode in @("clean", "dirty", "fail")) {',
                '    $result = Get-GitOutput -RepositoryRoot "C:\\repo" -GitArguments @($mode)',
                '    $outputMarker = $(if ($result.Output -eq "") { "<EMPTY>" } else { $result.Output })',
                '    Write-Output ($mode + "`t" + $result.Available + "`t" + $result.ExitCode + "`t" + $outputMarker)',
                '}',
                '',
            )
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            _windows_path(harness_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = {
        parts[0]: parts[1:]
        for parts in (line.split("\t") for line in result.stdout.splitlines() if line)
    }
    assert rows["clean"] == ["True", "0", "<EMPTY>"]
    assert rows["dirty"] == ["True", "0", " M tracked.txt"]
    assert rows["fail"] == ["False", "17", "<EMPTY>"]


def test_behavior_feature_script_records_git_command_and_input_provenance():
    text = _script_text()

    assert "GetUnresolvedProviderPathFromPSPath" in text
    assert '$pathExtensions -notcontains ".EXE"' in text
    assert 'process_pathext = $env:PATHEXT' in text
    assert '[string]$SourceGitCommit = ""' in text
    assert '[string]$SourceGitBranch = ""' in text
    assert "SourceGitCommit and SourceGitBranch must be provided together." in text
    assert "Git provenance is unavailable." in text
    assert '$gitCommitCommand = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("rev-parse", "HEAD")' in text
    assert '$gitBranchCommand = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("branch", "--show-current")' in text
    assert '$gitStatusCommand = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("status", "--short")' in text
    assert 'git_commit = $resolvedGitCommit' in text
    assert 'git_branch = $resolvedGitBranch' in text
    assert 'git_commit_command_available = [bool]$gitCommitCommand.Available' in text
    assert 'git_commit_command_exit_code = $gitCommitCommand.ExitCode' in text
    assert 'git_branch_command_available = [bool]$gitBranchCommand.Available' in text
    assert 'git_branch_command_exit_code = $gitBranchCommand.ExitCode' in text
    assert 'git_status_command_available = [bool]$gitStatusCommand.Available' in text
    assert 'git_status_command_exit_code = $gitStatusCommand.ExitCode' in text
    assert 'git_status_short = $detectedGitStatus' in text
    assert 'git_status_available = [bool]$gitStatusCommand.Available' in text
    assert 'git_status_interpretation = $gitStatusInterpretation' in text
    assert 'if (-not $gitStatusCommand.Available) { "unavailable" }' in text
    assert 'elseif ([string]::IsNullOrWhiteSpace($detectedGitStatus)) { "clean" }' in text
    assert 'else { "dirty" }' in text
    assert 'git_provenance_mode = $gitProvenanceMode' in text
    assert 'invocation_line = $MyInvocation.Line' in text
    assert 'process_command_line = [Environment]::CommandLine' in text
    assert 'integrity_candidate_manifest_sha256 = ' in text
    assert 'input_frame_contract_sha256 = $inputFrameContractSha256' in text
    assert 'Set-Location -LiteralPath $releaseRoot' in text

    full_dataset_gate = text.index('if ($MaxVideos -eq 0) {')
    openface_invocation = text.index('& $featureExtraction @arguments')
    assert full_dataset_gate < openface_invocation
    assert 'Full-dataset extraction requires git status --short to succeed; status is unavailable.' in text
    assert 'Full-dataset extraction requires a clean git checkout; git status --short is dirty.' in text
