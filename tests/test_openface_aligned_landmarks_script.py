from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_openface_aligned_landmarks.ps1"


def _script_text():
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_openface_script_requires_explicit_provenance_for_non_git_copies():
    text = _script_text()

    assert '[string]$SourceGitCommit = ""' in text
    assert '[string]$SourceGitBranch = ""' in text
    assert "SourceGitCommit and SourceGitBranch must be provided together." in text
    assert "Git provenance is unavailable." in text
    assert "git_provenance_mode = $gitProvenanceMode" in text
    assert "git_commit = $resolvedGitCommit" in text
    assert "git_branch = $resolvedGitBranch" in text


def test_openface_script_keeps_the_frozen_2d_only_contract():
    text = _script_text()

    assert "a29ba49cfc59039bfe5e2f141898b2a110da420f6f520d6a923a86ac78cd96ae" in text
    assert "7efbef33dbc3e54197960300827657f9fe7a42c0953ef52c2af054a6fdbc3598" in text
    assert "4ccdd65f992124db8127688a545a9344b537bdeb2d97371cfddfd65fc68a1d93" in text
    assert "-fdir $videoDir.FullName" in text
    assert "-2Dfp" in text
    assert "-mloc $modelPath" in text
    for forbidden_flag in ("-aus", "-gaze", "-hogalign", "-simalign", "-tracked"):
        assert forbidden_flag not in text


def test_openface_script_uses_the_frozen_release_package_layout():
    text = _script_text()

    assert '[string]$OpenFaceRoot = "D:\\Tools\\Openface_2.2.0_win_x64"' in text
    assert '$expectedPackageDirectory = "OpenFace_2.2.0_win_x64"' in text
    assert '$releaseRoot = Join-Path $OpenFaceRoot $expectedPackageDirectory' in text
    assert '$featureExtraction = Join-Path $releaseRoot "FeatureExtraction.exe"' in text
    assert '$modelPath = Join-Path $releaseRoot "model\\main_ceclm_general.txt"' in text
    assert '$readmePath = Join-Path $releaseRoot "readme.txt"' in text
    assert "OpenFace readme SHA-256 mismatch" in text
    assert "Set-Location -LiteralPath $releaseRoot" in text
    assert "openface_working_directory = $releaseRoot" in text
