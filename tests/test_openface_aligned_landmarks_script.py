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

    assert "5995ae5cce749c4969ac4dd7e62d3f740cc9f702961f9573be7e14c4ca5b7f86" in text
    assert "52f38548cffab1731f80e9e71f22a8b29373a2750eb6dc718069d56e82997543" in text
    assert "-fdir $videoDir.FullName" in text
    assert "-2Dfp" in text
    assert "-mloc $modelPath" in text
    for forbidden_flag in ("-aus", "-gaze", "-hogalign", "-simalign", "-tracked"):
        assert forbidden_flag not in text
