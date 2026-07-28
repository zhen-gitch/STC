[CmdletBinding()]
param(
    [string]$OpenFaceRoot = "D:\Tools\Openface_2.2.0_win_x64",
    [Parameter(Mandatory = $true)]
    [string]$ImageRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [Parameter(Mandatory = $true)]
    [string]$IntegrityComparisonSummary,
    [ValidateRange(0.0, 1.0)]
    [double]$MinSuccessRatio = 0.995,
    [string]$ExpectedFeatureExtractionSha256 = "a29ba49cfc59039bfe5e2f141898b2a110da420f6f520d6a923a86ac78cd96ae",
    [string]$ExpectedModelSha256 = "7efbef33dbc3e54197960300827657f9fe7a42c0953ef52c2af054a6fdbc3598",
    [string]$ExpectedReadmeSha256 = "4ccdd65f992124db8127688a545a9344b537bdeb2d97371cfddfd65fc68a1d93",
    [string]$SourceVideoContract = "",
    [string]$ExpectedSourceVideoContractSha256 = "12f50c2311b9d89dde81e27fa83891226ca5f447ed7d3d56fe38cf673a1c5a31",
    [string]$SourceGitCommit = "",
    [string]$SourceGitBranch = "",
    [ValidateRange(0, 2147483647)]
    [int]$MaxVideos = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# WSL can launch Windows PowerShell with PATHEXT reduced to ".CPL", which
# makes the call operator treat FeatureExtraction.exe as a document. Restore
# the standard executable extensions only when .EXE is absent.
$pathExtensions = @(([string]$env:PATHEXT).Split(";") | ForEach-Object { $_.ToUpperInvariant() })
if ($pathExtensions -notcontains ".EXE") {
    $env:PATHEXT = ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.CPL"
}

$featureProfile = "quality_2d_pose_au_no_gaze_no_hog_v1"
$downstreamBehaviorColumns = @("AU12_r", "AU14_r", "AU15_r", "pose_Rx", "pose_Ry", "pose_Rz")

function Get-NormalizedPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    # Resolve-Path can return a provider-qualified string for WSL UNC paths on
    # Windows PowerShell 5 (for example, Microsoft.PowerShell.Core\FileSystem::...).
    # Convert through the provider API before passing the path to System.IO.
    $providerPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    return [System.IO.Path]::GetFullPath($providerPath).TrimEnd("\")
}

function Test-PathInside {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Root
    )
    $rootPrefix = $Root.TrimEnd("\") + "\"
    return $Candidate.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        return ([System.BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

function Get-GitOutput {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string[]]$GitArguments
    )
    $exitCode = $null
    try {
        $lines = @(& git -C $RepositoryRoot @GitArguments 2>$null)
        $exitCode = [int]$LASTEXITCODE
        return [pscustomobject]@{
            Output = ($lines -join "`n").TrimEnd()
            ExitCode = $exitCode
            Available = $exitCode -eq 0
        }
    }
    catch {
        return [pscustomobject]@{
            Output = ""
            ExitCode = $exitCode
            Available = $false
        }
    }
}

function Get-SourceFrameId {
    param([Parameter(Mandatory = $true)][string]$BaseName)
    $matches = [regex]::Matches($BaseName, "\d+")
    if ($matches.Count -eq 0) {
        return $null
    }
    $frameId = 0L
    if (-not [long]::TryParse($matches[$matches.Count - 1].Value, [ref]$frameId)) {
        return $null
    }
    return $frameId
}

function Get-RequiredCsvColumns {
    $columns = [System.Collections.Generic.List[string]]::new()
    foreach ($column in @("frame", "face_id", "timestamp", "confidence", "success")) {
        $columns.Add($column)
    }
    foreach ($column in @("pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz")) {
        $columns.Add($column)
    }
    foreach ($prefix in @("x", "y")) {
        for ($index = 0; $index -lt 68; $index += 1) {
            $columns.Add("$prefix`_$index")
        }
    }
    foreach ($au in @("01", "02", "04", "05", "06", "07", "09", "10", "12", "14", "15", "17", "20", "23", "25", "26", "45")) {
        $columns.Add("AU$($au)_r")
    }
    foreach ($au in @("01", "02", "04", "05", "06", "07", "09", "10", "12", "14", "15", "17", "20", "23", "25", "26", "28", "45")) {
        $columns.Add("AU$($au)_c")
    }
    return $columns.ToArray()
}

function Get-CsvContract {
    param(
        [Parameter(Mandatory = $true)][string]$CsvPath,
        [Parameter(Mandatory = $true)][int]$ExpectedRows,
        [Parameter(Mandatory = $true)][double]$RequiredSuccessRatio,
        [Parameter(Mandatory = $true)][string[]]$RequiredColumns,
        [Parameter(Mandatory = $true)][string[]]$RequiredBehaviorColumns
    )
    $issues = [System.Collections.Generic.List[string]]::new()
    if (-not (Test-Path -LiteralPath $CsvPath -PathType Leaf)) {
        $issues.Add("missing_csv")
        return [pscustomobject]@{
            CsvRows = 0
            FirstFrame = $null
            LastFrame = $null
            SuccessCount = 0
            SuccessRatio = 0.0
            MeanConfidence = $null
            MinConfidence = $null
            FrameSequenceIssueCount = 0
            NonZeroTimestampCount = 0
            SchemaSha256 = ""
            SchemaColumnCount = 0
            Status = "FAIL"
            Issues = ($issues -join ";")
        }
    }

    $csvRows = 0
    $firstFrame = $null
    $lastFrame = $null
    $successCount = 0
    $confidenceCount = 0
    $confidenceSum = 0.0
    $minConfidence = $null
    $frameSequenceIssueCount = 0
    $frameParseIssueCount = 0
    $successParseIssueCount = 0
    $successDomainIssueCount = 0
    $confidenceParseIssueCount = 0
    $confidenceDomainIssueCount = 0
    $timestampParseIssueCount = 0
    $rowColumnCountIssueCount = 0
    $nonZeroTimestampCount = 0
    $schemaSha256 = ""
    $schemaColumnCount = 0
    $behaviorValueIssueCounts = [ordered]@{}
    $behaviorRangeIssueCounts = [ordered]@{}
    foreach ($column in $RequiredBehaviorColumns) {
        $behaviorValueIssueCounts[$column] = 0
        $behaviorRangeIssueCounts[$column] = 0
    }
    $behaviorValueRanges = @{
        "AU12_r" = @(0.0, 5.0)
        "AU14_r" = @(0.0, 5.0)
        "AU15_r" = @(0.0, 5.0)
        "pose_Rx" = @(-[Math]::PI, [Math]::PI)
        "pose_Ry" = @(-[Math]::PI, [Math]::PI)
        "pose_Rz" = @(-[Math]::PI, [Math]::PI)
    }

    $reader = [System.IO.File]::OpenText($CsvPath)
    try {
        $headerLine = $reader.ReadLine()
        if ([string]::IsNullOrWhiteSpace($headerLine)) {
            $issues.Add("missing_header")
            $columns = @()
        }
        else {
            $columns = @($headerLine.Split([char]",") | ForEach-Object { $_.Trim() })
            $schemaColumnCount = $columns.Count
            $schemaSha256 = Get-TextSha256 -Text ($columns -join ",")
            $seenColumns = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
            foreach ($column in $columns) {
                if (-not $seenColumns.Add($column)) {
                    $issues.Add("duplicate_column:$column")
                }
            }
            foreach ($column in $RequiredColumns) {
                if (-not $seenColumns.Contains($column)) {
                    $issues.Add("missing_column:$column")
                }
            }
            foreach ($column in $columns) {
                if ($RequiredColumns -notcontains $column) {
                    $issues.Add("unexpected_column:$column")
                }
            }
        }

        $frameIndex = [Array]::IndexOf($columns, "frame")
        $timestampIndex = [Array]::IndexOf($columns, "timestamp")
        $confidenceIndex = [Array]::IndexOf($columns, "confidence")
        $successIndex = [Array]::IndexOf($columns, "success")
        $leadingColumnsAvailable = $frameIndex -ge 0 -and $timestampIndex -ge 0 -and $confidenceIndex -ge 0 -and $successIndex -ge 0
        $behaviorColumnIndexes = [ordered]@{}
        foreach ($column in $RequiredBehaviorColumns) {
            $behaviorColumnIndexes[$column] = [Array]::IndexOf($columns, $column)
        }

        while ($null -ne ($line = $reader.ReadLine())) {
            $csvRows += 1
            $values = $line.Split([char]",")
            if ($values.Count -ne $columns.Count) {
                $rowColumnCountIssueCount += 1
                continue
            }
            if (-not $leadingColumnsAvailable) {
                $frameParseIssueCount += 1
                continue
            }

            $frameValue = 0
            if ([int]::TryParse($values[$frameIndex].Trim(), [ref]$frameValue)) {
                if ($null -eq $firstFrame) {
                    $firstFrame = $frameValue
                }
                $lastFrame = $frameValue
                if ($frameValue -ne $csvRows) {
                    $frameSequenceIssueCount += 1
                }
            }
            else {
                $frameParseIssueCount += 1
            }

            $timestampValue = 0.0
            if ([double]::TryParse(
                $values[$timestampIndex].Trim(),
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$timestampValue
            )) {
                if ($timestampValue -ne 0.0) {
                    $nonZeroTimestampCount += 1
                }
            }
            else {
                $timestampParseIssueCount += 1
            }

            $successValue = 0.0
            if ([double]::TryParse(
                $values[$successIndex].Trim(),
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$successValue
            )) {
                if ([double]::IsNaN($successValue) -or [double]::IsInfinity($successValue) -or
                    ($successValue -ne 0.0 -and $successValue -ne 1.0)) {
                    $successDomainIssueCount += 1
                }
                elseif ($successValue -eq 1.0) {
                    $successCount += 1
                }
            }
            else {
                $successParseIssueCount += 1
            }

            $confidenceValue = 0.0
            if ([double]::TryParse(
                $values[$confidenceIndex].Trim(),
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$confidenceValue
            )) {
                if ([double]::IsNaN($confidenceValue) -or [double]::IsInfinity($confidenceValue) -or
                    $confidenceValue -lt 0.0 -or $confidenceValue -gt 1.0) {
                    $confidenceDomainIssueCount += 1
                }
                else {
                    $confidenceCount += 1
                    $confidenceSum += $confidenceValue
                    if ($null -eq $minConfidence -or $confidenceValue -lt $minConfidence) {
                        $minConfidence = $confidenceValue
                    }
                }
            }
            else {
                $confidenceParseIssueCount += 1
            }

            foreach ($column in $RequiredBehaviorColumns) {
                $columnIndex = [int]$behaviorColumnIndexes[$column]
                if ($columnIndex -lt 0) {
                    continue
                }
                $behaviorValue = 0.0
                if (-not [double]::TryParse(
                    $values[$columnIndex].Trim(),
                    [System.Globalization.NumberStyles]::Float,
                    [System.Globalization.CultureInfo]::InvariantCulture,
                    [ref]$behaviorValue
                ) -or [double]::IsNaN($behaviorValue) -or [double]::IsInfinity($behaviorValue)) {
                    $behaviorValueIssueCounts[$column] = [int]$behaviorValueIssueCounts[$column] + 1
                    continue
                }
                if ($behaviorValueRanges.ContainsKey($column)) {
                    $range = $behaviorValueRanges[$column]
                    if ($behaviorValue -lt [double]$range[0] -or $behaviorValue -gt [double]$range[1]) {
                        $behaviorRangeIssueCounts[$column] = [int]$behaviorRangeIssueCounts[$column] + 1
                    }
                }
            }
        }
    }
    finally {
        $reader.Dispose()
    }

    if ($csvRows -eq 0) { $issues.Add("empty_csv") }
    if ($csvRows -ne $ExpectedRows) { $issues.Add("row_count:$csvRows!=$ExpectedRows") }
    if ($firstFrame -ne 1) { $issues.Add("first_frame:$firstFrame") }
    if ($lastFrame -ne $ExpectedRows) { $issues.Add("last_frame:$lastFrame!=$ExpectedRows") }
    if ($frameSequenceIssueCount -gt 0) { $issues.Add("non_sequential_frames:$frameSequenceIssueCount") }
    if ($rowColumnCountIssueCount -gt 0) { $issues.Add("row_column_count_errors:$rowColumnCountIssueCount") }
    if ($frameParseIssueCount -gt 0) { $issues.Add("frame_parse_errors:$frameParseIssueCount") }
    if ($successParseIssueCount -gt 0) { $issues.Add("success_parse_errors:$successParseIssueCount") }
    if ($successDomainIssueCount -gt 0) { $issues.Add("invalid_success_values:$successDomainIssueCount") }
    if ($confidenceParseIssueCount -gt 0) { $issues.Add("confidence_parse_errors:$confidenceParseIssueCount") }
    if ($confidenceDomainIssueCount -gt 0) { $issues.Add("invalid_confidence_values:$confidenceDomainIssueCount") }
    if ($timestampParseIssueCount -gt 0) { $issues.Add("timestamp_parse_errors:$timestampParseIssueCount") }
    if ($nonZeroTimestampCount -gt 0) { $issues.Add("unexpected_nonzero_timestamps:$nonZeroTimestampCount") }
    foreach ($column in $RequiredBehaviorColumns) {
        $issueCount = [int]$behaviorValueIssueCounts[$column]
        if ($issueCount -gt 0) {
            $issues.Add("invalid_behavior_values:$column`:$issueCount")
        }
        $rangeIssueCount = [int]$behaviorRangeIssueCounts[$column]
        if ($rangeIssueCount -gt 0) {
            $issues.Add("out_of_range_behavior_values:$column`:$rangeIssueCount")
        }
    }

    $successRatio = $(if ($csvRows -gt 0) { $successCount / [double]$csvRows } else { 0.0 })
    if ($successRatio -lt $RequiredSuccessRatio) {
        $issues.Add("success_ratio:$successRatio<$RequiredSuccessRatio")
    }
    $meanConfidence = $(if ($confidenceCount -gt 0) { $confidenceSum / $confidenceCount } else { $null })

    return [pscustomobject]@{
        CsvRows = $csvRows
        FirstFrame = $firstFrame
        LastFrame = $lastFrame
        SuccessCount = $successCount
        SuccessRatio = $successRatio
        MeanConfidence = $meanConfidence
        MinConfidence = $minConfidence
        FrameSequenceIssueCount = $frameSequenceIssueCount
        NonZeroTimestampCount = $nonZeroTimestampCount
        SchemaSha256 = $schemaSha256
        SchemaColumnCount = $schemaColumnCount
        Status = $(if ($issues.Count -eq 0) { "PASS" } else { "FAIL" })
        Issues = ($issues -join ";")
    }
}

$requiredCsvColumns = Get-RequiredCsvColumns
$expectedPackageDirectory = "OpenFace_2.2.0_win_x64"
$releaseRoot = Join-Path $OpenFaceRoot $expectedPackageDirectory
$featureExtraction = Join-Path $releaseRoot "FeatureExtraction.exe"
$modelPath = Join-Path $releaseRoot "model\main_ceclm_general.txt"
$readmePath = Join-Path $releaseRoot "readme.txt"
$auPredictorManifestPath = Join-Path $releaseRoot "AU_predictors\AU_all_best.txt"
$requiredModelRelativePaths = @(
    "model\patch_experts\cen_patches_0.25_of.dat",
    "model\patch_experts\cen_patches_0.35_of.dat",
    "model\patch_experts\cen_patches_0.50_of.dat",
    "model\patch_experts\cen_patches_1.00_of.dat"
)

foreach ($requiredPath in @($featureExtraction, $modelPath, $readmePath, $auPredictorManifestPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required OpenFace file does not exist: $requiredPath"
    }
    if ((Get-Item -LiteralPath $requiredPath).Length -le 0) {
        throw "Required OpenFace file is empty: $requiredPath"
    }
}
foreach ($requiredPath in @($ImageRoot, $IntegrityComparisonSummary)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required input path does not exist: $requiredPath"
    }
}
if (-not (Test-Path -LiteralPath $ImageRoot -PathType Container)) {
    throw "ImageRoot is not a directory: $ImageRoot"
}
if (-not (Test-Path -LiteralPath $IntegrityComparisonSummary -PathType Leaf)) {
    throw "IntegrityComparisonSummary is not a file: $IntegrityComparisonSummary"
}

foreach ($relativePath in $requiredModelRelativePaths) {
    $requiredModelPath = Join-Path $releaseRoot $relativePath
    if (-not (Test-Path -LiteralPath $requiredModelPath -PathType Leaf) -or
        (Get-Item -LiteralPath $requiredModelPath).Length -le 0) {
        throw "OpenFace CEN model dependency is missing or empty: $relativePath. Run download_models.ps1 from $releaseRoot."
    }
}

$auPredictorPaths = [System.Collections.Generic.List[string]]::new()
foreach ($line in @(Get-Content -LiteralPath $auPredictorManifestPath)) {
    $trimmed = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) {
        continue
    }
    $relativePredictorPath = ($trimmed -split "\s+")[0]
    $predictorPath = Join-Path (Split-Path -Parent $auPredictorManifestPath) $relativePredictorPath
    if (-not (Test-Path -LiteralPath $predictorPath -PathType Leaf) -or
        (Get-Item -LiteralPath $predictorPath).Length -le 0) {
        throw "OpenFace AU predictor listed by AU_all_best.txt is missing or empty: $relativePredictorPath"
    }
    $auPredictorPaths.Add($predictorPath)
}
if ($auPredictorPaths.Count -eq 0) {
    throw "OpenFace AU_all_best.txt contains no predictor entries."
}

$integrity = Get-Content -LiteralPath $IntegrityComparisonSummary -Raw | ConvertFrom-Json
$requiredIntegrityFields = @(
    "audit", "status", "reference_count", "candidate_count", "status_counts",
    "reference_manifest_sha256", "candidate_manifest_sha256"
)
foreach ($field in $requiredIntegrityFields) {
    if ($integrity.PSObject.Properties.Name -notcontains $field) {
        throw "Image integrity summary is missing required field: $field"
    }
}
if ($integrity.audit -ne "aligned image cross-machine comparison" -or $integrity.status -ne "EXACT_PASS") {
    throw "Image integrity gate is not EXACT_PASS: $($integrity.status)"
}
$integrityReferenceCount = [long]$integrity.reference_count
$integrityCandidateCount = [long]$integrity.candidate_count
if ($integrityReferenceCount -le 0 -or $integrityReferenceCount -ne $integrityCandidateCount) {
    throw "Image integrity reference/candidate counts are empty or unequal."
}
$exactMatchProperty = $integrity.status_counts.PSObject.Properties["EXACT_MATCH"]
if ($null -eq $exactMatchProperty -or [long]$exactMatchProperty.Value -ne $integrityCandidateCount) {
    throw "Image integrity status_counts does not prove exact equality for every candidate frame."
}
foreach ($field in @("reference_manifest_sha256", "candidate_manifest_sha256")) {
    $value = [string]$integrity.PSObject.Properties[$field].Value
    if ($value -notmatch "^[0-9a-fA-F]{64}$") {
        throw "Image integrity manifest hash is invalid: $field"
    }
}

$releaseDirectoryName = Split-Path -Leaf $releaseRoot
if ($releaseDirectoryName -ne $expectedPackageDirectory) {
    throw "Frozen OpenFace package directory mismatch: $releaseDirectoryName"
}
$featureExtractionSha256 = (Get-FileHash -LiteralPath $featureExtraction -Algorithm SHA256).Hash.ToLowerInvariant()
$modelSha256 = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
$readmeSha256 = (Get-FileHash -LiteralPath $readmePath -Algorithm SHA256).Hash.ToLowerInvariant()
$expectedFeatureExtractionSha256Normalized = $(
    if ($ExpectedFeatureExtractionSha256) { $ExpectedFeatureExtractionSha256.ToLowerInvariant() } else { "" }
)
$expectedModelSha256Normalized = $(
    if ($ExpectedModelSha256) { $ExpectedModelSha256.ToLowerInvariant() } else { "" }
)
$expectedReadmeSha256Normalized = $(
    if ($ExpectedReadmeSha256) { $ExpectedReadmeSha256.ToLowerInvariant() } else { "" }
)
if ($expectedFeatureExtractionSha256Normalized -and $featureExtractionSha256 -ne $expectedFeatureExtractionSha256Normalized) {
    throw "FeatureExtraction.exe SHA-256 mismatch: $featureExtractionSha256"
}
if ($expectedModelSha256Normalized -and $modelSha256 -ne $expectedModelSha256Normalized) {
    throw "OpenFace model SHA-256 mismatch: $modelSha256"
}
if ($expectedReadmeSha256Normalized -and $readmeSha256 -ne $expectedReadmeSha256Normalized) {
    throw "OpenFace readme SHA-256 mismatch: $readmeSha256"
}

$scriptPath = $MyInvocation.MyCommand.Path
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$gitCommitCommand = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("rev-parse", "HEAD")
$gitBranchCommand = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("branch", "--show-current")
$gitStatusCommand = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("status", "--short")
$detectedGitCommit = $(if ($gitCommitCommand.Available) { [string]$gitCommitCommand.Output } else { "" })
$detectedGitBranch = $(if ($gitBranchCommand.Available) { [string]$gitBranchCommand.Output } else { "" })
$detectedGitStatus = [string]$gitStatusCommand.Output
$gitStatusInterpretation = $(
    if (-not $gitStatusCommand.Available) { "unavailable" }
    elseif ([string]::IsNullOrWhiteSpace($detectedGitStatus)) { "clean" }
    else { "dirty" }
)
if ($MaxVideos -eq 0) {
    if (-not $gitStatusCommand.Available) {
        throw "Full-dataset extraction requires git status --short to succeed; status is unavailable."
    }
    if (-not [string]::IsNullOrWhiteSpace($detectedGitStatus)) {
        throw "Full-dataset extraction requires a clean git checkout; git status --short is dirty."
    }
}
$hasExplicitCommit = -not [string]::IsNullOrWhiteSpace($SourceGitCommit)
$hasExplicitBranch = -not [string]::IsNullOrWhiteSpace($SourceGitBranch)
if ($hasExplicitCommit -xor $hasExplicitBranch) {
    throw "SourceGitCommit and SourceGitBranch must be provided together."
}
if ($hasExplicitCommit) {
    $resolvedGitCommit = $SourceGitCommit.Trim().ToLowerInvariant()
    $resolvedGitBranch = $SourceGitBranch.Trim()
    if ($resolvedGitCommit -notmatch "^[0-9a-f]{7,40}$") {
        throw "SourceGitCommit must be a 7-40 character hexadecimal git commit."
    }
    if ($detectedGitCommit -and -not $detectedGitCommit.ToLowerInvariant().StartsWith($resolvedGitCommit)) {
        throw "Explicit SourceGitCommit does not match the detected checkout."
    }
    if ($detectedGitBranch -and $detectedGitBranch -ne $resolvedGitBranch) {
        throw "Explicit SourceGitBranch does not match the detected checkout."
    }
    $gitProvenanceMode = "explicit"
}
else {
    $resolvedGitCommit = $(if ($detectedGitCommit) { $detectedGitCommit.ToLowerInvariant() } else { "" })
    $resolvedGitBranch = $detectedGitBranch
    $gitProvenanceMode = "detected_checkout"
}
if ([string]::IsNullOrWhiteSpace($resolvedGitCommit) -or [string]::IsNullOrWhiteSpace($resolvedGitBranch)) {
    throw "Git provenance is unavailable. Pass both -SourceGitCommit and -SourceGitBranch when running from a non-git code copy."
}

$sourceVideoContractCandidate = $(
    if ([string]::IsNullOrWhiteSpace($SourceVideoContract)) {
        Join-Path $repositoryRoot "logs\au_region_tracking_audit\source_video_presence\tables\source_video_contract.csv"
    }
    else {
        $SourceVideoContract
    }
)
if (-not (Test-Path -LiteralPath $sourceVideoContractCandidate -PathType Leaf)) {
    throw "Source-video frame-time contract is missing: $sourceVideoContractCandidate"
}
$resolvedSourceVideoContract = Get-NormalizedPath -Path $sourceVideoContractCandidate
$sourceVideoContractSha256 = (Get-FileHash -LiteralPath $resolvedSourceVideoContract -Algorithm SHA256).Hash.ToLowerInvariant()
$expectedSourceVideoContractSha256Normalized = $(
    if ($ExpectedSourceVideoContractSha256) { $ExpectedSourceVideoContractSha256.ToLowerInvariant() } else { "" }
)
if ($expectedSourceVideoContractSha256Normalized -and
    $sourceVideoContractSha256 -ne $expectedSourceVideoContractSha256Normalized) {
    throw "Source-video frame-time contract SHA-256 mismatch: $sourceVideoContractSha256"
}

$resolvedImageRoot = Get-NormalizedPath -Path $ImageRoot
$resolvedOpenFaceRoot = Get-NormalizedPath -Path $OpenFaceRoot
$resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot).TrimEnd("\")
if (Test-Path -LiteralPath $resolvedOutputRoot) {
    throw "OutputRoot already exists; refusing to overwrite or resume into an existing directory: $resolvedOutputRoot"
}
foreach ($protectedRoot in @($resolvedImageRoot, $resolvedOpenFaceRoot)) {
    if ([System.StringComparer]::OrdinalIgnoreCase.Equals($resolvedOutputRoot, $protectedRoot) -or
        (Test-PathInside -Candidate $resolvedOutputRoot -Root $protectedRoot)) {
        throw "OutputRoot must be outside the aligned-image and OpenFace package trees: $resolvedOutputRoot"
    }
}

$allVideoDirs = @(
    Get-ChildItem -LiteralPath $resolvedImageRoot -Directory |
        Where-Object { $_.Name -like "*_video_aligned" } |
        Sort-Object Name
)
if ($allVideoDirs.Count -eq 0) {
    throw "No *_video_aligned directories found under ImageRoot."
}
$unexpectedImageDirs = @(
    Get-ChildItem -LiteralPath $resolvedImageRoot -Directory |
        Where-Object { $_.Name -notlike "*_video_aligned" } |
        Where-Object {
            @(Get-ChildItem -LiteralPath $_.FullName -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg") }).Count -gt 0
        }
)
if ($unexpectedImageDirs.Count -gt 0) {
    throw "Aligned ImageRoot contains JPG files outside *_video_aligned directories: $($unexpectedImageDirs[0].FullName)"
}

$sourceVideoRows = @(Import-Csv -LiteralPath $resolvedSourceVideoContract)
if ($sourceVideoRows.Count -eq 0) {
    throw "Source-video frame-time contract is empty."
}
$requiredSourceVideoColumns = @(
    "video_id", "aligned_frame_count", "raw_frame_count", "frame_count_match", "raw_fps", "contract_status"
)
foreach ($column in $requiredSourceVideoColumns) {
    if ($sourceVideoRows[0].PSObject.Properties.Name -notcontains $column) {
        throw "Source-video frame-time contract is missing required column: $column"
    }
}
$sourceVideoRowsById = @{}
foreach ($row in $sourceVideoRows) {
    if ([string]::IsNullOrWhiteSpace([string]$row.video_id) -or $sourceVideoRowsById.ContainsKey($row.video_id)) {
        throw "Source-video frame-time contract contains an empty or duplicate video_id: $($row.video_id)"
    }
    $rawFps = 0.0
    if (-not [double]::TryParse(
        ([string]$row.raw_fps).Trim(),
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$rawFps
    )) {
        throw "Source-video frame-time contract has an invalid raw_fps: $($row.video_id)"
    }
    if ($row.contract_status -ne "PASS" -or [int]$row.frame_count_match -ne 1 -or
        [long]$row.aligned_frame_count -ne [long]$row.raw_frame_count -or [Math]::Abs($rawFps - 30.0) -gt 0.000001) {
        throw "Source-video frame-time contract is not PASS/30 FPS/frame-count-matched: $($row.video_id)"
    }
    $sourceVideoRowsById[$row.video_id] = $row
}
if ($sourceVideoRowsById.Count -ne $allVideoDirs.Count) {
    throw "Source-video frame-time contract video count differs from aligned ImageRoot."
}
$videoDirs = @($allVideoDirs)
if ($MaxVideos -gt 0) {
    $videoDirs = @($videoDirs | Select-Object -First $MaxVideos)
}
$selectedVideoIds = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
foreach ($videoDir in $videoDirs) {
    [void]$selectedVideoIds.Add($videoDir.Name)
}

$inputFrameContractRows = [System.Collections.Generic.List[object]]::new()
$totalInputImages = 0L
foreach ($videoDir in $allVideoDirs) {
    if (-not $sourceVideoRowsById.ContainsKey($videoDir.Name)) {
        throw "Source-video frame-time contract is missing aligned video: $($videoDir.Name)"
    }
    $imageFiles = @(
        Get-ChildItem -LiteralPath $videoDir.FullName -File |
            Where-Object { $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg") } |
            Sort-Object Name
    )
    $issues = [System.Collections.Generic.List[string]]::new()
    $sequenceLines = [System.Collections.Generic.List[string]]::new()
    $firstSourceFrame = $null
    $lastSourceFrame = $null
    $unparsedFrameCount = 0
    $nonSequentialFrameCount = 0
    $seenSourceFrames = [System.Collections.Generic.HashSet[long]]::new()
    $ordinal = 0
    foreach ($imageFile in $imageFiles) {
        $ordinal += 1
        $sourceFrameId = Get-SourceFrameId -BaseName $imageFile.BaseName
        if ($null -eq $sourceFrameId) {
            $unparsedFrameCount += 1
            $sequenceLines.Add("$ordinal`t`t$($imageFile.Name)`t$($imageFile.Length)")
            continue
        }
        if ($null -eq $firstSourceFrame) {
            $firstSourceFrame = $sourceFrameId
        }
        $lastSourceFrame = $sourceFrameId
        if (-not $seenSourceFrames.Add([long]$sourceFrameId)) {
            $issues.Add("duplicate_source_frame:$sourceFrameId")
        }
        if ([long]$sourceFrameId -ne [long]$ordinal) {
            $nonSequentialFrameCount += 1
        }
        $sequenceLines.Add("$ordinal`t$sourceFrameId`t$($imageFile.Name)`t$($imageFile.Length)")
    }
    if ($imageFiles.Count -eq 0) { $issues.Add("no_images") }
    if ($unparsedFrameCount -gt 0) { $issues.Add("unparsed_source_frames:$unparsedFrameCount") }
    if ($nonSequentialFrameCount -gt 0) { $issues.Add("non_sequential_source_frames:$nonSequentialFrameCount") }
    if ($firstSourceFrame -ne 1) { $issues.Add("first_source_frame:$firstSourceFrame") }
    if ($lastSourceFrame -ne $imageFiles.Count) { $issues.Add("last_source_frame:$lastSourceFrame!=$($imageFiles.Count)") }
    $sourceVideoRow = $sourceVideoRowsById[$videoDir.Name]
    if ([long]$sourceVideoRow.aligned_frame_count -ne [long]$imageFiles.Count) {
        $issues.Add("source_contract_frame_count:$($sourceVideoRow.aligned_frame_count)!=$($imageFiles.Count)")
    }

    $totalInputImages += [long]$imageFiles.Count
    $inputFrameContractRows.Add([pscustomobject]@{
        video_id = $videoDir.Name
        image_dir = $videoDir.FullName
        image_count = $imageFiles.Count
        first_file = $(if ($imageFiles.Count -gt 0) { $imageFiles[0].Name } else { "" })
        last_file = $(if ($imageFiles.Count -gt 0) { $imageFiles[$imageFiles.Count - 1].Name } else { "" })
        first_source_frame = $firstSourceFrame
        last_source_frame = $lastSourceFrame
        unparsed_source_frame_count = $unparsedFrameCount
        non_sequential_source_frame_count = $nonSequentialFrameCount
        source_video_fps = $sourceVideoRow.raw_fps
        source_video_frame_count = $sourceVideoRow.raw_frame_count
        frame_sequence_sha256 = Get-TextSha256 -Text ($sequenceLines.ToArray() -join "`n")
        selected_for_run = $selectedVideoIds.Contains($videoDir.Name)
        status = $(if ($issues.Count -eq 0) { "PASS" } else { "FAIL" })
        issues = ($issues -join ";")
    })
}
if ($totalInputImages -ne $integrityCandidateCount) {
    throw "Aligned JPG count differs from the EXACT_PASS candidate count: $totalInputImages != $integrityCandidateCount"
}
$failedInputContracts = @($inputFrameContractRows | Where-Object { $_.status -ne "PASS" })
if ($failedInputContracts.Count -gt 0) {
    $examples = @($failedInputContracts | Select-Object -First 5 | ForEach-Object { "$($_.video_id):$($_.issues)" })
    throw "Aligned JPG frame contract failed: $($examples -join ', ')"
}

New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null
$auditRoot = Join-Path $resolvedOutputRoot "_audit"
$logRoot = Join-Path $auditRoot "logs"
New-Item -ItemType Directory -Path $auditRoot | Out-Null
New-Item -ItemType Directory -Path $logRoot | Out-Null

$inputFrameContractPath = Join-Path $auditRoot "input_frame_contract.csv"
$inputFrameContractRows | Export-Csv -LiteralPath $inputFrameContractPath -NoTypeInformation -Encoding UTF8
$inputFrameContractSha256 = (Get-FileHash -LiteralPath $inputFrameContractPath -Algorithm SHA256).Hash.ToLowerInvariant()

$binaryManifest = [System.Collections.Generic.List[object]]::new()
foreach ($file in @(Get-ChildItem -LiteralPath $releaseRoot -File | Where-Object { $_.Extension -in @(".exe", ".dll") } | Sort-Object Name)) {
    $binaryManifest.Add([pscustomobject]@{
        relative_path = $file.Name
        size = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}
if ($binaryManifest.Count -eq 0) {
    throw "OpenFace binary manifest is empty."
}
$binaryManifestPath = Join-Path $auditRoot "binary_manifest.csv"
$binaryManifest | Export-Csv -LiteralPath $binaryManifestPath -NoTypeInformation -Encoding UTF8
$binaryManifestSha256 = (Get-FileHash -LiteralPath $binaryManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()

$modelManifest = [System.Collections.Generic.List[object]]::new()
foreach ($modelDirectoryName in @("model", "AU_predictors")) {
    $modelDirectory = Join-Path $releaseRoot $modelDirectoryName
    if (-not (Test-Path -LiteralPath $modelDirectory -PathType Container)) {
        throw "OpenFace model directory is missing: $modelDirectory"
    }
    foreach ($file in @(Get-ChildItem -LiteralPath $modelDirectory -File -Recurse | Sort-Object FullName)) {
        $relativePath = $file.FullName.Substring($releaseRoot.Length)
        if ($relativePath.StartsWith([System.IO.Path]::DirectorySeparatorChar.ToString())) {
            $relativePath = $relativePath.Substring(1)
        }
        $modelManifest.Add([pscustomobject]@{
            relative_path = $relativePath
            size = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        })
    }
}
if ($modelManifest.Count -eq 0) {
    throw "OpenFace model manifest is empty."
}
$modelManifestPath = Join-Path $auditRoot "model_manifest.csv"
$modelManifest | Export-Csv -LiteralPath $modelManifestPath -NoTypeInformation -Encoding UTF8
$modelManifestSha256 = (Get-FileHash -LiteralPath $modelManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()

$requiredModelManifest = [System.Collections.Generic.List[object]]::new()
foreach ($relativePath in $requiredModelRelativePaths) {
    $requiredModelPath = Join-Path $releaseRoot $relativePath
    $requiredModelManifest.Add([pscustomobject]@{
        relative_path = $relativePath
        size = (Get-Item -LiteralPath $requiredModelPath).Length
        sha256 = (Get-FileHash -LiteralPath $requiredModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}
$requiredAuPredictorManifest = [System.Collections.Generic.List[object]]::new()
foreach ($predictorPath in $auPredictorPaths) {
    $file = Get-Item -LiteralPath $predictorPath
    $relativePath = $file.FullName.Substring($releaseRoot.Length)
    if ($relativePath.StartsWith([System.IO.Path]::DirectorySeparatorChar.ToString())) {
        $relativePath = $relativePath.Substring(1)
    }
    $requiredAuPredictorManifest.Add([pscustomobject]@{
        relative_path = $relativePath
        size = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}

$openFaceFeatureArguments = @("-2Dfp", "-pose", "-aus")
$provenance = [ordered]@{
    audit = "OpenFace original aligned-image behavior feature extraction"
    created_utc = [DateTime]::UtcNow.ToString("o")
    feature_profile = $featureProfile
    intended_use = "PB-P0 aligned-JPG AU/head source contract and fidelity audit; not training authorization"
    input_space = "original aligned JPG used by the RGB pipeline"
    behavior_source_columns = $downstreamBehaviorColumns
    computer_name = $env:COMPUTERNAME
    powershell_version = $PSVersionTable.PSVersion.ToString()
    script_path = $scriptPath
    script_sha256 = (Get-FileHash -LiteralPath $scriptPath -Algorithm SHA256).Hash.ToLowerInvariant()
    git_commit = $resolvedGitCommit
    git_branch = $resolvedGitBranch
    git_commit_command_available = [bool]$gitCommitCommand.Available
    git_commit_command_exit_code = $gitCommitCommand.ExitCode
    git_branch_command_available = [bool]$gitBranchCommand.Available
    git_branch_command_exit_code = $gitBranchCommand.ExitCode
    git_status_command_available = [bool]$gitStatusCommand.Available
    git_status_command_exit_code = $gitStatusCommand.ExitCode
    git_status_short = $detectedGitStatus
    git_status_available = [bool]$gitStatusCommand.Available
    git_status_interpretation = $gitStatusInterpretation
    git_provenance_mode = $gitProvenanceMode
    git_repository_root = $repositoryRoot
    invocation_line = $MyInvocation.Line
    process_command_line = [Environment]::CommandLine
    process_pathext = $env:PATHEXT
    openface_root = $OpenFaceRoot
    openface_package_root = $releaseRoot
    openface_working_directory = $releaseRoot
    openface_package_directory = $releaseDirectoryName
    expected_openface_package_directory = $expectedPackageDirectory
    feature_extraction = $featureExtraction
    feature_extraction_sha256 = $featureExtractionSha256
    expected_feature_extraction_sha256 = $expectedFeatureExtractionSha256Normalized
    model_path = $modelPath
    model_sha256 = $modelSha256
    expected_model_sha256 = $expectedModelSha256Normalized
    openface_readme = $readmePath
    openface_readme_sha256 = $readmeSha256
    expected_openface_readme_sha256 = $expectedReadmeSha256Normalized
    required_cen_model_files = $requiredModelManifest
    au_predictor_manifest = $auPredictorManifestPath
    au_predictor_manifest_sha256 = (Get-FileHash -LiteralPath $auPredictorManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    required_au_predictor_files = $requiredAuPredictorManifest
    binary_manifest = $binaryManifestPath
    binary_manifest_sha256 = $binaryManifestSha256
    model_manifest = $modelManifestPath
    model_manifest_sha256 = $modelManifestSha256
    image_root = $resolvedImageRoot
    output_root = $resolvedOutputRoot
    integrity_comparison_summary = (Get-NormalizedPath -Path $IntegrityComparisonSummary)
    integrity_comparison_summary_sha256 = (Get-FileHash -LiteralPath $IntegrityComparisonSummary -Algorithm SHA256).Hash.ToLowerInvariant()
    integrity_status = $integrity.status
    integrity_reference_count = $integrityReferenceCount
    integrity_candidate_count = $integrityCandidateCount
    integrity_reference_manifest_sha256 = ([string]$integrity.reference_manifest_sha256).ToLowerInvariant()
    integrity_candidate_manifest_sha256 = ([string]$integrity.candidate_manifest_sha256).ToLowerInvariant()
    source_video_contract = $resolvedSourceVideoContract
    source_video_contract_sha256 = $sourceVideoContractSha256
    expected_source_video_contract_sha256 = $expectedSourceVideoContractSha256Normalized
    source_video_contract_status = "$($sourceVideoRowsById.Count)/$($allVideoDirs.Count) PASS; aligned/raw frame counts matched; raw_fps=30.000000"
    input_video_count = $allVideoDirs.Count
    input_frame_count = $totalInputImages
    selected_video_count = $videoDirs.Count
    run_scope = $(if ($MaxVideos -gt 0) { "debug_subset" } else { "full_dataset" })
    debug_subset_must_not_be_treated_as_full_extraction = $MaxVideos -gt 0
    input_frame_contract = $inputFrameContractPath
    input_frame_contract_sha256 = $inputFrameContractSha256
    mapping_contract = "sorted complete aligned JPG sequence; source frame id parsed from final filename digit group must equal OpenFace row ordinal"
    feature_arguments = $openFaceFeatureArguments
    arguments = @("-fdir", "<video_dir>", "-out_dir", $resolvedOutputRoot) + $openFaceFeatureArguments + @("-mloc", $modelPath)
    required_csv_columns = $requiredCsvColumns
    downstream_behavior_columns = $downstreamBehaviorColumns
    quality_fields_authorized_for_mask_and_coverage_only = @("confidence", "success")
    csv_timestamp_semantics = "constant_zero_for_image_directory"
    csv_timestamp_authorized_for_head_velocity = $false
    head_velocity_timebase_contract = "derive d_pose/dt only from an external source-video FPS/frame-time contract; never from the OpenFace image-directory CSV timestamp"
    source_frame_time_formula = "t=(source_frame_id-1)/30"
    pose_delta_contract = "wrapped shortest signed angular difference between adjacent valid rows whose source_frame_id is consecutive"
    pose_translation_output_retained = $true
    pose_translation_authorized_for_behavior_targets = $false
    gaze_requested = $false
    gaze_training_access_count = 0
    gaze_normalizer_access_count = 0
    gaze_loss_access_count = 0
    hog_enabled = $false
    tracked_video_enabled = $false
    aligned_image_generation_enabled = $false
    min_success_ratio = $MinSuccessRatio
    max_videos = $MaxVideos
    output_root_must_be_new = $true
}
$runManifestPath = Join-Path $auditRoot "run_manifest.json"
$provenance | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $runManifestPath -Encoding UTF8

$runRows = [System.Collections.Generic.List[object]]::new()
$overallStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$index = 0
foreach ($videoDir in $videoDirs) {
    $index += 1
    $videoId = $videoDir.Name
    $expectedCsv = Join-Path $resolvedOutputRoot "$videoId.csv"
    $logPath = Join-Path $logRoot "$videoId.log"
    $inputContract = $inputFrameContractRows | Where-Object { $_.video_id -eq $videoId } | Select-Object -First 1
    if ($null -eq $inputContract -or $inputContract.status -ne "PASS") {
        throw "Selected video has no passing input frame contract: $videoId"
    }
    $imageCount = [int]$inputContract.image_count

    Write-Host "[$index/$($videoDirs.Count)] $videoId images=$imageCount"
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = 0
    $previousErrorActionPreference = $ErrorActionPreference
    $previousLocation = Get-Location
    try {
        # OpenFace model manifests contain relative paths; resolve them from the frozen release package.
        Set-Location -LiteralPath $releaseRoot
        $ErrorActionPreference = "Continue"
        $arguments = @("-fdir", $videoDir.FullName, "-out_dir", $resolvedOutputRoot) + $openFaceFeatureArguments + @("-mloc", $modelPath)
        & $featureExtraction @arguments 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        Set-Location -LiteralPath $previousLocation
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $stopwatch.Stop()
    $contract = Get-CsvContract -CsvPath $expectedCsv -ExpectedRows $imageCount -RequiredSuccessRatio $MinSuccessRatio -RequiredColumns $requiredCsvColumns -RequiredBehaviorColumns $downstreamBehaviorColumns
    $issues = [System.Collections.Generic.List[string]]::new()
    if ($exitCode -ne 0) { $issues.Add("exit_code:$exitCode") }
    if ($contract.Issues) { $issues.Add($contract.Issues) }
    $csvSizeBytes = 0L
    $csvSha256 = ""
    if (Test-Path -LiteralPath $expectedCsv -PathType Leaf) {
        $csvItem = Get-Item -LiteralPath $expectedCsv
        $csvSizeBytes = [long]$csvItem.Length
        $csvSha256 = (Get-FileHash -LiteralPath $expectedCsv -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    if ($contract.Status -eq "PASS" -and
        ($csvSizeBytes -le 0 -or [string]::IsNullOrWhiteSpace($csvSha256))) {
        $issues.Add("missing_csv_content_provenance")
    }
    $status = $(if ($issues.Count -eq 0) { "PASS" } else { "FAIL" })
    $runRows.Add([pscustomobject]@{
        video_id = $videoId
        image_dir = $videoDir.FullName
        image_count = $imageCount
        input_frame_sequence_sha256 = $inputContract.frame_sequence_sha256
        csv_path = $expectedCsv
        csv_rows = $contract.CsvRows
        csv_size_bytes = $csvSizeBytes
        csv_sha256 = $csvSha256
        first_frame = $contract.FirstFrame
        last_frame = $contract.LastFrame
        success_count = $contract.SuccessCount
        success_ratio = [Math]::Round($contract.SuccessRatio, 6)
        mean_confidence = $(if ($null -ne $contract.MeanConfidence) { [Math]::Round($contract.MeanConfidence, 6) } else { $null })
        min_confidence = $(if ($null -ne $contract.MinConfidence) { [Math]::Round($contract.MinConfidence, 6) } else { $null })
        frame_sequence_issue_count = $contract.FrameSequenceIssueCount
        nonzero_timestamp_count = $contract.NonZeroTimestampCount
        schema_sha256 = $contract.SchemaSha256
        schema_column_count = $contract.SchemaColumnCount
        exit_code = $exitCode
        elapsed_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
        status = $status
        issues = ($issues -join ";")
        log_path = $logPath
    })
    if ($status -ne "PASS") {
        Write-Warning "$videoId failed: $($issues -join ';')"
    }
}
$overallStopwatch.Stop()

$summaryPath = Join-Path $auditRoot "video_run_summary.csv"
$runRows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
$schemaRows = @(
    $runRows |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_.schema_sha256) } |
        Group-Object schema_sha256, schema_column_count |
        ForEach-Object {
            [pscustomobject]@{
                schema_sha256 = $_.Group[0].schema_sha256
                column_count = $_.Group[0].schema_column_count
                video_count = $_.Count
                example_csv = $_.Group[0].csv_path
            }
        }
)
$schemaManifestPath = Join-Path $auditRoot "csv_schema_manifest.csv"
$schemaRows | Export-Csv -LiteralPath $schemaManifestPath -NoTypeInformation -Encoding UTF8

$csvContentRows = @(
    $runRows |
        Sort-Object -Property video_id |
        ForEach-Object {
            [pscustomobject]@{
                video_id = $_.video_id
                csv_rows = $_.csv_rows
                schema_sha256 = $_.schema_sha256
                csv_size_bytes = $_.csv_size_bytes
                csv_sha256 = $_.csv_sha256
                status = $_.status
            }
        }
)
$csvContentManifestPath = Join-Path $auditRoot "csv_content_manifest.csv"
$csvContentRows | Export-Csv -LiteralPath $csvContentManifestPath -NoTypeInformation -Encoding UTF8
$csvContentManifestSha256 = (Get-FileHash -LiteralPath $csvContentManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()

$passCount = @($runRows | Where-Object { $_.status -eq "PASS" }).Count
$failCount = @($runRows | Where-Object { $_.status -ne "PASS" }).Count
$totalSelectedImages = ($runRows | Measure-Object -Property image_count -Sum).Sum
$totalRows = ($runRows | Measure-Object -Property csv_rows -Sum).Sum
$totalSuccesses = ($runRows | Measure-Object -Property success_count -Sum).Sum
$overallSuccessRatio = $(if ($totalRows -gt 0) { $totalSuccesses / [double]$totalRows } else { 0.0 })
$hogFiles = @(Get-ChildItem -LiteralPath $resolvedOutputRoot -File -Recurse -Filter "*.hog" -ErrorAction SilentlyContinue)
$trackedVideoFiles = @(
    Get-ChildItem -LiteralPath $resolvedOutputRoot -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension.ToLowerInvariant() -in @(".avi", ".mp4", ".mov", ".mkv") }
)
$regeneratedAlignedImages = @(
    Get-ChildItem -LiteralPath $resolvedOutputRoot -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -notlike "$auditRoot*" -and
            $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg", ".png", ".bmp")
        }
)
$schemaConsistent = $schemaRows.Count -eq 1
$finalStatus = $(
    if ($failCount -eq 0 -and $totalSelectedImages -eq $totalRows -and $schemaConsistent -and
        $hogFiles.Count -eq 0 -and $trackedVideoFiles.Count -eq 0 -and $regeneratedAlignedImages.Count -eq 0) {
        "PASS"
    }
    else {
        "FAIL"
    }
)
$finalSummary = [ordered]@{
    audit = "OpenFace original aligned-image behavior feature extraction summary"
    created_utc = [DateTime]::UtcNow.ToString("o")
    feature_profile = $featureProfile
    run_scope = $(if ($MaxVideos -gt 0) { "debug_subset" } else { "full_dataset" })
    video_count = $runRows.Count
    pass_count = $passCount
    fail_count = $failCount
    total_images = $totalSelectedImages
    total_csv_rows = $totalRows
    total_successes = $totalSuccesses
    success_ratio = [Math]::Round($overallSuccessRatio, 6)
    min_required_success_ratio = $MinSuccessRatio
    schema_count = $schemaRows.Count
    schema_consistent = $schemaConsistent
    hog_file_count = $hogFiles.Count
    tracked_video_file_count = $trackedVideoFiles.Count
    regenerated_aligned_image_count = $regeneratedAlignedImages.Count
    elapsed_seconds = [Math]::Round($overallStopwatch.Elapsed.TotalSeconds, 3)
    status = $finalStatus
    run_manifest = $runManifestPath
    input_frame_contract = $inputFrameContractPath
    video_run_summary = $summaryPath
    csv_schema_manifest = $schemaManifestPath
    csv_content_manifest = $csvContentManifestPath
    csv_content_manifest_sha256 = $csvContentManifestSha256
}
$extractionSummaryPath = Join-Path $auditRoot "extraction_summary.json"
$finalSummary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $extractionSummaryPath -Encoding UTF8
$extractionSummarySha256 = (Get-FileHash -LiteralPath $extractionSummaryPath -Algorithm SHA256).Hash.ToLowerInvariant()

# Finalize output provenance after the summary is immutable. The summary records
# only the run-manifest path, never its hash, so this rewrite does not create a
# run_manifest <-> extraction_summary hash cycle.
$provenance["csv_content_manifest"] = $csvContentManifestPath
$provenance["csv_content_manifest_sha256"] = $csvContentManifestSha256
$provenance["extraction_summary"] = $extractionSummaryPath
$provenance["extraction_summary_sha256"] = $extractionSummarySha256
$provenance | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $runManifestPath -Encoding UTF8

Write-Host "Aligned behavior OpenFace extraction complete: PASS=$passCount FAIL=$failCount images=$totalSelectedImages csv_rows=$totalRows success_ratio=$([Math]::Round($overallSuccessRatio, 6)) schemas=$($schemaRows.Count)"
if ($finalStatus -ne "PASS") {
    exit 1
}
