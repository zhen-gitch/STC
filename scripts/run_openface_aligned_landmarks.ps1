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
    [string]$SourceGitCommit = "",
    [string]$SourceGitBranch = "",
    [int]$MaxVideos = 0,
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-CsvContract {
    param(
        [Parameter(Mandatory = $true)][string]$CsvPath,
        [Parameter(Mandatory = $true)][int]$ExpectedRows,
        [Parameter(Mandatory = $true)][double]$RequiredSuccessRatio
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
    $confidenceParseIssueCount = 0

    $reader = [System.IO.File]::OpenText($CsvPath)
    try {
        $headerLine = $reader.ReadLine()
        if ([string]::IsNullOrWhiteSpace($headerLine)) {
            $issues.Add("missing_header")
        }
        else {
            $columns = @($headerLine.Split([char]",") | ForEach-Object { $_.Trim() })
            $requiredColumns = @("frame", "confidence", "success", "x_0", "x_67", "y_0", "y_67")
            foreach ($column in $requiredColumns) {
                if ($columns -notcontains $column) {
                    $issues.Add("missing_column:$column")
                }
            }

            $frameIndex = [Array]::IndexOf($columns, "frame")
            $confidenceIndex = [Array]::IndexOf($columns, "confidence")
            $successIndex = [Array]::IndexOf($columns, "success")
            $leadingIndex = [Math]::Max($frameIndex, [Math]::Max($confidenceIndex, $successIndex))
            $splitCount = $leadingIndex + 2
            $leadingColumnsAvailable = $frameIndex -ge 0 -and $confidenceIndex -ge 0 -and $successIndex -ge 0

            while ($null -ne ($line = $reader.ReadLine())) {
                $csvRows += 1
                if ([string]::IsNullOrWhiteSpace($line) -or -not $leadingColumnsAvailable) {
                    $frameParseIssueCount += 1
                    continue
                }
                $values = $line.Split([char[]]@(","), $splitCount)
                if ($values.Count -le $leadingIndex) {
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

                $successValue = 0
                if ([int]::TryParse($values[$successIndex].Trim(), [ref]$successValue)) {
                    if ($successValue -eq 1) {
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
                    $confidenceCount += 1
                    $confidenceSum += $confidenceValue
                    if ($null -eq $minConfidence -or $confidenceValue -lt $minConfidence) {
                        $minConfidence = $confidenceValue
                    }
                }
                else {
                    $confidenceParseIssueCount += 1
                }
            }
        }
    }
    finally {
        $reader.Dispose()
    }

    if ($csvRows -eq 0) {
        $issues.Add("empty_csv")
    }
    if ($csvRows -ne $ExpectedRows) {
        $issues.Add("row_count:$csvRows!=$ExpectedRows")
    }
    if ($firstFrame -ne 1) {
        $issues.Add("first_frame:$firstFrame")
    }
    if ($lastFrame -ne $ExpectedRows) {
        $issues.Add("last_frame:$lastFrame!=$ExpectedRows")
    }
    if ($frameSequenceIssueCount -gt 0) {
        $issues.Add("non_sequential_frames:$frameSequenceIssueCount")
    }
    if ($frameParseIssueCount -gt 0) {
        $issues.Add("frame_parse_errors:$frameParseIssueCount")
    }
    if ($successParseIssueCount -gt 0) {
        $issues.Add("success_parse_errors:$successParseIssueCount")
    }
    if ($confidenceParseIssueCount -gt 0) {
        $issues.Add("confidence_parse_errors:$confidenceParseIssueCount")
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
        Status = $(if ($issues.Count -eq 0) { "PASS" } else { "FAIL" })
        Issues = ($issues -join ";")
    }
}

function Get-GitOutput {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string[]]$GitArguments
    )
    try {
        $lines = @(& git -C $RepositoryRoot @GitArguments 2>$null)
        if ($LASTEXITCODE -ne 0) {
            return ""
        }
        return ($lines -join "`n").Trim()
    }
    catch {
        return ""
    }
}

$expectedPackageDirectory = "OpenFace_2.2.0_win_x64"
$releaseRoot = Join-Path $OpenFaceRoot $expectedPackageDirectory
$featureExtraction = Join-Path $releaseRoot "FeatureExtraction.exe"
$modelPath = Join-Path $releaseRoot "model\main_ceclm_general.txt"
$readmePath = Join-Path $releaseRoot "readme.txt"

foreach ($requiredPath in @($featureExtraction, $modelPath, $readmePath, $ImageRoot, $IntegrityComparisonSummary)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

$integrity = Get-Content -LiteralPath $IntegrityComparisonSummary -Raw | ConvertFrom-Json
if ($integrity.status -ne "EXACT_PASS") {
    throw "Image integrity gate is not EXACT_PASS: $($integrity.status)"
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
$detectedGitCommit = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("rev-parse", "HEAD")
$detectedGitBranch = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("branch", "--show-current")
$detectedGitStatus = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("status", "--short")
$hasExplicitCommit = -not [string]::IsNullOrWhiteSpace($SourceGitCommit)
$hasExplicitBranch = -not [string]::IsNullOrWhiteSpace($SourceGitBranch)
if ($hasExplicitCommit -xor $hasExplicitBranch) {
    throw "SourceGitCommit and SourceGitBranch must be provided together."
}

if ($hasExplicitCommit) {
    $resolvedGitCommit = $SourceGitCommit.Trim().ToLowerInvariant()
    $resolvedGitBranch = $SourceGitBranch.Trim()
    if ($resolvedGitCommit -notmatch "^[0-9a-f]{7,40}$") {
        throw "SourceGitCommit must be a 7-40 character hexadecimal git commit: $resolvedGitCommit"
    }
    if ($detectedGitCommit -and -not $detectedGitCommit.ToLowerInvariant().StartsWith($resolvedGitCommit)) {
        throw "Explicit SourceGitCommit does not match the detected checkout: $resolvedGitCommit != $detectedGitCommit"
    }
    if ($detectedGitBranch -and $detectedGitBranch -ne $resolvedGitBranch) {
        throw "Explicit SourceGitBranch does not match the detected checkout: $resolvedGitBranch != $detectedGitBranch"
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

$videoDirs = @(
    Get-ChildItem -LiteralPath $ImageRoot -Directory |
        Where-Object { $_.Name -like "*_video_aligned" } |
        Sort-Object Name
)
if ($MaxVideos -gt 0) {
    $videoDirs = @($videoDirs | Select-Object -First $MaxVideos)
}
if ($videoDirs.Count -eq 0) {
    throw "No *_video_aligned directories found under $ImageRoot"
}

if ((Test-Path -LiteralPath $OutputRoot) -and -not $Resume) {
    $existing = @(Get-ChildItem -LiteralPath $OutputRoot -Force -ErrorAction SilentlyContinue)
    if ($existing.Count -gt 0) {
        throw "OutputRoot is not empty. Use a new directory or pass -Resume: $OutputRoot"
    }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$auditRoot = Join-Path $OutputRoot "_audit"
$logRoot = Join-Path $auditRoot "logs"
New-Item -ItemType Directory -Path $auditRoot -Force | Out-Null
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

$binaryManifest = [System.Collections.Generic.List[object]]::new()
$binaryFiles = @(
    Get-ChildItem -LiteralPath $releaseRoot -File |
        Where-Object { $_.Extension -in @(".exe", ".dll") } |
        Sort-Object Name
)
foreach ($file in $binaryFiles) {
    $binaryManifest.Add([pscustomobject]@{
        relative_path = $file.Name
        size = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}
$binaryManifest | Export-Csv -LiteralPath (Join-Path $auditRoot "binary_manifest.csv") -NoTypeInformation -Encoding UTF8

$modelManifest = [System.Collections.Generic.List[object]]::new()
$modelRoot = Join-Path $releaseRoot "model"
foreach ($file in @(Get-ChildItem -LiteralPath $modelRoot -File -Recurse | Sort-Object FullName)) {
    $relativePath = $file.FullName.Substring($modelRoot.Length)
    if ($relativePath.StartsWith([System.IO.Path]::DirectorySeparatorChar.ToString())) {
        $relativePath = $relativePath.Substring(1)
    }
    $modelManifest.Add([pscustomobject]@{
        relative_path = $relativePath
        size = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}
$modelManifest | Export-Csv -LiteralPath (Join-Path $auditRoot "model_manifest.csv") -NoTypeInformation -Encoding UTF8

$provenance = [ordered]@{
    audit = "OpenFace aligned-image landmark extraction"
    created_utc = [DateTime]::UtcNow.ToString("o")
    computer_name = $env:COMPUTERNAME
    powershell_version = $PSVersionTable.PSVersion.ToString()
    script_path = $scriptPath
    script_sha256 = (Get-FileHash -LiteralPath $scriptPath -Algorithm SHA256).Hash.ToLowerInvariant()
    git_commit = $resolvedGitCommit
    git_branch = $resolvedGitBranch
    git_status_short = $detectedGitStatus
    git_status_available = [bool]$detectedGitCommit
    git_provenance_mode = $gitProvenanceMode
    git_repository_root = $repositoryRoot
    invocation_line = $MyInvocation.Line
    openface_root = $OpenFaceRoot
    openface_package_root = $releaseRoot
    openface_working_directory = $releaseRoot
    openface_package_directory = $releaseDirectoryName
    expected_openface_package_directory = $expectedPackageDirectory
    openface_readme = $readmePath
    openface_readme_sha256 = $readmeSha256
    expected_openface_readme_sha256 = $expectedReadmeSha256Normalized
    feature_extraction = $featureExtraction
    feature_extraction_sha256 = $featureExtractionSha256
    expected_feature_extraction_sha256 = $expectedFeatureExtractionSha256Normalized
    model_path = $modelPath
    model_sha256 = $modelSha256
    expected_model_sha256 = $expectedModelSha256Normalized
    image_root = $ImageRoot
    output_root = $OutputRoot
    integrity_comparison_summary = $IntegrityComparisonSummary
    integrity_comparison_summary_sha256 = (Get-FileHash -LiteralPath $IntegrityComparisonSummary -Algorithm SHA256).Hash.ToLowerInvariant()
    integrity_status = $integrity.status
    integrity_reference_count = $integrity.reference_count
    integrity_candidate_count = $integrity.candidate_count
    mapping_contract = "full sorted JPG sequence -> FeatureExtraction -fdir -> 2D landmarks only"
    arguments = @("-fdir", "<video_dir>", "-out_dir", $OutputRoot, "-2Dfp", "-mloc", $modelPath)
    min_success_ratio = $MinSuccessRatio
    max_videos = $MaxVideos
    resume = [bool]$Resume
}
$provenance | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $auditRoot "run_manifest.json") -Encoding UTF8

$runRows = [System.Collections.Generic.List[object]]::new()
$overallStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$index = 0
foreach ($videoDir in $videoDirs) {
    $index += 1
    $videoId = $videoDir.Name
    $expectedCsv = Join-Path $OutputRoot "$videoId.csv"
    $logPath = Join-Path $logRoot "$videoId.log"
    $imageCount = @(
        Get-ChildItem -LiteralPath $videoDir.FullName -File |
            Where-Object { $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg") }
    ).Count

    Write-Host "[$index/$($videoDirs.Count)] $videoId images=$imageCount"
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = 0
    $skipped = $false

    if ($Resume -and (Test-Path -LiteralPath $expectedCsv -PathType Leaf)) {
        $preContract = Get-CsvContract -CsvPath $expectedCsv -ExpectedRows $imageCount -RequiredSuccessRatio $MinSuccessRatio
        if ($preContract.Status -eq "PASS") {
            $skipped = $true
            Write-Host "  resume: existing CSV contract passed"
        }
    }

    if (-not $skipped) {
        $previousErrorActionPreference = $ErrorActionPreference
        $previousLocation = Get-Location
        try {
            # OpenFace model manifests contain relative paths; resolve them from the release package.
            Set-Location -LiteralPath $releaseRoot
            $ErrorActionPreference = "Continue"
            & $featureExtraction `
                -fdir $videoDir.FullName `
                -out_dir $OutputRoot `
                -2Dfp `
                -mloc $modelPath 2>&1 |
                Tee-Object -FilePath $logPath
            $exitCode = $LASTEXITCODE
        }
        finally {
            Set-Location -LiteralPath $previousLocation
            $ErrorActionPreference = $previousErrorActionPreference
        }
    }

    $stopwatch.Stop()
    $contract = Get-CsvContract -CsvPath $expectedCsv -ExpectedRows $imageCount -RequiredSuccessRatio $MinSuccessRatio
    $issues = [System.Collections.Generic.List[string]]::new()
    if ($exitCode -ne 0) {
        $issues.Add("exit_code:$exitCode")
    }
    if ($contract.Issues) {
        $issues.Add($contract.Issues)
    }
    $status = $(if ($issues.Count -eq 0) { "PASS" } else { "FAIL" })
    $runRows.Add([pscustomobject]@{
        video_id = $videoId
        image_dir = $videoDir.FullName
        image_count = $imageCount
        csv_path = $expectedCsv
        csv_rows = $contract.CsvRows
        first_frame = $contract.FirstFrame
        last_frame = $contract.LastFrame
        success_count = $contract.SuccessCount
        success_ratio = [Math]::Round($contract.SuccessRatio, 6)
        mean_confidence = $(if ($null -ne $contract.MeanConfidence) { [Math]::Round($contract.MeanConfidence, 6) } else { $null })
        min_confidence = $(if ($null -ne $contract.MinConfidence) { [Math]::Round($contract.MinConfidence, 6) } else { $null })
        frame_sequence_issue_count = $contract.FrameSequenceIssueCount
        exit_code = $exitCode
        skipped_by_resume = $skipped
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
$passCount = @($runRows | Where-Object { $_.status -eq "PASS" }).Count
$failCount = @($runRows | Where-Object { $_.status -ne "PASS" }).Count
$totalImages = ($runRows | Measure-Object -Property image_count -Sum).Sum
$totalRows = ($runRows | Measure-Object -Property csv_rows -Sum).Sum
$totalSuccesses = ($runRows | Measure-Object -Property success_count -Sum).Sum
$overallSuccessRatio = $(if ($totalRows -gt 0) { $totalSuccesses / [double]$totalRows } else { 0.0 })
$finalSummary = [ordered]@{
    audit = "OpenFace aligned-image landmark extraction summary"
    created_utc = [DateTime]::UtcNow.ToString("o")
    video_count = $runRows.Count
    pass_count = $passCount
    fail_count = $failCount
    total_images = $totalImages
    total_csv_rows = $totalRows
    total_successes = $totalSuccesses
    success_ratio = [Math]::Round($overallSuccessRatio, 6)
    min_required_success_ratio = $MinSuccessRatio
    elapsed_seconds = [Math]::Round($overallStopwatch.Elapsed.TotalSeconds, 3)
    status = $(if ($failCount -eq 0 -and $totalImages -eq $totalRows) { "PASS" } else { "FAIL" })
    video_run_summary = $summaryPath
}
$finalSummary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $auditRoot "extraction_summary.json") -Encoding UTF8

Write-Host "OpenFace extraction complete: PASS=$passCount FAIL=$failCount images=$totalImages csv_rows=$totalRows success_ratio=$([Math]::Round($overallSuccessRatio, 6))"
if ($finalSummary.status -ne "PASS") {
    exit 1
}
