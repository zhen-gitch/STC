[CmdletBinding()]
param(
    [string]$OpenFaceRoot = "D:\Tools\Openface_2.2.0_win_x64",
    [string]$RawVideoRoot = "D:\Project\dataset\AVEC2014",
    [Parameter(Mandatory = $true)]
    [string]$SourceVideoContract,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [string]$ExpectedSourceVideoContractSha256 = "12f50c2311b9d89dde81e27fa83891226ca5f447ed7d3d56fe38cf673a1c5a31",
    [string]$ExpectedFeatureExtractionSha256 = "a29ba49cfc59039bfe5e2f141898b2a110da420f6f520d6a923a86ac78cd96ae",
    [string]$ExpectedModelSha256 = "7efbef33dbc3e54197960300827657f9fe7a42c0953ef52c2af054a6fdbc3598",
    [string]$ExpectedReadmeSha256 = "4ccdd65f992124db8127688a545a9344b537bdeb2d97371cfddfd65fc68a1d93",
    [string]$ExpectedAuPredictorManifestSha256 = "b65b923db38e75ee53ea7f85d614882f2f834571c257c7e0c3637d348a0192d6",
    [string]$SourceGitCommit = "",
    [string]$SourceGitBranch = "",
    [ValidateRange(0, 2147483647)]
    [int]$MaxVideos = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pathExtensions = @(([string]$env:PATHEXT).Split(";") | ForEach-Object { $_.ToUpperInvariant() })
if ($pathExtensions -notcontains ".EXE") {
    $env:PATHEXT = ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.CPL"
}

$featureProfile = "quality_2d_pose_au_no_gaze_no_hog_v1"
$featureArguments = @("-2Dfp", "-pose", "-aus")
$coreAuColumns = @("AU12_r", "AU14_r", "AU15_r")
$expectedVideoCount = 300
$expectedFrameCount = 493141L

function Get-NormalizedPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $providerPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    return [System.IO.Path]::GetFullPath($providerPath).TrimEnd("\")
}

function Convert-RecordedPath {
    param([Parameter(Mandatory = $true)][string]$RecordedPath)
    if (Test-Path -LiteralPath $RecordedPath) {
        return Get-NormalizedPath -Path $RecordedPath
    }
    if ($RecordedPath -match "^/mnt/([A-Za-z])/(.+)$") {
        $drive = $Matches[1].ToUpperInvariant()
        $tail = $Matches[2] -replace "/", "\"
        return [System.IO.Path]::GetFullPath("${drive}:\$tail")
    }
    throw "Cannot convert recorded source path to Windows: $RecordedPath"
}

function Test-PathInside {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Root
    )
    $prefix = $Root.TrimEnd("\") + "\"
    return $Candidate.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
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
    try {
        $lines = @(& git -C $RepositoryRoot @GitArguments 2>$null)
        $exitCode = [int]$LASTEXITCODE
        return [pscustomobject]@{
            Output = ($lines -join "`n").TrimEnd()
            Available = $exitCode -eq 0
        }
    }
    catch {
        return [pscustomobject]@{ Output = ""; Available = $false }
    }
}

function Test-FileManifestUnchanged {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][System.Collections.IEnumerable]$Rows
    )
    foreach ($row in @($Rows)) {
        $path = Join-Path $Root ([string]$row.relative_path)
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
        $file = Get-Item -LiteralPath $path
        if ([long]$file.Length -ne [long]$row.size) { return $false }
        try {
            $sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        catch {
            return $false
        }
        if ($sha256 -ne [string]$row.sha256) { return $false }
    }
    return $true
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
        [Parameter(Mandatory = $true)][long]$ExpectedRows,
        [Parameter(Mandatory = $true)][string[]]$RequiredColumns,
        [Parameter(Mandatory = $true)][string[]]$SelectedAuColumns
    )
    $issues = [System.Collections.Generic.List[string]]::new()
    if (-not (Test-Path -LiteralPath $CsvPath -PathType Leaf)) {
        return [pscustomobject]@{
            CsvRows = 0L; FirstFrame = $null; LastFrame = $null; SuccessCount = 0L
            MeanConfidence = $null; MinConfidence = $null; SchemaSha256 = ""
            SchemaColumnCount = 0; Status = "FAIL"; Issues = "missing_csv"
        }
    }

    $csvRows = 0L
    $firstFrame = $null
    $lastFrame = $null
    $successCount = 0L
    $confidenceCount = 0L
    $confidenceSum = 0.0
    $minConfidence = $null
    $reader = [System.IO.File]::OpenText($CsvPath)
    try {
        $headerLine = $reader.ReadLine()
        if ([string]::IsNullOrWhiteSpace($headerLine)) {
            throw "CSV has no header: $CsvPath"
        }
        $columns = @($headerLine.Split([char]",") | ForEach-Object { $_.Trim() })
        $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
        foreach ($column in $columns) {
            if (-not $seen.Add($column)) { $issues.Add("duplicate_column:$column") }
        }
        foreach ($column in $RequiredColumns) {
            if (-not $seen.Contains($column)) { $issues.Add("missing_column:$column") }
        }
        foreach ($column in $columns) {
            if ($RequiredColumns -notcontains $column) { $issues.Add("unexpected_column:$column") }
        }
        $indexes = @{}
        foreach ($column in @("frame", "success", "confidence") + $SelectedAuColumns) {
            $indexes[$column] = [Array]::IndexOf($columns, $column)
        }
        while ($null -ne ($line = $reader.ReadLine())) {
            $csvRows += 1
            $values = $line.Split([char]",")
            if ($values.Count -ne $columns.Count) {
                $issues.Add("row_column_count:$csvRows")
                continue
            }
            $frame = 0L
            if (-not [long]::TryParse($values[[int]$indexes["frame"]].Trim(), [ref]$frame)) {
                $issues.Add("frame_parse:$csvRows")
            }
            else {
                if ($null -eq $firstFrame) { $firstFrame = $frame }
                $lastFrame = $frame
                if ($frame -ne $csvRows) { $issues.Add("frame_sequence:$csvRows`:$frame") }
            }
            $success = 0.0
            if (-not [double]::TryParse(
                $values[[int]$indexes["success"]].Trim(),
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$success
            ) -or [double]::IsNaN($success) -or [double]::IsInfinity($success) -or
                ($success -ne 0.0 -and $success -ne 1.0)) {
                $issues.Add("success_domain:$csvRows")
            }
            elseif ($success -eq 1.0) { $successCount += 1 }
            $confidence = 0.0
            if (-not [double]::TryParse(
                $values[[int]$indexes["confidence"]].Trim(),
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$confidence
            ) -or [double]::IsNaN($confidence) -or [double]::IsInfinity($confidence) -or
                $confidence -lt 0.0 -or $confidence -gt 1.0) {
                $issues.Add("confidence_domain:$csvRows")
            }
            else {
                $confidenceCount += 1
                $confidenceSum += $confidence
                if ($null -eq $minConfidence -or $confidence -lt $minConfidence) {
                    $minConfidence = $confidence
                }
            }
            foreach ($au in $SelectedAuColumns) {
                $value = 0.0
                if (-not [double]::TryParse(
                    $values[[int]$indexes[$au]].Trim(),
                    [System.Globalization.NumberStyles]::Float,
                    [System.Globalization.CultureInfo]::InvariantCulture,
                    [ref]$value
                ) -or [double]::IsNaN($value) -or [double]::IsInfinity($value) -or
                    $value -lt 0.0 -or $value -gt 5.0) {
                    $issues.Add("selected_au_domain:$au`:$csvRows")
                }
            }
        }
    }
    finally {
        $reader.Dispose()
    }
    if ($csvRows -ne $ExpectedRows) { $issues.Add("row_count:$csvRows!=$ExpectedRows") }
    if ($firstFrame -ne 1) { $issues.Add("first_frame:$firstFrame") }
    if ($lastFrame -ne $ExpectedRows) { $issues.Add("last_frame:$lastFrame!=$ExpectedRows") }
    return [pscustomobject]@{
        CsvRows = $csvRows
        FirstFrame = $firstFrame
        LastFrame = $lastFrame
        SuccessCount = $successCount
        MeanConfidence = $(if ($confidenceCount -gt 0) { $confidenceSum / $confidenceCount } else { $null })
        MinConfidence = $minConfidence
        SchemaSha256 = Get-TextSha256 -Text ($columns -join ",")
        SchemaColumnCount = $columns.Count
        Status = $(if ($issues.Count -eq 0) { "PASS" } else { "FAIL" })
        Issues = ($issues -join ";")
    }
}

$expectedPackageDirectory = "OpenFace_2.2.0_win_x64"
$releaseRoot = Join-Path (Get-NormalizedPath -Path $OpenFaceRoot) $expectedPackageDirectory
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
foreach ($path in @($featureExtraction, $modelPath, $readmePath, $auPredictorManifestPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -le 0) {
        throw "Required OpenFace file is missing or empty: $path"
    }
}
foreach ($relativePath in $requiredModelRelativePaths) {
    $path = Join-Path $releaseRoot $relativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -le 0) {
        throw "Required OpenFace CEN model dependency is missing or empty: $relativePath"
    }
}
$auPredictorPaths = [System.Collections.Generic.List[string]]::new()
foreach ($line in @(Get-Content -LiteralPath $auPredictorManifestPath)) {
    $trimmed = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) { continue }
    $relativePath = ($trimmed -split "\s+")[0]
    $path = Join-Path (Split-Path -Parent $auPredictorManifestPath) $relativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -le 0) {
        throw "Required AU predictor is missing or empty: $relativePath"
    }
    $auPredictorPaths.Add($path)
}
if ($auPredictorPaths.Count -eq 0) { throw "AU predictor manifest contains no predictor files." }

$featureExtractionSha256 = (Get-FileHash -LiteralPath $featureExtraction -Algorithm SHA256).Hash.ToLowerInvariant()
$modelSha256 = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
$readmeSha256 = (Get-FileHash -LiteralPath $readmePath -Algorithm SHA256).Hash.ToLowerInvariant()
$auPredictorManifestSha256 = (Get-FileHash -LiteralPath $auPredictorManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($featureExtractionSha256 -ne $ExpectedFeatureExtractionSha256.ToLowerInvariant()) {
    throw "FeatureExtraction.exe SHA-256 mismatch: $featureExtractionSha256"
}
if ($modelSha256 -ne $ExpectedModelSha256.ToLowerInvariant()) {
    throw "OpenFace model SHA-256 mismatch: $modelSha256"
}
if ($readmeSha256 -ne $ExpectedReadmeSha256.ToLowerInvariant()) {
    throw "OpenFace readme SHA-256 mismatch: $readmeSha256"
}
if ($auPredictorManifestSha256 -ne $ExpectedAuPredictorManifestSha256.ToLowerInvariant()) {
    throw "OpenFace AU predictor manifest SHA-256 mismatch: $auPredictorManifestSha256"
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$gitCommitResult = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("rev-parse", "HEAD")
$gitBranchResult = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("branch", "--show-current")
$gitStatusResult = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("status", "--short")
$detectedCommit = $(if ($gitCommitResult.Available) { [string]$gitCommitResult.Output } else { "" })
$detectedBranch = $(if ($gitBranchResult.Available) { [string]$gitBranchResult.Output } else { "" })
$detectedStatus = $(if ($gitStatusResult.Available) { [string]$gitStatusResult.Output } else { "" })
if ($MaxVideos -eq 0) {
    if (-not $gitStatusResult.Available) { throw "Full extraction requires available git status." }
    if (-not [string]::IsNullOrWhiteSpace($detectedStatus)) {
        throw "Full extraction requires a clean git checkout."
    }
}
$hasCommit = -not [string]::IsNullOrWhiteSpace($SourceGitCommit)
$hasBranch = -not [string]::IsNullOrWhiteSpace($SourceGitBranch)
if ($hasCommit -xor $hasBranch) { throw "SourceGitCommit and SourceGitBranch must be provided together." }
if ($hasCommit) {
    $resolvedCommit = $SourceGitCommit.Trim().ToLowerInvariant()
    $resolvedBranch = $SourceGitBranch.Trim()
    if ($resolvedCommit -notmatch "^[0-9a-f]{7,40}$") { throw "SourceGitCommit is not hexadecimal." }
    if ($detectedCommit -and -not $detectedCommit.ToLowerInvariant().StartsWith($resolvedCommit)) {
        throw "SourceGitCommit does not match the checkout."
    }
    if ($detectedBranch -and $detectedBranch -ne $resolvedBranch) {
        throw "SourceGitBranch does not match the checkout."
    }
}
else {
    $resolvedCommit = $detectedCommit.ToLowerInvariant()
    $resolvedBranch = $detectedBranch
}
if ([string]::IsNullOrWhiteSpace($resolvedCommit) -or [string]::IsNullOrWhiteSpace($resolvedBranch)) {
    throw "Git provenance is unavailable."
}

$resolvedRawVideoRoot = Get-NormalizedPath -Path $RawVideoRoot
$resolvedSourceVideoContract = Get-NormalizedPath -Path $SourceVideoContract
$resolvedRepositoryRoot = Get-NormalizedPath -Path $repositoryRoot
$resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot).TrimEnd("\")
if (-not (Test-Path -LiteralPath $resolvedRawVideoRoot -PathType Container)) {
    throw "RawVideoRoot is not a directory: $resolvedRawVideoRoot"
}
if (-not (Test-Path -LiteralPath $resolvedSourceVideoContract -PathType Leaf)) {
    throw "SourceVideoContract is not a file: $resolvedSourceVideoContract"
}
$sourceVideoContractSha256 = (Get-FileHash -LiteralPath $resolvedSourceVideoContract -Algorithm SHA256).Hash.ToLowerInvariant()
if ($sourceVideoContractSha256 -ne $ExpectedSourceVideoContractSha256.ToLowerInvariant()) {
    throw "Source-video contract SHA-256 mismatch: $sourceVideoContractSha256"
}
if (Test-Path -LiteralPath $resolvedOutputRoot) {
    throw "OutputRoot already exists; refusing overwrite or resume: $resolvedOutputRoot"
}
if ([System.StringComparer]::OrdinalIgnoreCase.Equals($resolvedOutputRoot, $resolvedRawVideoRoot)) {
    throw "OutputRoot cannot replace RawVideoRoot: $resolvedOutputRoot"
}
if ([System.StringComparer]::OrdinalIgnoreCase.Equals($resolvedOutputRoot, $releaseRoot) -or
    (Test-PathInside -Candidate $resolvedOutputRoot -Root $releaseRoot)) {
    throw "OutputRoot must be outside the OpenFace package root: $resolvedOutputRoot"
}
if ([System.StringComparer]::OrdinalIgnoreCase.Equals($resolvedOutputRoot, $resolvedRepositoryRoot) -or
    (Test-PathInside -Candidate $resolvedOutputRoot -Root $resolvedRepositoryRoot)) {
    throw "OutputRoot must be outside the git repository: $resolvedOutputRoot"
}

$sourceRows = @(Import-Csv -LiteralPath $resolvedSourceVideoContract)
if ($sourceRows.Count -ne $expectedVideoCount) {
    throw "Source-video contract row count mismatch: $($sourceRows.Count) != $expectedVideoCount"
}
$requiredSourceColumns = @(
    "split", "video_id", "raw_video_id", "task_name", "raw_video_path",
    "aligned_frame_count", "raw_frame_count", "frame_count_match", "raw_fps",
    "contract_status", "issues"
)
foreach ($column in $requiredSourceColumns) {
    if ($sourceRows[0].PSObject.Properties.Name -notcontains $column) {
        throw "Source-video contract is missing required column: $column"
    }
}
$seenVideoIds = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
$seenRawVideoIds = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
$inputRows = [System.Collections.Generic.List[object]]::new()
$totalFrames = 0L
foreach ($row in @($sourceRows | Sort-Object video_id)) {
    $videoId = [string]$row.video_id
    $rawVideoId = [string]$row.raw_video_id
    if ([string]::IsNullOrWhiteSpace($videoId) -or -not $seenVideoIds.Add($videoId)) {
        throw "Source-video contract has empty/duplicate video_id: $videoId"
    }
    if ([string]::IsNullOrWhiteSpace($rawVideoId) -or -not $seenRawVideoIds.Add($rawVideoId)) {
        throw "Source-video contract has empty/duplicate raw_video_id: $rawVideoId"
    }
    if ($videoId -ne "$rawVideoId`_aligned") {
        throw "Aligned/raw video identity mismatch: $videoId / $rawVideoId"
    }
    if ($row.task_name -notin @("Freeform", "Northwind") -or
        $videoId -notlike "*_$($row.task_name)_*") {
        throw "Task identity mismatch for $videoId"
    }
    if ($row.split -notin @("train", "val", "test")) {
        throw "Invalid physical split for ${videoId}: $($row.split)"
    }
    $alignedFrames = [long]$row.aligned_frame_count
    $rawFrames = [long]$row.raw_frame_count
    $fps = [double]$row.raw_fps
    if ($row.contract_status -ne "PASS" -or -not [string]::IsNullOrWhiteSpace([string]$row.issues) -or
        [int]$row.frame_count_match -ne 1 -or $alignedFrames -ne $rawFrames -or
        [Math]::Abs($fps - 30.0) -gt 0.000001) {
        throw "Source-video frame/FPS contract is not PASS for $videoId"
    }
    $rawVideoPath = Convert-RecordedPath -RecordedPath ([string]$row.raw_video_path)
    if (-not (Test-Path -LiteralPath $rawVideoPath -PathType Leaf)) {
        throw "Raw video does not exist: $rawVideoPath"
    }
    if (-not (Test-PathInside -Candidate $rawVideoPath -Root $resolvedRawVideoRoot)) {
        throw "Raw video is outside RawVideoRoot: $rawVideoPath"
    }
    if ([System.IO.Path]::GetFileNameWithoutExtension($rawVideoPath) -ne $rawVideoId -or
        [System.IO.Path]::GetExtension($rawVideoPath).ToLowerInvariant() -ne ".mp4") {
        throw "Raw video filename does not match raw_video_id: $rawVideoPath"
    }
    $totalFrames += $rawFrames
    $inputRows.Add([pscustomobject]@{
        split = [string]$row.split
        video_id = $videoId
        raw_video_id = $rawVideoId
        task_name = [string]$row.task_name
        raw_video_path = $rawVideoPath
        source_frame_count = $rawFrames
        source_fps = $fps
        selected_for_run = $false
        status = "PASS"
        issues = ""
    })
}
if ($totalFrames -ne $expectedFrameCount) {
    throw "Source-video total frame count mismatch: $totalFrames != $expectedFrameCount"
}
$sourceVideoDirectories = @($inputRows | ForEach-Object { Split-Path -Parent $_.raw_video_path } | Sort-Object -Unique)
foreach ($sourceVideoDirectory in $sourceVideoDirectories) {
    if ([System.StringComparer]::OrdinalIgnoreCase.Equals($resolvedOutputRoot, $sourceVideoDirectory) -or
        (Test-PathInside -Candidate $resolvedOutputRoot -Root $sourceVideoDirectory)) {
        throw "OutputRoot must not overlap a source-video directory: $resolvedOutputRoot"
    }
}
$selectedRows = @($inputRows)
if ($MaxVideos -gt 0) { $selectedRows = @($selectedRows | Select-Object -First $MaxVideos) }
$selectedIds = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
foreach ($row in $selectedRows) { [void]$selectedIds.Add($row.video_id) }
foreach ($row in $inputRows) { $row.selected_for_run = $selectedIds.Contains($row.video_id) }
$selectedFrames = [long](($selectedRows | Measure-Object -Property source_frame_count -Sum).Sum)

New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null
$auditRoot = Join-Path $resolvedOutputRoot "_audit"
$logRoot = Join-Path $auditRoot "logs"
New-Item -ItemType Directory -Path $auditRoot | Out-Null
New-Item -ItemType Directory -Path $logRoot | Out-Null

$inputContractPath = Join-Path $auditRoot "input_video_contract.csv"
$inputRows | Export-Csv -LiteralPath $inputContractPath -NoTypeInformation -Encoding UTF8
$inputContractSha256 = (Get-FileHash -LiteralPath $inputContractPath -Algorithm SHA256).Hash.ToLowerInvariant()

$binaryRows = [System.Collections.Generic.List[object]]::new()
foreach ($file in @(Get-ChildItem -LiteralPath $releaseRoot -File | Where-Object { $_.Extension -in @(".exe", ".dll") } | Sort-Object Name)) {
    $binaryRows.Add([pscustomobject]@{
        relative_path = $file.Name
        size = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}
if ($binaryRows.Count -eq 0) { throw "OpenFace binary manifest would be empty." }
$binaryManifestPath = Join-Path $auditRoot "binary_manifest.csv"
$binaryRows | Export-Csv -LiteralPath $binaryManifestPath -NoTypeInformation -Encoding UTF8
$binaryManifestSha256 = (Get-FileHash -LiteralPath $binaryManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()

$modelRows = [System.Collections.Generic.List[object]]::new()
foreach ($directoryName in @("model", "AU_predictors")) {
    $directory = Join-Path $releaseRoot $directoryName
    foreach ($file in @(Get-ChildItem -LiteralPath $directory -File -Recurse | Sort-Object FullName)) {
        $relativePath = $file.FullName.Substring($releaseRoot.Length)
        if ($relativePath.StartsWith([System.IO.Path]::DirectorySeparatorChar.ToString())) {
            $relativePath = $relativePath.Substring(1)
        }
        $modelRows.Add([pscustomobject]@{
            relative_path = $relativePath
            size = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        })
    }
}
if ($modelRows.Count -eq 0) { throw "OpenFace model manifest would be empty." }
$modelManifestPath = Join-Path $auditRoot "model_manifest.csv"
$modelRows | Export-Csv -LiteralPath $modelManifestPath -NoTypeInformation -Encoding UTF8
$modelManifestSha256 = (Get-FileHash -LiteralPath $modelManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()

$scriptPath = $MyInvocation.MyCommand.Path
$scriptSha256 = (Get-FileHash -LiteralPath $scriptPath -Algorithm SHA256).Hash.ToLowerInvariant()
$runManifestPath = Join-Path $auditRoot "run_manifest.json"
$provenance = [ordered]@{
    audit = "OpenFace raw-video core-AU fidelity reference extraction"
    implementation_package_id = "CODE-20260731-PB-P0C-CORE-AU-FIDELITY-v1"
    created_utc = [DateTime]::UtcNow.ToString("o")
    status = "RUNNING"
    script_path = $scriptPath
    script_sha256 = $scriptSha256
    invocation_line = $MyInvocation.Line
    process_command_line = [Environment]::CommandLine
    git_repository_root = $repositoryRoot
    git_commit = $resolvedCommit
    git_branch = $resolvedBranch
    git_status_available = $gitStatusResult.Available
    git_status_short = $detectedStatus
    openface_root = $OpenFaceRoot
    openface_package_root = $releaseRoot
    openface_working_directory = $releaseRoot
    feature_extraction = $featureExtraction
    feature_extraction_sha256 = $featureExtractionSha256
    expected_feature_extraction_sha256 = $ExpectedFeatureExtractionSha256.ToLowerInvariant()
    model_path = $modelPath
    model_sha256 = $modelSha256
    expected_model_sha256 = $ExpectedModelSha256.ToLowerInvariant()
    openface_readme = $readmePath
    openface_readme_sha256 = $readmeSha256
    expected_openface_readme_sha256 = $ExpectedReadmeSha256.ToLowerInvariant()
    au_predictor_manifest = $auPredictorManifestPath
    au_predictor_manifest_sha256 = $auPredictorManifestSha256
    expected_au_predictor_manifest_sha256 = $ExpectedAuPredictorManifestSha256.ToLowerInvariant()
    binary_manifest = $binaryManifestPath
    binary_manifest_sha256 = $binaryManifestSha256
    model_manifest = $modelManifestPath
    model_manifest_sha256 = $modelManifestSha256
    feature_profile = $featureProfile
    feature_arguments = $featureArguments
    input_mode = "raw_video"
    arguments = @("-f", "<raw_video_path>", "-out_dir", $resolvedOutputRoot) + $featureArguments + @("-mloc", $modelPath)
    raw_video_root = $resolvedRawVideoRoot
    source_video_contract = $resolvedSourceVideoContract
    source_video_contract_sha256 = $sourceVideoContractSha256
    expected_source_video_contract_sha256 = $ExpectedSourceVideoContractSha256.ToLowerInvariant()
    source_video_contract_accessed_columns = $requiredSourceColumns
    raw_video_path_field_access_count = $inputRows.Count
    raw_openface_csv_path_field_access_count = 0
    historical_raw_openface_feature_file_open_count = 0
    output_root = $resolvedOutputRoot
    output_root_must_be_new = $true
    input_video_count = $inputRows.Count
    input_frame_count = $totalFrames
    selected_video_count = $selectedRows.Count
    selected_frame_count = $selectedFrames
    run_scope = $(if ($MaxVideos -gt 0) { "debug_subset" } else { "full_dataset" })
    debug_subset_must_not_be_treated_as_full_extraction = $MaxVideos -gt 0
    max_videos = $MaxVideos
    input_video_contract = $inputContractPath
    input_video_contract_sha256 = $inputContractSha256
    selected_value_columns = $coreAuColumns
    quality_fields_authorized_for_mask_and_coverage_only = @("success", "confidence")
    pose_value_access_count = 0
    gaze_requested = $false
    gaze_value_access_count = 0
    extension_au_value_access_count = 0
    hog_enabled = $false
    tracked_video_enabled = $false
    aligned_image_generation_enabled = $false
    device = "cpu"
    precision = "OpenFace_2.2.0_binary_default"
    background_process = $false
}
$provenance | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $runManifestPath -Encoding UTF8

$requiredCsvColumns = Get-RequiredCsvColumns
$runRows = [System.Collections.Generic.List[object]]::new()
$overallStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$index = 0
foreach ($sourceRow in $selectedRows) {
    $index += 1
    $videoId = [string]$sourceRow.video_id
    $rawVideoId = [string]$sourceRow.raw_video_id
    $rawVideoPath = [string]$sourceRow.raw_video_path
    $expectedRows = [long]$sourceRow.source_frame_count
    $expectedCsv = Join-Path $resolvedOutputRoot "$rawVideoId.csv"
    $logPath = Join-Path $logRoot "$rawVideoId.log"
    Write-Host "[$index/$($selectedRows.Count)] $videoId frames=$expectedRows"
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = -1
    $processError = ""
    $previousPreference = $ErrorActionPreference
    $previousLocation = Get-Location
    try {
        Set-Location -LiteralPath $releaseRoot
        $ErrorActionPreference = "Continue"
        $arguments = @("-f", $rawVideoPath, "-out_dir", $resolvedOutputRoot) + $featureArguments + @("-mloc", $modelPath)
        & $featureExtraction @arguments 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = [int]$LASTEXITCODE
    }
    catch {
        $processError = $_.Exception.Message
    }
    finally {
        Set-Location -LiteralPath $previousLocation
        $ErrorActionPreference = $previousPreference
        $stopwatch.Stop()
    }
    $contract = Get-CsvContract `
        -CsvPath $expectedCsv `
        -ExpectedRows $expectedRows `
        -RequiredColumns $requiredCsvColumns `
        -SelectedAuColumns $coreAuColumns
    $issues = [System.Collections.Generic.List[string]]::new()
    if ($exitCode -ne 0) { $issues.Add("exit_code:$exitCode") }
    if ($processError) { $issues.Add("process_error:$processError") }
    if ($contract.Issues) { $issues.Add($contract.Issues) }
    $csvSize = 0L
    $csvSha256 = ""
    if (Test-Path -LiteralPath $expectedCsv -PathType Leaf) {
        $csvSize = (Get-Item -LiteralPath $expectedCsv).Length
        $csvSha256 = (Get-FileHash -LiteralPath $expectedCsv -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    if ($contract.Status -eq "PASS" -and ($csvSize -le 0 -or [string]::IsNullOrWhiteSpace($csvSha256))) {
        $issues.Add("missing_csv_content_provenance")
    }
    $status = $(if ($issues.Count -eq 0) { "PASS" } else { "FAIL" })
    $runRows.Add([pscustomobject]@{
        split = [string]$sourceRow.split
        video_id = $videoId
        raw_video_id = $rawVideoId
        task_name = [string]$sourceRow.task_name
        raw_video_path = $rawVideoPath
        source_frame_count = $expectedRows
        source_fps = [double]$sourceRow.source_fps
        csv_path = $expectedCsv
        csv_rows = $contract.CsvRows
        csv_size_bytes = $csvSize
        csv_sha256 = $csvSha256
        first_frame = $contract.FirstFrame
        last_frame = $contract.LastFrame
        success_count = $contract.SuccessCount
        success_ratio = $(if ($contract.CsvRows -gt 0) { $contract.SuccessCount / [double]$contract.CsvRows } else { 0.0 })
        mean_confidence = $contract.MeanConfidence
        min_confidence = $contract.MinConfidence
        schema_sha256 = $contract.SchemaSha256
        schema_column_count = $contract.SchemaColumnCount
        exit_code = $exitCode
        elapsed_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
        status = $status
        issues = ($issues -join ";")
        log_path = $logPath
    })
    if ($status -ne "PASS") { Write-Warning "$videoId failed: $($issues -join ';')" }
}
$overallStopwatch.Stop()

$videoSummaryPath = Join-Path $auditRoot "video_run_summary.csv"
$runRows | Export-Csv -LiteralPath $videoSummaryPath -NoTypeInformation -Encoding UTF8
$videoSummarySha256 = (Get-FileHash -LiteralPath $videoSummaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
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
$schemaManifestSha256 = (Get-FileHash -LiteralPath $schemaManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
$contentRows = @(
    $runRows |
        Sort-Object video_id |
        ForEach-Object {
            [pscustomobject]@{
                video_id = $_.video_id
                raw_video_id = $_.raw_video_id
                csv_rows = $_.csv_rows
                schema_sha256 = $_.schema_sha256
                csv_size_bytes = $_.csv_size_bytes
                csv_sha256 = $_.csv_sha256
                status = $_.status
            }
        }
)
$contentManifestPath = Join-Path $auditRoot "csv_content_manifest.csv"
$contentRows | Export-Csv -LiteralPath $contentManifestPath -NoTypeInformation -Encoding UTF8
$contentManifestSha256 = (Get-FileHash -LiteralPath $contentManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()

$passCount = @($runRows | Where-Object { $_.status -eq "PASS" }).Count
$failCount = $runRows.Count - $passCount
$totalOutputRows = [long](($runRows | Measure-Object -Property csv_rows -Sum).Sum)
$totalSuccesses = [long](($runRows | Measure-Object -Property success_count -Sum).Sum)
$hogFiles = @(Get-ChildItem -LiteralPath $resolvedOutputRoot -File -Recurse -Filter "*.hog" -ErrorAction SilentlyContinue)
$trackedFiles = @(
    Get-ChildItem -LiteralPath $resolvedOutputRoot -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notlike "$auditRoot*" -and $_.Extension.ToLowerInvariant() -in @(".avi", ".mp4", ".mov", ".mkv") }
)
$generatedImages = @(
    Get-ChildItem -LiteralPath $resolvedOutputRoot -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notlike "$auditRoot*" -and $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg", ".png", ".bmp") }
)
$endingGitCommitResult = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("rev-parse", "HEAD")
$endingGitBranchResult = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("branch", "--show-current")
$endingGitStatusResult = Get-GitOutput -RepositoryRoot $repositoryRoot -GitArguments @("status", "--short")
$scriptUnchanged = (
    (Get-FileHash -LiteralPath $scriptPath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $scriptSha256
)
$sourceVideoContractUnchanged = (
    (Get-FileHash -LiteralPath $resolvedSourceVideoContract -Algorithm SHA256).Hash.ToLowerInvariant() -eq
    $sourceVideoContractSha256
)
$binaryFilesUnchanged = Test-FileManifestUnchanged -Root $releaseRoot -Rows $binaryRows
$modelFilesUnchanged = Test-FileManifestUnchanged -Root $releaseRoot -Rows $modelRows
$gitProvenanceUnchanged = (
    $endingGitCommitResult.Available -and $endingGitBranchResult.Available -and
    $endingGitStatusResult.Available -and
    [string]$endingGitCommitResult.Output -eq $detectedCommit -and
    [string]$endingGitBranchResult.Output -eq $detectedBranch -and
    [string]$endingGitStatusResult.Output -eq $detectedStatus
)
$provenanceStable = (
    $scriptUnchanged -and $sourceVideoContractUnchanged -and $binaryFilesUnchanged -and
    $modelFilesUnchanged -and $gitProvenanceUnchanged
)
$finalStatus = $(
    if ($failCount -eq 0 -and $totalOutputRows -eq $selectedFrames -and $schemaRows.Count -eq 1 -and
        $hogFiles.Count -eq 0 -and $trackedFiles.Count -eq 0 -and $generatedImages.Count -eq 0 -and
        $provenanceStable) {
        "PASS"
    }
    else { "FAIL" }
)
$summary = [ordered]@{
    audit = "OpenFace raw-video core-AU fidelity reference extraction summary"
    created_utc = [DateTime]::UtcNow.ToString("o")
    feature_profile = $featureProfile
    run_scope = $(if ($MaxVideos -gt 0) { "debug_subset" } else { "full_dataset" })
    video_count = $runRows.Count
    pass_count = $passCount
    fail_count = $failCount
    total_source_frames = $selectedFrames
    total_csv_rows = $totalOutputRows
    total_successes = $totalSuccesses
    success_ratio = $(if ($totalOutputRows -gt 0) { $totalSuccesses / [double]$totalOutputRows } else { 0.0 })
    schema_count = $schemaRows.Count
    schema_consistent = $schemaRows.Count -eq 1
    hog_file_count = $hogFiles.Count
    tracked_video_file_count = $trackedFiles.Count
    generated_image_count = $generatedImages.Count
    provenance_stable = $provenanceStable
    script_unchanged = $scriptUnchanged
    source_video_contract_unchanged = $sourceVideoContractUnchanged
    binary_files_unchanged = $binaryFilesUnchanged
    model_files_unchanged = $modelFilesUnchanged
    git_provenance_unchanged = $gitProvenanceUnchanged
    elapsed_seconds = [Math]::Round($overallStopwatch.Elapsed.TotalSeconds, 3)
    status = $finalStatus
    csv_content_manifest_sha256 = $contentManifestSha256
}
$summaryPath = Join-Path $auditRoot "extraction_summary.json"
$summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
$summarySha256 = (Get-FileHash -LiteralPath $summaryPath -Algorithm SHA256).Hash.ToLowerInvariant()

$provenance["status"] = $finalStatus
$provenance["video_run_summary"] = $videoSummaryPath
$provenance["video_run_summary_sha256"] = $videoSummarySha256
$provenance["csv_schema_manifest"] = $schemaManifestPath
$provenance["csv_schema_manifest_sha256"] = $schemaManifestSha256
$provenance["csv_content_manifest"] = $contentManifestPath
$provenance["csv_content_manifest_sha256"] = $contentManifestSha256
$provenance["extraction_summary"] = $summaryPath
$provenance["extraction_summary_sha256"] = $summarySha256
$provenance["provenance_stable"] = $provenanceStable
$provenance["script_unchanged"] = $scriptUnchanged
$provenance["source_video_contract_unchanged"] = $sourceVideoContractUnchanged
$provenance["binary_files_unchanged"] = $binaryFilesUnchanged
$provenance["model_files_unchanged"] = $modelFilesUnchanged
$provenance["git_provenance_unchanged"] = $gitProvenanceUnchanged
$provenance | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $runManifestPath -Encoding UTF8

Write-Host "Raw AU reference extraction complete: status=$finalStatus PASS=$passCount FAIL=$failCount rows=$totalOutputRows"
if ($finalStatus -ne "PASS") { exit 1 }
