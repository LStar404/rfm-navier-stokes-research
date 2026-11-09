param([string]$Python = 'python', [int]$KnownServerPid = 68656,
      [int[]]$CompanionPids = @(34048, 41284))

$ErrorActionPreference = 'Stop'
$OutputDirectory = Join-Path $PSScriptRoot 'cpu_native_joint_output'
$Runner = Join-Path $PSScriptRoot 'cpu_native_joint.py'
if (-not (Test-Path -LiteralPath (Join-Path $OutputDirectory 'protocol.json'))) {
    throw 'The isolated timing protocol must already be frozen.'
}

function Read-PythonUsage {
    $Usage = @{}
    Get-Process -Name 'python*' -ErrorAction SilentlyContinue | ForEach-Object {
        $Usage[$_.Id] = [double]$_.CPU
    }
    return $Usage
}

function Read-GpuPercent {
    $Usage = (& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0) { throw 'Could not read GPU utilization.' }
    return [int]($Usage | Select-Object -First 1)
}

function Save-Monitor {
    $Temporary = "$MonitorPath.tmp"
    $Monitor | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Temporary -Encoding utf8NoBOM
    Move-Item -LiteralPath $Temporary -Destination $MonitorPath -Force
}

function Measure-PythonDeltas($Earlier, $Later, [int]$ExcludedPid = 0) {
    $Deltas = @()
    foreach ($ProcessId in $Later.Keys) {
        if ($ProcessId -eq $ExcludedPid) { continue }
        $Delta = if ($Earlier.ContainsKey($ProcessId)) { $Later[$ProcessId] - $Earlier[$ProcessId] } else { $Later[$ProcessId] }
        $Label = if ($ProcessId -eq $KnownServerPid) { 'known_server' } elseif ($CompanionPids -contains $ProcessId) { 'companion' } else { 'python' }
        $Deltas += [pscustomobject]@{ pid = $ProcessId; label = $Label; cpu_seconds = [Math]::Round($Delta, 4) }
    }
    return $Deltas
}

$Allowed = @($KnownServerPid) + @($CompanionPids)
$Before = Read-PythonUsage
foreach ($ProcessId in $Allowed) {
    if (-not $Before.ContainsKey($ProcessId)) { continue }
    $Identity = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId"
    $Pattern = if ($ProcessId -eq $KnownServerPid) { 'cli\.py.+\bserve\b' } else { '\bcompanion\b' }
    if ($null -eq $Identity -or $Identity.CommandLine -notmatch $Pattern) {
        throw "Background process identity changed: $ProcessId"
    }
}
if (@($Before.Keys | Where-Object { $Allowed -notcontains $_ }).Count -ne 0) {
    throw 'Other Python processes remain before the two-window assessment.'
}
$Preflight = @()
foreach ($Window in 1..2) {
    Start-Sleep -Seconds 10
    $Now = Read-PythonUsage
    $Background = @(Measure-PythonDeltas $Before $Now)
    $Gpu = Read-GpuPercent
    $Total = ($Background | Measure-Object -Property cpu_seconds -Sum).Sum
    if (@($Now.Keys | Where-Object { $Allowed -notcontains $_ }).Count -ne 0 -or
        @($Background | Where-Object { $_.cpu_seconds -gt 0.5 }).Count -ne 0 -or $Total -gt 1 -or $Gpu -gt 5) {
        throw "Background activity exceeded the registered preflight limits in window $Window."
    }
    $Preflight += [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); external_python = $Background; gpu_percent = $Gpu }
    $Before = $Now
}

$Stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$MonitorPath = Join-Path $OutputDirectory "isolation_$Stamp.json"
$StdoutPath = Join-Path $OutputDirectory "timing_$Stamp.log"
$StderrPath = Join-Path $OutputDirectory "timing_$Stamp.err.log"
$Monitor = [ordered]@{
    status = 'starting'
    start_utc = [DateTime]::UtcNow.ToString('o')
    initial_gpu_percent = $Gpu
    allowed_background_pids = $Allowed
    initial_external_python_pids = @($Before.Keys)
    preflight = $Preflight
    sampling_interval_seconds = 10
    allowed_background_individual_threshold_cpu_seconds = 0.5
    allowed_background_total_threshold_cpu_seconds = 1.0
    unexpected_python_threshold_cpu_seconds = 0.2
    samples = @()
    external_activity_detected = $false
}
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUNBUFFERED = '1'
$Process = Start-Process -FilePath $Python -ArgumentList @("`"$Runner`"", '--confirm-isolated') `
    -WindowStyle Hidden -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath -PassThru
$Monitor.status = 'running'
$Monitor.pid = $Process.Id
Save-Monitor
Write-Output "START $($Process.Id) $MonitorPath"

do {
    $Exited = $Process.WaitForExit(10000)
    $Now = Read-PythonUsage
    $External = @(Measure-PythonDeltas $Before $Now $Process.Id)
    $Background = @($External | Where-Object { $_.label -ne 'python' })
    $Total = ($Background | Measure-Object -Property cpu_seconds -Sum).Sum
    if (@($Background | Where-Object { $_.cpu_seconds -gt 0.5 }).Count -ne 0 -or $Total -gt 1 -or
        @($External | Where-Object { $_.label -eq 'python' -and $_.cpu_seconds -gt 0.2 }).Count -ne 0) {
        $Monitor.external_activity_detected = $true
    }
    $Monitor.samples += [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); external_python = $External }
    Save-Monitor
    $Before = $Now
} until ($Exited)

$Process.Refresh()
$Monitor.exit_code = $Process.ExitCode
$Monitor.end_utc = [DateTime]::UtcNow.ToString('o')
$Monitor.final_gpu_percent = Read-GpuPercent
$Monitor.status = if ($Process.ExitCode -ne 0) { 'failed' } elseif ($Monitor.external_activity_detected) {
    'completed_with_observed_external_activity'
} else { 'completed_no_sampled_external_activity' }
Save-Monitor
Write-Output "FINISH $($Monitor.status) $($Monitor.exit_code) $MonitorPath"
exit $Process.ExitCode
