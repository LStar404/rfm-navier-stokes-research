param([string]$Python = 'python', [int]$KnownServerPid = 68656,
      [int[]]$CompanionPids = @(34048, 41284))

$ErrorActionPreference = 'Stop'
$OutputDirectory = Join-Path $PSScriptRoot 'cpu_native_joint_repeat_output'
$ProtocolPath = Join-Path $OutputDirectory 'isolation_protocol.json'
$MonitorPath = Join-Path $OutputDirectory 'isolation_repeat.json'
$ExecutionPath = Join-Path $OutputDirectory 'execution.json'
$Runner = Join-Path $PSScriptRoot 'cpu_native_joint_repeat.py'
if (-not (Test-Path -LiteralPath $ProtocolPath) -or (Test-Path -LiteralPath $MonitorPath) -or
    (Test-Path -LiteralPath $ExecutionPath)) { throw 'Need one unused frozen repeat protocol.' }
$Protocol = Get-Content -LiteralPath $ProtocolPath -Raw | ConvertFrom-Json
if ($Protocol.status -ne 'frozen' -or $Protocol.window_seconds -ne 600 -or $Protocol.sample_seconds -ne 10 -or
    $Protocol.preflight_windows -ne 2) { throw 'Fixed-window protocol changed.' }
$Thresholds = $Protocol.thresholds
$ProtocolHash = (Get-FileHash -LiteralPath $ProtocolPath -Algorithm SHA256).Hash.ToLowerInvariant()

function Read-PythonUsage {
    $Usage = @{}
    Get-Process -Name 'python*' -ErrorAction SilentlyContinue | ForEach-Object { $Usage[$_.Id] = [double]$_.CPU }
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
    foreach ($ProcessId in $Later.Keys) {
        if ($ProcessId -eq $ExcludedPid) { continue }
        $Delta = if ($Earlier.ContainsKey($ProcessId)) { $Later[$ProcessId] - $Earlier[$ProcessId] } else { $Later[$ProcessId] }
        $Label = if ($ProcessId -eq $KnownServerPid) { 'known_server' } elseif ($CompanionPids -contains $ProcessId) { 'companion' } else { 'python' }
        [pscustomobject]@{ pid = $ProcessId; label = $Label; cpu_seconds = [Math]::Round($Delta, 4) }
    }
}

function Test-ExternalActivity($Rows) {
    $Background = @($Rows | Where-Object { $_.label -ne 'python' })
    $Total = ($Background | Measure-Object -Property cpu_seconds -Sum).Sum
    return (@($Background | Where-Object { $_.cpu_seconds -gt $Thresholds.background_individual_cpu_seconds }).Count -ne 0 -or
        $Total -gt $Thresholds.background_total_cpu_seconds -or
        @($Rows | Where-Object { $_.label -eq 'python' -and $_.cpu_seconds -gt $Thresholds.unexpected_python_cpu_seconds }).Count -ne 0)
}

$Allowed = @($KnownServerPid) + @($CompanionPids)
$Before = Read-PythonUsage
foreach ($ProcessId in $Allowed) {
    if (-not $Before.ContainsKey($ProcessId)) { continue }
    $Identity = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId"
    $Pattern = if ($ProcessId -eq $KnownServerPid) { 'cli\.py.+\bserve\b' } else { '\bcompanion\b' }
    if ($null -eq $Identity -or $Identity.CommandLine -notmatch $Pattern) { throw "Background identity changed: $ProcessId" }
}
if (@($Before.Keys | Where-Object { $Allowed -notcontains $_ }).Count -ne 0) { throw 'Other Python processes remain.' }
$Preflight = @()
foreach ($Window in 1..$Protocol.preflight_windows) {
    Start-Sleep -Seconds $Protocol.sample_seconds
    $Now = Read-PythonUsage
    $External = @(Measure-PythonDeltas $Before $Now)
    $Gpu = Read-GpuPercent
    if (@($Now.Keys | Where-Object { $Allowed -notcontains $_ }).Count -ne 0 -or
        (Test-ExternalActivity $External) -or $Gpu -gt $Thresholds.preflight_gpu_percent) {
        throw "Preflight window $Window did not pass. No repeat was started."
    }
    $Preflight += [ordered]@{ utc = [DateTimeOffset]::UtcNow.ToString('o'); external_python = $External; gpu_percent = $Gpu }
    $Before = $Now
}

$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUNBUFFERED = '1'
$Started = [DateTimeOffset]::UtcNow
$Deadline = $Started.AddSeconds($Protocol.window_seconds)
$Clock = [Diagnostics.Stopwatch]::StartNew()
$Monitor = [ordered]@{
    status = 'starting'; start_utc = $Started.ToString('o'); planned_end_utc = $Deadline.ToString('o')
    window_seconds = $Protocol.window_seconds; sampling_interval_seconds = $Protocol.sample_seconds
    isolation_protocol_sha256 = $ProtocolHash; preflight = $Preflight; initial_gpu_percent = $Gpu
    allowed_background_pids = $Allowed; samples = @(); external_activity_detected = $false
    sample_coverage_pass = $true
}
$Process = Start-Process -FilePath $Python -ArgumentList @("`"$Runner`"", '--execute') -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $OutputDirectory 'timing_repeat.log') `
    -RedirectStandardError (Join-Path $OutputDirectory 'timing_repeat.err.log') -PassThru
$Monitor.status = 'running'
$Monitor.pid = $Process.Id
Save-Monitor
Write-Output "START REPEAT $($Process.Id) $($Monitor.start_utc) $($Monitor.planned_end_utc)"

for ($Target = $Protocol.sample_seconds; $Target -le $Protocol.window_seconds; $Target += $Protocol.sample_seconds) {
    $Delay = [int][Math]::Ceiling(1000 * ($Target - $Clock.Elapsed.TotalSeconds))
    if ($Delay -gt 0) { Start-Sleep -Milliseconds $Delay }
    $Now = Read-PythonUsage
    $Elapsed = $Clock.Elapsed.TotalSeconds
    $External = @(Measure-PythonDeltas $Before $Now $Process.Id)
    if (Test-ExternalActivity $External) { $Monitor.external_activity_detected = $true }
    if ($Elapsed -lt $Target -or $Elapsed - $Target -gt $Thresholds.maximum_sample_lateness_seconds) {
        $Monitor.sample_coverage_pass = $false
    }
    $Monitor.samples += [ordered]@{
        target_elapsed_seconds = $Target; elapsed_seconds = [Math]::Round($Elapsed, 6)
        utc = [DateTimeOffset]::UtcNow.ToString('o'); external_python = $External
    }
    Save-Monitor
    $Before = $Now
}
$Monitor.end_utc = [DateTimeOffset]::UtcNow.ToString('o')
$Monitor.final_gpu_percent = Read-GpuPercent
$Process.WaitForExit()
$Process.Refresh()
$Monitor.exit_code = $Process.ExitCode
$Execution = if (Test-Path -LiteralPath $ExecutionPath) { Get-Content -LiteralPath $ExecutionPath -Raw | ConvertFrom-Json } else { $null }
$Monitor.execution_sha256 = if ($null -ne $Execution) { (Get-FileHash -LiteralPath $ExecutionPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$Monitor.solver_finished_within_window = ($null -ne $Execution -and $Execution.status -eq 'complete' -and
    $Execution.pid -eq $Process.Id -and $Execution.isolation_protocol_sha256 -eq $ProtocolHash -and
    [DateTimeOffset]$Execution.start_utc -ge $Started -and [DateTimeOffset]$Execution.end_utc -le $Deadline)
$Monitor.status = if ($Process.ExitCode -ne 0) { 'failed' } elseif (-not $Monitor.solver_finished_within_window -or
    -not $Monitor.sample_coverage_pass) { 'completed_outside_fixed_window' } elseif ($Monitor.external_activity_detected) {
    'completed_with_observed_external_activity'
} else { 'completed_no_sampled_external_activity' }
Save-Monitor
Write-Output "FINISH REPEAT $($Monitor.status) $($Monitor.exit_code)"
exit $Process.ExitCode
