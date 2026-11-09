param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$OutputDirectory = Join-Path $PSScriptRoot 'switch_formal_timing_output'
$ProtocolPath = Join-Path $OutputDirectory 'protocol.json'
$MonitorPath = Join-Path $OutputDirectory 'isolation.json'
$ExecutionPath = Join-Path $OutputDirectory 'execution.json'
$Runner = Join-Path $PSScriptRoot 'switch_formal_timing.py'
if (-not (Test-Path -LiteralPath $ProtocolPath) -or (Test-Path -LiteralPath $MonitorPath) -or
    (Test-Path -LiteralPath $ExecutionPath) -or (Test-Path -LiteralPath (Join-Path $OutputDirectory 'timing.json'))) {
    throw 'Need one unused frozen formal protocol.'
}
$Protocol = Get-Content -LiteralPath $ProtocolPath -Raw | ConvertFrom-Json
$Isolation = $Protocol.isolation
if ($Protocol.status -ne 'frozen' -or $Protocol.schedule.Count -ne 60 -or $Isolation.window_seconds -ne 900 -or
    $Isolation.sample_seconds -ne 10 -or $Isolation.preflight_windows -ne 2) { throw 'Fixed-window protocol changed.' }
$Thresholds = $Isolation.thresholds
$ProtocolHash = (Get-FileHash -LiteralPath $ProtocolPath -Algorithm SHA256).Hash.ToLowerInvariant()
$Server = [int]$Isolation.known_server_pid
$Companions = @($Isolation.companion_pids)
$Allowed = @($Server) + $Companions
$AllowedCreation = @{}
$RunnerPattern = [regex]::Escape($Runner) + '"?\s+'
$CoordinatorCommand = $RunnerPattern + '--execute\s*$'
$ChildCommand = $RunnerPattern + '--child\s+(?<slot>\d+)\s*$'

function Read-PythonUsage {
    $Result = @{}
    $Records = Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" -Property ProcessId,ParentProcessId,CreationDate,UserModeTime,KernelModeTime,CommandLine
    foreach ($Entry in $Records) {
        $Result[[int]$Entry.ProcessId] = @{
            parent = [int]$Entry.ParentProcessId
            creation = $Entry.CreationDate.ToUniversalTime().Ticks
            command = $Entry.CommandLine
            cpu = ([double]$Entry.UserModeTime + [double]$Entry.KernelModeTime) * 1e-7
        }
    }
    return $Result
}

function Read-GpuPercent {
    $Usage = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { throw 'Could not read GPU utilization.' }
    return [int]($Usage | Select-Object -First 1)
}

function Save-Monitor {
    $Temporary = "$MonitorPath.tmp"
    $Monitor | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Temporary -Encoding utf8NoBOM
    Move-Item -LiteralPath $Temporary -Destination $MonitorPath -Force
}

function Measure-External($Earlier, $Later, [int]$CoordinatorId = 0, $Children = @()) {
    foreach ($ProcessId in $Later.Keys) {
        if ($ProcessId -eq $CoordinatorId -or $Children -contains $ProcessId) { continue }
        $Process = $Later[$ProcessId]
        $Same = $Earlier.ContainsKey($ProcessId) -and $Earlier[$ProcessId].creation -eq $Process.creation
        $Delta = if ($Same) { $Process.cpu - $Earlier[$ProcessId].cpu } else { $Process.cpu }
        $Known = $AllowedCreation.ContainsKey($ProcessId) -and $AllowedCreation[$ProcessId] -eq $Process.creation
        $Label = if (-not $Known) { 'python' } elseif ($ProcessId -eq $Server) { 'known_server' } else { 'companion' }
        [pscustomobject]@{pid = $ProcessId; label = $Label; cpu_seconds = [Math]::Round($Delta, 4)}
    }
}

function Test-ExternalActivity($Rows) {
    $Background = @($Rows | Where-Object { $_.label -ne 'python' })
    $Total = ($Background | Measure-Object -Property cpu_seconds -Sum).Sum
    return (@($Background | Where-Object { $_.cpu_seconds -gt $Thresholds.background_individual_cpu_seconds }).Count -ne 0 -or
        $Total -gt $Thresholds.background_total_cpu_seconds -or
        @($Rows | Where-Object { $_.label -eq 'python' -and $_.cpu_seconds -gt $Thresholds.unexpected_python_cpu_seconds }).Count -ne 0)
}

$Before = Read-PythonUsage
foreach ($ProcessId in $Allowed) {
    if (-not $Before.ContainsKey($ProcessId)) { continue }
    $Pattern = if ($ProcessId -eq $Server) { 'cli\.py.+\bserve\b' } else { '\bcompanion\b' }
    if ($Before[$ProcessId].command -notmatch $Pattern) { throw "Background identity changed: $ProcessId" }
    $AllowedCreation[$ProcessId] = $Before[$ProcessId].creation
}
if (@($Before.Keys | Where-Object { -not $AllowedCreation.ContainsKey($_) }).Count -ne 0) {
    throw 'Other Python processes remain. No formal solve was started.'
}
$Preflight = @()
foreach ($Window in 1..$Isolation.preflight_windows) {
    Start-Sleep -Seconds $Isolation.sample_seconds
    $Now = Read-PythonUsage
    $External = @(Measure-External $Before $Now)
    $Gpu = Read-GpuPercent
    if (@($External | Where-Object label -eq 'python').Count -ne 0 -or
        (Test-ExternalActivity $External) -or $Gpu -gt $Thresholds.preflight_gpu_percent) {
        throw "Preflight window $Window did not pass. No formal solve was started."
    }
    $Preflight += [ordered]@{utc = [DateTimeOffset]::UtcNow.ToString('o'); external_python = $External; gpu_percent = $Gpu}
    $Before = $Now
}

$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUNBUFFERED = '1'
$env:RFM_SWITCH_FORMAL_PROTOCOL_SHA256 = $ProtocolHash
$Started = [DateTimeOffset]::UtcNow
$Deadline = $Started.AddSeconds($Isolation.window_seconds)
$Clock = [Diagnostics.Stopwatch]::StartNew()
$Monitor = [ordered]@{
    status = 'starting'; start_utc = $Started.ToString('o'); planned_end_utc = $Deadline.ToString('o')
    window_seconds = $Isolation.window_seconds; sampling_interval_seconds = $Isolation.sample_seconds
    protocol_sha256 = $ProtocolHash; preflight = $Preflight; initial_gpu_percent = $Gpu
    allowed_background_pids = $Allowed; samples = @(); external_activity_detected = $false
    sample_coverage_pass = $true; child_overlap_detected = $false
}
$Process = Start-Process -FilePath $Python -ArgumentList @('-B', '-u', "`"$Runner`"", '--execute') -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $OutputDirectory 'coordinator.stdout.log') `
    -RedirectStandardError (Join-Path $OutputDirectory 'coordinator.stderr.log') -PassThru
$Monitor.status = 'running'
$Monitor.pid = $Process.Id
try {
    $LaunchIdentity = Get-CimInstance Win32_Process -Filter "ProcessId=$($Process.Id)" -Property ProcessId,CreationDate,CommandLine
    if ($null -eq $LaunchIdentity -or $LaunchIdentity.CommandLine -notmatch $CoordinatorCommand) {
        throw 'The launched process identity does not match the coordinator.'
    }
    $CoordinatorCreation = $LaunchIdentity.CreationDate.ToUniversalTime().Ticks
    $Monitor.coordinator_creation_utc_ticks = $CoordinatorCreation
} catch {
    $CoordinatorCreation = $null
    $Monitor.coordinator_creation_utc_ticks = $null
    $Monitor.sample_coverage_pass = $false
}
Save-Monitor
Write-Output "START FORMAL $($Process.Id) $($Monitor.start_utc) $($Monitor.planned_end_utc)"

for ($Target = $Isolation.sample_seconds; $Target -le $Isolation.window_seconds; $Target += $Isolation.sample_seconds) {
    $Delay = [int][Math]::Ceiling(1000 * ($Target - $Clock.Elapsed.TotalSeconds))
    if ($Delay -gt 0) { Start-Sleep -Milliseconds $Delay }
    $SampleStarted = $Clock.Elapsed.TotalSeconds
    $SnapshotFailure = $null
    try { $Now = Read-PythonUsage } catch {
        $SnapshotFailure = $_.Exception.GetType().Name
        $Monitor.sample_coverage_pass = $false
        $Now = $Before
    }
    $Elapsed = $Clock.Elapsed.TotalSeconds
    $OwnerPresent = ($null -ne $CoordinatorCreation -and -not $Process.HasExited -and $Now.ContainsKey($Process.Id) -and
        $Now[$Process.Id].creation -eq $CoordinatorCreation -and $Now[$Process.Id].command -match $CoordinatorCommand)
    $ChildSlots = @()
    if ($OwnerPresent) {
        foreach ($ProcessId in $Now.Keys) {
            if ($Now[$ProcessId].parent -eq $Process.Id -and $Now[$ProcessId].command -match $ChildCommand) {
                $Slot = [int]$Matches.slot
                if ($Slot -ge 0 -and $Slot -lt $Protocol.schedule.Count) {
                    $ChildSlots += [ordered]@{pid = $ProcessId; slot = $Slot}
                }
            }
        }
    }
    $Children = @($ChildSlots | ForEach-Object { $_.pid })
    $CoordinatorId = if ($OwnerPresent) { $Process.Id } else { 0 }
    $External = @(Measure-External $Before $Now $CoordinatorId $Children)
    if (Test-ExternalActivity $External) { $Monitor.external_activity_detected = $true }
    if ($Children.Count -gt 1) { $Monitor.child_overlap_detected = $true }
    if ($Elapsed -lt $Target -or $Elapsed - $Target -gt $Thresholds.maximum_sample_lateness_seconds) {
        $Monitor.sample_coverage_pass = $false
    }
    $Monitor.samples += [ordered]@{
        target_elapsed_seconds = $Target; elapsed_seconds = [Math]::Round($Elapsed, 6)
        process_snapshot_seconds = [Math]::Round($Elapsed-$SampleStarted, 6)
        snapshot_failure_type = $SnapshotFailure
        utc = [DateTimeOffset]::UtcNow.ToString('o'); cohort_coordinator_present = $OwnerPresent
        cohort_children = $Children; cohort_child_slots = $ChildSlots; external_python = $External
    }
    Save-Monitor
    $Before = $Now
}
$Monitor.end_utc = [DateTimeOffset]::UtcNow.ToString('o')
try { $Monitor.final_gpu_percent = Read-GpuPercent } catch {
    $Monitor.final_gpu_percent = $null
    $Monitor.sample_coverage_pass = $false
}
$Process.WaitForExit()
$Process.Refresh()
$Monitor.exit_code = $Process.ExitCode
$Execution = if (Test-Path -LiteralPath $ExecutionPath) { Get-Content -LiteralPath $ExecutionPath -Raw | ConvertFrom-Json } else { $null }
$Monitor.execution_sha256 = if ($null -ne $Execution) { (Get-FileHash -LiteralPath $ExecutionPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$Monitor.solver_finished_within_window = ($null -ne $Execution -and $Execution.status -eq 'complete' -and
    $Execution.pid -eq $Process.Id -and $Execution.protocol_sha256 -eq $ProtocolHash -and
    [DateTimeOffset]$Execution.start_utc -ge $Started -and [DateTimeOffset]$Execution.end_utc -le $Deadline)
$Monitor.status = if ($Process.ExitCode -ne 0) { 'failed' } elseif (-not $Monitor.solver_finished_within_window -or
    -not $Monitor.sample_coverage_pass) { 'completed_outside_fixed_window' } elseif ($Monitor.child_overlap_detected) {
    'completed_with_overlapping_children'
} elseif ($Monitor.external_activity_detected) { 'completed_with_observed_external_activity' } else { 'completed_no_sampled_external_activity' }
Save-Monitor
Write-Output "FINISH FORMAL $($Monitor.status) $($Monitor.exit_code)"
exit $Process.ExitCode
