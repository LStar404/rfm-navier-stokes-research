param(
    [Parameter(Mandatory)][ValidateSet('mid', 'high')][string]$Scale,
    [Parameter(Mandatory)][string]$OutputDir,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDir).Path
$ProtocolPath = Join-Path $OutputDirectory 'protocol.json'
$MonitorPath = Join-Path $OutputDirectory 'isolation.json'
$TimingPath = Join-Path $OutputDirectory 'timing.json'
$ExecutionPath = Join-Path $OutputDirectory 'execution.json'
$LedgerPath = Join-Path $OutputDirectory 'identity_ledger.json'
$AdmissionPath = Join-Path $OutputDirectory 'coordinator_admission.json'
$Runner = Join-Path $PSScriptRoot 'prepared_formal_timing.py'
$CoordinatorOut = Join-Path $OutputDirectory 'coordinator.stdout.log'
$CoordinatorErr = Join-Path $OutputDirectory 'coordinator.stderr.log'
$AttemptLockPath = Join-Path $OutputDirectory '.prepared_monitor.lock'
if (-not (Test-Path -LiteralPath $ProtocolPath -PathType Leaf)) { throw 'Need a frozen prepared-cohort protocol.' }
foreach ($Path in @($MonitorPath, $ExecutionPath, $LedgerPath, $AdmissionPath, $CoordinatorOut, $CoordinatorErr, $TimingPath)) {
    if (Test-Path -LiteralPath $Path) { throw 'This prepared-cohort output directory already has an attempted launch.' }
}
$Protocol = Get-Content -LiteralPath $ProtocolPath -Raw | ConvertFrom-Json
$Isolation = $Protocol.isolation
$Count = if ($Scale -eq 'mid') { 55 } else { 40 }
if ($Protocol.status -ne 'frozen' -or $Protocol.scale -ne $Scale -or $Protocol.schedule.Count -ne $Count -or
    $Isolation.sample_seconds -ne 10 -or $Isolation.preflight_windows -ne 2 -or
    $Isolation.window_seconds -lt 10 -or $Isolation.window_seconds % 10 -ne 0 -or
    @($Isolation.known_background).Count -ne 1) { throw 'The frozen prepared-cohort window or identity configuration changed.' }
$Thresholds = $Isolation.thresholds
if ($Thresholds.preflight_gpu_percent -ne 5 -or $Thresholds.background_individual_cpu_seconds -ne .5 -or
    $Thresholds.background_total_cpu_seconds -ne 1 -or $Thresholds.unexpected_python_cpu_seconds -ne .2 -or
    $Thresholds.maximum_sample_lateness_seconds -ne 2) { throw 'The registered isolation thresholds changed.' }
$ProtocolHash = (Get-FileHash -LiteralPath $ProtocolPath -Algorithm SHA256).Hash.ToLowerInvariant()
$Schema = 'win32-process-identity/v1'
$RunToken = [Guid]::NewGuid().ToString('N')
Add-Type -Path (Join-Path $PSScriptRoot 'prepared_process_snapshot.cs')

function Identity-Key($Record) {
    if ($null -eq $Record -or $Record.pid -le 0 -or $Record.birth_filetime_100ns -le 0) {
        throw 'Missing exact Windows process identity.'
    }
    return "$([int]$Record.pid):$([long]$Record.birth_filetime_100ns)"
}

function Save-AtomicJson([string]$Path, $Record, [bool]$Replace) {
    $Temporary = "$Path.tmp"
    $Bytes = [Text.Encoding]::UTF8.GetBytes(($Record | ConvertTo-Json -Depth 14) + "`n")
    $Stream = [IO.File]::Open($Temporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $Stream.Write($Bytes, 0, $Bytes.Length); $Stream.Flush($true) } finally { $Stream.Dispose() }
    [IO.File]::Move($Temporary, $Path, $Replace)
}

function Read-GpuPercent {
    $Usage = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0 -or @($Usage).Count -lt 1) { throw 'Could not read GPU utilization.' }
    $Percent = [int]($Usage | Select-Object -First 1)
    if ($Percent -lt 0 -or $Percent -gt 100) { throw 'Invalid GPU utilization sample.' }
    return $Percent
}

function Read-ProcessSnapshot {
    $From = [DateTimeOffset]::UtcNow
    $Failure = $null
    $Rows = @()
    $Unresolved = @()
    try {
        $Snapshot = [PreparedProcessSnapshot]::Capture()
        $Rows = @($Snapshot.python | ForEach-Object { $_ })
        $Unresolved = @($Snapshot.unresolved_python | ForEach-Object { $_ })
    } catch { $Failure = $_.Exception.GetType().Name }
    return [ordered]@{
        snapshot_start_utc = $From.ToString('o'); snapshot_end_utc = [DateTimeOffset]::UtcNow.ToString('o')
        snapshot_failure_type = $Failure; python = $Rows; unresolved_python = $Unresolved
    }
}

function Compare-Snapshots($Earlier, $Later, $Roles) {
    $Previous = @{}
    foreach ($Entry in $Earlier.python) { $Previous[(Identity-Key $Entry)] = $Entry }
    foreach ($Entry in $Later.python) {
        $Key = Identity-Key $Entry
        $Old = $Previous[$Key]
        $Kernel = [long]$Entry.kernel_cpu_100ns - [long]$Old.kernel_cpu_100ns
        $User = [long]$Entry.user_cpu_100ns - [long]$Old.user_cpu_100ns
        $Role = if ($Roles.ContainsKey($Key)) { $Roles[$Key].label } else { 'external_python' }
        $Index = if ($Role -eq 'cohort_child') { $Roles[$Key].index } else { $null }
        [ordered]@{
            pid = $Entry.pid; birth_filetime_100ns = $Entry.birth_filetime_100ns; label = $Role; child_index = $Index
            kernel_delta_100ns = $Kernel; user_delta_100ns = $User; cpu_delta_100ns = $Kernel + $User
            previous_identity_observed = ($null -ne $Old); exited = ($Entry.exit_filetime_100ns -ne 0)
        }
    }
}

function Test-ExternalActivity($Rows) {
    $Total = [long]0
    $Active = $false
    foreach ($Entry in $Rows) {
        if ($Entry.label -eq 'known_background') {
            $Total += $Entry.cpu_delta_100ns
            if ($Entry.cpu_delta_100ns -gt [decimal]$Thresholds.background_individual_cpu_seconds * 10000000) { $Active = $true }
        } elseif ($Entry.label -eq 'external_python' -and
                  $Entry.cpu_delta_100ns -gt [decimal]$Thresholds.unexpected_python_cpu_seconds * 10000000) { $Active = $true }
    }
    return ($Active -or $Total -gt [decimal]$Thresholds.background_total_cpu_seconds * 10000000)
}

function Complete-IdentityRoles($Ledger, $Coordinator, $Roles, $Rows) {
    $Key = Identity-Key $Coordinator
    if ($null -eq $Ledger -or $Ledger.status -cne 'complete' -or $Ledger.schema -cne $Schema -or
        $Ledger.protocol_sha256 -cne $ProtocolHash -or $Ledger.run_token -cne $RunToken -or
        (Identity-Key $Ledger.coordinator) -cne $Key -or @($Ledger.children).Count -ne $Count -or @($Rows).Count -ne $Count) {
        return $false
    }
    $CandidateRoles = @{}
    $PreviousExit = [long]0
    $Index = 0
    foreach ($Entry in $Ledger.children) {
        $ChildKey = Identity-Key $Entry
        $Row = $Rows[$Index]
        if ($Entry.index -ne $Index -or $Roles.ContainsKey($ChildKey) -or $CandidateRoles.ContainsKey($ChildKey) -or
            $Entry.parent_pid -ne $Coordinator.pid -or
            $Entry.parent_birth_filetime_100ns -ne $Coordinator.birth_filetime_100ns -or $Entry.status -cne 'exited' -or
            $Entry.launch_monotonic_ns -le $PreviousExit -or $Entry.exit_monotonic_ns -le $Entry.launch_monotonic_ns -or
            $Entry.exit_filetime_100ns -le $Entry.birth_filetime_100ns -or $null -eq $Entry.exit_kernel_cpu_100ns -or
            $null -eq $Entry.exit_user_cpu_100ns -or $Entry.exit_kernel_cpu_100ns -lt 0 -or $Entry.exit_user_cpu_100ns -lt 0 -or
            $Row.index -ne $Index -or $Row.child_pid -ne $Entry.pid -or $Row.child_birth_filetime_100ns -ne $Entry.birth_filetime_100ns -or
            $Row.ready.index -ne $Index -or $Row.ready.run_token -cne $RunToken -or (Identity-Key $Row.ready) -cne $ChildKey -or
            $Row.marker.index -ne $Index -or $Row.marker.run_token -cne $RunToken -or (Identity-Key $Row.marker) -cne $ChildKey) {
            return $false
        }
        $CandidateRoles[$ChildKey] = @{label = 'cohort_child'; index = $Index; terminal = $Entry}
        $PreviousExit = [long]$Entry.exit_monotonic_ns
        $Index++
    }
    foreach ($ChildKey in $CandidateRoles.Keys) { $Roles[$ChildKey] = $CandidateRoles[$ChildKey] }
    return $true
}

function Wait-BoundedCoordinator([Diagnostics.Process]$Worker, [int]$Milliseconds) {
    if ($Worker.WaitForExit($Milliseconds)) { return $true }
    # This handle owns the cohort's launched process tree.
    try { $Worker.Kill($true) } catch [InvalidOperationException] {
        if (-not $Worker.WaitForExit(0)) { throw }
    }
    if (-not $Worker.WaitForExit(10000)) { throw 'The prepared coordinator did not exit after its owned tree was stopped.' }
    return $false
}

$KnownKey = Identity-Key $Isolation.known_background[0]
$Roles = @{}
$Roles[$KnownKey] = @{label = 'known_background'}
$Monitor = [ordered]@{
    status = 'preflight'; scale = $Scale; protocol_sha256 = $ProtocolHash; run_token = $RunToken
    identity_schema = $Schema; window_seconds = $Isolation.window_seconds; sampling_interval_seconds = $Isolation.sample_seconds
    post_window_exit_grace_seconds = 120; coordinator_force_stopped = $false
    known_background = @($Isolation.known_background); initial_snapshot = $null; preflight = @(); samples = @()
    coordinator = $null; coordinator_exit = $null; identity_ledger_sha256 = $null
    coordinator_admission_sha256 = $null; execution_sha256 = $null; timing_sha256 = $null
    sample_coverage_pass = $true; identity_ledger_valid = $false; solver_finished_within_window = $false
    external_activity_detected = $false; child_overlap_detected = $false
    observed_child_parent_consistent = $true
    observed_child_exit_consistent = $true
    scope = 'Sampled Python image prefixes and aggregate GPU endpoints. Query-only handles retain terminal CPU counters when observed. Activity entirely between observations can be missed.'
}
$Process = $null
$Deadline = $null
$AttemptLock = [IO.File]::Open($AttemptLockPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {
    if (Test-Path -LiteralPath $MonitorPath) { throw 'The prepared-cohort output directory was already claimed.' }
    $Before = Read-ProcessSnapshot
    $Monitor.initial_snapshot = $Before
    if ($Before.snapshot_failure_type -or $Before.unresolved_python.Count -ne 0 -or
        $Before.python.Count -ne 1 -or $Before.python[0].exit_filetime_100ns -ne 0 -or
        (Identity-Key $Before.python[0]) -cne $KnownKey) {
        $Monitor.status = 'preflight_rejected'
        Save-AtomicJson $MonitorPath $Monitor $false
        throw 'Initial preflight did not observe only the frozen background Python identity.'
    }
    foreach ($Window in 1..$Isolation.preflight_windows) {
        Start-Sleep -Seconds $Isolation.sample_seconds
        $Now = Read-ProcessSnapshot
        $Classification = @(Compare-Snapshots $Before $Now $Roles)
        $Gpu = $null
        try { $Gpu = Read-GpuPercent } catch { $Monitor.preflight_gpu_error = $_.Exception.GetType().Name }
        $Monitor.preflight += [ordered]@{window = $Window; snapshot = $Now; classification = $Classification; gpu_percent = $Gpu}
        if ($Now.snapshot_failure_type -or $Now.unresolved_python.Count -ne 0 -or $Now.python.Count -ne 1 -or
            $Now.python[0].exit_filetime_100ns -ne 0 -or (Identity-Key $Now.python[0]) -cne $KnownKey -or
            $null -eq $Gpu -or $Gpu -gt $Thresholds.preflight_gpu_percent -or
            @($Classification | Where-Object {$_.kernel_delta_100ns -lt 0 -or $_.user_delta_100ns -lt 0}).Count -ne 0 -or
            (Test-ExternalActivity $Classification)) {
            $Monitor.status = 'preflight_rejected'
            Save-AtomicJson $MonitorPath $Monitor $false
            throw "Preflight window $Window rejected the prepared timing cohort before launch."
        }
        $Before = $Now
    }

    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:RFM_PREPARED_PROTOCOL_SHA256 = $ProtocolHash
    $env:RFM_PREPARED_RUN_TOKEN = $RunToken
    $env:RFM_PREPARED_ADMISSION_PATH = $AdmissionPath
    $Started = [DateTimeOffset]::UtcNow
    $Clock = [Diagnostics.Stopwatch]::StartNew()
    $Deadline = $Started.AddSeconds($Isolation.window_seconds)
    $Monitor.start_utc = $Started.ToString('o')
    $Monitor.planned_end_utc = $Deadline.ToString('o')
    $Monitor.planned_end_filetime_100ns = $Deadline.UtcDateTime.ToFileTimeUtc()
    $Monitor.initial_gpu_percent = $Gpu
    $Monitor.status = 'launching'
    Save-AtomicJson $MonitorPath $Monitor $false
    $Process = Start-Process -FilePath $Python -ArgumentList @('-B', '-u', "`"$Runner`"", '--execute',
        '--output-dir', "`"$OutputDirectory`"", '--scale', $Scale) -WindowStyle Hidden `
        -RedirectStandardOutput $CoordinatorOut -RedirectStandardError $CoordinatorErr -PassThru
    $Coordinator = [PreparedProcessSnapshot]::IdentityFromHandle($Process.Handle)
    $CoordinatorKey = Identity-Key $Coordinator
    if ($Coordinator.pid -ne $Process.Id -or $Coordinator.exit_filetime_100ns -ne 0 -or $Roles.ContainsKey($CoordinatorKey)) {
        throw 'The launched coordinator did not provide a live, distinct Win32 identity.'
    }
    $Roles[$CoordinatorKey] = @{label = 'cohort_coordinator'}
    $Admission = [ordered]@{schema = $Schema; status = 'admitted'; protocol_sha256 = $ProtocolHash; run_token = $RunToken
        coordinator = $Coordinator; admitted_utc = [DateTimeOffset]::UtcNow.ToString('o')}
    Save-AtomicJson $AdmissionPath $Admission $false
    $Monitor.coordinator_admission_sha256 = (Get-FileHash -LiteralPath $AdmissionPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $Monitor.coordinator = $Coordinator
    $Monitor.status = 'running'
    Save-AtomicJson $MonitorPath $Monitor $true
    Write-Output "START PREPARED $Scale $($Process.Id) $($Monitor.start_utc) $($Monitor.planned_end_utc)"

    for ($Target = $Isolation.sample_seconds; $Target -le $Isolation.window_seconds; $Target += $Isolation.sample_seconds) {
        $Delay = [int][Math]::Ceiling(1000 * ($Target - $Clock.Elapsed.TotalSeconds))
        if ($Delay -gt 0) { Start-Sleep -Milliseconds $Delay }
        $SampleStarted = $Clock.Elapsed.TotalSeconds
        $Now = Read-ProcessSnapshot
        $Elapsed = $Clock.Elapsed.TotalSeconds
        if ($Now.snapshot_failure_type -or $Now.unresolved_python.Count -ne 0 -or $Elapsed -lt $Target -or
            $Elapsed - $Target -gt $Thresholds.maximum_sample_lateness_seconds) { $Monitor.sample_coverage_pass = $false }
        $Monitor.samples += [ordered]@{target_elapsed_seconds = $Target; sample_started_elapsed_seconds = $SampleStarted
            elapsed_seconds = $Elapsed; snapshot = $Now}
        Save-AtomicJson $MonitorPath $Monitor $true
    }
    $Monitor.end_utc = [DateTimeOffset]::UtcNow.ToString('o')
    $Monitor.end_elapsed_seconds = $Clock.Elapsed.TotalSeconds
    try { $Monitor.final_gpu_percent = Read-GpuPercent } catch {
        $Monitor.final_gpu_percent = $null
        $Monitor.sample_coverage_pass = $false
    }
    $Monitor.coordinator_force_stopped = -not (Wait-BoundedCoordinator $Process ($Monitor.post_window_exit_grace_seconds*1000))
    $Monitor.exit_code = $Process.ExitCode
    try { $Monitor.coordinator_exit = [PreparedProcessSnapshot]::IdentityFromHandle($Process.Handle) }
    catch { $Monitor.coordinator_exit_identity_failure_type = $_.Exception.GetType().Name }

    $Ledger = $null
    if (Test-Path -LiteralPath $LedgerPath -PathType Leaf) {
        $Monitor.identity_ledger_sha256 = (Get-FileHash -LiteralPath $LedgerPath -Algorithm SHA256).Hash.ToLowerInvariant()
        try {
            $Ledger = Get-Content -LiteralPath $LedgerPath -Raw | ConvertFrom-Json
            if (Test-Path -LiteralPath $TimingPath -PathType Leaf) {
                $Monitor.timing_sha256 = (Get-FileHash -LiteralPath $TimingPath -Algorithm SHA256).Hash.ToLowerInvariant()
                $Timing = Get-Content -LiteralPath $TimingPath -Raw | ConvertFrom-Json
                $Monitor.identity_ledger_valid = $Timing.status -ceq 'complete' -and $Timing.protocol_sha256 -ceq $ProtocolHash -and
                    (Complete-IdentityRoles $Ledger $Coordinator $Roles $Timing.rows)
            }
        } catch { $Monitor.ledger_failure_type = $_.Exception.GetType().Name }
    }
    $Execution = $null
    if (Test-Path -LiteralPath $ExecutionPath -PathType Leaf) {
        $Monitor.execution_sha256 = (Get-FileHash -LiteralPath $ExecutionPath -Algorithm SHA256).Hash.ToLowerInvariant()
        try { $Execution = Get-Content -LiteralPath $ExecutionPath -Raw | ConvertFrom-Json }
        catch { $Monitor.execution_failure_type = $_.Exception.GetType().Name }
    }
    $Monitor.solver_finished_within_window = ($null -ne $Execution -and $Execution.status -ceq 'complete' -and
        $Execution.pid -eq $Coordinator.pid -and $Execution.birth_filetime_100ns -eq $Coordinator.birth_filetime_100ns -and
        $Execution.protocol_sha256 -ceq $ProtocolHash -and $Execution.run_token -ceq $RunToken -and
        $Execution.identity_ledger_sha256 -ceq $Monitor.identity_ledger_sha256 -and
        [DateTimeOffset]$Execution.start_utc -ge [DateTimeOffset]$Admission.admitted_utc -and
        [DateTimeOffset]$Execution.end_utc -ge [DateTimeOffset]$Execution.start_utc -and [DateTimeOffset]$Execution.end_utc -le $Deadline -and
        $null -ne $Monitor.coordinator_exit -and (Identity-Key $Monitor.coordinator_exit) -ceq $CoordinatorKey -and
        $Monitor.coordinator_exit.exit_filetime_100ns -ge ([DateTimeOffset]$Execution.end_utc).UtcDateTime.ToFileTimeUtc() -and
        $Monitor.coordinator_exit.exit_filetime_100ns -le $Monitor.planned_end_filetime_100ns)

    foreach ($Sample in $Monitor.samples) {
        $Classification = @(Compare-Snapshots $Before $Sample.snapshot $Roles)
        $Sample.classification = $Classification
        $Sample.external_activity_detected = Test-ExternalActivity $Classification
        if ($Sample.external_activity_detected) { $Monitor.external_activity_detected = $true }
        if (@($Classification | Where-Object {$_.label -eq 'cohort_child' -and -not $_.exited}).Count -gt 1) {
            $Monitor.child_overlap_detected = $true
        }
        foreach ($Entry in $Sample.snapshot.python) {
            $Role = $Roles[(Identity-Key $Entry)]
            if ($Role.label -eq 'cohort_child' -and $Entry.enumerated_parent_pid -ne $Coordinator.pid) {
                $Monitor.observed_child_parent_consistent = $false
                $Monitor.sample_coverage_pass = $false
            }
            if ($Role.label -eq 'cohort_child' -and $Entry.exit_filetime_100ns -ne 0 -and
                ($Entry.exit_filetime_100ns -ne $Role.terminal.exit_filetime_100ns -or
                 $Entry.kernel_cpu_100ns -ne $Role.terminal.exit_kernel_cpu_100ns -or
                 $Entry.user_cpu_100ns -ne $Role.terminal.exit_user_cpu_100ns)) {
                $Monitor.observed_child_exit_consistent = $false
                $Monitor.sample_coverage_pass = $false
            }
        }
        if (@($Classification | Where-Object {$_.kernel_delta_100ns -lt 0 -or $_.user_delta_100ns -lt 0}).Count -ne 0) {
            $Monitor.sample_coverage_pass = $false
        }
        $Before = $Sample.snapshot
    }
    $Monitor.status = if ($Monitor.coordinator_force_stopped) { 'stopped_after_post_window_grace' }
        elseif ($Monitor.exit_code -ne 0) { 'failed' }
        elseif (-not $Monitor.identity_ledger_valid) { 'completed_with_invalid_identity_ledger' }
        elseif (-not $Monitor.solver_finished_within_window -or -not $Monitor.sample_coverage_pass) { 'completed_outside_fixed_window' }
        elseif ($Monitor.child_overlap_detected) { 'completed_with_overlapping_children' }
        elseif ($Monitor.external_activity_detected) { 'completed_with_observed_external_activity' }
        else { 'completed_no_sampled_external_activity' }
    Save-AtomicJson $MonitorPath $Monitor $true
    Write-Output "FINISH PREPARED $Scale $($Monitor.status) $($Monitor.exit_code)"
    exit $Monitor.exit_code
} catch {
    if ($Monitor.status -ne 'preflight_rejected' -and ($null -ne $Monitor.initial_snapshot -or -not (Test-Path -LiteralPath $MonitorPath))) {
        $Monitor.status = 'monitor_failed'
        $Monitor.sample_coverage_pass = $false
        $Monitor.monitor_failure_type = $_.Exception.GetType().Name
        if (-not (Test-Path -LiteralPath "$MonitorPath.tmp")) {
            Save-AtomicJson $MonitorPath $Monitor (Test-Path -LiteralPath $MonitorPath)
        }
    }
    throw
} finally {
    [PreparedProcessSnapshot]::Close()
    try {
        if ($null -ne $Process) {
            $Remaining = 35000
            if ($null -ne $Deadline -and (Test-Path -LiteralPath $AdmissionPath)) {
                $StopAt = $Deadline.AddSeconds($Monitor.post_window_exit_grace_seconds)
                $Remaining = [int][Math]::Max(0, ($StopAt - [DateTimeOffset]::UtcNow).TotalMilliseconds)
            }
            try { $null = Wait-BoundedCoordinator $Process $Remaining } finally { $Process.Dispose() }
        }
    } finally {
        $AttemptLock.Dispose()
        Remove-Item -LiteralPath $AttemptLockPath
    }
}
