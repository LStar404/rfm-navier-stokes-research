param([string]$Python = 'python', [string]$Output)

$ErrorActionPreference = 'Stop'
if ($Output -and (Test-Path -LiteralPath $Output)) { throw 'The identity audit output already exists.' }
$Wrapper = Join-Path $PSScriptRoot 'run_prepared_timing_isolated.ps1'
$Native = Join-Path $PSScriptRoot 'prepared_process_snapshot.cs'
$Identity = Join-Path $PSScriptRoot 'prepared_process_identity.py'

# Load only the production helper definitions. This never enters the fixed-window wrapper.
$ParseErrors = $null
$Tokens = $null
$Ast = [Management.Automation.Language.Parser]::ParseFile($Wrapper, [ref]$Tokens, [ref]$ParseErrors)
if ($ParseErrors.Count) { throw 'The prepared timing wrapper has parse errors.' }
$Helpers = @('Identity-Key', 'Compare-Snapshots', 'Test-ExternalActivity', 'Complete-IdentityRoles')
foreach ($Name in $Helpers) {
    $Definitions = @($Ast.FindAll({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Name}, $false))
    if ($Definitions.Count -ne 1) { throw "Missing or duplicate production helper $Name." }
    . ([scriptblock]::Create($Definitions[0].Extent.Text))
}
Add-Type -Path $Native

$ChildCode = @'
import json, sys, time
sys.path.insert(0, sys.argv[1])
from prepared_process_identity import current_process_identity
print(json.dumps(current_process_identity()), flush=True)
if sys.stdin.readline().strip() != "GO":
    raise RuntimeError("Expected identity audit barrier")
deadline = time.process_time() + float(sys.argv[2])
while time.process_time() < deadline:
    pass
'@

function New-IdentityChild([string]$CpuSeconds) {
    $StartInfo = [Diagnostics.ProcessStartInfo]::new($Python)
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardInput = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    foreach ($Argument in @('-B', '-u', '-c', $ChildCode, $PSScriptRoot, $CpuSeconds)) { $StartInfo.ArgumentList.Add($Argument) }
    return [Diagnostics.Process]::Start($StartInfo)
}

function Match-ChildHandle($Process) {
    $ReadyLine = $Process.StandardOutput.ReadLine()
    if (-not $ReadyLine) { throw 'The audit child did not return its Python identity.' }
    $Self = $ReadyLine | ConvertFrom-Json
    $Owned = [PreparedProcessSnapshot]::IdentityFromHandle($Process.Handle)
    if ((Identity-Key $Self) -cne (Identity-Key $Owned) -or $Owned.pid -ne $Process.Id -or $Owned.exit_filetime_100ns -ne 0) {
        throw 'Python current-process FILETIME and Windows owned-handle FILETIME differ.'
    }
    return $Owned
}

function MonotonicNs {
    return [long]([decimal][Diagnostics.Stopwatch]::GetTimestamp() * 1000000000 / [Diagnostics.Stopwatch]::Frequency)
}

function Journal-Identity($Child) {
    $Marker = @{index = $Child.index; pid = $Child.pid; birth_filetime_100ns = $Child.birth_filetime_100ns; run_token = $RunToken}
    return @{index = $Child.index; child_pid = $Child.pid; child_birth_filetime_100ns = $Child.birth_filetime_100ns
        ready = $Marker.Clone(); marker = $Marker.Clone()}
}

$Files = @($Identity, $Native, $Wrapper, $MyInvocation.MyCommand.Path)
$Sources = [ordered]@{}
foreach ($Path in $Files) { $Sources[(Split-Path -Leaf $Path)] = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
$Result = [ordered]@{status = 'failed'; source_sha256 = $Sources; scope = 'Transport only. No fluid solve, CUDA context, or timing cohort.'}
$Children = @()
try {
    $Schema = 'win32-process-identity/v1'
    $ProtocolHash = 'a'*64
    $RunToken = 'transport-identity-audit'
    $Count = 1
    $Thresholds = @{background_individual_cpu_seconds = .5; background_total_cpu_seconds = 1; unexpected_python_cpu_seconds = .2}
    $Launch = MonotonicNs
    $Registered = New-IdentityChild '.08'
    $Children += $Registered
    $External = New-IdentityChild '.35'
    $Children += $External
    $A = Match-ChildHandle $Registered
    $B = Match-ChildHandle $External
    $OwnerProcess = [Diagnostics.Process]::GetCurrentProcess()
    try { $Owner = [PreparedProcessSnapshot]::IdentityFromHandle($OwnerProcess.Handle) } finally { $OwnerProcess.Dispose() }
    $AK = Identity-Key $A
    $BK = Identity-Key $B
    $Keys = @($AK, $BK)
    $RawBefore = [PreparedProcessSnapshot]::Capture()
    $Before = @{python = @($RawBefore.python | Where-Object {(Identity-Key $_) -cin $Keys})}
    if ($Before.python.Count -ne 2 -or @($Before.python | Where-Object {$_.exit_filetime_100ns -ne 0 -or $_.enumerated_parent_pid -ne $Owner.pid}).Count) {
        throw 'The two held audit children were not live in the native snapshot.'
    }
    foreach ($Process in $Children) { $Process.StandardInput.WriteLine('GO'); $Process.StandardInput.Flush() }
    if (-not $Registered.WaitForExit(10000) -or $Registered.ExitCode -ne 0) { throw 'The registered audit child did not exit cleanly.' }
    $Exited = MonotonicNs
    if (-not $External.WaitForExit(10000) -or $External.ExitCode -ne 0) { throw 'The external audit child did not exit cleanly.' }
    $RawAfter = [PreparedProcessSnapshot]::Capture()
    $After = @{python = @($RawAfter.python | Where-Object {(Identity-Key $_) -cin $Keys})}
    if ($After.python.Count -ne 2 -or @($After.python | Where-Object {$_.exit_filetime_100ns -le 0}).Count) {
        throw 'Terminal CPU counters were lost after process exit.'
    }
    $TerminalA = @($After.python | Where-Object {(Identity-Key $_) -ceq $AK})[0]
    $TerminalB = @($After.python | Where-Object {(Identity-Key $_) -ceq $BK})[0]
    foreach ($Index in 0..1) {
        $OwnedTerminal = [PreparedProcessSnapshot]::IdentityFromHandle($Children[$Index].Handle)
        $SampledTerminal = @($TerminalA, $TerminalB)[$Index]
        if ((Identity-Key $OwnedTerminal) -cne (Identity-Key $SampledTerminal) -or
            $OwnedTerminal.exit_filetime_100ns -ne $SampledTerminal.exit_filetime_100ns -or
            $OwnedTerminal.kernel_cpu_100ns -ne $SampledTerminal.kernel_cpu_100ns -or
            $OwnedTerminal.user_cpu_100ns -ne $SampledTerminal.user_cpu_100ns) {
            throw 'Retained terminal snapshot differs from the owned process handle.'
        }
    }
    $Child = [ordered]@{index = 0; pid = $A.pid; birth_filetime_100ns = $A.birth_filetime_100ns
        parent_pid = $Owner.pid; parent_birth_filetime_100ns = $Owner.birth_filetime_100ns
        status = 'exited'; launch_monotonic_ns = $Launch; exit_monotonic_ns = $Exited
        exit_filetime_100ns = $TerminalA.exit_filetime_100ns; exit_kernel_cpu_100ns = $TerminalA.kernel_cpu_100ns
        exit_user_cpu_100ns = $TerminalA.user_cpu_100ns}
    $Ledger = [ordered]@{schema = $Schema; status = 'complete'; protocol_sha256 = $ProtocolHash
        run_token = $RunToken; coordinator = $Owner; children = @($Child)}
    $Journal = Journal-Identity $Child
    $Roles = @{}
    $Roles[(Identity-Key $Owner)] = @{label = 'cohort_coordinator'}
    if (-not (Complete-IdentityRoles $Ledger $Owner $Roles @($Journal))) { throw 'A complete exact ledger was rejected.' }
    $Classes = @(Compare-Snapshots $Before $After $Roles)
    $RegisteredClass = @($Classes | Where-Object {(Identity-Key $_) -ceq $AK})[0]
    $ExternalClass = @($Classes | Where-Object {(Identity-Key $_) -ceq $BK})[0]
    if ($RegisteredClass.label -cne 'cohort_child' -or $RegisteredClass.child_index -ne 0 -or $RegisteredClass.cpu_delta_100ns -le 0 -or
        $ExternalClass.label -cne 'external_python' -or $ExternalClass.cpu_delta_100ns -le 2000000 -or
        (Test-ExternalActivity @($RegisteredClass)) -or -not (Test-ExternalActivity @($ExternalClass))) {
        throw 'Exact post-exit cohort and external classifications differ.'
    }

    $Reused = @{python = @(@{pid = $A.pid; birth_filetime_100ns = $A.birth_filetime_100ns + 1
        kernel_cpu_100ns = 500000; user_cpu_100ns = 1700000; exit_filetime_100ns = 0})}
    $ReusedClass = @(Compare-Snapshots $After $Reused $Roles)[0]
    if ($ReusedClass.label -cne 'external_python' -or $ReusedClass.previous_identity_observed -or
        $ReusedClass.cpu_delta_100ns -ne 2200000 -or -not (Test-ExternalActivity @($ReusedClass))) {
        throw 'Synthetic PID reuse inherited a prior identity or a prior CPU count.'
    }
    $Count = 2
    $DuplicateRoles = @{}
    $Ledger.children = @($Child, $Child)
    if ((Complete-IdentityRoles $Ledger $Owner $DuplicateRoles @($Journal, $Journal)) -or $DuplicateRoles.Count) {
        throw 'A duplicate ledger entry was accepted or partially applied.'
    }
    $OverlapRoles = @{}
    $Other = [ordered]@{index = 1; pid = $B.pid; birth_filetime_100ns = $B.birth_filetime_100ns
        parent_pid = $Owner.pid; parent_birth_filetime_100ns = $Owner.birth_filetime_100ns; status = 'exited'
        launch_monotonic_ns = $Launch + 1; exit_monotonic_ns = $Exited + 1
        exit_filetime_100ns = $TerminalB.exit_filetime_100ns; exit_kernel_cpu_100ns = $TerminalB.kernel_cpu_100ns
        exit_user_cpu_100ns = $TerminalB.user_cpu_100ns}
    $Ledger.children = @($Child, $Other)
    $OtherJournal = Journal-Identity $Other
    if ((Complete-IdentityRoles $Ledger $Owner $OverlapRoles @($Journal, $OtherJournal)) -or $OverlapRoles.Count) {
        throw 'Overlapping registered child intervals were accepted or partially applied.'
    }
    $Other.launch_monotonic_ns = $Exited + 1
    $Other.exit_monotonic_ns = $Exited + 2
    $OtherJournal.ready.birth_filetime_100ns++
    $MismatchRoles = @{}
    if ((Complete-IdentityRoles $Ledger $Owner $MismatchRoles @($Journal, $OtherJournal)) -or $MismatchRoles.Count) {
        throw 'A nonmatching READY identity was accepted or partially applied.'
    }
    $OtherJournal.ready.birth_filetime_100ns--
    $OtherJournal.marker.birth_filetime_100ns++
    if ((Complete-IdentityRoles $Ledger $Owner $MismatchRoles @($Journal, $OtherJournal)) -or $MismatchRoles.Count) {
        throw 'A nonmatching RETURN identity was accepted or partially applied.'
    }

    $Result.status = 'passed'
    $Result.python_and_native_identity_exact = $true
    $Result.owned_and_retained_terminal_identity_and_counters_exact = $true
    $Result.raw_controlled_before = $Before.python
    $Result.raw_controlled_after = $After.python
    $Result.classifications = $Classes
    $Result.synthetic_pid_reuse = $ReusedClass
    $Result.duplicate_ledger_rejected = $true
    $Result.overlapping_ledger_rejected = $true
    $Result.mismatched_ready_and_return_rejected_without_partial_exemption = $true
} catch {
    $Result.failure = $_.Exception.Message
} finally {
    [PreparedProcessSnapshot]::Close()
    foreach ($Process in $Children) {
        $Process.StandardInput.Close()
        if (-not $Process.WaitForExit(10000)) { $Process.Kill(); $Process.WaitForExit() }
        $Process.Dispose()
    }
}
$Json = ($Result | ConvertTo-Json -Depth 12) + "`n"
if ($Output) {
    $Stream = [IO.File]::Open($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output),
        [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $Bytes = [Text.Encoding]::UTF8.GetBytes($Json); $Stream.Write($Bytes, 0, $Bytes.Length) }
    finally { $Stream.Dispose() }
    Write-Output $Result.status
} else { Write-Output $Json }
if ($Result.status -cne 'passed') { exit 1 }
