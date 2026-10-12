# Waits for a training snapshot, then plays it against TaflZero in the background.
#
#   powershell -File tools/evaluate_at.ps1 -Run runs/r1 -Step 16000 [-Nodes 400 | -Movetime 15000]
#       [-Pairs 16] [-Concurrency 8] [-Leaves 1] [-Pause] [-MinSpeed 1200] [-AboveNormal] [-GpuPriority]
#
# Starts a hidden process and returns at once. The snapshot and engine library are
# copied to runs/arena first (Windows keeps loaded files locked); results go to
# runs/arena/s<step>-<budget>[-paused].jsonl and .log. With -Pause, training is
# stopped for the match and resumed afterwards: running alongside training slows
# TaflZero far more than our engine (about 280 against 600 nodes per second), which
# made one match look much better than it was.
#
# Other people's jobs do the same. With -MinSpeed, a game counts only if TaflZero
# averaged that many nodes per second and no other process used the GPU meanwhile;
# other games are played again (see SpeedRule in arena.py). The match then starts only
# while the GPU is free, and training runs until then and whenever the match has to wait.
# -AboveNormal runs both engines above normal priority, which is enough to keep
# them at full speed while other jobs fill the processor. -GpuPriority does the same
# on the GPU: the match no longer waits for a free GPU, and other jobs on it get
# almost none of it while an engine is searching.
param(
    [Parameter(Mandatory = $true)][string]$Run,
    [Parameter(Mandatory = $true)][int]$Step,
    [int]$Nodes = 400,
    [int]$Movetime = 0,
    [int]$Pairs = 16,
    [int]$Concurrency = 8,
    [int]$Leaves = 1,
    [int]$MinSpeed = 0,
    [switch]$AboveNormal,
    [switch]$GpuPriority,
    [switch]$Pause,
    [switch]$Wait
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$arena = Join-Path $root "runs\arena"
$budget = if ($Movetime -gt 0) { "t$Movetime" } else { "n$Nodes" }
$name = "s$Step-$budget" + $(if ($Pause) { "-paused" } else { "" })
$snapshot = Join-Path $root ("$Run\checkpoints\step_{0:D7}.pt" -f $Step)
$limit = if ($Movetime -gt 0) { "--movetime $Movetime" } else { "--nodes $Nodes" }

if (-not $Wait) {
    $arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath, "-Run", $Run, "-Step", $Step,
        "-Nodes", $Nodes, "-Movetime", $Movetime, "-Pairs", $Pairs, "-Concurrency", $Concurrency, "-Leaves", $Leaves,
        "-MinSpeed", $MinSpeed, "-Wait")
    if ($Pause) { $arguments += "-Pause" }
    if ($AboveNormal) { $arguments += "-AboveNormal" }
    if ($GpuPriority) { $arguments += "-GpuPriority" }
    $p = Start-Process powershell -ArgumentList $arguments -WindowStyle Hidden -PassThru
    Write-Output "waiting for $snapshot in process $($p.Id); results in runs\arena\$name.log"
    exit 0
}

while (-not (Test-Path $snapshot)) { Start-Sleep -Seconds 30 }
Start-Sleep -Seconds 5  # let the trainer finish writing
New-Item -ItemType Directory -Force (Join-Path $arena "engine") | Out-Null
$model = Join-Path $arena (Split-Path -Leaf $snapshot)
Copy-Item $snapshot $model -Force
$lib = Join-Path $arena "engine\tafl_ffi_$name.dll"
Copy-Item (Join-Path $root "target\release\tafl_ffi.dll") $lib -Force
$env:TAFL_LIB = $lib
$env:PYTHONPATH = Join-Path $root "python"
$env:RAYON_NUM_THREADS = "2"
$python = Join-Path $root ".venv\Scripts\python.exe"
$out = Join-Path $arena "$name.jsonl"
$log = Join-Path $arena "$name.log"
$summary = Join-Path $arena "$name.summary.json"
$arguments = "-m tafl.arena --model `"$model`" --pairs $Pairs $limit --concurrency $Concurrency --leaves $Leaves --out `"$out`""
if ($MinSpeed -gt 0) { $arguments += " --min-opponent-speed $MinSpeed" }
if ($AboveNormal) { $arguments += " --above-normal" }
if ($GpuPriority) { $arguments += " --gpu-priority" }
$busy = 75  # the arena's exit status when the machine stayed too busy for a fair game

# Both readings below are given a time limit, so that a counter or WMI that does not
# answer on a busy machine cannot hold up the match.
function Get-Ours {
    while ($true) {
        try {
            return @(Get-CimInstance Win32_Process -OperationTimeoutSec 120 |
                    Where-Object { $_.CommandLine -match 'tafl\.(supervise|train|arena)' })
        } catch { Start-Sleep -Seconds 30 }
    }
}

function Get-Supervisor {
    Get-Ours | Where-Object { $_.CommandLine -match 'tafl\.supervise' }
}

# Percent of the GPU in use by processes other than ours, averaged over five seconds;
# 0 if the counters give no answer, and then the arena's own checks decide.
function Get-OtherGpuLoad {
    $ours = @(Get-Ours | ForEach-Object { [int]$_.ProcessId })
    $csv = Join-Path $arena "$name.gpu.csv"
    Remove-Item $csv -ErrorAction SilentlyContinue
    $typeperf = Start-Process typeperf -WindowStyle Hidden -PassThru `
        -ArgumentList "`"\GPU Engine(*)\Utilization Percentage`" -sc 5 -y -o `"$csv`""
    if (-not $typeperf.WaitForExit(120000)) { Stop-Process -Id $typeperf.Id -Force }
    $rows = if (Test-Path $csv) { @(Import-Csv $csv) } else { @() }
    $total = 0.0
    foreach ($row in $rows) {
        foreach ($cell in $row.PSObject.Properties) {
            if ($cell.Name -match 'pid_(\d+)_' -and $ours -notcontains [int]$Matches[1] -and $cell.Value.Trim()) {
                $total += [double]$cell.Value
            }
        }
    }
    $total / [Math]::Max($rows.Count, 1)
}

# With -Pause, a run that is training now is kept training whenever the match is not
# being played, also if something else stopped it in the meantime.
$training = $Pause -and [bool](Get-Supervisor)
$stop = Join-Path $root "$Run\STOP"
function Suspend-Training {
    if ($training -and (Get-Supervisor)) {
        New-Item -ItemType File -Force $stop | Out-Null
        while (Get-Supervisor) { Start-Sleep -Seconds 10 }
    }
}
function Resume-Training {
    if ($training -and -not (Get-Supervisor)) {
        Remove-Item $stop -ErrorAction SilentlyContinue  # left behind if the trainer was killed
        # Not through a pipe: the trainer would inherit its other end, and reading it would
        # last as long as the training. That held this script from 09:14 to 21:52 on
        # 2026-10-05, when it was replaced by hand.
        $starter = Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$(Join-Path $PSScriptRoot 'start_training.ps1')`"",
            "-Run", "`"$Run`"")
        $starter.WaitForExit()
    }
}

try {
    # A match that stopped early (an engine crash, a busy machine, or an earlier run of
    # this script) continues from its recorded games; a finished one (with a summary) is
    # not replayed. It is given up after five failures in a row that recorded no game;
    # waiting for a quiet machine is not a failure.
    $crashes = 0
    while ($crashes -lt 5 -and -not (Test-Path $summary)) {
        if ($MinSpeed -gt 0 -and -not $GpuPriority -and (Get-OtherGpuLoad) -ge 15) {
            Resume-Training
            while ((Get-OtherGpuLoad) -ge 15) { Start-Sleep -Seconds 300 }
        }
        Suspend-Training
        $recorded = if (Test-Path $out) { @(Get-Content $out).Count } else { 0 }
        $resume = if (Test-Path $out) { " --resume" } else { "" }
        $n = 1  # earlier logs are kept
        while (Test-Path ("$log" + $(if ($n -gt 1) { ".$n" } else { "" }))) { $n++ }
        $suffix = if ($n -gt 1) { ".$n" } else { "" }
        # Start-Process keeps Python's UTF-8 output as it is (PowerShell redirection would re-encode it).
        $p = Start-Process -FilePath $python -ArgumentList ($arguments + $resume) -WorkingDirectory $root -NoNewWindow `
            -PassThru -RedirectStandardOutput "$log$suffix" -RedirectStandardError "$log$suffix.err"
        $null = $p.Handle  # without this the exit code is lost
        $p.WaitForExit()
        if (Test-Path $summary) { break }
        if ((Test-Path $out) -and @(Get-Content $out).Count -gt $recorded) { $crashes = 0 }
        if ($p.ExitCode -eq $busy) {
            Resume-Training
            Start-Sleep -Seconds 900
        } else {
            $crashes++
            Start-Sleep -Seconds 60
        }
    }
} finally {
    Resume-Training
}
