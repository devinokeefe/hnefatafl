# Starts (or resumes) a training run detached from the terminal, under a supervisor
# that restarts it after crashes.
# The engine library is snapshotted into the run directory first, so that rebuilding
# the workspace never touches a DLL the run has loaded.
param(
    [Parameter(Mandatory = $true)][string]$Run,
    [int]$Threads = 12,
    [string[]]$Set = @()
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$runDir = Join-Path $root $Run
New-Item -ItemType Directory -Force (Join-Path $runDir 'engine') | Out-Null
$dll = Join-Path $runDir 'engine\tafl_ffi.dll'
Copy-Item (Join-Path $root 'target\release\tafl_ffi.dll') $dll -Force
$env:TAFL_LIB = $dll
$env:PYTHONPATH = Join-Path $root 'python'
$env:RAYON_NUM_THREADS = "$Threads"
$arguments = @('-m', 'tafl.supervise', $runDir)
if ($Set.Count -gt 0) { $arguments += '--set'; $arguments += $Set }
$p = Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList $arguments `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runDir 'stdout.txt') -RedirectStandardError (Join-Path $runDir 'stderr.txt')
Write-Output "started $Run as process $($p.Id)"
