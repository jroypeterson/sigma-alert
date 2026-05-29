# local_run.ps1 -- per-run worker for the local sigma-alert backstop.
#
# Invoked by Windows Task Scheduler (registered by setup_local_runner.ps1).
# This is a READ-ONLY consumer of the repo: it refreshes the disposable runner
# clone to origin/master, runs one screener mode, and exits. It never commits
# or pushes. If the GitHub Actions cron also fires the same slot you get two
# near-identical Slack alerts -- that duplication is intentional and accepted
# (the point is that the alert always goes out, even when GitHub is late or
# skips the scheduled run).
#
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File local_run.ps1 -Mode close

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('open', 'midday', 'close')]
    [string]$Mode
)

$ErrorActionPreference = 'Stop'

# Repo root = parent of the scripts dir this file lives in.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $ScriptDir
Set-Location $RepoDir

# Load .env (KEY=VALUE lines) into this process's environment.
$EnvFile = Join-Path $RepoDir '.env'
if (Test-Path $EnvFile) {
    foreach ($raw in (Get-Content $EnvFile)) {
        $line = $raw.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $idx = $line.IndexOf('=')
            $key = $line.Substring(0, $idx).Trim()
            $val = $line.Substring($idx + 1).Trim()
            Set-Item -Path ("Env:" + $key) -Value $val
        }
    }
}

if (-not $env:SLACK_WEBHOOK) {
    Write-Error "SLACK_WEBHOOK not set (expected in $EnvFile). Run setup_local_runner.ps1 first."
    exit 1
}

# Refresh the clone to match origin exactly. This checkout is disposable and
# is never edited by hand, so a hard reset is safe and avoids merge conflicts.
# It also pulls in the freshest Coverage-Manager / CI pushes (portfolio.json,
# ticker_metadata.json, distribution_cache.json, etc.) before screening.
git fetch --quiet origin master
if ($LASTEXITCODE -ne 0) {
    Write-Warning "git fetch failed; running against the current checkout."
} else {
    git reset --hard origin/master
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "git reset failed; running against the current checkout."
    }
}

# Resolve python: prefer PYTHON_EXE from .env, else 'python' on PATH.
$Python = $env:PYTHON_EXE
if (-not $Python) { $Python = 'python' }

$Screener = Join-Path $RepoDir 'scripts\sigma_screener.py'
Write-Host "[local_run] $(Get-Date -Format o) mode=$Mode python=$Python"
& $Python $Screener --mode $Mode
$rc = $LASTEXITCODE
Write-Host "[local_run] screener exited with code $rc"
exit $rc
