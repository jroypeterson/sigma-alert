# setup_local_runner.ps1 -- one-time bootstrap for the local sigma-alert backstop.
#
# What it does:
#   1. Clones sigma-alert to a dedicated directory OUTSIDE Dropbox (default
#      %LOCALAPPDATA%\sigma-alert-runner) so scheduled runs never touch your
#      Dropbox dev tree and Dropbox never fights the runner over the .git folder.
#   2. Installs the Python dependencies into the chosen interpreter.
#   3. Writes a gitignored .env holding the Slack webhook (and the python path).
#   4. Registers three weekday Scheduled Tasks (open / midday / close) that fire
#      within ~30 min of each market event, catch up if a start was missed
#      (laptop asleep or powered off at the trigger time), and run on battery.
#
# Re-running is safe and idempotent: it updates the clone, rewrites .env, and
# re-registers the tasks (-Force).
#
# Usage (run once, from anywhere -- e.g. your Dropbox checkout):
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup_local_runner.ps1 -SlackWebhook "https://hooks.slack.com/services/XXX/YYY/ZZZ"
#
# The webhook is the same value stored as the SLACK_WEBHOOK GitHub Actions
# secret (the #stock-price-alerts incoming webhook). It is not kept in this
# repo, so you must paste it in here once.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SlackWebhook,

    [string]$RunnerDir = (Join-Path $env:LOCALAPPDATA 'sigma-alert-runner'),

    [string]$RepoUrl = 'https://github.com/jroypeterson/sigma-alert.git',

    [string]$PythonExe = '',

    [string]$TaskPrefix = 'SigmaAlert'
)

$ErrorActionPreference = 'Stop'

# --- Resolve python -------------------------------------------------------
if (-not $PythonExe) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $PythonExe = $cmd.Source }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    $fallback = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'
    if (Test-Path $fallback) { $PythonExe = $fallback }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    Write-Error "Could not locate python.exe. Re-run with -PythonExe <full path>."
    exit 1
}
Write-Host "Using Python: $PythonExe"

# --- Clone or update the runner checkout ----------------------------------
if (Test-Path (Join-Path $RunnerDir '.git')) {
    Write-Host "Updating existing runner clone at $RunnerDir"
    Set-Location $RunnerDir
    git fetch --quiet origin master
    git reset --hard origin/master
} else {
    Write-Host "Cloning $RepoUrl -> $RunnerDir"
    git clone $RepoUrl $RunnerDir
    if ($LASTEXITCODE -ne 0) { Write-Error "git clone failed."; exit 1 }
    Set-Location $RunnerDir
}

# --- Install dependencies -------------------------------------------------
Write-Host "Installing Python dependencies..."
& $PythonExe -m pip install --quiet --upgrade -r (Join-Path $RunnerDir 'requirements.txt')

# --- Write .env (ASCII, gitignored) ---------------------------------------
$EnvFile = Join-Path $RunnerDir '.env'
$envLines = @(
    "SLACK_WEBHOOK=$SlackWebhook",
    "PYTHON_EXE=$PythonExe"
)
Set-Content -Path $EnvFile -Value $envLines -Encoding ASCII
Write-Host "Wrote $EnvFile"

# --- Register scheduled tasks ---------------------------------------------
$runScript = Join-Path $RunnerDir 'scripts\local_run.ps1'

# Mode -> local (machine-time, assumed ET) trigger, comfortably within 30 min
# of the market event. Market open 09:30 ET, close 16:00 ET. Close fires at
# 16:25 to give Yahoo time to post the official closing print.
$schedule = @(
    @{ Mode = 'open';   Time = '09:40' },
    @{ Mode = 'midday'; Time = '12:35' },
    @{ Mode = 'close';  Time = '16:25' }
)

foreach ($job in $schedule) {
    $taskName = "$TaskPrefix-$($job.Mode)"
    $argLine = "-NoProfile -ExecutionPolicy Bypass -File `"$runScript`" -Mode $($job.Mode)"
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argLine
    $trigger = New-ScheduledTaskTrigger -Weekly `
        -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
        -At ([datetime]$job.Time)
    # StartWhenAvailable: run ASAP after a missed start (asleep/off at trigger).
    # AllowStartIfOnBatteries + DontStopIfGoingOnBatteries: it's a laptop.
    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Hours 1)
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
        -Settings $settings `
        -Description "sigma-alert $($job.Mode) screener -- local backstop for the GitHub Actions cron" `
        -Force | Out-Null
    Write-Host "Registered task '$taskName' at $($job.Time) Mon-Fri (StartWhenAvailable, battery-OK)."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "  Runner clone : $RunnerDir"
Write-Host "  Tasks        : $TaskPrefix-open / -midday / -close (weekdays)"
Write-Host ""
Write-Host "These fire within ~30 min of each market event when the laptop is on, and"
Write-Host "run as soon as possible after a missed start. Duplicate alerts with the"
Write-Host "GitHub Actions runs are expected and harmless."
Write-Host ""
Write-Host "Smoke-test it now (posts a real alert to Slack):"
Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File `"$runScript`" -Mode close"
Write-Host ""
Write-Host "To remove later:"
Write-Host "  'open','midday','close' | ForEach-Object { Unregister-ScheduledTask -TaskName `"$TaskPrefix-`$_`" -Confirm:`$false }"
