"""local_run.ps1 must ride through a wake-time network race before screening (board #534).

2026-10-02: the SigmaAlert-open task fired at 09:40:00, the laptop left Modern
Standby at 09:40:09, the single bare `git fetch` failed, and the open cycle ran on
the previous day's checkout - code that predates the #445 cycle stamp. The fleet
freshness check then reported "return_map.cycles.json has no stamp for 'open'"
(the file held only `midday` and `close`, which fetched fine at 12:35 / 16:25).

These tests EXECUTE the real script under PowerShell with a fake `git` on PATH -
a grep of the source would pass with the retry present but off the read path.
"""
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "local_run.ps1"

pytestmark = pytest.mark.skipif(
    os.name != "nt" or shutil.which("powershell") is None,
    reason="local_run.ps1 is a Windows Task Scheduler worker",
)


def _run(tmp_path, fetch_failures, backoff="0,0,0"):
    repo = tmp_path / "runner"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "scripts" / "local_run.ps1")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    counter = tmp_path / "fails_left"
    counter.write_text(str(fetch_failures))

    (bindir / "fake_git.py").write_text(textwrap.dedent(f"""
        import sys
        from pathlib import Path
        log = Path(r"{log}"); counter = Path(r"{counter}")
        with log.open("a") as f:
            f.write("git " + " ".join(sys.argv[1:]) + "\\n")
        if sys.argv[1:2] == ["fetch"]:
            left = int(counter.read_text())
            if left > 0:
                counter.write_text(str(left - 1))
                sys.stderr.write("fatal: unable to access: Could not resolve host: github.com\\n")
                sys.exit(128)
        sys.exit(0)
    """))
    (bindir / "git.cmd").write_text(
        f'@"{sys.executable}" "%~dp0fake_git.py" %*\r\n@exit /b %ERRORLEVEL%\r\n')
    (bindir / "fake_python.cmd").write_text(
        f'@echo screener %* >> "{log}"\r\n@exit /b 0\r\n')
    (repo / ".env").write_text(
        "SLACK_WEBHOOK=https://example.invalid/hook\n"
        "SLACK_STATUS_REPORTS_WEBHOOK=https://example.invalid/status\n"
        f"PYTHON_EXE={bindir / 'fake_python.cmd'}\n")

    env = dict(os.environ)
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    env["SIGMA_FETCH_BACKOFF"] = backoff
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(repo / "scripts" / "local_run.ps1"), "-Mode", "open"],
        capture_output=True, text=True, env=env, timeout=120)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls


def test_the_2026_10_02_shape_one_failed_fetch_still_refreshes_the_checkout(tmp_path):
    """The production incident: the first fetch dies on DNS, the network comes up
    seconds later. The run must refresh to origin/master before screening."""
    proc, calls = _run(tmp_path, fetch_failures=1)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    fetches = [c for c in calls if c.startswith("git fetch")]
    assert len(fetches) == 2
    reset_at = calls.index("git reset --hard origin/master")
    screener_at = next(i for i, c in enumerate(calls) if c.startswith("screener"))
    assert reset_at < screener_at, "the screener ran before the checkout was refreshed"


def test_a_longer_outage_inside_the_budget_is_ridden_through(tmp_path):
    proc, calls = _run(tmp_path, fetch_failures=3)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sum(c.startswith("git fetch") for c in calls) == 4
    assert "git reset --hard origin/master" in calls


def test_a_real_outage_still_screens_but_says_the_checkout_is_stale(tmp_path):
    """The alert matters more than the refresh: an outage past the budget must not
    stop the screener, and must not reset onto an un-fetched ref."""
    proc, calls = _run(tmp_path, fetch_failures=99)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sum(c.startswith("git fetch") for c in calls) == 4
    assert not any(c.startswith("git reset") for c in calls)
    assert any(c.startswith("screener") for c in calls)
    assert "stale" in (proc.stdout + proc.stderr)


def test_a_healthy_fetch_makes_exactly_one_call(tmp_path):
    proc, calls = _run(tmp_path, fetch_failures=0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sum(c.startswith("git fetch") for c in calls) == 1


def test_the_scheduler_default_budget_covers_a_dns_on_wake_window():
    """The test hook shortens the sleeps; the defaults the scheduler actually uses
    must still span the 10-30s DNS-on-wake window (5+15+30 = 50s)."""
    src = SCRIPT.read_text(encoding="ascii")
    assert "$Backoff = @(5, 15, 30)" in src
