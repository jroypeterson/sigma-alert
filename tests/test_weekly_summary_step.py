"""The weekly-summary step of sigma-close.yml, executed for real.

2026-10-12..14 every close run failed AFTER the screener had succeeded:
`git add -A cache/weekly_snapshot.json` is `fatal: pathspec` (exit 128) when the
snapshot was never written, which is every session that is not the week's last.
The health fallback then named the step's last INFO line as the cause, because
git's stderr never reached /tmp/sigma_run.log.

These tests lift the step's `run:` block out of the workflow file and execute it
under bash in a throwaway git repo (with a bare remote), so they test the shell
that CI runs rather than a description of it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "sigma-close.yml"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None,
    reason="needs bash and git",
)


def _step_script() -> str:
    """The `run: |` block of the step that runs weekly_summary.py.

    Plain text extraction, not a YAML parse: PyYAML is not a declared dependency.
    """
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    hit = next(i for i, ln in enumerate(lines) if "python scripts/weekly_summary.py" in ln)
    start = max(i for i in range(hit) if lines[i].strip() == "run: |")
    indent = len(lines[start + 1]) - len(lines[start + 1].lstrip())
    body = []
    for ln in lines[start + 1:]:
        if ln.strip() and len(ln) - len(ln.lstrip()) < indent:
            break
        body.append(ln[indent:])
    script = "\n".join(body)
    assert "python scripts/weekly_summary.py" in script
    return script


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _run_step(tmp_path: Path, writes_snapshot: bool) -> tuple[subprocess.CompletedProcess, Path, Path]:
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    _git(tmp_path, "init", "--bare", "-q", str(remote))
    _git(tmp_path, "init", "-q", str(work))
    for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
        _git(work, "config", k, v)
    (work / "cache").mkdir()
    (work / "cache" / "keep.txt").write_text("x\n", encoding="utf-8")
    _git(work, "add", "cache/keep.txt")
    _git(work, "commit", "-q", "-m", "init")
    _git(work, "remote", "add", "origin", str(remote))
    _git(work, "push", "-q", "-u", "origin", "HEAD")

    log = tmp_path / "sigma_run.log"
    # Stand-in for the python step: the real one is self-gated and writes the
    # snapshot only on the week's last session.
    fake = 'echo "[INFO] 2026-10-14 is not the week\'s last NYSE session - no weekly summary"'
    if writes_snapshot:
        fake += '; echo "{}" > cache/weekly_snapshot.json'
    script = (_step_script()
              .replace("python scripts/weekly_summary.py", f"{{ {fake}; }}")
              .replace("/tmp/sigma_run.log", log.as_posix()))
    proc = subprocess.run(["bash", "-c", script], cwd=work, capture_output=True, text=True)
    return proc, work, log


def test_step_passes_when_no_snapshot_was_written(tmp_path):
    proc, _, log = _run_step(tmp_path, writes_snapshot=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pathspec" not in (proc.stdout + proc.stderr)
    assert "nothing to commit" in log.read_text(encoding="utf-8")


def test_step_commits_and_pushes_a_written_snapshot(tmp_path):
    proc, work, _ = _run_step(tmp_path, writes_snapshot=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=work,
                          capture_output=True, text=True).stdout.strip()
    remote_refs = subprocess.run(["git", "ls-remote", "origin"], cwd=work,
                                 capture_output=True, text=True).stdout
    shown = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=work,
                           capture_output=True, text=True).stdout
    assert "cache/weekly_snapshot.json" in shown
    assert "Weekly price summary snapshot" in subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=work, capture_output=True, text=True).stdout
    assert head and head in remote_refs  # the snapshot commit reached the remote


def test_a_git_failure_reaches_the_run_log_and_fails_the_step(tmp_path):
    """The fallback names the cause from the run log, so git's error must land there."""
    proc, work, log = _run_step(tmp_path, writes_snapshot=False)  # sets up repo
    # push target gone -> push must fail loudly
    _git(work, "remote", "set-url", "origin", (tmp_path / "no_such_remote.git").as_posix())
    (work / "cache" / "weekly_snapshot.json").write_text("{}\n", encoding="utf-8")
    script = (_step_script()
              .replace("python scripts/weekly_summary.py", "true")
              .replace("/tmp/sigma_run.log", log.as_posix()))
    proc = subprocess.run(["bash", "-c", script], cwd=work, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "fatal" in log.read_text(encoding="utf-8").lower()


def test_a_failed_git_add_fails_the_step_even_when_push_succeeds(tmp_path):
    """A failed `git add` must fail the step (Codex round 1 asked; measured, it does).

    A held index.lock makes `git add` fatal while the following push is a no-op
    that succeeds; the step must still fail, or the posted marker is lost and a
    duplicate close run re-posts the weekly summary.
    """
    proc, work, log = _run_step(tmp_path, writes_snapshot=False)  # sets up repo
    (work / "cache" / "weekly_snapshot.json").write_text("{}" + chr(10), encoding="utf-8")
    (work / ".git" / "index.lock").write_text("", encoding="utf-8")
    script = (_step_script()
              .replace("python scripts/weekly_summary.py", "true")
              .replace("/tmp/sigma_run.log", log.as_posix()))
    proc = subprocess.run(["bash", "-c", script], cwd=work, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "index.lock" in log.read_text(encoding="utf-8")
