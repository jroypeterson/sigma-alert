"""Build the #status-reports crash payload for the CI health backstop.

HEALTH_REPORTING.md 4.6: when the screener dies before its in-process heartbeat,
the workflow's `if: failure()` step posts the error line instead. That backstop
posted a fixed string — *"run crashed before the in-process heartbeat"* plus a
link — which is true and useless: it names no cause, carries no counters, and
costs a trip to the Actions log before anyone can even tell whether it is worth
looking at. It fired that way on 2026-07-31 and 2026-08-03 for what turned out
to be a **one-line `NameError`** that the message could have carried outright.

So this reads the tee'd run log and lifts the last exception line into the Slack
message. Lives in a file rather than a heredoc inside the YAML because three
workflows share it and because a shell heredoc containing JSON, backslashes and
`${}` is exactly where quoting bugs breed.

Contract: prints ONE line of JSON to stdout. Never raises — a payload builder
that crashes inside a failure handler turns a diagnosable failure into silence,
which is the whole thing this is trying to prevent. On any internal problem it
degrades to the old fixed message.

    python3 scripts/ci_health_payload.py > payload.json
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

LOG_PATH = os.environ.get("SIGMA_RUN_LOG", "/tmp/sigma_run.log")
MAX_REASON_CHARS = 300

# A Python exception's final line: `NameError: name 'mkey' is not defined`,
# `KeyError: 'AAA'`, `yfinance.exceptions.YFRateLimitError: ...`.
_EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit)\b")

_BASE = ":x: *sigma-alert - error*  |  health/v1: run crashed before the in-process heartbeat"


def extract_reason(text: str) -> str:
    """The most informative single line from a crashed run's output.

    Prefers the LAST exception line — with chained exceptions ("During handling
    of the above exception…") the final one is the one that actually killed the
    run. Falls back to the last non-empty line so a non-Python failure (an OOM
    kill, a shell error) still says something.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if _EXC_LINE.match(ln.strip()):
            return ln.strip()
    return lines[-1].strip() if lines else ""


def build_payload(reason: str, run_url: str) -> dict:
    body = _BASE
    if reason:
        # Fenced: a traceback line contains characters mrkdwn would otherwise
        # interpret, and it should read as output rather than as prose.
        body += "\n```" + reason[:MAX_REASON_CHARS] + "```"
    if run_url:
        body += f"\n<{run_url}|Actions log>"
    return {
        "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": body}}],
        "text": "sigma-alert error (crashed run)",
    }


def main() -> int:
    reason = ""
    try:
        p = pathlib.Path(LOG_PATH)
        if p.exists():
            reason = extract_reason(p.read_text(errors="replace"))
    except Exception:  # noqa: BLE001 — see the module docstring's contract
        reason = ""
    print(json.dumps(build_payload(reason, os.environ.get("RUN_URL", ""))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
