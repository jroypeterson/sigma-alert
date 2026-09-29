"""One-off repair of cache/skip_log.json from git history (M2, board #327).

Until 2026-09-29 a duplicate close run (watchdog dispatch + lagged cron — the
normal case in Sept 2026) OVERWROTE that day's skip-log entry with its own
skips: the ~750 names it found already scored. The file claimed 755-758 skips
per day where the run that published had 5-25, and the Friday report died on it.

The honest entry for each day is still in git: it is the FIRST version of the
(date, mode) entry ever committed — the publishing run commits before the
duplicate runs. This script derives that mechanically (no hand-kept commit
list) and replaces every entry whose committed-first version differs.

Default is a DRY RUN: prints, per day, current vs first-committed skip count
and the commit it would restore from. `--apply` writes the file (then commit it
yourself). Days whose current entry already equals the first one — cron-only
days, and the genuine 2026-09-28 loss — are left alone by construction.

Usage:  python scripts/repair_skip_log.py            # dry run
        python scripts/repair_skip_log.py --apply    # write cache/skip_log.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REL = "cache/skip_log.json"


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def first_committed_entries() -> dict:
    """{(date, mode): (entry, sha, committed_at)} for the first commit that
    carried each entry."""
    out: dict = {}
    log = _git("log", "--reverse", "--format=%H %cI", "--", REL).split("\n")
    for line in filter(None, log):
        sha, when = line.split(" ", 1)
        try:
            payload = json.loads(_git("show", f"{sha}:{REL}"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        for r in payload.get("runs") or []:
            key = (r.get("date"), r.get("mode"))
            if key not in out:
                out[key] = (r, sha[:7], when)
    return out


def plan(current: dict, first: dict) -> list:
    rows = []
    for r in current.get("runs") or []:
        key = (r.get("date"), r.get("mode"))
        if key not in first:
            continue
        orig, sha, when = first[key]
        if orig != r:
            rows.append((key, len(r.get("skipped") or []),
                         len(orig.get("skipped") or []), sha, when, orig))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="write the repaired file")
    args = ap.parse_args()

    path = ROOT / REL
    current = json.loads(path.read_text(encoding="utf-8"))
    rows = plan(current, first_committed_entries())
    if not rows:
        print("[OK] Every entry already equals its first-committed version.")
        return 0
    print(f"{'date':<11} {'mode':<6} {'now':>5} {'first':>5}  from")
    for (d, m), now, orig_n, sha, when, _ in rows:
        print(f"{d:<11} {m:<6} {now:>5} {orig_n:>5}  {sha} ({when})")
    total_now = sum(r[1] for r in rows)
    total_first = sum(r[2] for r in rows)
    print(f"{len(rows)} entr(ies) differ: {total_now} skips now -> {total_first} "
          f"after repair")
    if not args.apply:
        print("[DRY RUN] nothing written; re-run with --apply to write the file")
        return 0
    repl = {k: orig for k, _, _, _, _, orig in rows}
    current["runs"] = [repl.get((r.get("date"), r.get("mode")), r)
                       for r in current["runs"]]
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    print(f"[OK] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
