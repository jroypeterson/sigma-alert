"""Post a weekly skip-report summary to Slack.

Reads `cache/skip_log.json` (written by each EOD close run) and produces a
Slack Block Kit message summarizing the trailing 7 close-mode runs:

  - Chronic skips — tickers skipped on every run of the window (needs operator
    attention: delisted, renamed, or permanently broken in yfinance).
  - Reason breakdown — aggregate counts by skip reason.
  - Unresolved at week's end — tickers in the most recent run's skipped list.
  - Daily skip timeline — count of skips per trading day.

Fires via `.github/workflows/sigma-weekly-skip-report.yml` on Friday evening,
after the Friday close run has landed its updated skip_log.json commit.

The message goes to the `SLACK_STATUS_REPORTS_WEBHOOK` channel (#status-reports),
not the main sigma alerts channel. The message is always sent, even if nothing
was skipped, so absence of a report means the workflow didn't run.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent.parent
SKIP_LOG_PATH = ROOT / "cache" / "skip_log.json"
WATCHLIST_PATH = ROOT / "watchlist.txt"

ET = ZoneInfo("America/New_York")
WINDOW_DAYS = 7


def now_et() -> datetime:
    return datetime.now(ET)


def load_skip_log() -> dict | None:
    """Return the parsed skip log payload, or None if missing/unreadable."""
    if not SKIP_LOG_PATH.exists():
        return None
    try:
        with open(SKIP_LOG_PATH) as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            return data
        return None
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not read skip_log.json: {e}")
        return None


def load_watchlist_size() -> int:
    """Count the current watchlist — used for skip-rate calculations."""
    if not WATCHLIST_PATH.exists():
        return 0
    n = 0
    with open(WATCHLIST_PATH) as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                n += 1
    return n


def filter_window(runs: list[dict], window_days: int = WINDOW_DAYS) -> list[dict]:
    """Return only close-mode runs from the trailing window, sorted by date."""
    cutoff = (now_et().date() - timedelta(days=window_days)).strftime("%Y-%m-%d")
    out = [
        r for r in runs
        if r.get("mode") == "close" and r.get("date", "") >= cutoff
    ]
    out.sort(key=lambda r: r.get("date", ""))
    return out


def compute_stats(window_runs: list[dict], watchlist_size: int) -> dict:
    """Derive the stats we render. Keeps the Slack formatter dumb."""
    if not window_runs:
        return {
            "run_count": 0,
            "chronic": [],
            "reason_breakdown": Counter(),
            "unresolved": [],
            "daily_timeline": [],
            "total_skip_events": 0,
            "avg_skip_rate_pct": None,
            "window_start": None,
            "window_end": None,
        }

    # Per-ticker skip count across the window.
    skip_count_by_ticker: Counter = Counter()
    reasons_by_ticker: dict[str, set[str]] = defaultdict(set)
    reason_counter: Counter = Counter()
    daily_counts: list[tuple[str, int]] = []
    total_events = 0

    for run in window_runs:
        skipped = run.get("skipped") or []
        daily_counts.append((run.get("date", ""), len(skipped)))
        total_events += len(skipped)
        for event in skipped:
            ticker = event.get("ticker")
            reason = event.get("reason", "unknown")
            if not ticker:
                continue
            skip_count_by_ticker[ticker] += 1
            reasons_by_ticker[ticker].add(reason)
            reason_counter[reason] += 1

    # Chronic = skipped in every run in the window.
    run_count = len(window_runs)
    chronic = sorted(
        [
            {
                "ticker": t,
                "reasons": sorted(reasons_by_ticker[t]),
                "count": skip_count_by_ticker[t],
            }
            for t, n in skip_count_by_ticker.items()
            if n == run_count
        ],
        key=lambda x: x["ticker"],
    )

    # Most recent run's skipped list = "still broken as of today".
    latest = window_runs[-1]
    unresolved = sorted(
        [
            {"ticker": e.get("ticker", ""), "reason": e.get("reason", "unknown")}
            for e in (latest.get("skipped") or [])
            if e.get("ticker")
        ],
        key=lambda x: x["ticker"],
    )

    # Skip rate = mean over runs of (skipped / watchlist_size). Only meaningful
    # if we know the watchlist size.
    avg_skip_rate = None
    if watchlist_size > 0:
        rates = [len(r.get("skipped") or []) / watchlist_size for r in window_runs]
        avg_skip_rate = sum(rates) / len(rates) * 100

    return {
        "run_count": run_count,
        "chronic": chronic,
        "reason_breakdown": reason_counter,
        "unresolved": unresolved,
        "daily_timeline": daily_counts,
        "total_skip_events": total_events,
        "avg_skip_rate_pct": avg_skip_rate,
        "window_start": window_runs[0].get("date"),
        "window_end": window_runs[-1].get("date"),
    }


# Human-friendly labels for the skip reason codes emitted by sigma_screener.py.
REASON_LABELS = {
    "insufficient_history": "Insufficient history (<32 bars)",
    "distribution_nan": "Degenerate distribution",
    "fallback_insufficient": "Fallback retry: insufficient data",
    "fallback_exception": "Fallback retry: exception raised",
    "unknown": "Unknown",
}


def format_slack_payload(stats: dict, watchlist_size: int) -> dict:
    """Build the Block Kit payload. Always produces a message — even the
    no-skips case renders explicitly so the channel sees the workflow ran."""
    now = now_et()
    date_str = now.strftime("%Y-%m-%d")

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Sigma Weekly Skip Report — {date_str}",
            },
        }
    ]

    if stats["run_count"] == 0:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "No close-mode runs found in the trailing "
                    f"{WINDOW_DAYS} days. Either the screener hasn't run yet "
                    "or `cache/skip_log.json` is missing."
                ),
            },
        })
        return {"blocks": blocks}

    # Header context line
    window_label = f"{stats['window_start']} → {stats['window_end']}"
    rate_part = (
        f"  |  avg skip rate: {stats['avg_skip_rate_pct']:.2f}%"
        if stats["avg_skip_rate_pct"] is not None
        else ""
    )
    summary_line = (
        f"*Window:* {window_label}  "
        f"|  *Runs:* {stats['run_count']}  "
        f"|  *Total skip events:* {stats['total_skip_events']}  "
        f"|  *Watchlist:* {watchlist_size}{rate_part}"
    )
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": summary_line},
    })

    # Overall "clean week" short-circuit — still include timeline below.
    if stats["total_skip_events"] == 0:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":white_check_mark: No tickers were skipped in any run this week.",
            },
        })

    blocks.append({"type": "divider"})

    # Chronic skips — tickers skipped in every run.
    chronic = stats["chronic"]
    if chronic:
        lines = [":red_circle: *Chronic skips* — skipped on every run this week"]
        for c in chronic:
            reasons = ", ".join(c["reasons"])
            lines.append(f"• *{c['ticker']}* — {reasons}")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })
    else:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":red_circle: *Chronic skips:* none",
            },
        })

    # Reason breakdown.
    if stats["reason_breakdown"]:
        lines = [":bar_chart: *Skip reason breakdown* (all events in window)"]
        for reason, count in stats["reason_breakdown"].most_common():
            label = REASON_LABELS.get(reason, reason)
            lines.append(f"• {label}: *{count}*")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })

    # Unresolved at week's end — most recent run's skipped list.
    unresolved = stats["unresolved"]
    if unresolved:
        lines = [
            f":warning: *Unresolved at week's end* ({stats['window_end']}) — "
            f"{len(unresolved)} ticker(s) skipped in the most recent run"
        ]
        for u in unresolved:
            label = REASON_LABELS.get(u["reason"], u["reason"])
            lines.append(f"• *{u['ticker']}* — {label}")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })
    else:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":white_check_mark: *Unresolved at week's end:* none — most recent run was clean.",
            },
        })

    # Daily timeline — compact single-line.
    if stats["daily_timeline"]:
        timeline = "  ".join(
            f"{d}: {c}" for d, c in stats["daily_timeline"]
        )
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"*Daily skip counts:*  {timeline}",
                }
            ],
        })

    return {"blocks": blocks}


def send_slack(payload: dict, webhook_url: str | None) -> int:
    """Post to Slack. Returns 0 on success, 1 on failure. Falls back to
    stdout dump if the webhook env var isn't set — useful for local testing."""
    if not webhook_url:
        print("[ERROR] SLACK_STATUS_REPORTS_WEBHOOK not set; dumping payload:")
        print(json.dumps(payload, indent=2))
        return 1
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        print("[OK] Weekly skip report sent to Slack")
        return 0
    except requests.RequestException as e:
        print(f"[ERROR] Slack webhook failed: {e}")
        print("[FALLBACK] Payload:")
        print(json.dumps(payload, indent=2))
        return 1


def main() -> int:
    payload_raw = load_skip_log()
    watchlist_size = load_watchlist_size()

    runs = []
    if payload_raw:
        runs = filter_window(payload_raw.get("runs") or [])

    stats = compute_stats(runs, watchlist_size)
    print(
        f"[INFO] Window: {stats['window_start']} → {stats['window_end']} "
        f"| runs={stats['run_count']} | total_events={stats['total_skip_events']} "
        f"| chronic={len(stats['chronic'])} | unresolved={len(stats['unresolved'])}"
    )

    payload = format_slack_payload(stats, watchlist_size)
    webhook = os.environ.get("SLACK_STATUS_REPORTS_WEBHOOK")
    return send_slack(payload, webhook)


if __name__ == "__main__":
    sys.exit(main())
