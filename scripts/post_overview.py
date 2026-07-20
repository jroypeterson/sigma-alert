#!/usr/bin/env python3
"""Build (and optionally post) the pinned reference card for #stock-price-alerts.

This is the channel's "what am I looking at" overview — the message you pin at
the top of #stock-price-alerts so the digest format is self-documenting. It is
*generated from the live source files* (watchlist segments, ETF group lists,
etf_names.json) so it never drifts from what the screener actually posts:
re-run it after any structural change and re-pin.

Usage:
    python scripts/post_overview.py            # print the payload (dry run)
    python scripts/post_overview.py --post      # post to #stock-price-alerts

Webhook: reuses SLACK_WEBHOOK (the same incoming webhook the screener uses for
the main channel). Posting via webhook cannot pin — pin the message by hand
after it lands (Slack: ⋯ → Pin to channel).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sigma_screener as s  # noqa: E402

try:
    import requests
except ImportError:  # pragma: no cover - requests is a hard dep at runtime
    requests = None


def _named(tickers, names, weighting=None):
    """Render `TICKER (friendly name, weighting)` chips, source-order preserved.

    Mirrors the screener's returns-block rows exactly (same `etf_names.json` +
    `etf_weighting.json` data), so the pinned card can't drift from the digest.
    A `None` weighting (spot asset / yield / currency basket) renders the bare
    name, same as the digest. Em-dash join for the same reason the screener uses
    one: several display names already contain commas.
    """
    weighting = weighting or {}
    out = []
    for t in tickers:
        bits = [b for b in (names.get(t), weighting.get(t)) if b]
        out.append(f"`{t}` ({' — '.join(bits)})" if bits else f"`{t}`")
    return out


def _shared_weighting(tickers, weighting):
    """Return the single weighting shared by every ticker in a group, else "".

    Lets a homogeneous group (the 11 SPDR sectors are all cap-weighted) be
    summarized once instead of repeating the label 11 times. Derived from the
    data, so it degrades to "" the moment the group stops being homogeneous.
    """
    labels = {(weighting or {}).get(t) for t in tickers}
    if len(labels) == 1:
        return next(iter(labels)) or ""
    return ""


def _ordered(path):
    """Tickers from a sources/*.txt file in file order (skip comments/blanks)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        t = line.strip().upper()
        if t and not t.startswith("#"):
            out.append(t)
    return out


def build_blocks() -> dict:
    names = s.load_etf_names()
    weighting = s.load_etf_weighting()
    macro = _ordered(s.MACRO_PATH)
    index = _ordered(s.INDEX_ETFS_PATH)
    global_eq = _ordered(s.GLOBAL_EQUITY_ETFS_PATH)
    sector = _ordered(s.SECTOR_ETFS_PATH)
    healthcare = _ordered(s.HEALTHCARE_ETFS_PATH)
    tech = _ordered(s.TECH_ETFS_PATH)
    commodity = _ordered(s.COMMODITY_ETFS_PATH)
    credit = ", ".join(
        f"`{k}` ({s.CREDIT_SERIES[k]['label']})" for k in s.CREDIT_ORDER
    )
    # The 11 SPDR sectors are homogeneous, so the weighting is stated once for
    # the group rather than repeated per chip (derived, not hardcoded).
    _shared = _shared_weighting(sector, weighting)
    _sector_wt = f", all {_shared}" if _shared else ""
    curve = ", ".join(
        f"`{k}` (FRED {s.TREASURY_CURVE_SERIES[k]})" for k in s.TREASURY_CURVE_ORDER
    )

    blocks = [
        {"type": "header",
         "text": {"type": "plain_text", "text": "📌 #stock-price-alerts — what you're looking at"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            "*Sigma screener* — flags unusual daily moves across the coverage universe. "
            "Runs *3×/weekday* (Open ~9:40, Midday ~12:35, Close ~16:25 ET; a watchdog "
            "recovers any run GitHub Actions drops). "
            "Each move is a *z-score*: `z = (today's return − μ) / σ`, where μ/σ are the "
            "trailing ~252 daily returns. So z = how many standard deviations today's move is."
        )}},
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            "*Alert tiers*\n"
            "• *2σ+ Moves* — fires on the *entire watchlist* when `|z| ≥ 2.0` (3σ+ flagged inline)\n"
            "• *1σ Moves* — fires only on names you track analytically (`Core` + the five "
            "Position lists) when `1.0 ≤ |z| < 2.0`, to keep the lower bar from being noisy"
        )}},
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            "*Subcategory order within each tier*\n"
            "Portfolio → Researching → Ready to Buy → Ready to Short → Following for Interest "
            "→ Healthcare Services → MedTech → Large Pharma → Other (Tech/SaaS/Fin/Ind/Cons/"
            "Energy/Mat/RE) → S&P 500.  _A name in multiple buckets shows once per bucket._\n"
            "*Close digest also lists* any new *52-week highs / lows* from the session, "
            "grouped by the same subcategories (with an _Uncategorized_ catch-all)."
        )}},
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            "*Index, Sector & Macro Returns* (every digest, at the bottom)\n"
            "_Each index/ETF row names its *weighting methodology* — "
            "`market-cap weighted`, `equal-weighted`, or `price-weighted` (the Dow) — "
            "so the rows are comparable. Same-industry pairs are the point: "
            "`XBI` vs `IBB` (biotech) and `XHS` vs `IHF` (health care services / "
            "providers) each pair an equal- or small/mid-tilted read against a "
            "cap-weighted one. Rows with no label (spot gold/bitcoin, the 10Y "
            "yield, the dollar basket) have no meaningful weighting._\n"
            f"• *US Indices* — {', '.join(_named(index, names, weighting))}\n"
            f"• *Macro* — {', '.join(_named(macro, names, weighting))}  _(10Y: level + day bp + YTD bp from year-start; WTI/dollar: level + YTD %)_\n"
            f"• *Treasury Curve* — {curve} _(prior close; level + day bp + YTD bp; colored bond-style: yield up = 🟥)_\n"
            f"• *Credit* — {credit} _(effective yield + OAS spread from FRED; each as level + day bp; trailing `YTD: yield ±bp, OAS ±bp` labeled separately; colored by the spread move)_\n"
            f"• *Global Equity* — {', '.join(_named(global_eq, names, weighting))} _(country ETFs are USD, so they bundle local-equity + FX)_\n"
            f"• *Sectors* — SPDR Select Sector ETFs{_sector_wt} ({', '.join('`'+t+'`' for t in sector)})\n"
            f"• *Healthcare* — {', '.join(_named(healthcare, names, weighting))}\n"
            f"• *Tech Themes* — {', '.join(_named(tech, names, weighting))}\n"
            f"• *Commodities* — {', '.join(_named(commodity, names, weighting))}"
        )}},
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            "*Reading an alert line*\n"
            "```🟩/🟥  `TICKER` (name)  |  ±%chg  |  z = ±X.XX  |  $price  |  P% of 52w high  |  "
            "52w: $lo - $hi  |  2025: ±%  |  YTD: ±%```\n"
            "🟩 up / 🟥 down. `2025` = prior calendar-year return; `YTD` = year-to-date "
            "return vs the prior year-end close. ETF rows carry the same pair."
        )}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": (
            "Source: `jroypeterson/sigma-alert` · regenerate this card with "
            "`python scripts/post_overview.py --post` after structural changes, then re-pin."
        )}]},
    ]
    return {"blocks": blocks}


SLACK_MAX_BLOCKS = 50
SLACK_MAX_SECTION_CHARS = 3000


def validate_blocks(payload: dict) -> list[str]:
    """Pre-flight the payload against Slack's structural limits.

    Without this, an oversized card fails as an opaque HTTP 400 from Slack at
    the moment you try to pin it. The weighting labels added 2026-07-20 pushed
    the largest block to ~2689 of 3000 chars, so the margin is real but thin —
    a handful of new tickers would break it. Reports headroom either way.
    """
    problems: list[str] = []
    blocks = payload.get("blocks", [])
    if len(blocks) > SLACK_MAX_BLOCKS:
        problems.append(f"{len(blocks)} blocks exceeds Slack's limit of {SLACK_MAX_BLOCKS}")

    worst = 0
    for i, b in enumerate(blocks):
        text = ""
        if isinstance(b.get("text"), dict):
            text = b["text"].get("text", "") or ""
        # A context block's payload lives in elements[]; a bare `text` field
        # there is the documented invalid_blocks 400.
        if b.get("type") == "context":
            if "text" in b:
                problems.append(f"block {i} (context) has a bare `text` field — "
                                f"Slack requires elements[]")
            for e in b.get("elements") or []:
                if isinstance(e, dict):
                    text += e.get("text", "") or ""
        worst = max(worst, len(text))
        if len(text) > SLACK_MAX_SECTION_CHARS:
            problems.append(f"block {i} ({b.get('type')}) is {len(text)} chars, "
                            f"over the {SLACK_MAX_SECTION_CHARS} limit")

    headroom = SLACK_MAX_SECTION_CHARS - worst
    print(f"[overview] {len(blocks)} blocks (max {SLACK_MAX_BLOCKS}) · "
          f"largest {worst}/{SLACK_MAX_SECTION_CHARS} chars "
          f"({headroom} to spare)")
    if not problems and headroom < 400:
        print(f"[WARN] only {headroom} chars of headroom in the largest block — "
              f"adding a few more tickers will break this card. Split the group "
              f"across two section blocks before that happens.")
    return problems


def post(payload: dict) -> bool:
    webhook = os.environ.get("SLACK_WEBHOOK")
    if not webhook:
        print("[ERROR] SLACK_WEBHOOK not set — cannot post. (Dry-run output above.)")
        return False
    if requests is None:
        print("[ERROR] requests not installed — cannot post.")
        return False
    resp = requests.post(webhook, json=payload, timeout=10)
    resp.raise_for_status()
    print("[OK] Overview posted to #stock-price-alerts. Pin it: ⋯ → Pin to channel.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Post the #stock-price-alerts pinned overview card")
    ap.add_argument("--post", action="store_true",
                    help="actually post to Slack (default: print the payload only)")
    args = ap.parse_args()
    payload = build_blocks()
    # Gate BOTH paths, so a dry run tells you the card is broken rather than
    # letting you discover it from a Slack 400 when you try to pin it.
    problems = validate_blocks(payload)
    if problems:
        print("[ERROR] Block Kit validation failed:")
        for p in problems:
            print(f"  - {p}")
        if args.post:
            print("[ERROR] Refusing to post an invalid card.")
            sys.exit(1)
    if args.post:
        post(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("\n[DRY RUN] Pass --post to send to #stock-price-alerts.")


if __name__ == "__main__":
    main()
