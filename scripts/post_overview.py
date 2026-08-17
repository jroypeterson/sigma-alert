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

NOTE — USE EMOJI SHORTCODES, NOT LITERAL EMOJI, IN A PINNED CARD.
Slack rewrites a literal emoji to its shortcode when it stores the message (a literal
pushpin comes back as `:pushpin:`), so a card built with literals can never byte-match
what Slack stored, and the change-detector that keeps this card from being reposted
every week would fire forever. Arrows and maths symbols are not emoji and round-trip
fine.
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
         "text": {"type": "plain_text", "text": ":pushpin: #stock-price-alerts — what you're looking at"}},
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
            f"• *Macro* — {', '.join(_named(macro, names, weighting))}, `30Y Mortgage` (FRED {s.MORTGAGE_SERIES_ID})"
            "  _(10Y: level + day bp + YTD bp from year-start; WTI/dollar: level + YTD %; "
            "the mortgage row is Freddie Mac PMMS — *weekly*, so its change is `w/w` and it carries its observation date)_\n"
            f"• *Treasury Curve* — {curve} _(prior close; level + day bp + YTD bp; colored bond-style: yield up = :large_red_square:)_\n"
            f"• *Credit* — {credit} _(effective yield + OAS spread from FRED; each as level + day bp; trailing `YTD: yield ±bp, OAS ±bp` labeled separately; colored by the spread move)_"
        )}},
        # SPLIT INTO TWO SECTIONS ON PURPOSE (2026-08-17). One block carrying all
        # nine groups reached 2851 of Slack's 3000 chars when the 30Y mortgage row
        # was added — 149 to spare, under this file's own <400 warning threshold.
        # Slack renders consecutive sections contiguously, so the card reads the
        # same; the cut is between the rates/credit backdrop and the equity groups
        # because that is where the list already changes subject. Do NOT merge
        # them back: the next few tickers would fail as an opaque Slack 400 at the
        # moment of posting.
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            f"• *Global Equity* — {', '.join(_named(global_eq, names, weighting))} _(country ETFs are USD, so they bundle local-equity + FX)_\n"
            f"• *Sectors* — SPDR Select Sector ETFs{_sector_wt} ({', '.join('`'+t+'`' for t in sector)})\n"
            f"• *Healthcare* — {', '.join(_named(healthcare, names, weighting))}\n"
            f"• *Tech Themes* — {', '.join(_named(tech, names, weighting))}\n"
            f"• *Commodities* — {', '.join(_named(commodity, names, weighting))}"
        )}},
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            "*Reading an alert line*\n"
            "```:large_green_square:/:large_red_square:  `TICKER` (name)  |  ±%chg  |  z = ±X.XX  |  $price  |  P% of 52w high  |  "
            "52w: $lo - $hi  |  2025: ±%  |  YTD: ±%```\n"
            ":large_green_square: up / :large_red_square: down. `2025` = prior calendar-year return; `YTD` = year-to-date "
            "return vs the prior year-end close. ETF rows carry the same pair."
        )}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": (
            "Source: `jroypeterson/sigma-alert` · regenerate this card with "
            "`python scripts/post_overview.py --post` after structural changes — it retires the old card and pins itself."
        )}]},
    ]
    return {"blocks": blocks}


SLACK_MAX_BLOCKS = 50
SLACK_MAX_SECTION_CHARS = 3000
# A section may carry `fields[]` INSTEAD of `text`, with its own limits:
# at most 10 fields, each capped at 2000 chars (not 3000). Checking only
# `text.text` let a 3001-char fields[] section pass validation and then fail as
# an opaque Slack invalid_blocks (codex 2026-07-20). A false pass in a posting
# gate is worse than no gate, because it's trusted.
SLACK_MAX_FIELD_CHARS = 2000
SLACK_MAX_SECTION_FIELDS = 10


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

        # A section's alternative `fields[]` payload, which has its own limits.
        fields = b.get("fields")
        if fields is not None:
            if b.get("type") != "section":
                problems.append(f"block {i} ({b.get('type')}) has `fields`, "
                                f"which only a section block may carry")
            if not isinstance(fields, list):
                problems.append(f"block {i} `fields` must be a list")
                fields = []
            if len(fields) > SLACK_MAX_SECTION_FIELDS:
                problems.append(f"block {i} has {len(fields)} fields, over the "
                                f"{SLACK_MAX_SECTION_FIELDS} limit")
            for j, fld in enumerate(fields):
                ftext = fld.get("text", "") or "" if isinstance(fld, dict) else ""
                worst = max(worst, len(ftext))
                if len(ftext) > SLACK_MAX_FIELD_CHARS:
                    problems.append(
                        f"block {i} field {j} is {len(ftext)} chars, over the "
                        f"{SLACK_MAX_FIELD_CHARS} field limit")
        elif b.get("type") == "section" and not isinstance(b.get("text"), dict):
            # A section with neither text nor fields is rejected by Slack.
            problems.append(f"block {i} (section) has neither `text` nor `fields`")

    headroom = SLACK_MAX_SECTION_CHARS - worst
    print(f"[overview] {len(blocks)} blocks (max {SLACK_MAX_BLOCKS}) · "
          f"largest {worst}/{SLACK_MAX_SECTION_CHARS} chars "
          f"({headroom} to spare)")
    if not problems and headroom < 400:
        print(f"[WARN] only {headroom} chars of headroom in the largest block — "
              f"adding a few more tickers will break this card. Split the group "
              f"across two section blocks before that happens.")
    return problems


CHANNEL_ID = "C0AQXUERQG4"          # #stock-price-alerts (verified 2026-08-04)
# A stable string this card always contains. Retirement matches on THIS, not on the
# card's title -- retitling a card must not orphan its predecessor (that flaw left
# #13f with two pinned cards and #macro-and-markets with three).
CARD_MARKER = "scripts/post_overview.py"


def _bot_token() -> str | None:
    """SLACK_BOT_TOKEN from the environment, else a sibling project's .env.

    This script is MANUAL / local-only -- it is in no workflow -- so leaning on a
    workspace sibling is acceptable here in a way it would not be for a CI lane. The
    screener itself still uses the webhook and is untouched.
    """
    tok = os.environ.get("SLACK_BOT_TOKEN")
    if tok:
        return tok
    here = Path(__file__).resolve().parent.parent          # sigma-alert/
    for env in (here / ".env", here.parent / "portfolio_daily" / ".env"):
        if env.exists():
            for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("SLACK_BOT_TOKEN="):
                    return line.split("=", 1)[1].strip()
    return None


def post(payload: dict, force: bool = False) -> bool:
    """Post via chat.postMessage so the card can PIN ITSELF.

    Migrated off the incoming webhook on 2026-08-04. A webhook returns no message
    `ts`, so a webhook-posted card cannot be pinned by anything, ever -- which is why
    this channel's pinned card sat untouched from 2026-06-03 while the group lists it
    describes kept changing.

    Falls back to the webhook when no bot token is available: still posts, still warns
    loudly that it cannot pin. Degrading to "published but unpinnable" beats not
    publishing, but it must never be silent.
    """
    if requests is None:
        print("[ERROR] requests not installed — cannot post.")
        return False

    blocks = payload.get("blocks") or []
    text = payload.get("text") or "sigma-alert — channel overview"
    tok = _bot_token()

    if not tok:
        webhook = os.environ.get("SLACK_WEBHOOK")
        if not webhook:
            print("[ERROR] neither SLACK_BOT_TOKEN nor SLACK_WEBHOOK set — cannot post.")
            return False
        print("[WARN] no SLACK_BOT_TOKEN — posting via the webhook, which returns no "
              "message ts, so THIS CARD CANNOT BE PINNED. Set SLACK_BOT_TOKEN to get "
              "self-pinning back.")
        resp = requests.post(webhook, json=payload, timeout=10)
        resp.raise_for_status()
        print("[OK] Overview posted (unpinnable). Pin it by hand: ⋯ → Pin to channel.")
        return True

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import slack_pin

    if not force and slack_pin.pin_is_current(tok, CHANNEL_ID, text, blocks):
        print("[OK] pinned card is already current — nothing posted.")
        return True

    retired = slack_pin.retire_own_pins(tok, CHANNEL_ID, text, marker=CARD_MARKER)
    if retired:
        print(f"[OK] retired {retired} superseded pin(s)")

    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {tok}",
                 "Content-Type": "application/json; charset=utf-8"},
        json={"channel": CHANNEL_ID, "text": text,
              "blocks": slack_pin.stamp(blocks)}, timeout=15)
    body = resp.json()
    if not body.get("ok"):
        print(f"[ERROR] chat.postMessage failed: {body.get('error')}"
              + ("  → /invite @ClaudeBot in #stock-price-alerts"
                 if body.get("error") == "not_in_channel" else ""))
        return False
    pinned = slack_pin.pin(tok, CHANNEL_ID, body.get("ts"))
    print(f"[OK] Overview posted to #stock-price-alerts (ts={body.get('ts')}); "
          f"pinned={'yes' if pinned else 'NO — pin by hand'}")
    return True


def main():
    # The dry run prints a payload full of emoji; on the Windows cp1252 console that
    # raised UnicodeEncodeError and killed the ONE command documented as the safe way
    # to inspect this card. Ask for UTF-8, fall back to escaping, never crash.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="Post the #stock-price-alerts pinned overview card")
    ap.add_argument("--post", action="store_true",
                    help="actually post to Slack (default: print the payload only)")
    ap.add_argument("--force", action="store_true",
                    help="repost even when the pinned card is already current")
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
        post(payload, force=args.force)
    else:
        out = json.dumps(payload, indent=2, ensure_ascii=False)
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            out.encode(enc, errors="strict")
        except (UnicodeEncodeError, LookupError):
            out = json.dumps(payload, indent=2, ensure_ascii=True)
        print(out)
        print("\n[DRY RUN] Pass --post to send to #stock-price-alerts.")


if __name__ == "__main__":
    main()
