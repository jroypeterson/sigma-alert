"""
Sigma screener: flags stocks with 2+ standard deviation price moves
against their trailing 52-week daily return distribution.

Runs in three modes:
  --mode open   : compares today's open to prior close (gap detection)
  --mode midday : compares current price to prior close (intraday check)
  --mode close  : compares today's close to prior close (EOD move)

Price data:
  Uses yfinance with auto_adjust=True (the default since yfinance 0.2.x).
  This means Close prices reflect stock splits and dividends, which is the
  correct basis for computing daily return distributions. Raw/unadjusted
  prices would produce spurious sigma alerts on ex-dividend and split dates.
"""

import argparse
import json
import os
import re
import sys
import time
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
WATCHLIST_PATH = ROOT / "watchlist.txt"
CACHE_PATH = ROOT / "cache" / "distribution_cache.json"
SKIP_LOG_PATH = ROOT / "cache" / "skip_log.json"
METADATA_PATH = ROOT / "ticker_metadata.json"
MISSING_METADATA_PATH = ROOT / "cache" / "missing_metadata.json"
SP500_PATH = ROOT / "sources" / "sp500.txt"
SP500_NAMES_PATH = ROOT / "sources" / "sp500_names.json"
SECTOR_ETFS_PATH = ROOT / "sources" / "sector_etfs.txt"
INDEX_ETFS_PATH = ROOT / "sources" / "index_etfs.txt"
HEALTHCARE_ETFS_PATH = ROOT / "sources" / "healthcare_etfs.txt"
TECH_ETFS_PATH = ROOT / "sources" / "tech_etfs.txt"
MACRO_PATH = ROOT / "sources" / "macro.txt"
ETF_NAMES_PATH = ROOT / "sources" / "etf_names.json"
# Personal trading state pushed by Coverage Manager's weekly sigma_export step.
# Owned by Coverage Manager — do NOT edit by hand in this repo.
#
# Two files since 2026-05-03 (Coverage Manager Phase B):
#   portfolio.json    — names the user owns (Position == "Portfolio")
#   researching.json  — names the user is building a thesis on (Position == "Researching")
#
# Legacy core_watchlist.json is the union of the two and is still pushed for
# back-compat during the migration cycle. Once this screener is fully
# migrated to portfolio.json + researching.json, Coverage Manager will stop
# pushing core_watchlist.json.
CORE_WATCHLIST_PATH = ROOT / "core_watchlist.json"  # DEPRECATED — see below
PORTFOLIO_PATH = ROOT / "portfolio.json"
RESEARCHING_PATH = ROOT / "researching.json"
# Three additional position files added 2026-05-11 when Coverage Manager's
# Position taxonomy expanded from {Portfolio, Researching} to five values.
# All three render as their own Slack subcategories and are eligible for the
# 1σ alert tier on the same footing as Portfolio / Researching.
FOLLOWING_PATH = ROOT / "following_for_interest.json"
READY_TO_BUY_PATH = ROOT / "ready_to_buy.json"
READY_TO_SHORT_PATH = ROOT / "ready_to_short.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")

LOOKBACK_DAYS = 252          # ~1 trading year
SIGMA_THRESHOLD = 2.0
ONE_SIGMA_THRESHOLD = 1.0
THREE_SIGMA = 3.0

# Trailing window kept in cache/skip_log.json. Coverage Manager's weekly
# report reads trailing 7 days; keep 30 days so there's headroom for a
# longer-window view without growing the file unbounded.
SKIP_LOG_RETENTION_DAYS = 30

# 1σ tier eligibility — see `_is_one_sigma_eligible` below.
# 2σ+ alerts fire on the entire watchlist regardless of sector.
# 1σ used to be sector-based (Healthcare Services / MedTech / Other), but
# Coverage Manager's 2026-05-03 taxonomy expansion split "Other" into seven
# explicit sectors (Tech / Financials / Industrials / etc.), making the old
# gate brittle. The replacement gates 1σ on attention-based criteria from
# Coverage Manager: Core column == "Y" (per ticker_metadata.json schema v3),
# or membership in Portfolio / Researching.
SECTORS_GROUPED_AS_OTHER = frozenset({
    "Tech", "SaaS", "Financials", "Industrials",
    "Consumer", "Energy", "Materials", "Real Estate",
})


def _is_one_sigma_eligible(meta: dict, ticker: str,
                           portfolio_set, researching_set,
                           following_set=None, ready_to_buy_set=None,
                           ready_to_short_set=None) -> bool:
    """Return True if `ticker` is eligible for the 1σ alert tier.

    Triggers on:
      - Coverage Manager `Core` flag == "Y" (read from ticker_metadata.json
        schema v3+; falls back to False on older snapshots that lack the
        field), OR
      - ticker is in any of the five Position lists from Coverage Manager:
        Portfolio (held), Researching (active thesis), Ready to Buy or
        Ready to Short (thesis complete; waiting for entry trigger), or
        Following for Interest (passive earnings/signal tracking).

    2σ+ alerts always fire regardless; this gate only restricts the lower
    1σ tier so the noisier alerts only surface for names you care about.

    The three trailing set arguments default to None for back-compat with
    older callers / tests that only pass portfolio + researching.
    """
    if portfolio_set and ticker in portfolio_set:
        return True
    if researching_set and ticker in researching_set:
        return True
    if following_set and ticker in following_set:
        return True
    if ready_to_buy_set and ticker in ready_to_buy_set:
        return True
    if ready_to_short_set and ticker in ready_to_short_set:
        return True
    return (meta or {}).get("core", "").strip().upper() == "Y"

# Decision: using 400 calendar days for the yfinance download window.
# 252 trading days ≈ 365 calendar days, but we add buffer for holidays
# and weekends to ensure we always have enough data.
CALENDAR_DOWNLOAD_DAYS = 400


def now_et() -> datetime:
    """Return the current time in America/New_York, timezone-aware."""
    return datetime.now(ET)


def today_et() -> datetime:
    """Return today's date in ET."""
    return now_et().date()


# Common corporate suffixes stripped from company names for compact display
# in the 52-week high/low list. Order doesn't matter — pattern is rebuilt
# from the full list with word-boundary anchoring.
_COMPANY_SUFFIXES = [
    "Incorporated", "Inc",
    "Corporation", "Corp",
    "Limited", "Ltd",
    "PLC",
    "Holdings", "Holding",
    "Group",
    "Company", "Co",
    "AG", "SE", "SA", "NV", "BV", "AB", "ASA", "AS", "OYJ",
    "KGaA", "GmbH",
    "LLC", "LLP", "LP",
    "Tbk", "PT", "Bhd", "JSC", "OAO", "OJSC", "PJSC",
]
_SUFFIX_PATTERN = re.compile(
    r"[\s,]*\b(?:" + "|".join(_COMPANY_SUFFIXES) + r")\.?$",
    re.IGNORECASE,
)
# Catches suffixes glued onto the root word with no space, e.g.
# "AptarGroup" -> "Aptar", "MicroStrategyHoldings" -> "MicroStrategy".
_GLUED_SUFFIX_PATTERN = re.compile(
    r"(?<=[a-z])(?:Group|Holdings|Holding)$"
)


def short_company_name(name: str) -> str:
    """Strip common corporate suffixes for compact display.

    "Apple Inc." -> "Apple"
    "UnitedHealth Group Inc" -> "UnitedHealth"
    "Fresenius Medical Care AG" -> "Fresenius Medical Care"
    "JPMorgan Chase & Co" -> "JPMorgan Chase"

    Falls back to the original string if stripping would empty it.
    """
    if not name:
        return ""
    s = str(name).strip()
    # Remove "(publ)" annotations anywhere
    s = re.sub(r"\s*\(publ\)\s*", " ", s, flags=re.IGNORECASE).strip()
    # Strip suffixes iteratively to handle "UnitedHealth Group Inc" -> drop
    # "Inc" then drop "Group" on the next pass.
    prev = None
    while s != prev:
        prev = s
        s = _SUFFIX_PATTERN.sub("", s).strip()
        s = _GLUED_SUFFIX_PATTERN.sub("", s)
    # Trim trailing punctuation left over from "& Co", "Ltd.", etc.
    s = s.rstrip(",.& ")
    return s or name


def load_watchlist() -> list[str]:
    """Read tickers from watchlist.txt, one per line, skip blanks/comments."""
    tickers = []
    with open(WATCHLIST_PATH) as f:
        for line in f:
            t = line.strip().upper()
            if t and not t.startswith("#"):
                tickers.append(t)
    return tickers


def load_sp500_set() -> set[str]:
    """Load S&P 500 tickers from sources/sp500.txt into a set for membership checks."""
    if not SP500_PATH.exists():
        return set()
    out = set()
    with open(SP500_PATH) as f:
        for line in f:
            t = line.strip().upper()
            if t and not t.startswith("#"):
                out.add(t)
    return out


def load_sp500_names() -> dict:
    """Load `{TICKER: short company name}` fallback for S&P 500 tickers not in
    ticker_metadata.json. Coverage Manager only maintains metadata for the
    healthcare/MedTech/PA universe, so most S&P 500 names come from this file
    (populated from Wikipedia by refresh_sp500.py). Missing file is not fatal.
    """
    if not SP500_NAMES_PATH.exists():
        return {}
    try:
        with open(SP500_NAMES_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not read sp500_names.json: {e}")
        return {}
    if not isinstance(data, dict):
        return {}
    return {t.upper(): str(n) for t, n in data.items() if n}


def _load_ticker_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            t = line.strip().upper()
            if t and not t.startswith("#"):
                out.add(t)
    return out


def load_sector_etfs() -> set[str]:
    """Load sector ETF tickers (XLE/XLF/etc) from sources/sector_etfs.txt."""
    return _load_ticker_set(SECTOR_ETFS_PATH)


def load_index_etfs() -> set[str]:
    """Load broad-market index ETFs (SPYM/DIA/QQQ) from sources/index_etfs.txt.

    These render above sector ETFs in the Slack "Index & Sector Returns" block.
    """
    return _load_ticker_set(INDEX_ETFS_PATH)


def load_healthcare_etfs() -> set[str]:
    """Load healthcare sub-sector ETFs + the ^DRG pharma index
    (XBI/IBB/IHI/XHS/PPH/IHE/^DRG) from sources/healthcare_etfs.txt.

    These render under a `_Healthcare_` sub-header below `_Sectors_` in
    the Slack "Index & Sector Returns" block.
    """
    return _load_ticker_set(HEALTHCARE_ETFS_PATH)


def load_tech_etfs() -> set[str]:
    """Load tech-theme ETFs (MAGS/SMH/SOXX/IGV/DTCR/AIQ) from sources/tech_etfs.txt.

    Regular total-return ETFs that render under a `_Tech Themes_` sub-header in
    the Slack returns block (below `_Healthcare_`). Like the other ETF groups
    they carry prior-year/YTD period returns; they differ from `_Macro_` only in
    grouping/label.
    """
    return _load_ticker_set(TECH_ETFS_PATH)


def load_macro() -> set[str]:
    """Load macro / cross-asset tickers (^TNX/DX-Y.NYB/CL=F) from sources/macro.txt.

    These render under a `_Macro_` sub-header at the TOP of the Slack "Index &
    Sector Returns" block. They are unioned into `etf_set` so they go through
    the same download path and inherit the ETF exemptions (no alert, no
    missing-metadata flag), but are rendered by `_format_macro_line` rather than
    `_format_etf_line` (a bond yield is a level, not a price).
    """
    return _load_ticker_set(MACRO_PATH)


# Per-ticker render style for the _Macro_ sub-group. `yield` shows the level as
# a percent plus a basis-point change; `level` shows a bare index level plus a
# %change; `price` shows a $ price plus a %change. All three append the z-score.
# MACRO_ORDER fixes the display order (rates -> FX -> commodity); z-sorting
# across asset classes would be meaningless for three unlike instruments.
MACRO_STYLE = {
    "^TNX": "yield",
    "DX-Y.NYB": "level",
    "CL=F": "price",
}
MACRO_ORDER = ["^TNX", "DX-Y.NYB", "CL=F"]


def load_etf_names() -> dict:
    """Load `{TICKER: friendly name}` for index + sector ETFs.

    Coverage Manager doesn't maintain metadata for ETFs, so we keep the
    display names in this repo. Merged into the metadata dict at startup
    so the standard rendering path picks them up.
    """
    if not ETF_NAMES_PATH.exists():
        return {}
    try:
        with open(ETF_NAMES_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not read etf_names.json: {e}")
        return {}
    if not isinstance(data, dict):
        return {}
    return {t.upper(): str(n) for t, n in data.items() if n}


# Subcategory layout within each sigma tier. Order = render order.
# A ticker can appear in multiple subcategories (shown once per match).
# Alerts that match no bucket are dropped (mainly: Biotech / Specialty
# Pharma names that aren't Core / any Position list / S&P 500).
#
# Portfolio and Researching replaced the prior single "Core Watchlist"
# subcategory on 2026-05-03 (Coverage Manager Phase C). Held names render
# under "Portfolio"; thesis-building names render under "Researching".
#
# 2026-05-06: Large Pharma added (subsector == "Large Pharma"); the legacy
# "Other/PA" sector bucket dropped (CM no longer tags any row sector="Other"
# after the 2026-05-03 taxonomy expansion). Replacement bucket explicitly
# lists the seven sectors that absorbed "Other": Tech, SaaS, Financials,
# Industrials, Consumer, Energy, Materials, Real Estate.
#
# 2026-05-11: Coverage Manager's Position taxonomy expanded from 2 values
# to 5. Three new Position-derived subcategories render between Researching
# and the sector buckets, ordered by closeness-to-action: Ready to Buy
# (long thesis complete; waiting for trigger), Ready to Short (short
# thesis complete; waiting for trigger), Following for Interest (passive
# earnings/signal tracking; no intent to trade).
SUBCATEGORIES = [
    ("Portfolio", lambda a, sp500: a.get("in_portfolio", False)),
    ("Researching", lambda a, sp500: a.get("in_researching", False)),
    ("Ready to Buy", lambda a, sp500: a.get("in_ready_to_buy", False)),
    ("Ready to Short", lambda a, sp500: a.get("in_ready_to_short", False)),
    ("Following for Interest", lambda a, sp500: a.get("in_following_for_interest", False)),
    ("Healthcare Services", lambda a, sp500: a.get("sector") == "Healthcare Services"),
    ("MedTech", lambda a, sp500: a.get("sector") == "MedTech"),
    ("Large Pharma", lambda a, sp500: a.get("subsector") == "Large Pharma"),
    ("Other (Tech, SaaS, Fin, Ind, Cons, Energy, Mat, RE)",
     lambda a, sp500: a.get("sector") in SECTORS_GROUPED_AS_OTHER),
    ("S&P 500", lambda a, sp500: a["ticker"] in sp500),
]


def _load_position_set(path: Path, label: str) -> set[str]:
    """Generic loader for portfolio.json / researching.json.

    Both files are pushed by Coverage Manager's sigma_export step. Missing
    file is not an error — the corresponding Slack subcategory just renders
    empty.
    """
    if not path.exists():
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not read {label}: {e}")
        return set()
    if not isinstance(data, dict):
        return set()
    return {t.upper() for t in data.keys()}


def load_portfolio() -> set[str]:
    """Return the set of tickers with Position == 'Portfolio' (held names)."""
    return _load_position_set(PORTFOLIO_PATH, "portfolio.json")


def load_researching() -> set[str]:
    """Return the set of tickers with Position == 'Researching' (thesis-building)."""
    return _load_position_set(RESEARCHING_PATH, "researching.json")


def load_following_for_interest() -> set[str]:
    """Return the set of tickers with Position == 'Following for Interest'
    (passive earnings/signal tracking; no intent to trade)."""
    return _load_position_set(FOLLOWING_PATH, "following_for_interest.json")


def load_ready_to_buy() -> set[str]:
    """Return the set of tickers with Position == 'Ready to Buy'
    (long thesis complete; waiting for entry trigger)."""
    return _load_position_set(READY_TO_BUY_PATH, "ready_to_buy.json")


def load_ready_to_short() -> set[str]:
    """Return the set of tickers with Position == 'Ready to Short'
    (short thesis complete; waiting for entry trigger)."""
    return _load_position_set(READY_TO_SHORT_PATH, "ready_to_short.json")


def load_core_watchlist() -> set[str]:
    """DEPRECATED — back-compat wrapper. Returns the union of portfolio +
    researching from the new files. Falls back to legacy core_watchlist.json
    if neither new file exists yet (e.g. fresh sigma-alert clone before
    Coverage Manager's first push of the new files).

    Will be removed once Coverage Manager stops pushing core_watchlist.json.
    """
    portfolio = load_portfolio()
    researching = load_researching()
    if portfolio or researching:
        return portfolio | researching
    # Fallback to legacy file if the new ones haven't been pushed yet.
    if not CORE_WATCHLIST_PATH.exists():
        return set()
    try:
        with open(CORE_WATCHLIST_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not read core_watchlist.json: {e}")
        return set()
    if not isinstance(data, dict):
        return set()
    return {t.upper() for t in data.keys()}


def load_metadata() -> dict:
    """Load ticker metadata (company name, sector) if available."""
    if not METADATA_PATH.exists():
        return {}
    try:
        with open(METADATA_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def load_cache() -> dict | None:
    """Load distribution cache if it exists."""
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_cache(cache: dict) -> None:
    """Persist distribution cache to disk."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def write_missing_metadata_flag(tickers: list[str], metadata: dict,
                                exempt: set[str] | None = None) -> dict:
    """Identify watchlist tickers missing from ticker_metadata.json (or with
    a blank `name`) and write a flag file for Coverage Manager to pick up.

    Coverage Manager owns ticker_metadata.json and is the only system that can
    fix gaps. This file is written to `cache/missing_metadata.json` so it gets
    committed by the EOD CI run alongside the distribution cache, then the
    sibling Coverage Manager weekly build reads it and surfaces the gaps to
    the operator.

    `exempt` lists tickers whose names are owned locally (e.g. ETFs sourced
    from `etf_names.json`) and should not be reported as CM gaps.

    Returns the dict that was written (or an empty dict if no gaps).
    """
    metadata = metadata or {}
    exempt = exempt or set()
    gaps = {}
    for t in tickers:
        if t in exempt:
            continue
        meta = metadata.get(t)
        if meta is None:
            gaps[t] = "not_in_metadata"
        elif not (meta.get("name") or "").strip():
            gaps[t] = "missing_name"

    MISSING_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not gaps:
        # Clear any stale flag file so Coverage Manager doesn't keep warning.
        if MISSING_METADATA_PATH.exists():
            MISSING_METADATA_PATH.unlink()
        return {}

    payload = {
        "updated": now_et().isoformat(),
        "tickers": gaps,
    }
    with open(MISSING_METADATA_PATH, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[WARN] {len(gaps)} ticker(s) missing metadata — flagged for Coverage Manager: {sorted(gaps)}")
    return payload


def update_skip_log(skip_events: list[dict], mode: str) -> dict:
    """Append today's skip events to cache/skip_log.json and trim to the
    retention window. Consumed by Coverage Manager's weekly report.

    Schema:
        {
          "runs": [
            {"date": "YYYY-MM-DD", "mode": "close",
             "skipped": [{"ticker": "ABC", "reason": "insufficient_history"}, ...]}
          ]
        }

    Only the close run calls this — it's the canonical daily snapshot and
    the only mode whose cache directory gets committed by CI.

    Returns the trimmed payload that was written.
    """
    today_str = today_et().strftime("%Y-%m-%d")
    payload = {"runs": []}
    if SKIP_LOG_PATH.exists():
        try:
            with open(SKIP_LOG_PATH) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                payload = loaded
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Could not read skip_log.json, starting fresh: {e}")

    # Drop any prior entry for today+mode so re-runs overwrite cleanly.
    payload["runs"] = [
        r for r in payload["runs"]
        if not (r.get("date") == today_str and r.get("mode") == mode)
    ]
    payload["runs"].append({
        "date": today_str,
        "mode": mode,
        "skipped": sorted(skip_events, key=lambda e: e.get("ticker", "")),
    })

    # Trim to retention window.
    cutoff = (today_et() - timedelta(days=SKIP_LOG_RETENTION_DAYS)).strftime("%Y-%m-%d")
    payload["runs"] = [r for r in payload["runs"] if r.get("date", "") >= cutoff]
    payload["runs"].sort(key=lambda r: (r.get("date", ""), r.get("mode", "")))

    SKIP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SKIP_LOG_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Skip log updated: {len(skip_events)} skip(s) recorded for {today_str} {mode}")
    return payload


def is_cache_fresh(cache: dict) -> bool:
    """Check if cache date is the most recent prior trading day.

    Decision: we consider the cache 'fresh' if it was written yesterday or
    on Friday (for a Monday morning run). More precisely, we check that no
    more than 3 calendar days have passed — this covers weekends but will
    force a refresh after long holiday weekends, which is acceptable since
    we'd want updated distributions anyway.
    """
    if not cache or "date" not in cache:
        return False
    try:
        cache_date = datetime.strptime(cache["date"], "%Y-%m-%d").date()
    except ValueError:
        return False
    today = today_et()
    delta = (today - cache_date).days
    # Fresh if written within the last 3 calendar days (handles weekends)
    return 0 < delta <= 3


def validate_bar_date(bar_index: pd.DatetimeIndex, mode: str) -> bool:
    """Check that the latest bar in the data belongs to the current trading session.

    For open and midday modes, the latest bar must be today (the market is open).
    For close mode, the latest bar must also be today — the EOD run fires after
    the close, so today's bar should be present.

    Returns True if valid. Logs a warning and returns False if stale.
    """
    if len(bar_index) == 0:
        return False
    latest_bar_date = bar_index[-1].date()
    today = today_et()
    if latest_bar_date == today:
        return True
    # Allow 1-day staleness only on weekends/holidays where the runner fires
    # but the market was closed — in practice the cron only runs Mon-Fri,
    # but this guards against holiday edge cases.
    print(f"[WARN] Latest bar date is {latest_bar_date}, expected {today}")
    return False


def batch_download(tickers: list[str], period_start: str, period_end: str) -> pd.DataFrame | None:
    """Download OHLC data for all tickers in a single yfinance call.

    Note: yfinance auto_adjust=True is the default — Close prices are
    adjusted for splits and dividends.
    """
    try:
        data = yf.download(
            tickers,
            start=period_start,
            end=period_end,
            progress=False,
            threads=True,
        )
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"[WARN] Batch download failed: {e}")
        return None


def fallback_download_single(ticker: str, period_start: str, period_end: str) -> pd.DataFrame | None:
    """Download data for a single ticker as fallback."""
    try:
        data = yf.download(
            ticker,
            start=period_start,
            end=period_end,
            progress=False,
        )
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"[WARN] Fallback download failed for {ticker}: {e}")
        return None


def compute_distribution(close_series: pd.Series) -> tuple[float, float, int]:
    """Compute mean and std of daily returns from a close price series.

    Uses the prior 251 days of returns (excluding the most recent day)
    so that today's move is measured against a clean trailing distribution.
    """
    daily_returns = close_series.pct_change().dropna()
    # Exclude the last return (today's) — distribution is trailing only
    trailing = daily_returns.iloc[:-1]
    if len(trailing) < 30:
        # Decision: require at least 30 data points for a meaningful distribution.
        # Stocks with insufficient history are skipped rather than producing
        # unreliable z-scores.
        return (np.nan, np.nan, 0)
    mu = float(trailing.mean())
    sigma = float(trailing.std(ddof=1))  # sample std
    return (mu, sigma, len(trailing))


def compute_z_score(today_return: float, mu: float, sigma: float) -> float:
    """Calculate z-score of today's return vs trailing distribution."""
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return (today_return - mu) / sigma


def download_todays_prices(tickers: list[str]) -> dict:
    """Download only today's OHLC for open-mode with fresh cache.

    Decision: we use period='5d' which gives the last 5 trading days.
    We then validate that the latest bar is from today before using it.
    """
    prices = {}
    try:
        data = yf.download(tickers, period="5d", progress=False, threads=True)
        if data.empty:
            return prices

        if len(tickers) == 1:
            ticker = tickers[0]
            if not validate_bar_date(data.index, "open"):
                print(f"[WARN] Stale data for {ticker}, skipping")
                return prices
            if len(data) >= 2:
                prev_close = float(data["Close"].iloc[-2])
                today_open = float(data["Open"].iloc[-1])
                prices[ticker] = {"prev_close": prev_close, "today_open": today_open}
        else:
            # Validate bar date once from the shared index
            if not validate_bar_date(data.index, "open"):
                print("[WARN] Stale batch data, skipping all tickers in cached path")
                return prices
            for ticker in tickers:
                try:
                    close_col = data["Close"][ticker].dropna()
                    open_col = data["Open"][ticker].dropna()
                    if len(close_col) >= 2 and len(open_col) >= 1:
                        prev_close = float(close_col.iloc[-2])
                        today_open = float(open_col.iloc[-1])
                        prices[ticker] = {"prev_close": prev_close, "today_open": today_open}
                except (KeyError, IndexError):
                    continue
    except Exception as e:
        print(f"[WARN] Today's price batch download failed: {e}")
        for ticker in tickers:
            time.sleep(random.uniform(1, 2))
            try:
                d = yf.download(ticker, period="5d", progress=False)
                if len(d) >= 2 and validate_bar_date(d.index, "open"):
                    prev_close = float(d["Close"].iloc[-2])
                    today_open = float(d["Open"].iloc[-1])
                    prices[ticker] = {"prev_close": prev_close, "today_open": today_open}
            except Exception as e2:
                print(f"[WARN] Fallback today price failed for {ticker}: {e2}")
    return prices


def screen_open_cached(tickers: list[str], cache: dict,
                       metadata: dict | None = None,
                       portfolio_set: set[str] | None = None,
                       researching_set: set[str] | None = None,
                       following_set: set[str] | None = None,
                       ready_to_buy_set: set[str] | None = None,
                       ready_to_short_set: set[str] | None = None,
                       etf_set: set[str] | None = None) -> tuple[list[dict], dict, list[dict]]:
    """Open-mode screening using cached mu/sigma — only downloads today's prices.

    `etf_set` is the union of index + sector ETFs whose returns should be
    captured for the "Index & Sector Returns" Slack section regardless of
    whether they cross an alert threshold.

    Returns (alerts, run_stats, etf_returns).
    """
    alerts = []
    etf_returns = []
    _etfs = etf_set or set()
    stats = {"screened": 0, "skipped": 0, "stale": 0}
    prices = download_todays_prices(tickers)
    ticker_cache = cache.get("tickers", {})

    if not prices:
        # validate_bar_date already logged the warning
        stats["stale"] = len(tickers)
        return alerts, stats, etf_returns

    for ticker in tickers:
        if ticker not in ticker_cache:
            print(f"[INFO] {ticker} not in cache, skipping in cached open mode")
            stats["skipped"] += 1
            continue
        if ticker not in prices:
            print(f"[WARN] No today price for {ticker}, skipping")
            stats["skipped"] += 1
            continue

        stats["screened"] += 1
        mu = ticker_cache[ticker]["mu"]
        sigma = ticker_cache[ticker]["sigma"]
        # 52w high/low cached from prior EOD run — may be missing on first
        # post-upgrade run until cache is refreshed.
        high_52w = ticker_cache[ticker].get("high_52w")
        low_52w = ticker_cache[ticker].get("low_52w")
        # Prior year-end close cached by the last EOD close run — drives YTD.
        prior_year_end_close = ticker_cache[ticker].get("prior_year_end_close")
        prior_year_end_year = ticker_cache[ticker].get("prior_year_end_year")
        prev_close = prices[ticker]["prev_close"]
        today_open = prices[ticker]["today_open"]
        today_return = (today_open - prev_close) / prev_close

        # YTD vs the cached prior year-end close. Guard the year so a stale
        # December cache used on the first session of a new year doesn't
        # compute YTD off the wrong baseline — omit it for that one morning
        # until the next EOD close run refreshes the cache.
        ytd_return_pct = None
        if prior_year_end_close and prior_year_end_year == today_et().year - 1:
            ytd_return_pct = (today_open - prior_year_end_close) / prior_year_end_close * 100

        z = compute_z_score(today_return, mu, sigma)
        meta = (metadata or {}).get(ticker, {})
        sector = meta.get("sector", "")
        subsector = meta.get("subsector", "")
        abs_z = abs(z)

        # Collect ETF stats regardless of threshold (drives Index & Sector
        # Returns section). Indices vs. sectors are partitioned downstream.
        if ticker in _etfs:
            etf_returns.append({
                "ticker": ticker,
                "name": meta.get("name", ""),
                "z_score": z,
                "return_pct": today_return * 100,
                "price": today_open,
                "high_52w": high_52w,
                "low_52w": low_52w,
            })

        tier = None
        if abs_z >= SIGMA_THRESHOLD:
            tier = "2sigma"
        elif abs_z >= ONE_SIGMA_THRESHOLD and _is_one_sigma_eligible(
                meta, ticker, portfolio_set, researching_set,
                following_set=following_set,
                ready_to_buy_set=ready_to_buy_set,
                ready_to_short_set=ready_to_short_set):
            tier = "1sigma"
        if tier:
            alerts.append({
                "ticker": ticker,
                "name": meta.get("name", ""),
                "sector": sector,
                "subsector": subsector,
                "z_score": z,
                "return_pct": today_return * 100,
                "price": today_open,
                "high_52w": high_52w,
                "low_52w": low_52w,
                "ytd_return_pct": ytd_return_pct,
                "direction": "up" if today_return > 0 else "down",
                "three_sigma": abs_z >= THREE_SIGMA,
                "tier": tier,
                "in_portfolio": ticker in (portfolio_set or set()),
                "in_researching": ticker in (researching_set or set()),
                "in_following_for_interest": ticker in (following_set or set()),
                "in_ready_to_buy": ticker in (ready_to_buy_set or set()),
                "in_ready_to_short": ticker in (ready_to_short_set or set()),
            })
    return alerts, stats, etf_returns


def check_52w_high_low(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> str | None:
    """Check if today's bar hit a 52-week high or low.

    Compares today's high/low against the trailing highs/lows (excluding today).
    Returns "high", "low", or None.  If both occur on the same day, "high" wins
    (extremely rare — would require a massive intraday range spanning both extremes).
    """
    if len(high_series) < 2 or len(low_series) < 2:
        return None
    trailing_high = float(high_series.iloc[:-1].max())
    trailing_low = float(low_series.iloc[:-1].min())
    today_high = float(high_series.iloc[-1])
    today_low = float(low_series.iloc[-1])
    today_close = float(close_series.iloc[-1])

    if today_high >= trailing_high:
        return "high"
    if today_low <= trailing_low:
        return "low"
    return None


def _process_ticker_full(ticker: str, close: pd.Series, open_prices: pd.Series,
                         high_series: pd.Series | None, low_series: pd.Series | None,
                         mode: str, metadata: dict | None = None,
                         portfolio_set: set[str] | None = None,
                         researching_set: set[str] | None = None,
                         following_set: set[str] | None = None,
                         ready_to_buy_set: set[str] | None = None,
                         ready_to_short_set: set[str] | None = None) -> tuple[dict | None, dict | None, dict | None, dict | None, str | None]:
    """Process a single ticker in full-screen mode.

    Returns (alert_or_none, cache_entry_or_none, hi_lo_or_none, ticker_stats_or_none, skip_reason_or_none).
    ticker_stats is always populated when computation succeeds (used for sector ETF returns).
    skip_reason is set (and other values None) when the ticker cannot be screened.
    """
    if len(close) < 32:
        print(f"[WARN] {ticker}: insufficient data ({len(close)} days), skipping")
        return None, None, None, None, "insufficient_history"

    mu, sigma, sample_size = compute_distribution(close)
    if np.isnan(mu):
        print(f"[WARN] {ticker}: could not compute distribution, skipping")
        return None, None, None, None, "distribution_nan"

    # Compute 52-week high/low from the downloaded history (always, when available).
    # Trailing 252 sessions is ~1 year. If we have less, use what we've got.
    # `high_series`/`low_series` are preferred (capture intraday extremes);
    # falls back to close-only if those columns weren't downloaded. Use up to
    # the last 253 bars so today's intraday extreme can itself be the 52w edge.
    high_52w = None
    low_52w = None
    if high_series is not None and len(high_series) >= 2:
        high_52w = float(high_series.iloc[-min(len(high_series), 253):].max())
    elif len(close) >= 2:
        high_52w = float(close.iloc[-min(len(close), 253):].max())
    if low_series is not None and len(low_series) >= 2:
        low_52w = float(low_series.iloc[-min(len(low_series), 253):].min())
    elif len(close) >= 2:
        low_52w = float(close.iloc[-min(len(close), 253):].min())

    # Prior calendar year-end close → drives the YTD return shown on alert
    # rows, and is cached so the morning cached-open path can compute YTD
    # without a full history download. `close` carries a DatetimeIndex from
    # yfinance; the 400-day download window always reaches the prior year-end.
    # None when the series doesn't span into the prior year (e.g. a recent
    # IPO) or has no DatetimeIndex (some tests pass a plain-int index) — YTD
    # is then omitted downstream rather than failing.
    prior_year = today_et().year - 1
    prior_year_end_close = None
    try:
        py_closes = close[close.index.year == prior_year]
        if not py_closes.empty:
            prior_year_end_close = float(py_closes.iloc[-1])
    except (AttributeError, TypeError):
        prior_year_end_close = None

    cache_entry = {"mu": mu, "sigma": sigma, "sample_size": sample_size}
    if high_52w is not None:
        cache_entry["high_52w"] = high_52w
    if low_52w is not None:
        cache_entry["low_52w"] = low_52w
    if prior_year_end_close is not None:
        cache_entry["prior_year_end_close"] = prior_year_end_close
        cache_entry["prior_year_end_year"] = prior_year

    # Compute today's return based on mode
    prev_close = float(close.iloc[-2])
    if mode == "open":
        today_price = float(open_prices.iloc[-1])
    else:
        today_price = float(close.iloc[-1])

    today_return = (today_price - prev_close) / prev_close
    z = compute_z_score(today_return, mu, sigma)

    ytd_return_pct = None
    if prior_year_end_close:
        ytd_return_pct = (today_price - prior_year_end_close) / prior_year_end_close * 100

    meta = (metadata or {}).get(ticker, {})
    name = meta.get("name", "")
    sector = meta.get("sector", "")
    subsector = meta.get("subsector", "")

    # Always-populated stats for sector ETF returns section
    ticker_stats = {
        "ticker": ticker,
        "name": name,
        "z_score": z,
        "return_pct": today_return * 100,
        "price": today_price,
        "high_52w": high_52w,
        "low_52w": low_52w,
    }

    alert = None
    abs_z = abs(z)
    tier = None
    if abs_z >= SIGMA_THRESHOLD:
        tier = "2sigma"
    elif abs_z >= ONE_SIGMA_THRESHOLD and _is_one_sigma_eligible(
            meta, ticker, portfolio_set, researching_set,
            following_set=following_set,
            ready_to_buy_set=ready_to_buy_set,
            ready_to_short_set=ready_to_short_set):
        tier = "1sigma"

    if tier:
        alert = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "subsector": subsector,
            "z_score": z,
            "return_pct": today_return * 100,
            "price": today_price,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "ytd_return_pct": ytd_return_pct,
            "direction": "up" if today_return > 0 else "down",
            "three_sigma": abs_z >= THREE_SIGMA,
            "tier": tier,
            "in_portfolio": ticker in (portfolio_set or set()),
            "in_researching": ticker in (researching_set or set()),
            "in_following_for_interest": ticker in (following_set or set()),
            "in_ready_to_buy": ticker in (ready_to_buy_set or set()),
            "in_ready_to_short": ticker in (ready_to_short_set or set()),
        }

    # 52-week high/low check (only when high/low data is provided)
    hi_lo = None
    if high_series is not None and low_series is not None:
        result = check_52w_high_low(high_series, low_series, close)
        if result:
            hi_lo = {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "type": result,
                "price": float(close.iloc[-1]),
            }

    return alert, cache_entry, hi_lo, ticker_stats, None


def screen_full(tickers: list[str], mode: str, track_52w: bool = False,
                metadata: dict | None = None,
                portfolio_set: set[str] | None = None,
                researching_set: set[str] | None = None,
                following_set: set[str] | None = None,
                ready_to_buy_set: set[str] | None = None,
                ready_to_short_set: set[str] | None = None,
                etf_set: set[str] | None = None) -> tuple[list[dict], dict, dict, list[dict], list[dict], list[dict]]:
    """Full screening: downloads history, computes distributions.

    `etf_set` is the union of index + sector ETFs whose per-ticker stats
    should be captured for the "Index & Sector Returns" Slack block. They
    still go through the alert-tier logic — a 2σ ETF move is noteworthy.

    Returns (alerts, cache_data, run_stats, hi_lo_hits, etf_returns, skip_events).

    skip_events is a list of {ticker, reason} dicts for Coverage Manager's
    weekly report. Reasons: insufficient_history, distribution_nan,
    fallback_insufficient, fallback_exception. (Stale events are tracked
    separately in stats["stale"].)

    Note on end date: yf.download(end=...) is exclusive — to include today's
    bar we must pass tomorrow's date as the end boundary.
    """
    today = today_et()
    end_date = today + timedelta(days=1)  # exclusive upper bound
    start_date = today - timedelta(days=CALENDAR_DOWNLOAD_DAYS)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    alerts = []
    hi_lo_hits = []
    etf_returns = []
    skip_events: list[dict] = []
    _etfs = etf_set or set()
    cache_data = {"date": today.strftime("%Y-%m-%d"), "tickers": {}}
    stats = {"screened": 0, "skipped": 0, "stale": 0, "ref_date": None}

    # Attempt batch download
    data = batch_download(tickers, start_str, end_str)
    failed_tickers = []

    if data is not None:
        # Validate that the latest bar is from today's session
        if not validate_bar_date(data.index, mode):
            stats["stale"] = len(tickers)
            print(f"[ERROR] Batch data is stale — latest bar is not from {today}. Aborting screen.")
            return alerts, cache_data, stats, hi_lo_hits, etf_returns, skip_events

        stats["ref_date"] = str(data.index[-1].date())

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    close = data["Close"].dropna()
                    open_prices = data["Open"].dropna()
                    high_s = data["High"].dropna()
                    low_s = data["Low"].dropna()
                else:
                    close = data["Close"][ticker].dropna()
                    open_prices = data["Open"][ticker].dropna()
                    high_s = data["High"][ticker].dropna()
                    low_s = data["Low"][ticker].dropna()

                alert, cache_entry, hi_lo, ticker_stats, skip_reason = _process_ticker_full(
                    ticker, close, open_prices, high_s, low_s, mode, metadata,
                    portfolio_set=portfolio_set,
                    researching_set=researching_set,
                    following_set=following_set,
                    ready_to_buy_set=ready_to_buy_set,
                    ready_to_short_set=ready_to_short_set,
                )

                if cache_entry is None:
                    stats["skipped"] += 1
                    skip_events.append({"ticker": ticker, "reason": skip_reason or "unknown"})
                    continue

                cache_data["tickers"][ticker] = cache_entry
                stats["screened"] += 1
                if alert:
                    alerts.append(alert)
                if hi_lo and track_52w:
                    hi_lo_hits.append(hi_lo)
                if ticker_stats and ticker in _etfs:
                    etf_returns.append(ticker_stats)

            except (KeyError, IndexError) as e:
                print(f"[WARN] {ticker} failed in batch data: {e}")
                failed_tickers.append(ticker)
    else:
        failed_tickers = list(tickers)

    # Fallback: download individually for any tickers that failed in batch
    for ticker in failed_tickers:
        print(f"[INFO] Falling back to individual download for {ticker}")
        time.sleep(random.uniform(1, 2))
        single_data = fallback_download_single(ticker, start_str, end_str)
        if single_data is None or len(single_data) < 32:
            print(f"[WARN] {ticker}: insufficient data in fallback, skipping")
            stats["skipped"] += 1
            skip_events.append({"ticker": ticker, "reason": "fallback_insufficient"})
            continue

        if not validate_bar_date(single_data.index, mode):
            print(f"[WARN] {ticker}: stale data in fallback, skipping")
            stats["stale"] += 1
            continue

        if stats["ref_date"] is None:
            stats["ref_date"] = str(single_data.index[-1].date())

        try:
            close = single_data["Close"].dropna()
            open_prices = single_data["Open"].dropna()
            high_s = single_data["High"].dropna()
            low_s = single_data["Low"].dropna()

            alert, cache_entry, hi_lo, ticker_stats, skip_reason = _process_ticker_full(
                ticker, close, open_prices, high_s, low_s, mode, metadata,
                portfolio_set=portfolio_set,
                researching_set=researching_set,
                following_set=following_set,
                ready_to_buy_set=ready_to_buy_set,
                ready_to_short_set=ready_to_short_set,
            )

            if cache_entry is None:
                stats["skipped"] += 1
                skip_events.append({"ticker": ticker, "reason": skip_reason or "unknown"})
                continue

            cache_data["tickers"][ticker] = cache_entry
            stats["screened"] += 1
            if alert:
                alerts.append(alert)
            if hi_lo and track_52w:
                hi_lo_hits.append(hi_lo)
            if ticker_stats and ticker in _etfs:
                etf_returns.append(ticker_stats)

        except Exception as e:
            print(f"[WARN] {ticker} fallback processing failed: {e}")
            stats["skipped"] += 1
            skip_events.append({"ticker": ticker, "reason": "fallback_exception"})

    return alerts, cache_data, stats, hi_lo_hits, etf_returns, skip_events


def fetch_etf_period_returns(etf_set: set[str], mode: str) -> dict:
    """Fetch prior-year and YTD returns for a set of tickers.

    Despite the name, the caller passes both the index/sector ETFs and the
    alert tickers here so alert rows can carry the same prior-year/YTD suffix
    as the returns block.

    Uses a dedicated, longer-window download (~800 calendar days) so we
    reach the last close of the year *before* last — required as the
    baseline for the prior year's full-year return (e.g. 2024-12-31 for
    2025's return). The main batch download stays at 400 days to keep
    the ~1500-ticker call snappy; this side fetch covers only the ~14
    index + sector ETFs.

    YTD return uses today's open in `open` mode and today's close
    otherwise, matching the convention in `_process_ticker_full` /
    `screen_open_cached`.

    Returns `{ticker: {prior_year_label, prior_year_return_pct,
    ytd_return_pct}}`. Tickers whose required year-end closes can't be
    located (e.g. ETF inception after 2024-12-31) are omitted; their
    Slack rows then render without the suffix.
    """
    if not etf_set:
        return {}

    today = today_et()
    end_date = today + timedelta(days=1)  # yfinance end is exclusive
    start_date = today - timedelta(days=800)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    tickers = sorted(etf_set)
    try:
        data = yf.download(
            tickers,
            start=start_str,
            end=end_str,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"[WARN] ETF period-returns download failed: {e}")
        return {}
    if data is None or data.empty:
        print("[WARN] ETF period-returns download returned no data")
        return {}

    current_year = today.year
    prior_year = current_year - 1
    prior_prior_year = current_year - 2

    results: dict = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                close = data["Close"].dropna()
                open_s = data["Open"].dropna()
            else:
                close = data["Close"][ticker].dropna()
                open_s = data["Open"][ticker].dropna()
        except (KeyError, IndexError):
            continue

        if close.empty:
            continue

        years = close.index.year
        prior_year_closes = close[years == prior_year]
        prior_prior_year_closes = close[years == prior_prior_year]
        if prior_year_closes.empty or prior_prior_year_closes.empty:
            continue

        prior_year_end_close = float(prior_year_closes.iloc[-1])
        prior_prior_year_end_close = float(prior_prior_year_closes.iloc[-1])
        prior_year_return_pct = (
            (prior_year_end_close - prior_prior_year_end_close)
            / prior_prior_year_end_close * 100
        )

        if mode == "open" and not open_s.empty:
            today_price = float(open_s.iloc[-1])
        else:
            today_price = float(close.iloc[-1])

        ytd_return_pct = (
            (today_price - prior_year_end_close) / prior_year_end_close * 100
        )

        results[ticker] = {
            "prior_year_label": str(prior_year),
            "prior_year_return_pct": prior_year_return_pct,
            "ytd_return_pct": ytd_return_pct,
        }

    return results


def format_slack_message(alerts: list[dict], mode: str, total_tickers: int,
                         stats: dict, hi_lo_hits: list[dict] | None = None,
                         sp500_set: set[str] | None = None,
                         etf_returns: list[dict] | None = None,
                         index_etf_set: set[str] | None = None,
                         healthcare_etf_set: set[str] | None = None,
                         tech_etf_set: set[str] | None = None,
                         macro_etf_set: set[str] | None = None,
                         etf_period_returns: dict | None = None) -> dict:
    """Build Slack message payload using Block Kit for clean formatting."""
    current = now_et()
    date_str = current.strftime("%Y-%m-%d")
    time_str = current.strftime("%I:%M %p %Z")
    mode_label = {"open": "Open", "midday": "Midday", "close": "Close"}[mode]

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Sigma Alert — {mode_label} {date_str}",
            },
        },
    ]

    # Prior-year + YTD returns, keyed by ticker. Covers both the index/sector
    # ETFs and the alert tickers (main() folds the alert names into the same
    # long-window fetch, since the prior calendar-year return needs the
    # year-before-last's year-end close \u2014 older than the 400-day screen window).
    period_map = etf_period_returns or {}
    prior_year_label = str(current.year - 1)

    def _format_alert_line(a):
        marker = "\U0001F7E9" if a["direction"] == "up" else "\U0001F7E5"
        sigma_note = "  *\u26A0\uFE0F 3\u03C3+ move!*" if a["three_sigma"] else ""
        sign = "+" if a["return_pct"] > 0 else ""
        short = short_company_name(a.get("name", ""))
        name_part = f" ({short})" if short else ""
        price = a.get("price")
        price_part = f"  |  ${price:.2f}" if price is not None else ""
        lo, hi = a.get("low_52w"), a.get("high_52w")
        pct_of_high_part = (
            f"  |  {price / hi * 100:.0f}% of 52w high"
            if price is not None and hi is not None and hi > 0 else ""
        )
        range_part = f"  |  52w: ${lo:.2f} - ${hi:.2f}" if lo is not None and hi is not None else ""

        # Prior calendar-year return (e.g. "2025: +24.50%") comes from the
        # dedicated long-window fetch. When the year-before-last's year-end
        # close can't be located (e.g. an IPO after that cutover), the column
        # still renders as "2025: N/A" rather than being dropped.
        period = period_map.get(a["ticker"])
        py = period.get("prior_year_return_pct") if period else None
        if py is not None:
            py_sign = "+" if py > 0 else ""
            py_label = (period or {}).get("prior_year_label") or prior_year_label
            prior_year_part = f"  |  {py_label}: {py_sign}{py:.2f}%"
        else:
            prior_year_part = f"  |  {prior_year_label}: N/A"

        # YTD: prefer the inline value (computed from the screen history / cache,
        # so it's present even when the period fetch omits the ticker), falling
        # back to the period fetch's YTD when inline isn't available.
        ytd = a.get("ytd_return_pct")
        if ytd is None and period is not None:
            ytd = period.get("ytd_return_pct")
        ytd_part = ""
        if ytd is not None:
            ytd_sign = "+" if ytd > 0 else ""
            ytd_part = f"  |  YTD: {ytd_sign}{ytd:.2f}%"

        return (
            f"{marker}  `{a['ticker']}`{name_part}  "
            f"|  {sign}{a['return_pct']:.2f}%  |  z = {a['z_score']:+.2f}"
            f"{price_part}{pct_of_high_part}{range_part}{prior_year_part}{ytd_part}{sigma_note}"
        )

    def _append_section_chunked(blocks_list, header, lines, max_len=2900):
        """Append a section block, splitting into multiple blocks if text exceeds Slack's 3000-char limit."""
        chunk = [header]
        chunk_len = len(header) + 1
        for line in lines:
            line_len = len(line) + 1
            if chunk_len + line_len > max_len and len(chunk) > 1:
                blocks_list.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "\n".join(chunk)},
                })
                chunk = [line]
                chunk_len = line_len
            else:
                chunk.append(line)
                chunk_len += line_len
        if chunk:
            blocks_list.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(chunk)},
            })

    # Sort by signed z-score descending: biggest gainers on top,
    # biggest losers on the bottom, within each tier.
    sp500 = sp500_set or set()
    two_sig = sorted(
        [a for a in alerts if a.get("tier") == "2sigma"],
        key=lambda a: a["z_score"], reverse=True,
    )
    one_sig = sorted(
        [a for a in alerts if a.get("tier") == "1sigma"],
        key=lambda a: a["z_score"], reverse=True,
    )

    def _render_tier(tier_alerts, tier_header):
        """Render a tier as a header plus one subsection per SUBCATEGORIES match.
        An alert is duplicated across every category it matches. Alerts that
        match none are dropped — 1σ can't hit this path (the eligibility gate
        already requires Core or any Position-list membership, all of which
        map to a subcategory); 2σ alerts outside Position lists / HC sectors /
        Large Pharma / S&P 500 are intentionally hidden.
        """
        _append_section_chunked(blocks, tier_header, [])
        for label, predicate in SUBCATEGORIES:
            members = [a for a in tier_alerts if predicate(a, sp500)]
            if not members:
                continue
            sub_header = f"    _{label} ({len(members)})_"
            _append_section_chunked(
                blocks, sub_header, [_format_alert_line(a) for a in members]
            )

    if two_sig or one_sig:
        if two_sig:
            header_2 = f":bar_chart: *2\u03C3+ Moves ({len(two_sig)})*"
            _render_tier(two_sig, header_2)

        if one_sig:
            if two_sig:
                blocks.append({"type": "divider"})
            header_1 = (
                f":chart_with_upwards_trend: *1\u03C3 Moves ({len(one_sig)})* "
                f"— Core + Position lists only"
            )
            _render_tier(one_sig, header_1)
    else:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"No sigma moves detected across {total_tickers} tickers.",
            },
        })

    # 52-week high/low section (close mode only)
    if hi_lo_hits:
        blocks.append({"type": "divider"})

        highs = sorted([h for h in hi_lo_hits if h["type"] == "high"], key=lambda h: h["ticker"])
        lows = sorted([h for h in hi_lo_hits if h["type"] == "low"], key=lambda h: h["ticker"])

        def _format_hi_lo_ticker(h):
            short = short_company_name(h.get("name", ""))
            name_part = f" ({short})" if short else ""
            sector = f" [{h['sector']}]" if h.get("sector") else ""
            return f"`{h['ticker']}`{name_part}{sector}"

        hi_lo_lines = []
        if highs:
            tickers_str = ", ".join(_format_hi_lo_ticker(h) for h in highs)
            hi_lo_lines.append(f"\U0001F7E2 *52-Week Highs ({len(highs)}):*  {tickers_str}")
        if lows:
            tickers_str = ", ".join(_format_hi_lo_ticker(h) for h in lows)
            hi_lo_lines.append(f"\U0001F534 *52-Week Lows ({len(lows)}):*  {tickers_str}")

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(hi_lo_lines),
            },
        })

    # Index & sector ETF returns section. Indices (SPYM/DIA/QQQ) render
    # at the top, then sector ETFs underneath. Both groups sorted by
    # z-score descending so the strongest move within each group leads.
    if etf_returns:
        idx_set = index_etf_set or set()
        hc_set = healthcare_etf_set or set()
        tech_set = tech_etf_set or set()
        macro_set = macro_etf_set or set()
        index_rows = sorted(
            [s for s in etf_returns if s["ticker"] in idx_set],
            key=lambda s: s["z_score"], reverse=True,
        )
        healthcare_rows = sorted(
            [s for s in etf_returns if s["ticker"] in hc_set],
            key=lambda s: s["z_score"], reverse=True,
        )
        tech_rows = sorted(
            [s for s in etf_returns if s["ticker"] in tech_set],
            key=lambda s: s["z_score"], reverse=True,
        )
        sector_rows = sorted(
            [s for s in etf_returns
             if s["ticker"] not in idx_set and s["ticker"] not in hc_set
             and s["ticker"] not in tech_set and s["ticker"] not in macro_set],
            key=lambda s: s["z_score"], reverse=True,
        )
        # Macro rows render in fixed MACRO_ORDER (rates -> FX -> commodity),
        # not z-sorted — three unlike asset classes don't sort meaningfully.
        _macro_by_ticker = {s["ticker"]: s for s in etf_returns if s["ticker"] in macro_set}
        macro_rows = [_macro_by_ticker[t] for t in MACRO_ORDER if t in _macro_by_ticker]
        # Any macro ticker not in MACRO_ORDER (shouldn't happen) still shows.
        macro_rows += [s for s in etf_returns
                       if s["ticker"] in macro_set and s["ticker"] not in MACRO_ORDER]

        def _format_macro_line(s):
            """Render a macro row. Yields show level% + bp change; price/level
            rows show $price or bare level + %change. All append the z-score."""
            t = s["ticker"]
            style = MACRO_STYLE.get(t, "price")
            rp = s["return_pct"]
            z = s["z_score"]
            level = s.get("price")
            name = s.get("name") or t
            marker = "\U0001F7E9" if rp > 0 else "\U0001F7E5"
            sign = "+" if rp > 0 else ""
            if style == "yield" and level is not None:
                # level is the yield in percent (e.g. 4.45); recover the prior
                # close from the % change to express the move in basis points.
                denom = 1 + rp / 100
                bp_part = ""
                if denom != 0:
                    prev = level / denom
                    bp_part = f"  |  {(level - prev) * 100:+.1f}bp"
                core = f"{level:.2f}%{bp_part}  |  z = {z:+.2f}"
            elif style == "price" and level is not None:
                core = f"${level:.2f}  |  {sign}{rp:.2f}%  |  z = {z:+.2f}"
            elif level is not None:  # bare level (e.g. DXY)
                core = f"{level:.2f}  |  {sign}{rp:.2f}%  |  z = {z:+.2f}"
            else:
                core = f"{sign}{rp:.2f}%  |  z = {z:+.2f}"
            return f"{marker}  `{t}` ({name})  |  {core}"

        def _format_etf_line(s):
            marker = "\U0001F7E9" if s["return_pct"] > 0 else "\U0001F7E5"
            sign = "+" if s["return_pct"] > 0 else ""
            short = short_company_name(s.get("name", ""))
            name_part = f" ({short})" if short else ""
            price = s.get("price")
            price_part = f"  |  ${price:.2f}" if price is not None else ""
            lo, hi = s.get("low_52w"), s.get("high_52w")
            pct_of_high_part = (
                f"  |  {price / hi * 100:.0f}% of 52w high"
                if price is not None and hi is not None and hi > 0 else ""
            )
            range_part = f"  |  52w: ${lo:.2f} - ${hi:.2f}" if lo is not None and hi is not None else ""
            period = period_map.get(s["ticker"])
            period_part = ""
            if period:
                py_sign = "+" if period["prior_year_return_pct"] > 0 else ""
                ytd_sign = "+" if period["ytd_return_pct"] > 0 else ""
                period_part = (
                    f"  |  {period['prior_year_label']}: "
                    f"{py_sign}{period['prior_year_return_pct']:.2f}%"
                    f"  |  YTD: {ytd_sign}{period['ytd_return_pct']:.2f}%"
                )
            return (
                f"{marker}  `{s['ticker']}`{name_part}  "
                f"|  {sign}{s['return_pct']:.2f}%  |  z = {s['z_score']:+.2f}"
                f"{price_part}{pct_of_high_part}{range_part}{period_part}"
            )

        if index_rows or sector_rows or healthcare_rows or tech_rows or macro_rows:
            blocks.append({"type": "divider"})
            header = ":chart_with_upwards_trend: *Index, Sector & Macro Returns*"
            lines = []
            rendered_any = False
            if macro_rows:
                lines.append("_Macro_")
                lines.extend(_format_macro_line(s) for s in macro_rows)
                rendered_any = True
            if index_rows:
                if rendered_any:
                    lines.append("")  # blank spacer between groups
                lines.append("_Indices_")
                lines.extend(_format_etf_line(s) for s in index_rows)
                rendered_any = True
            if sector_rows:
                if rendered_any:
                    lines.append("")  # blank spacer between groups
                lines.append("_Sectors_")
                lines.extend(_format_etf_line(s) for s in sector_rows)
                rendered_any = True
            if healthcare_rows:
                if rendered_any:
                    lines.append("")  # blank spacer between groups
                lines.append("_Healthcare_")
                lines.extend(_format_etf_line(s) for s in healthcare_rows)
                rendered_any = True
            if tech_rows:
                if rendered_any:
                    lines.append("")  # blank spacer between groups
                lines.append("_Tech Themes_")
                lines.extend(_format_etf_line(s) for s in tech_rows)
                rendered_any = True
            _append_section_chunked(blocks, header, lines)

    blocks.append({"type": "divider"})

    # Audit context line
    ref_date = stats.get("ref_date", "unknown")
    screened = stats.get("screened", 0)
    skipped = stats.get("skipped", 0)
    stale = stats.get("stale", 0)
    context_parts = [f"Screened {screened}/{total_tickers} tickers at {time_str}"]
    context_parts.append(f"Bar date: {ref_date}")
    if skipped:
        context_parts.append(f"Skipped: {skipped}")
    if stale:
        context_parts.append(f"Stale: {stale}")

    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": "  |  ".join(context_parts),
            },
        ],
    })

    return {"blocks": blocks}


def send_slack(payload: dict) -> None:
    """Post message to Slack via incoming webhook."""
    webhook_url = os.environ.get("SLACK_WEBHOOK")
    if not webhook_url:
        print("[ERROR] SLACK_WEBHOOK environment variable not set")
        print("[FALLBACK] Alert payload:")
        print(json.dumps(payload, indent=2))
        return

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        print("[OK] Slack message sent successfully")
    except requests.RequestException as e:
        print(f"[ERROR] Slack webhook failed: {e}")
        print("[FALLBACK] Alert payload:")
        print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Stock sigma screener")
    parser.add_argument("--mode", choices=["open", "midday", "close"], required=True)
    args = parser.parse_args()

    tickers = load_watchlist()
    if not tickers:
        print("[ERROR] No tickers in watchlist")
        sys.exit(1)

    metadata_raw = load_metadata()
    if metadata_raw:
        print(f"[INFO] Loaded metadata for {len(metadata_raw)} tickers")

    sp500_set = load_sp500_set()
    if sp500_set:
        print(f"[INFO] Loaded {len(sp500_set)} S&P 500 tickers")

    # Fill in names for S&P 500 tickers that Coverage Manager doesn't maintain.
    # Coverage Manager owns ticker_metadata.json but only populates the
    # healthcare/MedTech/PA universe, so most S&P 500 names come from this
    # Wikipedia-sourced fallback file. Keep `metadata_raw` unmerged so
    # write_missing_metadata_flag still reports true CM gaps.
    metadata = {k: dict(v) for k, v in metadata_raw.items()}
    sp500_names = load_sp500_names()
    if sp500_names:
        filled = 0
        for ticker, name in sp500_names.items():
            entry = metadata.get(ticker)
            if entry is None:
                metadata[ticker] = {"name": name, "sector": "", "subsector": ""}
                filled += 1
            elif not (entry.get("name") or "").strip():
                entry["name"] = name
                filled += 1
        if filled:
            print(f"[INFO] Filled {filled} S&P 500 names from sp500_names.json")

    portfolio_set = load_portfolio()
    researching_set = load_researching()
    following_set = load_following_for_interest()
    ready_to_buy_set = load_ready_to_buy()
    ready_to_short_set = load_ready_to_short()
    if portfolio_set:
        print(f"[INFO] Loaded {len(portfolio_set)} Portfolio tickers")
    if researching_set:
        print(f"[INFO] Loaded {len(researching_set)} Researching tickers")
    if following_set:
        print(f"[INFO] Loaded {len(following_set)} Following-for-Interest tickers")
    if ready_to_buy_set:
        print(f"[INFO] Loaded {len(ready_to_buy_set)} Ready-to-Buy tickers")
    if ready_to_short_set:
        print(f"[INFO] Loaded {len(ready_to_short_set)} Ready-to-Short tickers")
    if not portfolio_set and not researching_set:
        # Fall back to legacy core_watchlist.json if neither new file is present
        # (e.g. fresh sigma-alert clone before Coverage Manager's first push of
        # the new files).
        legacy = load_core_watchlist()
        if legacy:
            print(f"[INFO] Loaded {len(legacy)} legacy core_watchlist tickers (no Portfolio/Researching split available yet — they all render under Portfolio)")
            portfolio_set = legacy

    index_etf_set = load_index_etfs()
    sector_etf_set = load_sector_etfs()
    healthcare_etf_set = load_healthcare_etfs()
    tech_etf_set = load_tech_etfs()
    macro_set = load_macro()
    # Tech-theme + macro tickers join etf_set so they share the download path,
    # alert suppression, and missing-metadata exemption — but render separately.
    etf_set = (index_etf_set | sector_etf_set | healthcare_etf_set
               | tech_etf_set | macro_set)
    if etf_set:
        print(
            f"[INFO] Loaded {len(index_etf_set)} index ETFs + "
            f"{len(sector_etf_set)} sector ETFs + "
            f"{len(healthcare_etf_set)} healthcare ETFs + "
            f"{len(tech_etf_set)} tech-theme ETFs + "
            f"{len(macro_set)} macro tickers for returns block"
        )

    # Merge ETF display names into metadata (CM doesn't track ETFs).
    # ETF tickers are authoritative here: `etf_names.json` overrides any
    # existing CM entry (name AND sector/subsector). Needed because ETF
    # tickers can collide with foreign equities — e.g. `DIA` is both the
    # SPDR DJIA ETF and DiaSorin S.p.A. on Borsa Italiana; letting CM's
    # "DiaSorin / MedTech" classification through would both mislabel the
    # Index & Sector Returns row and risk firing DIA as a 1σ MedTech alert.
    etf_names = load_etf_names()
    if etf_names:
        for ticker, name in etf_names.items():
            metadata[ticker] = {"name": name, "sector": "", "subsector": ""}
        print(f"[INFO] Applied {len(etf_names)} ETF names from etf_names.json")

    # Make sure ETFs are always screened even if a watchlist sync (e.g. from
    # Coverage Manager) drops them. Preserves watchlist order; appends any
    # ETF not already present.
    missing_etfs = [t for t in sorted(etf_set) if t not in tickers]
    if missing_etfs:
        print(f"[INFO] Adding {len(missing_etfs)} ETF(s) absent from watchlist: {missing_etfs}")
        tickers = tickers + missing_etfs

    print(f"[INFO] Mode: {args.mode} | Tickers: {len(tickers)} | Time: {now_et().isoformat()}")

    hi_lo_hits = []
    etf_returns = []

    if args.mode == "open":
        # Try cached path first — avoids full history download
        cache = load_cache()
        if cache and is_cache_fresh(cache):
            print("[INFO] Using cached distributions for open-mode screening")
            alerts, stats, etf_returns = screen_open_cached(
                tickers, cache, metadata,
                portfolio_set=portfolio_set, researching_set=researching_set,
                following_set=following_set,
                ready_to_buy_set=ready_to_buy_set,
                ready_to_short_set=ready_to_short_set,
                etf_set=etf_set,
            )
        else:
            print("[INFO] Cache stale or missing, running full download for open mode")
            alerts, _, stats, _, etf_returns, _ = screen_full(
                tickers, "open", metadata=metadata,
                portfolio_set=portfolio_set, researching_set=researching_set,
                following_set=following_set,
                ready_to_buy_set=ready_to_buy_set,
                ready_to_short_set=ready_to_short_set,
                etf_set=etf_set,
            )
            # Don't save cache on open runs — only EOD updates the cache
    elif args.mode == "midday":
        # Midday mode: same price comparison as close but don't update cache
        alerts, _, stats, _, etf_returns, _ = screen_full(
            tickers, "close", metadata=metadata,
            portfolio_set=portfolio_set, researching_set=researching_set,
            following_set=following_set,
            ready_to_buy_set=ready_to_buy_set,
            ready_to_short_set=ready_to_short_set,
            etf_set=etf_set,
        )
    else:
        # Close mode: full download, update cache, and check 52-week highs/lows
        alerts, cache_data, stats, hi_lo_hits, etf_returns, skip_events = screen_full(
            tickers, "close", track_52w=True, metadata=metadata,
            portfolio_set=portfolio_set, researching_set=researching_set,
            following_set=following_set,
            ready_to_buy_set=ready_to_buy_set,
            ready_to_short_set=ready_to_short_set,
            etf_set=etf_set,
        )
        save_cache(cache_data)
        print(f"[INFO] Cache saved with {len(cache_data['tickers'])} tickers")
        # Use the pre-fallback metadata so Coverage Manager still sees true gaps.
        # ETFs are exempt — their display names live in this repo
        # (sources/etf_names.json), not in CM's universe.
        write_missing_metadata_flag(tickers, metadata_raw, exempt=etf_set)
        # Persist today's skip events so Coverage Manager's weekly report can
        # surface chronic skips, reason breakdowns, and unresolved tickers.
        update_skip_log(skip_events, mode="close")

    # Report results
    if alerts:
        print(f"[ALERT] {len(alerts)} sigma moves detected:")
        for a in alerts:
            print(f"  {a['ticker']}: z={a['z_score']:+.2f}, return={a['return_pct']:+.2f}%")
    else:
        print("[INFO] No sigma moves detected")

    if hi_lo_hits:
        highs = [h for h in hi_lo_hits if h["type"] == "high"]
        lows = [h for h in hi_lo_hits if h["type"] == "low"]
        print(f"[INFO] 52-week highs: {len(highs)}, lows: {len(lows)}")

    if etf_returns:
        print(f"[INFO] Index/sector ETF returns captured: {len(etf_returns)} tickers")

    # Prior-year + YTD returns use their own longer-window download (the main
    # batch is capped at 400 days, which doesn't reach the year-before-last's
    # year-end close needed for the prior calendar-year return). Covers the
    # index/sector ETFs AND the alert tickers, so alert rows can show the same
    # `2025: ±% | YTD: ±%` suffix as the returns block. Macro tickers are
    # excluded — a prior-year/YTD "return" on a bond yield is misleading, and
    # the _Macro_ rows render level + day-change only.
    alert_tickers = {a["ticker"] for a in alerts}
    period_fetch_set = (etf_set - macro_set) | alert_tickers
    etf_period_returns = fetch_etf_period_returns(period_fetch_set, args.mode)
    if etf_period_returns:
        print(f"[INFO] Period returns computed: {len(etf_period_returns)} tickers "
              f"({len(alert_tickers)} alert + ETFs)")

    # Send to Slack
    payload = format_slack_message(
        alerts, args.mode, len(tickers), stats, hi_lo_hits, sp500_set,
        etf_returns=etf_returns, index_etf_set=index_etf_set,
        healthcare_etf_set=healthcare_etf_set,
        tech_etf_set=tech_etf_set,
        macro_etf_set=macro_set,
        etf_period_returns=etf_period_returns,
    )
    send_slack(payload)


if __name__ == "__main__":
    main()
