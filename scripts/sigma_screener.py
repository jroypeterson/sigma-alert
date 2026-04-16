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
ETF_NAMES_PATH = ROOT / "sources" / "etf_names.json"
# Core watchlist pushed by Coverage Manager's weekly sigma_export step.
# Owned by Coverage Manager — do NOT edit by hand in this repo.
CORE_WATCHLIST_PATH = ROOT / "core_watchlist.json"

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

# Sectors that get the lower 1σ alert tier (in-coverage tickers).
# 2σ+ alerts fire on the entire watchlist regardless of sector.
ONE_SIGMA_SECTORS = {"Healthcare Services", "MedTech", "PA"}

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
# Anything matching none lands in "Other" so no alert is dropped.
SUBCATEGORIES = [
    ("Core Watchlist", lambda a, sp500: a.get("on_watchlist", False)),
    ("Healthcare Services", lambda a, sp500: a.get("sector") == "Healthcare Services"),
    ("MedTech", lambda a, sp500: a.get("sector") == "MedTech"),
    ("Other/PA", lambda a, sp500: a.get("sector") == "PA"),
    ("S&P 500", lambda a, sp500: a["ticker"] in sp500),
]


def load_core_watchlist() -> set[str]:
    """Return the set of tickers on the core watchlist.

    The file is written by Coverage Manager's sigma_export step. Missing file
    is not an error — it just means no watchlist was pushed yet, and the
    "Core Watchlist" subcategory will be empty.
    """
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
                       core_watchlist: set[str] | None = None,
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
        prev_close = prices[ticker]["prev_close"]
        today_open = prices[ticker]["today_open"]
        today_return = (today_open - prev_close) / prev_close

        z = compute_z_score(today_return, mu, sigma)
        meta = (metadata or {}).get(ticker, {})
        sector = meta.get("sector", "")
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
        elif abs_z >= ONE_SIGMA_THRESHOLD and sector in ONE_SIGMA_SECTORS:
            tier = "1sigma"
        if tier:
            alerts.append({
                "ticker": ticker,
                "name": meta.get("name", ""),
                "sector": sector,
                "z_score": z,
                "return_pct": today_return * 100,
                "price": today_open,
                "high_52w": high_52w,
                "low_52w": low_52w,
                "direction": "up" if today_return > 0 else "down",
                "three_sigma": abs_z >= THREE_SIGMA,
                "tier": tier,
                "on_watchlist": ticker in (core_watchlist or set()),
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
                         core_watchlist: set[str] | None = None) -> tuple[dict | None, dict | None, dict | None, dict | None, str | None]:
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

    cache_entry = {"mu": mu, "sigma": sigma, "sample_size": sample_size}
    if high_52w is not None:
        cache_entry["high_52w"] = high_52w
    if low_52w is not None:
        cache_entry["low_52w"] = low_52w

    # Compute today's return based on mode
    prev_close = float(close.iloc[-2])
    if mode == "open":
        today_price = float(open_prices.iloc[-1])
    else:
        today_price = float(close.iloc[-1])

    today_return = (today_price - prev_close) / prev_close
    z = compute_z_score(today_return, mu, sigma)

    meta = (metadata or {}).get(ticker, {})
    name = meta.get("name", "")
    sector = meta.get("sector", "")

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
    elif abs_z >= ONE_SIGMA_THRESHOLD and sector in ONE_SIGMA_SECTORS:
        tier = "1sigma"

    if tier:
        alert = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "z_score": z,
            "return_pct": today_return * 100,
            "price": today_price,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "direction": "up" if today_return > 0 else "down",
            "three_sigma": abs_z >= THREE_SIGMA,
            "tier": tier,
            "on_watchlist": ticker in (core_watchlist or set()),
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
                core_watchlist: set[str] | None = None,
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
                    core_watchlist=core_watchlist,
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
                core_watchlist=core_watchlist,
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
            if ticker_stats and ticker in _sector_etfs:
                sector_returns.append(ticker_stats)

        except Exception as e:
            print(f"[WARN] {ticker} fallback processing failed: {e}")
            stats["skipped"] += 1
            skip_events.append({"ticker": ticker, "reason": "fallback_exception"})

    return alerts, cache_data, stats, hi_lo_hits, etf_returns, skip_events


def format_slack_message(alerts: list[dict], mode: str, total_tickers: int,
                         stats: dict, hi_lo_hits: list[dict] | None = None,
                         sp500_set: set[str] | None = None,
                         etf_returns: list[dict] | None = None,
                         index_etf_set: set[str] | None = None) -> dict:
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

    def _format_alert_line(a):
        marker = "\U0001F7E9" if a["direction"] == "up" else "\U0001F7E5"
        sigma_note = "  *\u26A0\uFE0F 3\u03C3+ move!*" if a["three_sigma"] else ""
        sign = "+" if a["return_pct"] > 0 else ""
        short = short_company_name(a.get("name", ""))
        name_part = f" ({short})" if short else ""
        price = a.get("price")
        price_part = f"  |  ${price:.2f}" if price is not None else ""
        lo, hi = a.get("low_52w"), a.get("high_52w")
        range_part = f"  |  52w: ${lo:.2f} - ${hi:.2f}" if lo is not None and hi is not None else ""
        return (
            f"{marker}  *{a['ticker']}*{name_part}  "
            f"|  {sign}{a['return_pct']:.2f}%  |  z = {a['z_score']:+.2f}{price_part}{range_part}{sigma_note}"
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
        match none are dropped — 1σ can't hit this path (already sector-filtered);
        2σ alerts outside HC Services/MedTech/PA/S&P 500 are intentionally hidden.
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
                f"— HC Services / MedTech / PA only"
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
            return f"*{h['ticker']}*{name_part}{sector}"

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
        index_rows = sorted(
            [s for s in etf_returns if s["ticker"] in idx_set],
            key=lambda s: s["z_score"], reverse=True,
        )
        sector_rows = sorted(
            [s for s in etf_returns if s["ticker"] not in idx_set],
            key=lambda s: s["z_score"], reverse=True,
        )

        def _format_etf_line(s):
            marker = "\U0001F7E9" if s["return_pct"] > 0 else "\U0001F7E5"
            sign = "+" if s["return_pct"] > 0 else ""
            short = short_company_name(s.get("name", ""))
            name_part = f" ({short})" if short else ""
            price = s.get("price")
            price_part = f"  |  ${price:.2f}" if price is not None else ""
            lo, hi = s.get("low_52w"), s.get("high_52w")
            range_part = f"  |  52w: ${lo:.2f} - ${hi:.2f}" if lo is not None and hi is not None else ""
            return (
                f"{marker}  *{s['ticker']}*{name_part}  "
                f"|  {sign}{s['return_pct']:.2f}%  |  z = {s['z_score']:+.2f}{price_part}{range_part}"
            )

        if index_rows or sector_rows:
            blocks.append({"type": "divider"})
            header = ":chart_with_upwards_trend: *Index & Sector Returns*"
            lines = []
            if index_rows:
                lines.append("_Indices_")
                lines.extend(_format_etf_line(s) for s in index_rows)
            if sector_rows:
                if index_rows:
                    lines.append("")  # blank spacer between groups
                lines.append("_Sectors_")
                lines.extend(_format_etf_line(s) for s in sector_rows)
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

    core_watchlist = load_core_watchlist()
    if core_watchlist:
        print(f"[INFO] Loaded {len(core_watchlist)} core watchlist tickers")

    index_etf_set = load_index_etfs()
    sector_etf_set = load_sector_etfs()
    etf_set = index_etf_set | sector_etf_set
    if etf_set:
        print(
            f"[INFO] Loaded {len(index_etf_set)} index ETFs + "
            f"{len(sector_etf_set)} sector ETFs for returns block"
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
                tickers, cache, metadata, core_watchlist=core_watchlist,
                etf_set=etf_set,
            )
        else:
            print("[INFO] Cache stale or missing, running full download for open mode")
            alerts, _, stats, _, etf_returns, _ = screen_full(
                tickers, "open", metadata=metadata, core_watchlist=core_watchlist,
                etf_set=etf_set,
            )
            # Don't save cache on open runs — only EOD updates the cache
    elif args.mode == "midday":
        # Midday mode: same price comparison as close but don't update cache
        alerts, _, stats, _, etf_returns, _ = screen_full(
            tickers, "close", metadata=metadata, core_watchlist=core_watchlist,
            etf_set=etf_set,
        )
    else:
        # Close mode: full download, update cache, and check 52-week highs/lows
        alerts, cache_data, stats, hi_lo_hits, etf_returns, skip_events = screen_full(
            tickers, "close", track_52w=True, metadata=metadata,
            core_watchlist=core_watchlist, etf_set=etf_set,
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

    # Send to Slack
    payload = format_slack_message(
        alerts, args.mode, len(tickers), stats, hi_lo_hits, sp500_set,
        etf_returns=etf_returns, index_etf_set=index_etf_set,
    )
    send_slack(payload)


if __name__ == "__main__":
    main()
