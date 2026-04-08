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
METADATA_PATH = ROOT / "ticker_metadata.json"
MISSING_METADATA_PATH = ROOT / "cache" / "missing_metadata.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")

LOOKBACK_DAYS = 252          # ~1 trading year
SIGMA_THRESHOLD = 2.0
ONE_SIGMA_THRESHOLD = 1.0
THREE_SIGMA = 3.0

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


def write_missing_metadata_flag(tickers: list[str], metadata: dict) -> dict:
    """Identify watchlist tickers missing from ticker_metadata.json (or with
    a blank `name`) and write a flag file for Coverage Manager to pick up.

    Coverage Manager owns ticker_metadata.json and is the only system that can
    fix gaps. This file is written to `cache/missing_metadata.json` so it gets
    committed by the EOD CI run alongside the distribution cache, then the
    sibling Coverage Manager weekly build reads it and surfaces the gaps to
    the operator.

    Returns the dict that was written (or an empty dict if no gaps).
    """
    metadata = metadata or {}
    gaps = {}
    for t in tickers:
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
                       metadata: dict | None = None) -> tuple[list[dict], dict]:
    """Open-mode screening using cached mu/sigma — only downloads today's prices.

    Returns (alerts, run_stats).
    """
    alerts = []
    stats = {"screened": 0, "skipped": 0, "stale": 0}
    prices = download_todays_prices(tickers)
    ticker_cache = cache.get("tickers", {})

    if not prices:
        # validate_bar_date already logged the warning
        stats["stale"] = len(tickers)
        return alerts, stats

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
        prev_close = prices[ticker]["prev_close"]
        today_open = prices[ticker]["today_open"]
        today_return = (today_open - prev_close) / prev_close

        z = compute_z_score(today_return, mu, sigma)
        meta = (metadata or {}).get(ticker, {})
        sector = meta.get("sector", "")
        abs_z = abs(z)
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
                "direction": "up" if today_return > 0 else "down",
                "three_sigma": abs_z >= THREE_SIGMA,
                "tier": tier,
            })
    return alerts, stats


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
                         mode: str, metadata: dict | None = None) -> tuple[dict | None, dict | None, dict | None]:
    """Process a single ticker in full-screen mode.

    Returns (alert_or_none, cache_entry_or_none, hi_lo_or_none).
    """
    if len(close) < 32:
        print(f"[WARN] {ticker}: insufficient data ({len(close)} days), skipping")
        return None, None, None

    mu, sigma, sample_size = compute_distribution(close)
    if np.isnan(mu):
        print(f"[WARN] {ticker}: could not compute distribution, skipping")
        return None, None, None

    cache_entry = {"mu": mu, "sigma": sigma, "sample_size": sample_size}

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
            "direction": "up" if today_return > 0 else "down",
            "three_sigma": abs_z >= THREE_SIGMA,
            "tier": tier,
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

    return alert, cache_entry, hi_lo


def screen_full(tickers: list[str], mode: str, track_52w: bool = False,
                metadata: dict | None = None) -> tuple[list[dict], dict, dict, list[dict]]:
    """Full screening: downloads history, computes distributions.

    Returns (alerts, cache_data, run_stats, hi_lo_hits).

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
            return alerts, cache_data, stats, hi_lo_hits

        stats["ref_date"] = str(data.index[-1].date())

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    close = data["Close"].dropna()
                    open_prices = data["Open"].dropna()
                    high_s = data["High"].dropna() if track_52w else None
                    low_s = data["Low"].dropna() if track_52w else None
                else:
                    close = data["Close"][ticker].dropna()
                    open_prices = data["Open"][ticker].dropna()
                    high_s = data["High"][ticker].dropna() if track_52w else None
                    low_s = data["Low"][ticker].dropna() if track_52w else None

                alert, cache_entry, hi_lo = _process_ticker_full(
                    ticker, close, open_prices, high_s, low_s, mode, metadata,
                )

                if cache_entry is None:
                    stats["skipped"] += 1
                    continue

                cache_data["tickers"][ticker] = cache_entry
                stats["screened"] += 1
                if alert:
                    alerts.append(alert)
                if hi_lo:
                    hi_lo_hits.append(hi_lo)

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
            high_s = single_data["High"].dropna() if track_52w else None
            low_s = single_data["Low"].dropna() if track_52w else None

            alert, cache_entry, hi_lo = _process_ticker_full(
                ticker, close, open_prices, high_s, low_s, mode, metadata,
            )

            if cache_entry is None:
                stats["skipped"] += 1
                continue

            cache_data["tickers"][ticker] = cache_entry
            stats["screened"] += 1
            if alert:
                alerts.append(alert)
            if hi_lo:
                hi_lo_hits.append(hi_lo)

        except Exception as e:
            print(f"[WARN] {ticker} fallback processing failed: {e}")
            stats["skipped"] += 1

    return alerts, cache_data, stats, hi_lo_hits


def format_slack_message(alerts: list[dict], mode: str, total_tickers: int,
                         stats: dict, hi_lo_hits: list[dict] | None = None) -> dict:
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
        sector_part = f"  [{a['sector']}]" if a.get("sector") else ""
        return (
            f"{marker}  *{a['ticker']}*{name_part}{sector_part}  "
            f"|  z = {a['z_score']:+.2f}  |  {sign}{a['return_pct']:.2f}%{sigma_note}"
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
    two_sig = sorted(
        [a for a in alerts if a.get("tier") == "2sigma"],
        key=lambda a: a["z_score"], reverse=True,
    )
    one_sig = sorted(
        [a for a in alerts if a.get("tier") == "1sigma"],
        key=lambda a: a["z_score"], reverse=True,
    )

    if two_sig or one_sig:
        if two_sig:
            header_2 = f":bar_chart: *2\u03C3+ Moves ({len(two_sig)})*"
            _append_section_chunked(
                blocks, header_2, [_format_alert_line(a) for a in two_sig]
            )

        if one_sig:
            if two_sig:
                blocks.append({"type": "divider"})
            header_1 = (
                f":chart_with_upwards_trend: *1\u03C3 Moves ({len(one_sig)})* "
                f"— HC Services / MedTech / PA only"
            )
            _append_section_chunked(
                blocks, header_1, [_format_alert_line(a) for a in one_sig]
            )
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

    metadata = load_metadata()
    if metadata:
        print(f"[INFO] Loaded metadata for {len(metadata)} tickers")

    print(f"[INFO] Mode: {args.mode} | Tickers: {len(tickers)} | Time: {now_et().isoformat()}")

    hi_lo_hits = []

    if args.mode == "open":
        # Try cached path first — avoids full history download
        cache = load_cache()
        if cache and is_cache_fresh(cache):
            print("[INFO] Using cached distributions for open-mode screening")
            alerts, stats = screen_open_cached(tickers, cache, metadata)
        else:
            print("[INFO] Cache stale or missing, running full download for open mode")
            alerts, _, stats, _ = screen_full(tickers, "open", metadata=metadata)
            # Don't save cache on open runs — only EOD updates the cache
    elif args.mode == "midday":
        # Midday mode: same price comparison as close but don't update cache
        alerts, _, stats, _ = screen_full(tickers, "close", metadata=metadata)
    else:
        # Close mode: full download, update cache, and check 52-week highs/lows
        alerts, cache_data, stats, hi_lo_hits = screen_full(tickers, "close", track_52w=True, metadata=metadata)
        save_cache(cache_data)
        print(f"[INFO] Cache saved with {len(cache_data['tickers'])} tickers")
        write_missing_metadata_flag(tickers, metadata)

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

    # Send to Slack
    payload = format_slack_message(alerts, args.mode, len(tickers), stats, hi_lo_hits)
    send_slack(payload)


if __name__ == "__main__":
    main()
