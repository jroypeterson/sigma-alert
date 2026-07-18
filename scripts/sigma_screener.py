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
GLOBAL_EQUITY_ETFS_PATH = ROOT / "sources" / "global_equity_etfs.txt"
HEALTHCARE_ETFS_PATH = ROOT / "sources" / "healthcare_etfs.txt"
TECH_ETFS_PATH = ROOT / "sources" / "tech_etfs.txt"
COMMODITY_ETFS_PATH = ROOT / "sources" / "commodity_etfs.txt"
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


# ---------------------------------------------------------------------------
# Foreign-ticker symbol normalization
# ---------------------------------------------------------------------------
# Coverage Manager pushes foreign coverage names as Bloomberg-style symbols
# (GETIB.SS, COH.AU, ...) and keys ticker_metadata.json + the position files by
# the BARE base symbol (GETIB, COH, ...). yfinance needs yet a third form
# (GETI-B.ST, COH.AX). So a foreign watchlist ticker carries three identities:
#   • display   — the watchlist ticker (Slack label, cache key, skip-log key)
#   • yf symbol — what we hand to yf.download()                (to_yf_symbol)
#   • meta key  — the base symbol CM keys metadata/positions by (to_metadata_key)
# US tickers and caret indices (^TNX) have all three identical. US class shares
# already use the dash form in the watchlist (BF-B, BRK-B), so a dotted ticker
# is always a foreign exchange listing — no US-class-share collision.
# All mappings below verified live against yfinance on 2026-06-13.

# Bloomberg exchange suffix → yfinance exchange suffix, for the suffixes that
# DIFFER. Suffixes already in yfinance form (.DE/.MI/.SW/.L/.SA/.T/.HK) need no
# remap and are intentionally absent here.
_BLOOMBERG_SUFFIX_MAP = {
    "SS": "ST",   # Stockholm (Nasdaq Stockholm)
    "AU": "AX",   # Australia (ASX)
    "IM": "MI",   # Milan (Borsa Italiana)
    "DC": "CO",   # Copenhagen
    "LN": "L",    # London (LSE)
    "FP": "PA",   # Paris (Euronext)
    "CH": "SW",   # Switzerland (SIX)
    "GY": "DE",   # Germany (Xetra)
}

# Every exchange suffix recognized as a foreign listing (Bloomberg forms +
# already-yfinance forms that appear in the watchlist). A dotted ticker whose
# suffix is in here is keyed in CM metadata/positions by its bare base symbol.
_EXCHANGE_SUFFIXES = set(_BLOOMBERG_SUFFIX_MAP) | {
    "ST", "AX", "MI", "CO", "L", "PA", "SW", "DE", "SA", "T", "HK",
}

# Explicit display → yfinance overrides the suffix rule can't derive:
#   - class shares, where Bloomberg glues the class letter onto the base
#     (GETIB → GETI-B), and
#   - bare foreign names carrying no exchange suffix in the watchlist.
# The 2026-06-13 batch was each verified to return live yfinance data whose
# name matches the CM metadata entry. Names NOT included (ASX/OSSFF/SHMZF)
# already resolve as-is; MMED is left untouched (ambiguous — "MMED" on Nasdaq
# is a different company than CM's metadata name). The 2026-06-21 additions
# (CVSG→CVSG.L, SFZS.SW→SFZN.SW) are web-confirmed Yahoo Finance symbols
# (quote pages exist) resolving the chronic insufficient_history skips; a live
# yfinance round-trip should confirm them on the next close run.
_YF_SYMBOL_OVERRIDES = {
    # Class shares
    "GETIB.SS": "GETI-B.ST",   # Getinge B
    "COLOB.DC": "COLO-B.CO",   # Coloplast B
    "AMBUSH.DC": "AMBU-B.CO",  # Ambu B
    "SECARE.SS": "SECT-B.ST",  # Sectra B
    # Bare foreign names (watchlist has no exchange suffix)
    "BIM": "BIM.PA",      # bioMerieux (Euronext Paris)
    "FRE": "FRE.DE",      # Fresenius (Xetra)
    "GXI": "GXI.DE",      # Gerresheimer (Xetra)
    "DAE": "DAE.SW",      # Daetwyler (SIX)
    "SOON": "SOON.SW",    # Sonova (SIX)
    "YPSN": "YPSN.SW",    # Ypsomed (SIX)
    "RDOR3": "RDOR3.SA",  # Rede D'Or (B3, Brazil)
    "CPH": "CPH.TO",      # Cipher Pharmaceuticals (TSX)
    "CVSG": "CVSG.L",     # CVS Group plc (LSE, UK animal-health / vet services)
    # Wrong-symbol correction — CM's watchlist symbol does not exist on yfinance:
    "SFZS.SW": "SFZN.SW",  # Siegfried Holding (SIX) — "SFZS" is a typo for SFZN
}


def to_yf_symbol(display: str) -> str:
    """Map a watchlist (display/Bloomberg) ticker to its yfinance symbol."""
    if display in _YF_SYMBOL_OVERRIDES:
        return _YF_SYMBOL_OVERRIDES[display]
    if "." not in display or display.startswith("^"):
        return display
    base, _, suffix = display.rpartition(".")
    if suffix in _BLOOMBERG_SUFFIX_MAP:
        return f"{base}.{_BLOOMBERG_SUFFIX_MAP[suffix]}"
    return display  # already a yfinance-form suffix (.DE/.MI/.SW/.L/.SA/.T/.HK)


def to_metadata_key(display: str) -> str:
    """Map a watchlist ticker to the base symbol Coverage Manager keys
    ticker_metadata.json + the position files by. A foreign exchange listing
    is keyed by its bare base (GETIB.SS → GETIB); US tickers, dash-form class
    shares (BF-B), and caret indices are returned unchanged."""
    if "." not in display or display.startswith("^"):
        return display
    base, _, suffix = display.rpartition(".")
    if suffix in _EXCHANGE_SUFFIXES:
        return base
    return display


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


def load_global_equity_etfs() -> set[str]:
    """Load global/international equity index ETFs (ACWI/EFA/EEM/VGK/EWJ/EWY/
    FXI/INDA) from sources/global_equity_etfs.txt.

    Regular total-return ETFs that render under a `_Global Equity_` sub-header
    below the `_Credit_` group (the `_US Indices_` group leads the section,
    then the macro/rates/credit backdrop), via the normal `_format_etf_line`
    (prior-year/YTD returns). The country ETFs are USD-denominated, so each
    bundles the local-equity move and the FX move vs the dollar.
    """
    return _load_ticker_set(GLOBAL_EQUITY_ETFS_PATH)


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


def load_commodity_etfs() -> set[str]:
    """Load commodity ETFs (GLD) from sources/commodity_etfs.txt.

    Regular total-return ETFs that render under a `_Commodities_` sub-header in
    the Slack returns block (below `_Tech Themes_`). Like the other ETF groups
    they carry prior-year/YTD period returns and go through `_format_etf_line`.
    """
    return _load_ticker_set(COMMODITY_ETFS_PATH)


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


# ---------------------------------------------------------------------------
# Credit indices (HY / IG) — sourced from FRED, NOT yfinance.
# ---------------------------------------------------------------------------
# JP asked for "HY and IG index levels and YTD yield changes" — actual credit
# index levels, not bond-ETF prices. The ICE BofA index families publish daily
# effective yields AND option-adjusted spreads (OAS, the rate-stripped credit
# signal) on FRED. FRED's public CSV endpoint needs no API key, so this adds a
# credit backdrop without a new secret. These are levels, not tradeable
# tickers, so they render in their own `_Credit_` sub-group (after `_Macro_`)
# via `_format_credit_line` and never produce alerts or join the download path.
#
# Series (yields + OAS both reported in percent on FRED; ×100 → basis points):
#   BAMLH0A0HYM2EY — ICE BofA US High Yield Index Effective Yield
#   BAMLH0A0HYM2   — ICE BofA US High Yield Index Option-Adjusted Spread
#   BAMLC0A0CMEY   — ICE BofA US Corporate (IG) Index Effective Yield
#   BAMLC0A0CM     — ICE BofA US Corporate (IG) Index Option-Adjusted Spread
CREDIT_SERIES = {
    "HY": {"label": "US High Yield", "yield_id": "BAMLH0A0HYM2EY", "oas_id": "BAMLH0A0HYM2"},
    "IG": {"label": "US Corp IG", "yield_id": "BAMLC0A0CMEY", "oas_id": "BAMLC0A0CM"},
}
CREDIT_ORDER = ["HY", "IG"]
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"


# ---------------------------------------------------------------------------
# US Treasury yield curve (2/10/30) — sourced from FRED, NOT yfinance.
# ---------------------------------------------------------------------------
# JP asked for the US yield curve (2, 10, 30-year) "as of yesterday's close".
# Yahoo/yfinance has no 2-year yield ticker (only ^FVX 5Y / ^TNX 10Y / ^TYX 30Y),
# so the full 2/10/30 curve must come from FRED's daily constant-maturity series
# (DGS2/DGS10/DGS30). FRED daily yields publish the prior business day's close
# next morning — i.e. exactly "yesterday's close". Rendered in its own
# `_Treasury Curve_` sub-group (after `_Macro_`, before `_Credit_`) via
# `_format_curve_line`: each maturity shows level% + 1-day bp move + YTD bp.
# Colored like a bond, NOT an equity: a yield RISE = price DOWN = 🟥, a yield
# FALL = 🟩 (the same inversion `_format_credit_line` applies to OAS). This is
# deliberately separate from the intraday ^TNX 10Y in `_Macro_` — that row is a
# live (today) tick; this curve is the prior-close snapshot across maturities,
# so the 10Y intentionally appears in both. No API key required (FRED CSV is
# public).
TREASURY_CURVE_SERIES = {
    "2Y": "DGS2",
    "10Y": "DGS10",
    "30Y": "DGS30",
}
TREASURY_CURVE_ORDER = ["2Y", "10Y", "30Y"]


def _fetch_fred_series(series_id: str, start: str | None = None,
                       timeout: int = 15, retries: int = 1) -> list[tuple[str, float]]:
    """Fetch one FRED series via the public no-key CSV endpoint.

    `start` bounds the download to observations on/after that date (FRED `cosd`
    param). These ICE BofA series go back ~25 years; left unbounded the CSV is
    multi-MB and times out, so the caller passes a ~2-year window — all that's
    needed for last/prev values + the prior-year-end YTD baseline.

    Returns an ascending list of `(YYYY-MM-DD, value)` with missing observations
    (`.`) dropped. Returns `[]` on any network/parse failure — the caller then
    renders the rest of the message without the credit block (warn-and-proceed,
    never a hard stop). One retry with a short backoff guards the wake/catch-up
    network race. Parses positionally (field 0 = date, field 1 = value) so it's
    robust to FRED's header-name changes (`DATE` → `observation_date`).
    """
    url = FRED_CSV_URL.format(series_id=series_id, start=start or "1900-01-01")
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    else:
        print(f"[WARN] FRED fetch failed for {series_id}: {last_err}")
        return []

    out: list[tuple[str, float]] = []
    for line in resp.text.splitlines()[1:]:  # skip header row
        parts = line.split(",")
        if len(parts) < 2:
            continue
        d, v = parts[0].strip(), parts[1].strip()
        if not v or v == ".":
            continue
        try:
            out.append((d, float(v)))
        except ValueError:
            continue
    return out


def fetch_credit_indices() -> dict:
    """Fetch HY/IG effective yields + OAS spreads from FRED for the `_Credit_`
    returns sub-group.

    Returns `{key: {label, yield_level, yield_bp_chg, [oas_bp, oas_bp_chg],
    [yield_ytd_bp], [oas_ytd_bp]}}` for each of HY/IG that resolves. Levels are
    in percent; the `*_bp*` fields are basis points (yield/OAS delta ×100). The
    two YTD fields are the year-to-date moves in bp from each metric's prior
    calendar-year-end level — one for the yield, one for the OAS spread — and
    are rendered separately labeled (a % "return" on a yield/spread is
    misleading, and a single YTD figure is ambiguous when both a yield and a
    spread are shown). A series that fails to resolve is omitted; an empty dict
    means the whole credit block is skipped.
    """
    prior_year = str(today_et().year - 1)
    # Bound the FRED download to Jan 1 of the prior year — enough for last/prev
    # values plus the prior-year-end YTD baseline, and small enough to be fast.
    start = f"{prior_year}-01-01"
    results: dict = {}
    for key in CREDIT_ORDER:
        cfg = CREDIT_SERIES[key]
        yseries = _fetch_fred_series(cfg["yield_id"], start=start)
        if len(yseries) < 2:
            print(f"[WARN] credit: insufficient yield data for {key}, skipping")
            continue
        y_last = yseries[-1][1]
        y_prev = yseries[-2][1]
        entry = {
            "label": cfg["label"],
            "yield_level": y_last,
            "yield_bp_chg": (y_last - y_prev) * 100,
        }
        prior_year_obs = [v for (d, v) in yseries if d[:4] == prior_year]
        if prior_year_obs:
            entry["yield_ytd_bp"] = (y_last - prior_year_obs[-1]) * 100

        oseries = _fetch_fred_series(cfg["oas_id"], start=start)
        if len(oseries) >= 2:
            o_last = oseries[-1][1]
            o_prev = oseries[-2][1]
            entry["oas_bp"] = o_last * 100
            entry["oas_bp_chg"] = (o_last - o_prev) * 100
            oas_prior_obs = [v for (d, v) in oseries if d[:4] == prior_year]
            if oas_prior_obs:
                entry["oas_ytd_bp"] = (o_last - oas_prior_obs[-1]) * 100

        results[key] = entry
    return results


def fetch_treasury_curve() -> dict:
    """Fetch the US Treasury 2/10/30 yield curve from FRED for the
    `_Treasury Curve_` returns sub-group.

    Returns `{key: {label, level, bp_chg, [ytd_bp]}}` for each maturity that
    resolves (key in TREASURY_CURVE_ORDER). `level` is the yield in percent as of
    the most recent FRED observation — the prior business-day close, since FRED's
    daily constant-maturity yields publish next morning ("yesterday's close").
    `bp_chg` is the 1-day move in basis points (latest minus prior observation,
    which skips weekends/holidays since FRED reports those as missing). `ytd_bp`
    is the move in bp from the prior calendar year-end level. A maturity that
    fails to resolve is omitted; an empty dict means the whole curve block is
    skipped (warn-and-proceed — never a hard stop). No API key required.
    """
    prior_year = str(today_et().year - 1)
    # Bound the FRED download to Jan 1 of the prior year — enough for last/prev
    # values plus the prior-year-end YTD baseline, and small enough to be fast.
    start = f"{prior_year}-01-01"
    results: dict = {}
    for key in TREASURY_CURVE_ORDER:
        series_id = TREASURY_CURVE_SERIES[key]
        series = _fetch_fred_series(series_id, start=start)
        if len(series) < 2:
            print(f"[WARN] treasury curve: insufficient data for {key} "
                  f"({series_id}), skipping")
            continue
        last = series[-1][1]
        prev = series[-2][1]
        entry = {
            "label": key,
            "level": last,
            "bp_chg": (last - prev) * 100,
        }
        prior_year_obs = [v for (d, v) in series if d[:4] == prior_year]
        if prior_year_obs:
            entry["ytd_bp"] = (last - prior_year_obs[-1]) * 100
        results[key] = entry
    return results


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
    """Load ticker metadata (company name, sector) if available.

    A missing or unreadable file is not fatal, but it must never be silent:
    with no metadata there is no Core 1σ eligibility, no company names, and
    no sectors, so unmatched 2σ alerts get dropped by the render path. Warn
    loudly and proceed (per the no-silent-failures invariant).
    """
    if not METADATA_PATH.exists():
        print(f"[WARN] ticker_metadata.json not found at {METADATA_PATH} — "
              "no Core 1σ eligibility, company names, or sectors this run")
        return {}
    try:
        with open(METADATA_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not read ticker_metadata.json ({e}) — "
              "no Core 1σ eligibility, company names, or sectors this run")
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


def _cache_has_tickers(cache_data: dict | None) -> bool:
    """True if a screen produced at least one ticker's distribution.

    A stale or failed batch download aborts `screen_full` early and returns a
    cache dict with an empty `tickers` map (only the `date` key). Persisting
    that empty-but-fresh-dated cache would clobber the prior good cache, and
    the next morning's cached-open run would then treat it as fresh and skip
    the entire universe ("not in cache"). main() uses this guard to keep the
    previous cache instead of overwriting it with nothing.
    """
    return bool((cache_data or {}).get("tickers"))


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
        # Foreign listings are keyed in CM metadata by their base symbol
        # (GETIB.SS → GETIB), so join on the base key — otherwise every
        # foreign name false-flags as a gap. The gap itself is still reported
        # under the display ticker so the operator sees the watchlist entry.
        meta = metadata.get(to_metadata_key(t))
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

    Retries with backoff: Yahoo intermittently throttles shared CI egress IPs
    (the 2026-07-07 open/midday runs screened 64/742 and 0/742 because the
    batch came back empty/partial), and a minute's patience usually recovers.

    Note: yfinance auto_adjust=True is the default — Close prices are
    adjusted for splits and dividends.
    """
    def _coverage(data) -> float:
        """Fraction of requested tickers with at least one Close bar."""
        if data is None or data.empty:
            return 0.0
        try:
            close = data["Close"]
        except KeyError:
            return 0.0
        if len(tickers) == 1 or not hasattr(close, "columns"):
            return 1.0 if close.notna().any() else 0.0
        covered = sum(
            1 for t in tickers
            if t in close.columns and close[t].notna().any()
        )
        return covered / max(len(tickers), 1)

    # A PARTIAL batch is the real-world failure mode (07-07 returned data for
    # 64/742 — non-empty, so an emptiness check alone never retries). Accept
    # only when most tickers actually came back; otherwise retry and keep the
    # best attempt so downstream still gets whatever Yahoo would give us.
    MIN_COVERAGE = 0.60
    best = None
    best_cov = 0.0
    for attempt in range(3):
        if attempt:
            delay = (15, 45)[attempt - 1]
            print(f"[WARN] Batch download failed/partial (coverage {best_cov:.0%}); "
                  f"retry {attempt}/2 in {delay}s")
            time.sleep(delay)
        try:
            data = yf.download(
                tickers,
                start=period_start,
                end=period_end,
                progress=False,
                threads=True,
            )
        except Exception as e:
            print(f"[WARN] Batch download failed: {e}")
            continue
        cov = _coverage(data)
        if cov >= MIN_COVERAGE:
            return data
        if cov > best_cov:
            best, best_cov = data, cov
    if best is not None:
        print(f"[WARN] Batch download degraded after retries: coverage {best_cov:.0%}")
        return best
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

    Uses the trailing ``LOOKBACK_DAYS`` (252, ~1 trading year) of daily
    returns, excluding the most recent (today's) return, so that today's move
    is measured against a clean, bounded trailing distribution. The 400-calendar-day
    download window yields ~271 usable returns; without the cap μ/σ would drift
    off the documented "trailing 252 daily returns" (CLAUDE.md) and pull in
    stale volatility regimes that shrink z-scores.
    """
    daily_returns = close_series.pct_change().dropna()
    # Exclude the last return (today's) — distribution is trailing only — then
    # cap to the trailing LOOKBACK_DAYS window (matches the documented spec).
    trailing = daily_returns.iloc[:-1].tail(LOOKBACK_DAYS)
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
    # Download under yfinance symbols (foreign names need them) but key the
    # returned dict by the display ticker so the caller's cache/metadata joins
    # — which are display/base-keyed — line up.
    yf_symbols = [to_yf_symbol(t) for t in tickers]
    try:
        data = yf.download(yf_symbols, period="5d", progress=False, threads=True)
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
            today = today_et()
            for ticker in tickers:
                yf_sym = to_yf_symbol(ticker)
                try:
                    close_col = data["Close"][yf_sym].dropna()
                    open_col = data["Open"][yf_sym].dropna()
                    # Per-ticker date alignment: the shared batch index having
                    # today's bar does NOT mean THIS ticker does. If today's
                    # open is missing, skip rather than score yesterday's open
                    # as today (and prev_close would slip to two-days-ago).
                    if len(open_col) == 0 or open_col.index[-1].date() != today:
                        print(f"[WARN] No today bar for {ticker} in cached-open batch, skipping")
                        continue
                    if len(close_col) >= 2 and len(open_col) >= 1:
                        today_open = float(open_col.iloc[-1])
                        # prev_close = last close strictly before today, so a
                        # missing today-close can't push us to two-days-ago.
                        before_today = close_col[close_col.index.date < today]
                        prev_close = (float(before_today.iloc[-1])
                                      if len(before_today) else float(close_col.iloc[-2]))
                        prices[ticker] = {"prev_close": prev_close, "today_open": today_open}
                except (KeyError, IndexError):
                    continue
    except Exception as e:
        print(f"[WARN] Today's price batch download failed: {e}")
        for ticker in tickers:
            time.sleep(random.uniform(1, 2))
            try:
                d = yf.download(to_yf_symbol(ticker), period="5d", progress=False)
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
        mkey = to_metadata_key(ticker)
        meta = (metadata or {}).get(mkey, {})
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
                meta, mkey, portfolio_set, researching_set,
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
                "in_portfolio": mkey in (portfolio_set or set()),
                "in_researching": mkey in (researching_set or set()),
                "in_following_for_interest": mkey in (following_set or set()),
                "in_ready_to_buy": mkey in (ready_to_buy_set or set()),
                "in_ready_to_short": mkey in (ready_to_short_set or set()),
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
                         ready_to_short_set: set[str] | None = None,
                         meta_key: str | None = None,
                         require_current_bar: bool = False) -> tuple[dict | None, dict | None, dict | None, dict | None, str | None]:
    """Process a single ticker in full-screen mode.

    `ticker` is the display identity (stamped on alerts/cache/stats — the
    Slack label and cache key). `meta_key` is the key Coverage Manager uses
    for ticker_metadata.json + the position sets; it differs from `ticker`
    only for foreign listings (GETIB.SS display → GETIB meta key). Defaults
    to `ticker` for US names, so existing callers/tests are unaffected.

    `require_current_bar` (set True by `screen_full`) enforces PER-TICKER date
    alignment: the shared batch index carrying today's bar does NOT guarantee
    THIS ticker has today's bar — a ticker with no non-null value today gets
    `dropna()`'d back to yesterday and would otherwise be z-scored on a stale
    bar (yesterday's move labelled "today" → a spurious 2σ). When enabled and
    the mode-relevant series is date-indexed and its latest bar is not today,
    the ticker is skipped with reason `stale_bar` instead of scored. Left False
    for plain-index test fixtures / callers that already validated freshness.

    Returns (alert_or_none, cache_entry_or_none, hi_lo_or_none, ticker_stats_or_none, skip_reason_or_none).
    ticker_stats is always populated when computation succeeds (used for sector ETF returns).
    skip_reason is set (and other values None) when the ticker cannot be screened.
    """
    meta_key = meta_key or ticker
    if len(close) < 32:
        print(f"[WARN] {ticker}: insufficient data ({len(close)} days), skipping")
        return None, None, None, None, "insufficient_history"

    if require_current_bar:
        # For open mode today's price comes from the open series; for
        # close/midday it comes from the close series. Validate whichever one
        # actually drives `today_price` below. Only enforced when the series is
        # date-indexed (real market data) — plain-index fixtures pass through.
        price_series = open_prices if mode == "open" else close
        try:
            latest_bar = price_series.index[-1].date()
        except (AttributeError, IndexError, TypeError):
            latest_bar = None
        if latest_bar is not None and latest_bar != today_et():
            print(f"[WARN] {ticker}: latest {mode} bar is {latest_bar}, "
                  f"expected {today_et()} — skipping (stale bar)")
            return None, None, None, None, "stale_bar"

    mu, sigma, sample_size = compute_distribution(close)
    # An unusable distribution is either NaN (too little history) OR a
    # degenerate zero sigma (30+ identical prior closes). A zero sigma cannot
    # produce a real z-score — compute_z_score returns 0.0, which would
    # silently swallow a genuine move (no alert AND no skip record). Treat it
    # as a skip so the name surfaces in the skip log instead of vanishing.
    if np.isnan(mu) or np.isnan(sigma) or sigma == 0:
        print(f"[WARN] {ticker}: unusable distribution (mu={mu}, sigma={sigma}), skipping")
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
        # prev_close must be the last close STRICTLY before today. When today's
        # (partial) close is present, iloc[-2] is yesterday — correct. But if
        # today's close is absent/NaN (dropna'd), iloc[-2] would be two-days-ago.
        # Select by date when the series is date-indexed to stay anchored on
        # yesterday's close regardless.
        try:
            before_today = close[close.index.date < today_et()]
            if len(before_today):
                prev_close = float(before_today.iloc[-1])
        except (AttributeError, TypeError):
            pass
    else:
        today_price = float(close.iloc[-1])

    today_return = (today_price - prev_close) / prev_close
    z = compute_z_score(today_return, mu, sigma)

    ytd_return_pct = None
    if prior_year_end_close:
        ytd_return_pct = (today_price - prior_year_end_close) / prior_year_end_close * 100

    meta = (metadata or {}).get(meta_key, {})
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
            meta, meta_key, portfolio_set, researching_set,
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
            "in_portfolio": meta_key in (portfolio_set or set()),
            "in_researching": meta_key in (researching_set or set()),
            "in_following_for_interest": meta_key in (following_set or set()),
            "in_ready_to_buy": meta_key in (ready_to_buy_set or set()),
            "in_ready_to_short": meta_key in (ready_to_short_set or set()),
        }

    # 52-week high/low check (only when high/low data is provided)
    hi_lo = None
    if high_series is not None and low_series is not None:
        result = check_52w_high_low(high_series, low_series, close)
        if result:
            # Carry the same subcategory-membership fields as alert dicts so the
            # 52-week list can be grouped by JP's taxonomy (Portfolio → … →
            # MedTech → HC Services → S&P 500 → Other) via the shared
            # SUBCATEGORIES predicates, instead of rendering as a flat list.
            hi_lo = {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "subsector": subsector,
                "type": result,
                "price": float(close.iloc[-1]),
                "in_portfolio": meta_key in (portfolio_set or set()),
                "in_researching": meta_key in (researching_set or set()),
                "in_following_for_interest": meta_key in (following_set or set()),
                "in_ready_to_buy": meta_key in (ready_to_buy_set or set()),
                "in_ready_to_short": meta_key in (ready_to_short_set or set()),
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
    stale_bar (this ticker had no today-bar in an otherwise-fresh batch),
    fallback_insufficient, fallback_exception. (Whole-batch stale aborts are
    tracked separately in stats["stale"].)

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

    # Attempt batch download. Foreign coverage names are downloaded under
    # their yfinance symbol (GETIB.SS → GETI-B.ST) while everything downstream
    # — cache key, alert identity, skip log — stays keyed by the display
    # ticker, and the CM metadata/position join uses the base symbol.
    yf_symbols = [to_yf_symbol(t) for t in tickers]
    data = batch_download(yf_symbols, start_str, end_str)
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
                yf_sym = to_yf_symbol(ticker)
                if len(tickers) == 1:
                    close = data["Close"].dropna()
                    open_prices = data["Open"].dropna()
                    high_s = data["High"].dropna()
                    low_s = data["Low"].dropna()
                else:
                    close = data["Close"][yf_sym].dropna()
                    open_prices = data["Open"][yf_sym].dropna()
                    high_s = data["High"][yf_sym].dropna()
                    low_s = data["Low"][yf_sym].dropna()

                alert, cache_entry, hi_lo, ticker_stats, skip_reason = _process_ticker_full(
                    ticker, close, open_prices, high_s, low_s, mode, metadata,
                    portfolio_set=portfolio_set,
                    researching_set=researching_set,
                    following_set=following_set,
                    ready_to_buy_set=ready_to_buy_set,
                    ready_to_short_set=ready_to_short_set,
                    meta_key=to_metadata_key(ticker),
                    require_current_bar=True,
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
        single_data = fallback_download_single(to_yf_symbol(ticker), start_str, end_str)
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
                meta_key=to_metadata_key(ticker),
                require_current_bar=True,
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
    ytd_return_pct, prior_year_end_close}}`. `prior_year_end_close` is the
    raw year-start level, used by the _Macro_ rows (the 10Y yield shows its
    year-start level + YTD basis-point move rather than a misleading % return).
    Tickers whose required year-end closes can't be located (e.g. ETF
    inception after 2024-12-31) are omitted; their Slack rows then render
    without the suffix.
    """
    if not etf_set:
        return {}

    today = today_et()
    end_date = today + timedelta(days=1)  # yfinance end is exclusive
    start_date = today - timedelta(days=800)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    tickers = sorted(etf_set)
    # Download under yfinance symbols (the alert tickers folded in here can be
    # foreign names) but key results by the display ticker so the caller's
    # join against alert/ETF rows lines up.
    yf_symbols = [to_yf_symbol(t) for t in tickers]
    try:
        data = yf.download(
            yf_symbols,
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
        yf_sym = to_yf_symbol(ticker)
        try:
            if len(tickers) == 1:
                close = data["Close"].dropna()
                open_s = data["Open"].dropna()
            else:
                close = data["Close"][yf_sym].dropna()
                open_s = data["Open"][yf_sym].dropna()
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
            "prior_year_end_close": prior_year_end_close,
        }

    return results


def format_slack_message(alerts: list[dict], mode: str, total_tickers: int,
                         stats: dict, hi_lo_hits: list[dict] | None = None,
                         sp500_set: set[str] | None = None,
                         etf_returns: list[dict] | None = None,
                         index_etf_set: set[str] | None = None,
                         global_equity_etf_set: set[str] | None = None,
                         healthcare_etf_set: set[str] | None = None,
                         tech_etf_set: set[str] | None = None,
                         commodity_etf_set: set[str] | None = None,
                         macro_etf_set: set[str] | None = None,
                         credit_data: dict | None = None,
                         curve_data: dict | None = None,
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

    # Degraded-run banner (no-silent-failures): when most tickers returned no
    # data (yfinance throttling CI egress IPs), the alert tiers AND the
    # index/sector returns groups below are silently thin — e.g. the 2026-07-07
    # open post rendered _Sectors_ with 2 of 11 ETFs and no XLV. Say so loudly
    # at the top instead of letting a hollow digest read as a quiet day.
    _screened = stats.get("screened", 0)
    if total_tickers and (_screened / total_tickers) < 0.5:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": (
            f":warning: *DEGRADED RUN — market data returned for only "
            f"{_screened}/{total_tickers} tickers.* Alert tiers and the "
            f"index/sector returns below are incomplete (missing rows, not "
            f"quiet markets). Likely transient Yahoo throttling; the next "
            f"scheduled run usually recovers."
        )}})

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

    # 52-week high/low section (close mode only). Grouped by the same
    # subcategory taxonomy as the alert tiers (Portfolio → Researching → …
    # → MedTech → HC Services → S&P 500) so the new-highs/lows readout is
    # legible by the buckets JP cares about, with an `Other` catch-all so
    # names outside every bucket are surfaced rather than hidden (unlike the
    # alert tiers, which drop unmatched names). A name matching multiple
    # buckets shows once per bucket, mirroring the alert duplication.
    if hi_lo_hits:
        blocks.append({"type": "divider"})

        highs = sorted([h for h in hi_lo_hits if h["type"] == "high"], key=lambda h: h["ticker"])
        lows = sorted([h for h in hi_lo_hits if h["type"] == "low"], key=lambda h: h["ticker"])

        def _format_hi_lo_ticker(h):
            short = short_company_name(h.get("name", ""))
            name_part = f" ({short})" if short else ""
            sector = f" [{h['sector']}]" if h.get("sector") else ""
            return f"`{h['ticker']}`{name_part}{sector}"

        def _grouped_hi_lo_lines(items):
            """Render one indented sub-line per non-empty subcategory, plus an
            `Other` line for names matching no bucket."""
            lines = []
            for label, predicate in SUBCATEGORIES:
                members = [h for h in items if predicate(h, sp500)]
                if members:
                    chips = ", ".join(_format_hi_lo_ticker(h) for h in members)
                    lines.append(f"    _{label} ({len(members)}):_  {chips}")
            # Names matching no bucket (e.g. non-Core Biotech / Specialty Pharma
            # not in S&P 500). Labeled "Uncategorized" to avoid colliding with
            # the "Other (Tech, SaaS, …)" sector subcategory above.
            others = [
                h for h in items
                if not any(predicate(h, sp500) for _, predicate in SUBCATEGORIES)
            ]
            if others:
                chips = ", ".join(_format_hi_lo_ticker(h) for h in others)
                lines.append(f"    _Uncategorized ({len(others)}):_  {chips}")
            return lines

        hi_lo_lines = []
        if highs:
            hi_lo_lines.append(f"\U0001F7E2 *52-Week Highs ({len(highs)})*")
            hi_lo_lines.extend(_grouped_hi_lo_lines(highs))
        if lows:
            if highs:
                hi_lo_lines.append("")  # blank spacer between highs and lows
            hi_lo_lines.append(f"\U0001F534 *52-Week Lows ({len(lows)})*")
            hi_lo_lines.extend(_grouped_hi_lo_lines(lows))

        # Chunk in case a heavy new-lows day overflows Slack's 3000-char limit
        # (the old single-block render could silently exceed it).
        if hi_lo_lines:
            _append_section_chunked(blocks, hi_lo_lines[0], hi_lo_lines[1:])

    # Index & sector ETF returns section. US Indices (SPYM/DIA/QQQ/^W5000/
    # ^RUT) lead, then the macro/rates/credit backdrop, then the remaining
    # ETF groups. Each equity group sorted by z-score descending so the
    # strongest move within each group leads.
    if etf_returns or credit_data or curve_data:
        _etf_returns = etf_returns or []
        idx_set = index_etf_set or set()
        global_eq_set = global_equity_etf_set or set()
        hc_set = healthcare_etf_set or set()
        tech_set = tech_etf_set or set()
        commodity_set = commodity_etf_set or set()
        macro_set = macro_etf_set or set()
        index_rows = sorted(
            [s for s in _etf_returns if s["ticker"] in idx_set],
            key=lambda s: s["z_score"], reverse=True,
        )
        global_equity_rows = sorted(
            [s for s in _etf_returns if s["ticker"] in global_eq_set],
            key=lambda s: s["z_score"], reverse=True,
        )
        healthcare_rows = sorted(
            [s for s in _etf_returns if s["ticker"] in hc_set],
            key=lambda s: s["z_score"], reverse=True,
        )
        tech_rows = sorted(
            [s for s in _etf_returns if s["ticker"] in tech_set],
            key=lambda s: s["z_score"], reverse=True,
        )
        commodity_rows = sorted(
            [s for s in _etf_returns if s["ticker"] in commodity_set],
            key=lambda s: s["z_score"], reverse=True,
        )
        sector_rows = sorted(
            [s for s in _etf_returns
             if s["ticker"] not in idx_set and s["ticker"] not in global_eq_set
             and s["ticker"] not in hc_set
             and s["ticker"] not in tech_set and s["ticker"] not in commodity_set
             and s["ticker"] not in macro_set],
            key=lambda s: s["z_score"], reverse=True,
        )
        # Macro rows render in fixed MACRO_ORDER (rates -> FX -> commodity),
        # not z-sorted — three unlike asset classes don't sort meaningfully.
        _macro_by_ticker = {s["ticker"]: s for s in _etf_returns if s["ticker"] in macro_set}
        macro_rows = [_macro_by_ticker[t] for t in MACRO_ORDER if t in _macro_by_ticker]
        # Any macro ticker not in MACRO_ORDER (shouldn't happen) still shows.
        macro_rows += [s for s in _etf_returns
                       if s["ticker"] in macro_set and s["ticker"] not in MACRO_ORDER]
        # Credit rows come from FRED (credit_data), not the yfinance returns.
        credit_rows = [k for k in CREDIT_ORDER if k in (credit_data or {})]
        # Treasury-curve rows also come from FRED (curve_data), in fixed
        # short->long maturity order.
        curve_rows = [k for k in TREASURY_CURVE_ORDER if k in (curve_data or {})]

        def _format_macro_line(s):
            """Render a macro row. Yields show level% + bp change; price/level
            rows show $price or bare level + %change. All append the z-score."""
            t = s["ticker"]
            style = MACRO_STYLE.get(t, "price")
            rp = s["return_pct"]
            z = s["z_score"]
            level = s.get("price")
            name = s.get("name") or t
            if style == "yield":
                # A bond yield RISE is a price decline, so color it like a loss
                # (red), the inverse of an equity return. Mirrors the inversion
                # in _format_credit_line (widening OAS = red) and the
                # international dashboard's yield rows. The price/level styles
                # (DXY, oil) keep the normal green-on-up rule below.
                marker = "\U0001F7E5" if rp > 0 else "\U0001F7E9"
            else:
                marker = "\U0001F7E9" if rp > 0 else "\U0001F7E5"
            sign = "+" if rp > 0 else ""
            period = period_map.get(t)
            if style == "yield" and level is not None:
                # level is the yield in percent (e.g. 4.45); recover the prior
                # close from the % change to express the move in basis points.
                denom = 1 + rp / 100
                bp_part = ""
                if denom != 0:
                    prev = level / denom
                    bp_part = f"  |  {(level - prev) * 100:+.1f}bp"
                core = f"{level:.2f}%{bp_part}  |  z = {z:+.2f}"
                # A "% return" on a yield is misleading, so for the 10Y we show
                # the year-start level (prior year-end close) and the YTD move
                # in basis points off it instead.
                start = period.get("prior_year_end_close") if period else None
                if start:
                    core += f"  |  YTD: {(level - start) * 100:+.1f}bp from {start:.2f}%"
            elif style == "price" and level is not None:
                core = f"${level:.2f}  |  {sign}{rp:.2f}%  |  z = {z:+.2f}"
                ytd = period.get("ytd_return_pct") if period else None
                if ytd is not None:
                    core += f"  |  YTD: {'+' if ytd > 0 else ''}{ytd:.2f}%"
            elif level is not None:  # bare level (e.g. DXY)
                core = f"{level:.2f}  |  {sign}{rp:.2f}%  |  z = {z:+.2f}"
                ytd = period.get("ytd_return_pct") if period else None
                if ytd is not None:
                    core += f"  |  YTD: {'+' if ytd > 0 else ''}{ytd:.2f}%"
            else:
                core = f"{sign}{rp:.2f}%  |  z = {z:+.2f}"
            return f"{marker}  `{t}` ({name})  |  {core}"

        def _format_credit_line(key, d):
            """Render a credit row, e.g.:
            `HY (US High Yield) | yield 7.42% +3.1bp | OAS 312bp +5bp |
             YTD: yield +18bp, OAS +12bp`.
            Each metric shows level + 1-day bp change inline; the trailing YTD
            segment carries BOTH year-to-date changes (yield level and OAS
            spread, in absolute bp), each labeled, so it's never ambiguous which
            move the YTD figure refers to. Colored by the spread move (widening =
            risk-off = red, tightening = green) — the OAS change is THE credit
            signal; falls back to the yield change when OAS is unavailable.
            Yields/spreads are levels, so no z-score and no $price."""
            color_chg = d.get("oas_bp_chg")
            if color_chg is None:
                color_chg = d.get("yield_bp_chg", 0.0)
            marker = "\U0001F7E5" if color_chg > 0 else "\U0001F7E9"
            parts = [f"yield {d['yield_level']:.2f}% {d['yield_bp_chg']:+.1f}bp"]
            if "oas_bp" in d:
                parts.append(f"OAS {d['oas_bp']:.0f}bp {d['oas_bp_chg']:+.0f}bp")
            ytd_bits = []
            if "yield_ytd_bp" in d:
                ytd_bits.append(f"yield {d['yield_ytd_bp']:+.0f}bp")
            if "oas_ytd_bp" in d:
                ytd_bits.append(f"OAS {d['oas_ytd_bp']:+.0f}bp")
            if ytd_bits:
                parts.append("YTD: " + ", ".join(ytd_bits))
            label = d.get("label") or key
            return f"{marker}  `{key}` ({label})  |  " + "  |  ".join(parts)

        def _format_curve_line(key, d):
            """Render a Treasury-curve row, e.g.:
            `🟥 `10Y` | 4.41% | +5.0bp | YTD: +33bp`.
            Level (prior-close yield, percent) + 1-day bp move + YTD bp from the
            prior year-end. Colored like a bond: a yield RISE is a price decline
            so it's red (🟥), a yield FALL is green (🟩) — the inverse of an
            equity return (same inversion as the credit OAS row). Yields are
            levels, so no z-score and no $price."""
            bp = d["bp_chg"]
            marker = "\U0001F7E5" if bp > 0 else "\U0001F7E9"
            parts = [f"{d['level']:.2f}%", f"{bp:+.1f}bp"]
            if "ytd_bp" in d:
                parts.append(f"YTD: {d['ytd_bp']:+.0f}bp")
            return f"{marker}  `{key}`  |  " + "  |  ".join(parts)

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

        if (index_rows or global_equity_rows or sector_rows or healthcare_rows
                or tech_rows or commodity_rows or macro_rows or credit_rows
                or curve_rows):
            blocks.append({"type": "divider"})
            header = ":chart_with_upwards_trend: *Index, Sector & Macro Returns*"
            lines = []
            rendered_any = False
            # US indices LEAD the section (US index price returns first), per
            # issue #20 — the broad-market read is the headline, the macro /
            # rates / credit backdrop follows.
            if index_rows:
                lines.append("_US Indices_")
                lines.extend(_format_etf_line(s) for s in index_rows)
                rendered_any = True
            if macro_rows:
                if rendered_any:
                    lines.append("")  # blank spacer between groups
                lines.append("_Macro_")
                lines.extend(_format_macro_line(s) for s in macro_rows)
                rendered_any = True
            if curve_rows:
                if rendered_any:
                    lines.append("")  # blank spacer between groups
                lines.append("_Treasury Curve_ (prior close, FRED)")
                lines.extend(_format_curve_line(k, curve_data[k]) for k in curve_rows)
                rendered_any = True
            if credit_rows:
                if rendered_any:
                    lines.append("")  # blank spacer between groups
                lines.append("_Credit_")
                lines.extend(_format_credit_line(k, credit_data[k]) for k in credit_rows)
                rendered_any = True
            if global_equity_rows:
                if rendered_any:
                    lines.append("")  # blank spacer between groups
                lines.append("_Global Equity_")
                lines.extend(_format_etf_line(s) for s in global_equity_rows)
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
            if commodity_rows:
                if rendered_any:
                    lines.append("")  # blank spacer between groups
                lines.append("_Commodities_")
                lines.extend(_format_etf_line(s) for s in commodity_rows)
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
    global_equity_etf_set = load_global_equity_etfs()
    sector_etf_set = load_sector_etfs()
    healthcare_etf_set = load_healthcare_etfs()
    tech_etf_set = load_tech_etfs()
    commodity_etf_set = load_commodity_etfs()
    macro_set = load_macro()
    # Global-equity + tech-theme + commodity + macro tickers join etf_set so
    # they share the download path, alert suppression, and missing-metadata
    # exemption — but render under their own sub-headers.
    etf_set = (index_etf_set | global_equity_etf_set | sector_etf_set
               | healthcare_etf_set | tech_etf_set | commodity_etf_set | macro_set)
    if etf_set:
        print(
            f"[INFO] Loaded {len(index_etf_set)} index ETFs + "
            f"{len(global_equity_etf_set)} global-equity ETFs + "
            f"{len(sector_etf_set)} sector ETFs + "
            f"{len(healthcare_etf_set)} healthcare ETFs + "
            f"{len(tech_etf_set)} tech-theme ETFs + "
            f"{len(commodity_etf_set)} commodity ETFs + "
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

    # Make sure every Coverage Manager Position-list name is screened even when
    # it's absent from watchlist.txt. `sync_watchlist.py` only merges
    # `sources/*.txt`, so held / researched / followed / trigger-ready names CM
    # pushes via the position JSONs (e.g. SPOT, TSM, WOOF, CROX) would otherwise
    # silently receive NO 1σ/2σ scan and NO skip event. Union them in the same
    # way ETFs are backfilled above so they're always in the universe (they're
    # already 1σ-eligible via _is_one_sigma_eligible). New names not yet in the
    # distribution cache get scored on the next full/close run that caches them.
    position_universe = (portfolio_set | researching_set | following_set
                         | ready_to_buy_set | ready_to_short_set)
    missing_positions = [t for t in sorted(position_universe) if t not in tickers]
    if missing_positions:
        print(f"[INFO] Adding {len(missing_positions)} Position-list ticker(s) absent "
              f"from watchlist: {missing_positions}")
        tickers = tickers + missing_positions

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
        if _cache_has_tickers(cache_data):
            save_cache(cache_data)
            print(f"[INFO] Cache saved with {len(cache_data['tickers'])} tickers")
        else:
            print("[WARN] Screen produced an empty cache (stale/failed batch) — "
                  "keeping the previous distribution cache rather than "
                  "overwriting it with an empty one")
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
    # index/sector ETFs, the alert tickers, AND the macro tickers. Alert/ETF
    # rows show the `2025: ±% | YTD: ±%` suffix; macro rows use the same fetch
    # for YTD too — WTI/dollar index as a YTD % return, and the 10Y yield as a
    # YTD basis-point move from its year-start level (a % return on a yield is
    # misleading, so we surface the level + bp move instead).
    alert_tickers = {a["ticker"] for a in alerts}
    period_fetch_set = etf_set | alert_tickers
    etf_period_returns = fetch_etf_period_returns(period_fetch_set, args.mode)
    if etf_period_returns:
        print(f"[INFO] Period returns computed: {len(etf_period_returns)} tickers "
              f"({len(alert_tickers)} alert + ETFs + macro)")

    # Credit indices (HY/IG effective yields + OAS spreads) from FRED's no-key
    # CSV. A separate fetch from the yfinance path — these are levels, not
    # tickers. Failure is non-fatal: an empty dict just omits the _Credit_ block.
    credit_data = fetch_credit_indices()
    if credit_data:
        print(f"[INFO] Credit indices fetched from FRED: {sorted(credit_data)}")
    else:
        print("[WARN] No credit data from FRED — _Credit_ block will be omitted")

    # US Treasury 2/10/30 yield curve from FRED's no-key CSV (DGS2/10/30).
    # Prior-close snapshot across maturities — separate from the intraday ^TNX
    # 10Y in _Macro_. Failure is non-fatal: an empty dict omits the curve block.
    curve_data = fetch_treasury_curve()
    if curve_data:
        print(f"[INFO] Treasury curve fetched from FRED: {sorted(curve_data)}")
    else:
        print("[WARN] No treasury-curve data from FRED — _Treasury Curve_ block will be omitted")

    # Send to Slack
    payload = format_slack_message(
        alerts, args.mode, len(tickers), stats, hi_lo_hits, sp500_set,
        etf_returns=etf_returns, index_etf_set=index_etf_set,
        global_equity_etf_set=global_equity_etf_set,
        healthcare_etf_set=healthcare_etf_set,
        tech_etf_set=tech_etf_set,
        commodity_etf_set=commodity_etf_set,
        macro_etf_set=macro_set,
        credit_data=credit_data,
        curve_data=curve_data,
        etf_period_returns=etf_period_returns,
    )
    send_slack(payload)

    # Persist the returns snapshot + rebuild the interactive return-map HTML
    # (BlackRock-style periodic table of returns). Reuses the returns already
    # computed above — no extra market-data fetch. Warn-and-proceed: a failure
    # here must never break the alert pipeline.
    try:
        import return_map
        snapshot = return_map.assemble_snapshot(
            etf_returns, etf_period_returns,
            index_set=index_etf_set,
            global_equity_set=global_equity_etf_set,
            sector_set=sector_etf_set,
            healthcare_set=healthcare_etf_set,
            tech_set=tech_etf_set,
            commodity_set=commodity_etf_set,
            macro_set=macro_set,
            macro_style=MACRO_STYLE,
            mode=args.mode,
            ref_date=stats.get("ref_date", ""),
        )
        snap_path = return_map.write_snapshot(snapshot)
        html_path = return_map.write_html(snapshot)
        print(f"[INFO] Return map updated: {snap_path.name} + {html_path}")
    except Exception as e:  # noqa: BLE001 — non-fatal by design
        print(f"[WARN] Return-map generation failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
