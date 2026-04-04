"""
Caching helpers for the sigma screener.

The distribution cache stores per-ticker mu and sigma so that the morning
open-mode run can skip downloading full 1-year history and only fetch
today's opening prices.

Cache format (JSON):
{
  "date": "2026-04-03",
  "tickers": {
    "AAPL": {"mu": 0.0008, "sigma": 0.018, "sample_size": 251},
    ...
  }
}

This module is kept separate for clarity but the core screener handles
all cache reads/writes directly. These utilities are available for any
future tooling that needs to inspect or manipulate the cache outside of
the main screening workflow.
"""

import json
from datetime import datetime
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "distribution_cache.json"


def read_cache() -> dict | None:
    """Read and return the cache dict, or None if missing/corrupt."""
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def cache_age_days() -> int | None:
    """Return the number of calendar days since the cache was written."""
    cache = read_cache()
    if not cache or "date" not in cache:
        return None
    try:
        cache_date = datetime.strptime(cache["date"], "%Y-%m-%d").date()
        return (datetime.now().date() - cache_date).days
    except ValueError:
        return None


def cache_summary() -> str:
    """Return a human-readable summary of the cache state."""
    cache = read_cache()
    if cache is None:
        return "No cache found"
    date = cache.get("date", "unknown")
    n_tickers = len(cache.get("tickers", {}))
    age = cache_age_days()
    return f"Cache date: {date} | Tickers: {n_tickers} | Age: {age} day(s)"


def invalidate_cache() -> bool:
    """Delete the cache file. Returns True if deleted, False if not found."""
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
        return True
    return False
