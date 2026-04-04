"""
Sigma screener: flags stocks with 2+ standard deviation price moves
against their trailing 52-week daily return distribution.

Runs in two modes:
  --mode open   : compares today's open to prior close (gap detection)
  --mode close  : compares today's close to prior close (EOD move)
"""

import argparse
import json
import os
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 252          # ~1 trading year
SIGMA_THRESHOLD = 2.0
THREE_SIGMA = 3.0

# Decision: using 400 calendar days for the yfinance download window.
# 252 trading days ≈ 365 calendar days, but we add buffer for holidays
# and weekends to ensure we always have enough data.
CALENDAR_DOWNLOAD_DAYS = 400


def load_watchlist() -> list[str]:
    """Read tickers from watchlist.txt, one per line, skip blanks/comments."""
    tickers = []
    with open(WATCHLIST_PATH) as f:
        for line in f:
            t = line.strip().upper()
            if t and not t.startswith("#"):
                tickers.append(t)
    return tickers


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
    today = datetime.now().date()
    delta = (today - cache_date).days
    # Fresh if written within the last 3 calendar days (handles weekends)
    return 0 < delta <= 3


def batch_download(tickers: list[str], period_start: str, period_end: str) -> pd.DataFrame | None:
    """Download OHLC data for all tickers in a single yfinance call."""
    try:
        # Single batch call to minimize Yahoo requests
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

    Decision: we use period='1d' which gives the current trading day's data.
    If the market hasn't opened yet, yfinance returns the prior day — the
    caller handles that by checking dates.
    """
    prices = {}
    try:
        data = yf.download(tickers, period="5d", progress=False, threads=True)
        if data.empty:
            return prices

        if len(tickers) == 1:
            # Single ticker: data is not multi-indexed
            ticker = tickers[0]
            if len(data) >= 2:
                prev_close = float(data["Close"].iloc[-2])
                today_open = float(data["Open"].iloc[-1])
                prices[ticker] = {"prev_close": prev_close, "today_open": today_open}
        else:
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
        # Fallback to individual downloads
        for ticker in tickers:
            time.sleep(random.uniform(1, 2))
            try:
                d = yf.download(ticker, period="5d", progress=False)
                if len(d) >= 2:
                    prev_close = float(d["Close"].iloc[-2])
                    today_open = float(d["Open"].iloc[-1])
                    prices[ticker] = {"prev_close": prev_close, "today_open": today_open}
            except Exception as e2:
                print(f"[WARN] Fallback today price failed for {ticker}: {e2}")
    return prices


def screen_open_cached(tickers: list[str], cache: dict) -> list[dict]:
    """Open-mode screening using cached mu/sigma — only downloads today's prices."""
    alerts = []
    prices = download_todays_prices(tickers)
    ticker_cache = cache.get("tickers", {})

    for ticker in tickers:
        if ticker not in ticker_cache:
            print(f"[INFO] {ticker} not in cache, skipping in cached open mode")
            continue
        if ticker not in prices:
            print(f"[WARN] No today price for {ticker}, skipping")
            continue

        mu = ticker_cache[ticker]["mu"]
        sigma = ticker_cache[ticker]["sigma"]
        prev_close = prices[ticker]["prev_close"]
        today_open = prices[ticker]["today_open"]
        today_return = (today_open - prev_close) / prev_close

        z = compute_z_score(today_return, mu, sigma)
        if abs(z) >= SIGMA_THRESHOLD:
            alerts.append({
                "ticker": ticker,
                "z_score": z,
                "return_pct": today_return * 100,
                "direction": "up" if today_return > 0 else "down",
                "three_sigma": abs(z) >= THREE_SIGMA,
            })
    return alerts


def screen_full(tickers: list[str], mode: str) -> tuple[list[dict], dict]:
    """Full screening: downloads history, computes distributions, returns alerts and cache data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=CALENDAR_DOWNLOAD_DAYS)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    alerts = []
    cache_data = {"date": datetime.now().strftime("%Y-%m-%d"), "tickers": {}}

    # Attempt batch download
    data = batch_download(tickers, start_str, end_str)
    failed_tickers = []

    if data is not None:
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    close = data["Close"].dropna()
                    open_prices = data["Open"].dropna()
                else:
                    close = data["Close"][ticker].dropna()
                    open_prices = data["Open"][ticker].dropna()

                if len(close) < 32:
                    print(f"[WARN] {ticker}: insufficient data ({len(close)} days), skipping")
                    continue

                mu, sigma, sample_size = compute_distribution(close)
                if np.isnan(mu):
                    print(f"[WARN] {ticker}: could not compute distribution, skipping")
                    continue

                # Save to cache
                cache_data["tickers"][ticker] = {
                    "mu": mu,
                    "sigma": sigma,
                    "sample_size": sample_size,
                }

                # Compute today's return based on mode
                prev_close = float(close.iloc[-2])
                if mode == "open":
                    today_price = float(open_prices.iloc[-1])
                else:
                    today_price = float(close.iloc[-1])

                today_return = (today_price - prev_close) / prev_close
                z = compute_z_score(today_return, mu, sigma)

                if abs(z) >= SIGMA_THRESHOLD:
                    alerts.append({
                        "ticker": ticker,
                        "z_score": z,
                        "return_pct": today_return * 100,
                        "direction": "up" if today_return > 0 else "down",
                        "three_sigma": abs(z) >= THREE_SIGMA,
                    })
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
            continue

        try:
            close = single_data["Close"].dropna()
            open_prices = single_data["Open"].dropna()

            mu, sigma, sample_size = compute_distribution(close)
            if np.isnan(mu):
                continue

            cache_data["tickers"][ticker] = {
                "mu": mu,
                "sigma": sigma,
                "sample_size": sample_size,
            }

            prev_close = float(close.iloc[-2])
            if mode == "open":
                today_price = float(open_prices.iloc[-1])
            else:
                today_price = float(close.iloc[-1])

            today_return = (today_price - prev_close) / prev_close
            z = compute_z_score(today_return, mu, sigma)

            if abs(z) >= SIGMA_THRESHOLD:
                alerts.append({
                    "ticker": ticker,
                    "z_score": z,
                    "return_pct": today_return * 100,
                    "direction": "up" if today_return > 0 else "down",
                    "three_sigma": abs(z) >= THREE_SIGMA,
                })
        except Exception as e:
            print(f"[WARN] {ticker} fallback processing failed: {e}")

    return alerts, cache_data


def format_slack_message(alerts: list[dict], mode: str, total_tickers: int) -> dict:
    """Build Slack message payload using Block Kit for clean formatting."""
    now_et = datetime.now()  # GitHub Actions will run in UTC; cron is set to ET-equivalent
    date_str = now_et.strftime("%Y-%m-%d")
    time_str = now_et.strftime("%I:%M %p")
    mode_label = "Open" if mode == "open" else "Close"

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Sigma Alert — {mode_label} {date_str}",
            },
        },
    ]

    if alerts:
        # Sort by absolute z-score descending so biggest moves are first
        alerts_sorted = sorted(alerts, key=lambda a: abs(a["z_score"]), reverse=True)

        lines = []
        for a in alerts_sorted:
            arrow = "\u2B06\uFE0F" if a["direction"] == "up" else "\u2B07\uFE0F"
            sigma_note = "  *\u26A0\uFE0F 3\u03C3+ move!*" if a["three_sigma"] else ""
            sign = "+" if a["return_pct"] > 0 else ""
            lines.append(
                f"{arrow}  *{a['ticker']}*  |  z = {a['z_score']:+.2f}  |  {sign}{a['return_pct']:.2f}%{sigma_note}"
            )

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(lines),
            },
        })
    else:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"No sigma moves detected across {total_tickers} tickers.",
            },
        })

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"Screened {total_tickers} tickers at {time_str} ET",
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
    parser.add_argument("--mode", choices=["open", "close"], required=True)
    args = parser.parse_args()

    tickers = load_watchlist()
    if not tickers:
        print("[ERROR] No tickers in watchlist")
        sys.exit(1)

    print(f"[INFO] Mode: {args.mode} | Tickers: {len(tickers)}")

    if args.mode == "open":
        # Try cached path first — avoids full history download
        cache = load_cache()
        if cache and is_cache_fresh(cache):
            print("[INFO] Using cached distributions for open-mode screening")
            alerts = screen_open_cached(tickers, cache)
        else:
            print("[INFO] Cache stale or missing, running full download for open mode")
            alerts, _ = screen_full(tickers, "open")
            # Don't save cache on open runs — only EOD updates the cache
    else:
        # Close mode: always do full download and update cache
        alerts, cache_data = screen_full(tickers, "close")
        save_cache(cache_data)
        print(f"[INFO] Cache saved with {len(cache_data['tickers'])} tickers")

    # Report results
    if alerts:
        print(f"[ALERT] {len(alerts)} sigma moves detected:")
        for a in alerts:
            print(f"  {a['ticker']}: z={a['z_score']:+.2f}, return={a['return_pct']:+.2f}%")
    else:
        print("[INFO] No sigma moves detected")

    # Send to Slack
    payload = format_slack_message(alerts, args.mode, len(tickers))
    send_slack(payload)


if __name__ == "__main__":
    main()
