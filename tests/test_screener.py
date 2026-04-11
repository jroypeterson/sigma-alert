"""Tests for sigma screener core logic."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Adjust path so we can import the screener
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import sigma_screener
from sigma_screener import (
    check_52w_high_low,
    compute_distribution,
    compute_z_score,
    format_slack_message,
    is_cache_fresh,
    load_core_watchlist,
    validate_bar_date,
)


# ---------------------------------------------------------------------------
# validate_bar_date
# ---------------------------------------------------------------------------

class TestValidateBarDate:
    def test_today_bar_is_valid(self):
        today = date.today()
        index = pd.DatetimeIndex([today - timedelta(days=1), today])
        assert validate_bar_date(index, "close") is True

    def test_yesterday_bar_is_stale(self):
        yesterday = date.today() - timedelta(days=1)
        index = pd.DatetimeIndex([yesterday - timedelta(days=1), yesterday])
        assert validate_bar_date(index, "close") is False

    def test_empty_index_is_invalid(self):
        index = pd.DatetimeIndex([])
        assert validate_bar_date(index, "open") is False


# ---------------------------------------------------------------------------
# is_cache_fresh
# ---------------------------------------------------------------------------

class TestCacheFreshness:
    def _make_cache(self, days_ago: int) -> dict:
        d = (date.today() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        return {"date": d, "tickers": {}}

    def test_yesterday_is_fresh(self):
        assert is_cache_fresh(self._make_cache(1)) is True

    def test_3_days_ago_is_fresh(self):
        # Covers Friday->Monday
        assert is_cache_fresh(self._make_cache(3)) is True

    def test_4_days_ago_is_stale(self):
        assert is_cache_fresh(self._make_cache(4)) is False

    def test_today_is_not_fresh(self):
        # Cache from today means it hasn't been through a trading day yet
        assert is_cache_fresh(self._make_cache(0)) is False

    def test_none_cache(self):
        assert is_cache_fresh(None) is False

    def test_missing_date_key(self):
        assert is_cache_fresh({"tickers": {}}) is False

    def test_bad_date_format(self):
        assert is_cache_fresh({"date": "not-a-date"}) is False


# ---------------------------------------------------------------------------
# compute_distribution
# ---------------------------------------------------------------------------

class TestComputeDistribution:
    def test_normal_series(self):
        # 100 days of prices with known small daily returns
        np.random.seed(42)
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100)))
        mu, sigma, n = compute_distribution(prices)
        assert not np.isnan(mu)
        assert not np.isnan(sigma)
        assert sigma > 0
        # n should be len-2 (one lost to pct_change, one excluded as "today")
        assert n == 98

    def test_insufficient_data(self):
        prices = pd.Series([100, 101, 102, 103, 104])
        mu, sigma, n = compute_distribution(prices)
        assert np.isnan(mu)
        assert n == 0

    def test_minimum_viable_length(self):
        # 32 prices -> 31 returns -> 30 trailing (excluding today)
        np.random.seed(0)
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.01, 32)))
        mu, sigma, n = compute_distribution(prices)
        assert not np.isnan(mu)
        assert n == 30


# ---------------------------------------------------------------------------
# compute_z_score
# ---------------------------------------------------------------------------

class TestComputeZScore:
    def test_zero_sigma(self):
        assert compute_z_score(0.05, 0.001, 0.0) == 0.0

    def test_nan_sigma(self):
        assert compute_z_score(0.05, 0.001, np.nan) == 0.0

    def test_normal_calculation(self):
        z = compute_z_score(0.05, 0.001, 0.02)
        assert abs(z - 2.45) < 0.01

    def test_negative_return(self):
        z = compute_z_score(-0.05, 0.001, 0.02)
        assert z < -2.0


# ---------------------------------------------------------------------------
# Watchlist sync dedup
# ---------------------------------------------------------------------------

class TestWatchlistSync:
    def test_dedup_across_sources(self, tmp_path):
        """Verify that tickers appearing in multiple sources are de-duplicated."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from sync_watchlist import load_source

        # Create two source files with overlapping tickers
        src1 = tmp_path / "src1.txt"
        src1.write_text("AAPL\nMSFT\nGOOG\n")
        src2 = tmp_path / "src2.txt"
        src2.write_text("MSFT\nAMZN\nGOOG\n")

        tickers1 = load_source(src1)
        tickers2 = load_source(src2)

        seen = set()
        unique1 = [t for t in tickers1 if t.upper() not in seen and not seen.add(t.upper())]
        unique2 = [t for t in tickers2 if t.upper() not in seen and not seen.add(t.upper())]

        assert unique1 == ["AAPL", "MSFT", "GOOG"]
        assert unique2 == ["AMZN"]  # MSFT and GOOG already seen

    def test_missing_source_returns_empty(self, tmp_path):
        from sync_watchlist import load_source
        result = load_source(tmp_path / "nonexistent.txt")
        assert result == []


# ---------------------------------------------------------------------------
# Alert vs no-alert behavior
# ---------------------------------------------------------------------------

class TestAlertBehavior:
    def test_no_alert_below_threshold(self):
        """A z-score below 2.0 should not trigger."""
        z = compute_z_score(0.01, 0.001, 0.02)  # z ≈ 0.45
        assert abs(z) < 2.0

    def test_alert_above_threshold(self):
        """A z-score above 2.0 should trigger."""
        z = compute_z_score(0.05, 0.001, 0.02)  # z ≈ 2.45
        assert abs(z) >= 2.0

    def test_three_sigma_flag(self):
        """A z-score above 3.0 should flag as three-sigma."""
        z = compute_z_score(0.07, 0.001, 0.02)  # z ≈ 3.45
        assert abs(z) >= 3.0


# ---------------------------------------------------------------------------
# 52-week high/low detection
# ---------------------------------------------------------------------------

class TestCheck52wHighLow:
    def _make_series(self, values):
        return pd.Series(values, dtype=float)

    def test_new_high(self):
        # Trailing highs peak at 150, today hits 155
        highs = self._make_series([140, 145, 150, 148, 155])
        lows = self._make_series([130, 135, 140, 138, 145])
        close = self._make_series([135, 142, 148, 145, 153])
        assert check_52w_high_low(highs, lows, close) == "high"

    def test_new_low(self):
        # Trailing lows bottom at 100, today hits 95
        highs = self._make_series([120, 115, 110, 108, 105])
        lows = self._make_series([110, 105, 100, 102, 95])
        close = self._make_series([115, 108, 103, 105, 97])
        assert check_52w_high_low(highs, lows, close) == "low"

    def test_no_extreme(self):
        # Today is within the trailing range
        highs = self._make_series([150, 148, 145, 147, 146])
        lows = self._make_series([130, 132, 135, 133, 134])
        close = self._make_series([140, 140, 140, 140, 140])
        assert check_52w_high_low(highs, lows, close) is None

    def test_equal_to_prior_high(self):
        # Touching the exact prior high counts as a 52-week high
        highs = self._make_series([140, 150, 145, 148, 150])
        lows = self._make_series([130, 135, 135, 138, 140])
        close = self._make_series([135, 148, 142, 145, 149])
        assert check_52w_high_low(highs, lows, close) == "high"

    def test_equal_to_prior_low(self):
        # Touching the exact prior low counts as a 52-week low
        highs = self._make_series([120, 115, 118, 116, 114])
        lows = self._make_series([110, 100, 105, 103, 100])
        close = self._make_series([115, 105, 110, 108, 102])
        assert check_52w_high_low(highs, lows, close) == "low"

    def test_insufficient_data(self):
        highs = self._make_series([100])
        lows = self._make_series([90])
        close = self._make_series([95])
        assert check_52w_high_low(highs, lows, close) is None


# ---------------------------------------------------------------------------
# Slack subcategory rendering
# ---------------------------------------------------------------------------

class TestSlackSubcategories:
    def _make_alert(self, ticker, sector, tier="2sigma", z=2.5, ret=5.0, price=100.0):
        return {
            "ticker": ticker,
            "name": f"{ticker} Corp",
            "sector": sector,
            "z_score": z,
            "return_pct": ret,
            "price": price,
            "direction": "up" if ret > 0 else "down",
            "three_sigma": abs(z) >= 3.0,
            "tier": tier,
        }

    def _all_text(self, payload):
        parts = []
        for b in payload["blocks"]:
            if b.get("type") == "section":
                parts.append(b["text"]["text"])
        return "\n".join(parts)

    def test_ticker_in_multiple_categories_shown_twice(self):
        alerts = [self._make_alert("UNH", "Healthcare Services")]
        sp500 = {"UNH"}
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, sp500)
        text = self._all_text(payload)
        assert text.count("*UNH*") == 2
        assert "Healthcare Services (1)" in text
        assert "S&P 500 (1)" in text

    def test_subcategory_order_and_labels(self):
        alerts = [
            self._make_alert("AAPL", "Tech", z=2.1),
            self._make_alert("ISRG", "MedTech", z=2.2),
            self._make_alert("HCA", "Healthcare Services", z=2.3),
            self._make_alert("NVO", "PA", z=2.4),
        ]
        sp500 = {"AAPL"}
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, sp500)
        text = self._all_text(payload)
        # Subcategories render in the declared order
        i_hc = text.index("Healthcare Services (1)")
        i_mt = text.index("MedTech (1)")
        i_pa = text.index("Other/PA (1)")
        i_sp = text.index("S&P 500 (1)")
        assert i_hc < i_mt < i_pa < i_sp
        # A 2σ ticker matching none of the 4 categories should be dropped,
        # not rendered under an "Other" bucket.
        alerts.append(self._make_alert("ADBE", "SaaS", z=2.0))
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, sp500)
        text = self._all_text(payload)
        assert "_Other (" not in text
        assert "*ADBE*" not in text

    def test_price_rendered_in_line(self):
        alerts = [self._make_alert("UNH", "Healthcare Services", price=512.34)]
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "$512.34" in text

    def test_core_watchlist_subcategory_renders_first(self):
        """A watchlist hit should render in the Core Watchlist subcategory
        at the top of its tier, and also in its sector subcategory below."""
        alerts = [
            self._make_alert("UNH", "Healthcare Services"),
            self._make_alert("INSM", "Biopharma"),
        ]
        alerts[1]["on_watchlist"] = True  # only INSM is on the watchlist
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)

        assert "Core Watchlist (1)" in text
        # INSM appears once under Core Watchlist; UNH appears under HC Services.
        i_pw = text.index("Core Watchlist (1)")
        i_hc = text.index("Healthcare Services (1)")
        assert i_pw < i_hc, "Core Watchlist must render before sector subcategories"
        # INSM should be reachable from the Core Watchlist bucket (it wouldn't
        # otherwise appear at all because Biopharma is not in the subcategory list)
        assert "*INSM*" in text

    def test_alert_without_on_watchlist_key_defaults_false(self):
        """Legacy alert dicts without the on_watchlist key must not crash
        and must not appear under Core Watchlist."""
        alerts = [self._make_alert("UNH", "Healthcare Services")]
        # Deliberately don't set on_watchlist
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "Core Watchlist" not in text

    def test_tier_separation(self):
        alerts = [
            self._make_alert("A", "Healthcare Services", tier="2sigma", z=2.5),
            self._make_alert("B", "MedTech", tier="1sigma", z=1.5),
        ]
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "2σ+ Moves (1)" in text
        assert "1σ Moves (1)" in text


# ---------------------------------------------------------------------------
# Core watchlist loader
# ---------------------------------------------------------------------------

class TestCoreWatchlistLoader:
    def test_missing_file_returns_empty_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sigma_screener, "CORE_WATCHLIST_PATH", tmp_path / "nope.json")
        assert load_core_watchlist() == set()

    def test_valid_file_returns_ticker_set(self, tmp_path, monkeypatch):
        path = tmp_path / "core_watchlist.json"
        path.write_text(json.dumps({
            "INSM": {"buy_price": 30, "target_price": 75, "sector": "Biopharma"},
            "ISRG": {"buy_price": 300, "target_price": 500, "sector": "MedTech"},
        }))
        monkeypatch.setattr(sigma_screener, "CORE_WATCHLIST_PATH", path)
        assert load_core_watchlist() == {"INSM", "ISRG"}

    def test_malformed_file_returns_empty_set(self, tmp_path, monkeypatch):
        path = tmp_path / "core_watchlist.json"
        path.write_text("{not valid json")
        monkeypatch.setattr(sigma_screener, "CORE_WATCHLIST_PATH", path)
        assert load_core_watchlist() == set()

    def test_list_instead_of_dict_returns_empty_set(self, tmp_path, monkeypatch):
        """Payload must be a {ticker: {...}} dict; a list is rejected cleanly."""
        path = tmp_path / "core_watchlist.json"
        path.write_text(json.dumps(["INSM", "ISRG"]))
        monkeypatch.setattr(sigma_screener, "CORE_WATCHLIST_PATH", path)
        assert load_core_watchlist() == set()
