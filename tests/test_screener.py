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
    _is_one_sigma_eligible,
    _process_ticker_full,
    check_52w_high_low,
    compute_distribution,
    compute_z_score,
    format_slack_message,
    is_cache_fresh,
    load_core_watchlist,
    load_portfolio,
    load_researching,
    validate_bar_date,
    write_missing_metadata_flag,
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
# 1σ tier eligibility (replaces the old sector-based ONE_SIGMA_SECTORS gate)
# ---------------------------------------------------------------------------

class TestOneSigmaEligibility:
    """1σ tier fires only on names you care about analytically:
    Coverage Manager Core flag, Portfolio, or Researching."""

    def test_core_y_is_eligible(self):
        meta = {"sector": "Tech", "core": "Y"}
        assert _is_one_sigma_eligible(meta, "AAPL", set(), set()) is True

    def test_core_blank_with_no_position_is_not_eligible(self):
        # Even if the sector used to fire 1σ under the old rule, a non-Core,
        # non-position name no longer qualifies.
        meta = {"sector": "MedTech", "core": ""}
        assert _is_one_sigma_eligible(meta, "ABT", set(), set()) is False

    def test_portfolio_membership_is_eligible(self):
        meta = {"sector": "Biopharma", "core": ""}
        assert _is_one_sigma_eligible(meta, "INSM", {"INSM"}, set()) is True

    def test_researching_membership_is_eligible(self):
        meta = {"sector": "Biopharma", "core": ""}
        assert _is_one_sigma_eligible(meta, "MRNA", set(), {"MRNA"}) is True

    def test_following_for_interest_membership_is_eligible(self):
        # Added 2026-05-11 with the Position taxonomy expansion. Following
        # is a passive-tracking bucket but still gets 1σ alerts because
        # the user wants to know about big moves on names they watch.
        meta = {"sector": "Biopharma", "core": ""}
        assert _is_one_sigma_eligible(
            meta, "FFIX", set(), set(), following_set={"FFIX"}
        ) is True

    def test_ready_to_buy_membership_is_eligible(self):
        meta = {"sector": "Biopharma", "core": ""}
        assert _is_one_sigma_eligible(
            meta, "RTBX", set(), set(), ready_to_buy_set={"RTBX"}
        ) is True

    def test_ready_to_short_membership_is_eligible(self):
        meta = {"sector": "Biopharma", "core": ""}
        assert _is_one_sigma_eligible(
            meta, "RTSX", set(), set(), ready_to_short_set={"RTSX"}
        ) is True

    def test_missing_metadata_falls_back_to_false(self):
        # Schema v3 hadn't shipped yet — a stale metadata file with no `core`
        # field is treated as "not Core". 1σ still fires for Portfolio/Researching.
        meta_no_core = {"sector": "Tech"}
        assert _is_one_sigma_eligible(meta_no_core, "AAPL", set(), set()) is False
        assert _is_one_sigma_eligible(meta_no_core, "AAPL", {"AAPL"}, set()) is True

    def test_none_meta_safe(self):
        assert _is_one_sigma_eligible(None, "AAPL", set(), set()) is False
        assert _is_one_sigma_eligible(None, "AAPL", {"AAPL"}, set()) is True


# ---------------------------------------------------------------------------
# Slack subcategory rendering
# ---------------------------------------------------------------------------

class TestSlackSubcategories:
    def _make_alert(self, ticker, sector, tier="2sigma", z=2.5, ret=5.0,
                    price=100.0, subsector=""):
        return {
            "ticker": ticker,
            "name": f"{ticker} Corp",
            "sector": sector,
            "subsector": subsector,
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
        assert text.count("`UNH`") == 2
        assert "Healthcare Services (1)" in text
        assert "S&P 500 (1)" in text

    def test_subcategory_order_and_labels(self):
        alerts = [
            self._make_alert("AAPL", "Tech", z=2.1),
            self._make_alert("ISRG", "MedTech", z=2.2),
            self._make_alert("HCA", "Healthcare Services", z=2.3),
            self._make_alert("LLY", "Biopharma", subsector="Large Pharma", z=2.4),
            self._make_alert("JPM", "Financials", z=2.5),
        ]
        sp500 = {"AAPL"}
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, sp500)
        text = self._all_text(payload)
        # Subcategories render in the declared order
        i_hc = text.index("Healthcare Services (1)")
        i_mt = text.index("MedTech (1)")
        i_lp = text.index("Large Pharma (1)")
        # The "Other" bucket label is long; match by its prefix.
        i_other = text.index("Other (Tech, SaaS, Fin")
        i_sp = text.index("S&P 500 (1)")
        assert i_hc < i_mt < i_lp < i_other < i_sp
        # A 2σ ticker that matches NO bucket should be dropped (mid-cap biotech
        # not in Portfolio/Researching, not in S&P 500).
        alerts.append(self._make_alert("MRNA", "Biopharma", subsector="Biotech", z=2.0))
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, sp500)
        text = self._all_text(payload)
        assert "`MRNA`" not in text

    def test_price_rendered_in_line(self):
        alerts = [self._make_alert("UNH", "Healthcare Services", price=512.34)]
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "$512.34" in text

    def test_ytd_rendered_in_line(self):
        """A positive YTD return renders with a + sign in the alert row."""
        alert = self._make_alert("UNH", "Healthcare Services", price=512.34)
        alert["ytd_return_pct"] = 18.42
        payload = format_slack_message([alert], "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "YTD: +18.42%" in text

    def test_negative_ytd_rendered_in_line(self):
        alert = self._make_alert("UNH", "Healthcare Services", price=512.34)
        alert["ytd_return_pct"] = -7.10
        payload = format_slack_message([alert], "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "YTD: -7.10%" in text

    def test_missing_ytd_omits_suffix(self):
        """Alerts without ytd_return_pct (e.g. recent IPO, stale cache) render
        no YTD suffix and must not crash."""
        alert = self._make_alert("UNH", "Healthcare Services", price=512.34)
        # ytd_return_pct deliberately absent
        payload = format_slack_message([alert], "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "YTD:" not in text

    def test_none_ytd_omits_suffix(self):
        """An explicit None ytd_return_pct also omits the suffix."""
        alert = self._make_alert("UNH", "Healthcare Services", price=512.34)
        alert["ytd_return_pct"] = None
        payload = format_slack_message([alert], "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "YTD:" not in text

    def test_prior_year_and_ytd_rendered_in_order(self):
        """The prior calendar-year return (from the period fetch) renders
        before the YTD return, mirroring the ETF rows."""
        alert = self._make_alert("UNH", "Healthcare Services", price=512.34)
        alert["ytd_return_pct"] = 6.10
        period = {"UNH": {
            "prior_year_label": "2025",
            "prior_year_return_pct": 24.50,
            "ytd_return_pct": 6.10,
        }}
        payload = format_slack_message(
            [alert], "close", 100, {"ref_date": "2026-04-10"}, None, set(),
            etf_period_returns=period,
        )
        text = self._all_text(payload)
        assert "2025: +24.50%" in text
        assert "YTD: +6.10%" in text
        assert text.index("2025: +24.50%") < text.index("YTD: +6.10%")

    def test_negative_prior_year_rendered(self):
        alert = self._make_alert("UNH", "Healthcare Services", price=512.34)
        alert["ytd_return_pct"] = 6.10
        period = {"UNH": {
            "prior_year_label": "2025",
            "prior_year_return_pct": -12.30,
            "ytd_return_pct": 6.10,
        }}
        payload = format_slack_message(
            [alert], "close", 100, {"ref_date": "2026-04-10"}, None, set(),
            etf_period_returns=period,
        )
        text = self._all_text(payload)
        assert "2025: -12.30%" in text

    def test_prior_year_na_when_no_period_data(self):
        """No period entry for the ticker → the prior-year column renders as
        N/A (not dropped), and the inline YTD still renders (recent-IPO case:
        prior-prior-year close unavailable). The N/A label uses the current
        year minus one."""
        alert = self._make_alert("UNH", "Healthcare Services", price=512.34)
        alert["ytd_return_pct"] = 6.10
        payload = format_slack_message([alert], "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        prior_year = str(date.today().year - 1)
        assert f"{prior_year}: N/A" in text
        assert "YTD: +6.10%" in text

    def test_ytd_falls_back_to_period_map(self):
        """When the inline YTD is absent (e.g. an old cache lacking the
        year-end close), YTD falls back to the period fetch's value."""
        alert = self._make_alert("UNH", "Healthcare Services", price=512.34)
        # No inline ytd_return_pct set.
        period = {"UNH": {
            "prior_year_label": "2025",
            "prior_year_return_pct": 24.50,
            "ytd_return_pct": 9.99,
        }}
        payload = format_slack_message(
            [alert], "close", 100, {"ref_date": "2026-04-10"}, None, set(),
            etf_period_returns=period,
        )
        text = self._all_text(payload)
        assert "YTD: +9.99%" in text

    def test_portfolio_subcategory_renders_first(self):
        """A Portfolio hit should render in the Portfolio subcategory
        at the top of its tier, and also in its sector subcategory below."""
        alerts = [
            self._make_alert("UNH", "Healthcare Services"),
            self._make_alert("INSM", "Biopharma"),
        ]
        alerts[1]["in_portfolio"] = True  # only INSM is in the portfolio
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)

        assert "Portfolio (1)" in text
        # INSM appears under Portfolio; UNH appears under HC Services.
        i_pw = text.index("Portfolio (1)")
        i_hc = text.index("Healthcare Services (1)")
        assert i_pw < i_hc, "Portfolio must render before sector subcategories"
        # INSM should be reachable from the Portfolio bucket (it wouldn't
        # otherwise appear because Biopharma is not in the subcategory list).
        assert "`INSM`" in text

    def test_researching_subcategory_renders_after_portfolio(self):
        """A Researching hit renders in its own subcategory between Portfolio and the sector blocks."""
        alerts = [
            self._make_alert("AAPL", "Other"),
            self._make_alert("ISRG", "MedTech"),
        ]
        alerts[1]["in_researching"] = True  # only ISRG is being researched
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "Researching (1)" in text
        i_re = text.index("Researching (1)")
        i_mt = text.index("MedTech (1)")
        assert i_re < i_mt, "Researching must render before sector subcategories"

    def test_new_position_subcategories_render_in_declared_order(self):
        """The three Position lists added 2026-05-11 (Ready to Buy, Ready to
        Short, Following for Interest) render between Researching and the
        sector buckets, in that order."""
        alerts = [
            self._make_alert("RTBX", "Biopharma"),
            self._make_alert("RTSX", "Biopharma"),
            self._make_alert("FFIX", "Biopharma"),
            self._make_alert("ISRG", "MedTech"),
        ]
        alerts[0]["in_ready_to_buy"] = True
        alerts[1]["in_ready_to_short"] = True
        alerts[2]["in_following_for_interest"] = True
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "Ready to Buy (1)" in text
        assert "Ready to Short (1)" in text
        assert "Following for Interest (1)" in text
        i_rb = text.index("Ready to Buy (1)")
        i_rs = text.index("Ready to Short (1)")
        i_fi = text.index("Following for Interest (1)")
        i_mt = text.index("MedTech (1)")
        assert i_rb < i_rs < i_fi < i_mt, (
            "Ready to Buy → Ready to Short → Following for Interest → sector"
        )

    def test_alert_without_position_keys_defaults_false(self):
        """Legacy alert dicts without in_portfolio/in_researching keys must not crash
        and must not appear under Portfolio or Researching subcategories."""
        alerts = [self._make_alert("UNH", "Healthcare Services")]
        # Deliberately don't set in_portfolio or in_researching
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "Portfolio (" not in text
        assert "Researching (" not in text

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
# Macro (rates / FX / commodity) returns rendering
# ---------------------------------------------------------------------------

class TestMacroRendering:
    MACRO_SET = {"^TNX", "DX-Y.NYB", "CL=F"}

    def _row(self, ticker, name, price, return_pct, z=0.5):
        return {
            "ticker": ticker, "name": name, "z_score": z,
            "return_pct": return_pct, "price": price,
            "high_52w": None, "low_52w": None,
        }

    def _text(self, etf_returns, **kw):
        payload = format_slack_message(
            [], "close", 100, {"ref_date": "2026-04-10"}, None, set(),
            etf_returns=etf_returns, macro_etf_set=self.MACRO_SET, **kw,
        )
        return "\n".join(
            b["text"]["text"] for b in payload["blocks"]
            if b.get("type") == "section"
        )

    def test_yield_renders_level_and_bp(self):
        # prev 4.00 -> 4.10 is +2.5% and +10.0bp
        text = self._text([self._row("^TNX", "10Y Treasury Yield", 4.10, 2.5, z=0.63)])
        assert "_Macro_" in text
        assert "4.10%" in text
        assert "+10.0bp" in text
        assert "z = +0.63" in text
        # A yield must NOT render as a $price.
        assert "$4.10" not in text

    def test_price_renders_dollar(self):
        text = self._text([self._row("CL=F", "WTI Crude Oil", 92.50, -1.20, z=-0.4)])
        assert "$92.50" in text
        assert "-1.20%" in text
        assert "\U0001F7E5" in text  # red marker on a down move

    def test_level_renders_bare(self):
        text = self._text([self._row("DX-Y.NYB", "US Dollar Index", 99.13, 0.30)])
        assert "99.13" in text
        assert "+0.30%" in text
        assert "$99.13" not in text  # the dollar index is a level, not a $price

    def test_macro_group_renders_first_and_separate_from_sectors(self):
        rows = [
            self._row("CL=F", "WTI Crude Oil", 92.50, 1.0),
            self._row("XLE", "Energy", 90.0, 1.0),
        ]
        text = self._text(rows, index_etf_set=set(), healthcare_etf_set=set())
        assert "Index, Sector & Macro Returns" in text
        assert "_Macro_" in text and "_Sectors_" in text
        assert text.index("_Macro_") < text.index("_Sectors_")
        # CL=F is macro, so it must appear above the _Sectors_ header, not within it.
        assert text.index("`CL=F`") < text.index("_Sectors_")

    def test_macro_order_is_fixed_not_z_sorted(self):
        # Provide rows out of order and with z that would re-sort them.
        rows = [
            self._row("CL=F", "WTI Crude Oil", 92.5, 1.0, z=9.0),
            self._row("DX-Y.NYB", "US Dollar Index", 99.1, 0.3, z=8.0),
            self._row("^TNX", "10Y Treasury Yield", 4.1, 2.5, z=0.1),
        ]
        text = self._text(rows)
        assert text.index("`^TNX`") < text.index("`DX-Y.NYB`") < text.index("`CL=F`")

    def test_yield_shows_year_start_level_and_ytd_bp(self):
        # Year started at 4.57%, now 4.10% → YTD -47.0bp from 4.57%.
        period = {"^TNX": {
            "prior_year_label": "2025",
            "prior_year_return_pct": None,
            "ytd_return_pct": -10.3,  # the misleading % — must NOT be shown
            "prior_year_end_close": 4.57,
        }}
        text = self._text(
            [self._row("^TNX", "10Y Treasury Yield", 4.10, 2.5, z=0.63)],
            etf_period_returns=period,
        )
        assert "YTD: -47.0bp from 4.57%" in text
        # The day-change bp is still present and the misleading % is not.
        assert "+10.0bp" in text
        assert "YTD: -10.30%" not in text

    def test_wti_shows_ytd_percent(self):
        period = {"CL=F": {
            "prior_year_label": "2025",
            "prior_year_return_pct": 5.0,
            "ytd_return_pct": 8.30,
            "prior_year_end_close": 70.0,
        }}
        text = self._text(
            [self._row("CL=F", "WTI Crude Oil", 92.50, -1.20, z=-0.4)],
            etf_period_returns=period,
        )
        assert "YTD: +8.30%" in text

    def test_dollar_index_shows_ytd_percent(self):
        period = {"DX-Y.NYB": {
            "prior_year_label": "2025",
            "prior_year_return_pct": 2.0,
            "ytd_return_pct": -3.10,
            "prior_year_end_close": 102.3,
        }}
        text = self._text(
            [self._row("DX-Y.NYB", "US Dollar Index", 99.13, 0.30)],
            etf_period_returns=period,
        )
        assert "YTD: -3.10%" in text

    def test_macro_without_period_data_omits_ytd(self):
        # No period entry → no YTD suffix, existing behavior preserved.
        text = self._text([self._row("^TNX", "10Y Treasury Yield", 4.10, 2.5)])
        assert "YTD:" not in text


class TestTechThemesRendering:
    TECH_SET = {"MAGS", "SMH", "IGV"}

    def _row(self, ticker, name, price, return_pct, z=0.5):
        return {
            "ticker": ticker, "name": name, "z_score": z,
            "return_pct": return_pct, "price": price,
            "high_52w": None, "low_52w": None,
        }

    def _text(self, etf_returns, **kw):
        payload = format_slack_message(
            [], "close", 100, {"ref_date": "2026-04-10"}, None, set(),
            etf_returns=etf_returns, tech_etf_set=self.TECH_SET, **kw,
        )
        return "\n".join(
            b["text"]["text"] for b in payload["blocks"]
            if b.get("type") == "section"
        )

    def test_tech_group_renders_with_holdings_name(self):
        text = self._text([self._row("SMH", "Semis: NVDA, TSM, AVGO", 250.0, 1.5)])
        assert "_Tech Themes_" in text
        assert "`SMH` (Semis: NVDA, TSM, AVGO)" in text

    def test_tech_separate_from_sectors_and_after_healthcare(self):
        rows = [
            self._row("SMH", "Semis", 250.0, 1.0, z=2.0),
            self._row("XLE", "Energy", 90.0, 1.0, z=1.0),
            self._row("XBI", "Biotech", 100.0, 1.0, z=1.0),
        ]
        text = self._text(rows, healthcare_etf_set={"XBI"})
        assert "_Sectors_" in text and "_Healthcare_" in text and "_Tech Themes_" in text
        # SMH is tech, not a sector — must not fall into the _Sectors_ group.
        assert text.index("_Healthcare_") < text.index("_Tech Themes_")
        assert text.index("`SMH`") > text.index("_Tech Themes_")
        # XLE (a real sector ETF) stays under _Sectors_, above _Tech Themes_.
        assert text.index("`XLE`") < text.index("_Tech Themes_")


class TestCommoditiesRendering:
    COMMO_SET = {"GLD"}

    def _row(self, ticker, name, price, return_pct, z=0.5):
        return {
            "ticker": ticker, "name": name, "z_score": z,
            "return_pct": return_pct, "price": price,
            "high_52w": None, "low_52w": None,
        }

    def _text(self, etf_returns, **kw):
        payload = format_slack_message(
            [], "close", 100, {"ref_date": "2026-04-10"}, None, set(),
            etf_returns=etf_returns, commodity_etf_set=self.COMMO_SET, **kw,
        )
        return "\n".join(
            b["text"]["text"] for b in payload["blocks"]
            if b.get("type") == "section"
        )

    def test_commodities_group_renders(self):
        text = self._text([self._row("GLD", "Gold", 244.10, 0.80)])
        assert "_Commodities_" in text
        assert "`GLD` (Gold)" in text

    def test_commodity_separate_from_sectors_and_after_tech(self):
        rows = [
            self._row("GLD", "Gold", 244.0, 0.8, z=2.0),
            self._row("XLE", "Energy", 90.0, 1.0, z=1.0),
            self._row("SMH", "Semis", 250.0, 1.0, z=1.0),
        ]
        text = self._text(rows, tech_etf_set={"SMH"})
        assert "_Sectors_" in text and "_Tech Themes_" in text and "_Commodities_" in text
        # Commodities renders last, after Tech Themes.
        assert text.index("_Tech Themes_") < text.index("_Commodities_")
        # GLD belongs to Commodities, not the catch-all _Sectors_ group.
        assert text.index("`GLD`") > text.index("_Commodities_")
        # XLE (a real sector ETF) stays under _Sectors_, above _Commodities_.
        assert text.index("`XLE`") < text.index("_Commodities_")


class TestGlobalEquityRendering:
    GLOBAL_SET = {"ACWI", "EFA", "EEM", "VGK", "EWJ", "EWY", "FXI", "INDA"}

    def _row(self, ticker, name, price, return_pct, z=0.5):
        return {
            "ticker": ticker, "name": name, "z_score": z,
            "return_pct": return_pct, "price": price,
            "high_52w": None, "low_52w": None,
        }

    def _text(self, etf_returns, **kw):
        payload = format_slack_message(
            [], "close", 100, {"ref_date": "2026-04-10"}, None, set(),
            etf_returns=etf_returns, global_equity_etf_set=self.GLOBAL_SET, **kw,
        )
        return "\n".join(
            b["text"]["text"] for b in payload["blocks"]
            if b.get("type") == "section"
        )

    def test_global_equity_group_renders(self):
        text = self._text([self._row("ACWI", "MSCI All-Country World", 120.0, 0.5)])
        assert "_Global Equity_" in text
        assert "`ACWI` (MSCI All-Country World)" in text

    def test_global_equity_after_indices_before_sectors_not_in_sectors(self):
        rows = [
            self._row("EEM", "Emerging Markets", 44.0, 1.0, z=2.0),
            self._row("SPYM", "S&P 500", 100.0, 1.0, z=1.0),
            self._row("XLE", "Energy", 90.0, 1.0, z=1.0),
        ]
        text = self._text(rows, index_etf_set={"SPYM"})
        assert "_Indices_" in text and "_Global Equity_" in text and "_Sectors_" in text
        # US indices first, then global equity, then sectors.
        assert text.index("_Indices_") < text.index("_Global Equity_") < text.index("_Sectors_")
        # EEM belongs to Global Equity, not the catch-all _Sectors_ group.
        assert text.index("`EEM`") > text.index("_Global Equity_")
        assert text.index("`EEM`") < text.index("_Sectors_")
        # XLE (a real sector ETF) stays under _Sectors_.
        assert text.index("`XLE`") > text.index("_Sectors_")


class TestCreditRendering:
    def _text(self, credit_data, **kw):
        payload = format_slack_message(
            [], "close", 100, {"ref_date": "2026-04-10"}, None, set(),
            credit_data=credit_data, **kw,
        )
        return "\n".join(
            b["text"]["text"] for b in payload["blocks"]
            if b.get("type") == "section"
        )

    def test_credit_renders_yield_oas_ytd(self):
        cd = {"HY": {"label": "US High Yield", "yield_level": 7.42,
                     "yield_bp_chg": 3.1, "oas_bp": 312, "oas_bp_chg": 5,
                     "yield_ytd_bp": 18, "oas_ytd_bp": 12}}
        text = self._text(cd)
        assert "_Credit_" in text
        assert "`HY` (US High Yield)" in text
        assert "yield 7.42% +3.1bp" in text
        assert "OAS 312bp +5bp" in text
        # Both YTD changes render, each labeled, in absolute bp.
        assert "YTD: yield +18bp, OAS +12bp" in text

    def test_credit_ytd_labels_each_change(self):
        # Only the yield YTD present → label it, no bare/ambiguous "YTD".
        d = {"HY": {"label": "x", "yield_level": 7.0, "yield_bp_chg": 1.0,
                    "oas_bp": 300, "oas_bp_chg": 1, "yield_ytd_bp": 20}}
        text = self._text(d)
        assert "YTD: yield +20bp" in text
        assert "OAS +" not in text.split("YTD:")[1]  # no OAS YTD when absent
        # Only the OAS YTD present → label it.
        d2 = {"IG": {"label": "y", "yield_level": 5.0, "yield_bp_chg": 0.0,
                     "oas_bp": 80, "oas_bp_chg": -1, "oas_ytd_bp": -8}}
        text2 = self._text(d2)
        assert "YTD: OAS -8bp" in text2

    def test_widening_spread_is_red_tightening_is_green(self):
        widen = {"HY": {"label": "x", "yield_level": 7.0, "yield_bp_chg": 1.0,
                        "oas_bp": 300, "oas_bp_chg": 5}}
        assert "\U0001F7E5" in self._text(widen)  # red on widening
        tighten = {"IG": {"label": "y", "yield_level": 5.0, "yield_bp_chg": -1.0,
                          "oas_bp": 80, "oas_bp_chg": -3}}
        assert "\U0001F7E9" in self._text(tighten)  # green on tightening

    def test_missing_oas_colors_by_yield_change(self):
        # No OAS field → color by the yield move; rising yield = risk-off = red.
        d = {"HY": {"label": "x", "yield_level": 7.0, "yield_bp_chg": 2.0,
                    "yield_ytd_bp": 10}}
        text = self._text(d)
        assert "OAS" not in text
        assert "\U0001F7E5" in text

    def test_missing_ytd_omits_suffix(self):
        d = {"HY": {"label": "x", "yield_level": 7.0, "yield_bp_chg": -1.0,
                    "oas_bp": 300, "oas_bp_chg": -2}}
        text = self._text(d)
        assert "YTD" not in text

    def test_credit_renders_without_etf_returns(self):
        # A credit-only message (no yfinance returns) still emits the section.
        d = {"HY": {"label": "x", "yield_level": 7.0, "yield_bp_chg": 1.0,
                    "oas_bp": 300, "oas_bp_chg": 1}}
        text = self._text(d)
        assert "Index, Sector & Macro Returns" in text
        assert "_Credit_" in text

    def test_credit_order_hy_before_ig(self):
        cd = {  # insertion order IG-first; render order must still be HY → IG
            "IG": {"label": "IG", "yield_level": 5.0, "yield_bp_chg": 0.0,
                   "oas_bp": 74, "oas_bp_chg": 1},
            "HY": {"label": "HY", "yield_level": 7.0, "yield_bp_chg": 0.0,
                   "oas_bp": 300, "oas_bp_chg": 1},
        }
        text = self._text(cd)
        assert text.index("`HY`") < text.index("`IG`")

    def test_credit_after_macro_before_indices(self):
        cd = {"HY": {"label": "HY", "yield_level": 7.0, "yield_bp_chg": 1.0,
                     "oas_bp": 300, "oas_bp_chg": 1}}
        macro_row = {"ticker": "^TNX", "name": "10Y", "z_score": 0.5,
                     "return_pct": 1.0, "price": 4.1, "high_52w": None, "low_52w": None}
        idx_row = {"ticker": "SPYM", "name": "S&P 500", "z_score": 0.5,
                   "return_pct": 1.0, "price": 100.0, "high_52w": None, "low_52w": None}
        payload = format_slack_message(
            [], "close", 100, {"ref_date": "2026-04-10"}, None, set(),
            etf_returns=[macro_row, idx_row], macro_etf_set={"^TNX"},
            index_etf_set={"SPYM"}, credit_data=cd,
        )
        text = "\n".join(b["text"]["text"] for b in payload["blocks"]
                         if b.get("type") == "section")
        assert text.index("_Macro_") < text.index("_Credit_") < text.index("_Indices_")


class TestFetchCreditIndices:
    def test_computes_bp_changes_and_ytd(self, monkeypatch):
        series = {
            "BAMLH0A0HYM2EY": [("2025-12-31", 6.50), ("2026-06-01", 6.87), ("2026-06-02", 6.88)],
            "BAMLH0A0HYM2": [("2025-12-31", 2.40), ("2026-06-01", 2.72), ("2026-06-02", 2.71)],
            "BAMLC0A0CMEY": [("2025-12-31", 4.84), ("2026-06-01", 5.14), ("2026-06-02", 5.14)],
            "BAMLC0A0CM": [("2026-06-01", 0.73), ("2026-06-02", 0.74)],
        }
        monkeypatch.setattr(sigma_screener, "_fetch_fred_series",
                            lambda sid, **kw: series.get(sid, []))
        cd = sigma_screener.fetch_credit_indices()
        assert set(cd) == {"HY", "IG"}
        assert cd["HY"]["yield_level"] == 6.88
        assert round(cd["HY"]["yield_bp_chg"], 1) == 1.0       # (6.88-6.87)*100
        assert round(cd["HY"]["yield_ytd_bp"], 0) == 38        # (6.88-6.50)*100
        assert round(cd["HY"]["oas_bp"], 0) == 271             # 2.71*100
        assert round(cd["HY"]["oas_bp_chg"], 0) == -1          # (2.71-2.72)*100
        assert round(cd["HY"]["oas_ytd_bp"], 0) == 31          # (2.71-2.40)*100
        assert round(cd["IG"]["oas_bp"], 0) == 74
        # IG OAS series has no prior-year obs → oas_ytd_bp omitted (not crash).
        assert "oas_ytd_bp" not in cd["IG"]

    def test_skips_series_with_insufficient_data(self, monkeypatch):
        # HY yield resolves; IG yield returns nothing → IG dropped, HY kept.
        series = {
            "BAMLH0A0HYM2EY": [("2026-06-01", 6.87), ("2026-06-02", 6.88)],
            "BAMLH0A0HYM2": [("2026-06-01", 2.72), ("2026-06-02", 2.71)],
        }
        monkeypatch.setattr(sigma_screener, "_fetch_fred_series",
                            lambda sid, **kw: series.get(sid, []))
        cd = sigma_screener.fetch_credit_indices()
        assert set(cd) == {"HY"}

    def test_fred_parse_drops_missing_and_is_header_agnostic(self, monkeypatch):
        class FakeResp:
            text = ("observation_date,BAMLH0A0HYM2EY\n"
                    "2026-05-30,.\n2026-06-01,6.89\n2026-06-02,6.88\n")
            def raise_for_status(self):
                pass
        monkeypatch.setattr(sigma_screener.requests, "get",
                            lambda *a, **k: FakeResp())
        out = sigma_screener._fetch_fred_series("X", start="2025-01-01")
        assert out == [("2026-06-01", 6.89), ("2026-06-02", 6.88)]

    def test_fred_network_failure_returns_empty(self, monkeypatch):
        import requests as _rq

        def boom(*a, **k):
            raise _rq.RequestException("timeout")
        monkeypatch.setattr(sigma_screener.requests, "get", boom)
        # retries=0 so the test doesn't sleep.
        assert sigma_screener._fetch_fred_series("X", retries=0) == []


class TestFiftyTwoWeekGrouping:
    """The 52-week high/low list is grouped by the alert subcategory taxonomy
    with an `Other` catch-all (added 2026-06-03)."""

    def _hilo(self, ticker, type_, sector="", subsector="", **flags):
        d = {
            "ticker": ticker, "name": f"{ticker} Corp", "sector": sector,
            "subsector": subsector, "type": type_, "price": 50.0,
            "in_portfolio": False, "in_researching": False,
            "in_following_for_interest": False, "in_ready_to_buy": False,
            "in_ready_to_short": False,
        }
        d.update(flags)
        return d

    def _text(self, hi_lo_hits, sp500=None):
        payload = format_slack_message(
            [], "close", 100, {"ref_date": "2026-04-10"}, hi_lo_hits, sp500 or set(),
        )
        return "\n".join(
            b["text"]["text"] for b in payload["blocks"]
            if b.get("type") == "section"
        )

    def test_lows_grouped_by_taxonomy_with_other(self):
        hits = [
            self._hilo("INSM", "low", sector="Biopharma", in_portfolio=True),
            self._hilo("ISRG", "low", sector="MedTech"),
            self._hilo("AAPL", "low", sector="Tech"),       # S&P 500 only
            self._hilo("ZZZZ", "low", sector="Biotech"),    # matches nothing → Other
        ]
        text = self._text(hits, sp500={"AAPL"})
        assert "*52-Week Lows (4)*" in text
        assert "_Portfolio (1):_" in text
        assert "_MedTech (1):_" in text
        assert "_S&P 500 (1):_" in text
        assert "_Uncategorized (1):_" in text
        # The uncategorized name is surfaced, not dropped.
        assert "`ZZZZ`" in text

    def test_name_in_multiple_buckets_duplicated(self):
        hits = [self._hilo("UNH", "low", sector="Healthcare Services", in_portfolio=True)]
        text = self._text(hits, sp500={"UNH"})
        # UNH is Portfolio, Healthcare Services, AND S&P 500 → three buckets.
        assert "_Portfolio (1):_" in text
        assert "_Healthcare Services (1):_" in text
        assert "_S&P 500 (1):_" in text
        assert text.count("`UNH`") == 3

    def test_highs_also_grouped(self):
        hits = [self._hilo("ISRG", "high", sector="MedTech")]
        text = self._text(hits)
        assert "*52-Week Highs (1)*" in text
        assert "_MedTech (1):_" in text

    def test_highs_and_lows_both_present(self):
        hits = [
            self._hilo("ISRG", "high", sector="MedTech"),
            self._hilo("HCA", "low", sector="Healthcare Services"),
        ]
        text = self._text(hits)
        assert "*52-Week Highs (1)*" in text
        assert "*52-Week Lows (1)*" in text
        assert text.index("52-Week Highs") < text.index("52-Week Lows")


class TestOverviewCard:
    def test_build_blocks_is_valid_and_lists_groups(self):
        import post_overview
        payload = post_overview.build_blocks()
        assert payload["blocks"][0]["type"] == "header"
        text = "\n".join(
            b["text"]["text"] for b in payload["blocks"] if b.get("type") == "section"
        )
        # Mentions each returns-block group and a couple of the live tickers,
        # proving it reads the source files rather than hard-coding.
        for token in ("_Macro_".strip("_"), "Indices", "Global Equity", "Sectors",
                      "Healthcare", "Tech Themes", "`^W5000`", "`SMH`", "`ACWI`",
                      "2σ+", "1σ"):
            assert token in text
        # Slack section hard limit.
        for b in payload["blocks"]:
            if b.get("type") == "section":
                assert len(b["text"]["text"]) < 3000


class TestLoadMacro:
    def test_load_tech_etfs_reads_tickers(self, tmp_path, monkeypatch):
        p = tmp_path / "tech.txt"
        p.write_text("# c\nMAGS\nsmh\nIGV\n")
        monkeypatch.setattr(sigma_screener, "TECH_ETFS_PATH", p)
        assert sigma_screener.load_tech_etfs() == {"MAGS", "SMH", "IGV"}

    def test_load_macro_reads_tickers(self, tmp_path, monkeypatch):
        p = tmp_path / "macro.txt"
        p.write_text("# comment\n^TNX\nDX-Y.NYB\ncl=f\n\n")
        monkeypatch.setattr(sigma_screener, "MACRO_PATH", p)
        assert sigma_screener.load_macro() == {"^TNX", "DX-Y.NYB", "CL=F"}

    def test_load_macro_missing_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sigma_screener, "MACRO_PATH", tmp_path / "nope.txt")
        assert sigma_screener.load_macro() == set()


# ---------------------------------------------------------------------------
# YTD return computation in the full-screen path
# ---------------------------------------------------------------------------

class TestYtdReturn:
    """_process_ticker_full computes YTD vs the prior calendar year-end close
    (from the downloaded history) and caches the year-end close so the morning
    cached-open path can reuse it."""

    def _make_series(self):
        """A business-day close series spanning the prior year-end through
        today, with a +10% jump on the last bar so a 2σ alert fires."""
        prior_year = date.today().year - 1
        idx = pd.bdate_range(start=f"{prior_year}-09-01", end=date.today())
        n = len(idx)
        np.random.seed(1)
        base = 100 * np.cumprod(1 + np.random.normal(0, 0.005, n - 1))
        close_vals = list(base) + [float(base[-1] * 1.10)]  # +10% today
        close = pd.Series(close_vals, index=idx)
        open_prices = close.copy()  # unused in close mode
        return close, open_prices, prior_year

    def test_ytd_on_alert_and_cache_entry(self):
        close, open_prices, prior_year = self._make_series()
        prior_year_end_close = float(close[close.index.year == prior_year].iloc[-1])
        today_price = float(close.iloc[-1])
        expected_ytd = (today_price - prior_year_end_close) / prior_year_end_close * 100

        alert, cache_entry, _hi_lo, _stats, skip = _process_ticker_full(
            "UNH", close, open_prices, None, None, "close",
        )
        assert skip is None
        # Alert fired (the +10% last bar is a multi-sigma move).
        assert alert is not None
        assert alert["ytd_return_pct"] == pytest.approx(expected_ytd, rel=1e-6)
        # Cache carries the prior year-end close + its year for the cached-open path.
        assert cache_entry["prior_year_end_close"] == pytest.approx(prior_year_end_close, rel=1e-6)
        assert cache_entry["prior_year_end_year"] == prior_year

    def test_ytd_none_when_no_prior_year_data(self):
        """A series confined to the current year has no prior-year-end close;
        YTD is None and the cache omits the year-end fields."""
        idx = pd.bdate_range(start=f"{date.today().year}-01-02", end=date.today())
        if len(idx) < 40:
            pytest.skip("too early in the year for a 40-bar current-year series")
        np.random.seed(2)
        base = 100 * np.cumprod(1 + np.random.normal(0, 0.005, len(idx) - 1))
        close = pd.Series(list(base) + [float(base[-1] * 1.10)], index=idx)
        alert, cache_entry, _hi_lo, _stats, skip = _process_ticker_full(
            "UNH", close, close.copy(), None, None, "close",
        )
        assert skip is None
        assert alert is not None
        assert alert["ytd_return_pct"] is None
        assert "prior_year_end_close" not in cache_entry
        assert "prior_year_end_year" not in cache_entry


# ---------------------------------------------------------------------------
# Portfolio / Researching / legacy core_watchlist loaders
# ---------------------------------------------------------------------------

class TestPortfolioAndResearchingLoaders:
    def _stub_paths(self, tmp_path, monkeypatch, portfolio=None, researching=None, core=None):
        monkeypatch.setattr(sigma_screener, "PORTFOLIO_PATH", tmp_path / "portfolio.json")
        monkeypatch.setattr(sigma_screener, "RESEARCHING_PATH", tmp_path / "researching.json")
        monkeypatch.setattr(sigma_screener, "CORE_WATCHLIST_PATH", tmp_path / "core_watchlist.json")
        if portfolio is not None:
            (tmp_path / "portfolio.json").write_text(json.dumps(portfolio))
        if researching is not None:
            (tmp_path / "researching.json").write_text(json.dumps(researching))
        if core is not None:
            (tmp_path / "core_watchlist.json").write_text(json.dumps(core))

    def test_load_portfolio_missing_returns_empty(self, tmp_path, monkeypatch):
        self._stub_paths(tmp_path, monkeypatch)
        assert load_portfolio() == set()

    def test_load_portfolio_returns_ticker_set(self, tmp_path, monkeypatch):
        self._stub_paths(tmp_path, monkeypatch, portfolio={
            "AAPL": {"position": "Portfolio"},
            "MSFT": {"position": "Portfolio"},
        })
        assert load_portfolio() == {"AAPL", "MSFT"}

    def test_load_researching_returns_ticker_set(self, tmp_path, monkeypatch):
        self._stub_paths(tmp_path, monkeypatch, researching={
            "INSM": {"position": "Researching"},
        })
        assert load_researching() == {"INSM"}

    def test_load_core_watchlist_unions_new_files(self, tmp_path, monkeypatch):
        """Back-compat wrapper: when new files exist, return their union."""
        self._stub_paths(
            tmp_path, monkeypatch,
            portfolio={"AAPL": {}}, researching={"INSM": {}},
        )
        assert load_core_watchlist() == {"AAPL", "INSM"}

    def test_load_core_watchlist_falls_back_to_legacy(self, tmp_path, monkeypatch):
        """When neither new file exists, fall back to legacy core_watchlist.json."""
        self._stub_paths(
            tmp_path, monkeypatch,
            core={"INSM": {}, "ISRG": {}},
        )
        assert load_core_watchlist() == {"INSM", "ISRG"}

    def test_load_core_watchlist_all_missing_returns_empty(self, tmp_path, monkeypatch):
        self._stub_paths(tmp_path, monkeypatch)
        assert load_core_watchlist() == set()

    def test_malformed_portfolio_returns_empty(self, tmp_path, monkeypatch):
        path = tmp_path / "portfolio.json"
        path.write_text("{not valid json")
        monkeypatch.setattr(sigma_screener, "PORTFOLIO_PATH", path)
        assert load_portfolio() == set()

    def test_list_instead_of_dict_returns_empty(self, tmp_path, monkeypatch):
        path = tmp_path / "portfolio.json"
        path.write_text(json.dumps(["INSM", "ISRG"]))
        monkeypatch.setattr(sigma_screener, "PORTFOLIO_PATH", path)
        assert load_portfolio() == set()


# ---------------------------------------------------------------------------
# Missing-metadata flag
# ---------------------------------------------------------------------------

class TestMissingMetadataFlag:
    def test_exempt_set_skips_etfs(self, tmp_path, monkeypatch):
        """ETFs whose names live in sources/etf_names.json must not be
        reported as CM gaps even though they're absent from ticker_metadata."""
        flag_path = tmp_path / "missing_metadata.json"
        monkeypatch.setattr(sigma_screener, "MISSING_METADATA_PATH", flag_path)
        tickers = ["AAPL", "SPYM", "DIA", "QQQ", "WIDGET"]
        metadata = {"AAPL": {"name": "Apple Inc"}}  # SPYM/DIA/QQQ/WIDGET missing
        result = write_missing_metadata_flag(
            tickers, metadata, exempt={"SPYM", "DIA", "QQQ"}
        )
        # Only WIDGET should be flagged; ETFs are exempt; AAPL has a name.
        assert set(result["tickers"].keys()) == {"WIDGET"}

    def test_no_gaps_clears_flag_file(self, tmp_path, monkeypatch):
        flag_path = tmp_path / "missing_metadata.json"
        flag_path.write_text("{}")  # stale file from a prior run
        monkeypatch.setattr(sigma_screener, "MISSING_METADATA_PATH", flag_path)
        result = write_missing_metadata_flag(
            ["SPYM"], {}, exempt={"SPYM"}
        )
        assert result == {}
        assert not flag_path.exists()
