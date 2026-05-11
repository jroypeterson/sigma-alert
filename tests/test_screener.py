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
        assert text.count("*UNH*") == 2
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
        assert "*MRNA*" not in text

    def test_price_rendered_in_line(self):
        alerts = [self._make_alert("UNH", "Healthcare Services", price=512.34)]
        payload = format_slack_message(alerts, "close", 100, {"ref_date": "2026-04-10"}, None, set())
        text = self._all_text(payload)
        assert "$512.34" in text

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
        assert "*INSM*" in text

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
