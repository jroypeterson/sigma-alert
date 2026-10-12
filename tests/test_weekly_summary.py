"""Weekly price summary (scripts/weekly_summary.py) + the screener's week baseline."""
import json
import math
from datetime import date

import pandas as pd
import pytest

import sigma_screener as ss
import weekly_summary as w

FRI = date(2026, 10, 9)


def _series(days, start=100.0, step=1.0):
    idx = pd.to_datetime(days)
    return pd.Series([start + i * step for i in range(len(idx))], index=idx)


# ── week_baseline (screener side) ────────────────────────────────────────────

def test_week_baseline_is_the_close_before_monday():
    days = pd.bdate_range("2026-09-21", "2026-10-09")
    s = _series(days)
    b = ss.week_baseline(s)
    assert b["week_base_date"] == "2026-10-02"          # prior Friday
    assert b["week_base_close"] == float(s.loc["2026-10-02"])
    assert b["last_close"] == float(s.iloc[-1])
    assert b["week_sessions"] == 5


def test_week_baseline_counts_only_sessions_traded():
    # Good Friday-style week: Thursday is the last bar; the baseline is still
    # the prior week's last close and n = 4, not 5.
    days = [d for d in pd.bdate_range("2026-03-23", "2026-04-02")]
    b = ss.week_baseline(_series(days))
    assert b["week_base_date"] == "2026-03-27"
    assert b["week_sessions"] == 4


def test_week_baseline_needs_a_calendar_index():
    assert ss.week_baseline(pd.Series([1.0, 2.0, 3.0])) == {}


def test_close_mode_cache_entry_carries_the_week_baseline():
    days = pd.bdate_range("2025-10-01", "2026-10-09")
    close = _series(days, step=0.1)
    _, entry, *_ = ss._process_ticker_full("ABC", close, close, None, None, "close")
    assert entry["week_base_date"] == "2026-10-02" and entry["week_sessions"] == 5
    _, entry_mid, *_ = ss._process_ticker_full("ABC", close, close, None, None, "midday")
    assert "week_base_close" not in entry_mid


# ── calendar gate ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("day,expected", [
    (date(2026, 10, 9), True),     # ordinary Friday
    (date(2026, 10, 8), False),    # Thursday with a Friday session to come
    (date(2026, 4, 2), True),      # Thursday before Good Friday 2026-04-03
    (date(2026, 11, 27), True),    # day after Thanksgiving (half-day) is a session
    (date(2026, 10, 10), False),   # Saturday
])
def test_is_last_session_of_week(day, expected):
    assert w.is_last_session_of_week(day) is expected


# ── weekly_move ──────────────────────────────────────────────────────────────

def _entry(**kw):
    e = {"mu": 0.001, "sigma": 0.02, "last_bar": "2026-10-09", "last_close": 110.0,
         "week_base_close": 100.0, "week_base_date": "2026-10-02", "week_sessions": 5,
         "prior_year_end_close": 80.0, "prior_year_end_year": 2025, "high_52w": 120.0}
    e.update(kw)
    return e


def test_weekly_z_scales_the_daily_distribution_by_root_n():
    mv = w.weekly_move(_entry(), "2026-10-09")
    assert mv["week_pct"] == pytest.approx(10.0)
    assert mv["z"] == pytest.approx((0.10 - 5 * 0.001) / (0.02 * math.sqrt(5)))
    assert mv["ytd_pct"] == pytest.approx(37.5)
    assert "through" not in mv


def test_a_late_bar_inside_the_week_counts_and_is_labeled():
    mv = w.weekly_move(_entry(last_bar="2026-10-08"), "2026-10-09")
    assert mv is not None and mv["through"] == "2026-10-08"


def test_a_carried_entry_from_last_week_is_not_this_weeks_move():
    assert w.weekly_move(_entry(last_bar="2026-10-02"), "2026-10-09") is None


def test_ytd_omitted_on_a_stale_year_baseline():
    assert "ytd_pct" not in w.weekly_move(_entry(prior_year_end_year=2024), "2026-10-09")


def test_fred_week_change_uses_the_last_observation_before_monday():
    series = [("2026-10-01", 4.0), ("2026-10-02", 4.10), ("2026-10-08", 4.02)]
    ch = w.fred_week_change(series, date(2026, 10, 5))
    assert ch["bp_chg"] == pytest.approx(-8.0) and ch["base_date"] == "2026-10-02"
    assert w.fred_week_change(series[:2], date(2026, 10, 5)) is None  # nothing this week


# ── snapshot + Slack payload ─────────────────────────────────────────────────

def _cache(tickers, **kw):
    return {"tickers": {t: _entry(**kw) for t in tickers},
            "published": {"date": "2026-10-09", "mode": "close"}}


@pytest.fixture
def universe(monkeypatch):
    names = [f"N{i:03d}" for i in range(60)]
    monkeypatch.setattr(ss, "load_watchlist", lambda: names + ["SPYM", "^TNX"])
    monkeypatch.setattr(ss, "load_metadata",
                        lambda: {t: {"name": f"{t} Inc", "sector": "MedTech", "subsector": ""}
                                 for t in names})
    monkeypatch.setattr(ss, "load_sp500_names", lambda: {})
    monkeypatch.setattr(ss, "load_sp500_set", lambda: set())
    monkeypatch.setattr(ss, "load_etf_names", lambda: {"SPYM": "S&P 500", "^TNX": "10Y"})
    monkeypatch.setattr(ss, "load_etf_weighting", lambda: {})
    for loader in ("load_portfolio", "load_researching", "load_following_for_interest",
                   "load_ready_to_buy", "load_ready_to_short", "load_global_equity_etfs",
                   "load_sector_etfs", "load_healthcare_etfs", "load_tech_etfs",
                   "load_commodity_etfs"):
        monkeypatch.setattr(ss, loader, lambda: set())
    monkeypatch.setattr(ss, "load_index_etfs", lambda: {"SPYM"})
    monkeypatch.setattr(ss, "load_macro", lambda: {"^TNX"})
    monkeypatch.setattr(w, "ETF_GROUPS", [(lbl, getattr(ss, f.__name__)) for lbl, f in w.ETF_GROUPS])
    return names


def test_snapshot_tiers_and_etfs_stay_out_of_alerts(universe):
    snap = w.build_snapshot(_cache(universe + ["SPYM", "^TNX"]), FRI)
    assert snap["coverage"] == {"current": 60, "total": 60, "missing": []}
    assert all(r["tier"] == "2sigma" for r in snap["names"])        # z ≈ +2.18
    assert all("MedTech" in r["categories"] for r in snap["names"])
    assert {r["ticker"] for r in snap["names"]}.isdisjoint({"SPYM", "^TNX"})
    assert snap["etf_groups"][0]["rows"][0]["ticker"] == "SPYM"


def test_slack_sections_stay_under_the_3000_char_limit(universe):
    snap = w.build_snapshot(_cache(universe + ["SPYM", "^TNX"]), FRI)
    payload = w.format_slack(snap)
    sections = [b["text"]["text"] for b in payload["blocks"] if b["type"] == "section"]
    assert len(sections) > 1 and max(map(len, sections)) <= 3000
    assert len(payload["blocks"]) <= 50
    # A `_Group_` header is never the last line of a block (it travels with its rows).
    assert not any(s.splitlines()[-1].startswith("_") and s.splitlines()[-1].endswith("_")
               and "none" not in s.splitlines()[-1] for s in sections)


def test_html_embeds_the_snapshot_and_escapes_script_close(universe):
    snap = w.build_snapshot(_cache(universe), FRI)
    snap["names"][0]["name"] = "</script><b>x"
    html = w.build_html(snap)
    assert html.startswith("<title>") and "</script><b>" not in html


# ── main(): gates ────────────────────────────────────────────────────────────

class _Posts:
    def __init__(self):
        self.calls = []

    def __call__(self, url, json=None, timeout=None):  # noqa: A002
        self.calls.append((url, json))

        class R:
            def raise_for_status(self):
                return None
        return R()


@pytest.fixture
def posts(monkeypatch):
    p = _Posts()
    monkeypatch.setattr(w.requests, "post", p)
    monkeypatch.setenv("SLACK_WEBHOOK", "https://hooks.example/main")
    monkeypatch.setenv("SLACK_STATUS_REPORTS_WEBHOOK", "https://hooks.example/status")
    return p


def test_posts_once_then_the_duplicate_close_run_is_a_noop(universe, posts, monkeypatch):
    monkeypatch.setattr(ss, "load_cache", lambda: _cache(universe + ["SPYM", "^TNX"]))
    assert w.main(["--date", "2026-10-09"]) == 0
    assert [u for u, _ in posts.calls] == ["https://hooks.example/main"]
    assert json.loads(w.SNAPSHOT_PATH.read_text())["posted"] is True
    assert w.HTML_PATH.exists()
    assert w.main(["--date", "2026-10-09"]) == 0
    assert len(posts.calls) == 1


def test_waits_for_the_delivered_close_digest(universe, posts, monkeypatch):
    c = _cache(universe)
    c["published"] = {"date": "2026-10-08", "mode": "close"}
    monkeypatch.setattr(ss, "load_cache", lambda: c)
    assert w.main(["--date", "2026-10-09"]) == 0
    assert posts.calls == [] and not w.SNAPSHOT_PATH.exists()


def test_not_on_a_mid_week_session(universe, posts, monkeypatch):
    # Delivered digest for that Thursday, so only the calendar gate can stop it.
    c = _cache(universe, last_bar="2026-10-08")
    c["published"] = {"date": "2026-10-08", "mode": "close"}
    monkeypatch.setattr(ss, "load_cache", lambda: c)
    assert w.main(["--date", "2026-10-08"]) == 0
    assert posts.calls == []


def test_a_thin_week_posts_nothing_to_the_channel_and_says_so(universe, posts, monkeypatch):
    c = _cache(universe)
    for t in universe[:20]:                     # 40/60 = 67% < 80% floor
        c["tickers"][t]["last_bar"] = "2026-10-02"
    monkeypatch.setattr(ss, "load_cache", lambda: c)
    assert w.main(["--date", "2026-10-09"]) == 0
    assert [u for u, _ in posts.calls] == ["https://hooks.example/status"]
    assert "NOT posted" in posts.calls[0][1]["text"]
    assert not w.SNAPSHOT_PATH.exists()


def test_chunk_moves_a_trailing_group_header_to_the_next_block():
    blocks = []
    w._chunk(blocks, "*Head*", ["x" * 50, "_Group_", "y" * 50], max_len=70)
    texts = [b["text"]["text"] for b in blocks]
    assert texts == ["*Head*\n" + "x" * 50, "_Group_\n" + "y" * 50]
