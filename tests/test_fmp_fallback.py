"""FMP fallback after the targeted Yahoo re-fetch (JP approved 2026-09-30).

Pins:
* FMP is asked ONLY for names still behind after the re-fetch, only on the
  `--mode close` route (midday ALSO calls screen_full(..., "close") — Codex R1),
  only for US symbols, and within FMP_MAX_CALLS / FMP_MAX_ERRORS /
  FMP_STAGE_BUDGET_S.
* The date guard: a quote is accepted only when it is stamped in the first
  minute after 16:00 ET today, the series it extends ends on the NYSE session
  immediately before today, and today's EOD row agrees with it. Intraday,
  after-hours, prior-day, gapped-series and disagreeing quotes are refused.
* No key = skip loudly ("FMP fallback: no key"), never fail the run.
* The merge keeps the 400-day series and appends exactly one bar for today.
* End to end through main(): the 2026-09-28 21:04 ET shape publishes when FMP
  has the close; the midday route never calls FMP.

The network is off suite-wide (conftest): FMP_API_KEY is removed and both FMP
network functions are stubbed; tests here substitute their own.
"""
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sigma_screener as ss  # noqa: E402
from test_duplicate_run import (  # noqa: E402
    ET, FRI_0925, MON_0928, _bdays_ending, _cache, _frame, _run_close, _shape_0928)

TODAY = MON_0928
# Captured at import, before conftest's autouse stub replaces it per test.
_REAL_FMP_GET_QUOTE = ss.fmp_get_quote
_REAL_CALENDAR = ss.fmp_corporate_action_symbols


def _stamp(d: date, h: int, m: int, s: int = 0) -> int:
    return int(datetime(d.year, d.month, d.day, h, m, s, tzinfo=ET).timestamp())


def _quote(prev=100.0, price=101.0, ts=None, **over):
    q = {"symbol": "X", "price": price, "open": 100.5, "dayHigh": 102.0,
         "dayLow": 99.5, "previousClose": prev,
         "timestamp": _stamp(TODAY, 16, 0, 2) if ts is None else ts}
    q.update(over)
    return q


def _bar(q, prev_close=100.0, last_bar=FRI_0925, today=TODAY):
    return ss.fmp_close_bar(q, today, prev_close, last_bar)


# ------------------------------------------------------------- the date guard
class TestDateGuard:
    def test_session_close_stamp_is_accepted(self):
        bar, why = _bar(_quote())
        assert why == "ok"
        assert bar == {"Close": 101.0, "Open": 100.5, "High": 102.0, "Low": 99.5}

    def test_intraday_stamp_is_refused(self):
        # 11:20 ET is the live shape probed on 2026-09-30.
        assert _bar(_quote(ts=_stamp(TODAY, 11, 20))) == (None, "intraday")

    def test_one_second_before_the_close_is_refused(self):
        assert _bar(_quote(ts=_stamp(TODAY, 15, 59, 59))) == (None, "intraday")

    def test_prior_day_close_is_refused(self):
        assert _bar(_quote(ts=_stamp(FRI_0925, 16, 0, 1))) == (None, "not_today")

    @pytest.mark.parametrize("hms", [(16, 10, 0), (16, 15, 0), (19, 45, 0)])
    def test_after_hours_stamp_is_refused(self, hms):
        # Codex R1: 16:15 with a plausible price was accepted by a 30-min window.
        # Board #558 moved the end from 16:01 to 16:10 on measured NYSE auction
        # stamps (16:01:11-16:02:57); see tests/test_fmp_close_auction.py.
        assert _bar(_quote(ts=_stamp(TODAY, *hms))) == (None, "after_close_window")

    def test_utc_date_rollover_is_judged_in_et(self):
        # 23:30 ET on 09-27 is 03:30 UTC 09-28 — "today" in UTC, NOT in ET.
        ts = _stamp(date(2026, 9, 27), 23, 30)
        assert _bar(_quote(ts=ts)) == (None, "not_today")

    def test_series_missing_the_prior_session_is_refused(self):
        # Codex R1 repro: Yahoo ends THURSDAY at 100; Friday closed 100.50;
        # Monday's quote previousClose=100.50 is within 1% of 100 — price
        # agreement is not date adjacency.
        q = _quote(prev=100.5, price=103.0, dayHigh=104.0)
        assert _bar(q, prev_close=100.0, last_bar=date(2026, 9, 24)) == (
            None, "gap_before_today")

    def test_prior_session_skips_nyse_holidays(self):
        # Tuesday after Labor Day 2026-09-07: the prior session is Friday 09-04.
        tue = date(2026, 9, 8)
        q = _quote(ts=_stamp(tue, 16, 0, 1))
        assert ss.fmp_close_bar(q, tue, 100.0, date(2026, 9, 4))[1] == "ok"
        assert ss.fmp_close_bar(q, tue, 100.0, date(2026, 9, 7))[1] == "gap_before_today"

    def test_previous_close_mismatch_is_refused(self):
        assert _bar(_quote(prev=100.0), prev_close=95.0) == (None, "prev_close_mismatch")

    def test_missing_timestamp_and_fields(self):
        q = _quote()
        q.pop("timestamp")
        assert _bar(q) == (None, "no_timestamp")
        assert _bar(_quote(price=None)) == (None, "missing_field")
        assert _bar(None) == (None, "no_quote")

    def test_price_outside_the_day_range_is_refused(self):
        assert _bar(_quote(price=103.0)) == (None, "price_outside_range")

    def test_eod_must_agree(self):
        bar = {"Close": 101.0}
        assert ss.fmp_eod_agrees(bar, 101.0)
        assert ss.fmp_eod_agrees(bar, 101.03)
        assert not ss.fmp_eod_agrees(bar, 101.2)
        assert not ss.fmp_eod_agrees(bar, None)


@pytest.mark.parametrize("today,prev", [
    ("2026-09-28", "2026-09-25"), ("2026-11-27", "2026-11-25"),
    ("2026-01-02", "2025-12-31"), ("2026-04-06", "2026-04-02"),
    ("2026-07-06", "2026-07-02"), ("2026-06-22", "2026-06-18"),
    ("2026-10-13", "2026-10-12"),   # Columbus Day: NYSE open
])
def test_prev_nyse_session(today, prev):
    assert ss.prev_nyse_session(date.fromisoformat(today)) == date.fromisoformat(prev)


class TestSymbolScope:
    @pytest.mark.parametrize("display,expected", [
        ("AAPL", "AAPL"), ("BRK-B", "BRK-B"),
        ("FRE", None),          # bare foreign name: the lane maps it to FRE.DE
        ("GETIB.SS", None), ("ROG.SW", None), ("^GSPC", None), ("CL=F", None),
        ("DX-Y.NYB", None),
    ])
    def test_us_only(self, display, expected):
        assert ss.fmp_symbol(display) == expected


# --------------------------------------------------- refetch_behind + FMP unit
def _behind_setup(n_us=4, foreign=("FRE",), newest=FRI_0925):
    tickers = [f"U{i}" for i in range(n_us)] + list(foreign)
    # Frame columns are yfinance symbols (FRE -> FRE.DE), as batch_download returns.
    spec = {ss.to_yf_symbol(t): newest for t in tickers}
    data = _frame(_bdays_ending(newest), spec)
    prior = {t: newest for t in tickers}
    return tickers, data, prior


class _Fmp:
    """Records every FMP call; answers with a session-close quote whose
    previousClose is the series' last close, and an agreeing EOD row."""

    def __init__(self, *, stamp=None, fail_auth=False, eod_bump=1.0, error=False,
                 actions=(), calendar_error=None):
        self.quotes, self.eods, self.calendars = [], [], []
        self.actions, self.calendar_error = set(actions), calendar_error
        self.data = None
        self.stamp = stamp or _stamp(TODAY, 16, 0, 3)
        self.fail_auth, self.eod_bump, self.error = fail_auth, eod_bump, error

    def quote(self, symbol, key):
        self.quotes.append((symbol, key))
        if self.fail_auth:
            raise ss.FMPAuthError("HTTP 402")
        if self.error:
            raise RuntimeError("network ReadTimeout")
        prev = float(self.data["Close"][symbol].dropna().iloc[-1])
        return {"symbol": symbol, "price": prev * 1.01, "open": prev,
                "dayHigh": prev * 1.02, "dayLow": prev * 0.99,
                "previousClose": prev, "timestamp": self.stamp}

    def eod(self, symbol, key, day):
        self.eods.append((symbol, day))
        prev = float(self.data["Close"][symbol].dropna().iloc[-1])
        return prev * 1.01 * self.eod_bump

    def calendar(self, key, day):
        self.calendars.append(day)
        if self.calendar_error:
            raise self.calendar_error
        return self.actions

    @property
    def calls(self):
        return len(self.quotes) + len(self.eods)


def _go(monkeypatch, *, allow=True, key="k-test", refetch=None, fmp=None,
        n_us=4, foreign=("FRE",), newest=FRI_0925):
    tickers, data, prior = _behind_setup(n_us, foreign, newest)
    if key:
        monkeypatch.setenv("FMP_API_KEY", key)
    f = fmp or _Fmp()
    f.data = data
    monkeypatch.setattr(ss, "fmp_get_quote", f.quote)
    monkeypatch.setattr(ss, "fmp_get_eod_price", f.eod)
    monkeypatch.setattr(ss, "fmp_corporate_action_symbols", f.calendar)
    monkeypatch.setattr(ss, "refetch_recent", lambda syms: refetch)
    stats = {}
    out = ss.refetch_behind(data, tickers, None, prior, stats, TODAY, allow_fmp=allow)
    return out, stats, f, data


class TestWhenFmpIsUsed:
    def test_still_behind_us_names_are_recovered(self, monkeypatch):
        out, stats, f, _ = _go(monkeypatch)
        assert sorted(out) == ["U0", "U1", "U2", "U3"]
        assert sorted(s for s, _ in f.quotes) == ["U0", "U1", "U2", "U3"]
        assert sorted(s for s, _ in f.eods) == ["U0", "U1", "U2", "U3"]
        assert all(k == "k-test" for _, k in f.quotes)
        assert stats["fmp_recovered"] == 4 and stats["fmp_attempted"] == 4
        assert stats["fmp_calls"] == 10          # 2 calendars + 2 per name
        assert f.calendars == [TODAY]
        assert stats["fmp_foreign"] == 1 and stats["fmp_foreign_names"] == ["FRE"]

    def test_only_names_the_re_fetch_did_not_recover(self, monkeypatch):
        spec = {"U0": MON_0928, "U1": FRI_0925, "U2": FRI_0925, "U3": FRI_0925}
        fresh = _frame(_bdays_ending(MON_0928, 5), spec)
        out, stats, f, _ = _go(monkeypatch, refetch=fresh, foreign=())
        assert stats["refetch_recovered"] == 1
        assert sorted(s for s, _ in f.quotes) == ["U1", "U2", "U3"]
        assert stats["fmp_attempted"] == 3 and stats["fmp_rejects"] == {}
        assert sorted(out) == ["U0", "U1", "U2", "U3"]

    def test_not_without_allow_fmp(self, monkeypatch):
        out, stats, f, _ = _go(monkeypatch, allow=False)
        assert f.calls == 0 and out == {} and "fmp_stage" not in stats

    def test_not_when_the_re_fetch_never_triggered(self, monkeypatch):
        # 1 of 10 behind is below REFETCH_BEHIND_FRACTION: neither stage runs.
        tickers = [f"U{i}" for i in range(10)]
        spec = {t: MON_0928 for t in tickers}
        spec["U0"] = FRI_0925
        data = _frame(_bdays_ending(MON_0928), spec)
        monkeypatch.setenv("FMP_API_KEY", "k")
        f = _Fmp()
        f.data = data
        monkeypatch.setattr(ss, "fmp_get_quote", f.quote)
        stats = {}
        ss.refetch_behind(data, tickers, None, {t: FRI_0925 for t in tickers},
                          stats, TODAY, allow_fmp=True)
        assert f.calls == 0 and "fmp_stage" not in stats

    def test_wrong_date_quote_is_rejected_and_nothing_recovered(self, monkeypatch):
        out, stats, f, _ = _go(monkeypatch, fmp=_Fmp(stamp=_stamp(FRI_0925, 16, 0, 1)))
        assert out == {} and stats["fmp_recovered"] == 0
        assert stats["fmp_rejects"] == {"not_today": 4}
        assert f.eods == []            # a refused quote costs no second call

    def test_intraday_quote_is_rejected(self, monkeypatch):
        out, stats, _, _ = _go(monkeypatch, fmp=_Fmp(stamp=_stamp(TODAY, 15, 30)))
        assert out == {} and stats["fmp_rejects"] == {"intraday": 4}

    def test_disagreeing_eod_row_is_rejected(self, monkeypatch):
        out, stats, _, _ = _go(monkeypatch, fmp=_Fmp(eod_bump=1.01))
        assert out == {} and stats["fmp_rejects"] == {"eod_disagrees": 4}

    def test_gapped_series_spends_no_call(self, monkeypatch):
        # Yahoo's newest bar is THURSDAY; today is Monday.
        out, stats, f, _ = _go(monkeypatch, newest=date(2026, 9, 24))
        assert out == {} and f.calls == 0
        assert stats["fmp_rejects"] == {"gap_before_today": 4}


class TestCorporateActions:
    """Codex R2: Yahoo is auto-adjusted, an FMP quote is raw. On an ex-date the
    stale series is unadjusted and previousClose is the raw prior close, so a
    $2 dividend reads as -2% and a 2:1 split as -50%, then the watermark makes
    it permanent. Refuse those names."""

    def test_ex_date_names_are_refused_without_a_call(self, monkeypatch):
        out, stats, f, _ = _go(monkeypatch, fmp=_Fmp(actions={"U1", "U3"}))
        assert sorted(out) == ["U0", "U2"]
        assert sorted(s for s, _ in f.quotes) == ["U0", "U2"]
        assert stats["fmp_rejects"] == {"corporate_action": 2}

    def test_calendar_unavailable_recovers_nothing(self, monkeypatch):
        out, stats, f, _ = _go(monkeypatch,
                               fmp=_Fmp(calendar_error=RuntimeError("HTTP 500")))
        assert out == {} and f.calls == 0
        assert stats["fmp_status"].startswith("corporate-action calendar unavailable")
        assert "corporate-action calendar unavailable" in ss.describe_unscreened(stats)

    def test_calendar_auth_error_is_reported_as_auth(self, monkeypatch):
        out, stats, f, _ = _go(monkeypatch,
                               fmp=_Fmp(calendar_error=ss.FMPAuthError("HTTP 402")))
        assert out == {} and f.calls == 0
        assert stats["fmp_status"].startswith("auth error")

    def test_calendar_parser_keeps_only_todays_ex_dates(self, monkeypatch):
        rows = {"dividends-calendar": [{"symbol": "aaa", "date": "2026-09-28"},
                                       {"symbol": "BBB", "date": "2026-09-29"}],
                "splits-calendar": [{"symbol": "CCC", "date": "2026-09-28",
                                     "numerator": 2, "denominator": 1}]}
        monkeypatch.setattr(ss, "_fmp_get", lambda path, params, key: rows[path])
        real = _REAL_CALENDAR
        assert real("k", TODAY) == {"AAA", "CCC"}


class TestSkipsCapsAndBudgets:
    def test_nyse_holiday_spends_no_call(self, monkeypatch):
        # Codex R4: Thanksgiving 2026-11-26 — the close cron fires, every series
        # correctly ends 11-25, and ~700 paid quotes would all come back not_today.
        tickers, data, prior = _behind_setup(newest=date(2026, 11, 25))
        monkeypatch.setenv("FMP_API_KEY", "k")
        f = _Fmp()
        f.data = data
        monkeypatch.setattr(ss, "fmp_get_quote", f.quote)
        monkeypatch.setattr(ss, "fmp_corporate_action_symbols", f.calendar)
        stats = {}
        out = ss.refetch_behind(data, tickers, None, prior, stats,
                                date(2026, 11, 26), allow_fmp=True)
        assert out == {} and f.calls == 0 and f.calendars == []
        assert stats["fmp_status"] == "not an NYSE session"

    def test_missing_key_skips_loudly_and_makes_no_call(self, monkeypatch):
        out, stats, f, _ = _go(monkeypatch, key=None)
        assert f.calls == 0 and out == {}
        assert stats["fmp_status"] == "no key"
        assert "FMP fallback: no key" in ss.describe_unscreened(stats)

    def test_cap_limits_calls_and_is_reported(self, monkeypatch):
        monkeypatch.setattr(ss, "FMP_MAX_CALLS", 6)   # 2 calendars + 2 names
        monkeypatch.setattr(ss, "FMP_WORKERS", 1)
        out, stats, f, _ = _go(monkeypatch)
        assert f.calls == 4 and len(out) == 2
        assert stats["fmp_capped"] == 2
        assert "CAPPED: 2 not tried" in ss.describe_unscreened(stats)

    def test_auth_error_stops_spending(self, monkeypatch):
        monkeypatch.setattr(ss, "FMP_WORKERS", 1)
        out, stats, f, _ = _go(monkeypatch, n_us=6, fmp=_Fmp(fail_auth=True))
        assert f.calls == 1 and out == {}
        assert stats["fmp_status"].startswith("auth error")

    def test_repeated_errors_trip_the_breaker(self, monkeypatch):
        # Codex R1: 712 names each timing out would block ~44 min.
        monkeypatch.setattr(ss, "FMP_WORKERS", 1)
        monkeypatch.setattr(ss, "FMP_MAX_ERRORS", 3)
        out, stats, f, _ = _go(monkeypatch, n_us=20, fmp=_Fmp(error=True))
        assert len(f.quotes) == 3 and out == {}
        assert stats["fmp_status"].startswith("stopped after 3 errors")
        assert "FMP stopped after 3 errors" in ss.describe_unscreened(stats)

    def test_stage_budget_stops_the_stage(self, monkeypatch):
        monkeypatch.setattr(ss, "FMP_STAGE_BUDGET_S", -1)
        out, stats, f, _ = _go(monkeypatch)
        assert f.calls == 0 and out == {}
        assert "budget" in stats["fmp_status"]

    def test_key_never_reaches_an_exception_message(self, monkeypatch):
        # requests embeds the URL (with ?apikey=) in its messages; the repo's
        # Actions logs are public.
        def _boom(*a, **k):
            raise ss.requests.ConnectionError(
                "HTTPSConnectionPool: /stable/quote?symbol=X&apikey=SECRETKEY")
        monkeypatch.setattr(ss.requests, "get", _boom)
        with pytest.raises(RuntimeError) as e:
            _REAL_FMP_GET_QUOTE("X", "SECRETKEY")
        assert "SECRETKEY" not in str(e.value)
        assert e.value.__cause__ is None and e.value.__suppress_context__


@pytest.mark.parametrize("status,text,payload", [
    (200, "Invalid API KEY: SECRETKEY", None),
    (200, "Restricted Endpoint for SECRETKEY", None),
    (200, '{"Error Message": "bad key SECRETKEY"}', {"Error Message": "bad key SECRETKEY"}),
])
def test_error_bodies_never_reach_the_status(monkeypatch, status, text, payload):
    """Codex R3: an error body that echoes the key must not become the
    exception text (it is printed and posted to #status-reports)."""
    class _R:
        status_code = status
        def __init__(self):
            self.text = text
        def json(self):
            if payload is None:
                raise ValueError
            return payload
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _R())
    with pytest.raises(ss.FMPAuthError) as e:
        _REAL_FMP_GET_QUOTE("X", "SECRETKEY")
    assert "SECRETKEY" not in str(e.value)


class TestMerge:
    def test_one_bar_appended_history_kept(self, monkeypatch):
        out, _, _, data = _go(monkeypatch)
        close, open_, high, low = out["U0"]
        base = data["Close"]["U0"].dropna()
        assert len(close) == len(base) + 1
        assert close.index[-1] == pd.Timestamp(TODAY)
        pd.testing.assert_series_equal(close.iloc[:-1], base, check_names=False,
                                       check_freq=False)
        prev = float(base.iloc[-1])
        assert close.iloc[-1] == pytest.approx(prev * 1.01)
        assert open_.iloc[-1] == pytest.approx(prev)
        assert high.iloc[-1] == pytest.approx(prev * 1.02)
        assert low.iloc[-1] == pytest.approx(prev * 0.99)
        assert close.index.is_monotonic_increasing and close.index.is_unique


# ------------------------------------------------------------ end to end: main
class TestSeptember28WithFmp:
    def _go(self, monkeypatch, tmp_path, key="k", mode="close"):
        spec, wm = _shape_0928()
        tickers = list(spec)
        frame = _frame(_bdays_ending(MON_0928), spec)
        fri_close = float(frame["Close"]["U0"].dropna().iloc[-1])  # Friday
        calls = []

        def _q(symbol, k):
            calls.append(symbol)
            return {"symbol": symbol, "price": fri_close * 1.001, "open": fri_close,
                    "dayHigh": fri_close * 1.01, "dayLow": fri_close * 0.99,
                    "previousClose": fri_close, "timestamp": _stamp(MON_0928, 16, 0, 1)}

        def _e(symbol, k, day):
            calls.append(symbol)
            return fri_close * 1.001
        if key:
            monkeypatch.setenv("FMP_API_KEY", key)
        monkeypatch.setattr(ss, "fmp_get_quote", _q)
        monkeypatch.setattr(ss, "fmp_get_eod_price", _e)
        run = _run_close(monkeypatch, tmp_path,
                         now_et=datetime(2026, 9, 28, 21, 4, tzinfo=ET),
                         tickers=tickers, frame=frame, mode=mode,
                         prior_cache=_cache(wm, FRI_0925), refetch_frame=None)
        return run, calls

    def test_fmp_recovers_the_lost_monday_and_the_run_publishes(self, monkeypatch, tmp_path):
        run, calls = self._go(monkeypatch, tmp_path)
        assert len(calls) == 2 * 712             # + 2 calendar calls (stubbed empty)
        assert len(run.digests) == 1
        hb = run.heartbeats[0]
        assert "730/730 tickers screened" in hb, hb
        assert "re-fetch recovered 0/712" in hb
        assert "fmp recovered 712/712 (1426 calls)" in hb, hb
        assert run.saved[-1]["tickers"]["U0"]["last_bar"] == "2026-09-28"

    def test_midday_route_never_calls_fmp(self, monkeypatch, tmp_path):
        # Codex R1: main()'s midday branch calls screen_full(..., "close").
        run, calls = self._go(monkeypatch, tmp_path, mode="midday")
        assert calls == []
        assert "fmp" not in run.heartbeats[0].lower()

    def test_without_a_key_the_run_is_the_old_error_and_says_why(self, monkeypatch, tmp_path):
        run, calls = self._go(monkeypatch, tmp_path, key=None)
        assert calls == []
        assert run.digests == []
        hb = run.heartbeats[0]
        assert "sigma-alert — error" in hb
        assert "FMP fallback: no key" in hb


def test_close_workflow_passes_the_key_to_the_screener_step():
    """The seam: code reads FMP_API_KEY, so the close run's screener step must
    receive it. (Midday/open never reach the FMP stage.)"""
    wf = (Path(__file__).resolve().parent.parent / ".github" / "workflows"
          / "sigma-close.yml").read_text(encoding="utf-8")
    step = wf.split("- name: Run sigma screener (close mode)", 1)[1].split("- name:", 1)[0]
    assert "FMP_API_KEY: ${{ secrets.FMP_API_KEY }}" in step
    assert "--mode close" in step
