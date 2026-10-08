"""Board #558: the FMP fallback refused 75 correct NYSE closes on 2026-10-07.

Measured (plans/558_561_plan_2026-10-08.md): Yahoo had no 10-07 bar for 700/756
names at 21:06 ET; FMP recovered 619/699 and refused 43 `after_close_window`
(NYSE closing-auction prints stamped 16:01:11-16:02:57, quote == EOD == Yahoo's
close) and 32 `intraday` (quote frozen at the last continuous trade, stamped
15:59:57-15:59:59; the EOD row is the official close). Every one NYSE-listed.

Pins:
* window end 16:10; 16:02:57 accepted, 16:10:00 refused.
* pre-auction stamps [15:59:00, 16:00:00) accepted ONLY once the run is at/after
  FMP_EOD_SETTLED_AT on the same ET day; 15:58:59 never.
* after settle, Close = the EOD row's price for EVERY accepted name (High/Low
  widened); before settle, the quote price (pre-#558 behaviour).
* the settle time is not later than the CI close cron in EST (winter).
* yesterday's FMP returns are reconciled against Yahoo's backfill and every
  disagreeing name is NAMED in the heartbeat.
"""
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sigma_screener as ss  # noqa: E402
from test_duplicate_run import (  # noqa: E402
    ET, _bdays_ending, _cache, _frame, _run_close)

WED = date(2026, 10, 7)
TUE = date(2026, 10, 6)
THU = date(2026, 10, 8)
AT_2106 = datetime(2026, 10, 7, 21, 6, tzinfo=ET)


def _stamp(d, h, m, s=0):
    return int(datetime(d.year, d.month, d.day, h, m, s, tzinfo=ET).timestamp())


def _q(ts, price=101.0, **over):
    q = {"symbol": "X", "price": price, "open": 100.5, "dayHigh": 102.0,
         "dayLow": 99.5, "previousClose": 100.0, "timestamp": ts}
    q.update(over)
    return q


def _bar(q, now):
    return ss.fmp_close_bar(q, WED, 100.0, TUE, now=now)


# ---------------------------------------------------------------- the window
class TestWindow:
    @pytest.mark.parametrize("hms", [(16, 1, 11), (16, 2, 57), (16, 9, 59)])
    def test_late_auction_stamps_are_accepted(self, hms):
        assert _bar(_q(_stamp(WED, *hms)), AT_2106)[1] == "ok"

    @pytest.mark.parametrize("hms", [(16, 10, 0), (16, 15, 0), (19, 45, 0)])
    def test_after_the_window_is_refused(self, hms):
        assert _bar(_q(_stamp(WED, *hms)), AT_2106) == (None, "after_close_window")

    def test_late_band_needs_settle_too(self):
        # Codex R1: a 16:05 stamp read at 16:20 may be after-hours; refused.
        early = datetime(2026, 10, 7, 16, 20, tzinfo=ET)
        assert _bar(_q(_stamp(WED, 16, 5)), early) == (None, "after_close_window")
        assert _bar(_q(_stamp(WED, 16, 0, 59)), early)[1] == "ok"
        assert _bar(_q(_stamp(WED, 16, 1, 0)), early) == (None, "after_close_window")

    @pytest.mark.parametrize("hms", [(15, 59, 0), (15, 59, 59)])
    def test_pre_auction_accepted_once_settled(self, hms):
        assert _bar(_q(_stamp(WED, *hms)), AT_2106)[1] == "ok"

    def test_pre_auction_window_start_is_exact(self):
        assert _bar(_q(_stamp(WED, 15, 58, 59)), AT_2106) == (None, "intraday")

    def test_pre_auction_refused_before_settle(self):
        h, m, s = ss.FMP_EOD_SETTLED_AT
        settle = datetime(2026, 10, 7, h, m, s, tzinfo=ET)
        ts = _stamp(WED, 15, 59, 59)
        assert _bar(_q(ts), settle - timedelta(seconds=1)) == (None, "intraday")
        assert _bar(_q(ts), settle)[1] == "ok"

    def test_pre_auction_refused_without_a_clock(self):
        assert _bar(_q(_stamp(WED, 15, 59, 59)), None) == (None, "intraday")

    def test_a_run_past_midnight_is_not_settled(self):
        # 21:06 start + the 10-07 lateness again lands past midnight ET.
        assert not ss.eod_is_settled(datetime(2026, 10, 8, 0, 40, tzinfo=ET), WED)
        assert not ss.eod_is_settled(datetime(2026, 10, 7, 21, 6), WED)   # naive

    def test_settle_is_judged_in_et_not_utc(self):
        # 20:35 UTC = 16:35 EDT: settled. 20:25 UTC = 16:25 EDT: not.
        from datetime import timezone
        assert ss.eod_is_settled(datetime(2026, 10, 7, 20, 35, tzinfo=timezone.utc), WED)
        assert not ss.eod_is_settled(datetime(2026, 10, 7, 20, 25, tzinfo=timezone.utc), WED)

    def test_price_outside_range_is_judged_on_the_quote(self):
        assert _bar(_q(_stamp(WED, 16, 2), price=103.0), AT_2106) == (
            None, "price_outside_range")


def test_eod_close_widens_high_and_low():
    bar = {"Close": 101.0, "Open": 100.5, "High": 101.02, "Low": 99.5}
    out = ss.fmp_with_eod_close(bar, 101.04)
    assert out == {"Close": 101.04, "Open": 100.5, "High": 101.04, "Low": 99.5}
    out = ss.fmp_with_eod_close({**bar, "Low": 101.0}, 100.98)
    assert out["Low"] == 100.98 and out["Close"] == 100.98


def test_settle_time_is_not_after_the_winter_close_cron():
    """Fable R1 H1: `30 21 * * 1-5` UTC is 17:30 EDT but 16:30 EST. A settle
    time after the EST cron would make the fix work only when GitHub is late."""
    wf = (Path(__file__).resolve().parent.parent / ".github" / "workflows"
          / "sigma-close.yml").read_text(encoding="utf-8")
    # The FIRST cron line (the workflow has one) and the SCHEDULED minute; a
    # real start is always later, the safe side.
    m = re.search(r"cron:\s*'(\d+)\s+(\d+)\s", wf)
    minute, hour_utc = int(m.group(1)), int(m.group(2))
    est = ((hour_utc - 5) % 24, minute, 0)
    assert ss.FMP_EOD_SETTLED_AT <= est, (ss.FMP_EOD_SETTLED_AT, est)


# ---------------------------------------------- the stage, 2026-10-07 shape
def _stage(monkeypatch, now, kinds):
    """`kinds` = {ticker: (stamp hms, quote price factor, eod factor)} over a
    series whose last close (Tuesday) is the base."""
    tickers = list(kinds)
    data = _frame(_bdays_ending(TUE), {t: TUE for t in tickers})
    merged = {t: tuple(ss._naive(ss._series_from(data, f, t))
                       for f in ("Close", "Open", "High", "Low")) for t in tickers}
    base = {t: float(merged[t][0].iloc[-1]) for t in tickers}

    def quote(sym, key):
        hms, qf, _ = kinds[sym]
        b = base[sym]
        return {"symbol": sym, "price": b * qf, "open": b, "dayHigh": b * 1.02,
                "dayLow": b * 0.98, "previousClose": b, "timestamp": _stamp(WED, *hms)}

    def eod(sym, key, day):
        return base[sym] * kinds[sym][2]

    monkeypatch.setenv("FMP_API_KEY", "k")
    monkeypatch.setattr(ss, "fmp_get_quote", quote)
    monkeypatch.setattr(ss, "fmp_get_eod_price", eod)
    monkeypatch.setattr(ss, "fmp_corporate_action_symbols", lambda k, d: set())
    monkeypatch.setattr(ss, "now_et", lambda: now)
    stats = {}
    out = ss.fmp_fallback(tickers, merged, {t: TUE for t in tickers}, stats, WED)
    return out, stats, base


SHAPE_1007 = {
    "LATE": ((16, 2, 37), 1.0099, 1.0100),       # A: quote 1 bp off EOD
    "PRE": ((15, 59, 59), 1.0097, 1.0100),       # C: quote 3 bp under the close
    "NASDAQ": ((16, 0, 1), 1.0099, 1.0100),      # passed before #558 too
    "MOVED": ((15, 59, 59), 1.0040, 1.0100),     # auction 60 bp away: refused
}


class TestStage1007:
    def test_after_settle_every_close_is_the_eod_row(self, monkeypatch):
        out, stats, base = _stage(monkeypatch, AT_2106, SHAPE_1007)
        assert sorted(out) == ["LATE", "NASDAQ", "PRE"]
        for t in out:
            assert out[t][0].iloc[-1] == pytest.approx(base[t] * 1.0100)
            assert out[t][0].index[-1] == pd.Timestamp(WED)
        assert stats["fmp_rejects"] == {"eod_disagrees": 1}
        assert stats["fmp_eod_settled"] is True
        assert stats["fmp_spread_bp_max"] == pytest.approx(59.4, abs=0.1)
        acc = stats["fmp_accepted"]
        assert sorted(acc) == ["LATE", "NASDAQ", "PRE"]
        assert all(e["src"] == "eod" and e["date"] == "2026-10-07" for e in acc.values())
        assert acc["PRE"]["ret"] == pytest.approx(0.0100)
        assert "max quote/EOD gap 59.4bp" in ss.describe_fmp(stats)

    def test_before_settle_it_is_the_pre_558_behaviour(self, monkeypatch):
        out, stats, base = _stage(monkeypatch, datetime(2026, 10, 7, 16, 20, tzinfo=ET),
                                  SHAPE_1007)
        # Codex R1: before settle the late band is closed too (pre-#558 window).
        assert sorted(out) == ["NASDAQ"]
        assert stats["fmp_rejects"] == {"intraday": 2, "after_close_window": 1}
        assert all(e["src"] == "quote" for e in stats["fmp_accepted"].values())
        # Before settle the QUOTE is the close, not the (possibly partial) EOD row.
        assert out["NASDAQ"][0].iloc[-1] == pytest.approx(base["NASDAQ"] * 1.0099)
        assert "before EOD settle" in ss.describe_fmp(stats)


# ----------------------------------------------------------- reconciliation
def _yahoo(days_to_close: dict, sym="U0"):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days_to_close])
    cols = pd.MultiIndex.from_product([["Close"], [sym]])
    return pd.DataFrame(list(days_to_close.values()), index=idx, columns=cols)


class TestReconcile:
    def test_agreeing_return_is_checked_and_dropped(self):
        data = _yahoo({TUE: 100.0, WED: 101.0})
        keep, rep = ss.reconcile_fmp_closes(
            {"U0": {"date": "2026-10-07", "ret": 0.0100, "src": "eod"}}, data, THU)
        assert keep == {} and rep["checked"] == 1 and rep["disagree"] == {}

    def test_a_later_dividend_adjustment_is_not_a_disagreement(self):
        # Yahoo is auto-adjusted: an ex-date on THU rescales every earlier close.
        data = _yahoo({TUE: 100.0 * 0.98, WED: 101.0 * 0.98})
        _, rep = ss.reconcile_fmp_closes(
            {"U0": {"date": "2026-10-07", "ret": 0.0100}}, data, THU)
        assert rep["disagree"] == {}

    def test_disagreement_is_named_with_its_gap(self):
        data = _yahoo({TUE: 100.0, WED: 101.0})
        # 3 bp is inside the 5 bp tolerance; 10 bp is not.
        keep, rep = ss.reconcile_fmp_closes(
            {"U0": {"date": "2026-10-07", "ret": 0.0097}}, data, THU)
        assert keep == {} and rep["checked"] == 1 and rep["disagree"] == {}
        _, rep = ss.reconcile_fmp_closes(
            {"U0": {"date": "2026-10-07", "ret": 0.0090}}, data, THU)
        assert list(rep["disagree"]) == ["U0"] and rep["disagree"]["U0"] == pytest.approx(9.9, abs=0.1)
        assert "U0 9.9bp" in ss.describe_fmp_reconcile(rep)

    def test_not_backfilled_stays_pending_then_expires(self):
        data = _yahoo({TUE: 100.0})
        e = {"U0": {"date": "2026-10-07", "ret": 0.01}}
        keep, rep = ss.reconcile_fmp_closes(e, data, THU)
        assert keep == e and rep["pending"] == 1 and rep["checked"] == 0
        keep, rep = ss.reconcile_fmp_closes(e, data, WED + timedelta(
            days=ss.FMP_RECONCILE_MAX_AGE_DAYS + 1))
        assert keep == {} and rep["expired"] == 1

    def test_a_gap_before_the_date_is_not_compared(self):
        # Yahoo lacks TUE: the bar before WED is MON, a two-session return.
        data = _yahoo({date(2026, 10, 5): 99.0, WED: 101.0})
        keep, rep = ss.reconcile_fmp_closes(
            {"U0": {"date": "2026-10-07", "ret": 0.01}}, data, THU)
        assert rep["checked"] == 0 and "U0" in keep

    def test_same_day_entry_is_kept(self):
        keep, rep = ss.reconcile_fmp_closes(
            {"U0": {"date": "2026-10-07", "ret": 0.01}}, None, WED)
        assert "U0" in keep and rep["checked"] == 0

    def test_a_zero_prior_close_is_named_not_raised(self):
        # Fable R2 High: this divided outside the try and killed the close run.
        data = _yahoo({TUE: 0.0, WED: 101.0})
        keep, rep = ss.reconcile_fmp_closes(
            {"U0": {"date": "2026-10-07", "ret": 0.01}}, data, THU)
        assert keep == {} and rep["checked"] == 1 and rep["disagree"] == {"U0": None}
        assert "U0 ?" in ss.describe_fmp_reconcile(rep)

    def test_duplicate_index_and_nan_do_not_raise(self):
        idx = pd.DatetimeIndex([pd.Timestamp(TUE), pd.Timestamp(WED), pd.Timestamp(WED)])
        cols = pd.MultiIndex.from_product([["Close"], ["U0"]])
        data = pd.DataFrame([100.0, float("nan"), 101.0], index=idx, columns=cols)
        _, rep = ss.reconcile_fmp_closes(
            {"U0": {"date": "2026-10-07", "ret": 0.01}}, data, THU)
        assert rep["checked"] == 1 and rep["disagree"] == {}

    def test_a_same_sign_bias_is_reported_per_source(self):
        # Fable R2: 3 bp under the line on every name reads "0 disagree"; the
        # mean SIGNED gap per src is what exposes an unsettled EOD row.
        cols = pd.MultiIndex.from_product([["Close"], ["U0", "U1"]])
        data = pd.DataFrame([[100.0, 100.0], [101.0, 101.0]],
                            index=pd.DatetimeIndex([pd.Timestamp(TUE), pd.Timestamp(WED)]),
                            columns=cols)
        pend = {t: {"date": "2026-10-07", "ret": 1.0103 / 1.0 - 1, "src": "eod"}
                for t in ("U0", "U1")}
        _, rep = ss.reconcile_fmp_closes(pend, data, THU)
        assert rep["disagree"] == {}
        text = ss.describe_fmp_reconcile(rep)
        assert "2 checked, 0 disagree; eod n=2 mean +3.0bp max 3.0bp" in text, text

    def test_a_long_weekend_is_still_inside_the_expiry(self):
        fri = date(2026, 10, 9)
        e = {"U0": {"date": fri.isoformat(), "ret": 0.01}}
        keep, rep = ss.reconcile_fmp_closes(e, _yahoo({TUE: 100.0}), date(2026, 10, 13))
        assert keep == e and rep["expired"] == 0

    def test_the_refetch_merged_series_wins(self):
        # Codex R1: raw batch agrees, the fresher re-fetch corrected WED to 100.
        data = _yahoo({TUE: 100.0, WED: 101.0})
        merged = (pd.Series([100.0, 100.0], index=pd.DatetimeIndex(
            [pd.Timestamp(TUE), pd.Timestamp(WED)])),)
        _, rep = ss.reconcile_fmp_closes(
            {"U0": {"date": "2026-10-07", "ret": 0.01}}, data, THU,
            overrides={"U0": merged})
        assert list(rep["disagree"]) == ["U0"]

    def test_non_dict_pending_and_future_dates(self):
        assert ss.reconcile_fmp_closes(["x"], None, THU) == (
            {}, {"checked": 0, "disagree": {}, "pending": 0, "expired": 1, "by_src": {}})
        keep, rep = ss.reconcile_fmp_closes(
            {"U0": {"date": "2027-01-01", "ret": 0.01}}, None, THU)
        assert keep == {} and rep["expired"] == 1

    def test_names_are_capped_for_slack(self):
        rep = {"checked": 300, "disagree": {f"T{i:03d}": 9.9 for i in range(300)}}
        text = ss.describe_fmp_reconcile(rep)
        assert "+275 more" in text and len(text) < 1000

    def test_garbage_never_raises(self):
        keep, rep = ss.reconcile_fmp_closes({"U0": {"date": "x"}, "U1": "nope"},
                                            None, THU)
        assert keep == {} and rep["expired"] == 2


# ------------------------------------------------------- two runs, end to end
class TestTwoCloses:
    """Wednesday's close recovers 30 names through FMP; Thursday's close sees
    Yahoo's backfilled Wednesday bar and reconciles them through main()."""

    N = 30

    def _wed(self, monkeypatch, tmp_path):
        spec = {f"U{i}": TUE for i in range(self.N)}
        frame = _frame(_bdays_ending(WED), spec)
        base = float(frame["Close"]["U0"].dropna().iloc[-1])

        def q(sym, k):
            return {"symbol": sym, "price": base * 1.001, "open": base,
                    "dayHigh": base * 1.01, "dayLow": base * 0.99,
                    "previousClose": base, "timestamp": _stamp(WED, 15, 59, 59)}
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.setattr(ss, "fmp_get_quote", q)
        monkeypatch.setattr(ss, "fmp_get_eod_price", lambda s, k, d: base * 1.0012)
        run = _run_close(monkeypatch, tmp_path, now_et=AT_2106, tickers=list(spec),
                         frame=frame, prior_cache=_cache(spec, TUE), refetch_frame=None)
        return run, frame, base

    def test_wednesday_recovers_pre_auction_names_and_records_them(self, monkeypatch, tmp_path):
        run, _, base = self._wed(monkeypatch, tmp_path)
        hb = run.heartbeats[0]
        assert f"fmp recovered {self.N}/{self.N}" in hb, hb
        saved = run.saved[-1]
        assert len(saved["fmp_closes"]) == self.N
        assert saved["fmp_closes"]["U0"]["ret"] == pytest.approx(0.0012)

    def _thu(self, monkeypatch, tmp_path, cache, mode="close"):
        spec = {f"U{i}": THU for i in range(self.N)}
        thu = _frame(_bdays_ending(THU), spec)
        for t in spec:
            bump = 1.0032 if t in ("U3", "U7") else 1.0012
            thu.loc[pd.Timestamp(WED), ("Close", t)] = (
                thu.loc[pd.Timestamp(TUE), ("Close", t)] * bump)
        return _run_close(monkeypatch, tmp_path,
                          now_et=datetime(2026, 10, 8, 21, 0, tzinfo=ET),
                          tickers=list(spec), frame=thu, prior_cache=cache,
                          refetch_frame=None, mode=mode)

    def test_a_clean_reconcile_is_still_shown_and_persisted(self, monkeypatch, tmp_path):
        # Codex R1: with 0 disagreements the bias line must still reach the
        # heartbeat, and the report must survive a failed status POST.
        run, _, _ = self._wed(monkeypatch, tmp_path)
        spec = {f"U{i}": THU for i in range(self.N)}
        thu = _frame(_bdays_ending(THU), spec)
        for t in spec:
            thu.loc[pd.Timestamp(WED), ("Close", t)] = (
                thu.loc[pd.Timestamp(TUE), ("Close", t)] * 1.0012)
        run2 = _run_close(monkeypatch, tmp_path,
                          now_et=datetime(2026, 10, 8, 21, 0, tzinfo=ET),
                          tickers=list(spec), frame=thu, prior_cache=run.saved[-1],
                          refetch_frame=None)
        hb = run2.heartbeats[0]
        assert f"prior FMP closes vs Yahoo: {self.N} checked, 0 disagree; eod n={self.N}" in hb, hb
        last = run2.saved[-1]["fmp_reconcile_last"]
        assert last["date"] == "2026-10-08" and last["checked"] == self.N

    def test_midday_does_not_reconcile(self, monkeypatch, tmp_path):
        # Fable R2 L5: only the close route reports (midday would repeat it).
        run, _, _ = self._wed(monkeypatch, tmp_path)
        run2 = self._thu(monkeypatch, tmp_path, run.saved[-1], mode="midday")
        assert "prior FMP closes" not in run2.heartbeats[0]

    def test_thursday_names_every_disagreeing_close(self, monkeypatch, tmp_path):
        run, frame, base = self._wed(monkeypatch, tmp_path)
        cache = run.saved[-1]
        # Thursday: Yahoo backfilled WED for everyone; U3 and U7 differ by ~20 bp.
        spec = {f"U{i}": THU for i in range(self.N)}
        thu = _frame(_bdays_ending(THU), spec)
        for t in spec:
            bump = 1.0032 if t in ("U3", "U7") else 1.0012
            thu.loc[pd.Timestamp(WED), ("Close", t)] = (
                thu.loc[pd.Timestamp(TUE), ("Close", t)] * bump)
        run2 = _run_close(monkeypatch, tmp_path,
                          now_et=datetime(2026, 10, 8, 21, 0, tzinfo=ET),
                          tickers=list(spec), frame=thu, prior_cache=cache,
                          refetch_frame=None)
        hb = run2.heartbeats[0]
        assert "prior FMP closes vs Yahoo:" in hb, hb
        m = re.search(r"prior FMP closes vs Yahoo: (\d+) checked, 2 disagree \(U3 [\d.]+bp, U7 [\d.]+bp\)", hb)
        assert m, hb
        assert run2.saved[-1].get("fmp_closes", {}) == {}


def test_corrupt_fmp_cache_keys_do_not_kill_the_run(monkeypatch):
    """Codex R2: a non-dict `fmp_closes` raised in dict() before any guard."""
    monkeypatch.setattr(ss, "batch_download", lambda *a, **k: None)
    out = ss.screen_full([], "close", prior_cache={"fmp_closes": ["x"],
                                                  "fmp_reconcile_last": "y"})
    assert "fmp_closes" not in out[1] and "fmp_reconcile_last" not in out[1]


def test_reconcile_report_survives_a_later_save(monkeypatch):
    """Codex R2: a later close with nothing pending must not erase the last
    report from the committed cache."""
    monkeypatch.setattr(ss, "batch_download", lambda *a, **k: None)
    last = {"date": "2026-10-08", "checked": 3, "disagree": {"U1": 9.0}}
    out = ss.screen_full([], "close", prior_cache={"fmp_reconcile_last": last})
    assert out[1]["fmp_reconcile_last"] == last
