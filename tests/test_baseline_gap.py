"""Board #551: a return must not span two sessions.

Production shape (2026-10-06 close): Yahoo had no 10-06 bar for any European
single stock. If 10-07 arrives first, dropna() leaves iloc[-2] = 10-05 and the
10-05 -> 10-07 move would be scored as one day.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sigma_screener as s  # noqa: E402


def _series(days, seed=0):
    idx = pd.to_datetime(days)
    rng = np.random.default_rng(seed)
    closes = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, len(idx))), index=idx)
    return closes, closes.shift(1).bfill()


def _weekdays_until(end, n=60, skip=()):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5 and d not in skip:
            out.append(d)
        d = d - pd.Timedelta(days=1)
    return sorted(out)


def test_a_missing_middle_session_is_refused_not_scored(monkeypatch):
    days = _weekdays_until(date(2026, 10, 7), skip={date(2026, 10, 6)})
    closes, opens = _series(days)
    monkeypatch.setattr(s, "today_et", lambda: date(2026, 10, 7))
    alert, entry, _, _, reason = s._process_ticker_full(
        "FRE", closes, opens, None, None, "close", {},
        require_current_bar=True, last_scored_bar=date(2026, 10, 5))
    assert reason == "gap_before_today"
    assert alert is None and entry is None


def test_adjacent_sessions_still_score(monkeypatch):
    days = _weekdays_until(date(2026, 10, 7))
    closes, opens = _series(days)
    monkeypatch.setattr(s, "today_et", lambda: date(2026, 10, 7))
    _, entry, _, _, reason = s._process_ticker_full(
        "FRE", closes, opens, None, None, "close", {},
        require_current_bar=True, last_scored_bar=date(2026, 10, 6))
    assert reason is None and entry is not None


def test_monday_after_a_weekend_is_adjacent():
    assert s._baseline_gap(*_series(_weekdays_until(date(2026, 10, 5))), "close") is None


def test_us_name_after_an_nyse_holiday_is_adjacent():
    # Thanksgiving 2026-11-26: a US series has no 11-26 bar; 11-27 vs 11-25 is one session.
    days = _weekdays_until(date(2026, 11, 27), skip={date(2026, 11, 26)})
    assert s._baseline_gap(*_series(days), "close") is None


def test_foreign_name_trading_through_an_nyse_holiday_is_adjacent():
    days = _weekdays_until(date(2026, 11, 27))  # Europe traded 11-26
    assert s._baseline_gap(*_series(days), "close") is None


def test_open_mode_checks_the_close_before_the_open_session():
    days = _weekdays_until(date(2026, 10, 7), skip={date(2026, 10, 6)})
    closes, opens = _series(days)
    # Open-mode shape: today's partial close may be absent; baseline is the last
    # close strictly before the open bar's own session.
    gap = s._baseline_gap(closes.iloc[:-1], opens, "open")
    assert gap == (date(2026, 10, 7), date(2026, 10, 5))


def test_plain_index_fixtures_are_not_judged():
    closes = pd.Series(np.arange(40, dtype=float) + 100)
    assert s._baseline_gap(closes, closes, "close") is None


def _batch(frames):
    """yf.download-shaped multi-ticker frame: columns (field, symbol)."""
    cols = {}
    for sym, (closes, opens) in frames.items():
        cols[("Close", sym)] = closes
        cols[("Open", sym)] = opens
    return pd.DataFrame(cols)


def test_cached_open_path_refuses_a_two_session_baseline(monkeypatch):
    """Codex #551 r2 High: the 14:35 UTC open run scores from scalars built in
    download_todays_prices and never reaches _baseline_gap. Production shape:
    FRE.DE 10-05 then 10-07 (no 10-06); a US name with every session."""
    full = _weekdays_until(date(2026, 10, 7), n=5)
    gapped = [d for d in full if d != date(2026, 10, 6)]
    fre_c, fre_o = _series(gapped)
    us_c, us_o = _series(full, seed=1)
    frame = _batch({"FRE.DE": (fre_c, fre_o), "ABT": (us_c, us_o)})
    monkeypatch.setattr(s, "today_et", lambda: date(2026, 10, 7))
    monkeypatch.setattr(s.yf, "download", lambda *a, **k: frame)
    prices = s.download_todays_prices(["FRE", "ABT"])
    assert "FRE" not in prices
    assert "ABT" in prices
