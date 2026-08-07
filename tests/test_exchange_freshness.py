"""A session that has not been scored yet must be scored, even if it is late.

The freshness guard compared every security against `today_et()`. It exists for
a real reason: a ticker with no bar for the current session gets `dropna()`'d
back to the previous one, and scoring that as today's move manufactures a
spurious 2 sigma. But `== today` conflates "is this bar current" with "is this
bar new", and only the second question is the one that matters.

GROUND TRUTH, from the 2026-08-06 close run's CI log (run 31136711177):

    26 x  latest close bar is 2026-08-05, expected 2026-08-06
     1 x  latest close bar is 2026-07-17, expected 2026-08-06

27 tickers skipped `stale_bar`, and 26 of them were behind by exactly one day —
the whole European book plus `BTC-USD` — because their exchange's EOD had not
reached Yahoo by the time the run fired. This had happened on every close run
since 07-20. Those sessions were real, and under the old rule they were never
scored at all.

A per-venue quorum was built first and rejected in adversarial review: a venue
whose feed has stalled has every one of its names agreeing on the same old date,
so the stale names certify each other and the SAME session is re-scored — and
re-alerted — every day inside the backstop window. A vote among tickers from one
feed cannot detect that feed being down.

Comparing against OUR OWN record of what we last scored cannot make that
mistake, and answers both edges with one condition.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sigma_screener as s  # noqa: E402

TODAY = date(2026, 8, 6)


# --- the rule -------------------------------------------------------------

def test_a_late_arriving_session_is_scored_not_skipped():
    """The live failure: 26 names carried an 08-05 bar on the 08-06 run.

    Those sessions were never scored. Under the old `== today` rule they never
    would be.
    """
    assert s.is_unscored_bar(date(2026, 8, 5), date(2026, 8, 4), today=TODAY)


def test_the_same_session_is_never_scored_twice():
    """The property the whole guard exists for, and the one the rejected
    per-venue design broke: an already-scored bar must not re-alert."""
    assert not s.is_unscored_bar(date(2026, 8, 5), date(2026, 8, 5), today=TODAY)


def test_a_bar_older_than_the_one_we_scored_is_not_new():
    assert not s.is_unscored_bar(date(2026, 8, 4), date(2026, 8, 5), today=TODAY)


def test_a_normal_current_bar_is_scored():
    assert s.is_unscored_bar(TODAY, date(2026, 8, 5), today=TODAY)


def test_a_bar_dated_after_today_is_refused():
    """A partial/forming session somewhere ahead of ET.

    Accepting it lets the SAME session be scored again by the next morning's
    cached-open run, which has no watermark of its own (Codex round 3). Nothing
    is given up by capping: the measured problem was always bars arriving LATE
    — the 2026-08-06 CI log is 26 x one-day-behind and zero ahead.
    """
    assert not s.is_unscored_bar(date(2026, 8, 7), date(2026, 8, 5), today=TODAY)
    assert not s.is_unscored_bar(date(2026, 8, 7), None, today=TODAY)


def test_a_stalled_feed_does_not_certify_itself():
    """The failure that killed the per-venue design, stated as a test.

    A feed stuck on 08-05 for three days: every day the bar is 08-05, and every
    day after the first it must be refused. A quorum among that venue's own
    tickers would have said 'the venue is on 08-05, so 08-05 is current' and
    re-alerted the same move daily.
    """
    scored = date(2026, 8, 5)
    for day in (date(2026, 8, 6), date(2026, 8, 7), date(2026, 8, 8)):
        assert not s.is_unscored_bar(date(2026, 8, 5), scored, today=day), (
            f"the 08-05 session must not be re-scored on {day}")


# --- the cold-cache fallback ----------------------------------------------

def test_no_record_falls_back_to_the_original_today_rule():
    """First run, or a cache written before `last_bar` existed.

    Being conservative on an unknown costs one cycle; guessing costs a false
    alert, so the pre-2026-08-07 behaviour is the fallback.
    """
    assert s.is_unscored_bar(TODAY, None, today=TODAY)
    assert not s.is_unscored_bar(date(2026, 8, 5), None, today=TODAY)


def test_a_missing_bar_is_not_this_guards_problem():
    """`latest_bar is None` means the series was not date-indexed (a plain
    fixture). The guard passes it through rather than inventing a verdict."""
    assert s.is_unscored_bar(None, date(2026, 8, 5), today=TODAY)


# --- reading the record off the cache -------------------------------------

def test_prior_bars_reads_last_bar_off_the_distribution_cache():
    cache = {"tickers": {"AAPL": {"mu": 0.0, "sigma": 1.0, "last_bar": "2026-08-05"},
                         "CVSG.L": {"mu": 0.0, "sigma": 1.0, "last_bar": "2026-08-04"}}}
    assert s.prior_bars_from_cache(cache) == {"AAPL": date(2026, 8, 5),
                                              "CVSG.L": date(2026, 8, 4)}


def test_entries_without_last_bar_are_absent_not_guessed():
    """An old cache entry must yield NO key, so the caller falls back to the
    today rule for that ticker instead of being handed a fabricated date."""
    cache = {"tickers": {"AAPL": {"mu": 0.0, "sigma": 1.0},
                         "MSFT": {"mu": 0.0, "sigma": 1.0, "last_bar": None},
                         "EA": {"mu": 0.0, "sigma": 1.0, "last_bar": "not-a-date"}}}
    assert s.prior_bars_from_cache(cache) == {}


def test_an_empty_or_missing_cache_is_not_an_error():
    for empty in (None, {}, {"tickers": {}}, {"tickers": None}):
        assert s.prior_bars_from_cache(empty) == {}


# --- the bootstrap, which decides whether any of this works ----------------
#
# `save_cache` REPLACES the file, so a ticker that skips vanishes from it. That
# is how the 26 late names came to have no cache entry at all. A ticker with no
# entry has no `last_bar`, so it falls back to the `== today` rule — the rule
# that was skipping it. Without carrying the refused bar forward, the whole fix
# is permanently inert for exactly the names it exists for.

def test_a_refused_ticker_keeps_a_record_so_it_can_recover_next_run():
    cache_data = {"tickers": {}}
    prior = {"CVSG.L": {"mu": 0.001, "sigma": 0.02, "sample_size": 252}}

    class _Idx:
        def __getitem__(self, i):
            class _TS:
                def date(self_inner):
                    return date(2026, 8, 5)
            return _TS()

    class _Series:
        index = _Idx()

    s._carry_refused_bar(cache_data, prior, "CVSG.L", _Series(), "stale_bar")

    entry = cache_data["tickers"]["CVSG.L"]
    assert entry["last_seen"] == "2026-08-05"
    assert "last_bar" not in entry, (
        "a REFUSED bar was not scored; writing it into the scored watermark "
        "would make that session permanently unscoreable if it was merely late")
    assert entry["mu"] == 0.001 and entry["sigma"] == 0.02, (
        "the prior distribution must survive, or the cached-open path breaks")
    # And the record is what lets the NEXT run pick the name up.
    recovered = s.prior_bars_from_cache(cache_data)
    assert s.is_unscored_bar(date(2026, 8, 6), recovered["CVSG.L"],
                             today=date(2026, 8, 6))


def test_a_refused_ticker_with_no_prior_distribution_is_parked_separately():
    """It must still get a record, but must not count as a screened ticker —
    `_cache_has_tickers` is what stops an empty batch overwriting a good cache.
    """
    cache_data = {"tickers": {}}

    class _Idx:
        def __getitem__(self, i):
            class _TS:
                def date(self_inner):
                    return date(2026, 8, 5)
            return _TS()

    class _Series:
        index = _Idx()

    s._carry_refused_bar(cache_data, {}, "AMBUSH.DC", _Series(), "stale_bar")
    assert cache_data["tickers"] == {}
    assert cache_data["refused_bars"] == {"AMBUSH.DC": "2026-08-05"}
    assert s.prior_bars_from_cache(cache_data) == {"AMBUSH.DC": date(2026, 8, 5)}


def _series_ending(d):
    class _Idx:
        def __getitem__(self, i):
            class _TS:
                def date(self_inner):
                    return d
            return _TS()

    class _Series:
        index = _Idx()

    return _Series()


def test_the_watermark_never_moves_backward():
    """Codex round 2. A refused bar is by definition NOT newer than what we
    scored, so writing it in unconditionally regresses the watermark whenever
    a feed falls back from D to D-1 — and when the feed recovers, D clears the
    strict `>` test and the same session is scored, and alerted, twice.
    """
    cache_data = {"tickers": {}}
    prior = {"CVSG.L": {"mu": 0.001, "sigma": 0.02, "last_bar": "2026-08-06"}}

    s._carry_refused_bar(cache_data, prior, "CVSG.L",
                         _series_ending(date(2026, 8, 5)), "stale_bar")

    assert cache_data["tickers"]["CVSG.L"]["last_bar"] == "2026-08-06", (
        "a D-1 regression must not lower a D watermark")
    # And the recovery must therefore NOT re-score 08-06.
    recovered = s.prior_bars_from_cache(cache_data)
    assert not s.is_unscored_bar(date(2026, 8, 6), recovered["CVSG.L"],
                                 today=date(2026, 8, 7))


def test_every_skip_reason_preserves_the_watermark():
    """Codex round 2. Only `stale_bar` used to be carried, so any other skip
    dropped the ticker from the replaced cache. A ticker that then recovered on
    a later close run the same day would find no record, hit the `>= today`
    cold-cache fallback, and score that session a second time.
    """
    for reason in ("insufficient_history", "distribution_nan",
                   "fallback_exception", "fallback_insufficient", None):
        cache_data = {"tickers": {}}
        prior = {"X": {"mu": 0.0, "sigma": 0.01, "last_bar": "2026-08-06"}}
        s._carry_refused_bar(cache_data, prior, "X",
                             _series_ending(date(2026, 8, 6)), reason)
        assert cache_data["tickers"]["X"]["last_bar"] == "2026-08-06", (
            f"{reason} must preserve the watermark")


def test_a_non_stale_skip_never_ADVANCES_the_watermark():
    """Preserving is not the same as advancing. `insufficient_history` means we
    never got a usable read, so it must not claim we scored through that bar.
    """
    cache_data = {"tickers": {}}
    prior = {"X": {"mu": 0.0, "sigma": 0.01, "last_bar": "2026-08-05"}}
    s._carry_refused_bar(cache_data, prior, "X",
                         _series_ending(date(2026, 8, 6)), "insufficient_history")
    assert cache_data["tickers"]["X"]["last_bar"] == "2026-08-05"


def test_every_skip_path_in_screen_full_carries_the_watermark():
    """The invariant stated over the source, not over one call.

    Four skip paths and one stale-fallback path all `continue` past the cache
    write. Each must carry the watermark, because `save_cache` replaces the
    file and a lost record silently re-enables the `>= today` fallback for that
    ticker — which is how the same session gets scored twice. A new skip path
    added without the call would be invisible; this makes it a test failure.
    """
    src = (Path(__file__).resolve().parents[1] / "scripts" /
           "sigma_screener.py").read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(src) if ln.startswith("def screen_full("))
    end = next(i for i, ln in enumerate(src[start + 1:], start + 1)
               if ln.startswith("def "))
    body = src[start:end]

    exits = [i for i, ln in enumerate(body)
             if 'skip_events.append(' in ln or 'stats["stale"] += 1' in ln]
    assert exits, "found no skip paths - the scan is looking in the wrong place"
    for i in exits:
        window = "\n".join(body[i:i + 6])
        assert "_carry_refused_bar" in window, (
            f"skip path at screen_full+{i} does not carry the watermark:\n"
            + "\n".join(body[i:i + 4]))


def test_a_watermark_parked_under_refused_bars_is_read_back_as_a_prior():
    """A ticker with no cached distribution parks its watermark under
    `refused_bars`, not `tickers`. `screen_full` must merge both when it builds
    the priors, or the monotonic guard is only half wired — it sees no prior
    for exactly those names and writes an older bar over a newer one.

    Found by the live four-run check, not by the unit tests above, which only
    ever exercised the `tickers` path. `EA`, `OSSFF`, `SHMZF` and `^W5000` all
    regressed a day.
    """
    cache = {"tickers": {"AAPL": {"mu": 0.0, "sigma": 0.01, "last_bar": "2026-08-07"}},
             "refused_bars": {"EA": "2026-08-07"}}
    merged = dict(cache["tickers"])
    for t, d in cache["refused_bars"].items():
        merged.setdefault(t, {"last_bar": d})

    cache_data = {"tickers": {}}
    s._carry_refused_bar(cache_data, merged, "EA",
                         _series_ending(date(2026, 8, 6)), "stale_bar")
    assert cache_data["refused_bars"]["EA"] == "2026-08-07", (
        "the 08-07 watermark must survive an 08-06 refusal")


def test_a_refused_session_stays_scoreable_if_it_was_merely_late():
    """Codex round 3. Writing a REFUSED bar into the scored watermark makes
    that session permanently unscoreable — the next run reads it as
    already-done. `last_seen` records what we looked at; `last_bar` records
    only what we scored, and the guard reads the newer of the two.
    """
    cache_data = {"tickers": {}}
    prior = {"CVSG.L": {"mu": 0.0, "sigma": 0.02}}          # cold: no watermark
    s._carry_refused_bar(cache_data, prior, "CVSG.L",
                         _series_ending(date(2026, 8, 5)), "stale_bar")

    entry = cache_data["tickers"]["CVSG.L"]
    assert entry["last_seen"] == "2026-08-05" and "last_bar" not in entry
    ref = s.prior_bars_from_cache(cache_data)["CVSG.L"]
    assert not s.is_unscored_bar(date(2026, 8, 5), ref, today=date(2026, 8, 6)), (
        "the same 08-05 bar is still not new")
    assert s.is_unscored_bar(date(2026, 8, 6), ref, today=date(2026, 8, 6)), (
        "but 08-06 arriving must be scoreable")


def test_the_fallback_path_uses_the_per_ticker_rule_not_the_legacy_one():
    """Codex round 3. `validate_bar_date` is the legacy `== today` rule.

    Applying it in the fallback path rejects a ticker BEFORE the per-ticker
    rule sees it — so a name chronically one session late, which is exactly the
    population the fallback serves and exactly the 26 the CI log measured, is
    rejected forever however new its bar is relative to what we scored.
    """
    src = (Path(__file__).resolve().parents[1] / "scripts" /
           "sigma_screener.py").read_text(encoding="utf-8")
    start = src.index("def screen_full(")
    end = src.index("\ndef ", start + 1)
    body = src[start:end]
    fb = body[body.index("Falling back to individual download"):]
    # Calls, not mentions — the comment explaining the decision names it too.
    code = "\n".join(ln for ln in fb.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "validate_bar_date(" not in code, (
        "the fallback path must not re-apply the legacy == today gate")
    assert "is_unscored_bar(" in code


def test_a_skip_with_nothing_on_record_writes_nothing():
    cache_data = {"tickers": {}}
    s._carry_refused_bar(cache_data, {}, "X",
                         _series_ending(date(2026, 8, 6)), "insufficient_history")
    assert cache_data == {"tickers": {}}, (
        "no prior watermark and no usable bar - there is nothing to claim")


# --- the round trip: what one run records, the next run reads --------------

def test_a_scored_run_records_the_bar_the_next_run_will_check(monkeypatch):
    """The two halves must agree on the key and the format, or the guard is
    silently inert — the cache would carry no usable record and every ticker
    would fall back to the today rule forever."""
    import numpy as np
    import pandas as pd

    idx = pd.to_datetime([date(2026, 7, 1) + pd.Timedelta(days=i)
                          for i in range(40)])
    rng = np.random.default_rng(0)
    closes = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, 40)), index=idx)
    opens = closes.shift(1).bfill()

    monkeypatch.setattr(s, "today_et", lambda: idx[-1].date())
    _, entry, _, _, reason = s._process_ticker_full(
        "CVSG.L", closes, opens, None, None, "close", {},
        require_current_bar=True, last_scored_bar=idx[-2].date())

    assert reason is None and entry is not None
    assert entry["last_bar"] == idx[-1].date().isoformat()
    assert s.prior_bars_from_cache({"tickers": {"CVSG.L": entry}}) == {
        "CVSG.L": idx[-1].date()}


def test_a_bar_already_scored_is_refused_end_to_end(monkeypatch):
    import numpy as np
    import pandas as pd

    idx = pd.to_datetime([date(2026, 7, 1) + pd.Timedelta(days=i)
                          for i in range(40)])
    rng = np.random.default_rng(1)
    closes = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, 40)), index=idx)
    opens = closes.shift(1).bfill()

    monkeypatch.setattr(s, "today_et", lambda: idx[-1].date() + pd.Timedelta(days=1))
    _, entry, _, _, reason = s._process_ticker_full(
        "CVSG.L", closes, opens, None, None, "close", {},
        require_current_bar=True, last_scored_bar=idx[-1].date())

    assert reason == "stale_bar" and entry is None


# --- the open-mode baseline, found by Codex --------------------------------

def test_open_mode_anchors_the_baseline_on_the_open_bars_own_session(monkeypatch):
    """Once a bar dated other than ET today can be scored, a `< today_et()`
    baseline picks the wrong close.

    With an 08-05 open accepted on 08-06, `< today_et()` selects the 08-05
    CLOSE — reporting the open against its own session's close. The baseline
    must be the last close strictly before the open bar's session.
    """
    import numpy as np
    import pandas as pd

    idx = pd.to_datetime([date(2026, 7, 1) + pd.Timedelta(days=i)
                          for i in range(40)])
    closes = pd.Series(np.linspace(100, 139, 40), index=idx)
    opens = pd.Series(np.linspace(100, 139, 40), index=idx)
    session = idx[-1].date()
    # ET "today" is deliberately a day AHEAD of the open bar's session — the
    # late-arriving-venue case. This is what makes the two baselines differ.
    monkeypatch.setattr(s, "today_et", lambda: session + pd.Timedelta(days=1))

    alert, entry, _, stats, reason = s._process_ticker_full(
        "CVSG.L", closes, opens, None, None, "open", {},
        require_current_bar=True, last_scored_bar=idx[-2].date())

    assert reason is None
    expected = (float(opens.iloc[-1]) - float(closes.iloc[-2])) / float(closes.iloc[-2])
    assert abs(stats["return_pct"] / 100 - expected) < 1e-9, (
        "the baseline must be the close before this open's own session, "
        f"not whatever precedes ET today (session={session})")
