"""Duplicate close runs and the evening missing-bar case (board #327, Fable
plan review 2026-09-29).

What these pin, from the recorded CI logs rather than invented shapes:

* **2026-09-21 20:11 ET** - the watchdog-dispatched close had scored Monday at
  18:55; the lagged cron run then got Friday's bar for 712 names. The old code
  counted "already scored" as "no data" and posted `error` at 0%. It is a
  duplicate: `ok`, no digest, no cache save, the skip-log entry untouched.
* **2026-09-28 21:04 ET** - the one genuine loss: 712 names' newest bar was
  Friday 09-25 and no earlier run had scored Monday. `error`, with the measured
  histogram instead of "(Yahoo throttling?)", and the re-fetch is attempted and
  its recovery counted.
* **Partial first run** - an earlier run scored 85%; this run scores the late
  15% with 2 alerts. Duplicate, the 2 alerts are NAMED in the heartbeat (H2
  minimum form), watermarks advance only for the late names, and the skip-log
  entry shrinks by exactly those names.
* The heartbeat's cycle date is the ET session date, not the UTC runner date.

Every scenario drives the real `main()` with only the I/O edges replaced, and
asserts the negative (nothing reaches Slack's digest webhook) at the transport.
"""
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import sigma_screener as ss  # noqa: E402

DIGEST_HOOK = "https://example.invalid/digest"
STATUS_HOOK = "https://example.invalid/status"


# ----------------------------------------------------------------- builders
def _bdays_ending(end: date, n: int = 60) -> pd.DatetimeIndex:
    return pd.bdate_range(end=pd.Timestamp(end), periods=n)


def _frame(index, spec: dict, jump: set | None = None) -> pd.DataFrame:
    """yf.download-shaped (field, symbol) MultiIndex frame.

    `spec` = {ticker: newest bar date, or None for no data at all}. Bars after
    a ticker's newest date are NaN, exactly how a batch presents a name Yahoo
    has no current bar for. Tickers in `jump` get a +15% move on their newest
    bar (a real 2-sigma+ alert)."""
    jump = jump or set()
    n = len(index)
    # Alternating +/-0.4% days: sigma ~0.4% and every ordinary day is ~1 sigma,
    # so ONLY the deliberate jumps can reach 2 sigma (random noise put ~5% of
    # a 113-name universe over the line and made alert counts non-deterministic).
    base = 100 * np.cumprod(1 + 0.004 * (-1.0) ** np.arange(n))
    dates = index.date
    close = {}
    for t, newest in spec.items():
        if newest is None:
            close[t] = np.full(n, np.nan)
            continue
        arr = base.copy()
        mask = dates <= newest
        if t in jump:
            k = int(np.flatnonzero(mask)[-1])
            arr[k] = arr[k - 1] * 1.15
        close[t] = np.where(mask, arr, np.nan)
    c = pd.DataFrame(close, index=index)
    # Built in one concat: per-column assignment into a 3,000-column MultiIndex
    # frame is quadratic and made this file take minutes.
    return pd.concat({"Open": c, "High": c * 1.001, "Low": c * 0.999, "Close": c},
                     axis=1)


def _cache(watermarks: dict, day: date, published: date | None = None) -> dict:
    """A distribution cache as the previous close run left it. `published` =
    the day an earlier close run's digest was DELIVERED (the explicit marker)."""
    extra = ({"published": {"date": published.isoformat(), "mode": "close"}}
             if published else {})
    return {"date": day.isoformat(), **extra, "tickers": {
        t: {"mu": 0.0, "sigma": 0.01, "sample_size": 252, "last_bar": wm.isoformat(),
            "high_52w": 110.0, "low_52w": 90.0}
        for t, wm in watermarks.items() if wm is not None}}


class _Run:
    def __init__(self):
        self.posts = []
        self.saved = []
        self.refetch_calls = []

    @property
    def digests(self):
        return [p for u, p in self.posts if u == DIGEST_HOOK]

    @property
    def heartbeats(self):
        return ["\n".join(bl["text"]["text"] for bl in p["blocks"])
                for u, p in self.posts if u == STATUS_HOOK]


def _run_close(monkeypatch, tmp_path, *, now_et: datetime, tickers, frame,
               prior_cache, skip_log=None, refetch_frame=None, mode="close",
               digest_fails=False) -> _Run:
    run = _Run()
    monkeypatch.setattr(ss, "now_et", lambda: now_et)
    monkeypatch.setattr(ss, "CACHE_PATH", tmp_path / "distribution_cache.json")
    monkeypatch.setattr(ss, "SKIP_LOG_PATH", tmp_path / "skip_log.json")
    monkeypatch.setattr(ss, "MISSING_METADATA_PATH", tmp_path / "missing_metadata.json")
    ss.CACHE_PATH.write_text(json.dumps(prior_cache))
    if skip_log is not None:
        ss.SKIP_LOG_PATH.write_text(json.dumps(skip_log))

    monkeypatch.setattr(ss, "load_watchlist", lambda: list(tickers))
    for name in ("load_metadata", "load_sp500_names", "load_etf_names",
                 "load_etf_weighting"):
        monkeypatch.setattr(ss, name, lambda: {})
    for name in ("load_sp500_set", "load_portfolio", "load_researching",
                 "load_following_for_interest", "load_ready_to_buy",
                 "load_ready_to_short", "load_core_watchlist", "load_index_etfs",
                 "load_global_equity_etfs", "load_sector_etfs",
                 "load_healthcare_etfs", "load_tech_etfs",
                 "load_commodity_etfs", "load_macro"):
        monkeypatch.setattr(ss, name, lambda: set())
    monkeypatch.setattr(ss, "batch_download", lambda *a, **k: frame)
    monkeypatch.setattr(ss, "fallback_download_single", lambda *a, **k: None)
    monkeypatch.setattr(ss, "fetch_etf_period_returns", lambda *a, **k: {})
    for name in ("fetch_credit_indices", "fetch_treasury_curve", "fetch_mortgage_rate"):
        monkeypatch.setattr(ss, name, lambda: {})

    def _refetch(symbols):
        run.refetch_calls.append(list(symbols))
        return refetch_frame
    monkeypatch.setattr(ss, "refetch_recent", _refetch, raising=False)

    real_save = ss.save_cache

    def _spy_save(cache):
        run.saved.append(cache)
        real_save(cache)
    monkeypatch.setattr(ss, "save_cache", _spy_save)

    monkeypatch.setenv("SLACK_WEBHOOK", DIGEST_HOOK)
    monkeypatch.setenv("SLACK_STATUS_REPORTS_WEBHOOK", STATUS_HOOK)

    class _Resp:
        def raise_for_status(self):
            return None

    def _post(url, json=None, timeout=None):
        assert url in (DIGEST_HOOK, STATUS_HOOK), f"unexpected network call to {url}"
        if url == DIGEST_HOOK and digest_fails:
            raise ss.requests.RequestException("500 from Slack")
        run.posts.append((url, json))
        return _Resp()
    monkeypatch.setattr(ss.requests, "post", _post)
    monkeypatch.setattr(sys, "argv", ["sigma_screener.py", "--mode", mode])
    ss.main()
    return run


ET = ss.ET


# ------------------------------------------------ 2026-09-21 20:11 ET shape
MON_0921, FRI_0918 = date(2026, 9, 21), date(2026, 9, 18)


def _shape_0921():
    """712 already scored (latest 09-18, last_bar 09-21) + 10 already scored
    with the Monday bar + 22 late European (latest = last_bar = 09-18) + 6 Tokyo
    (holiday; same) + 5 with no data."""
    spec, wm = {}, {}
    for i in range(712):
        spec[f"A{i}"], wm[f"A{i}"] = FRI_0918, MON_0921
    for i in range(10):
        spec[f"M{i}"], wm[f"M{i}"] = MON_0921, MON_0921
    for i in range(22):
        spec[f"E{i}.DE"], wm[f"E{i}.DE"] = FRI_0918, FRI_0918
    for i in range(6):
        spec[f"J{i}.T"], wm[f"J{i}.T"] = FRI_0918, FRI_0918
    for i in range(5):
        spec[f"N{i}"], wm[f"N{i}"] = None, None
    return spec, wm


class TestSeptember21Duplicate:
    @pytest.fixture
    def run(self, monkeypatch, tmp_path):
        spec, wm = _shape_0921()
        tickers = list(spec)
        earlier_entry = {"date": "2026-09-21", "mode": "close",
                         "skipped": [{"ticker": "E0.DE", "reason": "stale_bar"}]}
        self.skip_log = {"runs": [earlier_entry]}
        r = _run_close(monkeypatch, tmp_path,
                       now_et=datetime(2026, 9, 21, 20, 11, tzinfo=ET),
                       tickers=tickers, frame=_frame(_bdays_ending(MON_0921), spec),
                       prior_cache=_cache(wm, MON_0921, published=MON_0921),
                       skip_log=self.skip_log)
        r.skip_log_after = json.loads(ss.SKIP_LOG_PATH.read_text())
        return r

    def test_no_digest_is_posted(self, run):
        assert run.digests == []

    def test_heartbeat_is_ok_attempt_2_not_error(self, run):
        assert len(run.heartbeats) == 1
        hb = run.heartbeats[0]
        assert "sigma-alert — ok" in hb, hb
        assert "attempt: 2" in hb
        assert "722 already scored today" in hb
        assert "Yahoo throttling" not in hb

    def test_cache_is_not_saved(self, run):
        assert run.saved == []

    def test_skip_log_entry_is_untouched(self, run):
        assert run.skip_log_after == self.skip_log


# ------------------------------------------------ 2026-09-28 21:04 ET shape
MON_0928, FRI_0925 = date(2026, 9, 28), date(2026, 9, 25)


def _shape_0928():
    """712 US names on Friday's bar, never scored Monday; 18 (foreign/indices)
    carry Monday's bar."""
    spec, wm = {}, {}
    for i in range(712):
        spec[f"U{i}"], wm[f"U{i}"] = FRI_0925, FRI_0925
    for i in range(18):
        spec[f"F{i}"], wm[f"F{i}"] = MON_0928, FRI_0925
    return spec, wm


def _refetch_with_monday(symbols, n_recovered):
    """A period=5d frame in which the first `n_recovered` symbols DO carry the
    Monday bar (the question the re-fetch exists to answer)."""
    idx = _bdays_ending(MON_0928, 5)
    spec = {s: (MON_0928 if i < n_recovered else FRI_0925)
            for i, s in enumerate(symbols)}
    return _frame(idx, spec)


class TestSeptember28GenuineLoss:
    def _go(self, monkeypatch, tmp_path, n_recovered):
        spec, wm = _shape_0928()
        tickers = list(spec)
        behind = [t for t in tickers if t.startswith("U")]
        refetch = _refetch_with_monday(behind, n_recovered)
        return _run_close(monkeypatch, tmp_path,
                          now_et=datetime(2026, 9, 28, 21, 4, tzinfo=ET),
                          tickers=tickers, frame=_frame(_bdays_ending(MON_0928), spec),
                          prior_cache=_cache(wm, FRI_0925), refetch_frame=refetch)

    def test_no_recovery_is_error_with_the_measured_histogram(self, monkeypatch, tmp_path):
        run = self._go(monkeypatch, tmp_path, n_recovered=0)
        assert run.digests == []
        hb = run.heartbeats[0]
        assert "sigma-alert — error" in hb
        assert "behind 712 (newest bar: 2026-09-25 ×712)" in hb, hb
        assert "re-fetch recovered 0/712" in hb
        assert "Yahoo throttling" not in hb
        assert "attempt: 2" not in hb          # not a duplicate: nothing scored Monday

    def test_re_fetch_is_attempted_for_exactly_the_behind_names(self, monkeypatch, tmp_path):
        run = self._go(monkeypatch, tmp_path, n_recovered=0)
        assert len(run.refetch_calls) == 1
        assert sorted(run.refetch_calls[0]) == sorted(f"U{i}" for i in range(712))

    def test_recovered_names_are_scored_and_the_run_publishes(self, monkeypatch, tmp_path):
        run = self._go(monkeypatch, tmp_path, n_recovered=700)
        assert len(run.digests) == 1
        hb = run.heartbeats[0]
        assert "718/730 tickers screened" in hb, hb
        assert "re-fetch recovered 700/712" in hb
        saved = run.saved[-1]["tickers"]
        assert saved["U0"]["last_bar"] == "2026-09-28"      # merged frame scored
        assert saved["U0"]["high_52w"] is not None          # 400-day history kept (M3)
        assert saved["U711"]["last_bar"] == "2026-09-25"    # unrecovered: not advanced


# -------------------------------------------------- partial-first-run shape
TUE = date(2026, 9, 29)


class TestPartialFirstRun:
    @pytest.fixture
    def run(self, monkeypatch, tmp_path):
        spec, wm = {}, {}
        for i in range(643):                       # first run scored these (85%)
            spec[f"S{i}"], wm[f"S{i}"] = TUE - timedelta(days=1), TUE
        late = [f"L{i}" for i in range(113)]       # arrived after the first run
        for t in late:
            spec[t], wm[t] = TUE, TUE - timedelta(days=1)
        self.late = set(late)
        earlier = {"date": TUE.isoformat(), "mode": "close", "skipped":
                   [{"ticker": t, "reason": "stale_bar"} for t in late]
                   + [{"ticker": "DEAD", "reason": "insufficient_history"}]}
        self.earlier = earlier
        r = _run_close(monkeypatch, tmp_path,
                       now_et=datetime(2026, 9, 29, 19, 50, tzinfo=ET),
                       tickers=list(spec),
                       frame=_frame(_bdays_ending(TUE), spec, jump={"L0", "L1"}),
                       prior_cache=_cache(wm, TUE, published=TUE),
                       skip_log={"runs": [earlier]})
        r.skip_log_after = json.loads(ss.SKIP_LOG_PATH.read_text())
        return r

    def test_no_second_digest(self, run):
        assert run.digests == []

    def test_heartbeat_names_the_unposted_alerts(self, run):
        hb = run.heartbeats[0]
        assert "attempt: 2" in hb
        assert "NOT POSTED:* 2 alert(s)" in hb, hb
        assert "`L0`" in hb and "`L1`" in hb

    def test_watermark_advances_only_for_the_late_names(self, run):
        saved = run.saved[-1]["tickers"]
        assert all(saved[t]["last_bar"] == TUE.isoformat() for t in self.late)
        assert saved["S0"]["last_bar"] == TUE.isoformat()   # carried, unchanged
        assert saved["S0"]["mu"] == 0.0                     # the prior entry, not rescored

    def test_skip_log_entry_shrinks_by_exactly_the_late_names(self, run):
        entry = [r for r in run.skip_log_after["runs"] if r["date"] == TUE.isoformat()]
        assert len(entry) == 1
        assert entry[0]["skipped"] == [{"ticker": "DEAD", "reason": "insufficient_history"}]


# ------------------------------------------------------------ is_duplicate_run
class TestDuplicateDefinition:
    def test_watermarks_without_a_publication_are_not_a_duplicate(self):
        """Codex R1: run A scores 604 (79.9%, refused), run B the other 152
        (refused), so every last_bar is today and nothing was ever posted."""
        assert ss.is_duplicate_run({"screened": 0, "already_scored": 756,
                                    "earlier_published": False}, 756) is False

    def test_zero_already_scored_is_never_a_duplicate(self):
        assert ss.is_duplicate_run({"screened": 756, "already_scored": 0,
                                    "earlier_published": True}, 756) is False

    def test_the_0921_counts(self):
        assert ss.is_duplicate_run({"screened": 0, "already_scored": 722,
                                    "earlier_published": True}, 755) is True

    def test_the_marker_is_per_day_and_per_mode(self):
        c = {"published": {"date": "2026-09-21", "mode": "close"}}
        assert ss.earlier_published_today(c, "close", MON_0921) is True
        assert ss.earlier_published_today(c, "midday", MON_0921) is False
        assert ss.earlier_published_today(c, "close", date(2026, 9, 22)) is False


class TestUnpublishedAccumulation:
    """Codex R1 P1, end to end: every watermark is today, no marker."""

    def test_is_error_not_an_ok_duplicate(self, monkeypatch, tmp_path):
        spec, wm = _shape_0921()
        run = _run_close(monkeypatch, tmp_path,
                         now_et=datetime(2026, 9, 21, 20, 11, tzinfo=ET),
                         tickers=list(spec), frame=_frame(_bdays_ending(MON_0921), spec),
                         prior_cache=_cache(wm, MON_0921))           # no marker
        hb = run.heartbeats[0]
        assert "sigma-alert — error" in hb, hb
        assert "attempt: 2" not in hb
        assert run.digests == []


class TestMiddayAfterClose:
    """Codex R1 P2: a delayed midday after today's close cache landed is not a
    re-run of anything, and must not claim `attempt: 2`."""

    def test_midday_is_never_a_duplicate(self, monkeypatch, tmp_path):
        spec, wm = _shape_0921()
        run = _run_close(monkeypatch, tmp_path,
                         now_et=datetime(2026, 9, 21, 20, 11, tzinfo=ET),
                         tickers=list(spec), frame=_frame(_bdays_ending(MON_0921), spec),
                         prior_cache=_cache(wm, MON_0921, published=MON_0921),
                         mode="midday")
        assert "attempt: 2" not in run.heartbeats[0]


class TestWholeBatchStale:
    """Codex R1 P1: a batch whose SHARED index ends on Friday used to abort
    before the re-fetch could run."""

    def test_re_fetch_runs_before_the_stale_abort_and_recovers(self, monkeypatch, tmp_path):
        spec = {f"U{i}": FRI_0925 for i in range(730)}
        wm = dict(spec)
        run = _run_close(monkeypatch, tmp_path,
                         now_et=datetime(2026, 9, 28, 21, 4, tzinfo=ET),
                         tickers=list(spec), frame=_frame(_bdays_ending(FRI_0925), spec),
                         prior_cache=_cache(wm, FRI_0925),
                         refetch_frame=_refetch_with_monday(list(spec), 730))
        assert len(run.refetch_calls) == 1
        assert len(run.digests) == 1
        assert "730/730 tickers screened" in run.heartbeats[0]


class TestPublicationMarker:
    def _go(self, monkeypatch, tmp_path, **kw):
        spec = {f"U{i}": MON_0928 for i in range(100)}
        wm = {t: FRI_0925 for t in spec}
        _run_close(monkeypatch, tmp_path,
                   now_et=datetime(2026, 9, 28, 21, 4, tzinfo=ET),
                   tickers=list(spec), frame=_frame(_bdays_ending(MON_0928), spec),
                   prior_cache=_cache(wm, FRI_0925), **kw)
        return json.loads(ss.CACHE_PATH.read_text())

    def test_a_delivered_close_writes_the_marker(self, monkeypatch, tmp_path):
        saved = self._go(monkeypatch, tmp_path)
        assert saved["published"] == {"date": "2026-09-28", "mode": "close"}

    def test_a_failed_post_writes_no_marker(self, monkeypatch, tmp_path):
        saved = self._go(monkeypatch, tmp_path, digest_fails=True)
        assert "published" not in saved


class TestEveryUnpostedAlertIsNamed:
    def test_forty_alerts_are_all_listed(self):
        alerts = [{"ticker": f"LATE{i:02d}", "z_score": 2.5} for i in range(40)]
        _, payload = ss.build_health_payload(
            "close", 113, 756, 40, 0, published=False, reason="dup",
            already_scored=643, unposted=alerts)
        body = "\n".join(b["text"]["text"] for b in payload["blocks"])
        assert all(f"`LATE{i:02d}`" in body for i in range(40))
        assert all(len(b["text"]["text"]) < 3000 for b in payload["blocks"])


# ---------------------------------------------------------------- cycle date
class TestCycleDateIsTheETSession:
    def test_0104_utc_is_the_previous_et_day(self, monkeypatch):
        """The runner's clock is UTC. 2026-09-29 01:04 UTC is 2026-09-28 21:04
        EDT, and the heartbeat must file under the 09-28 close."""
        fixed = datetime(2026, 9, 29, 1, 4, tzinfo=timezone.utc)

        class _RunnerClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

        monkeypatch.setattr(ss, "datetime", _RunnerClock)
        _, payload = ss.build_health_payload("close", 39, 756, 0, 717)
        text = payload["blocks"][0]["text"]["text"]
        assert "cycle: 2026-09-28 close" in text, text


# ------------------------------------------------------------------- dry run
class TestDryRunCoversEverySideEffect:
    def test_every_write_path_leaves_the_repo_and_nothing_posts(self, monkeypatch, tmp_path):
        import return_map
        for name in ("DRY_RUN", "CACHE_PATH", "SKIP_LOG_PATH", "MISSING_METADATA_PATH"):
            monkeypatch.setattr(ss, name, getattr(ss, name))
        monkeypatch.setattr(return_map, "SNAPSHOT_PATH", return_map.SNAPSHOT_PATH)
        monkeypatch.setattr(return_map, "HTML_PATH", return_map.HTML_PATH)

        d = ss._enter_dry_run(tmp_path / "state")
        root = ss.ROOT.resolve()
        for p in (ss.CACHE_PATH, ss.SKIP_LOG_PATH, ss.MISSING_METADATA_PATH,
                  return_map.SNAPSHOT_PATH, return_map.HTML_PATH):
            assert root not in Path(p).resolve().parents, f"{p} still under the repo"
            assert Path(p).resolve().parent == d.resolve()

        def _boom(*a, **k):
            raise AssertionError("dry run reached the network")
        monkeypatch.setattr(ss.requests, "post", _boom)
        monkeypatch.setenv("SLACK_WEBHOOK", DIGEST_HOOK)
        monkeypatch.setenv("SLACK_STATUS_REPORTS_WEBHOOK", STATUS_HOOK)
        assert ss.send_slack({"blocks": []}) is True
        ss.post_health_heartbeat("close", {"screened": 1}, 1, 0)


# ------------------------------------------------------------ the workflow
class TestCloseWorkflowSerialisesAndRefreshes:
    """H3: a concurrency group alone does not refresh a queued run's checkout."""
    WF = (Path(__file__).resolve().parent.parent / ".github" / "workflows"
          / "sigma-close.yml").read_text(encoding="utf-8")

    def test_concurrency_group_without_cancel(self):
        assert "group: sigma-close" in self.WF
        assert "cancel-in-progress: false" in self.WF

    def test_fetch_and_reset_precede_the_screener(self):
        reset = self.WF.index("git reset --hard FETCH_HEAD")
        assert self.WF.index("git fetch --depth=1 origin master") < reset
        assert reset < self.WF.index("python scripts/sigma_screener.py --mode close")


class TestMarkerNeverWipesTheCache:
    """Fable post-build H1: an unreadable cache must not be replaced by an empty
    one carrying only the marker (the workflow would commit the wipe)."""

    def test_unreadable_cache_is_left_untouched(self, monkeypatch, tmp_path):
        path = tmp_path / "distribution_cache.json"
        path.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(ss, "CACHE_PATH", path)
        ss.mark_published("close")
        assert path.read_text(encoding="utf-8") == "{not json"

    def test_readable_cache_keeps_its_tickers(self, monkeypatch, tmp_path):
        path = tmp_path / "distribution_cache.json"
        path.write_text(json.dumps({"date": "2026-09-28",
                                    "tickers": {"AAA": {"last_bar": "2026-09-28"}}}),
                        encoding="utf-8")
        monkeypatch.setattr(ss, "CACHE_PATH", path)
        ss.mark_published("close")
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert "AAA" in saved["tickers"] and "published" in saved
