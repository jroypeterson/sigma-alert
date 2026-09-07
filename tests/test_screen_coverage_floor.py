"""Screen-coverage floor (board #327).

The defect these pin: from 2026-09-03 the close cycle screened 0/769, then
0/757, then 34/757 and 36/757 on 09-07 — and the 09-07 midday run still POSTED,
carrying 2 alerts drawn from 4.8% of the universe. `error` was a heartbeat
colour that changed nothing about what got published.

Every test here is written against a coverage value the OLD code would have
accepted, not against garbage — a floor that only rejects 0/757 would have
passed on 09-07 and is not the fix.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import sigma_screener as ss  # noqa: E402


# --------------------------------------------------------------- coverage math
class TestScreenCoverage:
    def test_empty_universe_is_zero_coverage_not_full(self):
        """0/0 must NOT read as 100%. An empty watchlist has nothing to cover,
        and the publish gate would otherwise wave through a run that screened
        literally nothing."""
        assert ss.screen_coverage(0, 0) == 0.0
        assert ss.coverage_is_publishable(0, 0) is False

    @pytest.mark.parametrize("screened,total,expected", [
        (0, 757, 0.0),
        (36, 757, 36 / 757),
        (757, 757, 1.0),
    ])
    def test_coverage_fraction(self, screened, total, expected):
        assert ss.screen_coverage(screened, total) == pytest.approx(expected)


# ------------------------------------------------------------- the floor itself
class TestPublishGate:
    def test_the_actual_2026_09_07_midday_run_is_refused(self):
        """36/757 = 4.8%. This exact run posted 2 alerts under the old logic."""
        assert ss.coverage_is_publishable(36, 757) is False

    def test_the_actual_2026_09_04_close_run_is_refused(self):
        assert ss.coverage_is_publishable(0, 757) is False

    def test_a_60_percent_screen_is_refused(self):
        """The old threshold was 0.5, so 60% published clean with no banner at
        all. It is now below the floor and publishes nothing — this is the
        plausible-wrong-value case, not the garbage case."""
        assert ss.coverage_is_publishable(454, 757) is False

    def test_just_below_the_floor_is_refused(self):
        assert ss.coverage_is_publishable(605, 757) is False   # 79.9%

    def test_exactly_at_the_floor_publishes(self):
        """The floor is inclusive — >= MIN_SCREEN_COVERAGE, not >."""
        total = 1000
        assert ss.coverage_is_publishable(int(total * ss.MIN_SCREEN_COVERAGE), total) is True

    def test_a_full_screen_publishes(self):
        assert ss.coverage_is_publishable(757, 757) is True

    def test_floor_sits_below_the_degraded_threshold(self):
        """If these two ever cross, the published-but-degraded band vanishes and
        the banner becomes unreachable."""
        assert ss.MIN_SCREEN_COVERAGE < ss.DEGRADED_SCREEN_COVERAGE


# ------------------------------------------------------------------- heartbeat
def _text(payload):
    return payload["blocks"][0]["text"]["text"]


class TestHealthPayload:
    def test_below_floor_is_error_and_says_it_did_not_publish(self):
        status, payload = ss.build_health_payload("midday", 36, 757, 2, 721)
        assert status == "error"
        body = _text(payload)
        assert "NOT PUBLISHED" in body
        assert "80%" in body            # the floor is named, not implied
        assert "36/757" in body

    def test_zero_screened_is_error(self):
        status, payload = ss.build_health_payload("close", 0, 757, 0, 757)
        assert status == "error"
        assert "NOT PUBLISHED" in _text(payload)

    def test_60_percent_now_errors_where_it_used_to_be_ok(self):
        """Direct regression on the old `frac < 0.5` rule: 60% coverage
        returned `ok` before this change."""
        status, _ = ss.build_health_payload("open", 454, 757, 5, 303)
        assert status == "error"

    def test_published_but_incomplete_is_partial_not_error(self):
        status, payload = ss.build_health_payload("open", 700, 757, 4, 57)  # 92.5%
        assert status == "partial"
        assert "NOT PUBLISHED" not in _text(payload)
        assert "alert tiers are incomplete" in _text(payload)

    def test_full_coverage_is_ok_and_carries_no_warning(self):
        status, payload = ss.build_health_payload("close", 757, 757, 3, 0)
        assert status == "ok"
        body = _text(payload)
        assert "NOT PUBLISHED" not in body
        assert "Warnings" not in body

    def test_published_flag_overrides_the_derived_value(self):
        """The heartbeat must report what HAPPENED. If a caller suppressed the
        post for its own reasons, the heartbeat may not claim it published.

        ⛑ TIGHTENED after Codex found this test PERMITTED a contradiction
        (Medium, 2026-09-07): it supplied 757/757 with published=False and
        asserted only on the words "NOT PUBLISHED", so a card claiming 100%
        coverage was "below the 80% floor" passed it clean. The reason is now
        asserted, not just the marker."""
        _, payload = ss.build_health_payload(
            "close", 757, 757, 3, 0, published=False,
            reason="the Slack POST did not succeed")
        body = _text(payload)
        assert "NOT PUBLISHED" in body
        assert "the Slack POST did not succeed" in body
        assert "below" not in body.split("NOT PUBLISHED")[1],             "a full-coverage run must not be told its coverage was below a floor"

    def test_a_full_coverage_run_is_never_told_its_coverage_was_low(self):
        """The plausible wrong value: 100% coverage, failed delivery."""
        _, payload = ss.build_health_payload(
            "close", 757, 757, 3, 0, published=False, reason="delivery failed")
        assert "80%" not in _text(payload)

    def test_a_below_floor_run_still_explains_the_floor_without_a_reason(self):
        """Default wording must survive when no reason is supplied."""
        _, payload = ss.build_health_payload("midday", 36, 757, 2, 721)
        body = _text(payload)
        assert "80%" in body and "5%" in body

    def test_status_is_still_derived_from_coverage_not_from_published(self):
        """`published=False` must not turn a fully-covered run into an `error`;
        the two axes are independent."""
        status, _ = ss.build_health_payload("close", 757, 757, 3, 0, published=False)
        assert status == "ok"


# ------------------------------------------------------------- digest banner
class TestDegradedBanner:
    def _banner(self, screened, total):
        blocks = ss.format_slack_message(
            [], "open", total, {"screened": screened, "skipped": total - screened},
            {}, set(),
        )["blocks"]
        return [b for b in blocks
                if b.get("type") == "section"
                and "DEGRADED RUN" in b.get("text", {}).get("text", "")]

    def test_banner_fires_in_the_published_but_incomplete_band(self):
        """92.5% publishes, and it must say it is incomplete. Under the old
        0.5 threshold this run rendered with no banner at all."""
        assert len(self._banner(700, 757)) == 1

    def test_banner_absent_on_a_full_screen(self):
        assert self._banner(757, 757) == []

    def test_banner_names_the_real_percentage(self):
        banner = self._banner(700, 757)[0]["text"]["text"]
        assert "700/757" in banner
        assert "92%" in banner


# -------------------------------------------------------------- the gate itself
class TestEnforcePublishGate:
    """Behavioural tests on the gate. It was inline in `main()` first, and
    mutation-testing proved that untestable: `if not publishable:` -> `if False:`
    disabled the whole gate with all 301 tests still green.
    """

    def test_returns_true_on_a_full_screen(self, capsys):
        assert ss.enforce_publish_gate(
            "close", {"screened": 757, "skipped": 0}, 757, 3) is True

    def test_returns_false_on_the_real_2026_09_07_midday_run(self):
        assert ss.enforce_publish_gate(
            "midday", {"screened": 36, "skipped": 721}, 757, 2) is False

    def test_returns_false_at_60_percent(self):
        """The plausible wrong value, not garbage: 60% published clean under the
        old 0.5 threshold."""
        assert ss.enforce_publish_gate(
            "open", {"screened": 454, "skipped": 303}, 757, 5) is False

    def test_refusal_posts_the_error_heartbeat_marked_unpublished(self, monkeypatch):
        """Silence on both channels is indistinguishable from a lane that never
        ran, so the refusal path must still say something — and it must say it
        did NOT publish."""
        seen = {}

        def _spy(mode, stats, total, n_alerts, published=None, reason=None):
            seen.update(mode=mode, total=total, published=published, reason=reason)
            seen["status"] = ss.build_health_payload(
                mode, stats.get("screened", 0), total, n_alerts,
                stats.get("skipped", 0), published=published, reason=reason)[0]

        monkeypatch.setattr(ss, "post_health_heartbeat", _spy)
        ss.enforce_publish_gate("midday", {"screened": 36, "skipped": 721}, 757, 2)
        assert seen["published"] is False
        assert seen["status"] == "error"

    def test_a_publishable_run_does_not_post_the_heartbeat_itself(self, monkeypatch):
        """main() posts the heartbeat on the happy path, after publishing. If the
        gate posted one too, every good run would emit two heartbeats."""
        calls = []
        monkeypatch.setattr(ss, "post_health_heartbeat",
                            lambda *a, **k: calls.append(k))
        ss.enforce_publish_gate("close", {"screened": 757, "skipped": 0}, 757, 3)
        assert calls == []

    def test_refusal_names_the_floor_and_the_suppressed_artifacts(self, capsys):
        ss.enforce_publish_gate("midday", {"screened": 36, "skipped": 721}, 757, 2)
        out = capsys.readouterr().out
        assert "80%" in out
        assert "digest" in out.lower()
        assert "return-map" in out.lower()


# ------------------------------------------------------------------ the wiring
class TestGateIsWiredIntoMain:
    """One structural claim remains: that `main()` honours a False. Labelled as
    structural — it reads source, not behaviour — but it fails the moment
    someone moves a write above the gate or drops the early return."""

    @staticmethod
    def _main_src():
        import inspect as _i
        return _i.getsource(ss.main)

    def test_gate_precedes_the_slack_digest(self):
        src = self._main_src()
        assert "enforce_publish_gate" in src, "main() does not consult the gate at all"
        assert src.index("enforce_publish_gate") < src.index("send_slack(payload)")

    def test_gate_precedes_the_return_map_rewrite(self):
        src = self._main_src()
        assert src.index("enforce_publish_gate") < src.index("return_map.write_html")

    def test_main_returns_on_a_refusal(self):
        """A gate whose False is ignored is not a gate. Matches a bare `return`
        STATEMENT — an earlier version asserted `"return" in head` and passed
        when the statement was mutated to `pass`, because the gate's own log
        line contains the words "the return-map rewrite"."""
        import re as _re
        src = self._main_src()
        gate = src[src.index("if not enforce_publish_gate"):]
        head = gate[:gate.index("# Send to Slack")]
        assert _re.search(r"^\s+return\s*$", head, _re.M), \
            "main() must RETURN when the gate refuses, not log and continue"


# ------------------------------------------------- the return-map denominator
class TestReturnMapCoverage:
    """Codex, High, 2026-09-07: one constant cannot gate two jobs whose
    denominators differ. The screen floor counts ~757 watchlist tickers; the
    return map is built from a ~43-symbol ETF/macro universe.
    """
    ETFS = {f"E{i}" for i in range(43)}

    @staticmethod
    def _returns(tickers):
        return [{"ticker": t} for t in tickers]

    def test_the_exact_scenario_the_screen_gate_misses(self):
        """Yahoo returns 714 of 757 watchlist names (94.3%, screen gate PASSES)
        but zero of the 43 return-map symbols. The map must not be rewritten."""
        assert ss.coverage_is_publishable(714, 757) is True
        assert ss.return_map_is_publishable([], self.ETFS) is False

    def test_a_full_etf_set_publishes(self):
        assert ss.return_map_is_publishable(self._returns(self.ETFS), self.ETFS) is True

    def test_half_the_etf_set_is_refused(self):
        half = list(self.ETFS)[:21]
        assert ss.return_map_is_publishable(self._returns(half), self.ETFS) is False

    def test_just_below_the_floor_is_refused(self):
        some = list(self.ETFS)[:34]           # 79.1%
        assert ss.return_map_is_publishable(self._returns(some), self.ETFS) is False

    def test_at_the_floor_publishes(self):
        some = list(self.ETFS)[:35]           # 81.4%
        assert ss.return_map_is_publishable(self._returns(some), self.ETFS) is True

    def test_an_empty_expected_set_never_publishes(self):
        """0/0 must not read as full coverage — same rule as screen_coverage."""
        assert ss.return_map_is_publishable([], set()) is False

    def test_extra_symbols_outside_the_set_do_not_inflate_coverage(self):
        """Counting the raw length of etf_returns would let 43 unrelated
        tickers satisfy the gate. It intersects with the expected set."""
        noise = self._returns({f"X{i}" for i in range(43)})
        assert ss.return_map_is_publishable(noise, self.ETFS) is False

    def test_rows_without_a_ticker_are_ignored(self):
        assert ss.return_map_is_publishable([{}, {"ticker": None}], self.ETFS) is False


# ------------------------------------------------------- delivery is observed
class TestSendSlackReportsDelivery:
    """Codex, High, 2026-09-07: `published` was a constant, so a failed POST at
    full coverage emitted `ok` with no NOT PUBLISHED line and exited zero."""

    def test_missing_webhook_returns_false(self, monkeypatch, capsys):
        monkeypatch.delenv("SLACK_WEBHOOK", raising=False)
        assert ss.send_slack({"text": "x"}) is False

    def test_http_error_returns_false(self, monkeypatch):
        monkeypatch.setenv("SLACK_WEBHOOK", "https://example.invalid/hook")

        def _boom(*a, **k):
            raise ss.requests.RequestException("429 Too Many Requests")

        monkeypatch.setattr(ss.requests, "post", _boom)
        assert ss.send_slack({"text": "x"}) is False

    def test_success_returns_true(self, monkeypatch):
        monkeypatch.setenv("SLACK_WEBHOOK", "https://example.invalid/hook")

        class _Resp:
            def raise_for_status(self):
                return None

        monkeypatch.setattr(ss.requests, "post", lambda *a, **k: _Resp())
        assert ss.send_slack({"text": "x"}) is True

    def test_main_passes_the_observed_result_not_a_constant(self):
        """Structural: `published=True` as a literal is the defect."""
        import inspect
        src = inspect.getsource(ss.main)
        assert "delivered = send_slack(payload)" in src
        assert "published=True" not in src,             "main() must pass the observed delivery result, never a constant"


# ------------------------------------- the SECOND return-map input (round 2)
class TestReturnMapPeriodCoverage:
    """Codex round 2, High: a defect in the round-1 fix. `assemble_snapshot`
    takes TWO inputs — `etf_returns` from the 400-day screen pull, and
    `etf_period_returns` from a SEPARATE ~800-day download (the screen window
    cannot reach the year-before-last's close). Round 1 gated only the first, so
    a throttled period fetch rewrote the map with every Prior Year and YTD
    column blank. Codex reproduced it: gate=True, assets=43, missing_prior=43,
    missing_ytd=43.

    Fixing the one-constant-two-jobs class once did not exempt the fix from it.
    """
    ETFS = {f"E{i}" for i in range(43)}

    @staticmethod
    def _rows(tickers):
        return [{"ticker": t} for t in tickers]

    def _full_period(self):
        return {t: {"ytd_return_pct": 1.0} for t in self.ETFS}

    def test_the_exact_round_2_repro(self):
        """All 43 in the screen pull, period pull empty."""
        assert ss.return_map_is_publishable(
            self._rows(self.ETFS), self.ETFS, {}) is False

    def test_both_inputs_full_publishes(self):
        assert ss.return_map_is_publishable(
            self._rows(self.ETFS), self.ETFS, self._full_period()) is True

    def test_thin_period_coverage_is_refused(self):
        thin = {t: {} for t in list(self.ETFS)[:21]}     # 48.8%
        assert ss.return_map_is_publishable(
            self._rows(self.ETFS), self.ETFS, thin) is False

    def test_just_below_the_floor_on_the_period_pull_is_refused(self):
        near = {t: {} for t in list(self.ETFS)[:34]}     # 79.1%
        assert ss.return_map_is_publishable(
            self._rows(self.ETFS), self.ETFS, near) is False

    def test_at_the_floor_on_the_period_pull_publishes(self):
        at = {t: {} for t in list(self.ETFS)[:35]}       # 81.4%
        assert ss.return_map_is_publishable(
            self._rows(self.ETFS), self.ETFS, at) is True

    def test_period_keys_outside_the_expected_set_do_not_inflate_coverage(self):
        noise = {f"X{i}": {} for i in range(43)}
        assert ss.return_map_is_publishable(
            self._rows(self.ETFS), self.ETFS, noise) is False

    def test_omitting_the_period_argument_skips_that_half(self):
        """None means 'not supplied' and must stay back-compatible; an EMPTY
        dict is a real measurement of zero and must NOT."""
        assert ss.return_map_is_publishable(self._rows(self.ETFS), self.ETFS) is True
        assert ss.return_map_is_publishable(
            self._rows(self.ETFS), self.ETFS, None) is True
        assert ss.return_map_is_publishable(
            self._rows(self.ETFS), self.ETFS, {}) is False

    def test_main_passes_the_period_returns_to_the_gate(self):
        import inspect
        src = inspect.getsource(ss.main)
        assert "return_map_is_publishable(etf_returns, etf_set, etf_period_returns)" in src,             "main() must gate on BOTH return-map inputs"


# --------------------------------------- the runner cannot be left silent
class TestLocalRunnerCannotBeInstalledSilent:
    """Codex round 2, High: the round-1 fix left -StatusWebhook optional AND
    made an idempotent re-run DELETE an existing one, because the script
    rewrites .env every time. Load-with-fallback feeding write-everything, on
    the single setting whose absence causes the silence being fixed.

    PowerShell is not executed here; these read the script as text, which is a
    weaker claim than running it and is labelled as such.
    """
    import pathlib
    SRC = (pathlib.Path(__file__).resolve().parent.parent
           / "scripts" / "setup_local_runner.ps1").read_text(encoding="ascii")

    def test_it_refuses_to_continue_without_a_status_webhook(self):
        assert 'if (-not $StatusWebhook) {' in self.SRC
        gate = self.SRC.split('if (-not $StatusWebhook) {')[-1]
        head = gate[:gate.index('}')]
        assert "Write-Error" in head and "exit 1" in head,             "a warning is not a mechanism; it must refuse to register the tasks"

    def test_an_existing_value_is_carried_forward_before_the_rewrite(self):
        """The clobber: .env is rewritten every run, so the read must happen
        BEFORE the write and must feed it."""
        assert "SLACK_STATUS_REPORTS_WEBHOOK" in self.SRC
        assert "$ExistingStatus" in self.SRC
        assert self.SRC.index("$ExistingStatus = ''") < self.SRC.index("$envLines = @(")
        assert self.SRC.index("$StatusWebhook = $ExistingStatus") < self.SRC.index("$envLines = @(")

    def test_the_status_webhook_is_always_written(self):
        """It used to be appended conditionally, so .env could be written
        without it. By the time we reach the write it is guaranteed non-empty."""
        block = self.SRC.split("$envLines = @(")[1]
        block = block[:block.index(")")]
        assert "SLACK_STATUS_REPORTS_WEBHOOK=$StatusWebhook" in block
        assert "if (" not in block, "the write must be unconditional"

    def test_the_documented_setup_command_passes_it(self):
        import pathlib
        readme = (pathlib.Path(__file__).resolve().parent.parent
                  / "README.md").read_text(encoding="utf-8")
        setup = readme.split("setup_local_runner.ps1")[1][:400]
        assert "-StatusWebhook" in setup,             "the documented command must not install the silent configuration"
