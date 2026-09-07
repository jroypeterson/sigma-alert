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
        post for its own reasons, the heartbeat may not claim it published."""
        _, payload = ss.build_health_payload("close", 757, 757, 3, 0, published=False)
        assert "NOT PUBLISHED" in _text(payload)

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

        def _spy(mode, stats, total, n_alerts, published=None):
            seen.update(mode=mode, total=total, published=published)
            seen["status"] = ss.build_health_payload(
                mode, stats.get("screened", 0), total, n_alerts,
                stats.get("skipped", 0), published=published)[0]

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
