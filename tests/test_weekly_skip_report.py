"""Tests for the weekly skip report's suppression logic and the foreign-symbol
overrides that resolve chronic insufficient_history skips (added 2026-06-21)."""

import json
import sys
from pathlib import Path

# Import both modules under test from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import sigma_screener
import weekly_skip_report as wsr


# ---------------------------------------------------------------------------
# _YF_SYMBOL_OVERRIDES — the two wrong/foreign-symbol fixes
# ---------------------------------------------------------------------------

class TestForeignSymbolOverrides:
    def test_cvsg_maps_to_lse(self):
        # CVS Group plc trades on the LSE as CVSG.L, not bare CVSG.
        assert sigma_screener.to_yf_symbol("CVSG") == "CVSG.L"

    def test_sfzs_typo_maps_to_siegfried(self):
        # CM's "SFZS.SW" is a typo for Siegfried Holding's real SIX symbol SFZN.SW.
        assert sigma_screener.to_yf_symbol("SFZS.SW") == "SFZN.SW"

    def test_plain_us_ticker_unchanged(self):
        # Sanity: the new entries don't perturb ordinary US tickers.
        assert sigma_screener.to_yf_symbol("AAPL") == "AAPL"


# ---------------------------------------------------------------------------
# active_suppressions — expiry handling
# ---------------------------------------------------------------------------

class TestActiveSuppressions:
    def test_dated_entry_active_before_expiry(self):
        supp = {"GMRS": "2026-07-20"}
        assert wsr.active_suppressions(supp, "2026-06-21") == {"GMRS"}

    def test_dated_entry_active_on_expiry_day(self):
        # `until` is inclusive.
        supp = {"GMRS": "2026-07-20"}
        assert wsr.active_suppressions(supp, "2026-07-20") == {"GMRS"}

    def test_dated_entry_expired_after(self):
        supp = {"GMRS": "2026-07-20"}
        assert wsr.active_suppressions(supp, "2026-07-21") == set()

    def test_indefinite_entry_always_active(self):
        supp = {"FOO": None}
        assert wsr.active_suppressions(supp, "2099-01-01") == {"FOO"}

    def test_malformed_until_treated_as_indefinite(self):
        # A garbled date must NOT silently expire the suppression.
        supp = {"BAR": "notadate"}
        assert wsr.active_suppressions(supp, "2026-06-21") == {"BAR"}


# ---------------------------------------------------------------------------
# compute_stats — suppression removes from chronic/unresolved, surfaces in hits
# ---------------------------------------------------------------------------

def _run(date_str, *tickers, reason="insufficient_history"):
    return {
        "date": date_str,
        "mode": "close",
        "skipped": [{"ticker": t, "reason": reason} for t in tickers],
    }


class TestChronicThreshold:
    """Chronic used to require `n == run_count` — skipped in EVERY run.

    Live failure 2026-07-24: the `stale_bar` guard shipped mid-window on
    07-18, so 28 tickers skipped 5 of the window's 6 runs and the digest
    reported ':red_circle: Chronic skips: none' while a fifth of the
    watchlist went unscreened every single day. Any run a ticker slips
    through — a new guard landing, a partial run, one lucky fetch — reset an
    otherwise-permanent skip to 'not chronic'. Chronic is now a ratio.
    """

    def test_five_of_six_runs_is_chronic(self):
        # The exact 2026-07-24 shape: clean on the first run, skipped after.
        runs = [_run("2026-07-17")] + [
            _run(d, "LONN.CH", reason="stale_bar")
            for d in ("2026-07-20", "2026-07-21", "2026-07-22",
                      "2026-07-23", "2026-07-24")
        ]
        stats = wsr.compute_stats(runs, watchlist_size=712)
        chronic = {c["ticker"]: c for c in stats["chronic"]}
        assert "LONN.CH" in chronic, (
            "a ticker unscreened on 5 of 6 runs is chronic by any reading"
        )
        assert chronic["LONN.CH"]["count"] == 5
        assert chronic["LONN.CH"]["run_count"] == 6

    def test_occasional_skip_is_not_chronic(self):
        runs = [_run("2026-07-20", "FOO")] + [
            _run(d) for d in ("2026-07-21", "2026-07-22",
                              "2026-07-23", "2026-07-24")
        ]
        stats = wsr.compute_stats(runs, watchlist_size=712)
        assert {c["ticker"] for c in stats["chronic"]} == set()

    def test_single_run_window_yields_no_chronic(self):
        """One run is never enough evidence to call something chronic."""
        runs = [_run("2026-07-24", "FOO")]
        stats = wsr.compute_stats(runs, watchlist_size=712)
        assert stats["chronic"] == []

    def test_every_run_still_chronic(self):
        runs = [_run(d, "FOO") for d in ("2026-07-21", "2026-07-22",
                                         "2026-07-23", "2026-07-24")]
        stats = wsr.compute_stats(runs, watchlist_size=712)
        assert {c["ticker"] for c in stats["chronic"]} == {"FOO"}

    def test_suppression_still_wins_over_the_ratio(self):
        runs = [_run("2026-07-17")] + [
            _run(d, "GMRS", "ZZZZ", reason="stale_bar")
            for d in ("2026-07-20", "2026-07-21", "2026-07-22",
                      "2026-07-23", "2026-07-24")
        ]
        stats = wsr.compute_stats(runs, watchlist_size=712, suppressed={"GMRS"})
        tickers = {c["ticker"] for c in stats["chronic"]}
        assert "GMRS" not in tickers
        assert "ZZZZ" in tickers

    def test_chronic_line_reports_the_ratio(self):
        runs = [_run("2026-07-17")] + [
            _run(d, "LONN.CH", reason="stale_bar")
            for d in ("2026-07-20", "2026-07-21", "2026-07-22",
                      "2026-07-23", "2026-07-24")
        ]
        stats = wsr.compute_stats(runs, watchlist_size=712)
        payload = wsr.format_slack_payload(stats, 712)
        text = json.dumps(payload)
        assert "5/6 runs" in text, text


class TestComputeStatsSuppression:
    def test_suppressed_ticker_excluded_from_chronic(self):
        runs = [
            _run("2026-06-19", "GMRS", "ZZZZ"),
            _run("2026-06-20", "GMRS", "ZZZZ"),
        ]
        stats = wsr.compute_stats(runs, watchlist_size=700, suppressed={"GMRS"})
        chronic_tickers = {c["ticker"] for c in stats["chronic"]}
        assert "GMRS" not in chronic_tickers
        assert "ZZZZ" in chronic_tickers  # non-suppressed chronic still shown

    def test_suppressed_ticker_excluded_from_unresolved(self):
        runs = [_run("2026-06-20", "GMRS", "ZZZZ")]
        stats = wsr.compute_stats(runs, watchlist_size=700, suppressed={"GMRS"})
        unresolved_tickers = {u["ticker"] for u in stats["unresolved"]}
        assert "GMRS" not in unresolved_tickers
        assert "ZZZZ" in unresolved_tickers

    def test_suppressed_hit_is_surfaced(self):
        runs = [_run("2026-06-20", "GMRS")]
        stats = wsr.compute_stats(runs, watchlist_size=700, suppressed={"GMRS"})
        hits = {h["ticker"] for h in stats["suppressed_hits"]}
        assert hits == {"GMRS"}

    def test_suppressed_ticker_absent_from_window_not_listed(self):
        # A suppressed ticker that never skipped this window isn't surfaced.
        runs = [_run("2026-06-20", "ZZZZ")]
        stats = wsr.compute_stats(runs, watchlist_size=700, suppressed={"GMRS"})
        assert stats["suppressed_hits"] == []

    def test_no_suppression_is_backward_compatible(self):
        runs = [_run("2026-06-20", "ZZZZ")]
        stats = wsr.compute_stats(runs, watchlist_size=700)
        assert {u["ticker"] for u in stats["unresolved"]} == {"ZZZZ"}
        assert stats["suppressed_hits"] == []

    def test_reason_breakdown_still_counts_suppressed(self):
        # Suppression hides the operator nag but must not falsify the event count.
        runs = [_run("2026-06-20", "GMRS", "ZZZZ")]
        stats = wsr.compute_stats(runs, watchlist_size=700, suppressed={"GMRS"})
        assert stats["reason_breakdown"]["insufficient_history"] == 2
        assert stats["total_skip_events"] == 2


# ---------------------------------------------------------------------------
# format_slack_payload — suppressed line renders and is well-formed
# ---------------------------------------------------------------------------

class TestSlackPayloadSuppression:
    def test_suppressed_context_block_present(self):
        runs = [_run("2026-06-20", "GMRS")]
        stats = wsr.compute_stats(runs, watchlist_size=700, suppressed={"GMRS"})
        payload = wsr.format_slack_payload(stats, watchlist_size=700)
        # context blocks use elements[] (the Slack invalid_blocks gotcha).
        texts = [
            el.get("text", "")
            for b in payload["blocks"]
            if b.get("type") == "context"
            for el in b.get("elements", [])
        ]
        assert any("Suppressed" in t and "GMRS" in t for t in texts)
