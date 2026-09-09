"""Tests for the cached-open screening path (`screen_open_cached`).

This function had **no tests at all**, and on 2026-07-31 and 2026-08-03 it died
in production with `NameError: name 'mkey' is not defined` — before the
in-process heartbeat, so `#status-reports` got a bare "crashed" line naming no
cause and no counters.

Two things made it survive:

1. The undefined name is inside the **1σ branch**, which only runs when a
   ticker's |z| lands between `ONE_SIGMA_THRESHOLD` and `SIGMA_THRESHOLD`. Any
   day where nothing sits in that band, the Open cycle passes. Intermittent, not
   daily — which is why it read as flaky rather than broken.
2. Only the Open cycle uses this function. Midday and Close go through
   `screen_full`, which has its own (correct) `meta_key`, so two of three cycles
   posted `ok` on the same days.

So the tests below deliberately drive a ticker INTO the 1σ band.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import sigma_screener as ss  # noqa: E402


def _cache(ticker: str, mu: float = 0.0, sigma: float = 0.01) -> dict:
    return {"tickers": {ticker: {"mu": mu, "sigma": sigma,
                                 "high_52w": 120.0, "low_52w": 80.0}}}


def _prices(ticker: str, prev_close: float, today_open: float) -> dict:
    return {ticker: {"prev_close": prev_close, "today_open": today_open}}


def _one_sigma_move(prev_close: float = 100.0) -> float:
    """A price that lands |z| squarely between the 1σ and 2σ thresholds.

    With mu=0 and sigma=0.01, z == return/0.01, so a 1.5% move gives z=1.5.
    """
    mid = (ss.ONE_SIGMA_THRESHOLD + ss.SIGMA_THRESHOLD) / 2
    return prev_close * (1 + mid * 0.01)


def test_one_sigma_branch_does_not_raise():
    """The regression. Before the fix this raised NameError and killed the run."""
    t = "AAA"
    with mock.patch.object(ss, "download_todays_prices",
                           return_value=_prices(t, 100.0, _one_sigma_move())):
        alerts, stats, _etf = ss.screen_open_cached(
            [t], _cache(t), metadata={t: {"name": "Alpha", "core": "Y"}},
            portfolio_set={t}, researching_set=set(),
        )
    assert stats["screened"] == 1
    assert [a["ticker"] for a in alerts] == [t]
    assert alerts[0]["tier"] == "1sigma"


def test_position_membership_resolves_under_v4_raw_keys():
    """CM schema v4 keys metadata AND the position lists by the RAW ticker, so a
    foreign line must match on its dotted symbol."""
    t = "DIA.MI"
    with mock.patch.object(ss, "download_todays_prices",
                           return_value=_prices(t, 100.0, _one_sigma_move())):
        alerts, _stats, _etf = ss.screen_open_cached(
            [t], _cache(t), metadata={t: {"name": "DiaSorin"}},
            portfolio_set={t}, researching_set=set(),
        )
    assert alerts and alerts[0]["in_portfolio"] is True


def test_position_membership_resolves_under_v3_stripped_keys():
    """...and on the other side of a CM republish, where both are keyed by the
    suffix-stripped base. The key used for membership must be the key that
    actually RESOLVED the metadata, not the raw ticker — otherwise a name is
    silently 'not in the portfolio' for a cycle."""
    t = "GETIB.SS"
    with mock.patch.object(ss, "download_todays_prices",
                           return_value=_prices(t, 100.0, _one_sigma_move())):
        alerts, _stats, _etf = ss.screen_open_cached(
            [t], _cache(t), metadata={"GETIB": {"name": "Getinge"}},
            portfolio_set={"GETIB"}, researching_set=set(),
        )
    assert alerts and alerts[0]["in_portfolio"] is True
    assert alerts[0]["name"] == "Getinge"


def test_unknown_metadata_still_screens_and_reports_not_held():
    """A ticker CM does not know must not crash and must not claim membership."""
    t = "ZZZ"
    with mock.patch.object(ss, "download_todays_prices",
                           return_value=_prices(t, 100.0, _one_sigma_move())):
        alerts, stats, _etf = ss.screen_open_cached(
            [t], _cache(t), metadata={}, portfolio_set=set(), researching_set=set(),
        )
    assert stats["screened"] == 1
    # Not core, not held → not 1σ-eligible, so no alert; and no exception.
    assert alerts == []


def test_two_sigma_fires_without_touching_the_eligibility_gate():
    """2σ alerts bypass the 1σ gate entirely — that path was always fine, and
    the test pins that the fix did not make 2σ depend on membership."""
    t = "BBB"
    big = 100.0 * (1 + (ss.SIGMA_THRESHOLD + 1) * 0.01)
    with mock.patch.object(ss, "download_todays_prices",
                           return_value=_prices(t, 100.0, big)):
        alerts, _stats, _etf = ss.screen_open_cached(
            [t], _cache(t), metadata={}, portfolio_set=set(), researching_set=set(),
        )
    assert alerts and alerts[0]["tier"] == "2sigma"
    assert alerts[0]["in_portfolio"] is False


@pytest.mark.parametrize("field,setname", [
    ("in_researching", "researching_set"),
    ("in_following_for_interest", "following_set"),
    ("in_ready_to_buy", "ready_to_buy_set"),
    ("in_ready_to_short", "ready_to_short_set"),
])
def test_every_position_set_is_wired(field, setname):
    """All five membership flags used `mkey`; a fix that only repaired the one
    in the eligibility call would leave four silently wrong."""
    t = "CCC"
    kwargs = {"portfolio_set": set(), "researching_set": set(), setname: {t}}
    with mock.patch.object(ss, "download_todays_prices",
                           return_value=_prices(t, 100.0, _one_sigma_move())):
        alerts, _stats, _etf = ss.screen_open_cached(
            [t], _cache(t), metadata={t: {"name": "Gamma"}}, **kwargs)
    assert alerts, f"{setname} membership did not make it 1σ-eligible"
    assert alerts[0][field] is True


def test_resolve_metadata_returns_the_matching_key():
    """The helper the fix rests on: entry AND the key it matched, so callers
    cannot use one without the other."""
    md = {"GETIB": {"name": "Getinge"}}
    entry, key = ss.resolve_metadata(md, "GETIB.SS", ss.foreign_collision_bases(["GETIB.SS"]))
    assert entry == {"name": "Getinge"} and key == "GETIB"

    entry, key = ss.resolve_metadata({"DIA.MI": {"n": 1}}, "DIA.MI", set())
    assert entry == {"n": 1} and key == "DIA.MI"

    # Nothing matches → no entry, and the key falls back to the raw ticker so
    # membership checks still ask a sensible question instead of crashing.
    entry, key = ss.resolve_metadata({}, "NOPE", set())
    assert entry is None and key == "NOPE"


def test_lookup_metadata_still_returns_just_the_entry():
    """Back-compat: the existing callers of lookup_metadata are unchanged."""
    assert ss.lookup_metadata({"AAA": {"n": 1}}, "AAA") == {"n": 1}
    assert ss.lookup_metadata({}, "AAA") is None


# ── the CI health backstop payload ───────────────────────────────────────────

def test_health_payload_lifts_the_exception_line():
    """The 2026-08-03 log, in miniature. The Slack card must NAME the cause."""
    import ci_health_payload as chp
    log = (
        "[WARN] No today bar for CSU in cached-open batch, skipping\n"
        "Traceback (most recent call last):\n"
        '  File "scripts/sigma_screener.py", line 2627, in main\n'
        "    alerts, stats, etf_returns = screen_open_cached(\n"
        "NameError: name 'mkey' is not defined\n"
    )
    assert chp.extract_reason(log) == "NameError: name 'mkey' is not defined"
    payload = chp.build_payload(chp.extract_reason(log), "https://example/run/1")
    text = payload["blocks"][0]["text"]["text"]
    assert "NameError: name 'mkey' is not defined" in text
    assert "https://example/run/1" in text


def test_health_payload_prefers_the_LAST_exception_in_a_chain():
    """'During handling of the above exception...' — the final one killed it."""
    import ci_health_payload as chp
    log = ("KeyError: 'AAA'\n"
           "During handling of the above exception, another exception occurred:\n"
           "RuntimeError: giving up\n")
    assert chp.extract_reason(log) == "RuntimeError: giving up"


def test_health_payload_falls_back_to_the_last_line_for_non_python_failures():
    """An OOM kill or a shell error has no exception line but still says
    something more useful than nothing."""
    import ci_health_payload as chp
    assert chp.extract_reason("starting\n/bin/bash: line 3: killed\n") == \
        "/bin/bash: line 3: killed"


def test_health_payload_is_valid_json_with_no_log_at_all():
    """The backstop runs INSIDE a failure handler. If it raises, a diagnosable
    failure becomes silence — the exact outcome it exists to prevent."""
    import json
    import ci_health_payload as chp
    doc = json.loads(json.dumps(chp.build_payload("", "")))
    assert doc["blocks"][0]["text"]["text"].startswith(":x: *sigma-alert - error*")


def test_health_payload_truncates_a_giant_traceback_line():
    import ci_health_payload as chp
    text = chp.build_payload("E" * 5000, "")["blocks"][0]["text"]["text"]
    assert len(text) < 600


def test_health_payload_escapes_rather_than_breaking_json():
    """A traceback containing quotes and backslashes is why this is built in
    python instead of hand-quoted into a curl --data string."""
    import json
    import ci_health_payload as chp
    nasty = 'ValueError: bad "quote" and \\ backslash and \n newline'
    doc = json.loads(json.dumps(chp.build_payload(nasty, "")))
    assert 'bad "quote"' in doc["blocks"][0]["text"]["text"]


def test_every_cycle_workflow_keeps_pipefail_with_the_tee():
    """Without `set -eo pipefail` the pipe's status is tee's (always 0), so a
    crashed screener would report SUCCESS and `if: failure()` would never fire —
    silencing the very backstop this change improves. Pinned because it is one
    deleted word away and nothing else would notice."""
    root = Path(__file__).resolve().parent.parent / ".github" / "workflows"
    for name in ("sigma-open.yml", "sigma-midday.yml", "sigma-close.yml"):
        text = (root / name).read_text(encoding="utf-8")
        assert "tee /tmp/sigma_run.log" in text, name
        assert "set -eo pipefail" in text, f"{name} pipes to tee without pipefail"
        assert "ci_health_payload.py" in text, name


def test_cached_path_carries_commercial_so_the_subcategory_can_match():
    """The Commercial Biopharma flag must survive the path Open ACTUALLY takes.

    ⛑ Codex High #2, 2026-09-08. `screen_full` copied `commercial` into its alert
    dict and this function did not, so on any morning with a fresh cache -- the
    normal case -- a commercial biotech's alert was built without the field. The
    subcategory predicate then matched nothing, and a ticker matching no
    subcategory is DROPPED at render, so the name went silently missing from the
    digest it had just been added to.

    Every existing test of that predicate passed throughout, because they all
    construct an already-enriched alert dict by hand and never run either
    screener. Both halves were tested; the seam between them was not.
    """
    t = "BIO"
    with mock.patch.object(ss, "download_todays_prices",
                           return_value=_prices(t, 100.0, 105.0)):
        alerts, _stats, _etf = ss.screen_open_cached(
            [t], _cache(t),
            metadata={t: {"name": "Biocorp", "sector": "Biopharma",
                          "subsector": "Biotech", "commercial": "Y"}},
            portfolio_set=set(), researching_set=set(),
        )
    assert len(alerts) == 1, "a 5% move on sigma=0.01 must alert"
    assert alerts[0].get("commercial") == "Y"


def test_both_screeners_emit_the_SAME_alert_fields():
    """A field added to one alert constructor and not the other is invisible.

    This is the structural version of the test above: it fails for ANY future
    field that lands on one path only, rather than for `commercial` specifically.
    Open reads the cached path and Midday/Close read `screen_full`, so a
    one-sided field means a section that works at 16:00 and silently does not at
    09:30 -- which is indistinguishable from a quiet market.
    """
    import ast

    src = ast.parse((Path(__file__).resolve().parent.parent
                     / "scripts" / "sigma_screener.py").read_text(encoding="utf-8"))
    # The full path builds its alert inside the per-ticker worker
    # `_process_ticker_full`, not in `screen_full` itself.
    fns = {n.name: n for n in ast.walk(src)
           if isinstance(n, ast.FunctionDef) and n.name in
           ("screen_open_cached", "_process_ticker_full")}
    assert set(fns) == {"screen_open_cached", "_process_ticker_full"}

    def alert_keys(fn):
        """Keys of the largest dict literal holding 'ticker' and 'z_score'.

        Both constructors are the only such dicts in their function; the ETF
        stats dict carries 'ticker' but no 'tier'/'direction' and is smaller.
        """
        best = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if {"ticker", "z_score", "direction"} <= keys and len(keys) > len(best):
                best = keys
        return best

    open_keys = alert_keys(fns["screen_open_cached"])
    full_keys = alert_keys(fns["_process_ticker_full"])
    assert open_keys, "could not locate the cached-path alert dict"
    assert full_keys, "could not locate the full-path alert dict"

    only_full = full_keys - open_keys
    only_open = open_keys - full_keys
    assert not only_full, f"fields only on the full path (Open would drop them): {sorted(only_full)}"
    assert not only_open, f"fields only on the cached path: {sorted(only_open)}"
