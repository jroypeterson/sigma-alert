"""M1 (Fable review 2026-09-29): the weekly skip report must be postable at the
size of the data it exists to surface.

On 2026-09-12, 09-19 and 09-26 the skip log carried ~755 skips per day (the
duplicate-close overwrite, board #327). Every chronic/unresolved chip went into
ONE section, Slack rejected it with a 400, and the report failed three Fridays
running with nothing saying so. Built from that recorded shape: 755 tickers
skipped on every run of a 5-run week.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import weekly_skip_report as wsr  # noqa: E402

SLACK_SECTION_LIMIT = 3000


def _week_of(n_tickers: int):
    days = ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25"]
    skipped = [{"ticker": f"T{i:03d}", "reason": "stale_bar"} for i in range(n_tickers)]
    return [{"date": d, "mode": "close", "skipped": list(skipped)} for d in days]


def _sections(payload):
    return [b["text"]["text"] for b in payload["blocks"]
            if b.get("type") == "section" and "text" in b]


def test_the_recorded_755_chip_week_fits_every_section():
    stats = wsr.compute_stats(_week_of(755), watchlist_size=756)
    assert len(stats["chronic"]) == 755 and len(stats["unresolved"]) == 755
    payload = wsr.format_slack_payload(stats, 756)
    for text in _sections(payload):
        assert len(text) < SLACK_SECTION_LIMIT, len(text)


def test_the_cap_says_how_many_it_left_out():
    stats = wsr.compute_stats(_week_of(755), watchlist_size=756)
    body = "\n".join(_sections(wsr.format_slack_payload(stats, 756)))
    assert f"and {755 - wsr.MAX_LISTED} more" in body


def test_a_normal_week_is_listed_in_full():
    stats = wsr.compute_stats(_week_of(12), watchlist_size=756)
    body = "\n".join(_sections(wsr.format_slack_payload(stats, 756)))
    assert "more" not in body
    assert all(f"`T{i:03d}`" in body for i in range(12))


def test_the_workflow_has_a_failure_backstop():
    wf = (Path(__file__).resolve().parent.parent / ".github" / "workflows"
          / "sigma-weekly-skip-report.yml").read_text(encoding="utf-8")
    assert "if: failure()" in wf
    assert wf.index("python scripts/weekly_skip_report.py") < wf.index("if: failure()")
