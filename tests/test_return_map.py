"""Tests for scripts/return_map.py — the interactive periodic-table-of-returns.

Pure rendering logic; no network, no market-data fetch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import return_map as rm  # noqa: E402


def _etf():
    return [
        {"ticker": "SPYM", "name": "S&P 500", "return_pct": 0.4},
        {"ticker": "^RUT", "name": "Russell 2000", "return_pct": -0.3},
        {"ticker": "^TNX", "name": "10Y", "return_pct": 1.2},  # yield — excluded
        {"ticker": "XLK", "name": "Technology", "return_pct": 0.9},
        {"ticker": "GLD", "name": "Gold", "return_pct": 0.6},
    ]


def _pm():
    return {
        "SPYM": {"ytd_return_pct": 6.1, "prior_year_return_pct": 24.5, "prior_year_label": "2025"},
        "^RUT": {"ytd_return_pct": 2.2, "prior_year_return_pct": 11.4, "prior_year_label": "2025"},
        "GLD": {"ytd_return_pct": 18.9, "prior_year_return_pct": 26.1, "prior_year_label": "2025"},
    }


def _snap():
    return rm.assemble_snapshot(
        _etf(), _pm(),
        index_set={"SPYM", "^RUT"}, sector_set={"XLK"},
        commodity_set={"GLD"}, macro_set={"^TNX"},
        macro_style={"^TNX": "yield"}, mode="close", ref_date="2026-06-26",
    )


def test_yield_style_macro_excluded():
    snap = _snap()
    tickers = {a["ticker"] for a in snap["assets"]}
    assert "^TNX" not in tickers  # a % return on a yield level is misleading
    assert {"SPYM", "^RUT", "XLK", "GLD"} == tickers


def test_groups_assigned():
    snap = _snap()
    by_t = {a["ticker"]: a for a in snap["assets"]}
    assert by_t["SPYM"]["group"] == "US Indices"
    assert by_t["^RUT"]["group"] == "US Indices"
    assert by_t["XLK"]["group"] == "Sectors"
    assert by_t["GLD"]["group"] == "Commodities"


def test_period_fields_carried_and_missing_is_none():
    snap = _snap()
    by_t = {a["ticker"]: a for a in snap["assets"]}
    assert by_t["SPYM"]["ytd_pct"] == 6.1
    assert by_t["SPYM"]["prior_year_pct"] == 24.5
    # XLK has no period entry -> None, not a crash / NaN.
    assert by_t["XLK"]["ytd_pct"] is None
    assert by_t["XLK"]["prior_year_pct"] is None


def test_nan_coerced_to_none():
    snap = rm.assemble_snapshot(
        [{"ticker": "SPYM", "name": "S&P 500", "return_pct": float("nan")}],
        {}, index_set={"SPYM"},
    )
    assert snap["assets"][0]["today_pct"] is None


def test_build_html_contains_columns_and_tickers():
    html = rm.build_html(_snap())
    assert "<!DOCTYPE html>" in html
    # Three period columns.
    assert "Prior Year" in html or "2025" in html
    assert "YTD" in html
    assert "Today" in html
    # Tiles present.
    for tk in ("SPYM", "^RUT", "XLK", "GLD"):
        assert tk in html
    # Self-contained — no external script/style sources.
    assert "http://" not in html and "https://" not in html
    assert "<script src" not in html


def test_build_html_empty_snapshot_is_graceful():
    html = rm.build_html({"assets": [], "generated_at": "", "mode": "", "ref_date": ""})
    assert "<!DOCTYPE html>" in html
    assert "No returns in the snapshot" in html


def test_roundtrip_write_load(tmp_path):
    snap = _snap()
    p = rm.write_snapshot(snap, tmp_path / "snap.json")
    loaded = rm.load_snapshot(p)
    assert loaded["assets"] == snap["assets"]
    out = rm.write_html(loaded, tmp_path / "out.html")
    assert out.exists() and out.stat().st_size > 0


def test_sample_renders():
    assert rm.main(["--sample", "--out", str(Path(rm.HTML_PATH))]) == 0
