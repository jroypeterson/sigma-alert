"""Tests for scripts/refresh_sp500.py — no live Wikipedia calls."""

from pathlib import Path

import pytest

from scripts.refresh_sp500 import (
    diff_tickers,
    load_existing,
    parse_sp500_tickers,
    write_sp500,
)


# Minimal fake HTML with the schema Wikipedia uses: a table whose first
# column is "Symbol". pd.read_html pulls tables from HTML directly.
FAKE_HTML = """
<html><body>
<table>
  <thead><tr><th>Symbol</th><th>Security</th><th>Sector</th></tr></thead>
  <tbody>
    <tr><td>AAPL</td><td>Apple Inc.</td><td>Technology</td></tr>
    <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>
    <tr><td>BF.B</td><td>Brown-Forman</td><td>Consumer Staples</td></tr>
    <tr><td>GOOGL[3]</td><td>Alphabet</td><td>Communication Services</td></tr>
    <tr><td>MSFT</td><td>Microsoft</td><td>Technology</td></tr>
  </tbody>
</table>
</body></html>
"""


def test_parse_extracts_symbols_from_first_table():
    tickers = parse_sp500_tickers(FAKE_HTML)
    assert "AAPL" in tickers
    assert "MSFT" in tickers


def test_parse_normalizes_dot_to_dash():
    tickers = parse_sp500_tickers(FAKE_HTML)
    assert "BRK-B" in tickers
    assert "BF-B" in tickers
    assert "BRK.B" not in tickers
    assert "BF.B" not in tickers


def test_parse_strips_footnote_markers():
    tickers = parse_sp500_tickers(FAKE_HTML)
    assert "GOOGL" in tickers
    assert "GOOGL[3]" not in tickers


def test_parse_returns_sorted_unique():
    tickers = parse_sp500_tickers(FAKE_HTML)
    assert tickers == sorted(set(tickers))


def test_parse_raises_on_no_symbol_column():
    bad_html = "<html><body><table><thead><tr><th>Name</th></tr></thead><tbody><tr><td>AAPL</td></tr></tbody></table></body></html>"
    with pytest.raises(ValueError, match="Symbol"):
        parse_sp500_tickers(bad_html)


# ── diff_tickers ────────────────────────────────────────────────────────────


def test_diff_detects_adds_and_removes():
    old = ["AAPL", "MSFT", "IBM"]
    new = ["AAPL", "MSFT", "NVDA", "PLTR"]
    added, removed = diff_tickers(old, new)
    assert added == ["NVDA", "PLTR"]
    assert removed == ["IBM"]


def test_diff_no_change():
    old = new = ["AAPL", "MSFT"]
    assert diff_tickers(old, new) == ([], [])


# ── load_existing / write_sp500 round trip ──────────────────────────────────


@pytest.fixture
def sp500_file(tmp_path: Path) -> Path:
    path = tmp_path / "sp500.txt"
    path.write_text(
        "# S&P 500 Constituents\n"
        "# Last updated: 2026-01-01\n"
        "# Check for reconstitution updates quarterly\n"
        "AAPL\n"
        "MSFT\n"
        "IBM\n"
        "\n"
        "BRK-B\n"
    )
    return path


def test_load_existing_skips_comments_and_blanks(sp500_file):
    assert load_existing(sp500_file) == ["AAPL", "BRK-B", "IBM", "MSFT"]


def test_write_sp500_header_and_body(tmp_path):
    path = tmp_path / "sp500.txt"
    write_sp500(["MSFT", "AAPL", "NVDA"], path=path, today="2026-04-11")
    content = path.read_text()
    assert "# Last updated: 2026-04-11" in content
    assert "# Source: https://en.wikipedia.org/wiki/List_of_S" in content
    # Body is sorted + deduped
    body = [line for line in content.splitlines() if line and not line.startswith("#")]
    assert body == ["AAPL", "MSFT", "NVDA"]


def test_write_sp500_dedupes(tmp_path):
    path = tmp_path / "sp500.txt"
    write_sp500(["AAPL", "AAPL", "MSFT"], path=path, today="2026-04-11")
    body = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    assert body == ["AAPL", "MSFT"]


def test_round_trip_load_write_load(tmp_path):
    path = tmp_path / "sp500.txt"
    tickers = ["AAPL", "BRK-B", "MSFT", "NVDA"]
    write_sp500(tickers, path=path, today="2026-04-11")
    assert load_existing(path) == tickers
