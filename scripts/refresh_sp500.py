"""Refresh sources/sp500.txt from Wikipedia's S&P 500 constituents page.

The S&P 500 rebalances quarterly (March, June, September, December) plus
ad-hoc for M&A. sigma-alert's S&P 500 subcategory in the Slack digest is
driven by a static ticker list at `sources/sp500.txt`, which this script
keeps up to date.

Source: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies — the
first table on the page is the current constituents list. pandas-datareader
and most OSS S&P 500 trackers use the same source; the schema has been
stable for years.

Ticker format: the file uses yfinance-style dashes (BRK-B, BF-B) because
sigma-alert does set-membership checks against yfinance-sourced ticker
strings. Wikipedia uses dot format (BRK.B) — we normalize on write.

Usage:
    python scripts/refresh_sp500.py            # fetch, diff, write
    python scripts/refresh_sp500.py --dry-run  # fetch, diff, report only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
SP500_PATH = ROOT / "sources" / "sp500.txt"
SP500_NAMES_PATH = ROOT / "sources" / "sp500_names.json"
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "sigma-alert-sp500-refresh/1.0 (https://github.com/jroypeterson/sigma-alert)"


def fetch_wikipedia_html(url: str = WIKIPEDIA_URL) -> str:
    """Fetch the Wikipedia S&P 500 page HTML. Wikipedia requires a UA header."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def _normalize_symbol(raw: str) -> str | None:
    """Normalize a Wikipedia symbol string to yfinance ticker format."""
    s = raw.strip().upper()
    if not s or s == "NAN":
        return None
    # Wikipedia footnote markers occasionally leak in (e.g. "GOOGL[3]")
    if "[" in s:
        s = s.split("[", 1)[0].strip()
    return s.replace(".", "-") or None


def parse_sp500_constituents(html: str) -> list[tuple[str, str]]:
    """Extract (symbol, security_name) pairs from Wikipedia's constituents table.

    Normalizes Wikipedia's dot-format (BRK.B) to yfinance dash-format
    (BRK-B). Returns a list sorted by symbol with duplicates removed.
    """
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError("No tables found in Wikipedia page")
    for t in tables:
        if "Symbol" in t.columns and "Security" in t.columns:
            constituents = t
            break
    else:
        raise ValueError("No table with Symbol and Security columns found")

    pairs: dict[str, str] = {}
    for sym, name in zip(
        constituents["Symbol"].astype(str),
        constituents["Security"].astype(str),
    ):
        s = _normalize_symbol(sym)
        if s is None:
            continue
        pairs[s] = name.strip()
    return sorted(pairs.items())


def parse_sp500_tickers(html: str) -> list[str]:
    """Backwards-compatible wrapper returning just the symbol list."""
    return [sym for sym, _ in parse_sp500_constituents(html)]


def load_existing(path: Path = SP500_PATH) -> list[str]:
    """Read the current sp500.txt, skipping comments and blank lines."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s.upper())
    return sorted(set(out))


def diff_tickers(old: list[str], new: list[str]) -> tuple[list[str], list[str]]:
    """Return (added, removed) — tickers entering and leaving the index."""
    old_set, new_set = set(old), set(new)
    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)
    return added, removed


def write_sp500(tickers: list[str], path: Path = SP500_PATH, today: str | None = None) -> None:
    """Overwrite sp500.txt with the new ticker list + refreshed header."""
    today = today or date.today().isoformat()
    header = [
        "# S&P 500 Constituents",
        f"# Last updated: {today}",
        "# Check for reconstitution updates quarterly (March, June, September, December)",
        f"# Source: {WIKIPEDIA_URL}",
    ]
    body = sorted(set(tickers))
    path.write_text("\n".join(header + body) + "\n")


def write_sp500_names(pairs: list[tuple[str, str]], path: Path = SP500_NAMES_PATH) -> None:
    """Overwrite sp500_names.json with {ticker: security_name} mapping.

    Used by sigma_screener.py as a name fallback — ticker_metadata.json is
    owned by Coverage Manager and only covers the healthcare/MedTech/PA
    universe, so most S&P 500 alerts would render without short names
    otherwise.
    """
    mapping = {sym: name for sym, name in pairs if name}
    path.write_text(json.dumps(dict(sorted(mapping.items())), indent=2) + "\n")


def print_report(added: list[str], removed: list[str], new_total: int) -> None:
    print(f"S&P 500 constituents from {WIKIPEDIA_URL}")
    print(f"New total: {new_total}")
    print(f"Added:   {len(added)}")
    print(f"Removed: {len(removed)}")
    if added:
        print("\n+ Added:")
        for t in added:
            print(f"  + {t}")
    if removed:
        print("\n- Removed:")
        for t in removed:
            print(f"  - {t}")
    if not added and not removed:
        print("\n(no changes — file already up to date)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and diff, but do not write sources/sp500.txt",
    )
    args = parser.parse_args()

    print(f"Fetching {WIKIPEDIA_URL} ...")
    try:
        html = fetch_wikipedia_html()
    except requests.RequestException as e:
        print(f"ERROR: fetch failed: {e}", file=sys.stderr)
        return 1

    try:
        new_pairs = parse_sp500_constituents(html)
    except Exception as e:
        print(f"ERROR: parse failed: {e}", file=sys.stderr)
        return 1

    new_tickers = [sym for sym, _ in new_pairs]

    if len(new_tickers) < 400 or len(new_tickers) > 600:
        print(
            f"ERROR: got {len(new_tickers)} tickers — that's way off from 500. "
            f"Refusing to overwrite; check the Wikipedia page schema.",
            file=sys.stderr,
        )
        return 1

    existing = load_existing()
    added, removed = diff_tickers(existing, new_tickers)
    print_report(added, removed, new_total=len(new_tickers))

    if args.dry_run:
        print("\n[dry-run] no file written")
        return 0

    # Always refresh the names file — even if the ticker set is unchanged,
    # company renames (e.g. Facebook -> Meta Platforms) should propagate.
    write_sp500_names(new_pairs)
    print(f"Wrote {len(new_pairs)} name mappings to {SP500_NAMES_PATH}")

    if not added and not removed:
        print("\nNo ticker changes — leaving sp500.txt untouched.")
        return 0

    write_sp500(new_tickers)
    print(f"\nWrote {len(new_tickers)} tickers to {SP500_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
