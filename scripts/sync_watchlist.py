"""
Watchlist sync script for sigma-alert.

Merges all source files in sources/ into a single watchlist.txt and
generates ticker_metadata.json with company names and sector tags
from the Coverage Manager CSV.

Source files:
  sources/hc_services.txt — Healthcare Services coverage
  sources/medtech.txt     — MedTech coverage
  sources/sp500.txt       — S&P 500 constituents
  sources/sector_etfs.txt — SPDR sector ETFs + S&P 500 portfolio ETF

Usage:
  python scripts/sync_watchlist.py

The script is designed to run both locally and in GitHub Actions.
When run in CI, the workflow handles committing the updated watchlist.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = REPO_ROOT / "sources"
WATCHLIST_PATH = REPO_ROOT / "watchlist.txt"
METADATA_PATH = REPO_ROOT / "ticker_metadata.json"

# Coverage Manager CSV — used for company name and sector lookup
COVERAGE_CSV = Path(__file__).resolve().parent.parent.parent / "Coverage Manager" / "data" / "coverage_universe_tickers.csv"

# Ordered list of (section_name, filename) — order determines priority for dedup
SOURCES = [
    ("Healthcare Services", "hc_services.txt"),
    ("MedTech", "medtech.txt"),
    ("S&P 500", "sp500.txt"),
    ("Sector ETFs", "sector_etfs.txt"),
]


def load_source(path: Path) -> list[str]:
    """Load tickers from a source file, preserving original casing."""
    if not path.exists():
        print(f"[WARN] Source file not found: {path}")
        return []
    tickers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tickers.append(line)
    return tickers


def build_watchlist() -> str:
    """Merge all sources into watchlist content string."""
    seen: set[str] = set()
    sections: list[tuple[str, list[str]]] = []

    for section_name, filename in SOURCES:
        path = SOURCES_DIR / filename
        tickers = load_source(path)
        unique = []
        for t in tickers:
            key = t.upper()
            if key not in seen:
                seen.add(key)
                unique.append(t)
        sections.append((section_name, unique))
        dupes = len(tickers) - len(unique)
        if dupes:
            print(f"[INFO] {section_name}: {len(unique)} unique, {dupes} duplicates removed")
        else:
            print(f"[INFO] {section_name}: {len(unique)} tickers")

    total = sum(len(t) for _, t in sections)
    date_str = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "# Sigma-Alert Watchlist",
        "# Synced from Coverage Manager (HC Services, MedTech, S&P 500, Sector ETFs)",
        f"# Last synced: {date_str}",
        "",
    ]
    for section_name, tickers in sections:
        lines.append(f"# --- {section_name} ({len(tickers)} tickers) ---")
        lines.extend(tickers)
        lines.append("")

    print(f"[INFO] Total unique tickers: {total}")
    return "\n".join(lines)


def build_ticker_metadata() -> dict:
    """Build metadata dict from Coverage Manager CSV.

    Returns {TICKER: {"name": "...", "sector": "..."}} for all tickers
    found in the coverage universe.
    """
    metadata = {}
    if not COVERAGE_CSV.exists():
        print(f"[WARN] Coverage CSV not found at {COVERAGE_CSV} — skipping metadata")
        return metadata

    with open(COVERAGE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("Ticker", "").strip().upper()
            if not ticker or ticker == "#N/A":
                continue
            # Strip any exchange suffix for matching (e.g. "ROG SW" -> "ROG")
            plain = ticker.split()[0] if " " in ticker else ticker
            plain = plain.split(".")[0] if "." in plain else plain
            name = row.get("Company Name", "").strip()
            sector = row.get("Sector (JP)", "").strip()
            subsector = row.get("Subsector (JP)", "").strip()
            metadata[plain] = {
                "name": name,
                "sector": sector,
                "subsector": subsector,
            }

    # Add sector ETFs manually (not in coverage CSV)
    etf_names = {
        "XLE": ("Energy Select Sector SPDR", "ETF"),
        "XLB": ("Materials Select Sector SPDR", "ETF"),
        "XLU": ("Utilities Select Sector SPDR", "ETF"),
        "XLP": ("Consumer Staples Select Sector SPDR", "ETF"),
        "XLI": ("Industrial Select Sector SPDR", "ETF"),
        "XLRE": ("Real Estate Select Sector SPDR", "ETF"),
        "XLC": ("Communication Services Select Sector SPDR", "ETF"),
        "XLV": ("Health Care Select Sector SPDR", "ETF"),
        "XLK": ("Technology Select Sector SPDR", "ETF"),
        "XLY": ("Consumer Discretionary Select Sector SPDR", "ETF"),
        "XLF": ("Financial Select Sector SPDR", "ETF"),
        "XBI": ("SPDR S&P Biotech ETF", "ETF"),
        "SPYM": ("SPDR Portfolio S&P 500 ETF", "ETF"),
    }
    for t, (name, sector) in etf_names.items():
        if t not in metadata:
            metadata[t] = {"name": name, "sector": sector, "subsector": ""}

    print(f"[INFO] Ticker metadata: {len(metadata)} entries")
    return metadata


def main():
    content = build_watchlist()

    # Check if anything changed
    if WATCHLIST_PATH.exists():
        existing = WATCHLIST_PATH.read_text()
        # Compare ignoring the "Last synced" date line
        def strip_date(s):
            return "\n".join(
                l for l in s.splitlines() if not l.startswith("# Last synced:")
            )
        if strip_date(existing) == strip_date(content):
            print("[INFO] Watchlist is already up to date — no changes")
        else:
            WATCHLIST_PATH.write_text(content)
            print(f"[OK] Watchlist written to {WATCHLIST_PATH}")

    # Always rebuild metadata (cheap operation, keeps it fresh)
    metadata = build_ticker_metadata()
    if metadata:
        METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        print(f"[OK] Metadata written to {METADATA_PATH}")


if __name__ == "__main__":
    main()
