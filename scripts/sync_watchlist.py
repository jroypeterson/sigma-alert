"""
Watchlist sync script for sigma-alert.

Merges all source files in sources/ into a single watchlist.txt.

Source files:
  sources/hc_services.txt — Healthcare Services coverage
  sources/medtech.txt     — MedTech coverage
  sources/sp500.txt       — S&P 500 constituents
  sources/index_etfs.txt  — Broad-market index ETFs (SPYM, DIA, QQQ)
  sources/sector_etfs.txt — SPDR Select Sector ETFs (XLE, XLF, ...)

Usage:
  python scripts/sync_watchlist.py

The script is designed to run both locally and in GitHub Actions.
When run in CI, the workflow handles committing the updated watchlist.

Note: ticker_metadata.json (company names + sector tags used by the
screener) is owned by Coverage Manager. Its weekly-build pipeline writes
the file directly into this repo and pushes it. Do NOT generate the
metadata file from this script — CI does not have access to the
Coverage Manager CSV, so doing so would corrupt the file.
"""

from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = REPO_ROOT / "sources"
WATCHLIST_PATH = REPO_ROOT / "watchlist.txt"

# Ordered list of (section_name, filename) — order determines priority for dedup
SOURCES = [
    ("Healthcare Services", "hc_services.txt"),
    ("MedTech", "medtech.txt"),
    ("S&P 500", "sp500.txt"),
    ("Index ETFs", "index_etfs.txt"),
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
        "# Synced from Coverage Manager (HC Services, MedTech, S&P 500) + index/sector ETFs",
        f"# Last synced: {date_str}",
        "",
    ]
    for section_name, tickers in sections:
        lines.append(f"# --- {section_name} ({len(tickers)} tickers) ---")
        lines.extend(tickers)
        lines.append("")

    print(f"[INFO] Total unique tickers: {total}")
    return "\n".join(lines)


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
            return

    WATCHLIST_PATH.write_text(content)
    print(f"[OK] Watchlist written to {WATCHLIST_PATH}")


if __name__ == "__main__":
    main()
