#!/usr/bin/env python3
"""Interactive "return map" — a BlackRock-style periodic-table-of-returns for
the asset classes / index groups the sigma screener already tracks.

This module does NOT fetch any market data. It reuses the returns the screener
has already computed for the "Index, Sector & Macro Returns" Slack block:

  * today's % move (``return_pct`` from ``etf_returns``)
  * year-to-date % (``ytd_return_pct`` from the period-returns fetch)
  * prior calendar-year % (``prior_year_return_pct`` from the same fetch)

``assemble_snapshot()`` packs those into a compact, self-describing JSON
snapshot (``cache/returns_snapshot.json``); ``build_html()`` renders that
snapshot into a single, dependency-free interactive HTML file
(``readable/return_map.html``) — three ranked, color-by-asset-class columns
(Prior Year / YTD / Today), the periodic-table signature.

The screener writes both artifacts at the end of every run (warn-and-proceed —
a failure here never breaks the alert pipeline). You can also rebuild the chart
by hand from the last committed snapshot, with no API calls:

    python scripts/return_map.py                 # snapshot -> readable/return_map.html
    python scripts/return_map.py --sample        # render bundled sample data (smoke test)
    python scripts/return_map.py --open          # also open it in the browser
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT / "cache" / "returns_snapshot.json"
HTML_PATH = ROOT / "readable" / "return_map.html"
ET = ZoneInfo("America/New_York")

# Group display order — US Indices lead (matches the Slack section after #20),
# then the rest of the equity-comparable groups. Credit / Treasury-curve rows
# are deliberately excluded: they're yield/spread *levels* in basis points, not
# %-returns, so they don't belong in a returns periodic table.
GROUP_ORDER = [
    "US Indices",
    "Global Equity",
    "Sectors",
    "Healthcare",
    "Tech Themes",
    "Commodities",
    "Macro",
]

# Stable, distinguishable color per asset class (the BlackRock periodic table
# colors tiles BY asset class, so a reader can trace one class across columns).
GROUP_COLORS = {
    "US Indices": "#1f4e79",
    "Global Equity": "#2e8b9e",
    "Sectors": "#c46210",
    "Healthcare": "#2e7d32",
    "Tech Themes": "#6a4c93",
    "Commodities": "#8c6d1f",
    "Macro": "#5f6b7a",
}
DEFAULT_COLOR = "#555555"

# The three return periods rendered as columns, in order.
PERIODS = [
    ("prior_year_pct", "Prior Year"),
    ("ytd_pct", "YTD"),
    ("today_pct", "Today"),
]


def assemble_snapshot(
    etf_returns,
    period_map,
    *,
    index_set=None,
    global_equity_set=None,
    sector_set=None,
    healthcare_set=None,
    tech_set=None,
    commodity_set=None,
    macro_set=None,
    macro_style=None,
    mode="",
    ref_date="",
    generated_at=None,
):
    """Build the return-map snapshot dict from the screener's already-computed
    returns. No network access. ``etf_returns`` is the list the screener passes
    to the Slack returns block; ``period_map`` is the
    ``fetch_etf_period_returns`` result keyed by display ticker.

    Macro rows styled as a yield (e.g. ^TNX) are dropped — a "% return" on a
    yield level is misleading and not comparable to the equity rows.
    """
    index_set = index_set or set()
    global_equity_set = global_equity_set or set()
    sector_set = sector_set or set()
    healthcare_set = healthcare_set or set()
    tech_set = tech_set or set()
    commodity_set = commodity_set or set()
    macro_set = macro_set or set()
    macro_style = macro_style or {}
    period_map = period_map or {}

    def group_of(ticker):
        if ticker in index_set:
            return "US Indices"
        if ticker in global_equity_set:
            return "Global Equity"
        if ticker in healthcare_set:
            return "Healthcare"
        if ticker in tech_set:
            return "Tech Themes"
        if ticker in commodity_set:
            return "Commodities"
        if ticker in macro_set:
            return "Macro"
        if ticker in sector_set:
            return "Sectors"
        return None

    assets = []
    for row in etf_returns or []:
        ticker = row.get("ticker")
        group = group_of(ticker)
        if group is None:
            continue
        # Skip yield-style macro rows (e.g. the 10Y) — not return-comparable.
        if group == "Macro" and macro_style.get(ticker) == "yield":
            continue
        period = period_map.get(ticker) or {}
        assets.append({
            "ticker": ticker,
            "name": row.get("name") or ticker,
            "group": group,
            "today_pct": _num(row.get("return_pct")),
            "ytd_pct": _num(period.get("ytd_return_pct")),
            "prior_year_pct": _num(period.get("prior_year_return_pct")),
            "prior_year_label": period.get("prior_year_label") or "Prior Year",
        })

    if generated_at is None:
        generated_at = datetime.now(ET).isoformat(timespec="seconds")

    return {
        "generated_at": generated_at,
        "mode": mode,
        "ref_date": ref_date or "",
        "assets": assets,
    }


def _num(v):
    """Coerce to float or None (never NaN)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def write_snapshot(snapshot, path=SNAPSHOT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return path


def load_snapshot(path=SNAPSHOT_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt_pct(v):
    if v is None:
        return "&mdash;"
    return f"{'+' if v >= 0 else ''}{v:.1f}%"


def build_html(snapshot) -> str:
    """Render the snapshot into a single self-contained interactive HTML page.
    No external assets / CDNs — works opened straight off disk and offline."""
    assets = snapshot.get("assets") or []
    generated_at = snapshot.get("generated_at", "")
    ref_date = snapshot.get("ref_date", "")
    mode = snapshot.get("mode", "")

    # Period labels: use the data's prior-year label (e.g. "2025") when present.
    prior_year_label = "Prior Year"
    for a in assets:
        if a.get("prior_year_label") and a["prior_year_label"] != "Prior Year":
            prior_year_label = a["prior_year_label"]
            break
    column_labels = {
        "prior_year_pct": prior_year_label,
        "ytd_pct": "YTD",
        "today_pct": "Today",
    }

    groups_present = [g for g in GROUP_ORDER if any(a["group"] == g for a in assets)]

    # Build the three ranked columns.
    columns_html = []
    for key, _default in PERIODS:
        ranked = sorted(
            [a for a in assets if a.get(key) is not None],
            key=lambda a: a[key], reverse=True,
        )
        # Assets missing this period render at the bottom in stable order.
        missing = [a for a in assets if a.get(key) is None]
        cells = []
        for a in ranked + missing:
            color = GROUP_COLORS.get(a["group"], DEFAULT_COLOR)
            val = a.get(key)
            cells.append(
                f'<div class="cell" style="background:{color}" '
                f'data-group="{html.escape(a["group"])}" '
                f'title="{html.escape(a["name"])} ({html.escape(a["ticker"])}) '
                f'&#10;{html.escape(column_labels[key])}: {_fmt_pct(val).replace("&mdash;","n/a")}">'
                f'<span class="tk">{html.escape(a["ticker"])}</span>'
                f'<span class="vp">{_fmt_pct(val)}</span>'
                f'</div>'
            )
        columns_html.append(
            f'<div class="col"><div class="colhead">{html.escape(column_labels[key])}</div>'
            + "".join(cells) + "</div>"
        )

    legend_html = "".join(
        f'<button class="lg" data-group="{html.escape(g)}" '
        f'style="--c:{GROUP_COLORS.get(g, DEFAULT_COLOR)}">'
        f'<span class="sw"></span>{html.escape(g)}</button>'
        for g in groups_present
    )

    sub = []
    if mode:
        sub.append(f"{html.escape(mode)} run")
    if ref_date:
        sub.append(f"bar {html.escape(ref_date)}")
    if generated_at:
        sub.append(f"generated {html.escape(generated_at)}")
    subtitle = " &middot; ".join(sub) if sub else "no run metadata"

    empty_note = ""
    if not assets:
        empty_note = (
            '<p class="empty">No returns in the snapshot yet. Run a screener '
            'close (<code>python scripts/sigma_screener.py --mode close</code>) '
            'or render the bundled sample with '
            '<code>python scripts/return_map.py --sample</code>.</p>'
        )

    return _TEMPLATE.format(
        subtitle=subtitle,
        legend=legend_html,
        columns="".join(columns_html),
        empty_note=empty_note,
    )


def write_html(snapshot, path=HTML_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(snapshot), encoding="utf-8")
    return path


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sigma Return Map</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; padding: 24px; background: #0f1115; color: #e6e6e6; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .sub {{ color: #9aa4b2; font-size: 13px; margin: 0 0 16px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 18px; }}
  .lg {{ display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
        border: 1px solid #2a2f3a; background: #171a21; color: #e6e6e6;
        padding: 4px 10px; border-radius: 14px; font-size: 12px; }}
  .lg .sw {{ width: 12px; height: 12px; border-radius: 3px; background: var(--c); }}
  .lg.off {{ opacity: 0.32; }}
  .grid {{ display: flex; gap: 14px; align-items: flex-start; overflow-x: auto; }}
  .col {{ flex: 1 1 0; min-width: 150px; }}
  .colhead {{ text-align: center; font-weight: 600; font-size: 13px;
             padding: 6px 0; color: #cfd6e0; border-bottom: 1px solid #2a2f3a;
             margin-bottom: 8px; }}
  .cell {{ border-radius: 6px; padding: 7px 9px; margin-bottom: 6px;
          display: flex; justify-content: space-between; align-items: baseline;
          color: #fff; font-size: 13px; box-shadow: 0 1px 2px rgba(0,0,0,.3);
          transition: opacity .15s, transform .15s; }}
  .cell .tk {{ font-weight: 700; letter-spacing: .2px; }}
  .cell .vp {{ font-variant-numeric: tabular-nums; opacity: .95; }}
  .cell.dim {{ opacity: 0.12; }}
  .cell.hot {{ transform: translateX(2px); outline: 2px solid #fff6; }}
  .empty {{ color: #9aa4b2; }}
  code {{ background: #171a21; padding: 1px 5px; border-radius: 4px; }}
  footer {{ margin-top: 18px; color: #6b7280; font-size: 11px; }}
</style>
</head>
<body>
  <h1>Sigma Return Map &mdash; periodic table of returns</h1>
  <p class="sub">{subtitle}</p>
  <div class="legend">{legend}</div>
  <div class="grid">{columns}</div>
  {empty_note}
  <footer>Tiles colored by asset class, ranked best&rarr;worst within each
  period. Hover a tile for detail; click a legend chip to isolate a class.
  Built from the sigma screener's existing returns block &mdash; no extra data
  fetch.</footer>
<script>
(function() {{
  var legend = document.querySelectorAll('.lg');
  var active = {{}};  // group -> isolated?
  function apply() {{
    var any = Object.keys(active).some(function(k) {{ return active[k]; }});
    document.querySelectorAll('.cell').forEach(function(c) {{
      var g = c.getAttribute('data-group');
      var on = !any || active[g];
      c.classList.toggle('dim', !on);
      c.classList.toggle('hot', any && active[g]);
    }});
    legend.forEach(function(b) {{
      var g = b.getAttribute('data-group');
      b.classList.toggle('off', any && !active[g]);
    }});
  }}
  legend.forEach(function(b) {{
    b.addEventListener('click', function() {{
      var g = b.getAttribute('data-group');
      active[g] = !active[g];
      apply();
    }});
  }});
}})();
</script>
</body>
</html>
"""


def _sample_snapshot():
    """Tiny built-in dataset so `--sample` renders without any run/snapshot."""
    return {
        "generated_at": "2026-06-29T16:30:00-04:00",
        "mode": "sample",
        "ref_date": "2026-06-26",
        "assets": [
            {"ticker": "SPYM", "name": "S&P 500", "group": "US Indices",
             "today_pct": 0.4, "ytd_pct": 6.1, "prior_year_pct": 24.5, "prior_year_label": "2025"},
            {"ticker": "^RUT", "name": "Russell 2000", "group": "US Indices",
             "today_pct": -0.3, "ytd_pct": 2.2, "prior_year_pct": 11.4, "prior_year_label": "2025"},
            {"ticker": "QQQ", "name": "Nasdaq 100", "group": "US Indices",
             "today_pct": 0.8, "ytd_pct": 9.7, "prior_year_pct": 28.1, "prior_year_label": "2025"},
            {"ticker": "EEM", "name": "Emerging Markets", "group": "Global Equity",
             "today_pct": 0.2, "ytd_pct": 7.3, "prior_year_pct": 5.0, "prior_year_label": "2025"},
            {"ticker": "XLV", "name": "Health Care", "group": "Sectors",
             "today_pct": -0.5, "ytd_pct": -1.2, "prior_year_pct": 3.4, "prior_year_label": "2025"},
            {"ticker": "XLK", "name": "Technology", "group": "Sectors",
             "today_pct": 1.1, "ytd_pct": 11.0, "prior_year_pct": 30.2, "prior_year_label": "2025"},
            {"ticker": "XBI", "name": "Biotech (Equal Wt)", "group": "Healthcare",
             "today_pct": -1.4, "ytd_pct": -4.6, "prior_year_pct": -8.0, "prior_year_label": "2025"},
            {"ticker": "SMH", "name": "Semis", "group": "Tech Themes",
             "today_pct": 1.7, "ytd_pct": 14.2, "prior_year_pct": 37.0, "prior_year_label": "2025"},
            {"ticker": "GLD", "name": "Gold", "group": "Commodities",
             "today_pct": 0.6, "ytd_pct": 18.9, "prior_year_pct": 26.1, "prior_year_label": "2025"},
            {"ticker": "BTC-USD", "name": "Bitcoin", "group": "Commodities",
             "today_pct": -2.1, "ytd_pct": 22.5, "prior_year_pct": 41.0, "prior_year_label": "2025"},
            {"ticker": "CL=F", "name": "WTI Crude Oil", "group": "Macro",
             "today_pct": 0.9, "ytd_pct": -3.3, "prior_year_pct": 1.1, "prior_year_label": "2025"},
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the sigma return-map HTML")
    parser.add_argument("--snapshot", default=str(SNAPSHOT_PATH),
                        help="returns snapshot JSON to render (default: cache/returns_snapshot.json)")
    parser.add_argument("--out", default=str(HTML_PATH),
                        help="output HTML path (default: readable/return_map.html)")
    parser.add_argument("--sample", action="store_true",
                        help="render the bundled sample data instead of a snapshot")
    parser.add_argument("--open", action="store_true", dest="open_browser",
                        help="open the rendered HTML in the default browser")
    args = parser.parse_args(argv)

    if args.sample:
        snapshot = _sample_snapshot()
    else:
        snap_path = Path(args.snapshot)
        if not snap_path.exists():
            print(f"[ERROR] snapshot not found: {snap_path}")
            print("        Run a screener close first, or use --sample.")
            return 1
        snapshot = load_snapshot(snap_path)

    out = write_html(snapshot, args.out)
    n = len(snapshot.get("assets") or [])
    print(f"[OK] return map written: {out} ({n} assets)")
    if args.open_browser:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
