"""Weekly stock-price summary — the last #stock-price-alerts message of the week.

Runs after the week's final close digest (Friday, or Thursday when Friday is an
NYSE holiday) and posts:

  * weekly sigma alerts across the coverage watchlist, in the same tiers and
    subcategory order as the daily digest (2σ on everything, 1σ on Core +
    the five Coverage Manager position lists);
  * the week's index / macro / rates / credit / sector moves.

NO MARKET-DATA PULL. Every price comes from `cache/distribution_cache.json`,
which the close run just wrote: `week_baseline()` stores each name's
`last_close`, `week_base_close` (last close before this week's Monday) and
`week_sessions`. Only the FRED rows (Treasury curve, credit, mortgage) are
fetched, from FRED's public no-key CSV, warn-and-proceed.

Weekly z-score: `(r_week - n*mu) / (sigma * sqrt(n))`, n = sessions traded this
week, mu/sigma = the trailing-252-day DAILY distribution already in the cache.
The sqrt-time scaling assumes daily returns are independent; it is the standard
approximation, not a fitted weekly distribution.

Writes `cache/weekly_snapshot.json` (committed by sigma-close.yml; it is also
the posted-this-week marker, so the routine duplicate close run does not post
twice) and `readable/weekly_summary.html`, the page republished to the claude.ai
artifact at WEEKLY_PAGE_URL.

    python scripts/weekly_summary.py              # gated: last session of week + digest delivered
    python scripts/weekly_summary.py --dry-run    # print the Slack payload, write nothing
    python scripts/weekly_summary.py --html-only  # rebuild the HTML from the committed snapshot
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sigma_screener as ss  # noqa: E402

ROOT = ss.ROOT
SNAPSHOT_PATH = ROOT / "cache" / "weekly_snapshot.json"
HTML_PATH = ROOT / "readable" / "weekly_summary.html"

# The claude.ai artifact this page is republished to (interactive job
# `sigma-weekly-page`). CI cannot publish an artifact, so the Slack post links
# the stable URL and the page states its own as-of week.
WEEKLY_PAGE_URL = "https://claude.ai/artifact/R775MwP4gSdxrPgw7xsYXu"

# Below this share of the watchlist carrying a current weekly figure, post
# nothing: same floor, same reason, as the daily digest's MIN_SCREEN_COVERAGE.
MIN_WEEKLY_COVERAGE = ss.MIN_SCREEN_COVERAGE

ETF_GROUPS = [
    ("US Indices", ss.load_index_etfs),
    ("Global Equity", ss.load_global_equity_etfs),
    ("Sectors", ss.load_sector_etfs),
    ("Healthcare", ss.load_healthcare_etfs),
    ("Tech Themes", ss.load_tech_etfs),
    ("Commodities", ss.load_commodity_etfs),
]
GREEN, RED = "\U0001F7E9", "\U0001F7E5"


# ── calendar ─────────────────────────────────────────────────────────────────

def is_last_session_of_week(day: date) -> bool:
    """True iff `day` is an NYSE session and no later weekday this week is."""
    if not ss.is_nyse_session(day):
        return False
    d = day + timedelta(days=1)
    while d.weekday() < 5:
        if ss.is_nyse_session(d):
            return False
        d += timedelta(days=1)
    return True


# ── math ─────────────────────────────────────────────────────────────────────

def weekly_move(entry: dict, ref_date: str) -> dict | None:
    """The week's move for one cache entry, or None when it is not current.

    Current = last scored on a bar dated THIS week (Monday..ref_date). Not
    "== ref_date": ~20 European names are routinely scored a session late
    (their EOD reaches Yahoo after the close run), and an exact-date test would
    drop them from every weekly summary. Their row carries `through` instead.
    A carried, refused entry from an earlier week is still excluded.
    """
    last_bar = entry.get("last_bar") or ""
    monday = (date.fromisoformat(ref_date)
              - timedelta(days=date.fromisoformat(ref_date).weekday())).isoformat()
    if not (monday <= last_bar <= ref_date):
        return None
    last, base = entry.get("last_close"), entry.get("week_base_close")
    n = entry.get("week_sessions") or 0
    mu, sigma = entry.get("mu"), entry.get("sigma")
    if not last or not base or n < 1 or mu is None or not sigma:
        return None
    r = last / base - 1
    z = (r - n * mu) / (sigma * math.sqrt(n))
    out = {"week_pct": r * 100, "z": z, "sessions": n, "price": last,
           "base": base, "base_date": entry.get("week_base_date")}
    if last_bar != ref_date:
        out["through"] = last_bar
    pyc = entry.get("prior_year_end_close")
    if pyc and entry.get("prior_year_end_year") == int(ref_date[:4]) - 1:
        out["ytd_pct"] = (last / pyc - 1) * 100
    hi = entry.get("high_52w")
    if hi:
        out["pct_of_high"] = last / hi * 100
    return out


def fred_week_change(series: list[tuple[str, float]], monday: date) -> dict | None:
    """Level now vs the last observation before `monday`, in bp."""
    if not series:
        return None
    before = [(d, v) for d, v in series if d < monday.isoformat()]
    if not before:
        return None
    (last_d, last), (base_d, base) = series[-1], before[-1]
    if last_d < monday.isoformat():
        return None  # nothing published this week yet
    return {"level": last, "bp_chg": (last - base) * 100,
            "obs_date": last_d, "base_date": base_d}


def fetch_fred_block(monday: date) -> dict:
    start = f"{monday.year - 1}-12-01"
    out = {"curve": [], "credit": [], "mortgage": None}
    for key in ss.TREASURY_CURVE_ORDER:
        ch = fred_week_change(ss._fetch_fred_series(ss.TREASURY_CURVE_SERIES[key], start=start), monday)
        if ch:
            out["curve"].append({"label": key, **ch})
    for key in ss.CREDIT_ORDER:
        cfg = ss.CREDIT_SERIES[key]
        y = fred_week_change(ss._fetch_fred_series(cfg["yield_id"], start=start), monday)
        o = fred_week_change(ss._fetch_fred_series(cfg["oas_id"], start=start), monday)
        if y:
            row = {"label": cfg["label"], "key": key, "yield": y}
            if o:
                row["oas"] = o
            out["credit"].append(row)
    # PMMS is weekly (Thursday), so last-vs-previous IS the weekly change.
    m = ss.fetch_mortgage_rate()
    if m:
        out["mortgage"] = m
    return out


# ── assembly ─────────────────────────────────────────────────────────────────

def load_display_metadata(tickers: list[str]) -> tuple[dict, set]:
    """CM metadata + S&P 500 / ETF display names, as main() builds it."""
    raw = ss.load_metadata()
    md = {k: dict(v) for k, v in raw.items()}
    collisions = ss.disambiguate_collision_metadata(md, raw, tickers)
    for t, name in ss.load_sp500_names().items():
        e = md.get(t)
        if e is None:
            md[t] = {"name": name, "sector": "", "subsector": ""}
        elif not (e.get("name") or "").strip():
            e["name"] = name
    for t, name in ss.load_etf_names().items():
        md[t] = {"name": name, "sector": "", "subsector": ""}
    return md, collisions


def build_snapshot(cache: dict, ref: date, fred: dict | None = None) -> dict:
    ref_date = ref.isoformat()
    monday = ref - timedelta(days=ref.weekday())
    tickers = ss.load_watchlist()
    md, collisions = load_display_metadata(tickers)
    sp500 = ss.load_sp500_set()
    sets = {
        "portfolio": ss.load_portfolio(), "researching": ss.load_researching(),
        "following": ss.load_following_for_interest(),
        "ready_to_buy": ss.load_ready_to_buy(), "ready_to_short": ss.load_ready_to_short(),
    }
    groups = [(label, loader()) for label, loader in ETF_GROUPS]
    macro = ss.load_macro()
    etf_set = set().union(*(g for _, g in groups)) | macro
    weighting = ss.load_etf_weighting()
    entries = cache.get("tickers", {})

    names, missing = [], []
    for t in tickers:
        if t in etf_set:
            continue
        mv = weekly_move(entries.get(t, {}), ref_date)
        if mv is None:
            missing.append(t)
            continue
        meta, key = ss.resolve_metadata(md, t, collisions)
        meta = meta or {}
        row = {
            "ticker": t, "name": ss.short_company_name(meta.get("name", "")),
            "sector": meta.get("sector", ""), "subsector": meta.get("subsector", ""),
            "commercial": meta.get("commercial", ""),
            "in_portfolio": key in sets["portfolio"],
            "in_researching": key in sets["researching"],
            "in_following_for_interest": key in sets["following"],
            "in_ready_to_buy": key in sets["ready_to_buy"],
            "in_ready_to_short": key in sets["ready_to_short"],
            **mv,
        }
        row["categories"] = [lbl for lbl, pred in ss.SUBCATEGORIES if pred(row, sp500)]
        abs_z = abs(mv["z"])
        if abs_z >= ss.SIGMA_THRESHOLD:
            row["tier"] = "2sigma"
        elif abs_z >= ss.ONE_SIGMA_THRESHOLD and ss._is_one_sigma_eligible(
                meta, key, sets["portfolio"], sets["researching"],
                following_set=sets["following"], ready_to_buy_set=sets["ready_to_buy"],
                ready_to_short_set=sets["ready_to_short"]):
            row["tier"] = "1sigma"
        names.append(row)

    def etf_row(t, group):
        mv = weekly_move(entries.get(t, {}), ref_date)
        if mv is None:
            return None
        return {"ticker": t, "group": group, "name": md.get(t, {}).get("name", t),
                "weighting": weighting.get(t), **mv}

    etf_groups = []
    for label, members in groups:
        rows = [r for r in (etf_row(t, label) for t in members) if r]
        rows.sort(key=lambda r: r["week_pct"], reverse=True)
        etf_groups.append({"group": label, "rows": rows,
                           "missing": sorted(set(members) - {r["ticker"] for r in rows})})
    macro_rows = [r for r in (etf_row(t, "Macro") for t in ss.MACRO_ORDER if t in macro) if r]
    for r in macro_rows:
        r["style"] = ss.MACRO_STYLE.get(r["ticker"], "price")

    n_total = len([t for t in tickers if t not in etf_set])
    return {
        "generated_at": ss.now_et().isoformat(timespec="seconds"),
        "ref_date": ref_date, "week_start": monday.isoformat(),
        "coverage": {"current": len(names), "total": n_total,
                     "missing": sorted(missing)},
        "names": sorted(names, key=lambda r: r["z"], reverse=True),
        "etf_groups": etf_groups, "macro": macro_rows,
        "fred": fred if fred is not None else fetch_fred_block(monday),
        "posted": False,
    }


# ── Slack ────────────────────────────────────────────────────────────────────

def _sign(x, nd=2, suffix="%"):
    return f"{'+' if x > 0 else ''}{x:.{nd}f}{suffix}"


def _name_line(r):
    mark = GREEN if r["week_pct"] > 0 else RED
    nm = f" ({r['name']})" if r.get("name") else ""
    # No price: foreign listings close in local currency, and a "$" on a yen
    # close (7701.T) misstates it ~150x. The page carries the close.
    parts = [f"{mark}  `{r['ticker']}`{nm}", _sign(r["week_pct"]) + " w/w",
             f"z = {r['z']:+.2f}"]
    if r.get("pct_of_high") is not None:
        parts.append(f"{r['pct_of_high']:.0f}% of 52w high")
    if r.get("ytd_pct") is not None:
        parts.append("YTD: " + _sign(r["ytd_pct"]))
    if r.get("through"):
        parts.append(f"thru {r['through']}")
    warn = "  *⚠️ 3σ+ week*" if abs(r["z"]) >= ss.THREE_SIGMA else ""
    return "  |  ".join(parts) + warn


def _etf_line(r):
    mark = GREEN if r["week_pct"] > 0 else RED
    w = f" — {r['weighting']}" if r.get("weighting") else ""
    ytd = f"  |  YTD: {_sign(r['ytd_pct'])}" if r.get("ytd_pct") is not None else ""
    return f"{mark}  `{r['ticker']}` ({r['name']}{w})  |  {_sign(r['week_pct'])} w/w  |  z = {r['z']:+.2f}{ytd}"


def _macro_line(r):
    if r.get("style") == "yield":  # ^TNX is quoted as yield x1 (4.12 = 4.12%)
        bp = (r["price"] - r["base"]) * 100
        mark = RED if bp > 0 else GREEN
        return f"{mark}  `{r['ticker']}` ({r['name']})  |  {r['price']:.2f}%  |  {_sign(bp, 0, 'bp')} w/w"
    return _etf_line(r)


def _bp_line(label, ch, invert=True):
    mark = (RED if ch["bp_chg"] > 0 else GREEN) if invert else (GREEN if ch["bp_chg"] > 0 else RED)
    return f"{mark}  `{label}`  |  {ch['level']:.2f}%  |  {_sign(ch['bp_chg'], 0, 'bp')} w/w  |  as of {ch['obs_date']}"


def _chunk(blocks, header, lines, max_len=2900):
    chunk, size = [header], len(header)
    for ln in lines:
        if size + len(ln) + 1 > max_len and len(chunk) > 1:
            # A trailing `_Group_` header travels with its rows.
            carry = [chunk.pop()] if len(chunk) > 1 and chunk[-1].startswith("_") else []
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(chunk)}})
            chunk, size = carry, sum(len(c) + 1 for c in carry)
        chunk.append(ln)
        size += len(ln) + 1
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(chunk)}})


def format_slack(snap: dict) -> dict:
    cov = snap["coverage"]
    week = f"{snap['week_start']} to {snap['ref_date']}"
    blocks = [{"type": "header", "text": {"type": "plain_text",
               "text": f"Weekly price summary — week of {snap['week_start']}"}}]
    idx = next((g for g in snap["etf_groups"] if g["group"] == "US Indices"), None)
    lead = ", ".join(f"{r['name']} {_sign(r['week_pct'])}" for r in (idx or {}).get("rows", [])[:3])
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text":
        f"Closes {week} (ET)  ·  {cov['current']}/{cov['total']} names with a current weekly close"
        + (f"  ·  {lead}" if lead else "")}]})

    for tier, title in (("2sigma", "*Weekly 2σ+ moves* (whole watchlist)"),
                        ("1sigma", "*Weekly 1σ moves* (Core + position lists)")):
        hits = [r for r in snap["names"] if r.get("tier") == tier]
        if not hits:
            _chunk(blocks, title, ["_none this week_"])
            continue
        lines = []
        for cat, _ in ss.SUBCATEGORIES:
            in_cat = [r for r in hits if cat in r["categories"]]
            if in_cat:
                lines.append(f"_{cat}_")
                lines += [_name_line(r) for r in in_cat]
        _chunk(blocks, title, lines)

    blocks.append({"type": "divider"})
    lines = []
    for g in snap["etf_groups"][:1]:
        lines.append("_US Indices_")
        lines += [_etf_line(r) for r in g["rows"]]
    fred = snap.get("fred") or {}
    if snap["macro"] or fred.get("mortgage"):
        lines.append("_Macro_")
        lines += [_macro_line(r) for r in snap["macro"]]
        m = fred.get("mortgage")
        if m:
            lines.append(_bp_line("30Y Mortgage", {"level": m["level"], "bp_chg": m["bp_chg"],
                                                   "obs_date": m["obs_date"]}))
    if fred.get("curve"):
        lines.append("_Treasury Curve_")
        lines += [_bp_line(c["label"], c) for c in fred["curve"]]
    if fred.get("credit"):
        lines.append("_Credit_ (OAS w/w; widening = \U0001F7E5)")
        for c in fred["credit"]:
            o = c.get("oas")
            if o:
                lines.append(f"{RED if o['bp_chg'] > 0 else GREEN}  `{c['key']}` ({c['label']})  |  "
                             f"yield {c['yield']['level']:.2f}% {_sign(c['yield']['bp_chg'], 0, 'bp')}  |  "
                             f"OAS {o['level'] * 100:.0f}bp {_sign(o['bp_chg'], 0, 'bp')}")
    for g in snap["etf_groups"][1:]:
        if g["rows"]:
            lines.append(f"_{g['group']}_")
            lines += [_etf_line(r) for r in g["rows"]]
    _chunk(blocks, "*Index, Sector & Macro — week*", lines)

    foot = (f"Weekly z = week return vs the trailing-252-day daily distribution scaled by √n sessions. "
            f"Full sortable table: <{WEEKLY_PAGE_URL}|weekly summary page>.")
    if cov["missing"]:
        shown = ", ".join(cov["missing"][:15]) + (" …" if len(cov["missing"]) > 15 else "")
        foot += f"  No current weekly close for {len(cov['missing'])}: {shown}"
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": foot}]})
    return {"text": f"Weekly price summary — {week}", "blocks": blocks}


def post_status(text: str) -> None:
    url = os.environ.get("SLACK_STATUS_REPORTS_WEBHOOK")
    print(text)
    if url:
        try:
            requests.post(url, json={"text": text}, timeout=10)
        except requests.RequestException as e:
            print(f"[WARN] status post failed: {e}")


# ── HTML page ────────────────────────────────────────────────────────────────

def build_html(snap: dict) -> str:
    data = json.dumps(snap, separators=(",", ":")).replace("</", "<\\/")
    title = "Weekly Price Summary"
    return (_HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__DATA__", data))


_HTML_TEMPLATE = r"""<title>__TITLE__</title>
<style>
:root{--bg:#fbfaf8;--panel:#fff;--ink:#1f1d1a;--muted:#6b665e;--line:#e7e3dc;--up:#1f7a4d;--dn:#b3372e;--upbg:#e5f3ea;--dnbg:#f8e6e3;--chip:#f1eee8;--accent:#3a5a8c}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171614;--panel:#1f1e1b;--ink:#ece8e1;--muted:#a19b91;--line:#34312c;--up:#5cc98f;--dn:#f07b6f;--upbg:#1d3326;--dnbg:#3a211e;--chip:#2a2824;--accent:#8fb0e3}}
:root[data-theme=dark]{--bg:#171614;--panel:#1f1e1b;--ink:#ece8e1;--muted:#a19b91;--line:#34312c;--up:#5cc98f;--dn:#f07b6f;--upbg:#1d3326;--dnbg:#3a211e;--chip:#2a2824;--accent:#8fb0e3}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 10px;letter-spacing:.02em}
.sub{color:var(--muted);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:16px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tile .k{color:var(--muted);font-size:12px}.tile .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;overflow-x:auto}
.card h3{font-size:13px;margin:0 0 6px;color:var(--muted);font-weight:600}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-size:12px;color:var(--muted);font-weight:600;cursor:pointer;user-select:none}
td:first-child,th:first-child,td.l,th.l{text-align:left}
tr:last-child td{border-bottom:0}
.up{color:var(--up)}.dn{color:var(--dn)}
.pill{display:inline-block;padding:1px 6px;border-radius:6px;font-weight:600}
.pill.up{background:var(--upbg)}.pill.dn{background:var(--dnbg)}
.tk{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;background:var(--chip);padding:1px 5px;border-radius:4px}
.nm{color:var(--muted);font-size:12.5px;max-width:240px;overflow:hidden;text-overflow:ellipsis}
.bar{position:relative;width:90px;height:10px;display:inline-block;vertical-align:middle}
.bar i{position:absolute;top:0;height:10px;border-radius:2px}
.bar:before{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;border-left:1px solid var(--line)}
.ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}
select,input{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:5px 8px;font:inherit}
.note{color:var(--muted);font-size:12px;margin-top:8px}
@media (max-width:600px){.nm{max-width:120px}.bar{width:60px}}
</style>
<main>
<h1>Weekly Price Summary</h1>
<div class="sub" id="sub"></div>
<div class="tiles" id="tiles"></div>
<h2>Weekly sigma moves</h2>
<div class="grid" id="alerts"></div>
<h2>Indices, macro, rates and sectors — week</h2>
<div class="grid" id="mkt"></div>
<h2>Every covered name</h2>
<div class="card">
<div class="ctl"><select id="cat"></select><input id="q" placeholder="Filter ticker or name" size="18"><span class="sub" id="cnt"></span></div>
<table id="all"><thead><tr><th class="l" data-k="ticker">Ticker</th><th class="l" data-k="name">Name</th><th data-k="week_pct">Week</th><th></th><th data-k="z">z</th><th data-k="price">Price</th><th data-k="ytd_pct">YTD</th><th data-k="pct_of_high">% of 52w hi</th></tr></thead><tbody></tbody></table>
</div>
<p class="note" id="foot"></p>
</main>
<script>
const S=__DATA__;
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct=(x,d=2)=>x==null?"—":(x>0?"+":"")+x.toFixed(d)+"%";
const bp=x=>x==null?"—":(x>0?"+":"")+x.toFixed(0)+"bp";
const cls=(x,inv)=>x==null?"":((inv?-x:x)>0?"up":"dn");
const maxAbs=Math.max(5,...S.names.map(r=>Math.abs(r.week_pct)));
function bar(x){if(x==null)return"";const w=Math.min(45,Math.abs(x)/maxAbs*45);return `<span class="bar"><i style="${x>=0?`left:50%`:`right:50%`};width:${w}px;background:var(--${x>=0?"up":"dn"})"></i></span>`}
$("sub").textContent=`Closes ${S.week_start} to ${S.ref_date} (ET) · ${S.coverage.current} of ${S.coverage.total} names with a current weekly close · built ${S.generated_at}`;
const idx=(S.etf_groups.find(g=>g.group==="US Indices")||{rows:[]}).rows;
const tnx=S.macro.find(r=>r.style==="yield");
let tiles=idx.map(r=>`<div class="tile"><div class="k">${esc(r.name)} <span class="tk">${esc(r.ticker)}</span></div><div class="v ${cls(r.week_pct)}">${pct(r.week_pct)}</div><div class="k">YTD ${pct(r.ytd_pct)}</div></div>`);
if(tnx){const b=(tnx.price-tnx.base)*100;tiles.push(`<div class="tile"><div class="k">10Y Treasury yield</div><div class="v ${cls(b,1)}">${tnx.price.toFixed(2)}%</div><div class="k">${bp(b)} w/w</div></div>`)}
$("tiles").innerHTML=tiles.join("");
function rowsTable(rows,cols){return `<table><tbody>${rows.map(r=>`<tr>${cols.map(c=>c(r)).join("")}</tr>`).join("")}</tbody></table>`}
const nameCols=[r=>`<td><span class="tk">${esc(r.ticker)}</span></td>`,r=>`<td class="l nm">${esc(r.name)}</td>`,r=>`<td><span class="pill ${cls(r.week_pct)}">${pct(r.week_pct)}</span></td>`,r=>`<td>z ${r.z>0?"+":""}${r.z.toFixed(2)}</td>`,r=>`<td>YTD ${pct(r.ytd_pct)}</td>`];
let al="";
for(const [tier,title] of [["2sigma","2σ+ — whole watchlist"],["1sigma","1σ — Core + position lists"]]){
  const hits=S.names.filter(r=>r.tier===tier);
  const cats=[...new Set(hits.flatMap(r=>r.categories))];
  al+=`<div class="card"><h3>${title} (${hits.length})</h3>`+(hits.length?cats.map(c=>`<div class="sub" style="margin:6px 0 2px">${esc(c)}</div>`+rowsTable(hits.filter(r=>r.categories.includes(c)),nameCols)).join(""):`<div class="sub">None this week.</div>`)+`</div>`;
}
$("alerts").innerHTML=al;
const etfCols=[r=>`<td><span class="tk">${esc(r.ticker)}</span></td>`,r=>`<td class="l nm" title="${esc(r.weighting||"")}">${esc(r.name)}</td>`,r=>`<td><span class="pill ${cls(r.week_pct)}">${pct(r.week_pct)}</span></td>`,r=>`<td>${bar(r.week_pct)}</td>`,r=>`<td>YTD ${pct(r.ytd_pct)}</td>`];
let mk="";
const F=S.fred||{};
const macroRows=S.macro.map(r=>r.style==="yield"?{t:r.ticker,n:r.name,v:`${r.price.toFixed(2)}%`,c:bp((r.price-r.base)*100),k:cls(r.price-r.base,1)}:{t:r.ticker,n:r.name,v:r.price.toFixed(2),c:pct(r.week_pct),k:cls(r.week_pct)});
if(F.mortgage)macroRows.push({t:"30Y Mortgage",n:"Freddie Mac PMMS, as of "+F.mortgage.obs_date,v:F.mortgage.level.toFixed(2)+"%",c:bp(F.mortgage.bp_chg),k:cls(F.mortgage.bp_chg,1)});
(F.curve||[]).forEach(c=>macroRows.push({t:"UST "+c.label,n:"as of "+c.obs_date,v:c.level.toFixed(2)+"%",c:bp(c.bp_chg),k:cls(c.bp_chg,1)}));
(F.credit||[]).forEach(c=>c.oas&&macroRows.push({t:c.key+" OAS",n:c.label,v:(c.oas.level*100).toFixed(0)+"bp",c:bp(c.oas.bp_chg),k:cls(c.oas.bp_chg,1)}));
mk+=`<div class="card"><h3>Macro, rates and credit (rate/spread rise = red)</h3><table><tbody>${macroRows.map(r=>`<tr><td><span class="tk">${esc(r.t)}</span></td><td class="l nm">${esc(r.n)}</td><td>${r.v}</td><td><span class="pill ${r.k}">${r.c}</span></td></tr>`).join("")}</tbody></table></div>`;
for(const g of S.etf_groups){if(g.rows.length)mk+=`<div class="card"><h3>${esc(g.group)}</h3>${rowsTable(g.rows,etfCols)}</div>`}
$("mkt").innerHTML=mk;
const cats=["All covered names",...new Set(S.names.flatMap(r=>r.categories))];
$("cat").innerHTML=cats.map(c=>`<option>${esc(c)}</option>`).join("");
let sortK="week_pct",sortD=-1;
function render(){const c=$("cat").value,q=$("q").value.trim().toLowerCase();
  let rows=S.names.filter(r=>(c===cats[0]||r.categories.includes(c))&&(!q||r.ticker.toLowerCase().includes(q)||(r.name||"").toLowerCase().includes(q)));
  rows.sort((a,b)=>{const x=a[sortK],y=b[sortK];if(x==null)return 1;if(y==null)return -1;return (typeof x==="string"?x.localeCompare(y):x-y)*sortD});
  $("cnt").textContent=rows.length+" names";
  document.querySelector("#all tbody").innerHTML=rows.map(r=>`<tr><td><span class="tk">${esc(r.ticker)}</span></td><td class="l nm">${esc(r.name)}${r.through?` <span class="sub">thru ${esc(r.through)}</span>`:""}</td><td class="${cls(r.week_pct)}">${pct(r.week_pct)}</td><td>${bar(r.week_pct)}</td><td>${r.z>0?"+":""}${r.z.toFixed(2)}</td><td>${r.price.toFixed(2)}</td><td class="${cls(r.ytd_pct)}">${pct(r.ytd_pct)}</td><td>${r.pct_of_high==null?"—":r.pct_of_high.toFixed(0)+"%"}</td></tr>`).join("")}
document.querySelectorAll("#all th[data-k]").forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortD=k===sortK?-sortD:(k==="ticker"||k==="name"?1:-1);sortK=k;render()});
$("cat").onchange=render;$("q").oninput=render;render();
$("foot").textContent=`Weekly z = (week return − n·μ) / (σ·√n), with μ, σ the trailing-252-day daily return distribution and n the sessions traded this week; the √n scaling assumes independent daily returns. Prices are local-currency closes. `+(S.coverage.missing.length?`No current weekly close for ${S.coverage.missing.length}: ${S.coverage.missing.join(", ")}.`:"");
</script>
"""


# ── main ─────────────────────────────────────────────────────────────────────

def load_snapshot() -> dict | None:
    try:
        return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_html(snap: dict) -> None:
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(build_html(snap), encoding="utf-8")
    print(f"[OK] wrote {HTML_PATH}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true", help="print the payload; write and post nothing")
    p.add_argument("--html-only", action="store_true", help="rebuild the HTML from the committed snapshot")
    p.add_argument("--force", action="store_true", help="skip the last-session and delivered-digest gates")
    p.add_argument("--date", help="reference session YYYY-MM-DD (default: today ET)")
    a = p.parse_args(argv)

    if a.html_only:
        snap = load_snapshot()
        if not snap:
            print("[ERROR] no weekly snapshot to render")
            return 1
        write_html(snap)
        return 0

    ref = date.fromisoformat(a.date) if a.date else ss.today_et()
    cache = ss.load_cache() or {}
    if not a.force:
        if not is_last_session_of_week(ref):
            print(f"[INFO] {ref} is not the week's last NYSE session — no weekly summary")
            return 0
        if cache.get("published") != ss.published_marker("close", ref):
            print(f"[INFO] no delivered close digest for {ref} yet — weekly summary waits for it")
            return 0
        prior = load_snapshot()
        if prior and prior.get("ref_date") == ref.isoformat() and prior.get("posted"):
            print(f"[INFO] weekly summary for {ref} already posted — duplicate close run, nothing to do")
            return 0

    snap = build_snapshot(cache, ref)
    cov = snap["coverage"]
    frac = cov["current"] / cov["total"] if cov["total"] else 0.0
    print(f"[INFO] weekly coverage {cov['current']}/{cov['total']} ({frac:.1%})")
    payload = format_slack(snap)

    if a.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if frac < MIN_WEEKLY_COVERAGE:
        post_status(f":x: sigma-alert weekly summary for {ref} NOT posted: only "
                    f"{cov['current']}/{cov['total']} names ({frac:.0%}) have a current weekly close "
                    f"(floor {MIN_WEEKLY_COVERAGE:.0%}).")
        return 0

    url = os.environ.get("SLACK_WEBHOOK")
    if not url:
        print("[ERROR] SLACK_WEBHOOK not set — weekly summary not posted")
        return 1
    try:
        requests.post(url, json=payload, timeout=15).raise_for_status()
    except requests.RequestException as e:
        post_status(f":x: sigma-alert weekly summary for {ref} NOT posted: Slack rejected it ({type(e).__name__}).")
        return 1
    snap["posted"] = True
    tmp = SNAPSHOT_PATH.with_name("weekly_snapshot.tmp.json")
    tmp.write_text(json.dumps(snap, indent=1), encoding="utf-8")
    os.replace(tmp, SNAPSHOT_PATH)
    write_html(snap)
    print(f"[OK] weekly summary for {ref} posted ({len(payload['blocks'])} blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
