# sigma-alert
> GitHub Actions stock screener that flags standard-deviation price moves vs. each ticker's trailing 52-week return distribution (2σ universe-wide, 1σ for covered names), posting tiered alerts to Slack.

- **Status:** live
- **Runtime/trigger:** Python via GitHub Actions (open 14:35, midday 17:35, close 21:30 UTC, Mon–Fri; hourly watchdog; weekly skip report Fri 22:30 UTC) · optional Windows local backstop
- **Reads:** yfinance OHLCV · Coverage Manager `ticker_metadata.json` + position lists · Wikipedia (S&P 500 reconstitution)
- **Writes:** Slack `#stock-price-alerts` (digests) · `#status-reports` (weekly skip report) · committed `cache/{distribution_cache,missing_metadata,skip_log}.json`
- **Run:** `python scripts/sigma_screener.py --mode {open|midday|close}`  ·  **Entry points:** `scripts/sigma_screener.py`, `scripts/sync_watchlist.py`, `scripts/weekly_skip_report.py`

GitHub Actions-based stock screener that flags standard-deviation price moves against each ticker's trailing 52-week daily return distribution. Alerts are posted to Slack with company names and sector tags.

## How it works

1. **Open check (~9:35–10:35 AM ET)** — compares today's opening price to the prior close to detect gap moves
2. **Midday check (~12:30–1:30 PM ET)** — compares the current price to the prior close for intraday moves
3. **Close check (~4:30–5:30 PM ET)** — compares today's closing price to the prior close for end-of-day moves

> Cron schedules use EST-aligned UTC offsets so they always fire after market open/close regardless of daylight saving time. During EDT months runs land ~1 hour later than the nominal time.
>
> A `Sigma Watchdog` workflow runs hourly during weekday market hours and recovers any of open/midday/close that GitHub Actions silently dropped (cron events on GitHub are best-effort and can be skipped during high load). When the watchdog has to recover a run, it posts a `:rotating_light: Sigma Watchdog` heartbeat to Slack so you know it happened.
4. For each ticker, a z-score is computed: `z = (today_return - μ) / σ` where μ and σ come from the trailing 251 daily returns
5. Alerts are split into two tiers in the Slack message:
   - **2σ+ Moves** — fires on the entire watchlist when `|z| ≥ 2.0` (3σ+ moves are flagged inline with a warning emoji)
   - **1σ Moves** — fires only on tickers tagged `Healthcare Services`, `MedTech`, or `PA` in the Coverage Manager CSV when `1.0 ≤ |z| < 2.0`. The narrow filter keeps the lower threshold from being noisy
6. **Close report only**: flags any ticker that hit a new 52-week high or low during the session
7. **Every digest**: an **Index, Sector & Macro Returns** section at the bottom. A `_Macro_` sub-header leads (10Y Treasury yield `^TNX` as `level% | ±bp`, US Dollar Index `DX-Y.NYB`, WTI crude `CL=F` — the cross-asset backdrop, from `sources/macro.txt`). Then broad-market indices (SPYM = S&P 500, DIA = Dow Jones, QQQ = Nasdaq 100, plus the `^W5000` Wilshire 5000 whole-market index) under `_Indices_`, then SPDR sector ETFs (XLV, XLK, XLE, …) under `_Sectors_`, then healthcare sub-sector ETFs (XBI, IBB, IHI, XHS, PPH, IHE) plus the `^DRG` NYSE Arca Pharmaceutical index under `_Healthcare_`, then tech-theme ETFs (`MAGS` Mag-7, `SMH`/`SOXX` semis, `IGV` software, `DTCR` data-center REITs, `AIQ` AI/tech) under `_Tech Themes_`. The index/sector/healthcare/tech groups sort by z-score descending; macro renders in a fixed rates→FX→commodity order. Each **ETF** row appends prior calendar year and YTD returns — e.g. `| 2025: +24.50% | YTD: +6.10%` (macro rows show level + day-change only). The screener unions `sources/index_etfs.txt`, `sources/sector_etfs.txt`, `sources/healthcare_etfs.txt`, `sources/tech_etfs.txt`, and `sources/macro.txt` into the download list at startup so the section keeps rendering even if a watchlist sync drops them from `watchlist.txt`. Today's column reuses already-downloaded data; the period columns come from a dedicated ~800-day ETF-only fetch (so the main 400-day batch download stays small)

Each alert line includes a direction marker (🟩 up / 🟥 down), ticker, short company name, percent move, z-score, price, 52-week low–high range, prior calendar-year return, and year-to-date return (vs the prior calendar year-end close):

```
🟩  *ISRG* (Intuitive Surgical)  |  +3.25%  |  z = +2.45  |  $485.20  |  52w: $310.40 - $502.10  |  2025: +24.50%  |  YTD: +18.40%
```

The YTD figure reuses the already-downloaded close history in the full-screen path (no extra download) and the cached prior year-end close in the morning cached-open path. The prior calendar-year return (`2025:`) needs the year-before-last's year-end close — older than the 400-day screen window — so the alert tickers are folded into the same dedicated ~800-day fetch that produces the ETF returns block. The prior-year column always renders on alert rows, showing `2025: N/A` when the baseline can't be located (e.g. a recent IPO); YTD is dropped only if its own baseline is unavailable.

Short names for S&P 500 tickers that Coverage Manager doesn't maintain come from `sources/sp500_names.json`, a Wikipedia-sourced fallback regenerated by `scripts/refresh_sp500.py`.

## Data and timing

- **Timezone**: All date logic uses `America/New_York` (ET). Slack timestamps display the actual ET time, not the runner's system clock.
- **Session validation**: Before computing signals, the screener verifies that the latest price bar is from today's trading session. If data is stale (e.g., market holiday, pre-open), the run aborts with a warning rather than silently alerting on old data.
- **Adjusted prices**: yfinance `auto_adjust=True` (the default) is used, so Close prices account for splits and dividends. This prevents spurious alerts on ex-dividend and split dates.
- **yfinance end date**: The download window uses `end = tomorrow` to work around yfinance's exclusive end-date behavior, ensuring today's bar is included.

## Caching

The EOD run saves each ticker's `{mu, sigma, sample_size, high_52w, low_52w, prior_year_end_close, prior_year_end_year}` to `cache/distribution_cache.json`. The morning open run loads this cache instead of re-downloading full history — it only needs today's opening prices. This dramatically reduces Yahoo Finance API calls, and lets the open digest render the 52-week range and YTD return without re-downloading history.

The EOD run also writes `cache/skip_log.json` — a 30-day rolling log of per-ticker skips with their reasons. It's consumed by the weekly skip report (see below) to surface data-quality issues.

## Setup

1. Create a Slack app with two incoming webhooks:
   - One pointed at your `#stock-price-alerts` channel (main sigma digests)
   - One pointed at your `#status-reports` channel (weekly skip report)
2. Add the webhook URLs as GitHub Actions secrets:
   - `SLACK_WEBHOOK` — for the main channel
   - `SLACK_STATUS_REPORTS_WEBHOOK` — for `#status-reports`
3. Edit `watchlist.txt` to add/remove tickers (one per line)
4. Push to GitHub — the cron schedules handle the rest

## Manual trigger

All workflows support `workflow_dispatch` so you can trigger them manually from the Actions tab for testing.

## Local backstop (optional)

GitHub Actions cron is best-effort — scheduled runs can land late or get skipped during high load. The `Sigma Watchdog` recovers *dropped* runs within the hour, but you can also have the screener fire **locally on your laptop** as an independent backstop so an alert still goes out (and goes out on time) whenever the machine is on.

The local runner is a **read-only consumer**: it refreshes a disposable clone to `origin/master`, runs the screener, and posts to Slack. It never commits or pushes. If GitHub Actions also fires the same slot you get two near-identical alerts — that duplication is intentional and harmless (the goal is that the alert always lands).

**One-time setup** (run from any PowerShell prompt):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\setup_local_runner.ps1 `
  -SlackWebhook "https://hooks.slack.com/services/XXX/YYY/ZZZ"
```

That script (`scripts/setup_local_runner.ps1`):

1. Clones sigma-alert to a dedicated directory **outside Dropbox** (default `%LOCALAPPDATA%\sigma-alert-runner`) so scheduled runs never touch your Dropbox dev tree.
2. Installs the Python dependencies.
3. Writes a gitignored `.env` holding the Slack webhook (the same value as the `SLACK_WEBHOOK` GitHub secret — it's not stored in the repo, so you paste it in once).
4. Registers three weekday Scheduled Tasks — `SigmaAlert-open` (09:40), `SigmaAlert-midday` (12:35), `SigmaAlert-close` (16:25), machine-time (assumed ET). Each is created with **`StartWhenAvailable`** (runs as soon as possible after a missed start — e.g. the laptop was asleep at the trigger time) and battery-friendly settings.

Each run is performed by `scripts/local_run.ps1 -Mode {open|midday|close}`, which you can also invoke by hand to smoke-test. Re-running the setup script is idempotent (updates the clone, rewrites `.env`, re-registers the tasks). To remove the tasks:

```powershell
'open','midday','close' | ForEach-Object { Unregister-ScheduledTask -TaskName "SigmaAlert-$_" -Confirm:$false }
```

> Times are picked to land within ~30 min of each market event. Open/midday/close fire at 09:40 / 12:35 / 16:25 ET; close is offset past 16:00 to let Yahoo post the official closing print. Since the runs are EST-aligned on GitHub's side but real-ET locally, the local run usually fires *first* during EDT months — but dedup isn't relied on, so order doesn't matter.

## Watchlist

The watchlist is built automatically from source files in `sources/`:

| Source | File | Description |
|--------|------|-------------|
| Healthcare Services | `sources/hc_services.txt` | Coverage Manager HC Services tickers |
| MedTech | `sources/medtech.txt` | Coverage Manager MedTech tickers |
| S&P 500 | `sources/sp500.txt` | S&P 500 constituents (update quarterly at reconstitution via `scripts/refresh_sp500.py`) |
| S&P 500 names | `sources/sp500_names.json` | Wikipedia-sourced ticker → company name map; used as a display fallback for S&P 500 tickers not maintained by Coverage Manager. Regenerated by `scripts/refresh_sp500.py` on every run |
| Macro | `sources/macro.txt` | Cross-asset macro tickers — 10Y Treasury yield (`^TNX`), US Dollar Index (`DX-Y.NYB`), WTI crude (`CL=F`). Render under `_Macro_` at the **top** of the returns section with a dedicated formatter (yields show level + bp change) |
| Index ETFs | `sources/index_etfs.txt` | Broad-market benchmarks (SPYM, DIA, QQQ, `^W5000` Wilshire 5000). Render under `_Indices_` in the **Index, Sector & Macro Returns** Slack section |
| Sector ETFs | `sources/sector_etfs.txt` | SPDR Select Sector ETFs (XLE, XLF, XLV, …). Render below the indices in the same section |
| Healthcare ETFs | `sources/healthcare_etfs.txt` | Healthcare sub-sector ETFs (XBI, IBB, IHI, XHS, PPH, IHE) plus the `^DRG` NYSE Arca Pharmaceutical index (caret index ticker, not an ETF). Render below the SPDR sectors under a `_Healthcare_` sub-header in the same section |
| Tech-theme ETFs | `sources/tech_etfs.txt` | Tech-theme ETFs — `MAGS` (Mag-7), `SMH`/`SOXX` (semis), `IGV` (software), `DTCR` (data-center REITs), `AIQ` (AI/tech). Render under a `_Tech Themes_` sub-header below `_Healthcare_`; names carry representative holdings |
| ETF names | `sources/etf_names.json` | `{TICKER: friendly name}` for index + sector + healthcare + macro tickers; pharma rows labeled by composition (large-cap vs diversified). Coverage Manager doesn't maintain this metadata, so display names live here and get merged into the metadata copy at startup |

### Syncing the watchlist

`scripts/sync_watchlist.py` merges all source files in `sources/` into `watchlist.txt`, de-duplicating across sources. It runs automatically via GitHub Actions:

- **On push** — whenever files in `sources/` change on master
- **Weekly** — every Monday at 8:00 AM ET as a drift check
- **Manual** — via `workflow_dispatch`

To update the watchlist, edit the relevant source file in `sources/` and push. The sync workflow will regenerate `watchlist.txt` and commit it.

You can also run the sync locally:

```bash
python scripts/sync_watchlist.py
```

### Ticker metadata

`ticker_metadata.json` is a `{TICKER: {"name", "sector", "subsector"}}` lookup that `sigma_screener.py` loads at startup. The screener uses it to:

- Show company names and sector tags in Slack alerts
- Filter the 1σ tier to Healthcare Services / MedTech / PA tickers only

**This file is owned by Coverage Manager.** Its `weekly-build` pipeline reads `coverage_universe_tickers.csv`, generates the metadata, writes it directly into this repo, and commits/pushes it (alongside `core_watchlist.json`) in a single commit. sigma-alert's CI does **not** regenerate it — the runner has no access to the Coverage Manager CSV, so any attempt to do so would corrupt the file.

If `ticker_metadata.json` is missing, the screener still runs — alerts just won't include company names or sector tags, and the 1σ tier won't fire because no ticker will match the sector filter.

### Core watchlist

`core_watchlist.json` is a `{TICKER: {buy_price, target_price, date_added, notes, name, sector, subsector}}` file also owned by Coverage Manager, pushed in the same commit as `ticker_metadata.json`. At startup the screener loads the key set; any alert whose ticker is on the list is tagged `on_watchlist=True` and renders under a "Core Watchlist" subcategory at the top of its sigma tier in the Slack digest. The file is optional — if missing, the Core Watchlist subcategory simply doesn't appear.

#### Reporting metadata gaps back to Coverage Manager

On the EOD close run, the screener checks every watchlist ticker against `ticker_metadata.json`. If any ticker is missing from the file (or has a blank `name`), it writes `cache/missing_metadata.json` listing the gaps; the close workflow commits that file alongside the distribution cache. Coverage Manager's weekly `sigma_export` step reads this flag from the sibling sigma-alert clone, logs a warning, and surfaces the missing tickers in its run summary so the operator can fix the source CSV.

> **Note on the Wikipedia name fallback:** `sources/sp500_names.json` is merged into a deep copy of the metadata at startup, not the original. That way the Wikipedia fill doesn't mask genuine CM coverage gaps — `write_missing_metadata_flag` still sees the pre-fallback metadata and reports true omissions.

## Weekly skip report

Every Friday at 22:30 UTC (one hour after the Friday close run), `sigma-weekly-skip-report.yml` runs `scripts/weekly_skip_report.py`, which reads `cache/skip_log.json`, computes:

- **Chronic skips** — tickers skipped on every run in the trailing 7 days
- **Reason breakdown** — aggregate counts by skip reason (`insufficient_history`, `distribution_nan`, `fallback_insufficient`, `fallback_exception`)
- **Unresolved at week's end** — tickers still failing in the most recent run
- **Daily timeline** — per-day skip count

...and posts a single Block Kit message to `#status-reports`. The message is always sent — if nothing was skipped, the message says so explicitly so the absence of a post means the workflow didn't run.

This is self-contained in sigma-alert — no Coverage Manager code is required to produce the report.

## Tests

```bash
python -m pytest tests/ -v
```

Covers: date validation, cache freshness, distribution math, z-score calculation, watchlist dedup, alert thresholds.
