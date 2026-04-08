# sigma-alert

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

Each alert line includes a direction marker (🟩 up / 🟥 down), the ticker, a short company name, and the sector tag, e.g.:

```
🟩  *ISRG* (Intuitive Surgical)  [MedTech]  |  z = +2.45  |  +3.25%
```

## Data and timing

- **Timezone**: All date logic uses `America/New_York` (ET). Slack timestamps display the actual ET time, not the runner's system clock.
- **Session validation**: Before computing signals, the screener verifies that the latest price bar is from today's trading session. If data is stale (e.g., market holiday, pre-open), the run aborts with a warning rather than silently alerting on old data.
- **Adjusted prices**: yfinance `auto_adjust=True` (the default) is used, so Close prices account for splits and dividends. This prevents spurious alerts on ex-dividend and split dates.
- **yfinance end date**: The download window uses `end = tomorrow` to work around yfinance's exclusive end-date behavior, ensuring today's bar is included.

## Caching

The EOD run saves each ticker's distribution parameters (μ, σ) to `cache/distribution_cache.json`. The morning open run loads this cache instead of re-downloading full history — it only needs today's opening prices. This dramatically reduces Yahoo Finance API calls.

## Setup

1. Create a Slack app with an incoming webhook pointed at your `#stock-price-alerts` channel
2. Add the webhook URL as a GitHub Actions secret named `SLACK_WEBHOOK`
3. Edit `watchlist.txt` to add/remove tickers (one per line)
4. Push to GitHub — the cron schedules handle the rest

## Manual trigger

All workflows support `workflow_dispatch` so you can trigger them manually from the Actions tab for testing.

## Watchlist

The watchlist is built automatically from source files in `sources/`:

| Source | File | Description |
|--------|------|-------------|
| Healthcare Services | `sources/hc_services.txt` | Coverage Manager HC Services tickers |
| MedTech | `sources/medtech.txt` | Coverage Manager MedTech tickers |
| S&P 500 | `sources/sp500.txt` | S&P 500 constituents (update quarterly at reconstitution) |
| Sector ETFs | `sources/sector_etfs.txt` | SPDR Select Sector ETFs + SPYM S&P 500 portfolio ETF |

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

**This file is owned by Coverage Manager.** Its `weekly-build` pipeline reads `coverage_universe_tickers.csv`, generates the metadata, writes it directly into this repo, and commits/pushes only that file. sigma-alert's CI does **not** regenerate it — the runner has no access to the Coverage Manager CSV, so any attempt to do so would corrupt the file.

If `ticker_metadata.json` is missing, the screener still runs — alerts just won't include company names or sector tags, and the 1σ tier won't fire because no ticker will match the sector filter.

#### Reporting metadata gaps back to Coverage Manager

On the EOD close run, the screener checks every watchlist ticker against `ticker_metadata.json`. If any ticker is missing from the file (or has a blank `name`), it writes `cache/missing_metadata.json` listing the gaps; the close workflow commits that file alongside the distribution cache. Coverage Manager's weekly `sigma_export` step reads this flag from the sibling sigma-alert clone, logs a warning, and surfaces the missing tickers in its run summary so the operator can fix the source CSV.

## Tests

```bash
python -m pytest tests/ -v
```

Covers: date validation, cache freshness, distribution math, z-score calculation, watchlist dedup, alert thresholds.
