# sigma-alert

GitHub Actions-based stock screener that flags 2+ standard deviation price moves against each ticker's trailing 52-week daily return distribution. Alerts are posted to Slack.

## How it works

1. **Open check (~9:35–10:35 AM ET)** — compares today's opening price to the prior close to detect gap moves
2. **Midday check (~12:30–1:30 PM ET)** — compares the current price to the prior close for intraday moves
3. **Close check (~4:30–5:30 PM ET)** — compares today's closing price to the prior close for end-of-day moves

> Cron schedules use EST-aligned UTC offsets so they always fire after market open/close regardless of daylight saving time. During EDT months runs land ~1 hour later than the nominal time.
4. For each ticker, a z-score is computed: `z = (today_return - μ) / σ` where μ and σ come from the trailing 251 daily returns
5. Any ticker with |z| ≥ 2.0 triggers a Slack alert
6. **Close report only**: flags any ticker that hit a new 52-week high or low during the session

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

### Syncing

`scripts/sync_watchlist.py` merges all source files into `watchlist.txt`, de-duplicating across sources. It runs automatically via GitHub Actions:

- **On push** — whenever files in `sources/` change on master
- **Weekly** — every Monday at 8:00 AM ET as a drift check
- **Manual** — via `workflow_dispatch`

To update the watchlist, edit the relevant source file in `sources/` and push. The sync workflow will regenerate `watchlist.txt` and commit it.

You can also run the sync locally:

```bash
python scripts/sync_watchlist.py
```

## Tests

```bash
python -m pytest tests/ -v
```

Covers: date validation, cache freshness, distribution math, z-score calculation, watchlist dedup, alert thresholds.
