# sigma-alert

GitHub Actions-based stock screener that flags 2+ standard deviation price moves against each ticker's trailing 52-week daily return distribution. Alerts are posted to Slack.

## How it works

1. **Open check (9:35 AM ET)** — compares today's opening price to the prior close to detect gap moves
2. **Close check (4:30 PM ET)** — compares today's closing price to the prior close for end-of-day moves
3. For each ticker, a z-score is computed: `z = (today_return - μ) / σ` where μ and σ come from the trailing 251 daily returns
4. Any ticker with |z| ≥ 2.0 triggers a Slack alert

## Caching

The EOD run saves each ticker's distribution parameters (μ, σ) to `cache/distribution_cache.json`. The morning open run loads this cache instead of re-downloading full history — it only needs today's opening prices. This dramatically reduces Yahoo Finance API calls.

## Setup

1. Create a Slack app with an incoming webhook pointed at your `#stock-price-alerts` channel
2. Add the webhook URL as a GitHub Actions secret named `SLACK_WEBHOOK`
3. Edit `watchlist.txt` to add/remove tickers (one per line)
4. Push to GitHub — the cron schedules handle the rest

## Manual trigger

Both workflows support `workflow_dispatch` so you can trigger them manually from the Actions tab for testing.

## Watchlist

Edit `watchlist.txt` — one ticker per line. Lines starting with `#` are ignored.
