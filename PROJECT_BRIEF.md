# Project Brief — read this first (for reviewers, human or AI)

This file exists so a reviewer can (1) judge how close the project is to its
intended goal and (2) understand the key design decisions **before** giving
feedback. For mechanics — data flow, schemas, the Slack layout, cron offsets —
see `README.md` and `CLAUDE.md`; this brief does not re-describe them.

> When reviewing, weigh findings against the **success criteria** and the
> **non-goals / accepted tradeoffs** below — several "obvious improvements"
> (dedup between GH Actions and the local backstop; a real-time feed; regenerating
> `ticker_metadata.json` in CI) were considered and deliberately declined. Say so
> if you think a declined option is actually worth it, but engage with the stated
> rationale.

---

## 1. Intended goal (the "why")

Give the user an **automated, zero-attention "what moved today, and is it
unusual?" screen** for a large equity universe, delivered to Slack three times a
trading day. "Unusual" is defined statistically: each ticker's move is scored as
a z-score against its **own** trailing 52-week daily-return distribution, so a
3% move in a sleepy large-cap and a 3% move in a volatile biotech are graded
differently. The point is **signal from noise** — surface the moves worth a
human glance and suppress the rest.

The universe is deliberately two-tiered to match the user's attention budget:
- a **broad 2σ net** over the whole watchlist (S&P 500 + the healthcare
  universe) catching genuinely large moves, and
- a **narrower 1σ net** over only the names the user actively cares about
  (Coverage Manager `Core` names + the five Position lists), where a smaller move
  is still worth knowing because it's a name they hold or are researching.

Context: the user is a solo, part-time, **healthcare-focused** investor
automating signal-from-noise. sigma-alert is the daily market-pulse layer; its
covered-name metadata comes from **Coverage Manager** (CM), the local source of
truth. Beyond the alerts, each digest carries an **Index, Sector & Macro
Returns** block (macro/credit/indices/global/sectors/healthcare/tech/commodities)
so the alerts land against a same-message read of the cross-asset backdrop.

## 2. Success criteria — and current status

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Per-ticker statistical scoring (not flat % thresholds) | ✅ Done | `z = (today_return − μ)/σ` from trailing ~251 daily returns; covered in `tests/test_screener.py` |
| 2 | Two-tier net: 2σ broad + 1σ for covered names only | ✅ Done | 2σ fires whole watchlist; 1σ gated by `_is_one_sigma_eligible()` (CM `Core=="Y"` OR membership in any of the five Position lists) |
| 3 | Three runs/day (open / midday / close), automated | ✅ Done | `sigma-open/midday/close.yml` cron-fired Mon–Fri; EST-aligned UTC offsets so they never fire early |
| 4 | Don't alert on stale/holiday data | ✅ Done | Session-validation aborts the run with a warning if the latest bar isn't today's session (README "Data and timing") |
| 5 | Don't fire spurious alerts on splits/dividends | ✅ Done | yfinance `auto_adjust=True`; documented rationale in `CLAUDE.md` |
| 6 | Keep Yahoo API load low | ✅ Done | EOD writes `cache/distribution_cache.json`; morning open reuses μ/σ/52w/prior-year-close instead of re-downloading history |
| 7 | Company names + sector tags in alerts | ✅ Done | Loads CM `ticker_metadata.json`; S&P 500 gaps filled from Wikipedia `sources/sp500_names.json` |
| 8 | Survive dropped GH Actions cron events | ✅ Done | `sigma-watchdog.yml` hourly during market hours recovers dropped runs via `workflow_dispatch` + posts a heartbeat only when it recovers something |
| 9 | Cross-asset context block in every digest | ✅ Done | Macro (`^TNX`/DXY/WTI) + Credit (FRED HY/IG yields+OAS, no key) + Indices/Global/Sectors/Healthcare/Tech/Commodities; download-list union so it renders even if a watchlist sync drops the ETFs |
| 10 | Never silently corrupt CM-owned metadata | ✅ Done | CI never regenerates `ticker_metadata.json` (no access to CM's CSV); reads only. Wikipedia fallback merged into a deep copy so true CM gaps still surface |
| 11 | Reverse-channel metadata gaps back to CM | ✅ Done | EOD writes `cache/missing_metadata.json`; CM's `sigma_export` reads it from the sibling clone and surfaces gaps |
| 12 | Data-quality visibility | 🟡 Partial | Weekly skip report (`weekly_skip_report.py`, Fri 22:30 UTC) posts a trailing-7-day Block Kit digest to `#status-reports`. But there is **no per-run `health/v1` ok/partial/error heartbeat** like the rest of the fleet — see §5 |
| 13 | On-time delivery even when GH cron lags | 🟡 Partial | Optional Windows local backstop (`setup_local_runner.ps1`, read-only clone) fires open/midday/close locally; opt-in, requires the laptop to be on, and intentionally un-deduped (see §4) |

**Overall: the v1 goal is met and the screener is live.** Open items are
hardening/observability (the missing health heartbeat in #12, the opt-in nature
of #13), not missing core function.

## 3. Key design decisions (and why)

1. **Two thresholds, two universes.** The 1σ tier deliberately fires on a narrow
   set (CM Core + Position lists) — a 1σ move is too common to broadcast over the
   whole S&P 500, but is exactly the kind of small move worth knowing on a name
   you hold. The narrow filter is what keeps the lower threshold from being noise.
2. **CM owns `ticker_metadata.json`; sigma-alert is a pure reader of it.** CM's
   weekly build writes the file directly into this repo and pushes it. The runner
   has **no access to CM's source CSV**, so any CI attempt to regenerate it would
   corrupt it. The screener degrades gracefully if it's missing (no names, 1σ tier
   simply doesn't fire).
3. **EOD cache so the morning run is cheap.** The expensive full-history download
   happens once at close; the morning open run loads μ/σ + 52w range +
   prior-year-end close from `distribution_cache.json` and only needs today's
   opens. Cuts Yahoo calls dramatically and lets the open digest still render the
   range/YTD.
4. **EST-aligned UTC cron, accepting ~1h late in EDT.** Crons are pinned to
   EST offsets so a run **never fires before** the market event year-round; the
   tradeoff is that during EDT months runs land ~1h late. Correctness (never early
   on stale data) over punctuality.
5. **Watchdog over tighter scheduling.** GH cron is best-effort and silently drops
   events under load. Rather than fight that, an hourly watchdog detects and
   recovers dropped open/midday/close runs and announces only when it actually
   recovers one.
6. **Local backstop is read-only and un-deduped — on purpose.** The optional
   laptop runner refreshes a disposable clone, runs, posts, and never
   commits/pushes. It does **not** coordinate with GH Actions; duplicate alerts are
   accepted as harmless because the only requirement is "the alert always lands."
7. **Credit from FRED's public no-key CSV; warn-and-proceed.** HY/IG yields+OAS
   come from FRED constants in-code (no source file, no secret), bounded to a
   ~2-year window so the multi-decade CSV doesn't time out; if FRED is unreachable
   the `_Credit_` block is simply omitted rather than failing the digest.
8. **Yields/spreads are levels, not prices.** Macro and credit rows have bespoke
   formatters: the 10Y yield shows a YTD **basis-point** move off its year-start
   level (a % "return" on a yield is misleading), and credit YTD labels the yield
   and OAS moves separately to disambiguate.
9. **`etf_names.json` authoritatively overrides metadata for ETF tickers**
   (name replaced, sector/subsector cleared) because ETF tickers can collide with
   foreign equities in CM's universe (e.g. `DIA` = SPDR DJIA **and** DiaSorin on
   Borsa Italiana) — without the override an ETF could be mislabeled or
   mis-tagged into a 1σ sector.

## 4. Non-goals / accepted tradeoffs

- **Not** a real-time / streaming feed — three batch snapshots a day (open /
  midday / close) is the intended cadence.
- **No dedup** between GitHub Actions and the optional local backstop — duplicate
  alerts on a doubly-fired slot are accepted by design (the user explicitly opted
  out of coordination).
- **Not** a portfolio P&L or sizing tool — it scores moves and tags ownership; it
  does not track cost basis, shares, or returns-since-entry.
- **2σ moves outside every subcategory bucket are dropped**, not shown in an
  "Other" catch-all — intentional, to keep the broad tier from re-flooding with
  uncovered names. (The close-only 52-week hi/lo list *does* keep an
  `Uncategorized` catch-all; different surface, different goal.)
- **S&P 500 membership is a static repo-maintained list** (`sources/sp500.txt`,
  refreshed manually at quarterly reconstitution via `refresh_sp500.py`), not a
  live feed — accepted because reconstitutions are infrequent and dated.
- CI does **not** own/maintain the CM-sourced metadata files (§3.2) — out of scope
  by design.

## 5. Known gaps / candidate next steps (feedback welcome here)

- **No `health/v1` heartbeat.** Unlike the rest of the fleet (per
  `HEALTH_REPORTING.md`), the open/midday/close runs don't post an
  ok/partial/error heartbeat to `#status-reports`. The watchdog only speaks up
  when it *recovers* a run, and the weekly skip report only covers data-quality
  skips — so a run that fails outright on a normal day (and isn't a dropped-cron
  case the watchdog catches) could be silently absent. This is the gap most worth
  closing first.
- **Local backstop wake-time network race.** The Scheduled Tasks use
  `StartWhenAvailable`, so a catch-up fire on wake can hit Slack before DNS is up.
  Per the workspace convention, the Slack POST path would benefit from a
  retry-with-backoff wrapper (`scheduled_jobs_monitor`'s `_urlopen_retry` is the
  reference) rather than a bare urlopen.
- **Backstop is opt-in and requires the laptop on** — so "always on time" (§2 #13)
  only holds when the machine is awake; otherwise it falls back to GH cron timing.
- **S&P 500 list drift** between manual reconstitution refreshes — a name added
  mid-quarter won't be in the broad net until the next `refresh_sp500.py` run.
- **`core_watchlist.json` is a deprecated back-compat path** still loaded as a
  fallback; cleanup is pending until CM stops pushing it.

## 6. How to evaluate

- **Mechanics, schemas, Slack layout:** `README.md`; agent/gotcha notes:
  `CLAUDE.md`.
- **Core logic** — data download, distribution math, z-score, alert-tier gating,
  subcategory grouping, all Slack formatting — lives in one file:
  `scripts/sigma_screener.py`. The 1σ eligibility gate is
  `_is_one_sigma_eligible()`; subcategory order is the `SUBCATEGORIES` list;
  credit/macro formatting is `_format_credit_line()` / `_format_macro_line()`.
- **Other entry points:** `scripts/sync_watchlist.py` (merges `sources/*` →
  `watchlist.txt`), `scripts/weekly_skip_report.py` (Friday digest),
  `scripts/refresh_sp500.py` (quarterly S&P 500 + name-map refresh),
  `scripts/post_overview.py` (pinned `#stock-price-alerts` reference card).
- **Run a screen:** `python scripts/sigma_screener.py --mode {open|midday|close}`.
- **Tests:** `python -m pytest tests/ -q` — **~111 tests** across
  `tests/test_screener.py` (100) and `tests/test_refresh_sp500.py` (11); they
  cover date/session validation, cache freshness, distribution math, z-score,
  watchlist dedup, and alert thresholds, and do **not** require network/Slack.
- **Most useful feedback:** (a) is the 2σ/1σ split (and the narrow 1σ universe)
  the right noise/signal tradeoff, or should the covered tier be broader/tighter;
  (b) the observability gap in §5 — is a `health/v1` heartbeat the right fix and
  what counts as `partial`; (c) correctness edge cases in the distribution math
  and the cached-open YTD/prior-year baseline logic (year-boundary handling);
  (d) whether dropping unbucketed 2σ moves (§4) ever hides something worth seeing.
