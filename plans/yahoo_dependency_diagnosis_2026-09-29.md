# Yahoo dependency diagnosis — sigma-alert (#327/#292) + post_earnings_movers (#376)

Written 2026-09-29 (overnight session). Diagnosis and plan only; no code changed.

## Bottom line

**Yahoo throttling is not the main cause.** Every close run in the sample got a full batch: 4 failed
downloads per run out of 756–772 tickers, and all 4 are dead symbols (`CSU`, `AMBUB.CO`, `SECTB.ST`, `FI`).
There was zero retry, zero "degraded after retries" and zero rate-limit text across 70 run logs.
The alerts say *"Yahoo throttling?"* because that is the guess hardcoded into the heartbeat
(`sigma_screener.py:2967`), not something the run measured.

Three different causes produced the symptoms:

| # | Symptom | Cause | Share |
|---|---|---|---|
| A | sigma close "0/758", "0/756", "0.1%" | **The close ran twice in one day.** The second run finds every bar already scored and counts that as missing data. | 11 of the 12 sub-floor close runs, 2026-09-09 to 2026-09-28. The 09-21 second run was both a duplicate AND a case of B |
| B | sigma close 2026-09-28 "39/756 (5.2%)" and 2026-09-21 second run (740 on the Friday bar) | **Yahoo served history without the current Monday's bar** for US equities in the evening (20:11 and 21:04 ET). Foreign names, indices and futures did get it. | 2 runs, both Mondays. Only 09-28 lost data; on 09-21 the first run had already scored the day |
| C | post_earnings_movers 2026-09-07 "empty dataset" | **The run fired on Labor Day, while the laptop was in Modern Standby.** yfinance's reason for the failure was captured and thrown away. | 1 run. It is the only occurrence of that warning in Slack history since go-live on 2026-07-22 |

## Evidence

### Where each lane fetches

- **sigma-alert** (`scripts/sigma_screener.py`):
  - The close and midday modes make one `yf.download(all ~756 symbols, start=today-400d, end=today+1, threads=True)` call (`batch_download`, line 1249).
  - That call retries at 15 s and 45 s when fewer than 60% of tickers come back, then keeps the best attempt.
  - The per-ticker `fallback_download_single` runs only when a ticker raises. It does **not** run when a ticker is merely missing today's bar.
  - Where it runs: GitHub Actions `ubuntu-latest`, yfinance **1.7.0**, pandas 3.0.6, plus a local read-only backstop (`SigmaAlert-close`, 16:25 ET, never commits).
  - There is a second ~800-day download for ETF period returns (line 2219).
- **post_earnings_movers** (`pem/prices.py`):
  - It reads the Coverage Manager close-only cache first (`Coverage Manager/cache/prices`).
  - It then makes one `yf.download(reporters, start=earliest_event-10d, end=run_date+1, threads=True, group_by="ticker")` call, with stdout and stderr **redirected into a buffer that is discarded** (lines 167–182).
  - Where it runs: local Windows task `PostEarningsMovers-Daily`, Mon–Fri 17:30 ET, `WakeToRun=True`, `StartWhenAvailable=True`.

### sigma-alert: coverage per run (GitHub Actions logs, 70 runs, 2026-08-06 to 2026-09-28)

This excerpt covers the close cycle only. `stale` counts the tickers skipped because their latest bar had already been scored. Full parse script: scratchpad `parse.py`.

| Close run (ET) | Trigger | Stale skips | Latest bar on stale tickers | Coverage |
|---|---|---|---|---|
| 09-10 18:51 | watchdog dispatch | 0 | — | full |
| 09-10 19:15 | lagged cron | 753 | 728 = **today** (already scored) | 0.0% → suppressed, `error` |
| 09-11 18:53 / 19:23 | dispatch / cron | 0 / 753 | today | full / 0.0% |
| 09-14 18:50 / 19:54 | dispatch / cron | 1 / 753 | today | full / 0.0% |
| 09-15, 16, 17, 18, 23, 24, 25 | same pattern | ~0–21 / ~751 | today | full / 0.0–0.1% |
| 09-21 18:55 | dispatch | 6 | 09-18 | full |
| 09-21 20:11 | lagged cron | 751 | **740 = 09-18 (Friday)** | 0.0% |
| 09-22 19:39 | cron only | 9 | — | full |
| 09-28 21:04 | cron only (no dispatch) | 712 | **712 = 09-25 (Friday)** | **5.2% (39/756)** |
| 08-06 21:02, 08-26 20:54, 08-28 23:12, 08-31 20:40 (Mon) | cron | 9–27 | — | full |

**Mechanism A (duplicate close):**

- The GitHub close cron (`30 21 * * 1-5`, i.e. 17:30 EDT) fires **1h45m to 3h35m late**; it has fired between 19:15 and 21:04 ET every day in the sample.
- `sigma-watchdog.yml` runs hourly. It sees no close run yet and dispatches one at ~17:50–18:55 ET. That run scores the day and commits the watermark (`last_bar`).
- The late scheduled run then gets today's bar for 728 names. `is_unscored_bar` correctly refuses them (`latest_bar == last_scored`) and logs them as `stale_bar`.
- `screen_coverage` counts "already scored" as "no data" and falls below the 80% floor. The run then posts an `error` heartbeat: *"market data returned for only 0/758 … (Yahoo throttling?)"*.
- Row #292's 2026-08-10 "0 of 772 after an earlier 767/772" has the same shape.
- This is `feedback_an_alert_is_the_guard_explaining_itself`: the diagnosis came from the alert text, and the text is a hardcoded guess.

**Collateral damage from A, found here and not on the board:**

- The duplicate run **overwrites that day's `cache/skip_log.json` entry**.
- The file now shows 755–758 skips per close for 09-15 through 09-25 (09-22 is the only honest day, with 14). The real first-run skips were about 0–25.
- Coverage Manager's weekly skip report reads this file, so its skip statistics are inflated roughly 30×.

**Mechanism B (a Monday bar missing in the evening):**

- On 2026-09-21 the 18:55 ET run received the Monday bar. At 20:11 ET, Yahoo returned history ending Friday 09-18 for 740 names.
- On 2026-09-28, the only run was at 21:04 ET, and 712 US names ended at Friday 09-25.
- The 18 names that did carry 09-28 were Brazil (`.SA`, `RDOR3`), Tokyo (`.T`), ASX, HK, `^RUT`, `^DRG`, `^TNX`, `DX-Y.NYB`, `CL=F`, `HUBB` and `YPSN`.
- The batch was not empty and not partial by count, so the 60%-coverage retry never fired. Every ticker had bars; they were just missing the newest one.
- **Real loss:** the 09-28 session was never scored for 712 names, and the next close scores 09-29 against 09-28, so Monday 2026-09-28 moves produced no sigma alerts at all.
- The time of day alone does not explain it. Four other close runs after 20:00 ET (including Monday 08-31 at 20:40) were fine. Both bad runs are **recent Mondays** after 20:00 ET. n = 2, so this is a hypothesis.

### Reproduction from here (2026-09-29 00:10 ET, 4 tiny batches, 50 of sigma-alert's US tickers)

| Mode | Result |
|---|---|
| Lane settings: `start/end` 400 d, `threads=True` (yfinance 1.4.1 local) | 49/49 with the 09-28 bar; `FRE` failed (known dead) — 3.4 s |
| `threads=False` | identical — 3.0 s |
| 5 chunks of 10, `threads=False` | identical — 3.4 s |
| Lane settings under **yfinance 1.7.0** (CI's version, isolated `--target` install) | identical — 2.4 s |

By midnight, Yahoo had backfilled the 09-28 bars. Thread count, chunk size and yfinance version make no difference when Yahoo is healthy. Threading is not the cause, and chunking would not have helped with A or B.

### post_earnings_movers, 2026-09-07

- 2026-09-07 was **Labor Day** (market closed), which is why "none resolved" is expected: no reaction bar exists.
- Windows System log: Kernel-Power **506 (entered Modern Standby) at 17:25:13 ET, 507 (exit) at 18:19:11 ET**. The 17:30 task ran while the machine was in connected standby. Its Slack status post did reach Slack at 17:30:27, so the network was at least partly up.
- The same call shape today (21 large caps, `2026-08-28 → 2026-09-08`) returns 6 rows, not empty. A holiday end date does not cause an empty frame.
- An all-tickers-failed batch returns an **empty DataFrame, not an exception**; the reason is printed to stdout. `_batched_yf_download` sends stdout to `buf` and never reads it, so the actual Yahoo or network error is lost for good.
- Ranked hypotheses:
  1. The first HTTPS call after standby failed: yfinance's cookie/crumb fetch fails, so every ticker fails (`feedback_wake_network_race`).
  2. A transient Yahoo error.
- What would distinguish them: the discarded `buf`, which is the fix below.
- Throughout, PEM's CM-cache fallback worked as designed.

## Recommended fix

### sigma-alert (fixes A in full; contains B)

1. **Make the close run idempotent: "already scored today" is not "missing data".**
   - Classify a `stale_bar` skip where `latest_bar == last_scored_bar == today` as `already_scored`, a separate reason.
   - If already-scored plus screened is at least the floor, the run is a **duplicate**. It should post an `ok` heartbeat saying "duplicate close — today already scored", with no digest and no return-map rewrite. Suppressing a second digest is correct; the first one already went out.
   - It must **not** replace that day's `skip_log.json` entry. Merge into the entry or leave it untouched.
   - Doing it in code, rather than stopping the duplicate, also covers the local-runner and manual-dispatch duplicates.
2. **Re-fetch only the tickers whose bar is behind.** This applies when the batch index carries today's date but more than 20% of tickers' latest bar is older than today.
   - Wait about 60 s, then run one `yf.download(stale_subset, period="5d")`. That is a different request shape, which yields today's bar and the previous close, enough to score today against the cached μ/σ, the way the cached-open path already does.
   - If still behind, fall back to **FMP `stable/quote`** per symbol for US names. It is on the paid Starter tier; 700 calls fit inside the 300/min limit in about 2.5 min.
   - This needs `FMP_API_KEY` as a GitHub secret, which is a cost/config decision. Stage it behind the Yahoo re-fetch, so FMP is spent only on a genuinely bad evening.
3. **Name the cause in the heartbeat.** Replace the hardcoded "(Yahoo throttling?)" with the measured breakdown: failed-download count, already-scored count, and the stale-bar date histogram. Then the next alert says "712 names' newest bar is 09-25" instead of guessing.
4. **Measure B before building more.** Run a cheap probe on Monday 2026-10-05, 18:00–22:00 ET, every 30 min: 20 US tickers under the lane's call shape, logging the newest bar per ticker.
   - If the Monday bar disappears after about 20:00 ET, the lagged cron is the exposure. Add a cron entry an hour earlier (`30 20 * * 1-5`, still after the 16:00 close in both EST and EDT) so the first run lands before the window.
   - If it does not disappear, 09-21 and 09-28 were one-off Yahoo backend events, and step 2 is sufficient.

### post_earnings_movers (C)

5. **Stop discarding `buf`.** When the frame is empty, append the last few lines of `buf` to `yf_err`. This is one line of code and turns the next occurrence into a diagnosis.
6. **Retry once after 30–60 s when the batch comes back empty**, which covers the wake race. Optionally add a network-ready pre-check. Leave the rest alone: the CM-cache fallback and the refusal to post a thin digest both behaved correctly.

### Rejected

| Alternative | Why rejected |
|---|---|
| Throttle yfinance with smaller chunks, `threads=False` or sleeps | The failures were not throttling (0 retries and 0 rate-limit messages in 70 logs), and chunking or `threads=False` returned the same data as the lane's settings in testing here |
| Switch sigma-alert wholesale to FMP | Batch/bulk EOD is gated on Starter, so it means 756 per-symbol 400-day history calls per run, three times a day. FMP coverage of the foreign book (`.T`, `.SA`, `.ST`, `.CO`) is sparse per the endpoint map, and Yahoo was healthy for more than 95% of runs |
| Use Coverage Manager's price cache in sigma-alert | The cache is close-only, local to the laptop (unreachable from GitHub Actions) and not refreshed intraday. It cannot provide today's bar, which is exactly what B lacks |
| Remove the watchdog to kill duplicates | The watchdog exists because the GitHub cron is hours late or dropped. Removing it trades a false alarm for a missed day. Idempotence (fix 1) fixes the class and keeps recovery |
| Lower the 80% floor | The floor correctly refused the 09-28 hollow digest. The defect is what gets counted against it |

## Open items for the implementing session

- Confirm the `skip_log.json` overwrite path in `update_skip_log` (the observed effect is certain; the exact line was not read).
- Correct the false premises on the board: #292/#327 attribute the 0% runs to Yahoo, and #376 treats the 09-07 run as a Yahoo outage when it was a holiday run in Modern Standby.
- The local runner (`SigmaAlert-close` 16:25 ET) posts to Slack without a watermark of its own, which is a third writer. This is out of scope here.

## Fable plan review (2026-09-29)

**Verdict: APPROVE WITH CHANGES.** The diagnosis holds on all three evidence claims; the scoped
plan (fix 1, fix 2 free half, fix 3, fix 4) is right in direction but under-specified on the one
rule everything hangs on — *what "already scored today" means* — and it misses two things the
logs show: the weekly skip report has been **dead for three Fridays** (a 400 from Slack on 755
chips), and every evening heartbeat this month was **filed under the wrong cycle date**.
Findings and an ordered build plan follow. No code was changed by this review.

### Evidence check (verified against `gh run list`, six run logs, git history of `cache/`)

| Claim | Verdict | What the record shows |
|---|---|---|
| (a) The <80% close runs are late scheduled runs after a watchdog dispatch already scored the day | **Confirmed, and it extends further back than the table.** | Every weekday 09-10 → 09-25 except 09-22 has a `workflow_dispatch` run at 21:47–22:56 UTC followed by a `schedule` run at 23:15–23:58 UTC (09-21: 00:11 UTC next day). The same pair exists on **09-03, 09-04, 09-07, 09-08** (`skip_log` shows 764/752/752/753 stale) — i.e. the "0/769, 0/757" runs that opened board #327 were duplicates too. In the 09-23 19:48 ET second run, **725 of 756 names had `latest=09-23, scored=09-23`** — the bar was there; it was already scored. 09-28 is the one genuine loss: `712 × latest=09-25, scored=09-25`, then `Cache saved with 752`, `error`. 11 of 12 sub-floor runs since 09-09 are duplicates; 16 of 17 since 09-03. |
| (b) The duplicate overwrites that day's `skip_log.json` entry, inflating the weekly report | **Confirmed — and understated.** | `update_skip_log` drops the existing `(today, mode)` entry and replaces it (`scripts/sigma_screener.py:1183-1192`, comment: *"so re-runs overwrite cleanly"*). The honest first-run entries still exist in git: `f76ce56` 09-10 = **5** skips, `8dd77e7` 09-15 = **5**, `604878f` 09-21 = **11**, `c575602` 09-25 = **25**; the committed file now says 758/758/756/755. **Consequence not in the doc:** `sigma-weekly-skip-report.yml` has **failed on 09-12, 09-19 and 09-26** (runs 34661862218, 35409620915, 36206126168) — `chronic=755 unresolved=755` then `Slack webhook failed: 400`. The report died on exactly the data it exists to surface, and nothing posts on that workflow's failure. **Misattribution:** the consumer is sigma-alert's own `scripts/weekly_skip_report.py`, not Coverage Manager — `rg --no-ignore skip_log` over Coverage Manager finds only codex_feedback quotes. The "Coverage Manager's weekly report" wording comes from stale comments at `sigma_screener.py:129` and `:1157`. |
| (c) A Monday bar goes missing in the evening | **Weakly supported; "Monday" is not the discriminator.** | Big regressions: 09-21 20:11 ET (712 names on the Friday bar, after the 18:55 run had Monday's) and 09-28 21:04 ET (712). But the evening runs also show **small** regressions on non-Mondays: 09-23 Wed 19:48 → **17** names back on 09-22; 09-25 Fri 19:58 → 3; 09-14 Mon 19:54 → 3. And 08-26 20:54, 08-31 (Mon) 20:40, 08-06 21:02 were clean. Best fit is *"after ~20:00 ET, since mid-September"*, n=2 large + 3 small. Treat as a hypothesis; the probe (doc step 4, not tonight) should run **every weekday evening**, not Mondays only. Side note confirming the guard works: the 6 `.T` names skipped on 09-21 18:55 were correct — 2026-09-21 was a Tokyo holiday (Respect for the Aged Day). |

Also verified: the 60%-coverage retry never fired in any of the six logs (4 failed downloads per run, all dead symbols); `_batched_yf_download` in `post_earnings_movers/pem/prices.py:163-186` captures `buf` and never reads it; an empty frame returns the fixed string at `:182`.

### Findings

**Critical**

- **C1 — The "already scored" rule as written (`latest == last_scored == today`) misclassifies both real cases.** (`sigma_screener.py:1791-1795`, doc fix 1 bullet 1.) On the 09-21 20:11 run the 712 names had `latest=09-18, last_scored=09-21` — `latest != today`, so the doc's rule leaves them `stale_bar` and the run still posts `error` although the day was scored. On a genuine loss the rule is vacuous the other way. The bar Yahoo returns *now* is irrelevant to whether today was scored; only our watermark is. Rule: **`already_scored_today := cache.tickers[t].last_bar >= today_et()`** — read the *scored* watermark (`last_bar`), not `prior_bars_from_cache()` which merges `last_seen` (`:1663-1691`). **`behind := latest_bar <= last_scored < today`** (nothing new AND today's session not scored) — the chronically-late European book falls here on every run (22 names on 09-21, correctly scored the next day with Friday's bar) and so did the 712 on 09-28. Duplicate := `(screened + already_scored_today) / total >= MIN_SCREEN_COVERAGE`. Check on the record: 09-21 20:11 → 723/756 = 95.6% → duplicate/`ok`; 09-23 19:48 → 743/756 → duplicate; 09-28 21:04 → 0 + 39 → 5.2% → `error` with a histogram saying `712 × newest bar 2026-09-25`. A legitimately partial first run stays partial on the second run's card because the status thresholds are unchanged, just applied to the right numerator.

**High**

- **H1 — Every close heartbeat after 20:00 EDT is filed under the next day's cycle.** `build_health_payload` stamps `cycle:` with naive `datetime.now()` (`sigma_screener.py:2958`); the runner is UTC, so the 09-28 21:04 ET `error` card says `cycle: 2026-09-29 close`, and all 11 duplicate cards this month carry tomorrow's date. §4.3 of `HEALTH_REPORTING.md` keys reruns on `cycle` + `attempt`, so attempt 2 of a day currently cannot even share a cycle with attempt 1. Use `today_et()`; add `attempt: 2 (duplicate — today already scored by an earlier run)` on the duplicate card. One-line fix, but it is the label the fleet monitor files by.
- **H2 — A duplicate run that screens late-arriving names loses their alerts silently.** Today: `save_cache` and `update_skip_log` run at `:3238-3251`, *before* `enforce_publish_gate` at `:3317`, so a run refused at the gate still advances the watermark — the session can never be re-alerted (that is exactly why 09-28 is unrecoverable). Under the plan as written, a duplicate run with a partial first run behind it (first run 85%, second run gets the missing 15%) advances 15% of watermarks and posts nothing. The small evening regressions above (17 names on 09-23) are the everyday version: names the *first* run skips as behind and the *second* run then scores. Resolve: in the duplicate branch, publish iff `screened > 0 and (alerts or hi_lo_hits)`, via the normal `format_slack_message` with a one-line `attempt 2 — late bars for N names` banner; the return-map gate already refuses the rewrite (ETF coverage <80%). If that is not built tonight, the minimum is that the heartbeat states `K alerts from N late names were NOT posted` — the loss must be visible.
- **H3 — No concurrency group and no refresh-before-act on `sigma-close.yml`.** The two runs have been ≥24 min apart in September (dispatch 21:30:05 vs cron 23:21 on 09-03 was the closest), so the second checkout has always contained the first run's commit — the idempotence check only works because of that luck. If the cron fires promptly at 21:30 while a watchdog dispatch is running, both read the pre-commit cache, both score, both post a full digest, and the second `git push` fails non-ff → the `if: failure()` step posts a **crash card for a run that published**. A concurrency group alone does not fix this (`feedback_a_concurrency_group_does_not_freeze_the_checkout`: the queued run still checks out the sha it was queued with). Add both: `concurrency: {group: sigma-close, cancel-in-progress: false}` and, before the screener step, `git fetch --depth=1 origin master && git reset --hard FETCH_HEAD` (the memory's verified recipe), so the serialized second run reads the first run's watermark. Three lines of YAML; do it in the same commit as fix 1 or fix 1 is only probabilistically correct.

**Medium**

- **M1 — The weekly skip report cannot post the thing it is for.** `weekly_skip_report.py:336-390` joins every chronic/unresolved chip into one section with no chunking or cap; 755 chips → Slack 400 → three consecutive failed Fridays, silent. Repairing the history (M2) fixes it for now; the report still needs a cap (`top 30 + "and N more"`) or the screener's `_append_section_chunked` pattern, and the workflow needs the same `if: failure()` backstop the cycle workflows have. The 09-28 entry (717, honest) will otherwise blow it up again on 10-02.
- **M2 — Repair the inflated history from git, mechanically.** For each duplicated date take the entry from the *earlier* "Update distribution cache" commit of that day: 09-03 `4ab71e7`, 09-04 `a75fb61`, 09-07 `2c6b6c2` (Labor Day — first run legitimately shows ~719 stale; keep it, it is honest), 09-08 `b3dd680`, 09-10 `f76ce56`, 09-11 `32581f6`, 09-14 `6d65f67`, 09-15 `8dd77e7`, 09-16 `5b1fb8c`, 09-17 `0a04d58`, 09-18 `0f56916`, 09-21 `604878f`, 09-23 `a6c4c3a`, 09-24 `b6b653d`, 09-25 `c575602`. Leave 09-09, 09-22 (honest, cron-only) and 09-28 (honest loss). One script, one commit, and the next Friday report is readable.
- **M3 — The re-fetch is a hypothesis; build it so it measures itself.** yfinance has no in-process price cache (cookie/crumb only), so the call does go out, and `period="5d"` sends `range=5d` instead of `period1/period2` — a different URL. Whether Yahoo answers differently is untested: the midnight reproduction ran when Yahoo was healthy. So: count `recovered / attempted` and put it in the heartbeat histogram (fix 3), so the first bad evening tells us whether the free half works before anyone spends the FMP half. Also **merge the 5-day frame onto the 400-day series** (concat, drop duplicate index, sort) and run `_process_ticker_full` normally — the doc's "score against cached μ/σ" path would skip the 52w range, the prior-year-end close and the cache entry that close mode exists to write.
- **M4 — Re-fetch trigger must count `behind`, not "latest < today".** On a duplicate run (C1) the 723 already-scored names also have `latest < today` (09-21 case) and would trigger a pointless 60 s wait + 756-symbol call. Trigger on `len(behind) / total > 0.20` with `behind` per C1. Note the 17-name regression on 09-23 is under the threshold and is served by H2 instead, not by the re-fetch.
- **M5 — Skip-log merge semantics for the duplicate.** "Leave untouched" is right when `screened == 0`. When the duplicate screened late names, the day's entry should be `earlier.skipped − tickers screened this run`; never add the duplicate's own `behind` list (those 22 European names were scored today with Friday's bar and are not skips). When there is no earlier entry (first run's commit missing), write normally.
- **M6 — The watchdog is date-keyed on UTC.** `sigma-watchdog.yml` uses `TODAY=$(date -u +%F)` and `gh run list --created "$TODAY"`. A close run created after 00:00 UTC (20:00 EDT) belongs to *tomorrow*: it is invisible for today (no recovery possible after 20:00 ET — on 09-28 the 21:00 and 23:00 UTC watchdog runs were dropped, the 00:43 UTC one saw "not yet due", and nothing recovered the day) and it **masks tomorrow's miss** (on 09-22 the 09-21 00:11 UTC run counted as "today's", so no dispatch was attempted). Not tonight's scope; the fix is `TZ=America/New_York` for `TODAY` plus filtering `createdAt` in ET. Fix 1 makes the duplicates harmless, so the watchdog can stay exactly as aggressive as it is.

**Low**

- **L1 — Duplicate branch should not `save_cache`.** With `screened == 0`, the carried-forward cache differs only by `last_seen` fields → a no-op commit every evening. Skip the save; `git diff --cached --quiet` then makes no commit.
- **L2 — Holiday evenings will read as `error` with the histogram** (e.g. `740 × newest bar 2026-09-04` on Labor Day). Correct per §4.2 only if the counters say why; a market-calendar check is a later item (post_earnings_movers has one in `config.session_closed`).
- **L3 — A session missed by every run of its day is unrecoverable by design.** The watermark scores only the newest bar, so on 09-29 the 732 names at `last_bar=09-25` will score 09-29 vs 09-28 and Monday's move is gone. Multi-session catch-up (score each unscored session between watermark and newest bar) is the deterministic fix for B if the re-fetch measures zero recovery. Separate row.
- **L4 — Midday duplicates post two identical digests** (no watermark write in midday mode). Accepted per JP's "duplicate alerts are fine"; noting it so nobody re-diagnoses it as a bug.
- **L5 — Fix 4 detail (post_earnings_movers).** Append the tail of `buf` to `yf_err` at `pem/prices.py:182` (and on the exception path). Retry once after 30–60 s **only** when the frame is empty or the call raised, never on a healthy run; a retry that succeeds must leave `yf_batch_ok=True` so `price_outage` (`pem/cli.py:333`) does not fire. The wake race clears in 10–30 s per `feedback_wake_network_race`, so 45 s is enough.

### Ordered build plan

1. **Workflow first (H3):** `sigma-close.yml` — add the concurrency group and the `fetch + reset --hard FETCH_HEAD` step before the screener runs. Commit separately so it is bisectable.
2. **Classification (C1):** in `screen_full`, compute `already_scored_today` from `prior_cache["tickers"][t]["last_bar"]` and `behind` per the C1 definitions; return both in `stats` (`stats["already_scored"]`, `stats["behind"]`, `stats["behind_dates"]` = `Counter` of newest-bar dates). Keep the per-ticker `stale_bar` skip reason as is for the skip log.
3. **Gate (C1 + H2 + L1 + M5):** extend `enforce_publish_gate`/`build_health_payload` to take `already_scored`. Duplicate branch = `(screened + already_scored)/total >= floor and already_scored > 0`: no `save_cache` unless `screened > 0`; skip-log merge per M5; publish only per H2; heartbeat `ok`/`partial` by the same thresholds on the combined numerator, `published` reported truthfully, `attempt: 2 (…)`.
4. **Heartbeat text (fix 3 + H1):** replace `"(Yahoo throttling?)"` (`:2964-2967`) with `failed downloads N · already scored N · behind N (newest bar: 2026-09-25 ×712, 2026-09-18 ×22)`; stamp `cycle:` with `today_et()`. Existing tests assert only on status, `NOT PUBLISHED`, the floor and the counters (`tests/test_screen_coverage_floor.py:77-143`), so the string is free to change.
5. **Re-fetch (fix 2 free half, M3 + M4):** after the batch and before the per-ticker loop, if `len(behind)/total > 0.20`: sleep 60 s, `yf.download(behind_symbols, period="5d")`, merge per ticker onto the 400-day frame, log and count `recovered`, carry the count into the heartbeat. No FMP.
6. **post_earnings_movers (fix 4, L5).**
7. **History repair (M2)** as a one-off script + commit, then **cap the report** (M1) and add the failure backstop to `sigma-weekly-skip-report.yml`.
8. **Comments/docs:** `sigma_screener.py:129`, `:1157`, `sigma-close.yml` commit step — the consumer is `weekly_skip_report.py`, not Coverage Manager. Board: correct #292/#327/#376 premises as the doc says.

### Tests the build must add (against the recorded shapes, not garbage)

- 09-21 20:11 shape: 712 `latest=09-18/last_bar=09-21`, 22 `latest=09-18/last_bar=09-18`, 6 Tokyo, 5 no-history, 10 `latest=09-21/last_bar=09-21` → duplicate, `ok`, nothing published, `save_cache` not called, skip-log entry unchanged.
- 09-28 21:04 shape: 712 `latest=09-25/last_bar=09-25`, 18 scored → `error`, histogram names `2026-09-25 ×712`, re-fetch attempted, `recovered` reported.
- Partial-first-run shape: first run 85% (`last_bar=today` for 643), second run screens 113 with 2 alerts → digest posted with the attempt-2 banner (or, fallback, heartbeat names 2 unposted alerts); watermark advanced only for the 113; skip-log entry shrinks by 113.
- `build_health_payload` cycle date at 01:04 UTC with ET = previous day → previous day.
- Mutation check per `feedback_a_green_suite_is_not_evidence`: flip `>= today` to `> today` in the C1 rule and confirm the 09-21 test fails.

### Follow-ups outside tonight's scope (file as rows, not build)

M6 (watchdog UTC date), L2 (holiday-aware counters), L3 (multi-session catch-up), the doc's step 4 probe (every weekday evening, not Mondays only), and the FMP half of fix 2 — which should wait for the `recovered` counter from step 5 to say whether it is needed.

## Build notes (2026-09-29, uncommitted — awaiting Codex)

- **Steps 1-5, 7, 8 built in sigma-alert; step 6 in post_earnings_movers.** Nothing committed or pushed. M2 is a script plus dry run only (`scripts/repair_skip_log.py`), not applied.
- **Deviation 1, C1 duplicate rule.** The build uses `already_scored` ALONE >= floor, not Fable's `(screened + already)`. An earlier run at 50% was refused at the gate and published nothing, so a later run completing it must not post an `ok` "already published" card. Status is still judged on the combined numerator, as Fable specified. Pinned by `test_an_earlier_refused_run_is_not_a_publication`.
- **Deviation 2, H2.** The build uses the minimum form: the heartbeat names each unposted alert. A supplementary digest would need its own formatter, because `format_slack_message` would stamp a "DEGRADED RUN 113/756" banner on it and render a near-empty returns block.
- **M2 commit list.** The mechanical derivation disagrees with Fable's hand list on 2 of 15 days:
  - 09-03 → `3bd8659` (5 skips); Fable listed `4ab71e7`, which has no 09-03 entry.
  - 09-04 → `923c663` (25 skips); Fable listed `a75fb61`, which has no 09-04 entry.
  - The other 13 days match.
- **Replays (real Yahoo data, `--dry-run`):**
  - **09-28 good data:** 747/756 screened, 30 alerts. This is the session JP never received.
  - **09-28 with the bar removed for 738 symbols:** 713 behind, re-fetch recovered 708 of 713, same 747/756 and 30 alerts. This shows the merge works when Yahoo is healthy. It does not show that the re-fetch recovers at 21:04 ET.
  - **09-21 duplicate:** `ok`, `attempt: 2`, 723 already scored. The 22 names that arrived late were scored, their 2 alerts are named as NOT POSTED, and the skip log is unchanged.
- **Follow-ups to file as board rows, not built:**
  - M6: the watchdog keys its date on UTC.
  - L2: holiday-aware counters.
  - L3: multi-session catch-up (the lost 09-28 session is unrecoverable under the current watermark).
  - The every-weekday-evening Yahoo probe.
  - The FMP half of fix 2, which should wait on the `recovered` counter.
  - A supplementary digest for H2.
  - Remove the "Likely transient Yahoo throttling" wording in `format_slack_message`'s DEGRADED banner. It is the same unmeasured guess the heartbeat line was.

## Fable post-build review (2026-09-29)

**Verdict: APPROVE — `85be1b7` stands as shipped. No Critical. One High (a latent cache-clobber in `mark_published`, one-line fix, not blocking tonight's runs). Both departures from the plan are correct, and departure 1 fixes a hole in the plan.** Line numbers are the HEAD file after `85be1b7`. No code was changed by this review. Verified by running, not by reading: `tests/test_duplicate_run.py` + `tests/test_weekly_skip_report_cap.py` 29/29 in 23 s, full suite 368/368 in 22 s, `git status` clean after both (the suite is not a pipeline run).

### The two departures

| Departure | Verdict | Why |
|---|---|---|
| **1. Duplicate := explicit `cache["published"] = {date, mode}` marker, written only after Slack accepts, AND `already_scored > 0`** — instead of the plan's `(screened + already_scored)/total >= floor`. | **Correct, and better than the plan.** | The plan's rule would have called two refused runs (79.9% + 20.1%, every `last_bar` at today, nothing ever posted) a duplicate and posted an `ok` "already published" card. Watermarks are written before the gate (`sigma_screener.py:3700` vs gate at `:3799`), so they can never say *delivered*. Codex R1 P1 was a real defect in my plan. Marker verified end to end: written only when `delivered` and close-only (`:3814-3815`); persisted through `save_cache` into `cache/distribution_cache.json` (`:1776-1786`), which the workflow's commit step adds (`sigma-close.yml:68`) — so the fetch/reset step at `:34-40` delivers it to the next run; carried through a duplicate's re-save (`:3684-3687`); dropped naturally the next day (marker date ≠ today → not copied); midday/open never write the cache (`:3626-3663`) so they cannot wipe it. Pinned at the transport by `test_a_failed_post_writes_no_marker`. Dropping the "≥ floor" clause is right: a day published at 85% followed by a zero-screen second run is `partial` / attempt 2, not `ok`. |
| **2. H2 as the minimum — heartbeat names every unposted alert — instead of a supplement digest.** | **Acceptable as the minimum the plan allowed.** | The reasoning holds: `format_slack_message` only knows this run's `screened`, so a supplement would carry a "DEGRADED RUN 113/756" banner and an empty returns block. Every alert is listed, chunked under 3,000 chars (`:3308-3323`, `test_forty_alerts_are_all_listed`). The supplement stays a follow-up row. Gap noted at L1: 52w hi/lo hits from late names reach only the log (`:3413`), not the card. |

### Checks asked for

- **C1 as built.** `already_scored_today` = `last_bar >= today_et()` via `scored_watermarks` (`:1719-1762`) — the scored watermark only, never `last_seen`/`refused_bars`, never what the feed returned this run. `behind` = `stale_latest` minus already/screened, histogram by `Counter` (`:2197-2216`). Status on the combined numerator (`:3269-3276`). Correct. The plan's mutation check (`>=` → `>`) was not run (no edits tonight); by inspection `TestSeptember21Duplicate` fails under it (already_scored → 0 → not a duplicate → `error`).
- **Concurrency + fetch/reset.** `sigma-close.yml:18-20` (`group: sigma-close`, `cancel-in-progress: false`) and `:34-40` (`git fetch --depth=1 origin master && git reset --hard FETCH_HEAD`) precede the screener step at `:50`. Watchdog dispatches are the same workflow, so the group covers them. `git push` from the reset branch is fast-forward. Correct. See L5 for the one case the group does not cover.
- **Marker committed.** Yes — `git add cache/distribution_cache.json` at `sigma-close.yml:68`, and `mark_published` runs before that step. A duplicate that scored nothing makes no commit (`git diff --cached --quiet`), which is the intended L1 behaviour.
- **Skip-log merge (M5).** Duplicate with an earlier entry → earlier minus names scored this run (`:1189-1203`), pinned by `test_skip_log_entry_shrinks_by_exactly_the_late_names`. See M3 for the no-earlier-entry fallback.
- **New top-level cache key is safe.** Readers, from an `rg --no-ignore` of the whole fleet root: `sigma_screener.py` (`cache["date"]`, `cache["tickers"]`), `scripts/cache_utils.py`, and one external consumer, `focus_today/sources/sigma.py:30`, which reads `json.load(f).get("tickers", {})`. None iterates top-level keys, so `published` is invisible to all three. (An earlier draft of this line said "no other repo reads the file" — that came from a narrower search that had timed out; corrected.)

### Findings

**High**

- **H1 — `mark_published` is load-with-fallback + save-everything (`sigma_screener.py:1781-1783`).** `load_cache()` returns `None` on `OSError` or `JSONDecodeError` (`:1080-1088`); the fallback `{"date": today, "tickers": {}}` then gets the marker and is written over the distribution cache — 756 distributions and every watermark gone, committed by the workflow, and the next morning's open run sees "not in cache" for the whole universe. The trigger is narrow in CI (the file was written by `save_cache` seconds earlier), which is why this is High and not Critical, but it is a data-destroying write on the publish path, the exact class in `feedback_load_with_fallback_then_save_destroys`, and one line removes it: pass `cache_data` in from `main()` and set the key on the dict already in memory, or return without saving when `load_cache()` is `None`. Recovery if it ever fires: `git checkout HEAD~1 -- cache/distribution_cache.json`.

**Medium**

- **M1 — A refused first run plus a completing second run loses the day, with a card that reads as a data problem.** `enforce_publish_gate` (`:3385-3386`) and the normal-path heartbeat (`:3859-3864`) judge on `screened` alone; `already_scored` enters the status only on the duplicate path. Run A at 79% is refused but advances 597 watermarks (`:3700`, save still before gate); run B scores the other 159 → not a duplicate (no marker — correct) → refused at 21% → `error`, "market data returned for only 159/756" — false; the `Data:` line's `already scored 597` is the only honest number. Not a regression, and the plan's rule was worse (a false `ok`). Follow-up row: persist each ticker's day z/return in its cache entry so a completing run can assemble the day's digest; at minimum, the warning line should say "N scored by a refused earlier run".
- **M2 — `describe_unscreened` prints unmeasured zeros on the open-mode cached path (`:3221-3245`).** `screen_open_cached` never runs `_classify_unscreened`, so a sub-floor cached open run's card says `failed downloads 0 · already scored 0 · behind 0`. Absent data is not a finding; return `None` when `"already_scored" not in stats`.
- **M3 — The M5 fallback re-inflates.** `update_skip_log` on a duplicate with no earlier entry writes the duplicate's own `skip_events` (`:1205`) — ~700 `stale_bar` rows for already-scored names, the inflation this commit exists to remove. Reachable when the first run's push failed (L5) so its skip-log entry never landed. My plan's M5 said "write normally" — that wording was mine and it was wrong; normal for a duplicate must exclude `already_scored` tickers.

**Low**

- **L1 — 52w hi/lo hits from late-scored names are counted in the log (`:3413`) but not named on the card.** Lesser loss than an alert; add to the `NOT POSTED` list when the supplement is built.
- **L2 — Whole-batch-stale abort reports `behind 0` (`:2299-2306`)** because `_stale_latest` is empty before the per-ticker loop; the card says `whole batch stale (N not today)` instead of the newest-bar histogram. The re-fetch `attempted/recovered` line still appears. Cosmetic.
- **L3 — Holidays now cost a 60 s sleep and one 756-symbol `period=5d` call before the abort (`:2133-2170`), twice (cron + watchdog).** Plan L2 accepted holiday `error`; noting the added cost until the market-calendar check lands.
- **L4 — Normal-path cards still carry no `attempt:` line**; the contract lists it as required (`HEALTH_REPORTING.md:71`). Only the duplicate card has one now. Pre-existing.
- **L5 — `git push` has no rebase-retry (`sigma-close.yml:82`).** The concurrency group covers close runs only; a push from `sync-watchlist.yml` / `refresh-sp500.yml` in the same minute makes the close push non-fast-forward, `if: failure()` posts a crash card for a run that published, and the marker never lands — the next run then re-publishes (accepted per JP) and hits M3. Pre-existing.
- **L6 — The weekly backstop card (`sigma-weekly-skip-report.yml:53-54`) has no `cycle:`/`attempt:` line.** Cosmetic against the contract; the close workflow's backstop builds its payload with `ci_health_payload.py` and could be reused.

### What to do before the next close run (2026-09-29 ~21:30 UTC)

Nothing is required. H1 is a one-liner worth landing in the morning; M1-M3 and L1-L6 are rows. The follow-ups already listed in the build notes (M6, L2, L3, the evening probe, the FMP half, the supplement digest, the DEGRADED banner wording) stand.
