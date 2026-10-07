# #551 — should a foreign listing with no same-day bar count in the coverage denominator?

Measured 2026-10-07 ~04:00 ET (overnight session). Run examined: close run
`37554018278`, started 2026-10-06 20:50 ET, heartbeat `partial`, 717/756 = 94.8%
(threshold `DEGRADED_SCREEN_COVERAGE` = 0.95, so 719 needed). Sources: the run's
GitHub Actions log, `cache/skip_log.json` (22 close runs, 2026-09-07..2026-10-06),
and live yfinance history re-pulled at ~04:00 ET 2026-10-07. No metered calls.

## Answer

**Keep foreign listings in the denominator. The 2026-10-06 `partial` was a true alarm.**
All 33 stale names belong to markets that DID trade on 2026-10-06, and their
bars were missing. An exchange-calendar-aware denominator is right in principle,
but it would have removed **0 of the 33** names on 2026-10-06. Across 22 runs it
would have changed **0** heartbeat statuses. So no code change is shipped.

## The 33 stale names on 2026-10-06, by cause

| Cause | Count | Names |
|---|---|---|
| Exchange holiday on 2026-10-06 | 0 | (none; Tuesday, no holiday on any of the 7 European venues) |
| Session already reflected given timezone | 0 | (none; the run was 20:50 ET, ~9h after every European close) |
| Dead / renamed symbol | 0 | (all 33 return live history today) |
| **Vendor miss: Yahoo per-equity European bar gap** | **23** | CVSG.L, CTEC.LN, ONT.L (LSE) · FRE, GXI, 1SXP.DE, AFX.DE, SRT.DE, SHL.GY (Xetra) · AMP.IM, DIA.MI (Milan) · BIM, ERF.FP (Paris) · LONN.CH, SFZN.SW, STMN.SW, DAE, SOON, YPSN (SIX) · COLOB.DC, GN.DC (Copenhagen) · GETIB.SS, SECARE.SS (Stockholm) |
| **Vendor miss: transient Yahoo lag, FMP could not fill** | **10** | IQV, VEEV, ZTS, ARES, CTVA, HPE, P, RDDT, XYZ, BTC-USD |

The other 6 skips are not stale bars: SECTB.SS, AMBUB.DC, CSU, FI and TALK are
`insufficient_history`, and PSKY had 7 days of history.

### Evidence that the 23 European names were genuine misses
- Yahoo's **index** bars for 2026-10-06 exist: ^GDAXI, ^FTSE, ^STOXX50E, ^FCHI and ^SSMI all carry 2026-10-06. So the venues traded.
- Yahoo's **single-stock** history skips the date for every European equity checked, including SAP.DE and NOVO-B.CO, which are outside the watchlist. At ~04:00 ET 2026-10-07 the series goes 2026-10-05 → 2026-10-07. The bar is still missing ~16h after the close.
- The run's FMP fallback logged "23 foreign name(s) out of scope". No recovery path exists for these names.
- The 10 US/crypto names all carry a 2026-10-06 bar now. At run time the re-fetch recovered 0/210. FMP recovered 176/187 and rejected 11 (9 `intraday`, 2 `gap_before_today`).

## History: 22 close runs, 2026-09-07..2026-10-06

117 foreign stale-bar skips in total (counted as name × run):

| Cause | Name × run | Detail |
|---|---|---|
| Exchange holiday (would correctly leave the denominator) | 21 | B3 Brazil Independence Day 2026-09-07 (2); Tokyo Silver Week 2026-09-21/22/23 (6 each = 18; Yahoo still has no bars for those dates); HKEX National Day 2026-10-01 (1, 2715.HK) |
| Vendor miss, market traded | 96 | **Pan-European gaps on 4 dates**: 2026-09-09 (23), 2026-09-18 (20), 2026-09-25 (20), 2026-10-06 (23) = 86. Smaller LSE/SIX lags on 2026-09-22 (3), 2026-09-28 (4) and 2026-10-01 (3) = 10 |

- On 09-09, 09-18 and 09-25 Yahoo backfilled the European bar later; FRE.DE history now has those dates. 10-06 has not been backfilled yet.
- The biggest holiday day had 6 names (Tokyo). That is under the ~37-name margin, so holiday-awareness flips no status in this window.

## Design call

- **Principle (accepted):** a listing whose exchange held no session should leave the denominator. A listing whose exchange traded but whose bar we lack is missing, and it must keep alarming.
- **What the data says:** the second kind dominates, 96 of 117 name × run. In particular, the four pan-European gaps are a recurring Yahoo defect, roughly weekly in September 2026. Removing foreign names from the denominator, or exempting "foreign and not yet at today", would have turned 2026-10-06 into `ok` (717/733 = 97.8%). That would hide a real 23-name hole in the alert tiers. **Rejected.**
- **Exchange-calendar exclusion:** correct, but it has zero measured effect on status. It would also add a holiday-calendar dependency for about 10 venues to a CI-deployed job. **Deferred, not built.** Revisit if a holiday ever flips a status, for example a large Japan or HK holding on a Golden Week day.
- Decided by the overnight worker on the measured data. No Fable consult, because the data removed the ambiguity.

## Follow-ups worth a board row (not done)
1. **Recovery path for foreign names.** The 4 pan-European days are the actual cause of `partial`. Options: a same-run re-fetch at `interval=1h` aggregated to a daily close, or a second free source for European equities (Stooq now sits behind a JS proof-of-work wall and is not scriptable). Alternatively, name the cause in `describe_unscreened`: "N European equities missing the YYYY-MM-DD bar while their index has it". The heartbeat would then say *Yahoo European gap* rather than a bare count.
2. **5 chronic dead/renamed symbols** in 22 of 22 close runs, each permanently in the denominator (0.7pp of the 5pp margin): AMBUB.DC and SECTB.SS (Position-list spellings with no `_YF_SYMBOL_OVERRIDES` entry; the watchlist forms AMBUSH.DC and SECARE.SS are mapped), CSU (Constellation Software, TSX, likely needs `CSU.TO`), FI (Fiserv, likely re-tickered), TALK. These are symbol fixes, not denominator questions.
3. **Hypothesis, not exercised: tonight's run may score two European sessions as one.** If Yahoo has not backfilled 2026-10-06 by tonight's close run, the 23 European names will carry a 2026-10-07 bar against a 2026-10-05 watermark. `is_unscored_bar` asks only whether the bar is *newer*. That would z-score a two-session move as a single day. The FMP path refuses this case (`gap_before_today`). Whether the Yahoo path has an equivalent guard has not been checked.
4. The close cron is `30 21 * * 1-5` (17:30 ET), but this run started at 20:50 ET, 3h20m late. That is not the cause here, because the European bar is still absent now.
