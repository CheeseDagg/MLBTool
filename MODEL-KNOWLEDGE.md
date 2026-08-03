# WHAT THE MODELS KNOW — the whats-what reference
*One page per tool: what's IN the model (validated), what was TESTED AND REJECTED
(don't re-argue it — the data spoke), and what the model is BLIND to (your manual
overlay zone — edges you might see that the model cannot). Updated 2026-07-27.*

Rule of thumb when a price looks wrong: first check BLIND SPOTS. If your read lives
there, the model isn't disagreeing with you — it can't see the thing you see.

---

## MLB — HR board (mlb_hr.py)

**IN (each validated or long-standing):**
batter season line blended with Marcel multi-year talent (age-adjusted, traded-player
stints summed, freshest-of live/snapshot) · season anchor (60% weight ≥60 PA; prior-
anchored below) · park factor (confidence-shrunk, +0.07 logit for low-conf parks) ·
temperature + wind direction vs park orientation · platoon (when handedness known) ·
opposing-starter composite (K/BB/flyball, league-centered; neutral if TBD/small) ·
bullpen HR rate · Savant barrels · bat-vs-pitch-zone heat (demoted weight) · batting-
order slot (PA share + hitter-quality gradient) · HOT HAND: homered in last graded
game = +0.18 logit (only the most recent game carries signal) · starter HR-allowed
pools seeded with his FULL PRIOR SEASON (validated: in-season-only was noise, prior-
year-matured cells are a robust 3/3-period holdout win; RE-CONFIRMED 2026-07-29 on the
two-season wide panel at +0.00025 pooled, winning 2024 and 2025 separately, 4/4 slices,
and a 20-trial start-level placebo fired 0/20 with max noise +0.00003. The effect is
MONOTONE IN CELL MATURITY — thin cells under 200 batters faced −0.00001, 200-600
+0.00014, 600+ **+0.00044 and robust** — which is the mechanism behind "needs a full
prior season", and the weight sweep has an interior optimum (rises to 0.45, plateaus to
0.60, −0.00092 by 1.40), so it is not a knife edge) · monthly recalibration
(leak-free, holdout-gated).

**TESTED & DEAD:** month/season phase · day of week · small-sample pitcher shrink
changes · pitcher residual momentum · recency-weighted calibration · umpire (for K;
never showed for HR context either) · day vs night game (null even with full data) ·
individual batter platoon splits beyond the flat league factors (null BOTH in-season
and with a full prior season maturing the cells — the flat factors already carry it) ·
handedness-split park factors (flickered, never robust, even with 2024 burn-in) ·
ROLLING CONTACT-QUALITY FORM — dead at every window from 7 to 60 days. Barrels are a
flat NULL (best holdout −0.00016, 0/3 periods). Hard-hit at a 60-day window technically
cleared the ship rule (train win, holdout +0.00013, 3/3) and even beat a 24-trial
shuffled placebo, but it does NOT ship: the 3/3 evaporates to 2/3 if the period cut
moves four days, and the window sweep is non-monotone (7d +0.00012, 14-45d ≈ 0, 60d
+0.00013) — a real form signal thickens with the window, it does not spike at both
ends of the grid with a dead middle. The KEYHOLE HYPOTHESIS IS NOW ANSWERED AND DEAD:
the short window was never the problem, there is simply nothing here beyond the season
line and the hot-hand flag. Two method upgrades came out of it and are permanent: the
form norm now EXCLUDES its own window (at W=60 vs a 200d norm the window was a sixth
of its own baseline, mechanically dragging the ratio to 1.0), and `contactform_placebo.py`
re-runs the entire tune-and-verdict pipeline on batter-shuffled contact — the standard
gate for any future win under ~+0.0005.

Career FAMILIARITY (Nth time a batter has faced this starter) is a clean null —
+0.00082 power ceiling, measured +0.00000.

**CANNOT BE SEEN AT THIS SAMPLE (not the same as dead — do not bury):** batter REST
days and 7-day WORKLOAD. Both read ≈ −0.0002, but their power ceilings are only
+0.00033 and +0.00046, because ~83% of games follow a game so the term is 1.0 on most
rows. Even a true ±40% effect would sit in the noise. Answerable only with a
multi-season panel. Same for within-season familiarity: 4% of rows are repeat
meetings, ceiling ≈ 0.
> **SUPERSEDED 2026-07-29 — see the fatigue-four block at the end of this section.**
> On the wide panel WORKLOAD and both familiarity terms are now **DEAD**, not
> invisible. REST stays invisible, but for a structural reason (70% of rows are
> exactly one day off), so a wider panel will not rescue it either.

**RE-READ ON THE WIDE PANEL (2026-07-29, holdout 24,332 rows — 3× the old one).** All
seven HR angles from batches 1 and 2 were re-scored across two seasons. Only pitcher-HR
flipped (see IN, above). The other six lost again — indiv platoon −0.00028, handed park
−0.00010, day/night −0.00002, travel +0.00000 (1/4 slices), home/away −0.00004, slot
shift −0.00007 — but their ORACLES are +0.00005 to +0.00023, meaning a generous
planted 15-18% effect would still be near-invisible even at 3× the rows. They are
filed CANNOT BE SEEN, not dead. The instructive part is the harness, not the baseball:
one angle that was NULL on a single-June holdout is a clean cross-season win on the
wide panel, so **any verdict filed on the narrow panel is provisional** — batter REST,
7-day WORKLOAD, within-season familiarity and rolling contact form all deserve the same
re-read before anyone treats them as settled.

**THAT RE-READ IS NOW DONE FOR THE FATIGUE FOUR** (`mlb_fatigue_wide.py`, 2026-07-29,
holdout 24,332). Three of them are no longer "cannot be seen" — they are **DEAD**, and
that is a real result, not a shrug. 7-day WORKLOAD, within-season FAMILIARITY and
career FAMILIARITY all had a planted +30% effect recovered *robustly* by the full
tune-and-verdict pipeline (3/3, 2/3 and 2/3 seeds), so the panel demonstrably can see
an effect that size — and all three measured negative (−0.00001, −0.00040, −0.00007).
When the panel can see it and it is not there, it is not there. Close them.

**Batter REST is the exception, and for a structural reason worth remembering.** Its
oracle is +0.00006 and a planted +30% effect was recovered **0 of 3 times** — the
pipeline cannot find rest even when rest is true by construction. The mechanism is in
the feature, not the sample: of 24,332 holdout rows, **17,018 are exactly one day off**
and 23,959 are three days or fewer. There is almost no contrast to fit, so tripling the
rows again will not help. Rest is not under-powered, it is nearly **constant**. Do not
re-run it on a wider panel; if anyone wants it, it needs a different feature (in-game
workload, catcher starts, day-after-night travel), not more rows.

Two harness lessons from this run, both of which cost a full re-run:

**A magic number was doing the detectability test.** The ceiling ladder read
`oracle < 0.0004 → cannot be seen`. LOAD's oracle is +0.00038 — a hair under — so it
printed "cannot be seen" on the same run where the fitted pipeline had just recovered a
planted LOAD effect 3/3. Detectability is *measured* by the plant; never hard-code a
cutoff next to an experiment that already answers the question. This is the THIRD bug
in this ladder (twice the ordering, once the cutoff), so it is now a tested function,
`read_ceiling()`, with LOAD's real numbers pinned as a regression case. **Any logic
that decides a verdict belongs in a function with a test, not in an if-chain inside
`main()` that only runs during a six-minute job.**

**Two bugs that only bite on a multi-season panel** were found while porting: within-
season familiarity was hard-coded to `d >= "2025-01-01"`, so every 2024 row carried
fam=0 — the feature was constant in exactly the holdout meant to measure it — and rest
was *capped* at 5 days rather than excluded, so a 183-day offseason gap read identically
to a Sunday off, which would have made all of April look like league-wide peak freshness.
When porting a narrow-panel angle wide, **re-read its feature builder for season-boundary
assumptions before trusting the port.**

> **BACKPORTED 2026-08-03.** Both bugs are now fixed in `mlb_fatigue_experiment.py` too,
> which is where they originated. `build_history()` keys every within-season accumulator
> — last-game, the 7d/30d workload list, and the familiarity counter — on
> `(entity, season)`, so the rest gap is simply **undefined** across a winter rather than
> capped at a big number. Sizes: **636 gaps over 45 days before, 128 after**; the 508 that
> disappeared were every returning batter's first game of 2025, all of them voting at the
> top of a scale that caps at 5. **No verdict moved** — REST −0.00020 → −0.00022 full
> sample, −0.00016 → −0.00008 on regulars, still NULL — but the direction is the warning:
> a season-opener genuinely IS a fresh bat, so the contamination was correlated with the
> effect under test, not clean noise. It flattered the term rather than diluting it.
>
> One claim above was itself a symptom and is now **withdrawn**: "4% of rows are repeat
> meetings" was produced by the `d >= "2025-01-01"` bug pinning every burn-in row to
> fam=0. Season-keyed, the panel is 17% and the scored June-2025 holdout was always
> 19.6%. FAMIL's verdict is unchanged (measured still exceeds its own ceiling, so it is
> still noise) but **it is not noise for lack of coverage** — do not quote the 4%.
>
> Still latent elsewhere, not a live bug and deliberately left alone: `build_contexts()`
> in `mlb_kfactors_experiment.py:142` computes pitcher `rest_days` the same raw way, and
> `rest_bucket()` sends anything over 5 days to `"long"`. Its dataset builder is pinned to
> a single season, so the path is unreachable today — but widening that panel the way the
> wide-panel work widened this one would drop every opening-day start into the `"long"`
> bucket and fit a ratio to it.

**THE CALIBRATION PANEL WAS LYING FOR A WEEK, AND THE WAY IT LIED IS THE LESSON
(2026-07-30).** On 2026-07-23 commit `1bbf7c9` added `hr_raw` to `GCOLS`. The append
path in `mlb_grade.py` writes `GCOLS`; the header line is written ONCE, at file
creation. So from that commit on, every row carried 17 fields under a 16-field
header, `csv.DictReader` shifted every column past the insertion point by one, and
`outcome` came back holding the neighbouring column's text (`heat +3%`). `summarize()`
keeps only `outcome in ("hr","no")` — so **217 of 662 rows, seven full days of
grading, were discarded in total silence** while the board went on publishing a
healthy-looking `n`. `migrate_graded()` existed and should have caught it, but it was
keyed on `"hr_n" in header` — the one migration it was born for — so it returned early
and stayed blind to every later column. Fixes: rows are now parsed by **their own
width** against known schema generations (`_rows_by_width`), the migration keys on
`header == GCOLS` exactly, and any row whose outcome is outside the vocabulary
surfaces as `panel["unparsed"]` instead of vanishing. **The general rule: an
append-only file whose header is written once is a schema-drift trap, and a
filter-then-aggregate step is where the evidence gets destroyed. Every silent drop
must become a counter.**

**THE SAME BUG WAS ALSO IN THE WORKFLOW, WHICH IS HOW YOU KNOW IT IS A CLASS AND NOT
AN ACCIDENT.** `.github/workflows/mlb-daily.yml` staged six `mlb/data` paths **by
name**. So `lineshop.json`, `bvp_board.json` and `recalib_report.json` were rebuilt
every run and thrown away, and `league_daily.csv` — a file that is worthless unless it
ACCUMULATES, since an Actions checkout is fresh every time — would never have held more
than one day. Now `git add -u mlb/data/` (tracked-only, so no stray cache or 9 MB
experiment dump rides along) plus a one-time explicit add for genuinely new artifacts.
**Hand-maintained lists of things-to-include rot silently and always in the direction
of dropping data.** Two independent instances of it in one codebase in one week.

**AND THE MISS I WENT LOOKING FOR WAS AN ARTEFACT OF THAT BUG.** The flag that
started this was the 21-date panel showing the 12-16% bucket 0-for-15 and 16-20%
1-for-29 — "the middle of the board is broken." On the repaired panel (n 347 → 552)
those become 2/35 and 7/79, and the honest verdict is **no shape evidence at all.**
Fitting the recalibration `a·logit(p)+b` on the live panel gives slope **a=1.448,
95% CI [0.564, 2.750]** — the point estimate is on the wrong side of 1 for the
"middle sags" story and the interval is useless. The reason is structural and worth
keeping: **the board only publishes rows in a ~15–30% band, so it has almost no
leverage to estimate calibration SHAPE.** Bucket-by-bucket readings on the live
ledger are not interpretable, ever. Shape has to come from the backtest panel.

**WHAT IS REAL: a modest LEVEL miss, marginal.** Live out-of-sample, actual − pred =
**−3.39 pts, date-block-bootstrap 95% CI [−6.68, −0.07]** (20 dates, n=552). A −0.219
logit shift would fix it (a 21.0% row prints 17.6%). Two independent methods agree on
direction: the recalibrator's own fit on the repaired ledger tops out at anchor
(21.6 → 18.3). Design effect from date clustering is **1.0×** — the day-to-day swings
(3.2% to 30.0% actual) look dramatic but are ordinary binomial noise at ~28 rows/day,
so naive SEs were not wrong here. **Do not read the swings as signal.**

**SELECTION / WINNER'S CURSE IS RULED OUT.** The obvious suspect was that publishing
only the top ~35 rows/day selects on noise. It does not: on the backtest, calibrated,
the top-35-per-day slice reads **pred 17.88% vs actual 18.01%** (n=3,675) and every
slice from top-50 down to top-5 is flat-to-favourable. The board's selection is clean;
the level miss is not selection.

**THE CONTROL THAT WAS MISSING, NOW BUILT.** The whole cold-week question — did the
model drift, or did the league go quiet? — was unanswerable because nothing recorded
the league's own rate. `league_day`/`record_league`/`league_context` now compute
league HR/PA per date **from the boxscores the grader already fetches** (zero extra
requests), write them idempotently to `mlb/data/league_daily.csv`, and surface
`panel["league"]["rel"]` — the window's rate over the season's. `rel < 1` is the share
of a board miss that is **not the model's fault**. It backfills from the next Actions
run; statsapi is blocked from the sandbox, so this could not be answered locally.
Guard worth noting: `league_context` tests `a is None`, not `not a`, because a window
where the league hit **zero** home runs is precisely what the control exists to catch.

**DO NOT APPLY THE PENDING RECALIBRATION YET.** `mlb_recalibrate.py --dry-run` on the
repaired ledger says APPLY (holdout Brier 0.13809 → 0.13762) — but that gain is
0.00047 against a `MIN_GAIN` of 0.0001, and **the holdout is the most recent 21 days,
which contains the anomalously cold 07-23..29 window.** A candidate curve that simply
predicts lower will win on a cold week whether or not it is right. That is exactly the
variance-chasing the recalibrator's own docstring warns against. Wait for
`league_daily.csv` to say whether that week was the league or the model.

**RESOLVED 07-30 — it was BOTH, and they separate cleanly.** Regressing daily
(actual − predicted) on the league HR index: **slope +20.33pt per unit index, 95% CI
[+7.19, +32.60], P(slope>0)=0.996** — theory says ~21pt because the mean prediction is
21%, so *day-to-day board swing is essentially the league environment* and is not a
model fault. What SURVIVES that control is a level bias: intercept −4.29pt, and
directly on the ledger the board predicted 20.97% and hit 17.57% over 20 days (n=552),
ratio 0.838, day-clustered 95% CI [0.683, 1.002], P(hot)=0.973. Shipped as
`LIVE_LEVEL` = 0.88 in `mlb_hr.py`, shrunk toward 1 by leave-one-day-out.

**THE CLASS OF BUG, worth carrying to every other sport:** a level gap between the
BACKTEST REPLAY and PRODUCTION is invisible to any refit that pools them. The monthly
recalibrator mixes 25k replay rows with ~550 live ones; live is 2% of the pool, so it
cannot move the anchors no matter how wrong production is. Shape and level need
separate estimators fit on separate data. Also: bucketing the live board by predicted
% made the bias look like a *shape* problem concentrated at 10–20%. It is not — those
rows were two thin-slate days, and with day fixed effects the slope on logit(p) is
1.39, CI [0.32, 2.88]. **Bucket tables on day-clustered data lie about shape.**

**07-27 (residual −15.58) is variance, not signal** — no park, temperature or team
cluster; P(≤1 of 25 at 21%) ≈ 2%, i.e. expected about once across 21 board days.

**AND A MISS:** slate temperature was flagged 07-29 as a live nowcast for the league
index (r=+0.469, R²=0.220). It does not survive leave-one-day-out — worth +0.00016 LL
at best shrinkage and actively harmful unshrunk. The R² was overfit on 13 days. The
oracle gap says a *working* league-index nowcast is worth ~+0.012 LL; temperature is
not it.

**BLIND SPOTS:** umpire assignment · travel/getaway days · lineup changes after build
(spot tags flag rookies/new bats, not scratches) · anything intra-day (board is
pregame) · calibration SHAPE from the live ledger (no leverage — see above).

## MLB — K props (mlb_kprops*.py)

**IN:** pitcher season K/BF, league-regressed · expected IP from his ACTUAL STARTS
only (game-log; relief innings excluded — the swingman guard, after Perkins pinned
7.0 IP off bulk-relief innings) · opponent OVERALL team K% shrunk 0.6 toward neutral
(walk-forward validated on 2,448 starts — 0.6 beat the old 0.5 in 3/3 periods) ·
push credit on whole-number lines · stale-slate guard + auto-rebuild when a fresh
slate publishes.

**TESTED & DEAD (all on real 2025 data):** umpire K-tendency · catcher framing ·
recent-form windows · home/away splits · days rest · workload (BF) trend ·
innings shrinkage toward league average (raw start-average won on holdout; shrink
actively hurts short-leash arms — openers really are short) · opponent K% split
by starter hand at the TEAM level (100% hand coverage, walk-forward: no shrink
level beat overall K% even in-sample — team-overall whiff already carries it).

**BLIND SPOTS:** the LINEUP-specific vs-hand read — who's actually in tonight and
how those bats whiff vs this hand (team-level splits tested & dead above; the
Schultz-type read stays YOURS, it needs tonight's lineup, not team history) ·
pitch-type matchups · scratches after build.

## NFL (nfl_model.py)

**IN:** Elo with margin-aware updates · ADAPTIVE home-field (leak-free running
estimate; currently ~34 Elo, down from ~57 in the 2010s — the fixed 48 was a hidden
look-ahead) · rest days (linear) · divisional-game shrink ×0.90 at prediction ·
preseason reversion.

**TESTED & DEAD:** play-by-play EPA + QB-EPA · QB win/loss residual rating · wind,
temperature, dome, surface · Thursday/short-week beyond rest · playoff temperature ·
late-season dead-rubber (favorites actually do BETTER) · bye-week nonlinearity ·
parameter retunes.

> **SUPERSEDED 2026-08-03 — READ THIS FIRST.** Everything below about QB change
> is still measured correctly. It is also **NOT SHIPPABLE**, and the four gates
> below could not see why. `QBNEW` reads `home_qb_id`, which is populated on
> 7,276 / 7,276 PLAYED games and **0 / 272 UNPLAYED** ones. It is perfectly
> time-ordered and completely unavailable at the moment a prediction is actually
> made. A new **GATE 0: AVAILABILITY** now vetoes it. Keep the absorption
> reasoning; discard the conclusion that it was ready. See the 2026-08-03 entry
> at the end of this document.

**QB CHANGE IS REAL, AND IT IS THE FIRST NFL ANGLE TO CLEAR ALL FOUR GATES
(2026-07-29, `nfl_qb_experiment.py`).** The absorption theorem says exactly which QB
question is worth asking. Team Elo can write itself as R_t = β·(QB quality)_t + skill_t,
so QB QUALITY is absorbable — which is why the EPA batch's per-QB EPA blend died 1/6
seasons and why that death said nothing about the blind spot. What a running average
over a team's history provably CANNOT represent is that the man who earned the rating is
not playing tonight. That is a CHANGE, not a level, and it is not absorbable.

`QBNEW` = share of the last 8 starts taken by someone other than tonight's starter,
signed home-minus-away. Panel 4,350 non-tie games 2010-2025, train 2,662 / holdout
1,688, baseline byte-identical to production (asserted: p_base reproduces
`nfl_model.run_elo` to max |dp| = 0). 18.6% of games carry a starter change on one
side; 65.5% carry some discontinuity in the 8-game window.

- GATE 3 (replication): +0.00685 holdout LL/game, 5/6 seasons, ROBUST WIN. With raw QB
  experience moved INTO the baseline (pass 2, the test that kills a change term that was
  really impersonating "the backup is inexperienced") it survives at +0.00447, still 5/6.
- GATE 1 (power): oracle at plant b=0.60 is +0.01015 and a planted effect was recovered
  3/3. The measured +0.00685 sits UNDER the oracle, so it is not a power artefact — the
  ceiling is not the binding constraint here. This is the first batch where gate 1 came
  back PASSED rather than DEAD or UNINFORMATIVE.
- GATE 2 (placebo): 0/200 within-season shuffles reached the measured gain, p=0.0050.
- GATE 4 (shape): on the 2,070 games with a discontinuity on exactly ONE side the effect
  is 2.5× stronger — +0.01134, and 6/6 seasons. The effect lives where the claim says.

`QBCHG` (plain since-last-game flag) also clears three gates (+0.00377 → +0.00240 pass 2,
p_eff 0.0050) but its shape read is only 4/6, and QBNEW beats it on both the real panel
and the synthetic one — a benching or injury costs a team for a STRETCH, and a window
fraction reads a stretch while a since-last-game flag only reads its first game.
`QBEARN` (new QB × how far the rating sits above 1500 — the absorption-aware version,
charging only teams whose rating was actually earned) is 6/6 in both passes at +0.00435,
which is the cleanest per-season record of the four, but it is nearly collinear with
QBNEW and buys nothing on top. `QBRES` (per-QB Elo minus team Elo) is a *level* dressed
as a residual: the refit baseline absorbs a plant at every strength down to b=0.15, so
its probe is uninformative and its 3/6 pass-2 record is not a verdict. **To ship: QBNEW,
one term.**

**THREE READER BUGS FOUND BY THE FIRST NFL ANGLE THAT ACTUALLY WON.** Every gate-reading
routine in this project was written against batches where the measured effect was ~zero,
and all three broke the first time a real one came through. (1) The ceiling reader's
last rung said "a planted effect was recovered 3/3, so a real one would have shown →
DEAD" — valid logic about a NULL, nonsense about a robust win, and it labelled QBNEW's
+0.00685 DEAD. It now takes gate 3's verdict as an argument. (2) The placebo counted how
often a shuffled column earned a ROBUST WIN, which is the SHIP RULE's false-positive
rate — a property of the gate, not the angle. That is why four angles with measured gains
spanning 7× all came back p ≈ 0.065-0.10. It now reports both: `alpha` (the gate's FPR,
~0.06-0.10 here, so a ROBUST WIN on its own is worth about p=0.08 and no more) and
`p_eff` (how often a shuffle reached the MEASURED gain — 0.0050 for the three
candidates). Read p_eff. (3) The plant ladder only stepped DOWN, for absorbed probes. An
oracle can also land BELOW the measured gain when the column is zero on most rows — that
is a probe calibrated too weakly, not a noisy measurement, and it mislabelled QBEARN
"noise by construction". The ladder now escalates to 0.90/1.40/2.00 until the bound
actually bounds. **All three were silent, all three read as "dead angle", and the fix in
each case makes the file MORE likely to report a win — which is the direction a
reader bug is least likely to be noticed from.**

**BLIND SPOTS:** all injuries other than the QB · WHICH backup (QBNEW knows a change
happened, not whether the replacement is competent — the natural next angle, and note
that the replacement's own quality is a level and therefore partly absorbable) · weather
as game-total context · market prices (deliberately excluded; disagreement vs market is
the displayed edge). QB change came OFF this list on 2026-07-29 — it was the #1 entry,
it is now measured, and it ships.

## Soccer (soccer_model/publish)

**IN:** Dixon-Coles fit on **xG** wherever understat joins (98% of matches — validated:
beats goals-fit 3/3 seasons, holdout Brier +0.0064) with goals fallback · time decay ·
home advantage · rescheduled matches settle within ±45 days · market anchors displayed.

**BLIND SPOTS:** lineups/rotation (biggest one — cup-week squad rotation) · injuries ·
manager changes · promoted teams start near league-average · congestion/fatigue ·
weather.

## UFC — research blend (ufc_blend_predict.py, in live A/B vs production)

**IN:** Elo (K=96) · striking + grappling margin EMAs · control · sub-threat EMA ·
experience & layoff · **AGE from DOB** (validated: −0.0095 Brier, the biggest UFC
finding) · **CHIN — cumulative KO/TKO losses** (validated vs Elo-only +0.0089
LL/bout AND vs Elo+age +0.0052, 3/3 periods both: damage accrual is real beyond
the age curve) · namesake guard (no more 2006 Rick Davis) · A/B graded
automatically against production every card.

**LAYOFF × AGE — the first angle to clear all three gates (2026-07-29).** Time off
costs an older fighter more than a younger one: layoff × (age − 30), holdout
+0.00213 on the age-complete subset, 3/3 periods, negative sign. Ceiling +0.00645,
so it is measuring at a third of what the panel could possibly show — the profile
of a real, modest effect. 0/24 shuffled placebos fired the ship rule and none beat
it — extended to 300 within-year permutations the joint placebo gives p=0.0033, and
it is a robust win in all three independent eras. Gate 1 (the power ceiling) is
structurally UNREADABLE for this term: the refit baseline absorbs any plant built
out of layoff and age, so the oracle comes back non-positive at every strength. That
is a broken probe, not a dead angle — gates 2 and 3 carry it. Shape holds: 3.4×
stronger per bout where a long gap actually exists (+0.00413
vs +0.00120 on even turnarounds), 3/3 when someone is 33+, 2/3 when both are under
31, and the pivot sweep peaks near 33 and decays to negative by 50 rather than
climbing forever — so it is not quietly repairing the layoff main effect. Pivot
stays at 30; the sweep is a shape check, not a tuning run.

**LAYAGE IS THE ONLY SURVIVOR IN SIX ROUNDS, AND ACTIV IS RETROACTIVELY DEAD
(2026-07-30).** The ceiling ladder had been planting effects at the STRONGEST grid
value and stepping DOWN only when the oracle came back non-positive — which answers
"could an effect ten times what the data claims be seen?", a question nobody asked.
Planting at the FITTED magnitude and stepping UP is the correct ladder. Under it ACTIV
measured +0.00063 against an oracle of +0.00014 for a true effect of its own size —
i.e. four times what FULL KNOWLEDGE of that effect is worth, which is noise by
construction. LAYAGE is unaffected (it already fit at the smallest magnitude on its
grid, so the old ladder had already stepped down to it). **A measured gain ABOVE the
oracle bound is not a strong result, it is a disproof.**

**ROUND 6 — SIX ACCUMULATORS, NOTHING SHIPS, AND THE TWO NEGATIVES ARE THE VALUE.**
CAGE (career cage time) won the holdout 3/3 and became a 0/3 NULL the instant plain
fight count entered the baseline: it was **experience in a wig**. Every accumulator
round from here must carry EXPER underneath it. MILEAGE (strikes absorbed) is the
sharper lesson: holdout +0.00322, 3/3 periods, 240 placebo shuffles never once reached
it (p=0.0083), and a perfectly monotone shape (quintile win rates 59.5 / 54.8 / 49.1 /
43.7 / 36.2). It still fails, on the two gates that decide: a true effect of its own
fitted size is recovered **0/12** seeded plants, so the gain cannot be one being
detected; and it replicates in exactly ONE era — the one the ship rule tunes on.
**An era-specific miscalibration of the BASELINE that happens to correlate with a
career accumulator will pass a placebo and pass a shape check and still not be real.**
That is what gate 3 is for. (KDABS unreadable; WTUP/WTNEW dead; QUICK unreadable
rather than refuted — only 287 of 6283 rows are non-zero.)

**PRODUCTION'S LEDGER p1 IS MIS-SCALED — IT EMITS ~0.5 FOR EVERYTHING (2026-07-30).**
`build_site.py` logs `p1 = σ(s1 − s2)` from `model_score` in `ufc_ratings.json`. That
treats a skill index as a log-odds with an implicit coefficient of exactly 1.0, and
nobody ever fit it. `model_score` has SD 0.421 over 1228 fighters, so a matched pair
differs by well under a logit and 13 of 14 logged predictions land between 0.45 and
0.66 (full range 0.269–0.657, SD 0.087). Regressing logit(model) on logit(market)
gives slope **0.152, SE 0.116** — production's number barely responds to the market's
at all. The research blend on the same card gives slope 0.688, SE 0.136, range
0.204–0.822: it is genuinely discriminating and merely shrunk. **Consequence: the live
A/B is not a fair contest between two models, it is a model versus a broken readout,
and production's apparent "disagreements with the market" are ignorance, not signal —
they must never be traded on.** The fix is blocked, not merely undone: `model_score`
is CURRENT-state, so fitting the coefficient on bout history leaks (that leaky fit
wants β≈2.3 and is an upper bound). It needs a walk-forward score or a market-anchored
scale, and until then the blend is the only UFC number worth reading.

**WHY MOST UFC ANGLES DIE — THE ABSORPTION THEOREM.** Elo scores a bout on
`r_i − r_j`. If the truth is `z = β·(x_i − x_j)` for ANY per-fighter quantity x, then
`r_i = β·x_i + skill_i` reproduces it exactly and the ratings converge there unaided.
So Elo absorbs any per-fighter *difference* term, not merely a fixed offset. A null on
a raw trait therefore says almost nothing about the sport — it says Elo already knows.
Only terms that vary WITHIN a career (mileage, layoff, age, activity), or that key on
where Elo is not yet informed, can leave a residual. This is the common thread under
win-streak momentum, KDABS, ABSORB, KO-power and raw reach all failing, and it is the
reason LAYAGE — two inputs that both move within a career — is the one that lived.

**THE THEOREM EXTENDS TO TYPES, AND THERE IT LEAVES A GAP (batch 5, 2026-07-29).**
Stance is not a number, it is a type, so a stance effect is an antisymmetric 3×3
matchup matrix M(s_i,s_j) with THREE free values: M(O,S), M(O,W), M(S,W). Everything
Elo can absorb has the form f(s_i) − f(s_j), which has only TWO free values and
therefore satisfies `M(O,W) = M(O,S) + M(S,W)`. The failure of that identity is the
one direction no Elo-family rating can ever hold: a rock-paper-scissors CYCLE. With k
types the absorbable subspace is k−1 dimensional inside a k(k−1)/2 dimensional space,
so every categorical feature with 3+ levels has a non-absorbable residual worth
testing. That is the general lesson; the specific UFC answer is under TESTED & DEAD.

**ABSORPTION IS A LIMIT, NOT AN INSTANT.** Every fighter debuts at 1500 regardless of
his frame or his stance, careers are short, and the inflow of debutants never stops —
so a fraction of any panel is always un-absorbed. Measured, not assumed: a 0.55-logit
per-fighter southpaw bonus planted into a synthetic roster with realistic turnover
comes back as a genuine ROBUST WIN on the plain indicator (+0.0063), and the
ignorance-weighted version reads it better still (+0.0068). Practical consequence: a
win on an "absorbable" control is evidence about Elo's convergence SPEED, not about
the trait — and the `TAU/(TAU + bouts)` weighting is the right way to isolate the
un-absorbed part.

**TESTED & DEAD:** reach · finish-aware K updates · opponent-adjusted striking ·
heavyweight-variance myth (HW favorites are RELIABLE; middleweight is the chaos
division) · win-streak momentum (beats Elo-only but only 2/3 age-adjusted — it's
mostly a youth/quality proxy, age already carries it) · WEAR-AND-TEAR BATCH, all
four dead on the age-complete subset: knockdowns ABSORBED (finer-grained than chin,
which only counts KO *losses*) 1/3 periods · career MILEAGE in hours fought 2/3 ·
significant strikes ABSORBED per minute 1/3 · weight-class CHANGE vs last bout, NULL
both ways. Read: **chin is the accrued-damage variable** — once chin and age are both
honestly in the baseline, finer trauma proxies add nothing. Method note that cost an
hour: a plain .lower() name join matched only 51% of fighters, which silently zeroed
the age term and made three of these look like ROBUST WINS. A diluted control is not
a control; the verdict is the AGE-COMPLETE subset (n=4,041 bouts, 1,486 holdout —
small, so these are "no evidence", not "proven zero"). Re-read at 49% DOB coverage:
all four still non-robust, so the verdict holds at higher power. Batch 3 added two
more deaths — career KO SHARE OF WINS (0/3, and it reads "cannot be seen" on the
full sample, so it is filed as unanswered rather than dead) and SIG STRIKES LANDED
PER MINUTE (2/3, against a fat +0.0204 ceiling — the panel could easily have seen
pace and did not). ACTIVITY (bouts in the last 365 days) is the instructive one: a
ROBUST WIN on the full sample that collapsed to 0/3 NULL on the age-complete subset.
It was an age proxy. That is precisely what the subset test is for. Re-run once more
at 97% DOB coverage, ACTIV is positive in 3/3 eras with a joint placebo of p=0.0133 —
a real candidate, but it does not clear the bar to ship, so it stays out.

**REACH IS DEAD (batch 4, 2026-07-29).** Five reach angles, two passes. Raw reach,
reach × grappling, reach × age and the height-adjusted version all die against
ceilings several times the measured gain. RCHNEW (reach weighted toward fighters Elo
has barely seen) looked like a robust win in the first pass and evaporated in the
second: once raw reach itself is in the baseline, RCHNEW is raw reach in a wig. Only
RCHENV — reach advantage conditioned on the environment it is used in — is genuinely
invisible rather than dead (a planted effect of its own size was recovered only 1/3),
so it is unanswered, not buried. The two-pass design is what caught this: any term
built on top of a main effect will impersonate that main effect unless the main
effect is already spoken for.

**STANCE IS DEAD (batch 5, 2026-07-29) — including the one direction Elo cannot
represent.** The round-8 sweep took holdout stance coverage from 1.0% (18 bouts, no
panel at all) to 91.8% (1,626 of 1,771), so the oldest folk claim in the sport was
finally testable on the decisive subset (n=7,108 age- AND stance-complete). Six
angles: the southpaw indicator −0.00012 against an oracle of +0.01351; the switch
indicator −0.00040 vs +0.00712; SPFAM (he has not seen that look, keyed on the
opponent's exposure to the stance rather than on stance itself) +0.00011 vs +0.01101;
and the CYCLE — orthodox > southpaw > switch > orthodox, the rock-paper-scissors
component that provably cannot be written as f(i) − f(j) and that no Elo-family
rating can hold — +0.00026 against an oracle of +0.01818. A planted effect of each of
those sizes was recovered 3/3, so a real one would have shown. Southpaw advantage is
not in this panel and the panel had 50–140× the power needed to see it. Two
caveats kept honest: SPNEW (stance where Elo is blind) reads NULL but a planted
effect of its own size was itself only recovered 1/3, so it is invisible, not dead;
and XSTNC's probe is uninformative because the refit baseline absorbs the plant at
every strength. Confirmation that the algebra is right: CYCLE's verdict barely moved
between the two passes (+0.00026 → −0.00013), which is exactly what orthogonality to
both indicators predicts.

**TWO METHOD LESSONS THAT COST REAL TIME.** (1) GRID EDGES. If a baseline
coefficient fits to the boundary of its search grid, the optimum was never searched
and the baseline is handicapped — any correlated angle then inherits credit that is
not its own. Every experiment now flags boundary fits, and a fit pinned at a grid
maximum is a CENSORED estimate that cannot be compared to another censored estimate.
(2) A NON-POSITIVE ORACLE IS A BROKEN PROBE, NOT A DEAD ANGLE. The ceiling probe
refits the baseline on the synthetic panel; if the plant correlates with baseline
terms, a well-specified baseline absorbs it, the refit carries it through inflated
main effects, and adding the true coefficient double-counts and HURTS. Both current
experiment files walk the plant strength down looking for one the baseline cannot
absorb, and print PROBE UNINFORMATIVE rather than a verdict when none exists.

**THERE IS NO PRIME (2026-07-29).** Age was validated on the subset with DOB *and
reach* for both corners — right when age and reach were one experiment, wrong now
that reach is dead, because it was still reporting "46% coverage" when DOB coverage
was 97%. Re-read on the wide panel: AGE HOLDS, −0.0088 Brier on 2,144 holdout bouts
(was 1,768), 3/3 periods, bootstrap [−0.0114, −0.0061]. But the age *curve* is not a
curve. The fitted quadratic peaks at −15.5 years against an observed range of
21.9–41.0, and buys −0.0002 Brier over a plain linear term. Over every age anyone
actually fights at, aging is a straight decline: there is no peak, no protected
prime window, and the shipped `age2_diff` is inert. A hinge sweep (does aging
ACCELERATE, in a shape the data can express) puts the best knot at 29 for −0.0005
over linear — an interior optimum, so real, but far too small to ship — and the
extra slope reverses at knot 38, which is survivorship: the men still fighting at
38 are the exceptional ones. Practical read for a card: a 24-year-old has no bonus
for being "in his prime", and a 34-year-old's decline is not a cliff. It is one
straight line, and Elo does not already know it.

**DATA GAP — CLOSED (round 7–8 backfill).** DOB coverage went from 47% of bouts with
BOTH ages to 97% (8,439 of 8,686 unique bouts, 1994-03-11..2026-06-14), and stance
from effectively nothing to 82% overall / 91.8% on the holdout. Every UFC verdict
above the batch-2 line was re-read at the higher power. Remaining: the shipped AGE
term and model A were tuned at 47% coverage and have not been re-validated at 97%.

**BLIND SPOTS:** short-notice replacements (Guskov-type — huge, manual) · weight-cut /
camp news · suspensions & why (the Temirov TMZ read was yours, not the model's) ·
round-by-round and method (blend prices win% only). Stance/southpaw matchups came
OFF this list on 2026-07-29 — they were tested four ways and are dead, not blind.

---
*The graders behind all four are leak-free and self-testing; every claim above traces
to an experiment verdict in the repos' experiments/ folders or a commit message.
When in doubt: the model handles the base rates — YOU own the blind-spot reads.*

**HOW A WIN IS NOW JUDGED (added after two fake ROBUST WINS in two days).** The old
gate was train win + holdout win + 3/3 sub-periods. At margins under ~+0.0005 that is
not enough, because the tuning grid can find a lucky cell. Two cheap checks now run
alongside it:
1. **POWER CEILING** (`power_ceiling()`) — re-roll the outcomes on the REAL panel so
   the angle IS true at a generous strength, then score at the true value. No fitted
   model can beat that number. A result at or above its own ceiling is noise by
   definition; that is how within-season familiarity was caught. A null far BELOW its
   ceiling means "we could not have seen it", which is not the same instruction as
   "it is not there" and must not be filed as dead.
2. **SHUFFLED PLACEBO** (`contactform_placebo.py` pattern) — permute the feature
   across subjects and re-run the entire tune-and-verdict pipeline N times. Reports
   how often the ship rule fires on pure noise.
3. **CROSS-SEASON REPLICATION** (`mlb_widepanel_experiment.py`, added 2026-07-29) —
   the win must clear baseline in the 2024 holdout AND the 2025 holdout SEPARATELY,
   not just pooled. Three consecutive ten-day slices of one June share a weather
   regime, a league ball, hot bats and park conditions; two different Augusts do not.
   This is the tooth that more rows alone cannot buy: it would have killed the 60-day
   hard-hit result without needing a placebo.
Also: check whether the effect's shape makes sense. Rolling contact form spiked at 7d
and 60d with a dead middle — a real signal does not do that.

**READ THE CEILING IN THE RIGHT ORDER (this bug has now been shipped and fixed twice,
once in UFC and once in MLB).** Test "did it win" BEFORE "is the oracle too small to
see". Getting it backwards prints a genuine cross-season robust win — one measuring
six times its own detection threshold — as "cannot be seen". The correct ladder:
measured ≥ oracle → noise by construction; measured inside the oracle's seed spread →
unproven, needs a placebo; won and above the FITTED threshold → live; did not win and
oracle tiny → cannot be seen, do not bury; otherwise dead.

**ONE RE-ROLL IS NOT A CEILING.** On an identical planted truth, four seeds gave
+0.00076 / +0.00451 / +0.00290 / +0.00396 — a 5× spread. Ceilings are averaged over
seeds with the spread reported, and a measured result that lands inside that spread is
UNPROVEN, not confirmed. Two more mechanics that bite: plants must be mean-normalised
(or the plant shifts the league level and the baseline re-fit steals the credit), and
plant randomness must use `zlib.crc32`, never `hash()` — CPython salts string hashing
per process, so ceiling seeds would silently measure different planted worlds.

**SCORE EVERY ROW YOU OWN.** Every MLB HR verdict through batch 2 was decided on ONE
holdout — June 2025, 8,037 batter-games — while 49,562 rows of 2024 sat on disk used
only to WARM the shrinkage cells, never scored. Widening the scoring windows
(warm 2024-03-20..05-31; train 2024-06..07 + 2025-04..05; hold 2024-08..09 + 2025-06)
quadrupled the holdout to 24,332 rows with ZERO new data pull. Before filing anything
as dead, check that the panel is actually being scored, not just warmed.

---

## 2026-08-03 — five findings from a full-day pass over every tool

**A FEATURE CAN BE PERFECTLY TIME-ORDERED AND STILL BE A PHANTOM. GATE 0:
AVAILABILITY.** Every leak proof written before today tested one thing: does a
future result move a past feature. That is necessary and it is not sufficient.
`QBNEW` passed all four gates — +0.00447 after the hardest baseline, 5/6
seasons, placebo 0/200, effect 2.5× stronger where the claim says it lives —
and reads `home_qb_id`, a column that is populated on **7,276 of 7,276 played
games and 0 of 272 unplayed games**. Nothing in the harness could see that,
because the harness only ever scored rows that had already been played. A
feature like this is not wrong in backtest; it is inert live, which is worse,
because the backtest keeps promising a gain the shipped model never delivers.

The gate is `columns_touched()` in `nfl_qb_experiment.py`: it intercepts
`itertuples`, `__getitem__`, `__setitem__` and `merge(on=/left_on=)` to
discover STRUCTURALLY which raw columns a feature builder reads, then verifies
that trace by BLANKING each un-traced suspect column and re-running — if the
feature moves, the trace missed a `.loc`/`.iloc` path and the run fails. It is
built on interception rather than declaration because a future feature cannot
forget to register itself. `verdict_of()` raises rather than judging an
unaudited column. Same gate now on `nfl_experiment.py` and
`nfl_epa_experiment.py`. Audit result: **no currently-shipped model input is
unavailable** (`spread_line` is 75.4% null but is display-only and already
guarded at `nfl_model.py:135`).

Side effect worth noting: separating the calibration counters while wiring this
in fixed a pre-existing degeneracy and flipped probe plant recovery 0/3 → 3/3,
turning one verdict from "STILL CANNOT BE SEEN — do not bury" into "DEAD".

**A SCORE IS NOT A LOG-ODDS, AND A TOOL THAT NEVER HAS AN OPINION CANNOT BE
CAUGHT BEING WRONG.** `model_score` in `ufc_ratings.json` is documented as
"logit units". It is an opponent-adjusted ridge coefficient normalised within
division; sd = 0.421. Every consumer fed it to a bare sigmoid, i.e. an implicit
temperature of 1 — the ledger in `build_site.py`, the ledger in
`refresh_odds.py`, and **six independent inline sites in the front end**. Across
24 logged bouts the mean distance from a coin flip was **0.049**; the most
lopsided call the tool had ever published was 65.7% while the market on the same
card ran to 80.8%. The production ledger read n=11, accuracy 36.4%, and the
model was right on 0.0% of its 5 disagreements with the market. The page copy
explained the flatness away as MMA variance.

Nothing threw. This is the failure mode a validator has to be *told* to look
for, so `validate_build.py` now hard-errors on a missing T, on `|T-1| < 0.05`,
and on "the widest score gap in the whole file still prices under 70%".

**TO PICK A CALIBRATION CONSTANT, FIT IT ERA BY ERA AND WATCH FOR A GRADIENT.**
`calibration.json` carried T=3.17 and applying it would have overshot by ~60%.
The ratings are recency-weighted AND fit on the same bout history they are
scored against, so a fight's own result is partly baked into its participants'
current scores — more so the fresher the fight. Fitting T per era exposes it:

    2005-2012  T=2.05  acc 62.0%      2020-2022  T=2.14  acc 68.2%
    2013-2016  T=2.02  acc 65.2%      2023-2024  T=2.87  acc 75.5%
    2017-2019  T=1.91  acc 67.1%      2025-2027  T=3.86  acc 76.8%

T and accuracy climbing together is memory, not skill. So T is fit on the
OLDEST cohort (`FIT_ERA = (2005, 2019)`, T=1.981), which biases it DOWN, and
down is the safe direction for a betting tool. Sanity check that it landed:
the honest walk-forward reports 60.7% accuracy and the fit era reports 62–65% —
the same model, not the 76% fantasy of the recent slice. `load_T()` **raises**
rather than defaulting to 1.0, because a silent default here is invisible.

**SEASON-BOUNDARY CONTAMINATION DOES NOT DILUTE THE EFFECT UNDER TEST — IT
FLATTERS IT.** `mlb_fatigue_experiment.py` computed rest-day gaps across the
winter, filing a ~190-day layoff in the same bucket as a genuine 8-day one. The
reflex is to call that noise. It is not: a season opener genuinely IS a fresh
bat, so the contaminated rows agreed with the hypothesis for the wrong reason.
636 → 128 contaminated observations, 508 → 0 season-crossing. Rows carry no
`year`, only an ISO `date`; an MLB season never straddles a calendar year, so
`date[:4]` IS the season. The identical pattern on pitchers at
`mlb_kfactors_experiment.py:142` is now fixed too — it was unreachable on a
single-season panel, which is exactly why nobody would have looked there after
widening the window.

**RANK BY WHAT HITS, NOT BY WHAT IS CHEAP — AND PUBLISH THE HIT RATE NEXT TO
IT.** Every MLB board sorted by edge/EV/Kelly. All four now rank by hit
probability with edge retained as a sortable column, and each row carries a
`has hit` column: the historical settle rate of the bucket that row falls in
(5-point buckets, suppressed to "—" under 30 graded observations rather than
printing a misleading small-sample rate). The first thing it showed:
**the board's 30–35% calls have historically come in at 18.1%.**

**PARLAY ARITHMETIC BELONGS IN ONE FILE (`parlay/slips.py`).** Slips were being
graded by throwaway scripts that each re-implemented the de-vig. Three things
are now permanent and tested (25/25):

- **Power de-vig is the default** (`board.py METHOD`, changed from `mult`).
  Multiplicative de-vig splits the overround in proportion to implied price,
  but books load margin onto the LONGSHOT side. A -5000 leg de-vigs to .9363
  under `mult` and .9737 under `power`. On a board that is nothing but heavy
  favourites, that bias is largest exactly where the tickets differ.
- **Same-market outcomes are handled exactly, not simulated.** Two outcomes of
  one market either nest (by-points ⊂ wins the fight → the intersection is the
  narrower one) or are disjoint (→ zero). P(at least one slip cashes) is
  inclusion–exclusion over the slips, exact, with `1 - ∏(1-pᵢ)` printed beside
  it so the cost of shared legs is visible. Cross-checked against a Monte Carlo
  that lays each market out on one uniform interval — rolling an independent
  coin per outcome lets a fighter both win and lose the same fight, which makes
  the simulation *more optimistic* than the arithmetic it is checking.
- **E[slips killed]** per shared leg = (tickets carrying it) × P(it loses). On
  the 2026-08-01 set of five, that number said 1.20 for one fighter before the
  card started; all five tickets died together on that fight. Kept as the
  regression case in `slips_2026-08-01.json`.

**STILL OPEN — THE UFC COVERAGE HOLE.** Only 2 of 13 bouts on the 2026-08-01
card resolve into the ledger; the other 11 fall outside the `ufc_bouts >= 5`
rating floor, and those are precisely the fighters actually being bet. The tool
silently OMITS them rather than saying "no opinion", which reads as coverage it
does not have. Raising the floor means inventing a prior for debutants — a real
modelling decision. Surfacing "no opinion" explicitly in the UI does not, and
should happen first.
