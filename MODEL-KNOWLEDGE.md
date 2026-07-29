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
Both live on in `mlb_fatigue_experiment.py`. When porting a narrow-panel angle wide,
**re-read its feature builder for season-boundary assumptions before trusting the port.**

**BLIND SPOTS:** umpire assignment · travel/getaway days · lineup changes after build
(spot tags flag rookies/new bats, not scratches) · anything intra-day (board is
pregame).

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

**BLIND SPOTS:** who the quarterback actually is (injuries/benchings — the team rating
carries it with a lag) · all injuries · weather as game-total context · market prices
(deliberately excluded; disagreement vs market is the displayed edge).

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
it. Shape holds: 3.4× stronger per bout where a long gap actually exists (+0.00413
vs +0.00120 on even turnarounds), 3/3 when someone is 33+, 2/3 when both are under
31, and the pivot sweep peaks near 33 and decays to negative by 50 rather than
climbing forever — so it is not quietly repairing the layoff main effect. Pivot
stays at 30; the sweep is a shape check, not a tuning run.

**WHY MOST UFC ANGLES DIE — Elo absorbs any FIXED per-fighter trait.** Elo estimates
*total* strength, so a constant per-fighter contribution to winning is folded into
the rating within a handful of bouts and leaves no residual for a second term. A
synthetic plant of a fixed trait is unrecoverable over an Elo baseline — the harness
refusing it is correct behaviour, not a bug. Only things that move WITHIN a career
(mileage, layoff, age, activity) are visible at all. This is the common thread under
win-streak momentum, KDABS, ABSORB and KO-power all failing, and it is the reason
LAYAGE — two inputs that both move within a career — is the one that lived.

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
It was an age proxy. That is precisely what the subset test is for.

**DATA GAP (fixable, highest-value UFC work):** the DOB cache covers 1,637 fighters
but the bout file has 2,678 names, so only 47% of bouts have BOTH ages. Every UFC
experiment runs against a half-strength age control until that pull is widened.

**BLIND SPOTS:** short-notice replacements (Guskov-type — huge, manual) · weight-cut /
camp news · suspensions & why (the Temirov TMZ read was yours, not the model's) ·
southpaw/style matchups · round-by-round and method (blend prices win% only).

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
