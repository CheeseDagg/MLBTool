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
year-matured cells are a robust 3/3-period holdout win) · monthly recalibration
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
small, so these are "no evidence", not "proven zero").

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
Also: check whether the effect's shape makes sense. Rolling contact form spiked at 7d
and 60d with a dead middle — a real signal does not do that.
