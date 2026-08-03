======================================================================
MLB HR ANGLES 3 — FATIGUE (rest, workload) and FAMILIARITY (Nth look)
baseline = shipped run-2 analog incl. prior-seeded pitcher HR
======================================================================
rows 74018  rest known 72773 (98%)  repeat-starter meetings 12277 (17%)
season-opening rows with no defined prior gap: 508 (these used to score as ~180 days of rest)

--- FULL SAMPLE
baseline [all]  TRAIN -0.30569 (n=16419)  HOLDOUT -0.32504 (n=8037)
REST    days off           w=-0.20  train_win=True   holdout dLL -0.00022  periods 1/3  -> NULL
LOAD    7d PA vs own norm  w=-0.20  train_win=True   holdout dLL -0.00019  periods 1/3  -> NULL
FAMIL   Nth look, season   w=+0.12  train_win=True   holdout dLL +0.00004  periods 3/3  -> ROBUST WIN
FAMCAR  Nth look, career   w=+0.03  train_win=False  holdout dLL +0.00000  periods 2/3  -> win, not robust

--- REGULARS ONLY (>=300 prior panel PA) — the proxy guard.
    On the full sample 'rested' partly means 'bench bat', and bench
    bats homer less because they are worse, not because they rested.
baseline [regulars]  TRAIN -0.33119 (n=11639)  HOLDOUT -0.35074 (n=6221)
REST    days off           w=-0.06  train_win=True   holdout dLL -0.00008  periods 1/3  -> NULL
LOAD    7d PA vs own norm  w=-0.12  train_win=True   holdout dLL -0.00011  periods 1/3  -> NULL
FAMIL   Nth look, season   w=-0.03  train_win=True   holdout dLL -0.00001  periods 1/3  -> NULL
FAMCAR  Nth look, career   w=-0.03  train_win=False  holdout dLL -0.00000  periods 1/3  -> NULL

--- POWER CEILING — how big could each angle possibly read?
    Real panel, HR outcomes re-rolled so the angle IS true at a
    generous strength, then scored at the true value. No fitted model
    can beat this. A null far below its ceiling is uninformative.
    REST    +/-40%         ceiling +0.00035 LL/game
    LOAD    +/-40%         ceiling +0.00046 LL/game
    FAMIL   +/-30%/look    ceiling -0.00001 LL/game
    FAMCAR  +/-30%/look    ceiling +0.00082 LL/game

Ship rule: ROBUST WIN (train + holdout + 3/3), and any margin under
+0.0005 must additionally clear a shuffled placebo before it counts.

========================================================================
READING THIS
========================================================================
Nothing ships. But the four results are not the same KIND of null, and the
power ceiling is what separates them.

FAMIL "ROBUST WIN" IS FAKE, AND THE CEILING PROVES IT IN ONE LINE. It scored
+0.00004 on the full sample with 3/3 periods. Its ceiling is -0.00001. The
angle measured LARGER than the absolute maximum it could produce if it were
completely true — which is only possible if what was measured is noise. It
also dies outright on the regulars-only pass. This is the second fake ROBUST
WIN in two days (hard-hit 60d was the first). The ship rule alone is not
sufficient at these effect sizes; the ceiling and the placebo are now both
part of the gate.

REST AND LOAD ARE UNINFORMATIVE NULLS, NOT DEAD ANGLES. Both read about
-0.0002. Both ceilings are +0.00035 and +0.00046. Even a genuine +/-40% effect
would barely clear the noise floor here, so measuring nothing tells us close to
nothing. The reason is structural: ~83% of games follow a game, so the rest
factor is 1.0 on most rows and there is nothing to grade. Do not record these
as "tested and dead" — record them as "cannot be seen at this sample". They
would need a multi-season panel to be answerable, which is a data project, not
a modelling one.

FAMCAR IS THE ONE REAL NULL. Career meetings with a starter has an +0.00082
ceiling — meaningful capacity — and it measured +0.00000 with the fitted weight
pinned at the grid floor. That is a genuine "this is not there". Batters do not
get measurably better at homering off a starter they have seen before, once
their own rate, the park, platoon and the pitcher's HR tendency are already
priced.

METHOD ADDED HERE, AND IT IS THE MAIN TAKEAWAY: power_ceiling() re-rolls HR
outcomes on the REAL panel so an angle is true at a stated strength, then
scores it at that true strength. No fitted model can beat that number. Running
it BEFORE believing any null costs seconds and turns "we found nothing" into
either "it is not there" or "we could not have seen it" — which are completely
different instructions about whether to come back to the idea.

========================================================================
SEASON-BOUNDARY REPAIR — what moved, and what did not
========================================================================
build_history() used to compute rest as a raw calendar difference between a
batter's consecutive appearances, with one last-game slot per batter and no
season in the key. On this panel (2024 burn-in + 2025) that made the WINTER
look like rest: 508 rows — every returning batter's first game of 2025 —
carried a gap of 150 to 423 days, and factor() caps rest at 5, so all 508 voted
at the very top of the scale. 636 gaps exceeded 45 days before the repair; 128
do now, and all 128 are within-season injury/IL/call-up returns, which is a
separate question this file does not claim to answer. The rest gap is now
undefined (None, no opinion) across a season boundary, and the trailing 7d/30d
workload accumulator is season-keyed for the same reason.

NO VERDICT CHANGES. All four angles land in the same category they did before.
REST moved -0.00020 -> -0.00022 on the full sample and -0.00016 -> -0.00008 on
regulars (fitted w relaxed from -0.12 to -0.06, periods 0/3 -> 1/3): still
NULL, and still an uninformative one. The contamination was mildly PUSHING the
rest term, which is the direction that matters — a season-opener is a fresh
bat, so the bug was quietly correlated with the thing being measured rather
than being clean noise.

ONE STATED CAUSE IN THIS DOCUMENT WAS WRONG AND IS NOW CORRECTED. The FAMIL
paragraph above used to explain its ~0 ceiling with "just 4% of panel rows are
a repeat meeting with the same starter". That 4% was itself a bug: the
within-season counter keyed off a hardcoded `d >= "2025-01-01"`, so every
burn-in row was pinned to fam=0 by construction. Keyed on the season instead,
the panel-wide figure is 17%, and the scored June-2025 holdout was always at
19.6%. The CONCLUSION is untouched — measured still exceeds the ceiling, so
FAMIL is still noise — but it is not noise for lack of coverage, and the 4%
number should not be quoted anywhere. Coverage on the scored windows:
  HOLDOUT Jun-2025   n=8037   rest 99.4%   fam 19.6%   famcar 40.8%
  TRAIN   Apr-May25  n=16419  rest 96.7%   fam  7.2%   famcar 37.0%
