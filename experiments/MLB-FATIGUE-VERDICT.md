======================================================================
MLB HR ANGLES 3 — FATIGUE (rest, workload) and FAMILIARITY (Nth look)
baseline = shipped run-2 analog incl. prior-seeded pitcher HR
======================================================================
rows 74018  rest known 73281 (99%)  repeat-starter meetings 2762 (4%)

--- FULL SAMPLE
baseline [all]  TRAIN -0.30569 (n=16419)  HOLDOUT -0.32504 (n=8037)
REST    days off           w=-0.20  train_win=True   holdout dLL -0.00020  periods 1/3  -> NULL
LOAD    7d PA vs own norm  w=-0.20  train_win=True   holdout dLL -0.00019  periods 1/3  -> NULL
FAMIL   Nth look, season   w=+0.12  train_win=True   holdout dLL +0.00004  periods 3/3  -> ROBUST WIN
FAMCAR  Nth look, career   w=+0.03  train_win=False  holdout dLL +0.00000  periods 2/3  -> win, not robust

--- REGULARS ONLY (>=300 prior panel PA) — the proxy guard.
    On the full sample 'rested' partly means 'bench bat', and bench
    bats homer less because they are worse, not because they rested.
baseline [regulars]  TRAIN -0.33119 (n=11639)  HOLDOUT -0.35074 (n=6221)
REST    days off           w=-0.12  train_win=True   holdout dLL -0.00016  periods 0/3  -> NULL
LOAD    7d PA vs own norm  w=-0.12  train_win=True   holdout dLL -0.00011  periods 1/3  -> NULL
FAMIL   Nth look, season   w=-0.03  train_win=True   holdout dLL -0.00001  periods 1/3  -> NULL
FAMCAR  Nth look, career   w=-0.03  train_win=False  holdout dLL -0.00000  periods 1/3  -> NULL

--- POWER CEILING — how big could each angle possibly read?
    Real panel, HR outcomes re-rolled so the angle IS true at a
    generous strength, then scored at the true value. No fitted model
    can beat this. A null far below its ceiling is uninformative.
    REST    +/-40%         ceiling +0.00033 LL/game
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
completely true — which is only possible if what was measured is noise. Cause:
just 4% of panel rows are a repeat meeting with the same starter, because the
panel is one 2024 burn-in plus April-June 2025. There is almost no signal
capacity in the column. It also dies outright on the regulars-only pass.
This is the second fake ROBUST WIN in two days (hard-hit 60d was the first).
The ship rule alone is not sufficient at these effect sizes; the ceiling and
the placebo are now both part of the gate.

REST AND LOAD ARE UNINFORMATIVE NULLS, NOT DEAD ANGLES. Both read about
-0.0002. Both ceilings are +0.00033 and +0.00046. Even a genuine +/-40% effect
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
