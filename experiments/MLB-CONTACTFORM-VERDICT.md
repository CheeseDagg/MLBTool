======================================================================
CONTACT-FORM EXPERIMENT — rolling barrels/hard-hit vs own norm
baseline includes the shipped prior-seeded pitcher HR factor
======================================================================
baseline (incl. prior-seeded pitcher HR)  TRAIN -0.30569 (n=16419)  HOLDOUT -0.32504 (n=8037)
BARRELS  W=7d {'tau_c': 50, 'w_c': 0.3}  train_win=False  holdout dLL -0.00016  periods 0/3  coverage 8014/8037  -> NULL
HARD-HIT W=60d {'tau_c': 10, 'w_c': 0.3}  train_win=True  holdout dLL +0.00013  periods 3/3  coverage 8014/8037  -> ROBUST WIN
--- window sweep (train-tuned knobs per W, holdout dLL — DIAGNOSTIC)
  BARRELS  7d -0.00016  14d -0.00032  21d -0.00038  30d -0.00029  45d -0.00015  60d +0.00012
  HARD-HIT 7d +0.00012  14d +0.00003  21d -0.00000  30d -0.00003  45d -0.00004  60d +0.00013
Ship rule: ROBUST WIN only (train win + holdout win + 3/3 periods).

========================================================================
ROUND 2 — THE KEYHOLE HYPOTHESIS, TESTED AND CLOSED
========================================================================
Round 1 buried this angle with a caveat: a 14-day window is only ~40 batted
balls, so maybe the window was the bottleneck rather than the idea. That is a
testable claim, so it got tested.

TWO FIXES FIRST, both permanent:
  * The norm now EXCLUDES its own window. The window is a subset of the norm
    span, so at W=60 against a 200-day norm the window was a sixth of the
    baseline it was being compared to, which drags the ratio toward 1.0 for
    reasons that have nothing to do with the hitter. Norm widened to 365d.
  * The selftest panel gained a 180-day burn-in. Without it the "own norm" was
    the adjacent few weeks, which - hot runs lasting ~25 days - is usually in
    the SAME state as the window. The old panel could not have detected a real
    effect under the corrected norm, and indeed the planted effect vanished
    until the burn-in was added. Real data has 2024 behind it; the panel must.

RESULT at W in (7, 14, 21, 30, 45, 60):
  BARRELS  best W=7d   holdout -0.00016  0/3 periods  -> NULL
  HARD-HIT best W=60d  holdout +0.00013  3/3 periods  -> clears the ship rule
  window sweep, holdout dLL (diagnostic, train-tuned knobs per W):
    BARRELS  7d -0.00016  14d -0.00032  21d -0.00038  30d -0.00029  45d -0.00015  60d +0.00012
    HARD-HIT 7d +0.00012  14d +0.00003  21d -0.00000  30d -0.00003  45d -0.00004  60d +0.00013

DOES THE SHIP RULE DESERVE TO BE BELIEVED AT +0.00013? contactform_placebo.py
permutes the contact tables across batters and re-runs the IDENTICAL
tune-on-train / verdict-on-June pipeline, full 6x3x3 grid, 24 times:
  BARRELS  fired 0/24  shuffled holdout dLL max +0.00007
  HARD-HIT fired 0/24  shuffled holdout dLL max +0.00007
So the rule is not trigger-happy, and +0.00013 sits above all 24 noise draws.

AND YET IT DOES NOT SHIP. Two things outweigh a clean placebo:
  1. The 3/3 is an artifact of where the cut lands. Shift the three-way period
     boundary by four days and it is 2/3; at four, five and six equal
     sub-periods it is 3/4, 3/5, 4/6. It is positive on average and
     inconsistent everywhere.
  2. The window sweep has the wrong SHAPE. A real form signal thickens as the
     window thickens. This one spikes at 7d, dies through 14-45d, and spikes
     again at 60d - two lucky cells at the ends of a grid, with nothing in
     between to connect them.
Even taken at face value it is ~1/70th the size of the pitcher-HR factor that
did ship. It cannot move a board price.

VERDICT: the keyhole hypothesis is answered. The window was never the
limitation. Recent contact quality carries no HR signal beyond the season line
and the hot-hand flag, at any window length. Closed - do not re-open without a
genuinely new data source (bat speed, swing decisions), not a new window.
