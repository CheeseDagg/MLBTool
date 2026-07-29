==========================================================================
HR WIDE PANEL — all seven angles re-read on a two-season holdout
rows 74018   warm 2024-03-20..2024-05-31   holdout n=24332 (was 8037 on June 2025 alone)
==========================================================================

  baseline HOLDOUT 2024: -0.32167 (n=16295)
  baseline HOLDOUT 2025: -0.32544 (n=8037)
  baseline tau_b=150 tau_park=3000  TRAIN -0.32124  HOLDOUT -0.32292 (n=24332)

angle               knob   holdout      2024      2025  slices  verdict
C indiv platoon      800  -0.00028  -0.00022  -0.00042     1/4  no
D handed park       2500  -0.00010  -0.00017  +0.00004     1/4  no
E day/night         1500  -0.00002  -0.00001  -0.00003     0/4  no
F pitcher HR         0.3  +0.00025  +0.00011  +0.00053     4/4  ROBUST WIN
G travel            8000  +0.00000  +0.00000  +0.00001     2/4  no
H home/away        20000  -0.00004  -0.00005  -0.00003     0/4  no
I slot shift        8000  -0.00007  -0.00007  -0.00007     1/4  no

--- POWER CEILINGS on the wide panel. ORACLE = what a model that knew
    the planted multiplier would gain (a hard bound). FITTED = what
    this pipeline actually recovers from a panel where the effect is
    real (the honest detection threshold). The gap is the tax.
C indiv platoon   ORACLE +0.00005 [-0.00004..+0.00013]  FITTED -0.00029 (0/3 robust)  measured -0.00028   STILL CANNOT BE SEEN - do not bury
D handed park     ORACLE +0.00023 [+0.00003..+0.00051]  FITTED +0.00014 (1/3 robust)  measured -0.00010   STILL CANNOT BE SEEN - do not bury
E day/night       ORACLE +0.00004 [-0.00007..+0.00015]  FITTED +0.00005 (1/3 robust)  measured -0.00002   inside the oracle's seed spread: unreadable
F pitcher HR      ORACLE +0.00037 [+0.00008..+0.00055]  FITTED +0.00004 (0/3 robust)  measured +0.00025   won but sits inside the oracle's own seed spread: UNPROVEN, needs a placebo
G travel          ORACLE +0.00017 [+0.00010..+0.00028]  FITTED +0.00021 (3/3 robust)  measured +0.00000   STILL CANNOT BE SEEN - do not bury
H home/away       ORACLE +0.00020 [+0.00011..+0.00026]  FITTED +0.00020 (2/3 robust)  measured -0.00004   STILL CANNOT BE SEEN - do not bury
I slot shift      ORACLE +0.00013 [+0.00003..+0.00019]  FITTED +0.00016 (2/3 robust)  measured -0.00007   STILL CANNOT BE SEEN - do not bury
