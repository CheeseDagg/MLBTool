==========================================================================
FATIGUE / FAMILIARITY — the four buried angles on the wide panel
rows 74018   holdout n=24332 (was 8037 when these were filed CANNOT BE SEEN)
==========================================================================
coverage on the holdout: rest 23959/24332 (98%)  load 20187/24332 (83%)  famil 6205/24332 (26%)  famcar 7914/24332 (33%)
  rest-day mix: 0d=233  1d=17018  2d=4655  3d=1189  4d=429  5d=193  6d=101  7d=71  8d=37  9d=18  10d=15  noned=373

--- SUBSET [all]  baseline tau_b=150 tau_park=3000 w_p=0.3  TRAIN -0.32114  HOLDOUT -0.32267 (n=24332)
angle                           w   holdout      2024      2025  slices  verdict
REST    days off            -0.12  -0.00003  -0.00003  -0.00003     1/4  no
LOAD    7d PA vs own norm   -0.03  -0.00001  -0.00001  -0.00001     1/4  no
FAMIL   Nth look, season    +0.40  -0.00040  -0.00061  +0.00004     2/4  no
FAMCAR  Nth look, career    +0.12  -0.00007  -0.00008  -0.00005     2/4  no

--- SUBSET [regulars]  baseline tau_b=150 tau_park=3000 w_p=0.3  TRAIN -0.35443  HOLDOUT -0.35399 (n=16298)
angle                           w   holdout      2024      2025  slices  verdict
REST    days off            +0.06  +0.00003  +0.00003  +0.00003     2/4  no
LOAD    7d PA vs own norm   -0.12  -0.00017  -0.00021  -0.00011     0/4  no
FAMIL   Nth look, season    +0.40  -0.00034  -0.00060  +0.00009     2/4  no
FAMCAR  Nth look, career    +0.12  -0.00005  -0.00003  -0.00008     2/4  no

--- POWER CEILINGS (plant w=+0.30, 3 seeds). ORACLE = a model
    handed the true multiplier. FITTED = what this pipeline recovers
    from a panel where the effect is real — the honest threshold.
REST    days off           ORACLE +0.00006 [+0.00002..+0.00011]  FITTED +0.00008 (0/3 robust)  measured -0.00003   STILL CANNOT BE SEEN - do not bury (a planted effect was itself only recovered 0/3)
LOAD    7d PA vs own norm  ORACLE +0.00038 [+0.00035..+0.00043]  FITTED +0.00037 (3/3 robust)  measured -0.00001   DEAD: a planted effect of this size was recovered 3/3, so a real one would have shown
FAMIL   Nth look, season   ORACLE +0.00045 [+0.00036..+0.00054]  FITTED +0.00050 (2/3 robust)  measured -0.00040   DEAD: a planted effect of this size was recovered 2/3, so a real one would have shown
FAMCAR  Nth look, career   ORACLE +0.00049 [+0.00042..+0.00058]  FITTED +0.00047 (2/3 robust)  measured -0.00007   DEAD: a planted effect of this size was recovered 2/3, so a real one would have shown
