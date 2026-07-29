======================================================================
CONTACT-FORM EXPERIMENT — rolling barrels/hard-hit vs own norm
baseline includes the shipped prior-seeded pitcher HR factor
======================================================================
baseline (incl. prior-seeded pitcher HR)  TRAIN -0.30569 (n=16419)  HOLDOUT -0.32504 (n=8037)
BARRELS  W=7d {'tau_c': 50, 'w_c': 0.3}  train_win=False  holdout dLL -0.00016  periods 0/3  coverage 7978/8037  -> NULL
HARD-HIT W=7d {'tau_c': 50, 'w_c': 0.3}  train_win=True  holdout dLL +0.00007  periods 2/3  coverage 7978/8037  -> win, not robust
Ship rule: ROBUST WIN only (train win + holdout win + 3/3 periods).
