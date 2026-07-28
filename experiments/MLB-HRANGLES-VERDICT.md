======================================================================
HR ANGLES EXPERIMENT — 2025-04-01..2025-06-30 (train to 2025-05-31)
======================================================================
baseline (bat x park x flat platoon) tuned: tau_b=75 tau_park=800  TRAIN LL -0.30596
baseline HOLDOUT LL -0.32578 (n=8037)
C indiv platoon  train_win=False  holdout dLL -0.00051  periods 0/3  -> NULL
D handed park    train_win=True  holdout dLL +0.00002  periods 1/3  -> win, not robust
E day/night      train_win=True  holdout dLL -0.00005  periods 1/3  -> NULL
F pitcher HR     train_win=True  holdout dLL +0.00074  periods 3/3  -> ROBUST WIN
======================================================================
VERDICT
  F pitcher HR     ROBUST WIN       holdout dLL +0.00074 (3/3 periods) params {'tau_b': 75, 'tau_park': 800, 'tau_s': 200, 'tau_ph': 1500, 'tau_dn': 3000, 'w_p': 0.6}
  D handed park    win, not robust  holdout dLL +0.00002 (1/3 periods) params {'tau_b': 75, 'tau_park': 800, 'tau_s': 200, 'tau_ph': 2500, 'tau_dn': 3000, 'w_p': 0.6}
  E day/night      NULL             holdout dLL -0.00005 (1/3 periods) params {'tau_b': 75, 'tau_park': 800, 'tau_s': 200, 'tau_ph': 1500, 'tau_dn': 1500, 'w_p': 0.6}
  C indiv platoon  NULL             holdout dLL -0.00051 (0/3 periods) params {'tau_b': 75, 'tau_park': 800, 'tau_s': 800, 'tau_ph': 1500, 'tau_dn': 3000, 'w_p': 0.6}
Ship rule: ROBUST WIN only (train win + holdout win + 3/3 periods).
