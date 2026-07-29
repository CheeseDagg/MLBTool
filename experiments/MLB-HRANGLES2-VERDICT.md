======================================================================
HR ANGLES 2 — travel, home/away, lineup-slot change, with ceilings
2025-04-01..2025-06-30 (train to 2025-05-31)   scorable rows: 74018
======================================================================
coverage: travel flag on 70582/74018 rows, slot-shift on 70582/74018

baseline (bat x park x flat platoon) tau_b=75 tau_park=800  TRAIN LL -0.30596
baseline HOLDOUT LL -0.32578 (n=8037)
G travel       tau_tr=8000   train_win=False holdout dLL +0.00001  periods 1/3  -> win, not robust
H home/away    tau_ha=20000  train_win=False holdout dLL -0.00003  periods 1/3  -> NULL
I slot shift   tau_sh=8000   train_win=False holdout dLL -0.00008  periods 0/3  -> NULL

--- POWER CEILINGS (panel re-rolled so the angle IS true; ORACLE is
    what a model that knew the answer would gain, FITTED is what this
    pipeline recovers from a panel where the effect is real)
G travel       plant [newx0.92, samex1.0]
               ORACLE +0.00018   FITTED +0.00017 (1/3 robust)   measured +0.00001   CANNOT BE SEEN AT THIS SAMPLE - do not bury
H home/away    plant [homex1.06, awayx0.94]
               ORACLE +0.00046   FITTED +0.00039 (3/3 robust)   measured -0.00003   dead: a real effect this size would have shown
I slot shift   plant [upx1.1, flatx1.0, downx0.9]
               ORACLE +0.00013   FITTED -0.00003 (0/3 robust)   measured -0.00008   CANNOT BE SEEN AT THIS SAMPLE - do not bury

Ship rule: ROBUST WIN, and the measured gain must clear the FITTED
detection threshold from a panel where the effect is known to be real.
