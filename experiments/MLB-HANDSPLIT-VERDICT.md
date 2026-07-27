==============================================================================
HAND-SPLIT WHIFF EXPERIMENT — 2025-04-01 .. 2025-06-30
==============================================================================
using committed kfactors dataset (reused) (2448 starts, 2448 team-game rows)
pitchers: 279 unique, hands cached for 0, missing 279
fetching throwing hand for 279 pitchers (bulk season list, then per-name fallback)
  bulk list resolved all but 0
dataset cached -> /home/runner/work/MLBTool/MLBTool/mlb/data/handsplit_dataset.json
dataset cached -> /home/runner/work/MLBTool/MLBTool/mlb/data/handsplit_dataset.json
dataset cached -> /home/runner/work/MLBTool/MLBTool/mlb/data/handsplit_dataset.json
hand coverage: 279/279 starters

==============================================================================
HAND-SPLIT WHIFF EXPERIMENT — HOLDOUT RESULT
(baseline = prior-rate * BF * overall-whiff factor, FIXED w=0.6)
==============================================================================
  train n=799  holdout n=923  cutoff=2025-05-23
  starter hand known for 100.0% of scorable starts
  tuned on TRAIN: tau=None min_split_pa=None  train dLL/start=+0.00000  train_win=False
  holdout starts actually using a split estimate: 0/923
  holdout LL delta +0.00 total (+0.00000/start, n=923)  MAE +0.0000  robust 0/3 periods
    period 2025-05-23..2025-06-05     n=304  LL delta +0.00 -> baseline  better
    period 2025-06-05..2025-06-18     n=308  LL delta +0.00 -> baseline  better
    period 2025-06-18..end            n=311  LL delta +0.00 -> baseline  better

==============================================================================
VERDICT
==============================================================================
  HAND-SPLIT OPP WHIFF: NO EDGE ON TRAIN (blend tuned to ~pure overall) — keep overall whiff
  => Keep the overall-whiff factor (OPP_W=0.6) unchanged in production.
==============================================================================
