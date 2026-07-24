==============================================================================
CATCHER-FRAMING EXPERIMENT — 2025-04-01 .. 2025-06-30
framing tables (prior seasons, leak rule): [2024]
==============================================================================
  savant 2024 header actually received: ['id', 'name', 'pitches', 'rv_tot', 'pct_tot', 'rv_11', 'pct_11', 'rv_12', 'pct_12', 'rv_13', 'pct_13', 'rv_14', 'pct_14', 'rv_16', 'pct_16', 'rv_17', 'pct_17', 'rv_18', 'pct_18', 'rv_19', 'pct_19']

baseballsavant UNREACHABLE — run on GitHub Actions.
  (probe error: ValueError: savant 2024: no strike_rate/runs column in ['id', 'name', 'pitches', 'rv_tot', 'pct_tot', 'rv_11', 'pct_11', 'rv_12', 'p)
  This is EXPECTED in the cloud sandbox (egress is blocked).
  The savant-probe workflow proves savant IS reachable on Actions.
  Action command:  python3 mlb_framing_experiment.py --start 2025-04-01 --end 2025-06-30
