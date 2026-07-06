"""
mlb_statcast.py  —  sharpen the pitcher input with Statcast expected stats
==========================================================================
The crude "runs his team allowed" proxy already moved the model above baseline.
This upgrades it to real EXPECTED stats -- xwOBA / xERA / xBA -- which strip out
defense and luck and predict a pitcher's future better than raw ERA.

The smart part: this is ONE call to Baseball Savant's expected-stats leaderboard,
NOT a 700k-pitch download. Fast and light.

Source: pybaseball.statcast_pitcher_expected_stats -> baseballsavant.com
(reachable from your machine, blocked from mine).

RUN:  python mlb_statcast.py
  -> ./data/pitcher_xstats.csv   (one row per pitcher: xwOBA, xERA, xBA, ...)

minPA is set low so mid-season starters who haven't "qualified" are still included.
If it returns very few pitchers, bump MIN_PA up a bit and re-run.
"""
import os, sys, datetime as dt
try:
    import pandas as pd
    from pybaseball import statcast_pitcher_expected_stats
except ImportError:
    sys.exit("Missing libs. Run:  pip install pybaseball pandas")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

YEAR = dt.date.today().year
MIN_PA = 1          # include everyone who's faced at least 1 batter (bump if too few)


def pull():
    print(f"[statcast expected stats] {YEAR}  (single leaderboard call) ...")
    df = statcast_pitcher_expected_stats(YEAR, MIN_PA)
    print(f"   pulled {len(df)} pitchers, {len(df.columns)} columns")
    print(f"   columns: {list(df.columns)}")
    path = os.path.join(DATA_DIR, "pitcher_xstats.csv")
    df.to_csv(path, index=False)
    print(f"   -> saved pitcher_xstats.csv")
    return path


if __name__ == "__main__":
    try:
        pull()
        print("\nDone. Open data/pitcher_xstats.csv and send me the header row + a couple "
              "rows so I can wire the right column (xwOBA / est_woba) into the model.")
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        print("If it's a 'too few pitchers' or empty result, raise MIN_PA in this file "
              "(e.g. 25) and re-run. Otherwise send me this error.")
