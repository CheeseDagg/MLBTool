"""
mlb_statcast_pitches.py  —  the heavy pull: per-pitcher-per-game xwOBA (LEAK-FREE input)
=======================================================================================
This is the 700k-pitch download that lets us backtest xwOBA properly (each pitcher's
form from his PRIOR starts, no leakage). It does the download AND the aggregation on
YOUR machine, so you upload a SMALL file (~a few thousand rows), not the raw pitches.

Honest heads-up: the download is slow (pybaseball chunks the season into ~5-day
requests) -- expect 10-30 min and some memory use. Let it run. Kick it off and go do
something else. It caches, so a re-run is fast.

Source: pybaseball.statcast -> baseballsavant.com (your machine can reach it, mine can't).

RUN:  python mlb_statcast_pitches.py
  -> ./data/pitcher_game_xwoba.csv   (pitcher, name, game_date, xwoba, batters_faced)

Expected payoff (I told you up front): confirms a ~1-point gain over the crude proxy,
turning "somewhere in 55-57%" into a proven number. No surprise above ~57%.
"""
import os, sys, datetime as dt
try:
    import pandas as pd, numpy as np
    from pybaseball import statcast
    try:
        from pybaseball import cache; cache.enable()
    except Exception:
        pass
except ImportError:
    sys.exit("Missing libs. Run:  pip install pybaseball pandas numpy")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# wOBA weights for the non-batted-ball outcomes (approx; used only where Statcast has
# no estimated_woba, i.e. K / BB / HBP). Batted balls use estimated_woba_using_speedangle.
WALK_WOBA, HBP_WOBA = 0.69, 0.72


def main(start="2026-03-25", end=None):
    if end is None:
        end = dt.date.today().strftime("%Y-%m-%d")
    print(f"[statcast pitches] {start} -> {end}   (this is the slow one; let it run)")
    df = statcast(start_dt=start, end_dt=end)
    if df is None or len(df) == 0:
        sys.exit("no statcast data returned — check the dates / your connection")
    print(f"   pulled {len(df):,} pitches, {len(df.columns)} columns")

    need = ["pitcher", "game_date", "woba_denom", "woba_value",
            "estimated_woba_using_speedangle"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"   NOTE: missing expected columns {miss} — send me df.columns and I'll adapt")

    # one row per plate appearance = the pitch where woba_denom is recorded
    pa = df[df["woba_denom"].notna()].copy()
    # xwOBA numerator: expected wOBA on batted balls; actual wOBA value otherwise (K=0, BB≈.69)
    pa["xnum"] = pa["estimated_woba_using_speedangle"]
    pa["xnum"] = pa["xnum"].fillna(pd.to_numeric(pa.get("woba_value"), errors="coerce"))
    pa["xnum"] = pd.to_numeric(pa["xnum"], errors="coerce").fillna(0.0)
    pa["woba_denom"] = pd.to_numeric(pa["woba_denom"], errors="coerce").fillna(0.0)

    name_col = "player_name" if "player_name" in pa.columns else None
    grp = ["pitcher"] + ([name_col] if name_col else []) + ["game_date"]
    agg = pa.groupby(grp, dropna=False).agg(
        xnum=("xnum", "sum"), denom=("woba_denom", "sum"), bf=("woba_denom", "size")
    ).reset_index()
    agg = agg[agg["denom"] > 0].copy()
    agg["xwoba"] = agg["xnum"] / agg["denom"]
    if name_col:
        agg = agg.rename(columns={name_col: "name"})
    cols = ["pitcher"] + (["name"] if name_col else []) + ["game_date", "xwoba", "bf"]
    out = agg[cols]

    path = os.path.join(DATA_DIR, "pitcher_game_xwoba.csv")
    out.to_csv(path, index=False)
    print(f"   aggregated to {len(out):,} pitcher-game rows")
    print(f"   league xwOBA (BF-weighted): {(out['xwoba']*out['bf']).sum()/out['bf'].sum():.4f}")
    print(f"   -> saved pitcher_game_xwoba.csv  (small — upload THIS one)")


if __name__ == "__main__":
    main()
