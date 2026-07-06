"""
mlb_data.py  —  Data layer for the MLB betting model  (step 1 of the pipeline)
==============================================================================
Pulls everything the model will need, mirroring the UFC pipeline's fetch stage:

  1. Team ratings     season batting + pitching  -> team offense / defense strength
  2. Game logs        every team's game-by-game results  -> the BACKTEST set
  3. Pitcher ratings  starter season stats  -> the single biggest per-game input
  4. Today's slate    scheduled games + probable starters  -> what we predict & price

WHY YOU RUN THIS (not me): the sandbox this was written in is firewalled to
GitHub/PyPI only and cannot reach baseball-reference / FanGraphs / statsapi.
Your machine can. So the data step lives with you; I write the code.

INSTALL (once):
    pip install pybaseball MLB-StatsAPI pandas

RUN:
    python mlb_data.py                 # current season + today's slate -> ./data/*.csv
    python mlb_data.py 2023            # a specific season's team/pitcher/game data

SOURCES:
    pybaseball   -> baseball-reference & FanGraphs   (team stats, pitcher stats, game logs)
    MLB-StatsAPI -> statsapi.mlb.com                 (schedule + probable pitchers)

HONEST NOTE (Claude, writing blind): I could not execute this against live data,
so the pybaseball / statsapi call signatures and column names are my best read of
those libraries, not something I verified by running. The code is defensive on
purpose (missing columns are skipped, each team pull is wrapped, quick steps run
first so it fails fast). If something breaks it's most likely a function name or a
column-name drift in those libs — send me the traceback and it's a quick fix.

NEXT STEP after this runs clean: mlb_model.py — team run-scoring ratings adjusted
by the game's starting pitcher -> win probability + run distribution, backtested
walk-forward on game_logs the same way the UFC model was.
"""

import os
import sys
import datetime as dt

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing pandas. Run:  pip install pybaseball MLB-StatsAPI pandas")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 30 MLB clubs, Baseball-Reference abbreviations (what schedule_and_record expects)
TEAMS = ["ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
         "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
         "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN"]

CURRENT_SEASON = dt.date.today().year


def _save(df, name):
    path = os.path.join(DATA_DIR, name)
    df.to_csv(path, index=False)
    print(f"   -> saved {name}  ({len(df)} rows, {len(df.columns)} cols)")
    return path


def pull_team_ratings(season=CURRENT_SEASON):
    """Season team batting + pitching -> one row/team with the rate stats that
    drive run scoring (offense) and run prevention (defense)."""
    from pybaseball import team_batting, team_pitching
    print(f"[team ratings] {season} ...")
    bat = team_batting(season)
    pit = team_pitching(season)
    # keep only the most predictive columns that are actually present
    bat_keep = [c for c in ["Team", "G", "PA", "R", "HR", "BB", "SO",
                            "AVG", "OBP", "SLG", "OPS", "wOBA", "wRC+"] if c in bat.columns]
    pit_keep = [c for c in ["Team", "G", "IP", "ERA", "FIP", "xFIP",
                            "WHIP", "K/9", "BB/9", "HR/9", "SIERA"] if c in pit.columns]
    m = bat[bat_keep].merge(pit[pit_keep], on="Team", suffixes=("_bat", "_pit"))
    if "R" in m.columns and "G_bat" in m.columns:
        m["R_per_G"] = m["R"] / m["G_bat"]
    return _save(m, f"team_ratings_{season}.csv")


def pull_game_logs(season=CURRENT_SEASON):
    """Every team's game-by-game results -> the backtest set. One pull per club.
    We don't assume specific columns here; schedule_and_record's raw output is
    saved as-is plus a `team` tag, and the model parses it."""
    from pybaseball import schedule_and_record
    print(f"[game logs] {season} ...  (30 pulls, one per club)")
    frames = []
    for i, tm in enumerate(TEAMS, 1):
        try:
            g = schedule_and_record(season, tm).copy()
            g["team"] = tm
            frames.append(g)
            print(f"   {i:>2}/30  {tm}: {len(g)} games")
        except Exception as e:
            print(f"   {i:>2}/30  {tm}: FAILED -> {type(e).__name__}: {e}")
    if not frames:
        raise RuntimeError("schedule_and_record failed for all 30 teams")
    return _save(pd.concat(frames, ignore_index=True), f"game_logs_{season}.csv")


def pull_pitcher_ratings(season=CURRENT_SEASON):
    """Every pitcher's season stats (qual=0 = include all) -> the per-game
    starter adjustment. We filter to starters (GS) in the model step."""
    from pybaseball import pitching_stats
    print(f"[pitcher ratings] {season} ...")
    p = pitching_stats(season, season, qual=0)
    keep = [c for c in ["Name", "Team", "IP", "GS", "ERA", "FIP", "xFIP",
                        "SIERA", "K/9", "BB/9", "HR/9", "WHIP", "WAR"] if c in p.columns]
    return _save(p[keep].copy(), f"pitcher_ratings_{season}.csv")


def pull_schedule(date=None):
    """Games on `date` (default today) with probable starters -> the slate we
    predict and price against the market."""
    import statsapi
    if date is None:
        date = dt.date.today().strftime("%m/%d/%Y")
    print(f"[schedule] {date} ...")
    games = statsapi.schedule(date=date)
    rows = [{
        "game_id": g.get("game_id"),
        "date": g.get("game_date"),
        "away": g.get("away_name"),
        "home": g.get("home_name"),
        "away_prob_pitcher": g.get("away_probable_pitcher"),
        "home_prob_pitcher": g.get("home_probable_pitcher"),
        "status": g.get("status"),
        "venue": g.get("venue_name"),
    } for g in games]
    return _save(pd.DataFrame(rows), f"schedule_{dt.date.today().isoformat()}.csv")


def main(season=CURRENT_SEASON):
    print("=" * 62)
    print(f"MLB DATA PULL  (season {season})  ->  ./data/")
    print("=" * 62)
    try:
        from pybaseball import cache
        cache.enable()   # speeds up re-runs a lot
    except Exception:
        pass

    # quick steps first so it fails fast if a library / network is off
    steps = [
        ("schedule + probables (today)", lambda: pull_schedule()),
        ("team ratings", lambda: pull_team_ratings(season)),
        ("pitcher ratings", lambda: pull_pitcher_ratings(season)),
        ("game logs  [slowest: 30 pulls]", lambda: pull_game_logs(season)),
    ]
    ok, fail = [], []
    for label, fn in steps:
        try:
            fn()
            ok.append(label)
        except Exception as e:
            print(f"   !! {label} FAILED: {type(e).__name__}: {e}")
            fail.append((label, f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 62)
    print(f"DONE  —  {len(ok)} ok, {len(fail)} failed")
    for label, e in fail:
        print(f"   FAILED  {label}\n           {e}")
    print(f"\noutput dir: {DATA_DIR}")
    if fail:
        print("\nSend me the FAILED lines above and I'll fix the pulls.")


if __name__ == "__main__":
    yr = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else CURRENT_SEASON
    main(yr)
