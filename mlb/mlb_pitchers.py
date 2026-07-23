"""
mlb_pitchers.py  —  pull each game's STARTING PITCHER + score, whole season
===========================================================================
Lean first step for the pitcher layer. Instead of scraping full pitcher stats,
this grabs one thing: every game's two starters and the final score, for the
entire season. From that I can compute each starter's rolling form (runs his
team allowed in his prior starts) LEAK-FREE, and test whether knowing the
starter adds any predictive signal before we invest in pulling ERA/FIP.

Source: MLB-StatsAPI (statsapi.mlb.com) — the schedule endpoint, which already
worked for you. This is basically that call over the full season date range.

RUN:  python mlb_pitchers.py
  -> ./data/game_starters.csv   (date, away, home, away_SP, home_SP, scores, status)

Written blind (I can't reach statsapi from my sandbox). If it errors, it's most
likely the date range being too big for one call, or a key name; send me the
error and it's a quick fix. Next: I fold this into mlb_model.py and re-backtest
on your real data to see if the starter moves the needle.
"""
import os, sys, datetime as dt
try:
    import pandas as pd
    import statsapi
except ImportError:
    sys.exit("Missing libs. Run:  pip install pybaseball MLB-StatsAPI pandas")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def pull_game_starters(start=None, end=None):
    # Default to March 1 of the CURRENT season (not a hardcoded year) so next season
    # doesn't silently span two years of games into one file.
    if start is None:
        start = f"03/01/{dt.date.today().year}"
    if end is None:
        end = dt.date.today().strftime("%m/%d/%Y")
    print(f"[game starters] {start} -> {end}")
    # pull month-by-month to keep each call small & so a mid-season failure
    # doesn't lose everything
    rows = []
    s = dt.datetime.strptime(start, "%m/%d/%Y").date()
    e = dt.datetime.strptime(end, "%m/%d/%Y").date()
    cur = s
    while cur <= e:
        nxt = min((cur.replace(day=28) + dt.timedelta(days=10)).replace(day=1), e + dt.timedelta(days=1))
        a, b = cur.strftime("%m/%d/%Y"), (nxt - dt.timedelta(days=1)).strftime("%m/%d/%Y")
        try:
            games = statsapi.schedule(start_date=a, end_date=b)
            for g in games:
                rows.append({
                    "date": g.get("game_date"),
                    "away": g.get("away_name"),
                    "home": g.get("home_name"),
                    "away_SP": g.get("away_probable_pitcher"),
                    "home_SP": g.get("home_probable_pitcher"),
                    "away_score": g.get("away_score"),
                    "home_score": g.get("home_score"),
                    "status": g.get("status"),
                })
            print(f"   {a}..{b}: {len(games)} games")
        except Exception as ex:
            print(f"   {a}..{b}: FAILED -> {type(ex).__name__}: {ex}")
        cur = nxt

    df = pd.DataFrame(rows)
    if df.empty:
        # every monthly pull failed (transient statsapi outage) — write an empty but
        # well-formed file instead of KeyError-crashing on df["home_score"].
        print("\n   total rows: 0 — all monthly pulls failed; writing empty game_starters.csv")
        path = os.path.join(DATA_DIR, "game_starters.csv")
        pd.DataFrame(columns=["date","away","home","away_SP","home_SP",
                              "away_score","home_score","status"]).to_csv(path, index=False)
        print(f"   -> saved game_starters.csv (empty)")
        return path
    # keep only completed games with a real score and both starters known
    have_score = df["home_score"].notna() & df["away_score"].notna()
    print(f"\n   total rows: {len(df)}  |  with final score: {int(have_score.sum())}")
    print(f"   games missing a starter name: "
          f"{int((df['home_SP'].isna() | df['away_SP'].isna()).sum())}")
    path = os.path.join(DATA_DIR, "game_starters.csv")
    df.to_csv(path, index=False)
    print(f"   -> saved game_starters.csv")
    return path


if __name__ == "__main__":
    pull_game_starters()
    print("\nDone. Open data/game_starters.csv, check it has real pitcher names and "
          "scores, and send me the header row + a couple of sample rows.")
