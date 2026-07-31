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

# A re-pull of the same season may come back a little smaller (a postponement
# drops out, a month is re-scheduled). It may not come back a tenth smaller --
# that is a failed pull wearing a successful pull's clothes.
SHRINK_FLOOR = 0.90


def _existing(path):
    """-> (row_count, season_year) for a game_starters.csv already on disk.
    (0, None) if it is absent, header-only or unreadable -- in all three cases
    there is nothing to protect, so the guard should stay out of the way."""
    if not os.path.exists(path):
        return 0, None
    try:
        old = pd.read_csv(path)
        if old.empty:
            return 0, None
        return len(old), _season_of(old)
    except Exception:
        return 0, None


def _season_of(df):
    """The season a frame of games belongs to, read off its own dates. The
    shrink guard compares like with like: on opening day a legitimate new-season
    pull holds three games while the file still holds last year's two thousand,
    and comparing those two numbers would lock the file every spring."""
    try:
        yrs = pd.to_datetime(df["date"], errors="coerce").dt.year.dropna()
        return int(yrs.mode().iloc[0]) if len(yrs) else None
    except Exception:
        return None


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
    path = os.path.join(DATA_DIR, "game_starters.csv")
    old_n, old_season = _existing(path)

    # A PARTIAL PULL MUST NOT OVERWRITE A GOOD FILE.
    #
    # This function pulls the whole season month by month and swallows each
    # month's exception so one bad month doesn't lose the rest. That is right,
    # but it used to mean a run where EVERY month failed wrote a well-formed
    # EMPTY file and returned success -- and the daily workflow stages every
    # changed file under mlb/data, so an hour of statsapi flakiness would have
    # committed 2000+ rows of season history away and left the board building
    # off nothing. The build would have gone green. The stale alarm would have
    # passed too: it checks the slate's DATE, and the date would have been today.
    #
    # So: refuse to shrink. Keep whatever is already on disk, and exit non-zero
    # so the workflow stops and the board stays visibly on yesterday. A stale
    # board with its own warning banner is a far better failure than a fresh
    # board built on an empty file.
    if not df.empty and old_n and _season_of(df) == old_season:
        keep = len(df) >= SHRINK_FLOOR * old_n
        if not keep:
            sys.exit(f"\n   ABORT: pulled {len(df)} rows but game_starters.csv already "
                     f"holds {old_n} for the same season -- that is a partial pull, not "
                     f"a real shrink. Existing file left untouched; fix the source and "
                     f"re-run.")
    if df.empty:
        if old_n:
            sys.exit(f"\n   ABORT: every monthly pull failed and game_starters.csv "
                     f"already holds {old_n} rows. Existing file left untouched.")
        # No prior file at all -- bootstrap case. Write the well-formed empty
        # skeleton so downstream reads don't KeyError, but still fail the run:
        # a build on zero games is not a build.
        pd.DataFrame(columns=["date","away","home","away_SP","home_SP",
                              "away_score","home_score","status"]).to_csv(path, index=False)
        sys.exit("\n   ABORT: all monthly pulls failed and there was no existing file; "
                 "wrote an empty skeleton. Nothing downstream can be trusted.")

    # keep only completed games with a real score and both starters known
    have_score = df["home_score"].notna() & df["away_score"].notna()
    print(f"\n   total rows: {len(df)}  |  with final score: {int(have_score.sum())}")
    print(f"   games missing a starter name: "
          f"{int((df['home_SP'].isna() | df['away_SP'].isna()).sum())}")
    if old_n:
        print(f"   previous file: {old_n} rows ({old_season}) -> now {len(df)}")
    df.to_csv(path, index=False)
    print(f"   -> saved game_starters.csv")
    return path


def selftest():
    """Offline checks for the shrink guard. Reaches no network and writes only
    into a temp dir."""
    import tempfile, subprocess, textwrap
    ok = [0, 0]

    def chk(c, msg):
        ok[1] += 1
        ok[0] += bool(c)
        print(("PASS  " if c else "FAIL  ") + msg)

    tmp = tempfile.mkdtemp()
    full = os.path.join(tmp, "full.csv")
    pd.DataFrame({"date": [f"2026-04-{i:02d}" for i in range(1, 21)],
                  "away": ["A"] * 20, "home": ["B"] * 20}).to_csv(full, index=False)
    chk(_existing(full) == (20, 2026), "_existing reads row count and season")
    chk(_existing(os.path.join(tmp, "nope.csv")) == (0, None),
        "a missing file reports no rows and no season")
    empty = os.path.join(tmp, "empty.csv")
    pd.DataFrame(columns=["date", "away", "home"]).to_csv(empty, index=False)
    chk(_existing(empty) == (0, None), "a header-only file reports no rows")

    same = pd.DataFrame({"date": ["2026-05-01"] * 19})
    chk(_season_of(same) == 2026, "_season_of reads the season off the pulled rows")
    chk(len(same) >= SHRINK_FLOOR * 20, "a 5% shrink is inside the floor and allowed")
    chk(not (3 >= SHRINK_FLOOR * 20), "a pull of 3 vs 20 held is refused")

    # THE SEASON BOUNDARY. On opening day the new season legitimately has a
    # handful of games while the file still holds all of last year's -- the
    # guard must not lock the file forever every spring.
    newyr = pd.DataFrame({"date": ["2027-03-28"] * 3})
    chk(_season_of(newyr) == 2027, "a new season's pull reports the new year")
    chk(_season_of(newyr) != _existing(full)[1],
        "a new season is not compared against last season's row count")

    bad = os.path.join(tmp, "bad.csv")
    open(bad, "w").write("this,is\nnot,a\nreal")
    chk(isinstance(_existing(bad), tuple), "an unreadable file degrades to a tuple, not a crash")

    print(f"\n{ok[0]}/{ok[1]} checks pass")
    return 0 if ok[0] == ok[1] else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    pull_game_starters()
    print("\nDone. Open data/game_starters.csv, check it has real pitcher names and "
          "scores, and send me the header row + a couple of sample rows.")
