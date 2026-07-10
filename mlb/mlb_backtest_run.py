#!/usr/bin/env python3
"""
mlb_backtest_run.py — executes the season replay ON GITHUB ACTIONS (statsapi reachable).

Flow, strictly walk-forward:
  for each date D from season start to yesterday:
      1. PREDICT D using AsOfState (only games < D). One row per projected lineup batter.
      2. Reveal D: pull D's boxscores, GRADE the predictions (HR count per batter).
      3. Fold D's results INTO the state (now visible to D+1, never to D).

Writes data/hr_backtest.csv (graded rows) and data/hr_backtest_panel.json (verdicts).
The predict-before-fold ordering is what guarantees no leakage in production, mirroring
the AsOfState.test_no_leakage invariant.

Network is confined to this file; mlb_backtest.py stays pure/testable.
"""
import os, csv, json, time, urllib.request, datetime as dt
import mlb_backtest as BT
import mlb_backtest_weather as WX
import mlb_hr as H

API = "https://statsapi.mlb.com/api/v1"
DATA = BT.DATA
SEASON_START = dt.date(2026, 3, 27)     # opening day; adjust if needed

def _get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(1.5 * (i + 1))

def schedule(date):
    d = date.isoformat()
    j = _get(f"{API}/schedule?sportId=1&date={d}&hydrate=probablePitcher")
    games = []
    for dd in j.get("dates", []):
        for g in dd.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                games.append(g)
    return games

def boxscore(game_pk):
    return _get(f"{API}/game/{game_pk}/boxscore")

def _bat_pit_lines(box):
    """Extract per-player HR/PA (batters) and HR/BF+components (pitchers) from a boxscore."""
    bats, pits = [], []
    for side in ("home", "away"):
        team = box.get("teams", {}).get(side, {})
        players = team.get("players", {})
        for pid, pdata in players.items():
            nm = (pdata.get("person") or {}).get("fullName", "")
            st = (pdata.get("stats") or {})
            bs = st.get("batting") or {}
            if bs:
                pa = int(bs.get("plateAppearances", 0) or 0)
                hr = int(bs.get("homeRuns", 0) or 0)
                if pa: bats.append({"name": nm, "pa": pa, "hr": hr})
            ps = st.get("pitching") or {}
            if ps:
                bf = int(ps.get("battersFaced", 0) or 0)
                hr = int(ps.get("homeRuns", 0) or 0)
                so = int(ps.get("strikeOuts", 0) or 0)
                bb = int(ps.get("baseOnBalls", 0) or 0)
                ao = int(ps.get("airOuts", 0) or 0)
                if bf: pits.append({"name": nm, "bf": bf, "hr": hr, "so": so, "bb": bb, "ao": ao})
    return bats, pits

def _hand(pdata):
    p = pdata.get("person") or {}
    bs = (p.get("batSide") or {}).get("code")
    ph = (p.get("pitchHand") or {}).get("code")
    return bs, ph

def predict_day(state, date, games):
    """One projected row per batter in each final game's lineup, priced as-of `date`."""
    rows = []
    for g in games:
        pk = g["gamePk"]
        try:
            box = boxscore(pk)
        except Exception:
            continue
        venue = (g.get("venue") or {}).get("name", "")
        for side in ("home", "away"):
            opp = "away" if side == "home" else "home"
            pp = (g.get("teams", {}).get(opp, {}).get("probablePitcher") or {})
            opp_sp = pp.get("fullName", "")
            sp_hand = (pp.get("pitchHand") or {}).get("code")
            team = box.get("teams", {}).get(side, {})
            order = team.get("battingOrder", []) or []
            players = team.get("players", {})
            slot = 0
            # first-pitch hour (local) for weather matching
            fp_hour = 19
            gd = g.get("gameDate")
            if gd:
                try:
                    fp_hour = dt.datetime.fromisoformat(gd.replace("Z","+00:00")).astimezone().hour
                except Exception:
                    pass
            wmult, wtag = WX.weather_for(venue, date, fp_hour)
            for pid in order[:9]:
                slot += 1
                pdata = players.get(f"ID{pid}") or players.get(str(pid)) or {}
                nm = (pdata.get("person") or {}).get("fullName", "")
                bat_side, _ = _hand(pdata)
                if not nm: continue
                r = BT.price_row(state, date, nm, bat_side, opp_sp, sp_hand, venue, slot)
                r = WX.reprice_row(r, wmult, wtag)      # fold in historical weather
                rows.append(r)
    return rows

def grade_day(pred_rows, box_by_pk, games):
    """Attach outcome + hr_n to each predicted row from that day's actual boxscores."""
    # build name -> hr count for the day
    hr_by_name = {}
    for g in games:
        pk = g["gamePk"]
        box = box_by_pk.get(pk)
        if not box: continue
        for side in ("home", "away"):
            players = box.get("teams", {}).get(side, {}).get("players", {})
            for pid, pdata in players.items():
                nm = BT.norm((pdata.get("person") or {}).get("fullName", ""))
                bs = (pdata.get("stats") or {}).get("batting") or {}
                if bs:
                    hr_by_name[nm] = hr_by_name.get(nm, 0) + int(bs.get("homeRuns", 0) or 0)
    graded = []
    for r in pred_rows:
        nm = BT.norm(r["player"])
        if nm not in hr_by_name:
            continue                     # didn't actually appear -> skip (no phantom grades)
        n = hr_by_name[nm]
        r = dict(r); r["outcome"] = "hr" if n >= 1 else "no"; r["hr_n"] = n
        graded.append(r)
    return graded

def main():
    yesterday = dt.date.today() - dt.timedelta(days=1)
    state = BT.AsOfState(heat_window=15)
    all_graded = []
    day = SEASON_START
    days_done = 0
    while day <= yesterday:
        try:
            games = schedule(day)
        except Exception as e:
            print(f"  {day}: schedule fetch failed ({type(e).__name__}); stopping early")
            break
        if games:
            # 1) predict using PAST-ONLY state
            preds = predict_day(state, day, games)
            # 2) reveal + grade
            box_by_pk = {}
            for g in games:
                try: box_by_pk[g["gamePk"]] = boxscore(g["gamePk"])
                except Exception: pass
            graded = grade_day(preds, box_by_pk, games)
            all_graded.extend(graded)
            # 3) fold the day's lines into state (now visible to tomorrow)
            for pk, box in box_by_pk.items():
                bats, pits = _bat_pit_lines(box)
                state.record_day(day, bats, pits)
            print(f"  {day}: {len(games)} games, {len(graded)} graded (cum {len(all_graded)})")
        day += dt.timedelta(days=1)
        days_done += 1
        time.sleep(0.3)   # be polite to statsapi

    os.makedirs(DATA, exist_ok=True)
    if all_graded:
        cols = ["date","player","opp_sp","slot","hr_pct","plat","heat","park","outcome","hr_n"]
        with open(BT.OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
            for r in all_graded: w.writerow(r)
    panel = BT.summarize(all_graded)
    panel["generated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")
    panel["season_start"] = SEASON_START.isoformat()
    json.dump(panel, open(os.path.join(DATA, "hr_backtest_panel.json"), "w"), indent=1)
    print(f"\nBACKTEST DONE: {panel.get('n',0)} graded predictions across "
          f"{panel.get('dates',0)} days | pred {panel.get('pred_mean')}% -> "
          f"actual {panel.get('actual')}% | Brier {panel.get('brier')}")
    a = panel.get("a_tier")
    if a: print(f"A-tier (25%+ & heat>=+10): {a['hits']}/{a['n']} = {a['actual']}% "
                f"(said {a['pred']}%)  <-- the verdict on n={a['n']}")

if __name__ == "__main__":
    main()
