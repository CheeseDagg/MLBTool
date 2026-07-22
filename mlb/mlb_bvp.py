#!/usr/bin/env python3
"""BvP leaderboard — the batters with the most CAREER home runs off the pitcher they
face today. Writes mlb/data/bvp_board.json for the dashboard's "Best HR history vs
today's starter" panel.

History / color ONLY. This does NOT feed the model's HR% — it sits next to it.

Data source: MLB StatsAPI (no key).
  1. schedule?date=...&hydrate=probablePitcher,lineups,team  -> today's matchups WITH player IDs
     (no fragile name matching; every batter/pitcher comes with its MLBAM id)
  2. people/{batterId}/stats?stats=vsPlayerTotal&opposingPlayerId={pitcherId}  -> career line vs that arm

Runs on a machine that can reach statsapi.mlb.com (the sandbox can't — same as every other live pull).
Fails soft: any error writes an empty board rather than crashing the daily run.

Usage:
  python3 mlb_bvp.py                 # today
  python3 mlb_bvp.py 2026-07-22      # a specific date
  python3 mlb_bvp.py --selftest      # offline logic + parser checks
  python3 mlb_bvp.py --probe         # today, prints the matchups it found then exits (debug)
"""
import json, os, sys, datetime, urllib.request

API      = "https://statsapi.mlb.com/api/v1"
MIN_PA   = 6      # career PA vs that pitcher to qualify (kills lone 1-for-1 flukes). tweak here.
TOP_N    = 15
TIMEOUT  = 20


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mlb-bvp/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


# ---- pure parsers (unit-tested in --selftest; no network) ------------------
def _parse_matchups(data):
    """schedule JSON -> [(batter_id, batter_name, team_abbr, pitcher_id, pitcher_name), ...]
    Home batters face the away probable starter and vice-versa."""
    out = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            teams = g.get("teams", {})
            home, away = teams.get("home", {}), teams.get("away", {})
            hp = home.get("probablePitcher") or {}
            ap = away.get("probablePitcher") or {}
            lu = g.get("lineups", {}) or {}
            # (which lineup, the pitcher they oppose, their own team block)
            for side, opp_pit, own in (("homePlayers", ap, home), ("awayPlayers", hp, away)):
                pid, pname = opp_pit.get("id"), opp_pit.get("fullName")
                if not pid:
                    continue
                t = own.get("team") or {}
                tabbr = t.get("abbreviation") or t.get("triCode") or ""
                for b in lu.get(side, []) or []:
                    if b.get("id") and b.get("fullName"):
                        out.append((b["id"], b["fullName"], tabbr, pid, pname))
    return out


def _parse_vs(data):
    """vsPlayerTotal JSON -> {pa,ab,h,hr,avg,slg} for the first split with PA>0, else None."""
    for st in data.get("stats", []):
        for sp in st.get("splits", []):
            s = sp.get("stat", {})
            pa = int(s.get("plateAppearances", 0) or 0)
            if pa <= 0:
                continue
            return {"pa": pa,
                    "ab": int(s.get("atBats", 0) or 0),
                    "h": int(s.get("hits", 0) or 0),
                    "hr": int(s.get("homeRuns", 0) or 0),
                    "avg": s.get("avg", "-"),
                    "slg": s.get("slg", "-")}
    return None


def _slgf(r):
    try:
        return float(r["slg"])
    except (TypeError, ValueError):
        return -1.0


def _rank(rows):
    """gate (HR>=1 and PA>=MIN_PA) then most HR -> higher SLG -> more PA. Returns top TOP_N."""
    keep = [r for r in rows if r.get("hr", 0) >= 1 and r.get("pa", 0) >= MIN_PA]
    keep.sort(key=lambda r: (-r["hr"], -_slgf(r), -r["pa"]))
    return keep[:TOP_N]


# ---- live pull -------------------------------------------------------------
def todays_matchups(date):
    url = f"{API}/schedule?sportId=1&date={date}&hydrate=probablePitcher,lineups,team"
    return _parse_matchups(_get(url))


def career_vs(batter_id, pitcher_id):
    url = (f"{API}/people/{batter_id}/stats?stats=vsPlayerTotal"
           f"&opposingPlayerId={pitcher_id}&group=hitting&sportId=1")
    try:
        return _parse_vs(_get(url))
    except Exception:
        return None


def build(date):
    rows, seen = [], set()
    for bid, bname, tabbr, pid, pname in todays_matchups(date):
        if (bid, pid) in seen:
            continue
        seen.add((bid, pid))
        line = career_vs(bid, pid)
        if not line:
            continue
        rows.append({"batter": bname, "team": tabbr, "opp_sp": pname, **line})
    return _rank(rows)


def _write(date, rows):
    out = {"generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
           "date": date, "min_pa": MIN_PA, "rows": rows}
    here = os.path.dirname(os.path.abspath(__file__))
    dpath = os.path.join(here, "data", "bvp_board.json")
    if not os.path.isdir(os.path.dirname(dpath)):
        dpath = os.path.join(here, "bvp_board.json")
    with open(dpath, "w") as f:
        json.dump(out, f, indent=1)
    return dpath


def main(date):
    try:
        rows = build(date)
    except Exception as e:                       # fail soft — never crash the daily run
        rows = []
        sys.stderr.write(f"[mlb_bvp] pull failed ({type(e).__name__}: {e}); wrote empty board\n")
    path = _write(date, rows)
    print(f"bvp_board.json: {len(rows)} bats (min {MIN_PA} PA, HR>=1) -> {path}")
    for r in rows[:10]:
        print(f"  {r['hr']} HR  {r['batter']:<20} vs {r['opp_sp']:<18} ({r['pa']} PA, {r['slg']} SLG)")


# ---- offline self-test -----------------------------------------------------
def selftest():
    # 1) ranking + gate
    fake = [
        {"batter": "A", "opp_sp": "P", "pa": 16, "hr": 3, "slg": ".800"},
        {"batter": "B", "opp_sp": "P", "pa": 40, "hr": 3, "slg": ".700"},   # same HR, lower SLG -> below A
        {"batter": "C", "opp_sp": "Q", "pa": 12, "hr": 2, "slg": "1.000"},
        {"batter": "D", "opp_sp": "Q", "pa": 3,  "hr": 2, "slg": "1.333"},  # PA<6 -> dropped
        {"batter": "E", "opp_sp": "R", "pa": 25, "hr": 0, "slg": ".500"},   # 0 HR -> dropped
    ]
    got = [r["batter"] for r in _rank(fake)]
    assert got == ["A", "B", "C"], f"rank/gate wrong: {got}"

    # 2) schedule parser: home bats vs away SP, away bats vs home SP, IDs carried
    sched = {"dates": [{"games": [{
        "teams": {
            "home": {"team": {"abbreviation": "PHI"}, "probablePitcher": {"id": 11, "fullName": "Ranger Suarez"}},
            "away": {"team": {"abbreviation": "LAD"}, "probablePitcher": {"id": 22, "fullName": "Eric Lauer"}}},
        "lineups": {"homePlayers": [{"id": 100, "fullName": "Kyle Schwarber"}],
                     "awayPlayers": [{"id": 200, "fullName": "Mookie Betts"}]}}]}]}
    mm = _parse_matchups(sched)
    assert (100, "Kyle Schwarber", "PHI", 22, "Eric Lauer") in mm, "home batter should face away SP"
    assert (200, "Mookie Betts", "LAD", 11, "Ranger Suarez") in mm, "away batter should face home SP"

    # 3) missing probable pitcher -> that side skipped, no crash
    sched2 = {"dates": [{"games": [{
        "teams": {"home": {"team": {"abbreviation": "NYY"}, "probablePitcher": {}},
                   "away": {"team": {"abbreviation": "BOS"}, "probablePitcher": {"id": 9, "fullName": "Arm"}}},
        "lineups": {"homePlayers": [{"id": 1, "fullName": "A B"}], "awayPlayers": [{"id": 2, "fullName": "C D"}]}}]}]}
    mm2 = _parse_matchups(sched2)
    assert all(p[3] != None for p in mm2) and len(mm2) == 1, f"TBD pitcher not skipped: {mm2}"

    # 4) vsPlayer parser
    vs = {"stats": [{"splits": [{"stat": {"plateAppearances": 16, "atBats": 15, "hits": 4,
                                          "homeRuns": 3, "avg": ".267", "slg": ".800"}}]}]}
    assert _parse_vs(vs) == {"pa": 16, "ab": 15, "h": 4, "hr": 3, "avg": ".267", "slg": ".800"}
    assert _parse_vs({"stats": [{"splits": [{"stat": {"plateAppearances": 0}}]}]}) is None, "0-PA split should be None"

    print("selftest OK:")
    print("  rank/gate  -> HR desc, SLG desc, PA desc; drops <6 PA and 0-HR ->", got)
    print("  matchups   -> home/away crossed correctly, IDs carried, TBD pitcher skipped")
    print("  vsPlayer   -> parsed 16 PA / 3 HR line; 0-PA -> None")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--probe" in sys.argv:
        dt = datetime.date.today().isoformat()
        for m in todays_matchups(dt):
            print(m)
    else:
        arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else datetime.date.today().isoformat()
        main(arg)
