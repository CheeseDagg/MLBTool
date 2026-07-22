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
def _parse_teams(data):
    """schedule JSON -> [(team_id, team_abbr, opp_pitcher_id, opp_pitcher_name, lineup_or_None), ...]
    Home batters face the away probable starter and vice-versa. lineup_or_None is the posted
    [(batter_id, name), ...] when the card is up, else None (caller falls back to the roster)."""
    out = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            teams = g.get("teams", {})
            home, away = teams.get("home", {}), teams.get("away", {})
            hp = home.get("probablePitcher") or {}
            ap = away.get("probablePitcher") or {}
            lu = g.get("lineups", {}) or {}
            for own, opp_pit, side_key in ((home, ap, "homePlayers"), (away, hp, "awayPlayers")):
                pid, pname = opp_pit.get("id"), opp_pit.get("fullName")
                if not pid:
                    continue
                t = own.get("team") or {}
                tid = t.get("id")
                if not tid:
                    continue
                tabbr = t.get("abbreviation") or t.get("triCode") or ""
                posted = [(b["id"], b["fullName"]) for b in (lu.get(side_key) or [])
                          if b.get("id") and b.get("fullName")]
                out.append((tid, tabbr, pid, pname, posted or None))
    return out


def _parse_roster(data):
    """active-roster JSON -> [(batter_id, name), ...] position players only (skip pitchers)."""
    out = []
    for r in data.get("roster", []):
        pos = (r.get("position") or {}).get("abbreviation", "")
        if pos == "P":
            continue
        p = r.get("person") or {}
        if p.get("id") and p.get("fullName"):
            out.append((p["id"], p["fullName"]))
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
def todays_teams(date):
    url = f"{API}/schedule?sportId=1&date={date}&hydrate=probablePitcher,lineups,team"
    return _parse_teams(_get(url))


def team_batters(team_id):
    """active-roster position players; [] on any error (fails soft per team)."""
    try:
        return _parse_roster(_get(f"{API}/teams/{team_id}/roster?rosterType=active"))
    except Exception:
        return []


def career_vs(batter_id, pitcher_id):
    url = (f"{API}/people/{batter_id}/stats?stats=vsPlayerTotal"
           f"&opposingPlayerId={pitcher_id}&group=hitting&sportId=1")
    try:
        return _parse_vs(_get(url))
    except Exception:
        return None


def build(date):
    rows, seen = [], set()
    for tid, tabbr, pid, pname, posted in todays_teams(date):
        # confirmed lineup when the card is up, otherwise the whole active roster
        batters = posted if posted is not None else team_batters(tid)
        for bid, bname in batters:
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

    # 2) schedule parser: team faces the OTHER team's SP; carries team id + posted lineup
    sched = {"dates": [{"games": [{
        "teams": {
            "home": {"team": {"id": 143, "abbreviation": "PHI"}, "probablePitcher": {"id": 11, "fullName": "Ranger Suarez"}},
            "away": {"team": {"id": 119, "abbreviation": "LAD"}, "probablePitcher": {"id": 22, "fullName": "Eric Lauer"}}},
        "lineups": {"homePlayers": [{"id": 100, "fullName": "Kyle Schwarber"}], "awayPlayers": []}}]}]}
    tm = _parse_teams(sched)
    phi = [t for t in tm if t[1] == "PHI"][0]
    lad = [t for t in tm if t[1] == "LAD"][0]
    assert phi[:4] == (143, "PHI", 22, "Eric Lauer"), f"PHI should face Lauer: {phi}"          # home bats vs away SP
    assert phi[4] == [(100, "Kyle Schwarber")], "posted PHI lineup should carry through"
    assert lad[:4] == (119, "LAD", 11, "Ranger Suarez"), f"LAD should face Suarez: {lad}"       # away bats vs home SP
    assert lad[4] is None, "empty LAD lineup -> None (triggers roster fallback)"

    # 3) missing probable pitcher -> that side skipped, no crash
    sched2 = {"dates": [{"games": [{
        "teams": {"home": {"team": {"id": 147, "abbreviation": "NYY"}, "probablePitcher": {}},
                   "away": {"team": {"id": 111, "abbreviation": "BOS"}, "probablePitcher": {"id": 9, "fullName": "Arm"}}},
        "lineups": {}}]}]}
    tm2 = _parse_teams(sched2)
    # NYY's opponent (BOS) has a known starter -> NYY kept; BOS's opponent (NYY) is TBD -> BOS dropped
    assert len(tm2) == 1 and tm2[0][1] == "NYY" and tm2[0][3] == "Arm", f"TBD handling wrong: {tm2}"

    # 4) roster parser: position players only (pitchers dropped)
    roster = {"roster": [
        {"person": {"id": 1, "fullName": "Trea Turner"}, "position": {"abbreviation": "SS"}},
        {"person": {"id": 2, "fullName": "Some Reliever"}, "position": {"abbreviation": "P"}},
        {"person": {"id": 3, "fullName": "Shohei Ohtani"}, "position": {"abbreviation": "DH"}}]}
    rb = _parse_roster(roster)
    assert (1, "Trea Turner") in rb and (3, "Shohei Ohtani") in rb, "position players kept"
    assert all(n != "Some Reliever" for _, n in rb), "pitchers dropped from roster"

    # 5) vsPlayer parser
    vs = {"stats": [{"splits": [{"stat": {"plateAppearances": 16, "atBats": 15, "hits": 4,
                                          "homeRuns": 3, "avg": ".267", "slg": ".800"}}]}]}
    assert _parse_vs(vs) == {"pa": 16, "ab": 15, "h": 4, "hr": 3, "avg": ".267", "slg": ".800"}
    assert _parse_vs({"stats": [{"splits": [{"stat": {"plateAppearances": 0}}]}]}) is None, "0-PA split should be None"

    print("selftest OK:")
    print("  rank/gate -> HR desc, SLG desc, PA desc; drops <6 PA and 0-HR ->", got)
    print("  schedule  -> team faces opp SP; posted lineup carried, empty lineup -> None (roster fallback)")
    print("  roster    -> position players kept, pitchers dropped")
    print("  vsPlayer  -> parsed 16 PA / 3 HR line; 0-PA -> None")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--probe" in sys.argv:
        dt = datetime.date.today().isoformat()
        teams = todays_teams(dt)
        print(f"{dt}: {len(teams)} team-sides")
        for tid, tabbr, pid, pname, posted in teams:
            src = f"lineup ({len(posted)})" if posted is not None else "roster fallback (no card yet)"
            print(f"  {tabbr:4} vs {pname:<20} [{src}]")
    else:
        arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else datetime.date.today().isoformat()
        main(arg)
