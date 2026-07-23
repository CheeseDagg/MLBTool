#!/usr/bin/env python3
"""
mlb_marcel_run.py — builds the Marcel talent table ON GITHUB ACTIONS.

For every batter on the current board (and, cheaply, all rostered hitters), pull the
last 3+ seasons of HR/PA from statsapi's yearByYear endpoint — the same numbers
Baseball Reference shows, structured and free — compute the Marcel HR/PA talent
estimate, and write data/marcel_talent.json:

    { "generated": ..., "lg_hrpa": 0.0312, "season": 2026,
      "players": { "aaron judge": {"marcel": 0.0631, "age": 34, "seasons": 3}, ... } }

The live model (mlb_hr.build_board) loads this and uses marcel HR/PA as the batter
`base` instead of the single-season rate — with the current season blended in via
mlb_marcel.blend_with_current so it updates as the year progresses.

Player identity: statsapi player IDs are canonical, so the yearByYear pull is keyed by
ID (no name-merge fragility). The board maps name->ID via the same roster pull the
board already uses. Missing/insufficient history -> that player simply falls back to
the league-average base (Marcel's own rule), which the live model already handles.
"""
import os, json, time, urllib.request, datetime as dt
import mlb_marcel as MC

API = "https://statsapi.mlb.com/api/v1"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def _get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return json.loads(r.read())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.0 * (i + 1))

def norm(s):
    if not isinstance(s, str):
        return ""
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower().replace(".", "").replace("'", "").strip()

def active_hitters(season):
    """All hitters on 40-man rosters — the pool we build talent for."""
    out = {}
    teams = _get(f"{API}/teams?sportId=1&season={season}").get("teams", [])
    for t in teams:
        tid = t["id"]
        try:
            roster = _get(f"{API}/teams/{tid}/roster?rosterType=40Man&season={season}")
        except Exception:
            continue
        for r in roster.get("roster", []):
            pos = (r.get("position") or {}).get("abbreviation", "")
            if pos == "P":
                continue                     # hitters only
            pid = r.get("person", {}).get("id")
            nm = r.get("person", {}).get("fullName", "")
            if pid and nm:
                out[pid] = nm
        time.sleep(0.15)
    return out

def collapse_seasons(splits):
    """statsapi yearByYear returns ONE split PER TEAM per season, so a player traded
    mid-year appears as multiple same-year stints. Sum them into a single row per
    season — otherwise a deadline trade is counted as two separate seasons, which
    double-weights that year in Marcel's 5/4/3 window (and pushes a real prior season
    out), and the current-season capture below sees only one stint. Idempotent when
    statsapi already returns one row per season. Returns most-recent-first."""
    by_year = {}
    for s in splits:
        st = s.get("stat", {}) or {}
        yr = s.get("season")
        pa = st.get("plateAppearances")
        hr = st.get("homeRuns")
        if yr is None or pa is None or hr is None:
            continue
        y = int(yr)
        e = by_year.setdefault(y, {"year": y, "hr": 0, "pa": 0})
        e["hr"] += int(hr); e["pa"] += int(pa)
    return sorted(by_year.values(), key=lambda r: -r["year"])

def year_by_year(pid):
    """[{year, hr, pa}] most-recent-first from statsapi yearByYear hitting, with
    same-year (traded-player) stints summed per season."""
    try:
        d = _get(f"{API}/people/{pid}/stats?stats=yearByYear&group=hitting")
        splits = d.get("stats", [{}])[0].get("splits", [])
    except Exception:
        return []
    return collapse_seasons(splits)

def player_age(pid, season):
    try:
        d = _get(f"{API}/people/{pid}")
        bd = d.get("people", [{}])[0].get("birthDate")
        if bd:
            by = int(bd[:4])
            return season - by
    except Exception:
        pass
    return None

def league_hrpa(season):
    """League HR/PA for the season, to regress toward."""
    try:
        d = _get(f"{API}/stats?stats=season&group=hitting&season={season}&sportId=1")
        st = d.get("stats", [{}])[0].get("splits", [{}])[0].get("stat", {})
        hr = int(st.get("homeRuns", 0)); pa = int(st.get("plateAppearances", 0))
        if pa:
            return hr / pa
    except Exception:
        pass
    return MC.LG_HRPA_DEFAULT

def main():
    season = dt.date.today().year
    lg = league_hrpa(season)
    print(f"league HR/PA {season}: {lg:.4f}")
    hitters = active_hitters(season)
    print(f"building Marcel talent for {len(hitters)} rostered hitters...")
    players = {}
    done = 0
    for pid, nm in hitters.items():
        seasons = year_by_year(pid)
        # only prior seasons count as talent history (current season handled via blend live)
        prior = [s for s in seasons if s["year"] < season]
        age = player_age(pid, season)
        marcel = MC.marcel_hrpa(prior, age, lg_hrpa=lg)
        # also capture current-season line so the live blend can update it
        cur = next((s for s in seasons if s["year"] == season), {"hr": 0, "pa": 0})
        players[norm(nm)] = {"marcel": round(marcel, 5), "age": age,
                             "seasons": len([s for s in prior if s["pa"] >= MC.MIN_SEASON_PA]),
                             "cur_hr": cur["hr"], "cur_pa": cur["pa"]}
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(hitters)}")
        time.sleep(0.12)
    out = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
           "season": season, "lg_hrpa": round(lg, 5), "players": players}
    os.makedirs(DATA, exist_ok=True)
    json.dump(out, open(os.path.join(DATA, "marcel_talent.json"), "w"))
    # a quick sanity line: top-10 talent
    top = sorted(players.items(), key=lambda kv: -kv[1]["marcel"])[:10]
    print(f"\nMarcel talent written: {len(players)} hitters")
    print("top 10 HR/PA talent:")
    for nm, p in top:
        print(f"  {nm:<24} {p['marcel']:.4f}  (age {p['age']}, {p['seasons']} seasons)")

def selftest():
    # traded mid-2025 (two team stints) + full 2024 + traded 2023 (two stints).
    splits = [
        {"season": "2025", "stat": {"plateAppearances": 300, "homeRuns": 12}},  # team A
        {"season": "2025", "stat": {"plateAppearances": 320, "homeRuns": 16}},  # team B (traded)
        {"season": "2024", "stat": {"plateAppearances": 640, "homeRuns": 30}},
        {"season": "2023", "stat": {"plateAppearances": 250, "homeRuns": 9}},   # team A
        {"season": "2023", "stat": {"plateAppearances": 300, "homeRuns": 11}},  # team B (traded)
    ]
    rows = collapse_seasons(splits)
    assert [r["year"] for r in rows] == [2025, 2024, 2023], rows           # one row per season
    assert rows[0] == {"year": 2025, "hr": 28, "pa": 620}, rows[0]         # 2025 stints summed
    assert rows[2] == {"year": 2023, "hr": 20, "pa": 550}, rows[2]         # 2023 stints summed
    # malformed / null-stat splits are dropped, not crashed on
    assert collapse_seasons([{"season": None, "stat": {}}, {"season": "2022", "stat": {}}]) == []
    # idempotent when statsapi already returns one row per season
    one = [{"season": "2025", "stat": {"plateAppearances": 500, "homeRuns": 25}}]
    assert collapse_seasons(one) == [{"year": 2025, "hr": 25, "pa": 500}]
    print("MARCEL-RUN SELFTEST PASS — traded-player same-year stints summed; malformed dropped; idempotent")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    main()
