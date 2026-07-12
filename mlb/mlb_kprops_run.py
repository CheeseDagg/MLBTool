#!/usr/bin/env python3
"""
mlb_kprops_run.py — pitcher strikeout props: model board + line shop, ON ACTIONS.

Flow:
  1. Read today's slate.json -> tonight's starters and matchups.
  2. Pull each starter's season SO/IP/GS and each opponent's team SO/PA
     (statsapi — same source the daily build already uses).
  3. mlb_kprops.project_start() -> lambda + fair prices for the line ladder.
  4. Pull pitcher_strikeouts odds across the books (The Odds API), group by
     (pitcher, line), devig, and apply the SAME DOUBLE-GATE as HR Shop:
     a play must be +EV vs our fair AND vs the vig-free consensus.
  5. Write data/kprops.json: the model board + any +EV plays.

Key from ODDS_API_KEY secret only. Degrades gracefully: no key -> model board
only (still useful — the fair ladder IS the read); no stats for a pitcher ->
skip him rather than fake a number.
"""
import os, json, time, urllib.request, urllib.parse, datetime as dt, unicodedata
import mlb_kprops as KP
import mlb_lineshop as LS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
API_KEY = os.environ.get("ODDS_API_KEY", "")
BOOKS = ["draftkings", "fanduel", "betrivers", "williamhill_us"]
SPORT = "baseball_mlb"

def norm(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return "".join(c for c in s.lower() if c.isalnum())

# ---------------------------------------------------------------- stats pulls
def pull_pitcher_stats(names):
    """{norm_name: {'so':, 'ip':, 'gs':}} via statsapi lookup+stats."""
    import statsapi
    out = {}
    for nm in names:
        try:
            hits = statsapi.lookup_player(nm)
            if not hits: continue
            pid = hits[0]["id"]
            data = statsapi.player_stat_data(pid, group="pitching", type="season")
            for s in data.get("stats", []):
                st = s.get("stats", {})
                ip = float(st.get("inningsPitched", 0) or 0)
                out[norm(nm)] = {
                    "so": int(st.get("strikeOuts", 0) or 0),
                    "ip": ip,
                    "gs": int(st.get("gamesStarted", 0) or 0),
                }
                break
        except Exception as e:
            print(f"  [stats] {nm}: {type(e).__name__}")
        time.sleep(0.2)
    return out

def pull_team_whiff():
    """{team_name_norm: {'so':, 'pa':}} — team BATTING strikeouts, via statsapi
    season hitting stats per team."""
    import statsapi
    out = {}
    try:
        teams = statsapi.get("teams", {"sportId": 1}).get("teams", [])
        for t in teams:
            tid, tname = t["id"], t["name"]
            try:
                st = statsapi.get("team_stats", {"teamId": tid, "group": "hitting",
                                                  "stats": "season"})
                for grp in st.get("stats", []):
                    for sp in grp.get("splits", []):
                        s = sp.get("stat", {})
                        out[norm(tname)] = {
                            "so": int(s.get("strikeOuts", 0) or 0),
                            "pa": int(s.get("plateAppearances", 0) or 0),
                        }
            except Exception:
                pass
            time.sleep(0.15)
    except Exception as e:
        print(f"  [team whiff] {type(e).__name__}")
    return out

# ---------------------------------------------------------------- odds pulls
QUOTA = {"remaining": None, "used": None}
def _get(url):
    with urllib.request.urlopen(url, timeout=25) as r:
        h = r.headers
        if h.get("x-requests-remaining") is not None:
            QUOTA["remaining"] = h.get("x-requests-remaining")
            QUOTA["used"] = h.get("x-requests-used")
        return json.loads(r.read())

def list_events():
    return _get(f"https://api.the-odds-api.com/v4/sports/{SPORT}/events?apiKey={API_KEY}&dateFormat=iso")

def event_k_odds(event_id):
    """{(pitcher_norm, line): {'over': {book: am}, 'under': {book: am}}}"""
    q = urllib.parse.urlencode({
        "apiKey": API_KEY, "regions": "us", "markets": "pitcher_strikeouts",
        "oddsFormat": "american", "bookmakers": ",".join(BOOKS)})
    data = _get(f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds?{q}")
    out = {}
    for bk in data.get("bookmakers", []):
        book = bk.get("key")
        for mk in bk.get("markets", []):
            if mk.get("key") != "pitcher_strikeouts": continue
            for oc in mk.get("outcomes", []):
                pname = oc.get("description") or oc.get("participant") or ""
                side = (oc.get("name") or "").lower()
                line = oc.get("point"); price = oc.get("price")
                if not pname or side not in ("over","under") or line is None or price is None:
                    continue
                key = (norm(pname), float(line))
                out.setdefault(key, {"over": {}, "under": {}})[side][book] = price
    return out

# ---------------------------------------------------------------- main
def main():
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")
    slate = json.load(open(os.path.join(DATA, "slate.json")))
    games = slate.get("games", [])

    # tonight's starters + their opponent team names
    matchups = []   # (pitcher_name, opposing_team_name)
    for g in games:
        if g.get("away_sp"): matchups.append((g["away_sp"], g.get("home","")))
        if g.get("home_sp"): matchups.append((g["home_sp"], g.get("away","")))
    print(f"starters on slate: {len(matchups)}")

    pstats = pull_pitcher_stats([m[0] for m in matchups])
    whiff  = pull_team_whiff()
    print(f"pitcher stats pulled: {len(pstats)} | team whiff rows: {len(whiff)}")

    board = []
    fair_by_pitcher = {}      # norm_name -> {line: p_over}
    for pname, opp in matchups:
        ps = pstats.get(norm(pname))
        tw = whiff.get(norm(opp))
        if not ps:
            continue
        so_t, pa_t = (tw["so"], tw["pa"]) if tw else (0, 0)
        card = KP.project_start(pname, ps["so"], ps["ip"], ps["gs"], so_t, pa_t, opp)
        board.append(card)
        fair_by_pitcher[norm(pname)] = {l["line"]: l["p_over"] for l in card["lines"]}
    board.sort(key=lambda c: -c["lam"])
    print(f"projected starts: {len(board)}")

    # ---- line shop (if key present) ----
    plays = []
    n_markets = 0
    err_counts = {}
    if API_KEY and board:
        try:
            events = list_events()
        except Exception as e:
            events = []; print(f"odds events failed: {type(e).__name__}")
        err_counts = {}
        for ev in events:
            eid = ev.get("id")
            if not eid: continue
            try:
                odds = event_k_odds(eid)
            except Exception as e:
                code = getattr(e, "code", None) or type(e).__name__
                err_counts[str(code)] = err_counts.get(str(code), 0) + 1
                continue
            for (pn, line), sides in odds.items():
                fairs = fair_by_pitcher.get(pn)
                if not fairs: continue
                # model P(over) at the book's exact line (interpolate ladder if needed)
                if line in fairs:
                    fp = fairs[line]
                else:
                    import mlb_kprops as _kp
                    lam = next((c["lam"] for c in board if norm(c["pitcher"])==pn), None)
                    if lam is None: continue
                    fp = _kp.p_over(line, lam)
                over, under = sides.get("over", {}), sides.get("under", {})
                if not over: continue
                a = LS.analyze_player(f"{pn} O{line}", fp, over, under or None)
                if a:
                    n_markets += 1
                    plays.append(a)
            time.sleep(0.3)
        plays = LS.rank_board(plays, min_ev_fair=0.0, min_books=2)

    diag = f"errors {err_counts if API_KEY else 'no-key'} · credits remaining {QUOTA['remaining']} (used {QUOTA['used']})"
    print("DIAG:", diag)
    out = {"generated": ts, "board": board, "diag": diag,
           "n_markets": n_markets, "n_plays": len(plays), "plays": plays,
           "books": BOOKS,
           "note": "Board = model fair ladder per starter. Plays (if any) are +EV vs BOTH "
                   "our fair AND the vig-free consensus at the book's exact line."}
    json.dump(out, open(os.path.join(DATA, "kprops.json"), "w"), indent=1)
    print(f"\nK PROPS: {len(board)} starts projected | {n_markets} markets shopped | {len(plays)} plays")
    for p in plays[:8]:
        print(f"  {p['player']:<26} {p['best_book']:<11} {p['best_price']:+d}  EV(min) {p['conservative_ev_pct']}%")

if __name__ == "__main__":
    main()
