"""
mlb_grade.py  —  the grader: turns the prediction log into calibration
=======================================================================
Every board publishes ~35 falsifiable P(homer) claims into hr_predictions.csv.
This module settles them against reality and answers the only questions that
matter for tuning:

  * Does 30% mean 30%?             (buckets: predicted vs actual, Brier)
  * Does each factor earn its keep? (lift: heat+/-, platoon class, card vs proj)
  * Do the +EV rows make money?     (flat-stake ROI at the logged book price)

OUTCOMES per row:
  hr      player homered in the matched game
  no      player had >=1 PA in the matched game, no homer
  void    player logged 0 PA (scratched / never entered) — books void these,
          so they're excluded from calibration but COUNTED: void-rate grades
          the lineup layer itself
  pending game not final yet (graded on a later run; grading is idempotent)

Doubleheaders: a player has one row per game; each row is matched to its game
by the opposing starter's name. Unmatchable rows fall to void-ambiguous.

RUN:  python mlb_grade.py            # grade all settleable past dates
      python mlb_grade.py --selftest # offline validation, no network
Publish hook: mlb_grade.summarize(rows) -> panel dict for slate.json
"""
import os, sys, json, csv, math, glob, unicodedata, datetime as dt
import urllib.request, urllib.parse

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PLOG = os.path.join(DATA, "hr_predictions.csv")
GRADED = os.path.join(DATA, "hr_graded.csv")
GCOLS = ["date","player","team","opp_sp","slot","lu","hr_pct","fair",
         "book_price","ev_pct","park","temp","plat","heat","outcome"]

def norm(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode()
    s = s.lower().replace(".","").replace("'","")
    for suf in (" jr"," sr"," ii"," iii"," iv"):
        if s.endswith(suf): s = s[:-len(suf)]
    return " ".join(s.split())

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0 (MLBTool grader)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

# ---------------------------------------------------------------------------
# results fetch: per date -> per team, list of games with starter + batter facts
def fetch_day_results(date_iso):
    """-> (games, all_final)
    games: [{'teams': {full_name: {'opp_sp': norm_name,
                                   'bat': {norm_player: {'pa': int, 'hr': int}}}}}]"""
    mdy = dt.date.fromisoformat(date_iso).strftime("%m/%d/%Y")
    base = "https://statsapi.mlb.com/api/v1"
    sched = _get(f"{base}/schedule?sportId=1&date={mdy}")
    games, all_final = [], True
    for d0 in sched.get("dates", []):
        for g in d0.get("games", []):
            state = ((g.get("status") or {}).get("abstractGameState") or "")
            if state != "Final":
                if state not in ("",):
                    all_final = False
                continue
            try:
                box = _get(f"{base}/game/{g['gamePk']}/boxscore")
            except Exception:
                all_final = False; continue
            entry = {"teams": {}}
            names, starters = {}, {}
            for side in ("home","away"):
                t = (box.get("teams") or {}).get(side) or {}
                full = ((g.get("teams") or {}).get(side) or {}).get("team",{}).get("name","")
                bat = {}
                for pid_key, p in (t.get("players") or {}).items():
                    nm = norm((p.get("person") or {}).get("fullName",""))
                    st = ((p.get("stats") or {}).get("batting") or {})
                    if nm:
                        bat[nm] = {"pa": int(st.get("plateAppearances",0) or 0),
                                   "hr": int(st.get("homeRuns",0) or 0)}
                pit = t.get("pitchers") or []
                sp = ""
                if pit:
                    sp = norm(((t.get("players") or {}).get(f"ID{pit[0]}",{})
                               .get("person") or {}).get("fullName",""))
                names[side], starters[side] = full, sp
                entry["teams"][full] = {"bat": bat, "opp_sp": ""}
            # each team's opponent starter
            entry["teams"][names["home"]]["opp_sp"] = starters["away"]
            entry["teams"][names["away"]]["opp_sp"] = starters["home"]
            games.append(entry)
    return games, all_final

# ---------------------------------------------------------------------------
# pure: settle one prediction row against a day's games
def settle_row(row, games):
    """-> 'hr' | 'no' | 'void' | 'pending' (pending = team not found in finals)"""
    pn = norm(row.get("player",""))
    want_sp = norm((row.get("opp_sp","") or "").replace(" *","").replace("TBD",""))
    cands = []
    for g in games:
        for full, t in g["teams"].items():
            if pn in t["bat"]:
                cands.append(t)
    if not cands: return "pending"
    if len(cands) > 1 and want_sp:                 # doubleheader: match by starter
        m = [t for t in cands if t["opp_sp"] == want_sp]
        if len(m) == 1: cands = m
        elif not m:     return "void"              # row's game unidentifiable
        else:           cands = m[:1]
    t = cands[0]
    b = t["bat"][pn]
    if b["pa"] <= 0: return "void"
    return "hr" if b["hr"] >= 1 else "no"

# ---------------------------------------------------------------------------
# pure: aggregate graded rows into the calibration panel
def _dec(am):
    try:
        a = int(am);  return a/100+1 if a > 0 else 100/(-a)+1
    except Exception: return None

def summarize(rows):
    """rows: graded dicts (outcome in hr/no/void). -> panel dict (JSON-safe)."""
    live = [r for r in rows if r.get("outcome") in ("hr","no")]
    voids = sum(1 for r in rows if r.get("outcome") == "void")
    n = len(live)
    panel = {"n": n, "voids": voids, "dates": len({r["date"] for r in rows}) if rows else 0}
    if not n: return panel
    p = [float(r["hr_pct"])/100 for r in live]
    y = [1.0 if r["outcome"]=="hr" else 0.0 for r in live]
    panel["pred_mean"] = round(100*sum(p)/n, 1)
    panel["actual"]    = round(100*sum(y)/n, 1)
    panel["brier"]     = round(sum((pi-yi)**2 for pi,yi in zip(p,y))/n, 4)
    # buckets
    edges = [(0,12),(12,16),(16,20),(20,25),(25,100)]
    bks = []
    for lo,hi in edges:
        sel = [(pi,yi) for pi,yi in zip(p,y) if lo <= pi*100 < hi]
        if sel:
            bks.append({"bucket": f"{lo}-{hi if hi<100 else '+'}",
                        "n": len(sel),
                        "pred": round(100*sum(a for a,_ in sel)/len(sel),1),
                        "actual": round(100*sum(b for _,b in sel)/len(sel),1)})
    panel["buckets"] = bks
    # factor lift
    def grp(label, sel):
        if not sel: return None
        pp = [float(r["hr_pct"])/100 for r in sel]
        yy = [1.0 if r["outcome"]=="hr" else 0.0 for r in sel]
        return {"g": label, "n": len(sel),
                "pred": round(100*sum(pp)/len(sel),1),
                "actual": round(100*sum(yy)/len(sel),1)}
    lifts = []
    lifts.append(grp("heat +", [r for r in live if str(r.get("heat","")).startswith("heat +") and r.get("heat")!="heat +0%"]))
    lifts.append(grp("heat −", [r for r in live if str(r.get("heat","")).startswith("heat -")]))
    for cls in ("LvL","LvR","RvL","RvR"):
        lifts.append(grp(cls, [r for r in live if str(r.get("plat","")).startswith(cls)]))
    lifts.append(grp("card", [r for r in live if r.get("lu")=="card"]))
    lifts.append(grp("proj", [r for r in live if r.get("lu")=="proj"]))
    panel["lift"] = [x for x in lifts if x]
    # +EV tier: flat 1u at logged book price
    ev = [r for r in live if r.get("ev_pct") not in ("", None) and float(r["ev_pct"]) > 0
          and r.get("book_price") not in ("", None)]
    if ev:
        pl = 0.0
        for r in ev:
            d = _dec(r["book_price"])
            if d is None: continue
            pl += (d-1) if r["outcome"]=="hr" else -1.0
        panel["ev_tier"] = {"n": len(ev),
                            "hits": sum(1 for r in ev if r["outcome"]=="hr"),
                            "roi": round(100*pl/len(ev),1)}
    return panel

# ---------------------------------------------------------------------------
def load_csv(path):
    if not os.path.exists(path): return []
    with open(path) as f: return list(csv.DictReader(f))

def grade_all():
    preds = load_csv(PLOG)
    if not preds:
        print("no prediction log yet"); return
    done = {(r["date"], norm(r["player"]), r.get("opp_sp","")) for r in load_csv(GRADED)}
    today = dt.date.today().isoformat()
    dates = sorted({r["date"] for r in preds if r["date"] < today})
    new = []
    for d in dates:
        rows = [r for r in preds if r["date"]==d
                and (d, norm(r["player"]), r.get("opp_sp","")) not in done]
        if not rows: continue
        try:
            games, _fin = fetch_day_results(d)
        except Exception as e:
            print(f"  {d}: results fetch failed ({type(e).__name__}) — retry next run"); continue
        settled = 0
        for r in rows:
            o = settle_row(r, games)
            if o == "pending": continue
            rec = {k: r.get(k,"") for k in GCOLS[:-1]}; rec["outcome"] = o
            new.append(rec); settled += 1
        print(f"  {d}: settled {settled}/{len(rows)}")
    if new:
        exists = os.path.exists(GRADED)
        with open(GRADED, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=GCOLS)
            if not exists: w.writeheader()
            for r in new: w.writerow(r)
    allg = load_csv(GRADED)
    pan = summarize(allg)
    print(f"graded total: {pan.get('n',0)} live rows, {pan.get('voids',0)} voids "
          f"across {pan.get('dates',0)} day(s)"
          + (f" | actual {pan['actual']}% vs pred {pan['pred_mean']}% | Brier {pan['brier']}"
             if pan.get("n") else ""))

def panel_for_publish():
    """publish hook — never raises."""
    try:
        return summarize(load_csv(GRADED))
    except Exception as e:
        return {"n": 0, "error": type(e).__name__}

# ---------------------------------------------------------------------------
def selftest():
    G = [{"teams": {
        "New York Yankees": {"opp_sp": "gopher gary",
            "bat": {"slug mcpower": {"pa":4,"hr":1},
                    "mid bat": {"pa":4,"hr":0},
                    "benched guy": {"pa":0,"hr":0}}},
        "Boston Red Sox": {"opp_sp": "ace groundall",
            "bat": {"sox star": {"pa":5,"hr":0}}}}}]
    row = lambda **k: dict({"date":"2026-07-07","player":"","opp_sp":"","hr_pct":"20",
                            "lu":"card","plat":"","heat":"","ev_pct":"","book_price":""}, **k)
    assert settle_row(row(player="Slug McPower", opp_sp="Gopher Gary"), G) == "hr"
    assert settle_row(row(player="Mid Bat", opp_sp="Gopher Gary"), G) == "no"
    assert settle_row(row(player="Benched Guy", opp_sp="Gopher Gary"), G) == "void"
    assert settle_row(row(player="Ghost Man", opp_sp="X"), G) == "pending"
    # doubleheader: same player two games, starter disambiguates
    G2 = [ {"teams": {"Milwaukee Brewers": {"opp_sp":"starter one",
                        "bat": {"jake bauers":{"pa":4,"hr":1}}}}},
           {"teams": {"Milwaukee Brewers": {"opp_sp":"starter two",
                        "bat": {"jake bauers":{"pa":3,"hr":0}}}}} ]
    assert settle_row(row(player="Jake Bauers", opp_sp="Starter One"), G2) == "hr"
    assert settle_row(row(player="Jake Bauers", opp_sp="Starter Two"), G2) == "no"
    assert settle_row(row(player="Jake Bauers", opp_sp="Starter Three"), G2) == "void"
    # summarize math
    rows = [
        row(player="A", hr_pct="30", outcome="hr",  heat="heat +5%", plat="RvL +12%", lu="card", ev_pct="10", book_price="200"),
        row(player="B", hr_pct="30", outcome="no",  heat="heat +3%", plat="RvL +12%", lu="card", ev_pct="12", book_price="300"),
        row(player="C", hr_pct="10", outcome="no",  heat="heat -4%", plat="LvL -22%", lu="proj"),
        row(player="D", hr_pct="10", outcome="void"),
    ]
    p = summarize(rows)
    assert p["n"] == 3 and p["voids"] == 1
    assert p["pred_mean"] == round(100*(0.3+0.3+0.1)/3,1)
    assert p["brier"] == round((0.7**2 + 0.3**2 + 0.1**2)/3, 4), p["brier"]
    assert any(b["bucket"]=="25-+" and b["n"]==2 for b in p["buckets"])
    hplus = [x for x in p["lift"] if x["g"]=="heat +"][0]
    assert hplus["n"]==2 and hplus["actual"]==50.0
    ev = p["ev_tier"]     # A wins at +200 (+2u), B loses (-1u) -> +1u/2 = +50%
    assert ev["n"]==2 and ev["hits"]==1 and ev["roi"]==50.0
    json.dumps(p)
    print("GRADER SELFTEST PASS — settle/void/pending/DH + Brier/buckets/lift/ROI all exact")
    return 0

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    grade_all()
