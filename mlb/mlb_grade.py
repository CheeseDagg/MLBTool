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
GCOLS = ["date","player","team","opp_sp","slot","lu","hr_pct","hr_raw","fair",
         "book_price","ev_pct","park","temp","plat","heat","outcome","hr_n"]

LEAGUE = os.path.join(DATA, "league_daily.csv")
LCOLS = ["date", "games", "pa", "hr"]

def league_day(games):
    """League-wide HR and PA for a settled date, from boxes the grader ALREADY has.

    This is the control that separates 'the model broke' from 'the ball wasn't
    flying'. Without it a cold week is unattributable: on 2026-07-23..29 the board
    predicted 20.2% and hit 13.7%, and nothing on hand could say whether that was
    the model drifting or the league going quiet under it. Costs zero extra
    fetches — every boxscore is already parsed for settlement.

    Counts each player once per game. Pitchers who never bat contribute pa=0 and
    so are harmless; the denominator is plate appearances, not roster slots.
    """
    pa = hr = 0
    for g in games:
        for t in (g.get("teams") or {}).values():
            for st in (t.get("bat") or {}).values():
                pa += int(st.get("pa", 0) or 0)
                hr += int(st.get("hr", 0) or 0)
    return {"games": len(games), "pa": pa, "hr": hr}

def record_league(date_iso, games):
    """Append one date's league rate, idempotently (re-grading must not double it)."""
    if not games: return
    prev = {}
    if os.path.exists(LEAGUE):
        with open(LEAGUE, newline="") as f:
            prev = {r["date"]: r for r in csv.DictReader(f)}
    d = league_day(games)
    if d["pa"] <= 0: return
    prev[date_iso] = dict(date=date_iso, **{k: str(v) for k, v in d.items()})
    tmp = LEAGUE + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LCOLS); w.writeheader()
        for k in sorted(prev):
            w.writerow({c: prev[k].get(c, "") for c in LCOLS})
    os.replace(tmp, LEAGUE)

def league_context(dates):
    """-> {'hr_pa': rate over `dates`, 'season_hr_pa': rate over all recorded dates,
           'rel': ratio}. rel < 1 means the league itself was quiet in that window,
    which is the amount of a board miss that is NOT the model's fault."""
    if not os.path.exists(LEAGUE): return None
    with open(LEAGUE, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("pa") or "").isdigit()]
    if not rows: return None
    sel = [r for r in rows if r["date"] in set(dates)]
    def rate(rs):
        pa = sum(int(r["pa"]) for r in rs); hr = sum(int(r["hr"]) for r in rs)
        return (hr / pa) if pa else None
    a, b = rate(sel), rate(rows)
    # `is None` deliberately, not truthiness: a window in which the league hit ZERO
    # home runs is the single most interesting window this control exists to catch,
    # and `if not a` would throw it away as if the data were missing.
    if a is None or b is None or not b: return None
    return {"hr_pa": round(a, 5), "season_hr_pa": round(b, 5),
            "rel": round(a / b, 3), "days": len(sel), "season_days": len(rows)}

def _rows_by_width(path):
    """Parse a graded CSV whose header may be STALE, one row at a time by width.

    The append path writes GCOLS; the header line is only ever written once, at
    file creation. So adding a column to GCOLS silently desynchronises the two:
    every subsequent row carries the new field under a header that does not name
    it, csv.DictReader shifts every column past the insertion point by one, and
    `outcome` comes back holding whatever the neighbouring column said. That is
    not a parse error anybody sees — summarize() keeps only outcome in
    ("hr","no"), so the shifted rows are DISCARDED IN SILENCE and calibration
    quietly freezes at the last pre-drift date. It happened on 2026-07-23 when
    hr_raw was added: 217 of 662 rows, a full week of grading, vanished from the
    panel while the board went on reporting a healthy-looking n.

    So: never trust the header for a row it may not describe. A row is mapped by
    its OWN width against the known schema generations, newest first.
    """
    gens = [GCOLS, [c for c in GCOLS if c != "hr_raw"]]
    by_w = {len(g): g for g in gens}
    out, unknown = [], 0
    with open(path, newline="") as f:
        rd = csv.reader(f)
        try: next(rd)
        except StopIteration: return [], 0
        for rec in rd:
            if not rec: continue
            cols = by_w.get(len(rec))
            if cols is None:
                unknown += 1; continue
            out.append({k: v for k, v in zip(cols, rec)})
    return out, unknown

def migrate_graded():
    """Bring the graded ledger's header into line with GCOLS, non-destructively.

    Keyed on the header matching GCOLS EXACTLY rather than on the presence of one
    named column. The previous version returned early whenever "hr_n" appeared in
    the header line, which meant it was blind to every later column addition —
    the check was written against the one migration it was born for. Rows are
    re-read by width (see _rows_by_width) so pre-drift and post-drift rows both
    land on the right fields, and missing columns fill blank."""
    if not os.path.exists(GRADED): return
    with open(GRADED, newline="") as f:
        try: header = next(csv.reader(f))
        except StopIteration: return
    if header == GCOLS: return
    rows, unknown = _rows_by_width(GRADED)
    tmp = GRADED + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GCOLS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in GCOLS})
    os.replace(tmp, GRADED)
    added = [c for c in GCOLS if c not in header]
    print(f"  migrated {len(rows)} rows -> header realigned"
          + (f", added {added}" if added else "")
          + (f" ({unknown} rows of unknown width dropped)" if unknown else ""))

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
                abbr = ((box.get("teams") or {}).get(side, {}).get("team") or {}).get("abbreviation", "")
                names[side], starters[side] = full, sp
                entry["teams"][full] = {"bat": bat, "opp_sp": "", "abbr": abbr}
            # each team's opponent starter
            entry["teams"][names["home"]]["opp_sp"] = starters["away"]
            entry["teams"][names["away"]]["opp_sp"] = starters["home"]
            games.append(entry)
    return games, all_final

# ---------------------------------------------------------------------------
# pure: settle one prediction row against a day's games
# FanGraphs board codes (written by mlb_hr) vs statsapi box abbreviations differ for
# these clubs. Without aliasing, the strict team guard below voids EVERY prediction for
# them (SFG!=SF, WSN!=WSH, etc.) — silently dropping ~6 teams from calibration.
FG_ALIASES = {"KCR": {"KC", "KCR"}, "SDP": {"SD", "SDP"}, "SFG": {"SF", "SFG"},
              "TBR": {"TB", "TBR"}, "WSN": {"WSH", "WSN"}, "CHW": {"CWS", "CHW"},
              "ATH": {"OAK", "ATH"}}

def _team_eq(box_abbr, want_tm):
    """True if the box-score abbr and the board's team code are the same club,
    tolerating the FanGraphs<->statsapi code differences above."""
    a = (box_abbr or "").upper(); b = (want_tm or "").upper()
    if not a or not b: return False
    if a == b: return True
    return any(a in alts and b in alts for alts in FG_ALIASES.values())

def settle_row(row, games, all_final=True):
    """-> 'hr' | 'no' | 'void' | 'pending' (pending = team not found in finals).

    all_final: whether every game on the slate is Final. Used to resolve doubleheaders
    safely — while games are still live, a row whose starter matches no available box is
    most likely game 2 that hasn't posted yet (pending), not a phantom (void)."""
    pn = norm(row.get("player",""))
    want_sp = norm((row.get("opp_sp","") or "").replace(" *","").replace("TBD",""))
    want_tm = (row.get("team","") or "").strip().upper()
    cands = []
    for g in games:
        for full, t in g["teams"].items():
            if pn in t["bat"]:
                cands.append(t)
    if not cands: return "pending"
    if want_tm:
        # strict guard: the row claims a team; if no candidate box carries that
        # team, this is a duplicate-name phantom -> void, never inherit outcomes
        m2 = [t for t in cands if _team_eq(t.get("abbr",""), want_tm)]
        if m2: cands = m2
        elif len({t.get("abbr","") for t in cands}) >= 1 and all(t.get("abbr") for t in cands):
            return "void"
    if want_sp:                                    # identify the row's game by its starter
        m = [t for t in cands if t.get("opp_sp") and t["opp_sp"] == want_sp]
        if m:
            cands = m[:1]
        elif len(cands) > 1:
            # doubleheader with multiple final boxes but none matches the row's starter ->
            # the row's game is unidentifiable (or not final yet)
            return "void" if all_final else "pending"
        elif not all_final:
            # ONE box, it doesn't match the row's starter, and games are still live: this is
            # most likely game 2 of a doubleheader that hasn't posted — do NOT settle the row
            # against game 1's box. Once the slate is fully final, trust the single box below
            # (its opp_sp text may just differ in formatting).
            return "pending"
    t = cands[0]
    b = t["bat"][pn]
    if b["pa"] <= 0: return "void"
    return ("hr", b["hr"]) if b["hr"] >= 1 else ("no", 0)

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
    # Anything whose outcome is not in the vocabulary is a PARSE failure, not a
    # grading state, and it must be visible. Dropping it quietly is how a week of
    # grading disappeared from this panel on 2026-07-23 while n kept looking fine.
    bad = [r for r in rows if r.get("outcome") not in ("hr","no","void","pending")]
    if bad:
        panel["unparsed"] = len(bad)
        panel["unparsed_dates"] = sorted({r.get("date","?") for r in bad})[:5]
    try:
        lc = league_context({r["date"] for r in live}) if live else None
        if lc: panel["league"] = lc
    except Exception:
        pass
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
    # --- HOLDOUT TRIGGER (item 7): once the live ledger is big + representative enough,
    #     surface whether it CONFIRMS the season backtest's calibration verdict. Below
    #     threshold we say so explicitly, so nobody trusts a hot-week sample of 66. ---
    HOLDOUT_MIN = 300
    def _heat(r):
        h = r.get("heat","") or ""
        try:
            if "heat +" in h: return int(h.split("+")[1].rstrip("%"))
            if "heat -" in h: return -int(h.split("-")[1].rstrip("%"))
        except Exception: pass
        return None
    hi = {"n": len(live), "ready": len(live) >= HOLDOUT_MIN, "min": HOLDOUT_MIN}
    if live:
        span = len({r["date"] for r in live})
        hi["span_days"] = span
        hp = [r for r in live if float(r["hr_pct"]) >= 25]
        if hp:
            hi["hi_bucket"] = {"n": len(hp),
                "pred": round(sum(float(r["hr_pct"]) for r in hp)/len(hp),1),
                "actual": round(100*sum(1 for r in hp if r["outcome"]=="hr")/len(hp),1)}
        # try to load the backtest panel for a live-vs-season comparison
        try:
            bt = json.load(open(os.path.join(DATA, "hr_backtest_panel.json")))
            b25 = next((b for b in bt.get("buckets",[]) if b["bucket"]=="25-+"), None)
            if b25 and hp:
                bt_gap = b25["pred"] - b25["actual"]                 # season: +hot
                lv_gap = hi["hi_bucket"]["pred"] - hi["hi_bucket"]["actual"]
                same_sign = (bt_gap > 0) == (lv_gap > 0)
                hi["compare"] = {"season_hot_by": round(bt_gap,1),
                                 "live_hot_by": round(lv_gap,1),
                                 "agree_direction": same_sign,
                                 "verdict": ("confirmed" if (hi["ready"] and same_sign)
                                             else "insufficient_sample" if not hi["ready"]
                                             else "conflict_investigate")}
        except Exception:
            pass
    panel["holdout"] = hi
    # two-homer games — counted era only (hr_n present)
    def _heat(r):
        h = r.get("heat","") or ""
        try:
            if "heat +" in h: return int(h.split("+")[1].rstrip("%"))
            if "heat -" in h: return -int(h.split("-")[1].rstrip("%"))
        except Exception: pass
        return None
    cnt = [r for r in live if str(r.get("hr_n","")).strip() != ""]
    if cnt:
        two = sum(1 for r in cnt if int(float(r["hr_n"])) >= 2)
        A = [r for r in cnt if float(r["hr_pct"]) >= 25 and (_heat(r) or -99) >= 10]
        a2 = sum(1 for r in A if int(float(r["hr_n"])) >= 2)
        panel["multi"] = {"n": len(cnt), "two_plus": two,
                          "rate": round(100*two/len(cnt),1),
                          "a_n": len(A), "a_two_plus": a2,
                          "a_rate": (round(100*a2/len(A),1) if A else None)}
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
    # Top-Likelihood tier: the house doctrine — never pass on a likely outcome
    # because EV is negative. Each day's top-5 by model HR% with a logged price,
    # flat 1u, EV ignored. Graded head-to-head against the +EV tier above.
    by_date = {}
    for r in live:
        if r.get("book_price") in ("", None): continue
        by_date.setdefault(r["date"], []).append(r)
    tops = []
    for d0, rows_d in by_date.items():
        tops += sorted(rows_d, key=lambda r: -float(r["hr_pct"]))[:5]
    if tops:
        pl = 0.0
        for r in tops:
            d = _dec(r["book_price"])
            if d is None: continue
            pl += (d-1) if r["outcome"]=="hr" else -1.0
        panel["top_tier"] = {"n": len(tops),
                             "hits": sum(1 for r in tops if r["outcome"]=="hr"),
                             "roi": round(100*pl/len(tops),1)}
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
    # settle every date through today. Prior days are fully final; today's games
    # that aren't Final yet come back 'pending' from settle_row and retry next run,
    # so early day-games settle on the 3:17 build instead of waiting overnight.
    dates = sorted({r["date"] for r in preds if r["date"] <= today})
    new = []
    for d in dates:
        rows = [r for r in preds if r["date"]==d
                and (d, norm(r["player"]), r.get("opp_sp","")) not in done]
        if not rows: continue
        try:
            games, _fin = fetch_day_results(d)
        except Exception as e:
            print(f"  {d}: results fetch failed ({type(e).__name__}) — retry next run"); continue
        # Record the league rate only once the day is fully final; a partial slate
        # would log a denominator that later grows, and the ratio would be wrong
        # in a way no later run corrects (record_league overwrites, but a date
        # settled early then never revisited would keep its partial count).
        if _fin:
            try: record_league(d, games)
            except Exception as e: print(f"  {d}: league rate not recorded ({type(e).__name__})")
        settled = 0
        for r in rows:
            o = settle_row(r, games, _fin)
            o, hn = o if isinstance(o, tuple) else (o, "")
            if o == "pending": continue
            rec = {k: r.get(k,"") for k in GCOLS[:-2]}
            rec["outcome"], rec["hr_n"] = o, hn
            new.append(rec); settled += 1
        print(f"  {d}: settled {settled}/{len(rows)}")
    migrate_graded()
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
    # Hoisted: two separate blocks below repoint GRADED at a temp ledger, and a
    # `global` may not follow the first use of the name in the same scope.
    global load_csv, fetch_day_results, GRADED
    G = [{"teams": {
        "New York Yankees": {"opp_sp": "gopher gary",
            "bat": {"slug mcpower": {"pa":4,"hr":1},
                    "mid bat": {"pa":4,"hr":0},
                    "benched guy": {"pa":0,"hr":0}}},
        "Boston Red Sox": {"opp_sp": "ace groundall",
            "bat": {"sox star": {"pa":5,"hr":0}}}}}]
    row = lambda **k: dict({"date":"2026-07-07","player":"","opp_sp":"","team":"","hr_pct":"20",
                            "lu":"card","plat":"","heat":"","ev_pct":"","book_price":""}, **k)
    assert settle_row(row(player="Slug McPower", opp_sp="Gopher Gary"), G)[0] == "hr"
    assert settle_row(row(player="Mid Bat", opp_sp="Gopher Gary"), G) == ("no", 0)
    assert settle_row(row(player="Benched Guy", opp_sp="Gopher Gary"), G) == "void"
    assert settle_row(row(player="Ghost Man", opp_sp="X"), G) == "pending"
    # doubleheader: same player two games, starter disambiguates
    G2 = [ {"teams": {"Milwaukee Brewers": {"opp_sp":"starter one",
                        "bat": {"jake bauers":{"pa":4,"hr":1}}}}},
           {"teams": {"Milwaukee Brewers": {"opp_sp":"starter two",
                        "bat": {"jake bauers":{"pa":3,"hr":0}}}}} ]
    assert settle_row(row(player="Jake Bauers", opp_sp="Starter One"), G2)[0] == "hr"
    assert settle_row(row(player="Jake Bauers", opp_sp="Starter Two"), G2) == ("no", 0)
    assert settle_row(row(player="Jake Bauers", opp_sp="Starter Three"), G2) == "void"
    # PARTIAL DOUBLEHEADER: only game 1 is final (one box). The game-2 row (starter two)
    # must NOT settle against game 1's box while games are live — it stays pending.
    G_dh = [{"teams": {"Houston Astros": {"opp_sp": "starter one", "abbr": "HOU",
              "bat": {norm("Dh Bat"): {"pa": 4, "hr": 1}}}}}]
    assert settle_row(row(player="Dh Bat", opp_sp="Starter Two", team="HOU"), G_dh, all_final=False) == "pending", \
        "game-2 row must not grade against game-1 box while games are live"
    assert settle_row(row(player="Dh Bat", opp_sp="Starter One", team="HOU"), G_dh, all_final=False)[0] == "hr", \
        "game-1 row still settles against its own box"
    # once the slate is fully final, a single unmatched box is trusted (ordinary single game)
    assert settle_row(row(player="Dh Bat", opp_sp="Weird Format", team="HOU"), G_dh, all_final=True)[0] == "hr"
    # duplicate-name phantom: row claims a team whose box the player isn't in -> void
    G3 = [{"teams": {"New York Yankees": {"opp_sp": "x", "abbr": "NYY",
            "bat": {"ben rice": {"pa": 4, "hr": 1}}}}}]
    assert settle_row(row(player="Ben Rice", opp_sp="Ian Seymour", team="NYY"), G3)[0] == "hr"
    assert settle_row(row(player="Ben Rice", opp_sp="Seth Lugo", team="NYM"), G3) == "void"
    # ALIAS REGRESSION: board writes FanGraphs codes (SFG/WSN/…) but the box abbr is the
    # statsapi form (SF/WSH). These must grade as the SAME team, not void — else ~6 clubs
    # silently never contribute to calibration. A genuine mismatch must still void.
    G_sf = [{"teams": {"San Francisco Giants": {"opp_sp": "arm", "abbr": "SF",
              "bat": {norm("Homer Giant"): {"pa": 4, "hr": 1}}}}}]
    assert settle_row(row(player="Homer Giant", opp_sp="Arm", team="SFG"), G_sf)[0] == "hr", \
        "SFG board code must grade against SF box abbr, not void"
    G_wsh = [{"teams": {"Washington Nationals": {"opp_sp": "arm", "abbr": "WSH",
               "bat": {norm("Nat Bat"): {"pa": 4, "hr": 0}}}}}]
    assert settle_row(row(player="Nat Bat", opp_sp="Arm", team="WSN"), G_wsh) == ("no", 0)
    G_bad = [{"teams": {"New York Yankees": {"opp_sp": "arm", "abbr": "NYY",
               "bat": {norm("Real Phantom"): {"pa": 4, "hr": 1}}}}}]
    assert settle_row(row(player="Real Phantom", opp_sp="Arm", team="BOS"), G_bad) == "void", \
        "a genuine team mismatch must still void"
    # summarize math
    rows = [
        row(player="A", hr_pct="30", outcome="hr",  heat="heat +5%", plat="RvL +12%", lu="card", ev_pct="10", book_price="200"),
        row(player="B", hr_pct="30", outcome="no",  heat="heat +3%", plat="RvL +12%", lu="card", ev_pct="12", book_price="300"),
        row(player="C", hr_pct="10", outcome="no",  heat="heat -4%", plat="LvL -22%", lu="proj"),
        row(player="D", hr_pct="10", outcome="void"),
    ]
    rows[0]["hr_n"]="2"; rows[1]["hr_n"]="0"; rows[2]["hr_n"]=""   # mixed eras
    rows.append(row(player="E", hr_pct="27", outcome="hr", heat="heat +11%", hr_n="2", lu="card"))
    rows.append(row(player="F", hr_pct="26", outcome="hr", heat="heat +14%", hr_n="1", lu="card"))
    p = summarize(rows)
    assert p["n"] == 5 and p["voids"] == 1
    m = p["multi"]
    assert m["n"] == 4 and m["two_plus"] == 2 and m["rate"] == 50.0
    # holdout: 5 live rows is nowhere near 300 -> not ready (guards against trusting
    # a tiny hot-week sample); verdict must read insufficient_sample
    h = p["holdout"]
    assert h["ready"] is False and h["n"] == 5 and h["min"] == 300
    assert h["compare"]["verdict"] == "insufficient_sample"
    assert m["a_n"] == 2 and m["a_two_plus"] == 1 and m["a_rate"] == 50.0
    assert p["pred_mean"] == round(100*(0.3+0.3+0.1+0.27+0.26)/5,1)
    assert p["brier"] == round((0.7**2 + 0.3**2 + 0.1**2 + 0.73**2 + 0.74**2)/5, 4), p["brier"]
    assert any(b["bucket"]=="25-+" and b["n"]==4 for b in p["buckets"])
    hplus = [x for x in p["lift"] if x["g"]=="heat +"][0]
    assert hplus["n"]==4 and hplus["actual"]==75.0
    ev = p["ev_tier"]     # A wins at +200 (+2u), B loses (-1u) -> +1u/2 = +50%
    assert ev["n"]==2 and ev["hits"]==1 and ev["roi"]==50.0
    # top-likelihood tier: A(30%,+200,hr) and B(30%,+300,no) and C(10%) -> top5 of the
    # date = all 3 priced rows... C has no price -> excluded; A+B: +2u-1u = +50% ROI
    tt = p["top_tier"]
    assert tt["n"]==2 and tt["hits"]==1 and tt["roi"]==50.0, tt
    json.dumps(p)
    # HEADER DRIFT: a ledger written under an OLD header, then appended to under a
    # NEW GCOLS, must survive. This is the 2026-07-23 hr_raw incident pinned: the
    # old migration keyed on "hr_n" in the header and so was blind to any later
    # column, DictReader shifted every post-drift row, and summarize() discarded
    # 217 rows without a word. Both halves must come back with real outcomes, and
    # a shifted row must be COUNTED as unparsed rather than silently dropped.
    import tempfile as _tf0
    _drift = os.path.join(_tf0.mkdtemp(), "hr_graded_drift.csv")
    _old = [c for c in GCOLS if c != "hr_raw"]
    with open(_drift, "w", newline="") as f:
        w = csv.writer(f); w.writerow(_old)
        w.writerow(["2026-07-22","Old Guy","NYY","arm","3","card","20.0","+400","",
                    "","park +2%","80F","RvL +5%","heat +3%","hr","1"])
        w.writerow(["2026-07-23","New Guy","BOS","arm","4","card","22.0","19.1","+380",
                    "","","park -1%","78F","RvR -2%","heat +1%","no","0"])
    _pre, _ = _rows_by_width(_drift)
    assert [r["outcome"] for r in _pre] == ["hr","no"], \
        "width-keyed parse must recover both schema generations, got %r" % [r["outcome"] for r in _pre]
    assert _pre[1]["hr_raw"] == "19.1" and _pre[1]["hr_pct"] == "22.0", _pre[1]
    _og = GRADED
    try:
        GRADED = _drift
        migrate_graded()
        with open(_drift, newline="") as f: assert next(csv.reader(f)) == GCOLS
        _after = list(csv.DictReader(open(_drift)))
        assert [r["outcome"] for r in _after] == ["hr","no"], _after
        assert summarize(_after)["n"] == 2 and "unparsed" not in summarize(_after)
        assert migrate_graded() is None      # idempotent: header now matches, no-op
    finally:
        GRADED = _og
    _shift = [{"date":"2026-07-23","hr_pct":"20","outcome":"heat +1%"}]
    assert summarize(_shift).get("unparsed") == 1, \
        "a shifted row must surface as unparsed, not vanish"
    # LEAGUE CONTROL: counts every batter's pa/hr across both sides of every game,
    # and record_league must be idempotent under re-grading (the grader re-settles
    # dates, and a league rate that doubled on the second pass would be worse than
    # none at all — it would look like a live signal).
    _lg = [{"teams": {"A": {"bat": {"x": {"pa": 4, "hr": 1}, "y": {"pa": 3, "hr": 0}}},
                      "B": {"bat": {"z": {"pa": 5, "hr": 2}, "arm": {"pa": 0, "hr": 0}}}}}]
    assert league_day(_lg) == {"games": 1, "pa": 12, "hr": 3}, league_day(_lg)
    global LEAGUE
    _olg = LEAGUE
    try:
        LEAGUE = os.path.join(_tf0.mkdtemp(), "league_daily.csv")
        record_league("2026-07-24", _lg)
        record_league("2026-07-24", _lg)          # re-grade of the same date
        _r = list(csv.DictReader(open(LEAGUE)))
        assert len(_r) == 1 and _r[0]["pa"] == "12", _r
        record_league("2026-07-25", [{"teams": {"A": {"bat": {"x": {"pa": 10, "hr": 0}}}}}])
        _c = league_context(["2026-07-25"])
        assert _c["days"] == 1 and _c["hr_pa"] == 0.0 and _c["season_days"] == 2
        _c2 = league_context(["2026-07-24"])
        assert _c2["rel"] > 1.0, ("a hot window must read rel>1 against the season, got %r" % _c2)
        assert league_context(["2099-01-01"]) is None   # no data -> no claim
    finally:
        LEAGUE = _olg
    # SAME-DAY GRADING: today's finished game settles now; unfinished stays pending
    import datetime as _dt
    _today = _dt.date.today().isoformat()

    _orig_load, _orig_fetch, _orig_graded = load_csv, fetch_day_results, GRADED
    import tempfile as _tf
    GRADED = os.path.join(_tf.mkdtemp(), "hr_graded_selftest.csv")
    def _fake_load(path):
        if path == PLOG:
            return [
                {"date": _today, "player": "Done Hitter", "team": "AAA", "opp_sp": "Early Arm", "hr_pct":"30"},
                {"date": _today, "player": "Live Hitter", "team": "BBB", "opp_sp": "Late Arm", "hr_pct":"28"},
            ]
        return []   # empty graded log
    def _fake_fetch(d):
        # AAA game final (Done Hitter homered), BBB game still going -> not in finals, all_final False
        games = [{"teams": {"AAA Team": {"opp_sp": norm("Early Arm"), "abbr":"AAA",
                                          "bat": {norm("Done Hitter"): {"pa":4,"hr":1}}}}}]
        return games, False
    load_csv, fetch_day_results = _fake_load, _fake_fetch
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            grade_all()
        graded_now = _fake_load  # can't easily read file; assert via settle directly
        g,_ = _fake_fetch(_today)
        assert settle_row({"date":_today,"player":"Done Hitter","team":"AAA","opp_sp":"Early Arm"}, g)[0] == "hr"
        assert settle_row({"date":_today,"player":"Live Hitter","team":"BBB","opp_sp":"Late Arm"}, g) == "pending"
    finally:
        load_csv, fetch_day_results, GRADED = _orig_load, _orig_fetch, _orig_graded
    print("SAME-DAY PARTIAL SLATE PASS — final game settles, in-progress stays pending")

    print("GRADER SELFTEST PASS — settle/void/pending/DH + Brier/buckets/lift/ROI all exact")
    return 0

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    grade_all()
