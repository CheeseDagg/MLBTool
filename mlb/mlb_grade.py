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
GCOLS = ["date","player","team","game_id","opp_sp","slot","lu","hr_pct","hr_raw","hr_model",
         "fair","book_price","ev_pct","park","temp","plat","heat","outcome","hr_n"]

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
    # A ratio needs a baseline to be a ratio. With one recorded day the window IS
    # the season and rel comes back exactly 1.000 — a number that looks like a
    # clean reading and carries no information at all. Say nothing until the
    # denominator is real; the file backfills one day per Actions run.
    MIN_SEASON_DAYS = 10
    if len({r["date"] for r in rows}) < MIN_SEASON_DAYS:
        return {"building": True, "season_days": len({r["date"] for r in rows}),
                "need": MIN_SEASON_DAYS}
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

BACKFILL_PER_RUN = 5

def backfill_league(preds, limit=BACKFILL_PER_RUN, today=None):
    """Fill league rates for dates the grader has already settled.

    grade_all only fetches a date that still has UNGRADED rows, so once the ledger
    caught up, league_daily.csv would gain exactly one date per day and the control
    would not have a usable baseline until ~10 days out. This walks the prediction
    log's own dates, newest first, and fetches the few that are missing.

    Bounded at `limit` per run on purpose: three runs a day makes ~15 dates/day, so
    a season backfills in a couple of days without ever issuing a burst that looks
    like abuse to statsapi. Failures are skipped, not retried in-loop — the next
    run picks them up because they are still missing.
    """
    today = today or dt.date.today().isoformat()
    have = set()
    if os.path.exists(LEAGUE):
        with open(LEAGUE, newline="") as f:
            have = {r["date"] for r in csv.DictReader(f)}
    want = sorted({r["date"] for r in preds if r["date"] < today} - have, reverse=True)
    if not want: return 0
    done = 0
    for d in want[:limit]:
        try:
            games, fin = fetch_day_results(d)
        except Exception as e:
            print(f"  league backfill {d}: fetch failed ({type(e).__name__})"); continue
        if not games or not fin: continue
        record_league(d, games); done += 1
    if done:
        print(f"  league backfill: +{done} date(s), {len(want)-done} still missing")
    return done

def _gens():
    """Every schema the graded ledger has ever been written in, NEWEST FIRST.

    Each must have a DISTINCT width — that is the only thing that lets a row be
    identified without trusting the header line. Add a column here when you add one
    to GCOLS; the assert in _rows_by_width fires if two generations ever collide.
    """
    return [GCOLS,                                                          # 19: +hr_model (2026-08-04)
            [c for c in GCOLS if c != "hr_model"],                          # 18: +game_id  (2026-08-03)
            [c for c in GCOLS if c not in ("hr_model", "game_id")],         # 17: +hr_raw   (2026-07-23)
            [c for c in GCOLS if c not in ("hr_model", "game_id", "hr_raw")]]  # 16: original

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
    gens = _gens()
    by_w = {len(g): g for g in gens}
    assert len(by_w) == len(gens), "graded schema generations must have distinct widths"
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

def read_graded(path=None):
    """PUBLIC width-safe reader for hr_graded.csv. Use this, never csv.DictReader.

    Every other module that reads this ledger (mlb_recalibrate, mlb_livelevel) runs
    in its OWN workflow and never calls migrate_graded(), so it can meet the file in
    the window where the header is a generation behind the rows. csv.DictReader
    would then shift every field past the newest column and the module would read
    `outcome` out of `heat`, drop every row as un-graded, and refit — or decline to
    refit — on a silently truncated sample. _rows_by_width does not care what the
    header says.
    """
    p = path or GRADED
    if not os.path.exists(p): return []
    with open(p, newline="") as f:
        try: header = next(csv.reader(f))
        except StopIteration: return []
    # Width-mapping is ONLY valid for a file this module wrote. Applied to some other
    # CSV -- a test fixture, a hand-made export -- every row would land in the "unknown
    # width" bucket and be silently dropped, which is the exact failure this function
    # exists to prevent. So: bypass the header only when the header is itself a known
    # generation of OUR schema. Anything else is somebody else's file; trust its header.
    if header not in _gens():
        with open(p, newline="") as f: return list(csv.DictReader(f))
    rows, unknown = _rows_by_width(p)
    if unknown:
        print(f"  read_graded: {unknown} row(s) of unknown width skipped in {os.path.basename(p)}")
    return rows

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

def drop_tbd_shadows():
    """One-time repair: remove ledger rows that are a re-probe of a game already graded.

    WHAT THE BUG WAS. Before game_id, a settled row was keyed on the announced
    OPPOSING STARTER. That name changes during the day, so a game first boarded with
    a "TBD *" probable and later re-boarded with the real name graded TWICE — same
    player, same game, same outcome, a slightly different hr_pct — and both copies
    counted in calibration and in the reliability buckets.

    WHAT IT IS NOT. Most same-(date, player) pairs in this ledger are NOT that bug and
    must be left alone. Verified against game_starters.csv: 2026-07-11 PIT, 07-17
    TBR@BOS, 07-18 PIT@CLE, 07-19 LAD@NYY, 07-22 PIT@NYY, 07-28 CLE@CIN and 07-29
    ATL@NYM were all real doubleheaders, and the two Max Muncys on 07-22 (LAD and ATH)
    are two different men. Deleting those would destroy real, correctly-graded results
    — including JJ Bleday, who went `no` in game one on 07-28 and `hr` in game two.

    THE RULE. A team plays at most two games in a day, so a player cannot legitimately
    hold three rows on one date. Where he does, and one of them is the TBD placeholder
    while the others are named, the TBD row is the shadow. That is the entire test:
    exact, needs no schedule lookup, and cannot touch a two-row doubleheader. On the
    2026 ledger it matches exactly two rows (Junior Caminero 07-17, Ben Rice 07-19) out
    of 812 — 0.25%. The mechanism was real; the measured damage was small.

    A two-row (TBD, named) pair is deliberately NOT touched. Shohei Ohtani and Max
    Muncy on 07-19 each have one TBD and one named row against a real LAD@NYY
    doubleheader, so the TBD is most likely the second game whose probable was never
    announced — a legitimate row, not a duplicate. Guessing there could delete a real
    result to remove a maybe. game_id makes the question moot going forward.

    Idempotent: a second call finds nothing to drop and rewrites nothing.
    """
    if not os.path.exists(GRADED): return 0
    rows, _ = _rows_by_width(GRADED)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r.get("date",""), norm(r.get("player",""))), []).append(i)
    kill = set()
    for _, idx in groups.items():
        if len(idx) < 3: continue                      # <= 2 is a legal doubleheader
        tbd = [i for i in idx if (rows[i].get("opp_sp") or "").startswith("TBD")]
        if tbd and len(idx) - len(tbd) >= 2:           # keep the two named games
            kill.update(tbd)
    if not kill: return 0
    keep = [r for i, r in enumerate(rows) if i not in kill]
    tmp = GRADED + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GCOLS); w.writeheader()
        for r in keep: w.writerow({k: r.get(k, "") for k in GCOLS})
    os.replace(tmp, GRADED)
    for i in sorted(kill):
        print(f"  dropped re-probe shadow: {rows[i].get('date')} {rows[i].get('player')} "
              f"(opp_sp {rows[i].get('opp_sp')!r}, a 3rd row on a 2-game day)")
    return len(kill)

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

def _model_pct(r):
    """The MODEL's number for a graded row. Since 2026-08-04 priced rows publish a
    market-anchored hr_pct (mlb_hr MKT_W) and keep the model's own calibrated
    number in hr_model; before that (and on unpriced rows) hr_pct IS the model.
    Screens defined as statements about the MODEL (top-of-board, model+market
    agreement) must read this, or they quietly become rankings of the book's own
    prices instead of tests of the model."""
    v = str(r.get("hr_model", "") or "").strip()
    return float(v) if v else float(r["hr_pct"])

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
        # top-5 by the MODEL's number (hr_model on anchored rows) — sorting on the
        # published hr_pct would, post-anchor, just pick the book's shortest prices
        tops += sorted(rows_d, key=lambda r: -_model_pct(r))[:5]
    if tops:
        pl = 0.0
        for r in tops:
            d = _dec(r["book_price"])
            if d is None: continue
            pl += (d-1) if r["outcome"]=="hr" else -1.0
        panel["top_tier"] = {"n": len(tops),
                             "hits": sum(1 for r in tops if r["outcome"]=="hr"),
                             "roi": round(100*pl/len(tops),1)}
    # The claimed-edge curve: hit rate bucketed by the model's OWN claimed EV at
    # the logged price. This is the winner's-curse detector. If "edge" were
    # real, hit rate would hold (or rise) as claimed EV rises; what the first
    # 163 priced rows actually showed was monotone DECAY — 16.4% at negative EV,
    # 8.7% at 0-15, 8.0% at 15-30, 5.0% above 30. The bigger the model's
    # disagreement with the book, the more often the book was right. The board
    # renders this curve so nobody has to take that sentence on faith.
    pr = [r for r in live if r.get("ev_pct") not in ("", None)
          and r.get("book_price") not in ("", None)]
    curve = []
    for lo, hi, lab in [(-1e9, 0, "book likes it more"), (0, 15, "claimed edge 0-15%"),
                        (15, 30, "claimed edge 15-30%"), (30, 1e9, "claimed edge 30%+")]:
        ss = [r for r in pr if lo <= float(r["ev_pct"]) < hi]
        if not ss:
            continue
        h = sum(1 for r in ss if r["outcome"] == "hr")
        curve.append({"band": lab, "n": len(ss), "hits": h,
                      "hit_pct": round(100 * h / len(ss), 1)})
    if len(curve) >= 2:
        panel["ev_curve"] = curve
    # The agreement screen: model AND market both call him a top homer threat
    # (model >= 20% and price no longer than +300). This is the only priced
    # subset that has ever been above water here (7/22, +7.5% on the first
    # sample) — consistent with "short prices hit", NOT proof of an edge. It is
    # tracked so its sample can grow into a verdict instead of an anecdote.
    ag = [r for r in pr if _model_pct(r) >= 20 and (_dec(r["book_price"]) or 99) <= _dec(300)]
    if ag:
        pl = 0.0
        for r in ag:
            d = _dec(r["book_price"])
            if d is None: continue
            pl += (d - 1) if r["outcome"] == "hr" else -1.0
        panel["agree_tier"] = {"n": len(ag),
                               "hits": sum(1 for r in ag if r["outcome"] == "hr"),
                               "roi": round(100 * pl / len(ag), 1)}
    # ANCHOR GRADE (2026-08-04): priced rows publish a market-anchored hr_pct and
    # keep the model's own number in hr_model, so the blend is itself on the
    # ledger and gets graded here — Brier of what the board PUBLISHED vs Brier of
    # what the model would have said, same rows, head-to-head. This is the number
    # that decides whether MKT_W stays where it is (see mlb_hr --refit-anchor).
    anch = [r for r in live if str(r.get("hr_model", "") or "").strip() != ""]
    if anch:
        pa = [float(r["hr_pct"]) / 100 for r in anch]
        pm = [float(r["hr_model"]) / 100 for r in anch]
        ya = [1.0 if r["outcome"] == "hr" else 0.0 for r in anch]
        panel["anchor_tier"] = {"n": len(anch),
            "brier_published": round(sum((a-b)**2 for a,b in zip(pa,ya))/len(anch), 4),
            "brier_model":     round(sum((a-b)**2 for a,b in zip(pm,ya))/len(anch), 4)}
    return panel

# ---------------------------------------------------------------------------
def load_csv(path):
    if not os.path.exists(path): return []
    with open(path) as f: return list(csv.DictReader(f))

def _done_keys(graded):
    """Which (date, player, game) triples the ledger has already settled.

    TWO key spaces, deliberately. The ledger used to be deduped on
    (date, player, opp_sp) — the ANNOUNCED OPPOSING STARTER. opp_sp is in the key
    because it separates the two halves of a doubleheader, but it is a field that
    CHANGES during the day: a probable firms up from "TBD *" to a name, or is
    swapped. Each change minted a fresh key, so the same player in the same game
    was graded and appended a SECOND time — same outcome, slightly different
    hr_pct — and both copies then counted in calibration and in the reliability
    buckets. Ben Rice on 2026-07-19 is in there three times.

    game_id is MLB's gamePk: unique per game, distinct across the halves of a
    doubleheader, and it does not move. It is the key going forward.

    The opp_sp set is kept as a fallback for rows written before game_id existed,
    which have no gamePk to match on. A row is "done" if EITHER key hits, so the
    new key can only ever dedupe more than the old one, never less.
    """
    by_gid, by_sp = set(), set()
    for r in graded:
        k = (r.get("date",""), norm(r.get("player","")))
        gid = (r.get("game_id") or "").strip()
        if gid: by_gid.add(k + (gid,))
        by_sp.add(k + (r.get("opp_sp",""),))
    return by_gid, by_sp

def _is_done(r, by_gid, by_sp):
    k = (r.get("date",""), norm(r.get("player","")))
    gid = (r.get("game_id") or "").strip()
    if gid and k + (gid,) in by_gid: return True
    return k + (r.get("opp_sp",""),) in by_sp

def grade_all():
    preds = load_csv(PLOG)
    if not preds:
        print("no prediction log yet"); return
    # Migrate BEFORE reading the ledger. load_csv is a DictReader and trusts the
    # header line; on a file whose header is a generation behind, inserting a
    # column shifts every field past it, so opp_sp would come back holding the
    # game_id and every historical row would look un-graded and be re-appended.
    migrate_graded()
    drop_tbd_shadows()
    by_gid, by_sp = _done_keys(load_csv(GRADED))
    today = dt.date.today().isoformat()
    # settle every date through today. Prior days are fully final; today's games
    # that aren't Final yet come back 'pending' from settle_row and retry next run,
    # so early day-games settle on the 3:17 build instead of waiting overnight.
    dates = sorted({r["date"] for r in preds if r["date"] <= today})
    new = []
    for d in dates:
        rows = [r for r in preds if r["date"]==d and not _is_done(r, by_gid, by_sp)]
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
    backfill_league(preds)
    # migrate_graded() already ran at the top of this function, before the ledger
    # was read — it has to, or the dedup key is parsed off a stale header.
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
# HONEST CONFIDENCE — what a board number has historically been worth
# ---------------------------------------------------------------------------
# A published 23% is only usable if the reader knows what THIS model's 23%s have
# actually done. Same idea as the calibration panel, but per-row instead of buried
# in a tab: every board row carries the settled hit rate of its own bucket.
# Deliberately 5-point buckets (fine enough to be about that row, coarse enough to
# fill) and hard-suppressed under REL_MIN_N — a 9-row bucket moves 11 points on one
# homer, and a number that precise-looking is worse than no number at all.
REL_WIDTH = 5      # bucket width in probability points
REL_MIN_N = 30     # below this the bucket publishes actual=None -> board shows "—"
BACKTEST = os.path.join(DATA, "hr_backtest.csv")

def _rel_key(r):
    """Identity of a graded prediction, for de-duping the season replay against the
    live ledger (they overlap on the days the replay ran through)."""
    return (str(r.get("date", ""))[:10], norm(r.get("player", "")), norm(r.get("opp_sp", "")))

def _rel_tally(rows, width=REL_WIDTH, skip=None):
    """settled rows -> {bucket_lo: [n, hits, sum_predicted]}. Voids/pending excluded:
    a void is not a miss, and counting it as one would understate every bucket."""
    out = {}
    for r in rows:
        if r.get("outcome") not in ("hr", "no"):
            continue
        if skip and _rel_key(r) in skip:
            continue
        try:
            p = float(r.get("hr_pct"))
        except (TypeError, ValueError):
            continue
        if not (0.0 <= p <= 100.0):
            continue
        e = out.setdefault(int(p // width) * width, [0, 0, 0.0])
        e[0] += 1
        e[1] += 1 if r["outcome"] == "hr" else 0
        e[2] += p
    return out

def reliability(min_n=REL_MIN_N, width=REL_WIDTH):
    """Per-bucket historical hit rate for the board's own numbers -> JSON-safe dict.

    Two graded sources, kept SEPARATE on purpose rather than pooled:
      live   — hr_graded.csv, the board exactly as it published, settled next morning.
               This is the honest read of the deployed model, and it is what a bucket
               shows whenever it has the rows.
      season — hr_backtest.csv, the 25k-prediction walk-forward replay. Deeper, but a
               pre-recalibration scale, so it is only the fallback for buckets the live
               ledger cannot fill (the 30%+ tail, where the board is at its loudest and
               the live ledger is at its thinnest). Rows already in the live ledger are
               dropped from it so nothing is counted twice.
    Each bucket reports which source it came from; the UI says so on hover, because a
    number whose provenance is hidden is the thing this column exists to prevent.
    """
    live_rows = load_csv(GRADED)
    try:
        season_rows = load_csv(BACKTEST)
    except Exception:
        season_rows = []
    live = _rel_tally(live_rows, width)
    season = _rel_tally(season_rows, width, skip={_rel_key(r) for r in live_rows})

    def _stat(e):
        n, hits, s = e
        return {"n": n, "pred": round(s / n, 1), "actual": round(100.0 * hits / n, 1)}

    buckets = []
    for lo in sorted(set(live) | set(season)):
        b = {"lo": lo, "hi": lo + width}
        lv = _stat(live[lo]) if lo in live else {"n": 0}
        sn = _stat(season[lo]) if lo in season else {"n": 0}
        b["live"], b["season"] = lv, sn
        if lv["n"] >= min_n:
            src = ("live", lv)
        elif sn["n"] >= min_n:
            src = ("season", sn)
        else:
            src = (None, None)
        b["src"] = src[0]
        b["n"] = src[1]["n"] if src[1] else (lv["n"] + sn["n"])
        b["pred"] = src[1]["pred"] if src[1] else None
        b["actual"] = src[1]["actual"] if src[1] else None   # None -> the board prints "—"
        buckets.append(b)

    ld = sorted({str(r.get("date", ""))[:10] for r in live_rows if r.get("outcome") in ("hr", "no")})
    return {"width": width, "min_n": min_n, "buckets": buckets,
            "n_live": sum(v[0] for v in live.values()),
            "n_season": sum(v[0] for v in season.values()),
            "live_from": ld[0] if ld else None, "live_to": ld[-1] if ld else None}

def reliability_for_publish():
    """publish hook — never raises; an empty table just means the column shows "—"."""
    try:
        return reliability()
    except Exception as e:
        return {"width": REL_WIDTH, "min_n": REL_MIN_N, "buckets": [],
                "n_live": 0, "n_season": 0, "error": type(e).__name__}

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
    # MARKET ANCHOR (2026-08-04): a priced row publishes an anchored hr_pct and
    # carries the model's own number in hr_model. The panel must (a) grade the
    # published number against the model number head-to-head, and (b) keep the
    # model-defined screens reading hr_model — the top-of-board tier must rank G
    # (model 30) over H (model 18) even though H's PUBLISHED number is higher.
    a_rows = [
        row(player="G", hr_pct="21.4", hr_model="30", outcome="hr", ev_pct="26.0",
            book_price="280", game_id="1"),
        row(player="H", hr_pct="28.0", hr_model="18", outcome="no", ev_pct="-10.0",
            book_price="220", game_id="1"),
    ]
    pa = summarize(a_rows)
    at = pa["anchor_tier"]
    assert at["n"] == 2
    assert at["brier_published"] == round(((0.214-1)**2 + 0.28**2)/2, 4), at
    assert at["brier_model"]     == round(((0.30-1)**2 + 0.18**2)/2, 4), at
    # published (market) lost this pair on purpose: the grade must be able to say so
    assert at["brier_published"] > at["brier_model"], at
    assert _model_pct(a_rows[0]) == 30.0 and _model_pct(row(player="X", hr_pct="12")) == 12.0
    tops_pair = summarize(a_rows + [row(player="I", hr_pct="25", hr_model="10",
                                        outcome="no", book_price="150", game_id="1")])
    # top-5 window fits all 3 here, but agree_tier (model>=20, price<=+300) must
    # read the MODEL number: only G qualifies (model 30, +280) — H's published 28
    # with model 18 must NOT sneak in. 1 hit at +280 pays +2.8u on 1u -> ROI +280%
    agt = tops_pair["agree_tier"]
    assert agt["n"] == 1 and agt["hits"] == 1 and agt["roi"] == 280.0, agt
    # HEADER DRIFT: a ledger written under an OLD header, then appended to under a
    # NEW GCOLS, must survive. This is the 2026-07-23 hr_raw incident pinned: the
    # old migration keyed on "hr_n" in the header and so was blind to any later
    # column, DictReader shifted every post-drift row, and summarize() discarded
    # 217 rows without a word. Both halves must come back with real outcomes, and
    # a shifted row must be COUNTED as unparsed rather than silently dropped.
    import tempfile as _tf0
    _drift = os.path.join(_tf0.mkdtemp(), "hr_graded_drift.csv")
    # The header written here is the ORIGINAL 16-column generation. Beneath it sit
    # one 16-wide row (which it describes) and one 17-wide row from after hr_raw was
    # added (which it does not) — the exact on-disk state that lost a week of grading.
    _gen16 = [c for c in GCOLS if c not in ("hr_model", "game_id", "hr_raw")]
    assert len(_gen16) == 16, _gen16
    with open(_drift, "w", newline="") as f:
        w = csv.writer(f); w.writerow(_gen16)
        w.writerow(["2026-07-22","Old Guy","NYY","arm","3","card","20.0","+400","",
                    "","park +2%","80F","RvL +5%","heat +3%","hr","1"])
        w.writerow(["2026-07-23","New Guy","BOS","arm","4","card","22.0","19.1","+380",
                    "","","park -1%","78F","RvR -2%","heat +1%","no","0"])
    _pre, _ = _rows_by_width(_drift)
    assert [r["outcome"] for r in _pre] == ["hr","no"], \
        "width-keyed parse must recover both schema generations, got %r" % [r["outcome"] for r in _pre]
    assert _pre[1]["hr_raw"] == "19.1" and _pre[1]["hr_pct"] == "22.0", _pre[1]
    assert _pre[0].get("game_id","") == "" and _pre[1].get("game_id","") == "", \
        "neither legacy generation carries a gamePk; it must fill blank, not shift"
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
        # Two days is not a baseline: the control must REFUSE to report a ratio.
        assert league_context(["2026-07-25"]) == {"building": True, "season_days": 2,
                                                  "need": 10}
        for i in range(3, 13):                      # fill past MIN_SEASON_DAYS
            record_league("2026-08-%02d" % i,
                          [{"teams": {"A": {"bat": {"x": {"pa": 100, "hr": 3}}}}}])
        _c = league_context(["2026-07-25"])
        assert _c["days"] == 1 and _c["hr_pa"] == 0.0, _c
        _c2 = league_context(["2026-07-24"])
        assert _c2["rel"] > 1.0, ("a hot window must read rel>1 against the season, got %r" % _c2)
        assert league_context(["2099-01-01"]) is None   # no data -> no claim
        # BACKFILL: bounded, skips dates already recorded, never touches today
        # (today's slate is still live and its denominator would be partial).
        global fetch_day_results
        _of = fetch_day_results
        _calls = []
        def _fake(d):
            _calls.append(d)
            return [{"teams": {"A": {"bat": {"x": {"pa": 40, "hr": 1}}}}}], True
        try:
            fetch_day_results = _fake
            _p = [{"date": "2026-08-%02d" % i} for i in range(14, 26)]
            _p += [{"date": "2026-08-26"}]                    # == today below
            n1 = backfill_league(_p, limit=5, today="2026-08-26")
            assert n1 == 5 and len(_calls) == 5, (n1, _calls)
            assert "2026-08-26" not in _calls, "must not backfill today's live slate"
            assert _calls == sorted(_calls, reverse=True), \
                "newest-first, so the freshest window becomes usable first: %r" % _calls
            before = set(_calls)
            _calls.clear()
            backfill_league(_p, limit=5, today="2026-08-26")
            assert not (set(_calls) & before), \
                "a second run must not re-fetch dates already recorded: %r" % _calls
        finally:
            fetch_day_results = _of
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

    # ---- honest-confidence table -------------------------------------------
    # The whole point of this column is that a thin bucket must NOT print a number,
    # and that the season replay must never be double-counted against the live
    # ledger. Both are checked on synthetic ledgers, not on whatever today's data
    # happens to look like.
    global BACKTEST
    _og2, _ob = GRADED, BACKTEST
    _rd = _tf.mkdtemp()
    GRADED  = os.path.join(_rd, "g.csv")
    BACKTEST = os.path.join(_rd, "b.csv")
    try:
        def _w(path, rows):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["date","player","opp_sp","hr_pct","outcome"])
                w.writeheader()
                for r in rows: w.writerow(r)
        # live: 40 rows at 21% (10 hr = 25%), 4 rows at 31% (all hr) -> thin, must suppress
        lv  = [{"date":"2026-07-01","player":f"L{i}","opp_sp":"arm","hr_pct":"21.0",
                "outcome":("hr" if i < 10 else "no")} for i in range(40)]
        lv += [{"date":"2026-07-02","player":f"H{i}","opp_sp":"arm","hr_pct":"31.0",
                "outcome":"hr"} for i in range(4)]
        lv += [{"date":"2026-07-03","player":"V","opp_sp":"arm","hr_pct":"21.0","outcome":"void"}]
        # season: the SAME 40 live 21% rows re-listed (must be de-duped away) plus 50
        # fresh 31% rows at a 20% hit rate -> fills the bucket live can't
        bt  = [dict(r) for r in lv[:40]]
        bt += [{"date":"2026-06-01","player":f"S{i}","opp_sp":"arm","hr_pct":"31.0",
                "outcome":("hr" if i < 10 else "no")} for i in range(50)]
        _w(GRADED, lv); _w(BACKTEST, bt)
        rel = reliability()
        bk = {b["lo"]: b for b in rel["buckets"]}
        assert rel["n_live"] == 44, rel["n_live"]
        assert bk[20]["src"] == "live" and bk[20]["n"] == 40 and bk[20]["actual"] == 25.0, bk[20]
        assert bk[20]["season"]["n"] == 0, "season rows already in the live ledger were double-counted"
        assert bk[30]["src"] == "season" and bk[30]["n"] == 50 and bk[30]["actual"] == 20.0, bk[30]
        assert bk[30]["live"]["n"] == 4, bk[30]
        # a bucket no source can fill must publish nothing at all
        _w(BACKTEST, [])
        thin = {b["lo"]: b for b in reliability()["buckets"]}
        assert thin[30]["src"] is None and thin[30]["actual"] is None, thin[30]
        json.dumps(rel)
    finally:
        GRADED, BACKTEST = _og2, _ob
    print("HONEST-CONFIDENCE PASS — 5-pt buckets, live-first, season de-duped, <30 suppressed")

    # ---- game_id dedup key --------------------------------------------------
    # The old key was (date, player, opp_sp) — the ANNOUNCED opposing starter, a field
    # that changes during the day. These checks pin the two properties that matter:
    # a re-probe of one game must be recognised as already done, and the two halves of
    # a doubleheader must NOT be.
    by_gid, by_sp = _done_keys([
        {"date":"2026-07-19","player":"Ben Rice","game_id":"778001","opp_sp":"Yoshinobu Yamamoto"},
        {"date":"2026-07-19","player":"Ben Rice","game_id":"778002","opp_sp":"Will Klein"},
        {"date":"2026-07-01","player":"Old Row","game_id":"","opp_sp":"Some Arm"},   # pre-migration
    ])
    reprobe = {"date":"2026-07-19","player":"Ben Rice","game_id":"778002","opp_sp":"TBD *"}
    assert _is_done(reprobe, by_gid, by_sp), \
        "a game already graded must stay done when its probable is re-announced"
    dh_new = {"date":"2026-07-19","player":"Ben Rice","game_id":"778003","opp_sp":"Ryan Yarbrough"}
    assert not _is_done(dh_new, by_gid, by_sp), \
        "a THIRD distinct gamePk is a new game and must still be graded"
    assert _is_done({"date":"2026-07-19","player":"Ben Rice","game_id":"778001",
                     "opp_sp":"Yoshinobu Yamamoto"}, by_gid, by_sp)
    # legacy rows with no gamePk fall back to the old opp_sp key, unchanged
    assert _is_done({"date":"2026-07-01","player":"Old Row","game_id":"","opp_sp":"Some Arm"},
                    by_gid, by_sp), "pre-migration rows must still dedupe on opp_sp"
    assert not _is_done({"date":"2026-07-01","player":"Old Row","game_id":"","opp_sp":"Other Arm"},
                        by_gid, by_sp)
    # accents/suffixes normalised on BOTH sides of the key
    b2, s2 = _done_keys([{"date":"2026-07-01","player":"Julio Rodríguez Jr.","game_id":"9","opp_sp":"x"}])
    assert _is_done({"date":"2026-07-01","player":"Julio Rodriguez","game_id":"9","opp_sp":"y"}, b2, s2)
    print("DEDUP-KEY PASS — re-probe collapses on gamePk, DH halves stay distinct, legacy falls back")

    # ---- schema generations must stay unambiguous ---------------------------
    assert len({len(g) for g in _gens()}) == len(_gens()), \
        "graded schema generations collided on width — _rows_by_width cannot disambiguate"
    assert "game_id" in GCOLS and GCOLS[-2:] == ["outcome","hr_n"], GCOLS

    # ---- read_graded must not eat a file it did not write -------------------
    # Width-mapping is right for OUR ledger and catastrophic for anything else: a
    # narrow fixture would land every row in the unknown-width bucket and vanish,
    # which is precisely the silent-data-loss failure this reader exists to stop.
    _fk = os.path.join(_tf.mkdtemp(), "foreign.csv")
    with open(_fk, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date","player","hr_pct","outcome"]); w.writeheader()
        w.writerow({"date":"2026-07-01","player":"A","hr_pct":"20","outcome":"hr"})
        w.writerow({"date":"2026-07-01","player":"B","hr_pct":"18","outcome":"no"})
    _fr = read_graded(_fk)
    assert [r["outcome"] for r in _fr] == ["hr","no"], \
        "a foreign header must be trusted, not width-mapped into oblivion: %r" % _fr
    # and a file with OUR header but drifted row widths must still be width-mapped
    _dr = os.path.join(os.path.dirname(_fk), "ours.csv")
    with open(_dr, "w", newline="") as f:
        w = csv.writer(f); w.writerow([c for c in GCOLS if c not in ("game_id","hr_raw")])
        w.writerow(["2026-07-23","New Guy","BOS","arm","4","card","22.0","19.1","+380",
                    "","","park -1%","78F","RvR -2%","heat +1%","no","0"])
    assert [r["outcome"] for r in read_graded(_dr)] == ["no"], \
        "our own stale header must still be bypassed in favour of row width"

    # ---- the shadow-row repair ----------------------------------------------
    # It must remove a 3rd row on a 2-game day and must leave a real doubleheader,
    # a lone TBD row, and a (TBD, named) pair completely alone.
    _og3 = GRADED
    GRADED = os.path.join(_tf.mkdtemp(), "shadow.csv")
    try:
        def _wg(rows):
            with open(GRADED, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=GCOLS); w.writeheader()
                for r in rows: w.writerow({k: r.get(k,"") for k in GCOLS})
        base = lambda **k: dict({"date":"2026-07-17","outcome":"no","hr_pct":"20"}, **k)
        _wg([base(player="Junior Caminero", opp_sp="Jake Bennett"),
             base(player="Junior Caminero", opp_sp="TBD *"),
             base(player="Junior Caminero", opp_sp="Eduardo Rivera *"),
             base(player="JJ Bleday", opp_sp="Slade Cecconi"),          # real DH, keep both
             base(player="JJ Bleday", opp_sp="Gavin Williams", outcome="hr"),
             base(player="Ty France", opp_sp="TBD *"),                  # lone TBD, keep
             base(player="Shohei Ohtani", opp_sp="TBD *"),              # (TBD, named) pair,
             base(player="Shohei Ohtani", opp_sp="Cam Schlittler")])    # ambiguous -> keep both
        import io as _io, contextlib as _cl
        _b = _io.StringIO()
        with _cl.redirect_stdout(_b): n = drop_tbd_shadows()
        assert n == 1, f"expected exactly 1 shadow dropped, got {n}"
        left, _ = _rows_by_width(GRADED)
        assert len(left) == 7, len(left)
        cam = [r["opp_sp"] for r in left if r["player"] == "Junior Caminero"]
        assert cam == ["Jake Bennett","Eduardo Rivera *"], cam
        bl = sorted(r["outcome"] for r in left if r["player"] == "JJ Bleday")
        assert bl == ["hr","no"], f"a real doubleheader was collapsed: {bl}"
        assert sum(1 for r in left if r["player"] == "Ty France") == 1
        assert sum(1 for r in left if r["player"] == "Shohei Ohtani") == 2, \
            "an ambiguous (TBD, named) pair must not be guessed at"
        with _cl.redirect_stdout(_io.StringIO()):
            assert drop_tbd_shadows() == 0, "repair is not idempotent"
    finally:
        GRADED = _og3
    print("SHADOW-REPAIR PASS — 3rd-row re-probe dropped, doubleheaders and lone TBDs untouched")

    print("GRADER SELFTEST PASS — settle/void/pending/DH + Brier/buckets/lift/ROI all exact")
    return 0

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    grade_all()
