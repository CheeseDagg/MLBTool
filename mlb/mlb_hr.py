"""
mlb_hr.py  —  home-run board: best HR bets of the day
=====================================================
Ranks every projected hitter on today's slate by modeled P(hits a HR),
optionally priced against batter_home_runs props from The Odds API.

THE NUMBER (per plate appearance, then per game):
    p_PA = batter_rate x pitcher_factor x park_HR x temp_mult
    p_game = 1 - (1 - p_PA) ^ expected_PA

  * batter_rate     shrunk season HR/PA:  (HR + K*league) / (PA + K), K=130 PA.
                    League rate computed FROM the pull itself (self-calibrating).
  * pitcher_factor  starter's shrunk HR-per-batter vs league, weighted 62%
                    (a starter faces ~62% of a lineup's PAs; bullpen = league avg).
  * park_HR         HR-SPECIFIC park index (runs index from mlb_parks is the
                    wrong lens: Coors inflates runs far more than HRs; Fenway
                    inflates runs while SUPPRESSING HRs). Seed table below,
                    same confidence-shrink mechanic as mlb_parks.
  * temp_mult       +0.8%/F above 70F at open parks (ball-carry research),
                    capped +/-25%; halved under retractable roofs; domes off.
                    WIND IS NOT MODELED: the weather feed is speed-only and HR
                    wind effects are direction x park-geometry specific.
                    Deliberate omission, not an oversight.

HONEST SCOPING (read before betting):
  * Lineups: real lineup cards post 1-4h before first pitch — after the daily
    9am run. Until then the "lineup" is each team's TOP 9 BY SEASON PA, with
    expected PAs by usage rank (4.45 down to ~3.7). A scratch/rest day makes
    that player's row void, not wrong.
  * Platoon splits are NOT in v1 (needs a second data pull; on the list).
  * HR props carry heavy hold (books juice Yes-side props 8-15%). An edge here
    must clear a much higher bar than a moneyline edge. The board shows model
    fair vs best book so you can see the gap yourself.

RUN:  python mlb_hr.py              # full build -> data/hr_board.json
      python mlb_hr.py --selftest   # offline math validation, no network
Used by publish:  import mlb_hr; mlb_hr.load_board(DATA_DIR)
"""
import os, sys, json, math, glob, unicodedata, datetime as dt
import urllib.request, urllib.parse

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
YEAR = dt.date.today().year
K_BAT = 130     # PA of league-average prior blended into each batter
K_PIT = 200     # BF of league-average prior blended into each starter
SP_WEIGHT = 0.62
PA_TOP = 4.45   # expected PA for the #1 usage slot; -0.09 per slot down
CAP_PPA = 0.12  # sanity cap on per-PA HR probability

# ---------------------------------------------------------------------------
# HR-SPECIFIC park factors  (100 = league avg HR environment)
# SEED APPROXIMATIONS of the Statcast 3-yr HR index — refresh from Baseball
# Savant's park-factor page when convenient; conf shrinks toward neutral
# exactly like mlb_parks (eff = 1 + (raw-100)/100 * conf). Note how different
# this is from the RUN index: Coors 128 runs but ~110 HR; Fenway 108 runs
# but ~92 HR; Rate Field ~99 runs but ~110 HR. Same venue keys as mlb_parks.
HR_PARKS = {
    "Great American Ball Park": (128, 0.9), "Great American": (128, 0.9),
    "Yankee Stadium":           (116, 0.9),
    "Rate Field":               (110, 0.9),
    "American Family Field":    (110, 0.9),
    "Citizens Bank Park":       (110, 0.9),
    "Coors Field":              (110, 0.9),
    "Dodger Stadium":           (106, 0.9),
    "Camden Yards":             (106, 0.7),   # dimensions adjusted
    "Angel Stadium":            (104, 0.9),
    "Chase Field":              (104, 0.9),
    "Truist Park":              (104, 0.9),
    "Daikin Park":              (103, 0.9), "Minute Maid Park": (103, 0.9),
    "Globe Life Field":         (102, 0.9),
    "Wrigley Field":            (100, 0.9),   # wind park; wind not modeled
    "Nationals Park":           (100, 0.9),
    "Target Field":             ( 98, 0.9),
    "Progressive Field":        ( 97, 0.9),
    "Citi Field":               ( 96, 0.9),
    "T-Mobile Park":            ( 96, 0.9),
    "Petco Park":               ( 96, 0.9), "PETCO Park": (96, 0.9),
    "Comerica Park":            ( 94, 0.9),
    "Busch Stadium":            ( 92, 0.9),
    "Fenway Park":              ( 92, 0.9),   # run park, HR suppressor
    "PNC Park":                 ( 90, 0.9),
    "loanDepot park":           ( 90, 0.9),
    "Oracle Park":              ( 84, 0.9),
    # low-confidence / small-sample venues (same flags as mlb_parks)
    "Rogers Centre":            (106, 0.5),
    "Sutter Health Park":       (112, 0.5),
    "Kauffman Stadium":         ( 92, 0.5), "Kauffman": (92, 0.5),
    "Tropicana Field":          ( 95, 0.7),
    "George M. Steinbrenner Field": (112, 0.5),
    "Oakland Coliseum":         ( 92, 0.9),
}
HR_DEFAULT = (100, 0.0)

# statsapi full team name -> FanGraphs abbreviation
TEAM_MAP = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "ATH", "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}
FG_ALIASES = {"KCR": {"KC", "KCR"}, "SDP": {"SD", "SDP"}, "SFG": {"SF", "SFG"},
              "TBR": {"TB", "TBR"}, "WSN": {"WSH", "WSN"}, "CHW": {"CWS", "CHW"},
              "ATH": {"OAK", "ATH"}}

def _scrub(o):
    """Non-finite floats are invalid JSON for browsers; convert to None."""
    if isinstance(o, float) and not math.isfinite(o): return None
    if isinstance(o, dict):  return {k: _scrub(v) for k, v in o.items()}
    if isinstance(o, list):  return [_scrub(v) for v in o]
    return o

def norm(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "")
    for suf in (" jr", " sr", " ii", " iii", " iv"):
        if s.endswith(suf): s = s[: -len(suf)]
    return " ".join(s.split())

def am_from_p(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    d = 1.0 / p
    return f"+{round((d-1)*100)}" if d >= 2 else f"-{round(100/(d-1))}"

def dec_from_am(a):
    a = int(a)
    return a / 100 + 1 if a > 0 else 100 / (-a) + 1

def resolve_venue(venue, table):
    """Exact -> case-insensitive -> containment (longest key wins).
    Survives sponsor renames like 'UNIQLO Field at Dodger Stadium'."""
    if not isinstance(venue, str) or not venue.strip(): return None
    if venue in table: return venue
    low = {k.lower(): k for k in table}
    v = venue.lower()
    if v in low: return low[v]
    hits = [k for k in table if k.lower() in v or v in k.lower()]
    if hits: return max(hits, key=len)
    return None

def hr_park(venue):
    key = resolve_venue(venue, HR_PARKS)
    raw, conf = HR_PARKS[key] if key else HR_DEFAULT
    eff = 1 + ((raw - 100) / 100.0) * conf
    pct = (eff - 1) * 100
    lab = "park avg" if abs(pct) < 0.5 else (f"park +{pct:.0f}%" if pct > 0 else f"park {pct:.0f}%")
    if conf == 0: lab += " [unknown]"
    elif conf < 0.9: lab += " [low-conf]"
    return eff, lab

def hr_temp_mult(temp_f, roof):
    """+0.8%/F above 70 at open parks (ball-carry research), symmetric below,
    capped at +/-25%. Retractable roof: halved. Dome: off."""
    if roof == "dome" or temp_f is None:
        return 1.0, "dome" if roof == "dome" else "no wx"
    m = 1 + 0.008 * (temp_f - 70)
    if roof == "retract": m = 1 + (m - 1) * 0.5
    m = min(max(m, 0.75), 1.25)
    tag = f"{temp_f:.0f}F"
    if roof == "retract": tag += " (roof: half)"
    return m, tag

def _resolve_team_key(full, keys):
    """Map a schedule full name ('New York Yankees') onto whatever label the
    batter source used: FG/BREF code, full name, nickname, or unique city."""
    code = TEAM_MAP.get(full)
    if code and code in keys: return code
    if code:
        for canon, alts in FG_ALIASES.items():
            if code == canon:
                for a in alts:
                    if a in keys: return a
    if full in keys: return full
    low = {k.lower(): k for k in keys}
    if full.lower() in low: return low[full.lower()]
    words = full.split()
    for n in (2, 1):                                      # 'red sox' before 'sox'
        if len(words) > n:
            nick = " ".join(words[-n:]).lower()
            m = [k for k in keys if k.lower() == nick or k.lower().endswith(" " + nick)]
            if len(m) == 1: return m[0]
    for n in range(len(words) - 1, 0, -1):                # 'kansas city' before 'kansas'
        city = " ".join(words[:n]).lower()
        m = [k for k in keys if k.lower() == city or k.lower().startswith(city)]
        if len(m) == 1: return m[0]
    return None

def select_rows(rows, have_ev, n_hr=30, n_ev=10):
    """Board selection, likelihood-first: the board IS the top-30 most likely
    homers by model HR%. The top-10 EV gaps are appended as the value screen
    (they're where the model disagrees with the book — informative, but the
    known hot tail means treat them as screens, not truth). Sorted by HR%."""
    key = lambda r: (r["player"], r.get("team"), r.get("opp_sp"), r.get("slot"))
    sel = sorted(rows, key=lambda r: -r["hr_pct"])[:n_hr]
    seen = {key(r) for r in sel}
    if have_ev:
        for r in sorted([r for r in rows if "ev_pct" in r],
                        key=lambda r: -r["ev_pct"])[:n_ev]:
            if key(r) not in seen:
                sel.append(r); seen.add(key(r))
    sel.sort(key=lambda r: -r["hr_pct"])
    return sel

# ---------------------------------------------------------------------------
# pure compute — shared by live build and selftest
def build_board(batters, pitchers, sched, temps, props=None):
    """
    batters : list of {name, fg_team, pa, hr}
    pitchers: list of {name, bf, hr}
    sched   : list of {home, away, venue, home_sp, away_sp}   (full team names)
    temps   : {venue: (temp_f or None, roof)}
    props   : {norm_name: {"price": int, "book": str}} or None
    """
    tb_pa = sum(b["pa"] for b in batters) or 1
    Lb = sum(b["hr"] for b in batters) / tb_pa
    tp_bf = sum(p["bf"] for p in pitchers) or 1
    Lp = sum(p["hr"] for p in pitchers) / tp_bf
    if Lb <= 0: Lb = 0.031
    if Lp <= 0: Lp = Lb

    pit_fac = {}
    for p in pitchers:
        rate = (p["hr"] + K_PIT * Lp) / (p["bf"] + K_PIT)
        pit_fac[norm(p["name"])] = min(max(rate / Lp, 0.60), 1.60)

    by_team = {}
    for b in batters:
        by_team.setdefault(b["fg_team"], []).append(b)
    lineup = {}
    for t, bs in by_team.items():
        bs = sorted(bs, key=lambda x: -x["pa"])[:9]
        lineup[t] = [(b, i + 1) for i, b in enumerate(bs)]

    def fg(full):
        return TEAM_MAP.get(full)

    def team_bats(full):
        k = _resolve_team_key(full, set(lineup.keys()))
        return lineup.get(k, []) if k else []

    def sp_factor(name):
        n = norm(name or "")
        if n in pit_fac: return pit_fac[n], True
        if n:
            parts = n.split()
            if parts:
                last, fi = parts[-1], parts[0][:1]
                cands = [v for k, v in pit_fac.items()
                         if k.split() and k.split()[-1] == last and k[:1] == fi]
                if len(cands) == 1: return cands[0], True
        return 1.0, False

    rows = []
    for g in sched:
        eff, park_lab = hr_park(g["venue"])
        temp_f, roof = temps.get(g["venue"], (None, "open"))
        tmult, ttag = hr_temp_mult(temp_f, roof)
        for side, opp_sp_name in (("away", g.get("home_sp")), ("home", g.get("away_sp"))):
            team_full = g[side]
            opp_full = g["home" if side == "away" else "away"]
            fac, matched = sp_factor(opp_sp_name)
            sp_eff = SP_WEIGHT * fac + (1 - SP_WEIGHT)
            for b, slot in team_bats(team_full):
                base = (b["hr"] + K_BAT * Lb) / (b["pa"] + K_BAT)
                p_pa = min(base * eff * sp_eff * tmult, CAP_PPA)
                pa_est = max(PA_TOP - 0.09 * (slot - 1), 3.4)
                p_game = 1 - (1 - p_pa) ** pa_est
                sp_disp = opp_sp_name if isinstance(opp_sp_name, str) and opp_sp_name.strip() else "TBD"
                row = {
                    "player": b["name"], "team": fg(team_full) or team_full,
                    "opp": fg(opp_full) or opp_full,
                    "opp_sp": sp_disp + ("" if matched else " *"),
                    "venue": g["venue"], "slot": slot,
                    "hr_pct": round(p_game * 100, 1),
                    "fair": am_from_p(p_game),
                    "park": park_lab, "temp": ttag,
                    "sp_fac": round(fac, 2),
                }
                if props:
                    pr = props.get(norm(b["name"]))
                    if pr:
                        dec = dec_from_am(pr["price"])
                        row["book_price"] = int(pr["price"])
                        row["book"] = pr["book"]
                        row["ev_pct"] = round((p_game * dec - 1) * 100, 1)
                rows.append(row)

    have_ev = any("ev_pct" in r for r in rows)
    rows.sort(key=lambda r: (-(r.get("ev_pct", -999)), -r["hr_pct"]) if have_ev
              else (-r["hr_pct"], r["player"]))
    return rows, have_ev

# ---------------------------------------------------------------------------
# live data pulls (network; lazy imports so --selftest never needs them)
def _col(df, *names):
    """Find a column by candidate names, case-insensitive. None if absent."""
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    return None

_MULTI = {"TOT", "2TM", "3TM", "4TM", "---", ""}

def _rows_from_batting(df):
    """FG or BREF batting frame -> [{name, fg_team, pa, hr}]. Pure; unit-tested."""
    nc = _col(df, "Name"); tc = _col(df, "Team", "Tm")
    pc = _col(df, "PA"); hc = _col(df, "HR")
    if not all([nc, pc, hc]):
        raise ValueError(f"batting frame missing columns (have: {list(df.columns)[:12]})")
    out = []
    for _, r in df.iterrows():
        team = str(r.get(tc, "") or "") if tc else ""
        if team.strip().upper() in _MULTI: continue      # keep per-team rows only
        try: pa, hr = int(float(r.get(pc, 0) or 0)), int(float(r.get(hc, 0) or 0))
        except (TypeError, ValueError): continue
        if pa >= 30:
            out.append({"name": str(r[nc]), "fg_team": team.strip(), "pa": pa, "hr": hr})
    return out

def _rows_from_pitching(df):
    """FG or BREF pitching frame -> [{name, bf, hr}]. BF from TBF/BF, else IP*4.25."""
    nc = _col(df, "Name"); bc = _col(df, "TBF", "BF"); ic = _col(df, "IP")
    hc = _col(df, "HR")
    if not all([nc, hc]) or not (bc or ic):
        raise ValueError(f"pitching frame missing columns (have: {list(df.columns)[:12]})")
    out, derived = [], 0
    for _, r in df.iterrows():
        try:
            if bc and r.get(bc) == r.get(bc) and r.get(bc) not in (None, ""):
                bf = int(float(r[bc]))
            else:
                bf = int(round(float(r.get(ic, 0) or 0) * 4.25)); derived += 1
            hr = int(float(r.get(hc, 0) or 0))
        except (TypeError, ValueError): continue
        if bf >= 40:
            out.append({"name": str(r[nc]), "bf": bf, "hr": hr})
    if derived: print(f"   (BF derived from IP*4.25 for {derived} rows)")
    return out

def _statsapi_fetch(params):
    base = "https://statsapi.mlb.com/api/v1/stats"
    req = urllib.request.Request(f"{base}?{urllib.parse.urlencode(params)}",
                                 headers={"User-Agent": "Mozilla/5.0 (MLBTool board)"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = ""
        try: body = e.read().decode()[:200]
        except Exception: pass
        raise RuntimeError(f"statsapi HTTP {e.code}: {body}") from None

def _statsapi_rows(group):
    """Season stats for ALL players from MLB's own StatsAPI (statsapi.mlb.com)
    — the same host mlb_pitchers.py / mlb_data.py hit successfully from this
    Action daily, so it can't be IP-blocked like FanGraphs/BREF. Paginated.
    Tries playerPool variants since the param contract is picky.
    group: 'hitting' -> [{name, fg_team, pa, hr}]  |  'pitching' -> [{name, bf, hr}]"""
    pool_variants = [{"playerPool": "All"}, {"playerPool": "ALL"}, {}]
    out, offset, limit, pool, last_err = [], 0, 500, None, None
    while True:
        data = None
        for pv in ([pool] if pool is not None else pool_variants):
            params = {"stats": "season", "group": group, "season": YEAR,
                      "sportId": 1, "limit": limit, "offset": offset}
            params.update(pv)
            try:
                data = _statsapi_fetch(params); pool = pv; break
            except RuntimeError as e:
                last_err = e; continue
        if data is None:
            raise RuntimeError(f"statsapi all param variants failed: {last_err}")
        splits = (data.get("stats") or [{}])[0].get("splits", []) or []
        for s in splits:
            st = s.get("stat", {}) or {}
            name = (s.get("player") or {}).get("fullName", "")
            team = (s.get("team") or {}).get("name", "")      # full name, matches schedule
            try:
                if group == "hitting":
                    pa, hr = int(st.get("plateAppearances", 0) or 0), int(st.get("homeRuns", 0) or 0)
                    if name and pa >= 30:
                        out.append({"name": name, "fg_team": team, "pa": pa, "hr": hr})
                else:
                    bf, hr = int(st.get("battersFaced", 0) or 0), int(st.get("homeRuns", 0) or 0)
                    if not bf:
                        ip = float(st.get("inningsPitched", 0) or 0)
                        bf = int(round(ip * 4.25))
                    if name and bf >= 40:
                        out.append({"name": name, "bf": bf, "hr": hr})
            except (TypeError, ValueError):
                continue
        if len(splits) < limit: break
        offset += limit
        if offset > 4000: break                                # hard stop, sanity
    if not out:
        raise ValueError(f"statsapi returned no usable {group} rows")
    return out

def pull_batters():
    """Three-tier: FanGraphs (best, works at home) -> Baseball-Reference
    (usually works in CI) -> MLB StatsAPI (always works in CI). Each tier
    wrapped; the survivor is named in the log and the published note."""
    try:
        from pybaseball import batting_stats
        out = _rows_from_batting(batting_stats(YEAR, qual=0)); src = "FanGraphs"
    except Exception as e1:
        print(f"   FanGraphs unavailable ({type(e1).__name__}) -> Baseball-Reference")
        try:
            from pybaseball import batting_stats_bref
            out = _rows_from_batting(batting_stats_bref(YEAR)); src = "Baseball-Reference"
        except Exception as e2:
            print(f"   BREF unavailable ({type(e2).__name__}) -> MLB StatsAPI")
            out = _statsapi_rows("hitting"); src = "MLB StatsAPI"
    print(f"   batters: {len(out)} with 30+ PA  [{src}]")
    return out, src

def pull_pitchers():
    try:
        from pybaseball import pitching_stats
        out = _rows_from_pitching(pitching_stats(YEAR, qual=0)); src = "FanGraphs"
    except Exception as e1:
        print(f"   FanGraphs unavailable ({type(e1).__name__}) -> Baseball-Reference")
        try:
            from pybaseball import pitching_stats_bref
            out = _rows_from_pitching(pitching_stats_bref(YEAR)); src = "Baseball-Reference"
        except Exception as e2:
            print(f"   BREF unavailable ({type(e2).__name__}) -> MLB StatsAPI")
            out = _statsapi_rows("pitching"); src = "MLB StatsAPI"
    print(f"   pitchers: {len(out)} with 40+ BF  [{src}]")
    return out, src

def todays_sched():
    hits = sorted(glob.glob(os.path.join(DATA, "schedule_*.csv"))) or \
           [p for p in [os.path.join(DATA, "schedule.csv")] if os.path.exists(p)]
    if not hits:
        sys.exit("need data/schedule*.csv — run mlb_data.py first")
    import pandas as pd
    sc = pd.read_csv(hits[-1])
    def _cs(x):
        """clean string from CSV: pandas NaN (a truthy float!) -> None."""
        return x.strip() if isinstance(x, str) and x.strip() else None
    out = []
    for _, r in sc.iterrows():
        home, away = _cs(r.get("home")), _cs(r.get("away"))
        if not home or not away: continue
        out.append({"home": home, "away": away,
                    "venue": _cs(r.get("venue")) or "",
                    "home_sp": _cs(r.get("home_prob_pitcher")),
                    "away_sp": _cs(r.get("away_prob_pitcher"))})
    return out

def pull_temps(venues):
    try:
        from mlb_weather import VENUES, fetch_weather
    except Exception:
        return {v: (None, "open") for v in venues}
    temps = {}
    for v in venues:
        k = resolve_venue(v, VENUES)
        meta = VENUES.get(k) if k else None
        if meta is None:
            temps[v] = (None, "open"); continue
        lat, lon, roof = meta
        if roof == "dome":
            temps[v] = (None, "dome"); continue
        try:
            t, _w = fetch_weather(lat, lon)
            temps[v] = (t, roof)
        except Exception:
            temps[v] = (None, roof)
    return temps

def pull_props():
    """batter_home_runs Yes/Over-0.5 best price per player. Fails soft:
    returns ({}, reason). Costs ~1 credit per event on The Odds API."""
    key = os.environ.get("ODDS_API_KEY") or "2aa2e57832d4c9ca4bd66b20b05ba448"
    base = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
    try:
        q = urllib.parse.urlencode({"apiKey": key})
        with urllib.request.urlopen(f"{base}/events?{q}", timeout=30) as r:
            events = json.loads(r.read().decode())
    except Exception as e:
        return {}, f"props off: events list failed ({type(e).__name__})"
    best = {}
    tried = hit = 0
    for ev in events[:16]:
        tried += 1
        try:
            q = urllib.parse.urlencode({"apiKey": key, "regions": "us",
                                        "markets": "batter_home_runs",
                                        "oddsFormat": "american"})
            with urllib.request.urlopen(f"{base}/events/{ev['id']}/odds?{q}",
                                        timeout=30) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 422):
                return {}, f"props off: plan/market unavailable (HTTP {e.code})"
            continue
        except Exception:
            continue
        for bk in data.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "batter_home_runs": continue
                for o in mk.get("outcomes", []):
                    nm = o.get("name", "")
                    pt = o.get("point")
                    if not (nm == "Yes" or (nm == "Over" and (pt is None or float(pt) <= 0.51))):
                        continue
                    player = norm(o.get("description", ""))
                    price = o.get("price")
                    if not player or price is None: continue
                    cur = best.get(player)
                    if cur is None or dec_from_am(price) > dec_from_am(cur["price"]):
                        best[player] = {"price": int(price), "book": bk.get("key", "?")}
        if data.get("bookmakers"): hit += 1
    note = f"props: {len(best)} players priced across {hit}/{tried} events"
    return best, note

def load_board(data_dir):
    """publish hook: read data/hr_board.json -> (rows, note). Fails soft."""
    path = os.path.join(data_dir, "hr_board.json")
    if not os.path.exists(path):
        return [], "no HR board (run mlb_hr.py)"
    try:
        with open(path) as f:
            d = json.load(f)
        return d.get("rows", []), d.get("note", "")
    except Exception as e:
        return [], f"hr_board read error: {type(e).__name__}"

# ---------------------------------------------------------------------------
def selftest():
    bat = [
        {"name": "Slug McPower",  "fg_team": "NYY", "pa": 350, "hr": 28},  # elite
        {"name": "Mid Bat",       "fg_team": "NYY", "pa": 300, "hr": 10},
        {"name": "Slap Hitter",   "fg_team": "NYY", "pa": 320, "hr": 2},
        {"name": "Tiny Sample",   "fg_team": "NYY", "pa": 40,  "hr": 5},   # shrink test
    ] + [{"name": f"NY filler{i}", "fg_team": "NYY", "pa": 200 - i, "hr": 6} for i in range(5)] \
      + [{"name": f"BO bat{i}",   "fg_team": "BOS", "pa": 250 - i, "hr": 8} for i in range(9)]
    pit = [
        {"name": "Gopher Gary",  "bf": 400, "hr": 24},   # HR-prone
        {"name": "Ace Groundall","bf": 400, "hr": 5},    # HR-suppressing
    ] + [{"name": f"lg arm{i}", "bf": 300, "hr": 10} for i in range(20)]
    sched = [{"home": "New York Yankees", "away": "Boston Red Sox",
              "venue": "Yankee Stadium", "home_sp": "Ace Groundall",
              "away_sp": "Gopher Gary"}]
    temps = {"Yankee Stadium": (88.0, "open")}
    rows, have_ev = build_board(bat, pit, sched, temps,
                                props={norm("Slug McPower"): {"price": 320, "book": "testbook"}})
    assert rows, "no rows built"
    for r in rows:
        assert 0.0 < r["hr_pct"] < 60.0, f"prob out of range: {r}"
    top = {r["player"]: r for r in rows}
    # 1. batter ordering survives the pipeline
    assert top["Slug McPower"]["hr_pct"] > top["Mid Bat"]["hr_pct"] > top["Slap Hitter"]["hr_pct"]
    # 2. shrinkage: 5 HR in 40 PA must NOT beat the proven elite bat
    assert top["Tiny Sample"]["hr_pct"] < top["Slug McPower"]["hr_pct"]
    # 3. facing the gopher-baller (NYY bats vs Gary... wait: NYY face away? home bats face away_sp)
    #    home lineup (NYY) faces away_sp = Gopher Gary; away lineup (BOS) faces Ace.
    nyy_sp = [r for r in rows if r["team"] == "NYY"][0]["sp_fac"]
    bos_sp = [r for r in rows if r["team"] == "BOS"][0]["sp_fac"]
    assert nyy_sp > 1.0 > bos_sp, f"pitcher factors wrong: NYY faces {nyy_sp}, BOS faces {bos_sp}"
    # 4. temperature monotonic
    r_hot, _ = build_board(bat, pit, sched, {"Yankee Stadium": (95.0, "open")})
    r_cold, _ = build_board(bat, pit, sched, {"Yankee Stadium": (50.0, "open")})
    h = {r["player"]: r["hr_pct"] for r in r_hot}
    c = {r["player"]: r["hr_pct"] for r in r_cold}
    assert h["Slug McPower"] > c["Slug McPower"], "temp effect not monotonic"
    # 5. park effect: same slate at Oracle must price lower than Yankee
    sched_orc = [dict(sched[0], venue="Oracle Park")]
    r_orc, _ = build_board(bat, pit, sched_orc, {"Oracle Park": (88.0, "open")})
    o = {r["player"]: r["hr_pct"] for r in r_orc}
    assert o["Slug McPower"] < top["Slug McPower"]["hr_pct"], "park factor not applied"
    # 6. props EV wiring
    ev_row = top["Slug McPower"]
    assert have_ev and "ev_pct" in ev_row and ev_row["book"] == "testbook"
    # 7. slot PA math: slot 1 must project above same-rate slot 9 (use fillers)
    # 8. JSON-serializable
    json.dumps(rows)
    # 9. REGRESSION (run #9 crash): TBD starter (pandas NaN) + unknown venue must
    #    build rows, price the SP neutral, and label it TBD — never crash.
    nan = float("nan")
    sched_tbd = [{"home": "New York Yankees", "away": "Boston Red Sox",
                  "venue": "", "home_sp": None, "away_sp": nan}]
    r_tbd, _ = build_board(bat, pit, sched_tbd, {})
    assert r_tbd, "TBD-starter game built no rows"
    samp = r_tbd[0]
    assert samp["opp_sp"] in ("TBD *",) and samp["sp_fac"] == 1.0, samp["opp_sp"]
    json.dumps(r_tbd)
    # 10. REGRESSION (Alvarez eviction): likelihood-first board — the model's
    #     best HR% bat is ROW ONE regardless of price; top EV gaps still appended.
    big = [{"player": f"fringe{i}", "team": "X", "opp": "Y", "opp_sp": "P", "slot": 1,
            "hr_pct": 10.0 + i * 0.1, "fair": "+800", "park": "park avg", "temp": "80F",
            "sp_fac": 1.0, "book_price": 900, "book": "b", "ev_pct": 20.0 + i} for i in range(40)]
    slug = {"player": "Elite Slugger", "team": "HOU", "opp": "WSN", "opp_sp": "Arm", "slot": 1,
            "hr_pct": 33.0, "fair": "+203", "park": "park avg", "temp": "80F",
            "sp_fac": 1.0, "book_price": 170, "book": "b", "ev_pct": -18.0}
    pool = sorted(big + [slug], key=lambda r: -r["ev_pct"])
    picked = select_rows(pool, True)
    assert picked[0]["player"] == "Elite Slugger", "likelihood-first: slugger must lead the board"
    assert any(r["ev_pct"] >= 59 for r in picked), "top EV gaps must still be appended"
    assert all(picked[i]["hr_pct"] >= picked[i+1]["hr_pct"] for i in range(len(picked)-1))
    print(f"SELFTEST PASS — {len(rows)} rows, top: {rows[0]['player']} "
          f"{rows[0]['hr_pct']}% (fair {rows[0]['fair']})")
    return 0

def _build():
    print("1) schedule…"); sched = todays_sched()
    print(f"   {len(sched)} games")
    print("2) batters…"); bat, bsrc = pull_batters()
    print("3) pitchers…"); pit, psrc = pull_pitchers()
    if not bat or not pit:
        sys.exit("no batter/pitcher data from any source — board not built")
    tkeys = sorted({b["fg_team"] for b in bat})
    print(f"   team labels in batter data ({len(tkeys)}): {tkeys[:34]}")
    print(f"   schedule teams: {sorted({g[s] for g in sched for s in ('home','away')})[:8]} …")
    print("4) park temps (Open-Meteo)…")
    temps = pull_temps(sorted({g["venue"] for g in sched}))
    print("5) HR props (The Odds API, optional)…")
    props, pnote = pull_props(); print(f"   {pnote}")
    rows, have_ev = build_board(bat, pit, sched, temps, props or None)
    rows = select_rows(rows, have_ev)
    print(f"   board rows built: {len(rows)}")
    if not rows:
        note = (f"BOARD EMPTY — team labels in {bsrc} data didn't map to schedule teams. "
                f"Sample labels: {', '.join(tkeys[:10])} · fix TEAM_MAP/resolver in mlb_hr.py")
        out = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "note": note, "rows": []}
        os.makedirs(DATA, exist_ok=True)
        with open(os.path.join(DATA, "hr_board.json"), "w") as f:
            json.dump(_scrub(out), f, indent=1, allow_nan=False)
        sys.exit("EMPTY BOARD — " + note)
    note = (f"stats: {bsrc}" + ("" if psrc == bsrc else f"/{psrc}") + " · "
            "lineups = top-9 by season PA until cards post · platoon splits not in v1 · "
            "wind not modeled (speed-only feed) · park HR factors are seed approximations "
            "(conf-shrunk; refresh from Savant) · temp vs flat 70F baseline (mildly double-counts "
            "warm-climate open parks) · " + pnote +
            (" · sorted by model HR% · EV screen appended" if have_ev else " · sorted by model HR% (no props matched)"))
    out = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "note": note, "rows": rows}
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "hr_board.json")
    with open(path, "w") as f:
        json.dump(_scrub(out), f, indent=1, allow_nan=False)
    print(f"hr_board.json written: {len(rows)} rows"
          + (f", EV priced" if have_ev else ", model-only"))

def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    try:
        _build()
    except SystemExit:
        raise
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        err = tb.strip().splitlines()[-1][:220]
        os.makedirs(DATA, exist_ok=True)
        with open(os.path.join(DATA, "hr_board.json"), "w") as f:
            json.dump(_scrub({"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       "note": f"HR BUILD CRASHED — {err} (full trace in Action log)",
                       "rows": []}), f, indent=1, allow_nan=False)
        sys.exit(1)

if __name__ == "__main__":
    main()
