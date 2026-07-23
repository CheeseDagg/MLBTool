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
try:
    import mlb_marcel as _mc
except Exception:
    _mc = None
K_BAT = 130     # PA of league-average prior blended into each batter
K_PIT = 200     # BF of league-average prior blended into each starter
SP_WEIGHT = 0.62

# ---------------------------------------------------------------------------
# PLATOON — league-average HR-rate factors by (batter side, starter throws).
# Per-batter splits at seasonal PA are mostly noise; the sound v2 is heavy
# regression to the league effect: LHB-vs-LHP is the big penalty, the rest
# mild. Approximate, conservative, labeled — refine when calibration says to.
# Applied to the STARTER'S share only (SP_WEIGHT); bullpen hands unknowable.
# Switch hitters take the favorable side. Unknown handedness -> neutral.
PLATOON = {"LL": 0.78, "LR": 1.08, "RL": 1.12, "RR": 0.97}

def platoon_factor(bat_side, pitch_hand):
    """Effective per-PA multiplier vs a starter of known hand. Returns
    (multiplier, tag) — tag empty when neutral/unknown."""
    if not bat_side or not pitch_hand or pitch_hand not in ("L", "R"):
        return 1.0, ""
    b = bat_side
    if b == "S":                       # switch: take the platoon-favorable side
        b = "R" if pitch_hand == "L" else "L"
    raw = PLATOON.get(b + pitch_hand)
    if raw is None:
        return 1.0, ""
    eff = SP_WEIGHT * raw + (1 - SP_WEIGHT)          # starter share only
    pct = (raw - 1) * 100
    tag = f"{bat_side}v{pitch_hand} {'+' if pct >= 0 else ''}{pct:.0f}%"
    return eff, tag


def hands_get(hands, name, want="bat"):
    """Duplicate-name safe lookup. want='bat' prefers position players,
    want='pit' prefers pitchers. -> (batSide, pitchHand, mlbam_id) or (None,)*3."""
    cands = (hands or {}).get(norm(name)) or []
    if isinstance(cands, tuple): cands = [cands]     # legacy single-entry shape
    if not cands: return (None, None, None)
    if len(cands) > 1:
        pit = [c for c in cands if (c[3] if len(c) > 3 else "") == "P"]
        bat = [c for c in cands if (c[3] if len(c) > 3 else "") != "P"]
        pick = (pit or cands) if want == "pit" else (bat or cands)
        c = pick[0]
    else:
        c = cands[0]
    return (c[0], c[1], c[2] if len(c) > 2 else None)


# ---------------------------------------------------------------------------
# BULLPEN — the (1-SP_WEIGHT) share of PAs was priced league-flat. Team relief
# HR-allowed rates (gs==0 arms, MLB StatsAPI), conf-shrunk. Fail-soft neutral.
K_PEN = 350

BARREL_W = 0.35
SEASON_W  = 0.6   # weight on the bat's CURRENT-SEASON rate vs the model's talent-prior projection.
                  # Walk-forward validated on the 25,128-prediction backtest: w=0.6 is the Brier
                  # minimum on train AND unseen test independently (base .108573 -> .108249);
                  # decliners (Ohtani-shape) were overrated -2.9, surgers underrated +2.1 (>2SE).
SEASON_K  = 250   # PA of league rate the season term is shrunk with (insensitive 170-380 in test)
SEASON_MIN_PA = 60  # below this the season sample is noise - blend inactive

def pull_barrels():
    """Savant barrels-per-PA leaderboard -> {mlbam_id: brl_pa_pct}. One request,
    same host pattern as the xwOBA pull already running in this Action."""
    try:
        from pybaseball import statcast_batter_exitvelo_barrels
        df = statcast_batter_exitvelo_barrels(YEAR, minBBE=25)
        col = "brl_pa" if "brl_pa" in df.columns else ("brl_percent" if "brl_percent" in df.columns else None)
        idc = "player_id" if "player_id" in df.columns else None
        if col is None or idc is None:
            return {}, "barrel off (columns changed)"
        out = {}
        for _, r in df.iterrows():
            try: out[int(r[idc])] = float(r[col])
            except (TypeError, ValueError): continue
        return out, f"barrel blend {int(BARREL_W*100)}% ({len(out)} bats)"
    except Exception as e:
        return {}, f"barrel off ({type(e).__name__})"

def pull_bullpens():
    try:
        rows = _statsapi_rows("pitching")
    except Exception as e:
        return {}, f"bullpen off ({type(e).__name__})"
    rp = [r for r in rows if r.get("gs", 1) == 0 and r.get("team")]
    if not rp:
        return {}, "bullpen off (no relief rows)"
    L_hr = sum(r["hr"] for r in rp); L_bf = sum(r["bf"] for r in rp) or 1
    L = L_hr / L_bf
    agg = {}
    for r in rp:
        a = agg.setdefault(r["team"], [0, 0]); a[0] += r["hr"]; a[1] += r["bf"]
    out = {}
    for team, (hr, bf) in agg.items():
        rate = (hr + K_PEN * L) / (bf + K_PEN)
        out[team] = round(rate / L, 3)
    return out, f"bullpen on ({len(out)} pens)"

# MLB StatsAPI team ids -> our codes. Used as a fallback when currentTeam
# carries an id but no name. Ids are stable across seasons.
TEAM_ID_MAP = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KCR", 119: "LAD",
    120: "WSN", 121: "NYM", 133: "ATH", 134: "PIT", 135: "SDP", 136: "SEA",
    137: "SFG", 138: "STL", 139: "TBR", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CHW", 146: "MIA", 147: "NYY", 158: "MIL",
}
# norm(name) -> current team code, from the StatsAPI roster. Authoritative for
# WHICH CLUB a player is on today. Season stats tables carry the team a player
# ACCUMULATED STATS for, which is a different question and goes stale the moment
# anyone changes clubs — that is how Juan Soto sat on the Yankees board in 2026.
ROSTER_TEAM = {}
# norm(name) -> team code for players on a 26-man ACTIVE roster, i.e. who can
# actually play today. Season stats cannot know about the IL: a hurt star keeps
# his big PA total and gets projected straight into the lineup for weeks after he
# stops playing. Judge (IL since Jun 5) and Buxton (IL since Jul 6) both did.
ACTIVE_ROSTER = {}
ROSTER_OK = set()          # team codes whose roster we actually fetched

def pull_active_rosters():
    """26-man active roster per club. Positive list: on it = available today.
    Fail-soft per team — a club we cannot fetch is simply never filtered."""
    ACTIVE_ROSTER.clear(); ROSTER_OK.clear()
    for tid, code in TEAM_ID_MAP.items():
        try:
            req = urllib.request.Request(
                f"https://statsapi.mlb.com/api/v1/teams/{tid}/roster?rosterType=active&season={YEAR}",
                headers={"User-Agent": "Mozilla/5.0 (MLBTool board)"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode())
            got = 0
            for e in (data.get("roster") or []):
                nm = norm(((e.get("person") or {}).get("fullName")) or "")
                if nm:
                    ACTIVE_ROSTER[nm] = code; got += 1
            if got:
                ROSTER_OK.add(code)
        except Exception:
            continue
    return len(ROSTER_OK)

def pull_handedness():
    """batSide / pitchHand AND current team for every player from MLB StatsAPI's
    bulk roster endpoint (one call, unblockable host). Fail-soft: {} -> platoon off."""
    try:
        req = urllib.request.Request(
            f"https://statsapi.mlb.com/api/v1/sports/1/players?season={YEAR}",
            headers={"User-Agent": "Mozilla/5.0 (MLBTool board)"})
        with urllib.request.urlopen(req, timeout=45) as r:
            data = json.loads(r.read().decode())
        out = {}
        ROSTER_TEAM.clear()
        for p in data.get("people", []):
            n = norm(p.get("fullName", ""))
            if n:
                pos = ((p.get("primaryPosition") or {}).get("abbreviation") or "")
                out.setdefault(n, []).append(((p.get("batSide") or {}).get("code"),
                          (p.get("pitchHand") or {}).get("code"),
                          p.get("id"), pos))
                ct = p.get("currentTeam") or {}
                code = TEAM_MAP.get(ct.get("name") or "") or TEAM_ID_MAP.get(ct.get("id"))
                if code:
                    ROSTER_TEAM[n] = code
        return out, f"platoon on ({sum(len(v) for v in out.values())} players)"
    except Exception as e:
        return {}, f"platoon off (handedness fetch failed: {type(e).__name__})"
PA_TOP = 4.45   # expected PA for the #1 usage slot; -0.09 per slot down
CAP_PPA = 0.12  # sanity cap on per-PA HR probability

# --- Season-backtest-derived corrections (25,074 predictions, 105 days) ---
# The raw model runs progressively HOT above ~16%: 25%+ bucket said 28.3% but only
# 22.2% happened (+6.1pts). Calibrate on the GAME probability with a piecewise
# shrink that leaves the well-calibrated low end (0-12%: +0.4) untouched and pulls
# the high end down to observed. Anchors are (predicted%, observed%) bucket midpoints.
CALIB_ANCHORS = [(0.0, 0.0), (8.4, 8.8), (13.7, 12.3), (17.7, 16.0),
                 (22.0, 20.1), (28.3, 22.2), (40.0, 30.5)]   # last extrapolates the slope
def calibrate_pct(p_pct):
    """Map a raw game-HR% to the backtest-calibrated %. Piecewise-linear, monotone."""
    a = CALIB_ANCHORS
    if p_pct <= a[0][0]: return p_pct
    for (x0, y0), (x1, y1) in zip(a, a[1:]):
        if p_pct <= x1:
            t = (p_pct - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    # above the top anchor: continue the final slope
    (x0, y0), (x1, y1) = a[-2], a[-1]
    slope = (y1 - y0) / (x1 - x0) if x1 > x0 else 1.0
    return y1 + slope * (p_pct - x1)

# ---- "spot" READABILITY layer (NOT a probability change) -------------------
# The board is calibrated by park (verified on the 25k-prediction backtest: high-HR
# parks project 12.0% -> 11.7% actual, gap -0.3, inside 2SE). So a bat like a Coors
# regular CORRECTLY sits near the top every home game -- which makes the board hard
# to READ day to day: is today special for him, or just his floor? This tags each
# bat by today's HR% vs its OWN recent median so standout days (soft SP, wind out,
# good platoon) separate from the perennial floor. Display only; hr_pct/fair untouched.
SPOT_K, SPOT_MIN = 8, 3                    # look-back games; min history to fire
SPOT_STANDOUT, SPOT_FLOOR = 3.0, -2.0      # pts above/below own median

def _spot_median(xs):
    s = sorted(xs); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0

def _spot_history(plog_path, today):
    """{player: [most-recent-first HR% ...][:SPOT_K]} from the prediction log,
    excluding today. Pure stdlib; fails soft to {}."""
    import csv
    hist = {}
    if not os.path.exists(plog_path):
        return hist
    try:
        with open(plog_path) as f:
            for r in csv.DictReader(f):
                if r.get("date") == today:
                    continue
                try:
                    v = float(r["hr_pct"])
                except (KeyError, ValueError, TypeError):
                    continue
                hist.setdefault(r["player"], []).append((r.get("date", ""), v))
    except Exception:
        return {}
    return {pl: [v for _, v in sorted(x, reverse=True)][:SPOT_K] for pl, x in hist.items()}

def annotate_spots(rows, plog_path, today):
    """Add base_self / edge_self / spot to each row. edge_self = today - own median.
    spot in {STANDOUT, floor, "", new}. Never touches hr_pct."""
    hist = _spot_history(plog_path, today)
    for r in rows:
        vals = hist.get(r["player"], [])
        if len(vals) < SPOT_MIN:
            r["base_self"], r["edge_self"], r["spot"] = None, None, "new"
            continue
        base = _spot_median(vals)
        d = round(r["hr_pct"] - base, 1)
        r["base_self"] = round(base, 1)
        r["edge_self"] = d
        r["spot"] = "STANDOUT" if d >= SPOT_STANDOUT else ("floor" if d <= SPOT_FLOOR else "")
    return rows

def print_spotlight(rows, n=8):
    """Console leaderboard of today's biggest movers vs their own norm."""
    scored = sorted((r for r in rows if r.get("edge_self") is not None),
                    key=lambda r: -r["edge_self"])
    hits = [r for r in scored[:n] if r["edge_self"] >= SPOT_STANDOUT]
    if not hits:
        print("   spotlight: no bats meaningfully above their own norm today")
        return
    print("   SPOTLIGHT - elevated vs own recent norm (readability, not a price):")
    for r in hits:
        print(f"     +{r['edge_self']:.1f} over own {r['base_self']:.0f}%   "
              f"{r['player']:<22} {r['hr_pct']:.1f}%   ({r.get('opp_sp','')}, {r.get('park','')})")

# Heat DEMOTED. Over 7,824 heat>=+10 predictions the factor UNDERperformed its own
# projection (said 16.1%, happened 14.2%) and barely beat negative-heat (9.3%). It was
# adding to the hot bias, not signal. Shrink its deviation-from-1 toward neutral until a
# rebuilt (zone/window) version proves out. 0.0 = fully neutral; 1.0 = original weight.
HEAT_WEIGHT = 0.25

# Observed two-homer rate among all graded predictions (season): the base rate combine
# markets should be priced against.
TWO_HR_BASE_RATE = 0.007

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

# ---------------------------------------------------------------------------
# WIND — direction now modeled. PARK_ORIENT = approximate compass bearing from
# home plate to CENTER FIELD (deg). SEED APPROXIMATIONS, conf-weighted like the
# HR parks; verify any venue against satellite when convenient. Wind FROM
# (bearing+180) blows straight OUT. Effect: +0.9% HR per mph of out-component,
# capped ±18%, halved under a retractable roof, off in domes.
PARK_ORIENT = {
    "Yankee Stadium": (75, 0.9), "Fenway Park": (52, 0.9), "Wrigley Field": (45, 0.9),
    "Oriole Park at Camden Yards": (31, 0.8), "Camden Yards": (31, 0.8),
    "Great American Ball Park": (120, 0.8), "Great American": (120, 0.8),
    "Citizens Bank Park": (10, 0.8), "Citi Field": (30, 0.8), "Nationals Park": (28, 0.8),
    "PNC Park": (115, 0.8), "Truist Park": (145, 0.7), "Busch Stadium": (62, 0.8),
    "Target Field": (90, 0.7), "Progressive Field": (0, 0.7), "Comerica Park": (145, 0.7),
    "Kauffman Stadium": (45, 0.8), "Kauffman": (45, 0.8), "Rate Field": (127, 0.7),
    "Angel Stadium": (65, 0.7), "Dodger Stadium": (25, 0.8), "Oracle Park": (85, 0.9),
    "Petco Park": (0, 0.7), "Coors Field": (10, 0.8), "T-Mobile Park": (48, 0.7),
    "Oakland Coliseum": (55, 0.7), "Sutter Health Park": (35, 0.5),
    "George M. Steinbrenner Field": (70, 0.5), "loanDepot park": (78, 0.6),
    "Daikin Park": (345, 0.6), "Minute Maid Park": (345, 0.6),
    "Chase Field": (0, 0.6), "Globe Life Field": (95, 0.6), "American Family Field": (128, 0.7),
    "Rogers Centre": (348, 0.6), "Tropicana Field": (45, 0.3),
}
WIND_PER_MPH = 0.009
WIND_CAP = 0.18

def hr_weather_mult(temp_f, roof, wind_mph=None, wind_dir=None, venue=None):
    """Temp (+0.8%/F over 70) x wind out/in component. Returns (mult, tag).
    Dome: off. Retractable: both effects halved. Unknown orientation: temp only."""
    if roof == "dome":
        return 1.0, "dome"
    if temp_f is None:
        return 1.0, "no wx"
    tmult = 1 + 0.008 * (temp_f - 70)
    wmult, wtag = 1.0, ""
    ok = resolve_venue(venue, PARK_ORIENT) if venue else None
    if ok is not None and wind_mph and wind_dir is not None:
        bearing, conf = PARK_ORIENT[ok]
        out_from = (bearing + 180) % 360
        delta = math.radians((wind_dir - out_from + 180) % 360 - 180)
        comp = wind_mph * math.cos(delta) * conf          # +out / -in, conf-shrunk
        wmult = 1 + max(min(comp * WIND_PER_MPH, WIND_CAP), -WIND_CAP)
        if abs(comp) >= 3:
            wtag = f" · wind {'out' if comp > 0 else 'in'} {abs(comp):.0f}"
    m = tmult * wmult
    if roof == "retract":
        m = 1 + (m - 1) * 0.5
    m = min(max(m, 0.70), 1.35)
    tag = f"{temp_f:.0f}F{wtag}"
    if roof == "retract": tag += " (roof: half)"
    return m, tag

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
# HEAT MAPS — Savant zone overlap: does this starter locate where this batter
# does HR damage? Gameday zones (1-9 in-zone, 11-14 chase). Batter HR-rate per
# pitch by zone (shrunk to his own overall, K_ZONE pitches); starter's pitch
# mix by zone, split by batter side. Overlap = Sigma w_z * rate_z / overall,
# clipped conservatively, applied to the STARTER SHARE like platoon. Fully
# fail-soft; per-row pill; HR_HEAT=0 env disables without redeploy.
K_ZONE = 150
HEAT_CLIP = (0.85, 1.18)
ZONES = [1,2,3,4,5,6,7,8,9,11,12,13,14]

def batter_zone_profile(df):
    """statcast batter frame -> {'z':{zone: shrunk HR-per-pitch}, 'ov': overall, 'n': pitches}"""
    if df is None or not len(df) or "zone" not in df.columns: return None
    d = df[df["zone"].notna()]
    n = len(d)
    if n < 400: return None                       # too thin to trust zones
    hr = (d["events"] == "home_run") if "events" in d.columns else None
    if hr is None: return None
    ov = float(hr.sum()) / n
    prof, share = {}, {}
    for z in ZONES:
        m = d["zone"] == z
        nz = int(m.sum()); hz = int((hr & m).sum())
        prof[int(z)] = (hz + K_ZONE * ov) / (nz + K_ZONE)
        share[int(z)] = nz / n
    return {"z": prof, "s": share, "ov": ov if ov > 0 else 1e-5, "n": n, "ver": 2}

def pitcher_zone_mix(df, stand):
    """statcast pitcher frame -> zone weights vs batters of `stand` ('L'/'R')."""
    if df is None or not len(df) or "zone" not in df.columns: return None
    d = df[df["zone"].notna()]
    if "stand" in d.columns and stand in ("L", "R"):
        ds = d[d["stand"] == stand]
        if len(ds) >= 250: d = ds                 # side-split only when thick enough
    n = len(d)
    if n < 300: return None
    w = {}
    for z in ZONES:
        w[int(z)] = float((d["zone"] == z).sum()) / n
    return {"w": w, "n": n}

def heat_factor(bat_prof, pit_mix):
    """Overlap multiplier ~1.0, clipped. None if either side missing.
    Key-agnostic (JSON cache stringifies zone keys). SELF-MIX CENTERED:
    denominator is the batter's shrunk profile under HIS OWN pitch diet, so
    the shrinkage bias that flattened sluggers' hot zones cancels — a pitcher
    throwing the batter's average diet scores exactly 1.000. Falls back to the
    raw-overall denominator for pre-v2 cached profiles."""
    if not bat_prof or not pit_mix: return None
    bz, pw = bat_prof["z"], pit_mix["w"]
    gb = lambda z: bz.get(z, bz.get(str(z), bat_prof["ov"]))
    gw = lambda z: pw.get(z, pw.get(str(z), 0.0))
    num = sum(gw(z) * gb(z) for z in ZONES)
    cov = sum(gw(z) for z in ZONES)
    if cov <= 0: return None
    bs = bat_prof.get("s")
    if bs:
        gs = lambda z: bs.get(z, bs.get(str(z), 0.0))
        den_num = sum(gs(z) * gb(z) for z in ZONES)
        den_cov = sum(gs(z) for z in ZONES)
        base = (den_num / den_cov) if den_cov > 0 else bat_prof["ov"]
    else:
        base = bat_prof["ov"]                      # pre-v2 cache: documented bias
    raw = (num / cov) / base
    return min(max(raw, HEAT_CLIP[0]), HEAT_CLIP[1])

def _zone_cache_path(): return os.path.join(DATA, "zone_cache.json")

def load_zone_cache():
    try:
        with open(_zone_cache_path()) as f: return json.load(f)
    except Exception: return {}

def save_zone_cache(c):
    try:
        os.makedirs(DATA, exist_ok=True)
        with open(_zone_cache_path(), "w") as f: json.dump(_scrub(c), f)
    except Exception: pass

def fetch_zone_profiles(bat_ids, pit_ids, hands):
    """Pull Savant pitch data for exactly the needed players (cached aggregates,
    ~5-day freshness). Returns (bat_profiles{id}, pit_mixes{(id,stand)}, note)."""
    if os.environ.get("HR_HEAT", "1") == "0":
        return {}, {}, "heat off (HR_HEAT=0)"
    today = dt.date.today().isoformat()
    start = f"{YEAR}-03-01"
    cache = load_zone_cache()
    def fresh(e, key=""):
        try:
            if key.startswith("b:") and (e.get("v") or {}).get("ver") != 2:
                return False                       # pre-v2 batter profile: refresh for self-mix centering
            return (dt.date.fromisoformat(today) - dt.date.fromisoformat(e["d"])).days < 5
        except Exception: return False
    bats, pits, pulled, failed = {}, {}, 0, 0
    try:
        from pybaseball import statcast_batter, statcast_pitcher
    except Exception as e:
        return {}, {}, f"heat off (pybaseball import: {type(e).__name__})"
    for pid in bat_ids:
        key = f"b:{pid}"
        if key in cache and fresh(cache[key], key):
            bats[pid] = cache[key]["v"]; continue
        try:
            prof = batter_zone_profile(statcast_batter(start, today, pid)); pulled += 1
        except Exception:
            prof = None; failed += 1
        bats[pid] = prof
        cache[key] = {"d": today, "v": prof}
    for pid in pit_ids:
        for stand in ("L", "R"):
            key = f"p:{pid}:{stand}"
            if key in cache and fresh(cache[key]):
                pits[(pid, stand)] = cache[key]["v"]; continue
            try:
                df = statcast_pitcher(start, today, pid); pulled += 1
                mix = pitcher_zone_mix(df, stand)
            except Exception:
                mix = None; failed += 1
            pits[(pid, stand)] = mix
            cache[key] = {"d": today, "v": mix}
    save_zone_cache(cache)
    ok_b = sum(1 for v in bats.values() if v); ok_p = sum(1 for v in pits.values() if v)
    return bats, pits, (f"heat on ({ok_b} bats x {ok_p} pitcher-sides, "
                        f"{pulled} pulls, {failed} fails)")

# ---------------------------------------------------------------------------
# pure compute — shared by live build and selftest
def build_board(batters, pitchers, sched, temps, props=None, hands=None, heats=None, cards=None, pens=None, barrels=None, marcel=None):
    """
    batters : list of {name, fg_team, pa, hr}
    pitchers: list of {name, bf, hr}
    sched   : list of {home, away, venue, home_sp, away_sp}   (full team names)
    temps   : {venue: (temp_f or None, roof)}
    props   : {norm_name: {"price": int, "book": str}} or None
    hands   : {norm_name: (batSide, pitchHand)} or None -> platoon off
    """
    hands = hands or {}
    heats = heats or {}
    cards = cards or {}
    pens = pens or {}
    barrels = barrels or {}
    L_brl = (sum(barrels.values())/len(barrels)) if barrels else None
    tb_pa = sum(b["pa"] for b in batters) or 1
    Lb = sum(b["hr"] for b in batters) / tb_pa
    tp_bf = sum(p["bf"] for p in pitchers) or 1
    Lp = sum(p["hr"] for p in pitchers) / tp_bf
    if Lb <= 0: Lb = 0.031
    if Lp <= 0: Lp = Lb

    # Composite starter factor. HR/BF is the slowest-stabilizing pitcher stat
    # because it bundles skill (contact allowed, fly-ball rate — both stabilize
    # ~80 BF) with luck (HR per fly ball — barely signal under ~400 FB). When
    # component data exists (statsapi tier: so/bb/ao) decompose and shrink each
    # at its own speed; league-average inputs compose to exactly 1.000 by
    # construction. Without components, legacy HR/BF shrink (K_PIT) unchanged.
    K_CT, K_FB, K_HRFB = 80.0, 80.0, 400.0
    comp = [p for p in pitchers if p.get("so") is not None and p.get("ao") is not None]
    kpool = [p for p in pitchers if p.get("so") is not None]
    CL = FL = HL = C2 = H2 = None
    if kpool:
        tb2 = sum(p["bf"] for p in kpool) or 1
        tct2 = sum(p["bf"] - p["so"] - p.get("bb", 0) for p in kpool)
        C2 = tct2 / tb2
        H2 = (sum(p["hr"] for p in kpool) / tct2) if tct2 else 0.0
    if comp:
        tb = sum(p["bf"] for p in comp) or 1
        tct = sum(p["bf"] - p["so"] - p.get("bb", 0) for p in comp)
        tfb = sum(p["ao"] + p["hr"] for p in comp)
        thr = sum(p["hr"] for p in comp)
        CL = tct / tb                      # league contact (BIP+HR) per BF
        FL = tfb / tct if tct else 0.0     # league fly per contact
        HL = thr / tfb if tfb else 0.0     # league HR per fly
    pit_fac, pit_bf = {}, {}
    for p in pitchers:
        n = norm(p["name"]); pit_bf[n] = p["bf"]
        if CL and p.get("so") is not None and p.get("ao") is not None:
            ct = p["bf"] - p["so"] - p.get("bb", 0)
            fb = p["ao"] + p["hr"]
            c = ((ct + K_CT * CL) / (p["bf"] + K_CT)) / CL
            f = (((fb + K_FB * FL) / (ct + K_FB)) / FL) if FL else 1.0
            h = (((p["hr"] + K_HRFB * HL) / (fb + K_HRFB)) / HL) if HL else 1.0
            fac = c * f * h
        elif C2 and p.get("so") is not None:
            # partial composite (FG/BREF tier: K and BB known, batted-ball mix not):
            # fast-stabilizing contact skill x hard-shrunk HR-per-contact
            K_H2 = 350.0
            ct = p["bf"] - p["so"] - p.get("bb", 0)
            c = ((ct + K_CT * C2) / (p["bf"] + K_CT)) / C2
            h = (((p["hr"] + K_H2 * H2) / (ct + K_H2)) / H2) if H2 else 1.0
            fac = c * h
        else:
            rate = (p["hr"] + K_PIT * Lp) / (p["bf"] + K_PIT)
            fac = rate / Lp
        pit_fac[n] = min(max(fac, 0.60), 1.60)

    by_team = {}
    seen_name = {}                                  # norm name -> index of kept batter
    deduped = []
    for b in sorted(batters, key=lambda x: -x["pa"]):
        nn = norm(b["name"])
        if nn in seen_name:
            continue                                # duplicate name: keep only the higher-PA real player
        seen_name[nn] = True
        deduped.append(b)
    # Team of record = today's roster, not the season-stats table. Without this a
    # player who changed clubs is projected into his OLD lineup for the whole year.
    moved = dropped = 0
    keep = []
    for b in deduped:
        nn = norm(b["name"])
        rt = ACTIVE_ROSTER.get(nn) or ROSTER_TEAM.get(nn)
        if rt and rt != b.get("fg_team"):
            b["fg_team"] = rt; moved += 1
        # Drop only when we hold that club's active roster and he is not on it —
        # i.e. injured/optioned. Never drop on a roster we failed to fetch.
        if ACTIVE_ROSTER and b["fg_team"] in ROSTER_OK and nn not in ACTIVE_ROSTER:
            dropped += 1; continue
        keep.append(b)
    deduped = keep
    if ROSTER_TEAM or ACTIVE_ROSTER:
        print(f"   roster: {moved} re-teamed, {dropped} dropped as inactive/IL "
              f"({len(ACTIVE_ROSTER)} active across {len(ROSTER_OK)} clubs)")
    for b in deduped:
        by_team.setdefault(b["fg_team"], []).append(b)
    lineup = {}
    for t, bs in by_team.items():
        bs = sorted(bs, key=lambda x: -x["pa"])[:9]
        lineup[t] = [(b, i + 1) for i, b in enumerate(bs)]

    def fg(full):
        return TEAM_MAP.get(full)

    bat_by_name = {norm(b["name"]): b for b in batters}
    def team_bats(full):
        card = cards.get(full)
        if card:
            out9 = []
            for nm, slot in card:
                b = bat_by_name.get(norm(nm))
                if b is None:
                    b = {"name": nm, "fg_team": full, "pa": 0, "hr": 0}   # call-up: league prior via shrink
                out9.append((b, slot))
            return out9
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
        tw = temps.get(g["venue"], (None, "open"))
        temp_f, roof = tw[0], tw[1]
        wspd = tw[2] if len(tw) > 2 else None
        wdir = tw[3] if len(tw) > 3 else None
        tmult, ttag = hr_weather_mult(temp_f, roof, wspd, wdir, g["venue"])
        for side, opp_sp_name in (("away", g.get("home_sp")), ("home", g.get("away_sp"))):
            team_full = g[side]
            opp_full = g["home" if side == "away" else "away"]
            fac, matched = sp_factor(opp_sp_name)
            bp_fac = 1.0
            if pens:
                k = opp_full if opp_full in pens else _resolve_team_key(opp_full, set(pens.keys()))
                bp_fac = pens.get(k, 1.0) if k else 1.0
            sp_eff = SP_WEIGHT * fac + (1 - SP_WEIGHT) * bp_fac
            pen_tag = f"pen {bp_fac:.2f}x" if abs(bp_fac - 1.0) >= 0.05 else ""
            for b, slot in team_bats(team_full):
                # BASE: Marcel multi-year talent (5/4/3, regressed, age-adjusted) blended
                # with this season's line — the pro approach — else the legacy shrink estimate.
                mt = (marcel or {}).get(norm(b["name"])) if marcel else None
                if mt and mt.get("marcel"):
                    base = _mc.blend_with_current(mt["marcel"], mt.get("cur_hr", 0),
                                                  mt.get("cur_pa", 0), lg_hrpa=Lb, k=200)
                    b_src = "mrc"
                else:
                    base = (b["hr"] + K_BAT * Lb) / (b["pa"] + K_BAT)
                    b_src = "shr"
                brl_tag = ""
                if barrels and L_brl:
                    bid = hands_get(hands, b["name"], "bat")[2]
                    bp_ = barrels.get(bid)
                    if bp_ is not None and base > 0:
                        implied = (bp_ / L_brl) * Lb
                        blended = (1 - BARREL_W) * base + BARREL_W * implied
                        if abs(blended / base - 1) >= 0.05:
                            brl_tag = f"brl {'+' if blended > base else ''}{(blended/base-1)*100:.0f}%"
                        base = blended
                sp_hand = hands_get(hands, opp_sp_name, "pit")[1] if isinstance(opp_sp_name, str) else None
                bat_side = hands_get(hands, b["name"], "bat")[0]
                plat_eff, plat_tag = platoon_factor(bat_side, sp_hand)
                hf = heats.get((norm(b["name"]), norm(opp_sp_name or "")))
                if hf:
                    hf_raw = SP_WEIGHT * hf + (1 - SP_WEIGHT)     # starter-share, as before
                    heat_eff = 1.0 + HEAT_WEIGHT * (hf_raw - 1.0) # demoted toward neutral
                else:
                    heat_eff = 1.0
                heat_tag = (f"heat {'+' if hf >= 1 else ''}{(hf-1)*100:.0f}%") if hf else ""
                p_pa = min(base * eff * sp_eff * tmult * plat_eff * heat_eff, CAP_PPA)
                pa_est = max(PA_TOP - 0.09 * (slot - 1), 3.4)
                p_game = 1 - (1 - p_pa) ** pa_est
                # SEASON ANCHOR - blend toward what the bat is doing THIS season (context-free),
                # because the walk-forward test shows the talent prior + factor stack is
                # systematically overconfident vs current-season reality. Display tag: szn.
                szn_tag = ""
                _chr = (mt.get("cur_hr", 0) if mt else b.get("hr", 0)) or 0
                _cpa = (mt.get("cur_pa", 0) if mt else b.get("pa", 0)) or 0
                if _cpa >= SEASON_MIN_PA:
                    # enough season sample: anchor toward the bat's own context-free
                    # CURRENT-SEASON rate (validated blend — unchanged).
                    _anchor = min((_chr + SEASON_K * Lb) / (_cpa + SEASON_K), CAP_PPA)
                else:
                    # season sample too noisy to trust the bat's own rate, but the factor-stack
                    # overconfidence still needs correcting: anchor toward the shrunk talent
                    # prior (context-free) so a tiny-sample flier can't sit at a full boosted
                    # number and tie a proven bat. Prior is already league-shrunk (K_BAT), so
                    # this collapses toward league exactly when the sample is thinnest.
                    _anchor = min(base, CAP_PPA)
                _gszn = 1 - (1 - _anchor) ** pa_est
                _blend = (1 - SEASON_W) * p_game + SEASON_W * _gszn
                if p_game > 0 and abs(_blend / p_game - 1) >= 0.05:
                    szn_tag = f"szn {'+' if _blend > p_game else ''}{(_blend/p_game-1)*100:.0f}%"
                p_game = _blend
                p_game = calibrate_pct(p_game * 100) / 100      # backtest recalibration
                sp_disp = opp_sp_name if isinstance(opp_sp_name, str) and opp_sp_name.strip() else "TBD"
                row = {
                    "player": b["name"], "team": fg(team_full) or team_full,
                    "opp": fg(opp_full) or opp_full,
                    "opp_sp": sp_disp + ("" if matched else " *"),
                    "venue": g["venue"], "slot": slot,
                    "hr_pct": round(p_game * 100, 1),
                    "fair": am_from_p(p_game),
                    "park": park_lab, "temp": ttag,
                    "sp_fac": round(fac, 2), "sp_small": bool(matched and pit_bf.get(norm(opp_sp_name or ""), 999) < 60), "plat": plat_tag, "heat": heat_tag, "pen": pen_tag, "brl": brl_tag, "szn": szn_tag, "lu": ("card" if cards.get(team_full) else "proj"), "b_src": b_src,
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
    hc = _col(df, "HR"); sc = _col(df, "SO", "K"); wc = _col(df, "BB")
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
            row = {"name": str(r[nc]), "bf": bf, "hr": hr}
            try:
                if sc and r.get(sc) == r.get(sc): row["so"] = int(float(r[sc]))
                if wc and r.get(wc) == r.get(wc): row["bb"] = int(float(r[wc]))
            except (TypeError, ValueError): pass
            out.append(row)
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
                        out.append({"name": name, "bf": bf, "hr": hr,
                                    "team": team, "gs": int(st.get("gamesStarted", 0) or 0),
                                    "so": int(st.get("strikeOuts", 0) or 0),
                                    "bb": int(st.get("baseOnBalls", 0) or 0),
                                    "ao": int(st.get("airOuts", 0) or 0)})
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
            t, w, wd = fetch_weather(lat, lon)
            temps[v] = (t, roof, w, wd)
        except Exception:
            temps[v] = (None, roof)
    return temps

def pull_lineups(sched):
    """Actual posted lineup cards from MLB StatsAPI boxscores (post ~1-4h
    pregame). Returns ({full_team_name: [(player_name, slot1..9)]}, note).
    Games without a posted card fall back to season-usage projection."""
    base = "https://statsapi.mlb.com/api/v1"
    def _get(url):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (MLBTool board)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    try:
        today = dt.date.today().strftime("%m/%d/%Y")
        s = _get(f"{base}/schedule?sportId=1&date={today}")
        pks = {}
        for d0 in s.get("dates", []):
            for g in d0.get("games", []):
                h = ((g.get("teams") or {}).get("home") or {}).get("team", {}).get("name")
                a = ((g.get("teams") or {}).get("away") or {}).get("team", {}).get("name")
                if h and a: pks[(h, a)] = g.get("gamePk")
    except Exception as e:
        return {}, f"lineups: schedule fetch failed ({type(e).__name__}) — season-usage for all"
    cards, posted, total = {}, 0, 0
    for g in sched:
        total += 1
        pk = pks.get((g["home"], g["away"]))
        if not pk: continue
        try:
            box = _get(f"{base}/game/{pk}/boxscore")
        except Exception:
            continue
        for side in ("home", "away"):
            t = (box.get("teams") or {}).get(side) or {}
            order = t.get("battingOrder") or []
            if len(order) < 9: continue
            players = t.get("players") or {}
            lu = []
            for i, pid in enumerate(order[:9]):
                p = players.get(f"ID{pid}") or {}
                nm = (p.get("person") or {}).get("fullName", "")
                if nm: lu.append((nm, i + 1))
            if len(lu) == 9:
                cards[g[side]] = lu
                if side == "home": posted += 1     # count per game via home card
    return cards, f"lineups: {posted}/{total} cards posted (rest = season-usage)"

def pull_props():
    """batter_home_runs Yes/Over-0.5 best price per player. Fails soft:
    returns ({}, reason). Costs ~1 credit per event on The Odds API."""
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        return {}, "props off (ODDS_API_KEY secret not set)"
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
    # 11. PLATOON: LHB must price lower vs LHP than vs RHP; switch immune to the
    #     penalty; unknown handedness neutral; tag present when applied.
    hands = {"slug mcpower": ("L", None), "ace groundall": (None, "L"), "gopher gary": (None, "R")}
    r_lhp, _ = build_board(bat, pit, [dict(sched[0])], {"Yankee Stadium": (88.0, "open")}, hands=hands)
    # BOS bats face Ace (LHP); NYY bats face Gary (RHP). Slug is NYY -> faces RHP.
    hands2 = dict(hands); hands2["ace groundall"] = (None, "R")
    r_rhp, _ = build_board(bat, pit, [dict(sched[0])], {"Yankee Stadium": (88.0, "open")}, hands=hands2)
    # flip: make Slug face the lefty by swapping which SP is L
    hands3 = {"slug mcpower": ("L", None), "gopher gary": (None, "L")}
    r_vL, _ = build_board(bat, pit, [dict(sched[0])], {"Yankee Stadium": (88.0, "open")}, hands=hands3)
    pv = lambda rows: [r for r in rows if r["player"] == "Slug McPower"][0]
    assert pv(r_vL)["hr_pct"] < pv(r_lhp)["hr_pct"], "LHB vs LHP must price below vs RHP"
    assert "Lv" in pv(r_vL)["plat"], pv(r_vL)["plat"]
    assert pv(r_lhp)["plat"].startswith("LvR"), pv(r_lhp)["plat"]
    hands4 = {"slug mcpower": ("S", None), "gopher gary": (None, "L")}
    r_sw, _ = build_board(bat, pit, [dict(sched[0])], {"Yankee Stadium": (88.0, "open")}, hands=hands4)
    assert pv(r_sw)["hr_pct"] > pv(r_vL)["hr_pct"], "switch hitter must dodge the LL penalty"
    r_unk, _ = build_board(bat, pit, [dict(sched[0])], {"Yankee Stadium": (88.0, "open")})
    assert pv(r_unk)["plat"] == "" 
    # 16. GARCIA REGRESSION: duplicate names must resolve by role, both directions
    dup = {"luis garcia": [("L", "R", 111, "2B"), (None, "R", 222, "P")]}
    assert hands_get(dup, "Luis García Jr.", "bat") == ("L", "R", 111)
    assert hands_get(dup, "Luis Garcia", "pit") == (None, "R", 222)
    assert hands_get(dup, "Nobody Here", "bat") == (None, None, None)
    # 17. BULLPEN: soft opposing pen must raise the row; unknown team neutral
    pens = {"Boston Red Sox": 1.20}
    r_pen, _ = build_board(bat, pit, sched, {"Yankee Stadium": (88.0, "open")}, pens=pens)
    nyy_p = [r for r in r_pen if r["team"] == "NYY"][0]      # NYY bats face BOS pen
    nyy_0 = [r for r in rows if r["team"] == "NYY" and r["player"] == nyy_p["player"]][0]
    assert nyy_p["hr_pct"] > nyy_0["hr_pct"] and nyy_p["pen"] == "pen 1.20x"
    bos_p = [r for r in r_pen if r["team"] == "BOS"][0]
    assert bos_p["pen"] == "" and abs(bos_p["hr_pct"] - [r for r in rows if r["player"]==bos_p["player"]][0]["hr_pct"]) < 0.05
    # 18. BARREL: high-barrel bat rises with tag; barrel-less bat untouched
    hands_b = {"slug mcpower": [("L", None, 500, "DH")]}
    brl = {500: 12.0}   # league mean will be 12 -> implied == Lb; craft asym:
    brl = {500: 24.0, 999: 8.0}                      # mean 16 -> slug implied 1.5x league
    r_b, _ = build_board(bat, pit, sched, {"Yankee Stadium": (88.0, "open")}, hands=hands_b, barrels=brl)
    slug_b = [r for r in r_b if r["player"] == "Slug McPower"][0]
    slug_0 = [r for r in rows if r["player"] == "Slug McPower"][0]
    assert slug_b["brl"].startswith("brl ") and abs(slug_b["hr_pct"] - slug_0["hr_pct"]) > 0.3
    other = [r for r in r_b if r["player"] == "Mid Bat"][0]
    assert other["brl"] == "" 
    # 19. PHANTOM (Ben Rice NYM): a duplicate name across two teams must collapse
    #     to the higher-PA real player — never appear on both teams' boards.
    #     (Production feeds one consistent team format; fixture mirrors that.)
    base_bats = [{"name": f"Y{i}", "fg_team": "New York Yankees", "pa": 250 - i, "hr": 10} for i in range(8)]
    base_bats += [{"name": f"M{i}", "fg_team": "New York Mets", "pa": 250 - i, "hr": 10} for i in range(8)]
    dup_bats = [{"name": "Ben Rice", "fg_team": "New York Yankees", "pa": 300, "hr": 15},
                {"name": "Ben Rice", "fg_team": "New York Mets", "pa": 8, "hr": 1}]
    two_g = [{"home": "New York Yankees", "away": "New York Mets", "venue": "Yankee Stadium",
              "home_sp": "arm x", "away_sp": "arm y"}]
    rr, _ = build_board(dup_bats + base_bats, [{"name": "arm x", "bf": 300, "hr": 10},
                                               {"name": "arm y", "bf": 300, "hr": 10}], two_g, {})
    rices = [r for r in rr if r["player"] == "Ben Rice"]
    assert len(rices) == 1, f"phantom survived on teams: {[r['team'] for r in rices]}"
    assert rices[0]["team"] in ("NYY", "New York Yankees"), rices[0]["team"]
    # 20-22. COMPOSITE PITCHER FACTOR (the Burns fix)
    # 20: centering by construction — in an all-average pool, every factor is exactly 1.000
    avg = [{"name": f"avg{i}", "bf": 400, "hr": 12, "so": 90, "bb": 30, "ao": 108} for i in range(10)]
    g1 = [{"home": "New York Yankees", "away": "Boston Red Sox", "venue": "Yankee Stadium",
           "home_sp": "avg0", "away_sp": "avg1"}]
    rows20, _ = build_board(bat, avg, g1, {})
    assert all(r["sp_fac"] == 1.0 for r in rows20), [r["sp_fac"] for r in rows20]
    # 21: with outliers in the pool — elite-K arm (100 BF) drops well below 1.0;
    #     low-K flyball arm reads high; HR/FB luck outlier moves far less than raw HR/BF would
    burns = {"name": "Ace Burns", "bf": 100, "hr": 3, "so": 46, "bb": 6, "ao": 18}
    gopher = {"name": "Gopher Guy", "bf": 400, "hr": 12, "so": 80, "bb": 30, "ao": 150}
    lucky = {"name": "HRFB Outlier", "bf": 100, "hr": 8, "so": 22, "bb": 8, "ao": 22}
    pool = avg + [burns, gopher, lucky]
    g2 = [{"home": "New York Yankees", "away": "Boston Red Sox", "venue": "Yankee Stadium",
           "home_sp": "Gopher Guy", "away_sp": "Ace Burns"},
          {"home": "New York Mets", "away": "Atlanta Braves", "venue": "Citi Field",
           "home_sp": "HRFB Outlier", "away_sp": "avg0"}]
    rows21, _ = build_board(bat + [{"name": "Met Bat", "fg_team": "New York Mets", "pa": 300, "hr": 10},
                                   {"name": "Brave Bat", "fg_team": "Atlanta Braves", "pa": 300, "hr": 10}],
                            pool, g2, {})
    f_burns = [r for r in rows21 if r["team"] == "NYY"][0]["sp_fac"]
    f_gopher = [r for r in rows21 if r["team"] == "BOS"][0]["sp_fac"]
    f_lucky = [r for r in rows21 if r["team"] == "ATL"][0]["sp_fac"]
    assert f_burns <= 0.90, f_burns
    assert f_gopher >= 1.10, f_gopher
    assert f_lucky <= 1.13 and f_lucky < f_gopher, (f_lucky, f_gopher)
    # raw HR/BF would have priced the luck outlier at (8+200*Lp)/(100+200)/Lp — far higher
    # 22: small-sample caveat flag under 60 BF; legacy path (no so/ao) still works
    tiny = {"name": "Tiny Arm", "bf": 45, "hr": 1, "so": 15, "bb": 4, "ao": 12}
    g3 = [{"home": "New York Yankees", "away": "Boston Red Sox", "venue": "Yankee Stadium",
           "home_sp": "avg0", "away_sp": "Tiny Arm"}]
    rows22, _ = build_board(bat, avg + [tiny], g3, {})
    assert [r for r in rows22 if r["team"] == "NYY"][0]["sp_small"] is True
    assert [r for r in rows22 if r["team"] == "BOS"][0]["sp_small"] is False
    # 23. PARTIAL TIER (BREF/FG: so/bb but no ao) — centering exact; elite-K arm still drops
    avgp = [{"name": f"pavg{i}", "bf": 400, "hr": 12, "so": 90, "bb": 30} for i in range(10)]
    pburns = {"name": "K Burns", "bf": 100, "hr": 3, "so": 46, "bb": 6}
    g4 = [{"home": "New York Yankees", "away": "Boston Red Sox", "venue": "Yankee Stadium",
           "home_sp": "pavg0", "away_sp": "K Burns"}]
    r23a, _ = build_board(bat, avgp, g4, {})
    assert all(r["sp_fac"] == 1.0 for r in r23a), [r["sp_fac"] for r in r23a]
    r23b, _ = build_board(bat, avgp + [pburns], g4, {})
    f_kburns = [r for r in r23b if r["team"] == "NYY"][0]["sp_fac"]
    assert f_kburns <= 0.93, f_kburns
    # 12. REAL LINEUPS: posted card overrides usage — order, exclusion, call-up prior
    card = {"New York Yankees": [("NY filler4", 1), ("Slug McPower", 2), ("Callup Kid", 3),
            ("Mid Bat", 4), ("Slap Hitter", 5), ("NY filler0", 6), ("NY filler1", 7),
            ("NY filler2", 8), ("NY filler3", 9)]}
    r_card, _ = build_board(bat, pit, sched, {"Yankee Stadium": (88.0, "open")}, cards=card)
    nyy = [r for r in r_card if r["team"] == "NYY"]
    assert len(nyy) == 9 and all(r["lu"] == "card" for r in nyy)
    assert not any(r["player"] == "Tiny Sample" for r in nyy), "benched player must vanish"
    kid = [r for r in nyy if r["player"] == "Callup Kid"][0]
    assert 0 < kid["hr_pct"] < 25 and kid["slot"] == 3          # league-prior via shrink
    slugc = [r for r in nyy if r["player"] == "Slug McPower"][0]
    assert slugc["slot"] == 2
    bos = [r for r in r_card if r["team"] == "BOS"]
    assert bos and all(r["lu"] == "proj" for r in bos), "no card -> usage projection"
    # 13. WIND: out > calm > in at same temp; dome off; unknown orientation temp-only
    from mlb_hr import hr_weather_mult
    m_out,_t1 = hr_weather_mult(80, "open", 14, (75+180)%360, "Yankee Stadium")   # straight out
    m_cal,_t2 = hr_weather_mult(80, "open", 0, 0, "Yankee Stadium")
    m_in ,t3  = hr_weather_mult(80, "open", 14, 75, "Yankee Stadium")             # straight in
    assert m_out > m_cal > m_in and "wind in" in t3
    assert hr_weather_mult(95, "dome", 20, 0, "Yankee Stadium")[0] == 1.0
    m_unk, t_unk = hr_weather_mult(80, "open", 14, 0, "Estadio Nowhere")
    assert abs(m_unk - (1+0.008*10)) < 1e-9 and "wind" not in t_unk 
    # 14. REGRESSION (uniform -15% board): profiles cached through JSON come back
    #     with STRING zone keys — the factor must be identical fresh vs reloaded.
    bp = {"z": {z: 0.01 + (0.02 if z == 5 else 0) for z in [1,2,3,4,5,6,7,8,9,11,12,13,14]},
          "ov": 0.012, "n": 2000}
    pm = {"w": {5: 0.5, 13: 0.5}, "n": 1500}
    f_fresh = heat_factor(bp, pm)
    bp2, pm2 = json.loads(json.dumps(bp)), json.loads(json.dumps(pm))
    f_cached = heat_factor(bp2, pm2)
    assert f_fresh is not None and abs(f_fresh - f_cached) < 1e-12, (f_fresh, f_cached)
    assert f_fresh > 1.02, f_fresh                      # zone-5 heavy mix must read hot, not floor
    assert heat_factor(bp, {"w": {}, "n": 500}) is None  # zero coverage -> None, never a fake 0.85
    # 15. SELF-MIX CENTERING: pitcher throwing the batter's exact diet -> 1.000;
    #     heart-heavy vs that diet -> >1; chase-heavy -> <1; legacy (no "s") still works.
    diet = {z: (0.10 if z in (4,5,6) else 0.07) for z in [1,2,3,4,5,6,7,8,9,11,12,13,14]}
    tot = sum(diet.values()); diet = {z: v/tot for z, v in diet.items()}
    bpc = {"z": {z: (0.030 if z in (4,5,6) else 0.006) for z in diet}, "s": dict(diet),
           "ov": 0.011, "n": 2500, "ver": 2}
    f_same = heat_factor(bpc, {"w": dict(diet), "n": 1500})
    assert abs(f_same - 1.0) < 1e-9, f_same
    f_heart = heat_factor(bpc, {"w": {5: 1.0}, "n": 1500})
    f_chase = heat_factor(bpc, {"w": {13: 1.0}, "n": 1500})
    assert f_heart > 1.0 > f_chase, (f_heart, f_chase)
    legacy = {"z": bpc["z"], "ov": 0.011, "n": 2500}
    assert heat_factor(legacy, {"w": dict(diet), "n": 1500}) is not None
    print(f"SELFTEST PASS — {len(rows)} rows, top: {rows[0]['player']} "
          f"{rows[0]['hr_pct']}% (fair {rows[0]['fair']})")
    return 0

def _build():
    print("1) schedule…"); sched = todays_sched()
    print(f"   {len(sched)} games")
    print("2) batters…"); bat, bsrc = pull_batters()
    print("3) pitchers…"); pit, psrc = pull_pitchers()
    print("3b) handedness (MLB StatsAPI)…"); hands, hnote = pull_handedness(); print(f"   {hnote}")
    print("3c) active rosters (MLB StatsAPI)…")
    _nt = pull_active_rosters()
    print(f"   active rosters: {len(ACTIVE_ROSTER)} players across {_nt}/30 clubs"
          + ("" if _nt >= 25 else "  ** thin — IL filter will be partial **"))
    if not bat or not pit:
        sys.exit("no batter/pitcher data from any source — board not built")
    tkeys = sorted({b["fg_team"] for b in bat})
    print(f"   team labels in batter data ({len(tkeys)}): {tkeys[:34]}")
    print(f"   schedule teams: {sorted({g[s] for g in sched for s in ('home','away')})[:8]} …")
    print("4) park temps (Open-Meteo)…")
    temps = pull_temps(sorted({g["venue"] for g in sched}))
    print("3c) bullpens (StatsAPI)…"); pens, pen_note = pull_bullpens(); print(f"   {pen_note}")
    print("3d) barrels (Savant)…"); barrels, brl_note = pull_barrels(); print(f"   {brl_note}")
    print("4b) lineup cards (StatsAPI)…"); cards, lu_note = pull_lineups(sched); print(f"   {lu_note}")
    print("5) HR props (The Odds API, optional)…")
    props, pnote = pull_props(); print(f"   {pnote}")
    prelim, _ = build_board(bat, pit, sched, temps, None, hands or None, None, cards or None, pens or None, barrels or None)
    cand = sorted(prelim, key=lambda r: -r["hr_pct"])[:70]
    def _pid(nm, want="bat"):
        return hands_get(hands, nm, want)[2] if hands else None
    bat_ids = sorted({_pid(r["player"]) for r in cand if _pid(r["player"])})
    sp_names = sorted({(g.get(s) or "") for g in sched for s in ("home_sp", "away_sp") if isinstance(g.get(s), str)})
    pit_ids = sorted({_pid(n, "pit") for n in sp_names if _pid(n, "pit")})
    print(f"5b) heat maps (Savant zones) — {len(bat_ids)} bats, {len(pit_ids)} starters…")
    bz, pz, heat_note = fetch_zone_profiles(bat_ids, pit_ids, hands or {})
    print(f"   {heat_note}")
    heats = {}
    if bz or pz:
        for r in cand:
            bid = _pid(r["player"])
            spn = r["opp_sp"].replace(" *", "").strip()
            spid = _pid(spn, "pit")
            side = hands_get(hands, r["player"], "bat")[0] if hands else None
            if side == "S" and hands and _pid(spn, "pit"):
                sph = hands_get(hands, spn, "pit")[1]
                side = "R" if sph == "L" else "L"
            if bid and spid and side in ("L", "R"):
                hf = heat_factor(bz.get(bid), pz.get((spid, side)))
                if hf: heats[(norm(r["player"]), norm(spn))] = hf
    marcel = None
    try:
        import json as _json
        mp = _json.load(open(os.path.join(DATA, "marcel_talent.json")))
        marcel = mp.get("players")
        print(f"   Marcel talent: {len(marcel)} hitters (generated {mp.get('generated','?')})")
    except Exception as _e:
        print(f"   Marcel talent: not available ({type(_e).__name__}) — using single-season base")
    rows, have_ev = build_board(bat, pit, sched, temps, props or None, hands or None, heats or None, cards or None, pens or None, barrels or None, marcel=marcel)
    rows = select_rows(rows, have_ev)
    annotate_spots(rows, os.path.join(DATA, "hr_predictions.csv"), dt.date.today().isoformat())
    _mrc = sum(1 for r in rows if r.get("b_src")=="mrc")
    print(f"   base: {_mrc}/{len(rows)} rows using Marcel talent, rest single-season fallback")
    print(f"   board rows built: {len(rows)}")
    print_spotlight(rows)
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
            f"{lu_note} · "
            f"{hnote} (league factors, starter share) · {heat_note} · {pen_note} · {brl_note} · "
            "wind: direction x park bearing (conf-shrunk) · park HR factors are seed approximations "
            "(conf-shrunk; refresh from Savant) · temp vs flat 70F baseline (mildly double-counts "
            "warm-climate open parks) · " + pnote +
            (" · sorted by model HR% · EV screen appended" if have_ev else " · sorted by model HR% (no props matched)"))
    out = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "note": note, "rows": rows}
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "hr_board.json")
    with open(path, "w") as f:
        json.dump(_scrub(out), f, indent=1, allow_nan=False)
    try:
        import csv
        plog = os.path.join(DATA, "hr_predictions.csv")
        today = dt.date.today().isoformat()
        hdr = ["date","player","team","opp_sp","slot","lu","hr_pct","fair","book_price","ev_pct","park","temp","plat","heat","edge_self","spot"]
        old_rows = []
        if os.path.exists(plog):
            with open(plog) as f:
                old_rows = [r for r in csv.DictReader(f) if r.get("date") != today]
        with open(plog, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
            for r in old_rows: w.writerow({k: r.get(k, "") for k in hdr})
            for r in rows:
                w.writerow({"date": today, "player": r["player"], "team": r["team"],
                            "opp_sp": r["opp_sp"], "slot": r["slot"], "lu": r.get("lu",""),
                            "hr_pct": r["hr_pct"], "fair": r["fair"],
                            "book_price": r.get("book_price",""), "ev_pct": r.get("ev_pct",""),
                            "park": r["park"], "temp": r["temp"], "plat": r.get("plat",""),
                            "heat": r.get("heat",""),
                            "edge_self": ("" if r.get("edge_self") is None else r.get("edge_self")),
                            "spot": r.get("spot","")})
        print(f"   prediction log: {len(rows)} rows for {today} ({len(old_rows)} historical kept)")
    except Exception as e:
        print(f"   prediction log skipped: {type(e).__name__}")
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
