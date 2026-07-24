#!/usr/bin/env python3
"""
mlb_framing_experiment.py — DOES THE CATCHER'S FRAMING SHARPEN PITCHER-K PROPS?
================================================================================
STANDALONE EXPERIMENT. Does not import or modify mlb_kprops.py or any production
file. It only builds evidence and prints a VERDICT.

THE HYPOTHESIS
--------------
Pitch framing moves called strikes, and called strikes move strikeouts. A good
framer adds roughly 1-2% called-strike rate. The validated baseline projects a
start's strikeouts as Poisson(lambda) with
    lambda_b = pitcher_season_rate(strictly-prior, league-regressed) * BF
               * opp_whiff_factor(w=0.6)
TREATMENT multiplies lambda_b by a framing factor built from the strike-rate-
above-average of the catcher STARTING BEHIND THE PLATE for the pitcher's own
team, with ONE weight w_f tuned on TRAIN (grid includes 0 = baseline).

CRITICAL LEAK RULE
------------------
For a start in season Y the framing table is season Y-1 (prior season, frozen
pregame knowledge). Same-season framing would leak in-season information; the
selftest constructs a case where using it flips the answer.

DATA
----
(1) Catcher framing: Baseball Savant catcher-framing leaderboard CSV
    https://baseballsavant.mlb.com/leaderboard/catcher-framing?year={Y}&min=q&sort=4,1&csv=true
    Column drift is tolerated: the header actually received is printed, the name
    column(s) and a strike-rate (preferred) or framing-runs (fallback) column
    are auto-detected.
(2) Per-game starting catcher + starts + team batting: statsapi schedule +
    boxscores, the same endpoints production uses. The starting catcher is the
    position-'C' player holding a battingOrder slot (100..900) on the pitcher's
    own side of the boxscore.
Dataset cached to mlb/data/framing_dataset.json so re-runs don't re-pull.

NAME JOIN
---------
Savant lists 'Last, First'; statsapi lists 'First Last'. Both are normalized
(accents stripped, punctuation dropped, comma form flipped) plus a
last-name+first-initial fallback. The match rate over scorable starts is
reported; unmatched catchers are neutral (factor 1.0). If <70% of starts get a
framing value the join is too weak and the verdict says so LOUDLY.

NETWORK
-------
In THIS sandbox egress to statsapi/savant is blocked — EXPECTED. On GitHub
Actions both are reachable (production mlb_hr pulls barrels from savant; a
savant-probe workflow exists). If blocked, the pull prints a clear message and
exits 0. It never fakes numbers.

RUN
---
  python3 mlb_framing_experiment.py --selftest        # offline, no network, must pass
  python3 mlb_framing_experiment.py                   # live pull + verdict (on Actions)
  python3 mlb_framing_experiment.py --start 2025-04-01 --end 2025-06-30
"""
import os, sys, io, csv, json, math, time, argparse, unicodedata, datetime as dt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASET_PATH = os.path.join(DATA, "framing_dataset.json")
REPORT_PATH = os.path.join(DATA, "framing_experiment.json")

# ---- experiment-local constants (production untouched) ----------------------
LG_K_PER_BF = 0.222     # league K per batter faced fallback (~league K%)
REG_BF = 100.0          # league BF added to a pitcher's rate (same shrink as baseline)
MIN_PRIOR_STARTS = 3    # a pitcher needs this many prior starts to be scored
MIN_OPP_PA = 300        # opponent needs this many prior batting PAs (else neutral)
W_OPP = 0.60            # baseline opponent-whiff weight — VALIDATED, fixed, both arms
OPP_LO, OPP_HI = 0.80, 1.20
TRAIN_FRAC = 0.5        # earlier fraction of dates = TRAIN (tuning); later = HOLDOUT
N_PERIODS = 3           # holdout robustness buckets
TRAIN_WIN_MARGIN = 5e-4 # train mean-LL/start improvement needed to claim a train win
MIN_COVERAGE = 0.70     # below this framing-join coverage the verdict is INVALID

# framing factor: 1 + w_f * delta, delta = prior-season strike-rate above average
# (fraction, e.g. +0.015 = +1.5 points). Grid contains 0.0 = baseline exactly.
FR_W_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
FR_LO, FR_HI = 0.85, 1.15
RUNS_TO_RATE_STD = 0.012  # runs-column fallback: 1 cross-sectional std ~= 1.2 rate pts


def clip(x, lo, hi):
    return min(hi, max(lo, x))


# ===========================================================================
# POISSON MATH (pure, exact) — the scoring core, testable to hand values.
# ===========================================================================
def poisson_logpmf(k, lam):
    """log P(X = k) for X ~ Poisson(lam). Exact: k*ln(lam) - lam - ln(k!)."""
    lam = max(float(lam), 1e-9)
    k = int(k)
    return k * math.log(lam) - lam - math.lgamma(k + 1)


# ===========================================================================
# NAME NORMALIZATION — savant 'Last, First' <-> statsapi 'First Last'.
# ===========================================================================
def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s or "")
                   if not unicodedata.combining(c))


def norm(s):
    if not isinstance(s, str):
        return ""
    return "".join(c for c in strip_accents(s).lower() if c.isalnum())


def name_key(s):
    """Canonical key for a person name. Accepts 'Last, First' (savant) and
    'First Last' (statsapi); strips accents/punctuation. Both formats of the
    same name produce the SAME key."""
    s = strip_accents(s or "").strip()
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    return norm(s)


def alt_key_first_last(s):
    """Fallback key from 'First Last': last token + first initial."""
    toks = strip_accents(s or "").split()
    if len(toks) < 2:
        return None
    return "~" + norm(toks[-1]) + norm(toks[0])[:1]


def alt_key_last_first(last, first):
    l, f = norm(last), norm(first)
    if not l or not f:
        return None
    return "~" + l + f[:1]


def table_from_savant_rows(rows):
    """rows: [(savant_name_or_(last,first), delta_fraction)]. Builds a lookup
    dict with canonical keys plus last+initial fallback keys (fallback dropped
    on collision — ambiguity must never mis-join)."""
    table, alt_seen = {}, {}
    for name, delta in rows:
        if isinstance(name, tuple):
            last, first = name
            disp = f"{last}, {first}"
        else:
            disp = name
            last, _, first = name.partition(",")
        table[name_key(disp)] = delta
        ak = alt_key_last_first(last, first)
        if ak:
            if ak in alt_seen:
                table.pop(ak, None)       # collision -> ambiguous -> drop
            else:
                alt_seen[ak] = True
                table[ak] = delta
    return table


def framing_delta(catcher_name, season, framing):
    """(delta, matched) for a catcher in a start of `season`, read STRICTLY from
    the PRIOR season's table — the one leak rule of this experiment."""
    table = framing.get(season - 1) or {}
    k = name_key(catcher_name)
    if k and k in table:
        return table[k], True
    ak = alt_key_first_last(catcher_name)
    if ak and ak in table:
        return table[ak], True
    return 0.0, False


# ===========================================================================
# LEAK-FREE CONTEXT BUILDER — identical discipline to the kfactors experiment:
# one chronological pass; every feature in a context comes from rows dated
# strictly BEFORE the start; state folds AFTER a date is predicted. The framing
# delta is prior-SEASON (frozen pregame), attached at scoring time via
# framing_delta() and therefore also leak-free by construction.
# ===========================================================================
def build_contexts(starts, team_bat):
    """
    starts   : [{date(dt.date), pitcher, so, bf, home(bool), team, opp, catcher}]
    team_bat : [{date(dt.date), team, bat_so, bat_pa}]  (one row per team per game)
    """
    all_dates = sorted({s["date"] for s in starts} | {t["date"] for t in team_bat})
    starts_by_date, bat_by_date = {}, {}
    for s in starts:
        starts_by_date.setdefault(s["date"], []).append(s)
    for t in team_bat:
        bat_by_date.setdefault(t["date"], []).append(t)

    pit = {}                 # norm(pitcher) -> [(date, so, bf)] chronological
    team_tot = {}            # norm(team) -> (so_cum, pa_cum)
    lg_so = lg_pa = 0

    ctxs = []
    for date in all_dates:
        # 1) SNAPSHOT this date's starts from strictly-prior state
        for s in starts_by_date.get(date, []):
            prior = list(pit.get(norm(s["pitcher"]), []))
            oso, opa = team_tot.get(norm(s.get("opp", "")), (0, 0))
            lg = (lg_so / lg_pa) if lg_pa > 0 else LG_K_PER_BF
            ctxs.append({
                "date": date, "season": date.year,
                "pitcher": s["pitcher"], "k": int(s["so"]), "bf": int(s["bf"]),
                "catcher": s.get("catcher", ""),
                "opp": s.get("opp", ""),
                "prior": prior, "n_prior": len(prior),
                "lg": lg, "opp_so": oso, "opp_pa": opa,
            })
        # 2) FOLD this date's rows into state (visible to LATER dates only)
        for s in starts_by_date.get(date, []):
            pit.setdefault(norm(s["pitcher"]), []).append(
                (date, int(s["so"]), int(s["bf"])))
        for t in bat_by_date.get(date, []):
            key = norm(t["team"])
            cso, cpa = team_tot.get(key, (0, 0))
            team_tot[key] = (cso + int(t["bat_so"]), cpa + int(t["bat_pa"]))
            lg_so += int(t["bat_so"])
            lg_pa += int(t["bat_pa"])
    return ctxs


# ===========================================================================
# THE MODELS — pure functions of a frozen context (+ framing tables + params).
# ===========================================================================
def rate_season(ctx):
    """K/BF over strictly-prior starts, league-regressed."""
    so = sum(p[1] for p in ctx["prior"])
    bf = sum(p[2] for p in ctx["prior"])
    return (so + REG_BF * ctx["lg"]) / (bf + REG_BF)


def factor_opp(ctx):
    """VALIDATED baseline component: opponent whiff, w fixed at 0.6 (both arms)."""
    if ctx["opp_pa"] < MIN_OPP_PA or ctx["lg"] <= 0:
        return 1.0
    raw = (ctx["opp_so"] / ctx["opp_pa"]) / ctx["lg"]
    return clip(1.0 + W_OPP * (raw - 1.0), OPP_LO, OPP_HI)


def lam_baseline(ctx):
    return rate_season(ctx) * ctx["bf"] * factor_opp(ctx)


def factor_framing(ctx, w, framing):
    """TREATMENT: 1 + w * prior-season strike-rate-above-average of the starting
    catcher behind the pitcher. Unmatched catcher -> neutral 1.0."""
    delta, matched = framing_delta(ctx["catcher"], ctx["season"], framing)
    if not matched or w == 0.0:
        return 1.0
    return clip(1.0 + w * delta, FR_LO, FR_HI)


def lam_framing(ctx, w, framing):
    return lam_baseline(ctx) * factor_framing(ctx, w, framing)


# ===========================================================================
# TUNING (TRAIN ONLY) + HOLDOUT SCORING
# ===========================================================================
def _ll(ctxs, lam_fn):
    return sum(poisson_logpmf(c["k"], lam_fn(c)) for c in ctxs)


def tune_framing(train_ctxs, framing):
    """Grid-tune w on TRAIN. The grid contains 0.0 (= baseline exactly), so
    best-train >= baseline-train by construction; the reported delta is the
    margin over neutral."""
    n = max(1, len(train_ctxs))
    base_ll = _ll(train_ctxs, lam_baseline)
    best = max(FR_W_GRID,
               key=lambda w: _ll(train_ctxs, lambda c: lam_framing(c, w, framing)))
    dll = (_ll(train_ctxs, lambda c: lam_framing(c, best, framing)) - base_ll) / n
    return {"param": {"w": best}, "train_dll_per_start": dll,
            "train_win": dll > TRAIN_WIN_MARGIN}


def _score_holdout(hold_ctxs, w, framing, n_periods=N_PERIODS):
    """Treatment vs baseline on the holdout: total & per-start Poisson LL delta,
    MAE delta, per-period robustness."""
    rows = []
    for c in hold_ctxs:
        lam_b = lam_baseline(c)
        lam_m = lam_framing(c, w, framing)
        rows.append({"date": c["date"],
                     "ll_b": poisson_logpmf(c["k"], lam_b),
                     "ll_m": poisson_logpmf(c["k"], lam_m),
                     "ae_b": abs(c["k"] - lam_b), "ae_m": abs(c["k"] - lam_m)})
    n = len(rows)
    if not n:
        return {"n": 0}
    ll_b = sum(r["ll_b"] for r in rows)
    ll_m = sum(r["ll_m"] for r in rows)
    out = {
        "n": n,
        "ll_b": round(ll_b, 3), "ll_m": round(ll_m, 3),
        "ll_delta_total": round(ll_m - ll_b, 3),
        "ll_delta_per_start": round((ll_m - ll_b) / n, 5),
        "mae_b": round(sum(r["ae_b"] for r in rows) / n, 4),
        "mae_m": round(sum(r["ae_m"] for r in rows) / n, 4),
    }
    out["mae_delta"] = round(out["mae_m"] - out["mae_b"], 4)   # negative = sharper
    dates = sorted({r["date"] for r in rows})
    periods = []
    if len(dates) >= n_periods:
        chunk = max(1, len(dates) // n_periods)
        bounds = [dates[i * chunk] for i in range(n_periods)]
        for i in range(n_periods):
            lo = bounds[i]
            hi = bounds[i + 1] if i + 1 < len(bounds) else None
            sub = [r for r in rows if r["date"] >= lo and (hi is None or r["date"] < hi)]
            if not sub:
                continue
            d = sum(r["ll_m"] for r in sub) - sum(r["ll_b"] for r in sub)
            periods.append({"period": f"{lo}..{'end' if hi is None else hi}",
                            "n": len(sub), "ll_delta_total": round(d, 3),
                            "treatment_better": d > 0})
    out["periods"] = periods
    out["periods_better"] = sum(1 for p in periods if p["treatment_better"])
    out["periods_total"] = len(periods)
    return out


def evaluate(starts, team_bat, framing, train_frac=TRAIN_FRAC,
             min_prior_starts=MIN_PRIOR_STARTS, n_periods=N_PERIODS):
    """Full experiment: contexts -> temporal split -> tune w on TRAIN -> score
    FRAMING vs baseline on HOLDOUT. Also measures the catcher-join coverage."""
    ctxs = [c for c in build_contexts(starts, team_bat)
            if c["n_prior"] >= min_prior_starts and c["bf"] > 0]
    if not ctxs:
        return {"error": "no scorable starts"}

    matched = sum(1 for c in ctxs
                  if framing_delta(c["catcher"], c["season"], framing)[1])
    coverage = matched / len(ctxs)

    all_dates = sorted({c["date"] for c in ctxs})
    cutoff = all_dates[min(int(len(all_dates) * train_frac), len(all_dates) - 1)]
    train = [c for c in ctxs if c["date"] < cutoff]
    hold = [c for c in ctxs if c["date"] >= cutoff]
    if not train or not hold:
        return {"error": "empty train or holdout after split"}

    tuned = tune_framing(train, framing)
    res = _score_holdout(hold, tuned["param"]["w"], framing, n_periods)

    report = {
        "cutoff": cutoff.isoformat(),
        "n_train": len(train), "n_holdout": len(hold),
        "framing_coverage": round(coverage, 4),
        "coverage_ok": coverage >= MIN_COVERAGE,
        "constants": {"reg_bf": REG_BF, "min_prior_starts": min_prior_starts,
                      "min_opp_pa": MIN_OPP_PA, "w_opp_baseline": W_OPP,
                      "train_frac": train_frac, "train_win_margin": TRAIN_WIN_MARGIN,
                      "min_coverage": MIN_COVERAGE, "fr_w_grid": FR_W_GRID},
        "tuned": {"param": tuned["param"],
                  "train_dll_per_start": round(tuned["train_dll_per_start"], 5),
                  "train_win": tuned["train_win"]},
        "holdout": res,
    }
    report["verdict"] = _verdict(report)
    return report


def _verdict(report):
    res = report["holdout"]
    if res.get("n", 0) == 0:
        return "INSUFFICIENT DATA"
    trained_win = report["tuned"]["train_win"]
    improved = res["ll_delta_total"] > 0
    robust = res["periods_total"] > 0 and \
        res["periods_better"] >= (res["periods_total"] + 1) // 2
    if trained_win and improved and robust:
        v = "HELPS (robust)"
    elif trained_win and improved:
        v = "MARGINAL (holdout up, not robust across periods)"
    elif not trained_win:
        v = "NO EDGE ON TRAIN (w tuned to ~neutral)"
    else:
        v = "DOES NOT HELP on holdout"
    if not report["coverage_ok"]:
        v += " — BUT JOIN COVERAGE TOO LOW, VERDICT INVALID"
    return v


def print_report(report):
    print("\n" + "=" * 78)
    print("CATCHER-FRAMING EXPERIMENT — HOLDOUT RESULT")
    print("(baseline = prior-rate * BF * opp-whiff(w=0.6); treatment *= framing)")
    print("=" * 78)
    if "error" in report:
        print("  ERROR:", report["error"])
        return
    print(f"  train n={report['n_train']}  holdout n={report['n_holdout']}  "
          f"cutoff={report['cutoff']}")
    cov = report["framing_coverage"]
    print(f"  catcher->framing join coverage: {cov:.1%} of scorable starts "
          f"(unmatched = neutral 1.0)")
    if not report["coverage_ok"]:
        print("  " + "!" * 74)
        print(f"  !! JOIN COVERAGE {cov:.1%} < {MIN_COVERAGE:.0%} — THE NAME JOIN IS TOO "
              f"WEAK. A poor join")
        print("  !! invalidates this test: too many starts scored with a neutral "
              "placeholder.")
        print("  !! Fix the savant<->statsapi name matching before trusting any verdict.")
        print("  " + "!" * 74)
    t = report["tuned"]
    res = report["holdout"]
    print(f"  FRAMING tuned={t['param']}  train dLL/start="
          f"{t['train_dll_per_start']:+.5f}  train_win={t['train_win']}")
    print(f"      holdout LL delta {res['ll_delta_total']:+.2f} total "
          f"({res['ll_delta_per_start']:+.5f}/start, n={res['n']})  "
          f"MAE {res['mae_delta']:+.4f}  "
          f"robust {res['periods_better']}/{res['periods_total']} periods")
    for p in res["periods"]:
        flag = "treatment" if p["treatment_better"] else "baseline "
        print(f"        period {p['period']:<26} n={p['n']:<4} "
              f"LL delta {p['ll_delta_total']:+.2f} -> {flag} better")
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  FRAMING  {report['verdict']}")
    if report["verdict"].startswith("HELPS") and report["coverage_ok"]:
        print(f"\n  => The starting catcher's prior-season framing robustly sharpens "
              f"the K projection (w={t['param']['w']}).")
    else:
        print("\n  => No robust, valid framing edge over the validated baseline. "
              "Leave mlb_kprops.py unchanged.")
    print("=" * 78)


# ===========================================================================
# DATA PULL 1/2 — Baseball Savant catcher-framing leaderboard CSV.
# Tolerant of column drift: prints the header actually received, auto-detects
# the name column(s) and a strike-rate (preferred) / runs (fallback) column.
# ===========================================================================
SAVANT_URL = ("https://baseballsavant.mlb.com/leaderboard/catcher-framing"
              "?year={year}&min=q&sort=4,1&csv=true")
UA = {"User-Agent": "Mozilla/5.0 (MLBTool framing experiment)"}


def _http_get(url, tries=3, timeout=30):
    import urllib.request
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def _find_col(low_header, exact=(), contains=()):
    for i, h in enumerate(low_header):
        if h in exact:
            return i
    for i, h in enumerate(low_header):
        if any(all(tok in h for tok in c.split()) for c in contains):
            return i
    return None


def parse_framing_csv(text, year):
    """CSV text -> (lookup_table, info). Table maps normalized-name keys to
    strike-rate-above-average as a FRACTION (e.g. +0.015 = +1.5 points)."""
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    if not rows:
        raise ValueError(f"savant {year}: empty CSV")
    header = [h.strip() for h in rows[0]]
    print(f"  savant {year} header actually received: {header}")
    low = [h.lower() for h in header]

    i_name = _find_col(low, exact=("last_name, first_name", "name", "player_name",
                                   "catcher", "entity_name", "player"))
    i_last = _find_col(low, exact=("last_name",))
    i_first = _find_col(low, exact=("first_name",))
    if i_name is None and (i_last is None or i_first is None):
        raise ValueError(f"savant {year}: no name column in header {header}")

    i_rate = _find_col(low, exact=("strike_rate",), contains=("strike rate",))
    i_runs = _find_col(low, contains=("runs",)) if i_rate is None else None
    if i_rate is None and i_runs is None:
        raise ValueError(f"savant {year}: no strike_rate/runs column in {header}")
    i_n = _find_col(low, exact=("n_called_pitches",), contains=("pitches",))

    recs = []
    for r in rows[1:]:
        if len(r) < len(header):
            continue
        if i_name is not None:
            nm = r[i_name].strip()
        else:
            nm = f"{r[i_last].strip()}, {r[i_first].strip()}"
        try:
            v = float(r[i_rate if i_rate is not None else i_runs])
        except (ValueError, TypeError):
            continue
        try:
            w = float(r[i_n]) if i_n is not None else 1.0
        except (ValueError, TypeError):
            w = 1.0
        if nm:
            recs.append((nm, v, max(w, 1.0)))
    if not recs:
        raise ValueError(f"savant {year}: no parseable rows")

    wsum = sum(w for _, _, w in recs)
    mean = sum(v * w for _, v, w in recs) / wsum
    if i_rate is not None:
        # strike_rate is in percent points -> delta as fraction
        entries = [(nm, (v - mean) / 100.0) for nm, v, _ in recs]
        metric = header[i_rate]
    else:
        # runs fallback: standardize, map 1 std to RUNS_TO_RATE_STD rate points
        var = sum(w * (v - mean) ** 2 for _, v, w in recs) / wsum
        std = math.sqrt(var) or 1.0
        entries = [(nm, (v - mean) / std * RUNS_TO_RATE_STD) for nm, v, _ in recs]
        metric = header[i_runs] + " (std-mapped)"
    table = table_from_savant_rows(entries)
    print(f"  savant {year}: {len(recs)} catchers via column '{metric}' "
          f"(mean {mean:.3f})")
    return table, {"year": year, "n_catchers": len(recs), "metric": metric,
                   "header": header}


def pull_framing_year(year):
    raw = _http_get(SAVANT_URL.format(year=year))
    return parse_framing_csv(raw.decode("utf-8-sig", "replace"), year)


# ===========================================================================
# DATA PULL 2/2 — statsapi (same endpoints as production / kfactors). Extends
# the kfactors pull with each side's STARTING CATCHER. Never fakes data.
# ===========================================================================
API = "https://statsapi.mlb.com/api/v1"


def _get(url, tries=3):
    import urllib.request
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return json.loads(r.read())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def _final_games(date):
    j = _get(f"{API}/schedule?sportId=1&date={date.isoformat()}")
    out = []
    for dd in j.get("dates", []):
        for g in dd.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                out.append(g["gamePk"])
    return out


def _starting_catcher(team):
    """Starting catcher for one boxscore side: the position-'C' player holding a
    battingOrder slot. A true starter's battingOrder is a multiple of 100
    (100..900); substitutes carry 101/205/etc. Prefer the multiple-of-100 'C';
    fall back to the lowest-order 'C' with any battingOrder."""
    best = None
    for pdata in (team.get("players") or {}).values():
        pos = ((pdata.get("position") or {}).get("abbreviation") or "").upper()
        if pos != "C":
            allpos = pdata.get("allPositions") or []
            if not any((p.get("abbreviation") or "").upper() == "C" for p in allpos):
                continue
        bo_raw = pdata.get("battingOrder")
        if bo_raw in (None, ""):
            continue
        try:
            bo = int(str(bo_raw))
        except ValueError:
            continue
        name = ((pdata.get("person") or {}).get("fullName") or "").strip()
        if not name:
            continue
        if 100 <= bo <= 900 and bo % 100 == 0:
            return name
        if best is None or bo < best[0]:
            best = (bo, name)
    return best[1] if best else ""


def _team_batting_totals(box, side):
    """(bat_so, bat_pa) for one side. Primary: teamStats.batting; PA falls back
    to component sum; final fallback: the OTHER side's pitching battersFaced."""
    team = box.get("teams", {}).get(side, {})
    bat = ((team.get("teamStats") or {}).get("batting")) or {}
    so = int(bat.get("strikeOuts", 0) or 0)
    pa = int(bat.get("plateAppearances", 0) or 0)
    if pa <= 0:
        pa = sum(int(bat.get(k, 0) or 0) for k in
                 ("atBats", "baseOnBalls", "hitByPitch", "sacBunts", "sacFlies",
                  "catchersInterference"))
    if pa <= 0 or so <= 0:
        other = "away" if side == "home" else "home"
        pso = pbf = 0
        for pdata in (box.get("teams", {}).get(other, {}).get("players") or {}).values():
            ps = (pdata.get("stats") or {}).get("pitching") or {}
            if ps:
                pso += int(ps.get("strikeOuts", 0) or 0)
                pbf += int(ps.get("battersFaced", 0) or 0)
        if pa <= 0:
            pa = pbf
        if so <= 0:
            so = pso
    return so, pa


def _parse_boxscore(box, game_date):
    """Return (starts, bat_rows) from one boxscore.
    starts   : [{date, pitcher, so, bf, home, team, opp, catcher}] per side's
               STARTER — catcher = the starting catcher on the PITCHER'S OWN side
    bat_rows : [{date, team, bat_so, bat_pa}] one per side."""
    names = {}
    for side in ("home", "away"):
        names[side] = (box.get("teams", {}).get(side, {}).get("team") or {}).get("name", "")
    if not names["home"] or not names["away"]:
        return None, None
    starts, bat_rows = [], []
    for side in ("home", "away"):
        other = "away" if side == "home" else "home"
        team = box.get("teams", {}).get(side, {})
        players = team.get("players", {})
        order = team.get("pitchers", []) or []
        starter_pid = order[0] if order else None
        catcher = _starting_catcher(team)
        for pid, pdata in players.items():
            ps = (pdata.get("stats") or {}).get("pitching") or {}
            if not ps:
                continue
            bf = int(ps.get("battersFaced", 0) or 0)
            so = int(ps.get("strikeOuts", 0) or 0)
            person = pdata.get("person") or {}
            is_starter = (person.get("id") == starter_pid) or \
                         int(ps.get("gamesStarted", 0) or 0) == 1
            if is_starter and bf > 0:
                starts.append({"date": game_date, "pitcher": person.get("fullName", ""),
                               "so": so, "bf": bf, "home": side == "home",
                               "team": names[side], "opp": names[other],
                               "catcher": catcher})
        bso, bpa = _team_batting_totals(box, side)
        if bpa > 0:
            bat_rows.append({"date": game_date, "team": names[side],
                             "bat_so": bso, "bat_pa": bpa})
    if len(bat_rows) < 2:
        return None, None
    return starts, bat_rows


def pull_range(start_date, end_date):
    """Pull (starts, team_bat) across a date range from statsapi.
    Raises on the first network failure so the caller can degrade cleanly."""
    starts, team_bat = [], []
    day = start_date
    ngames = 0
    while day <= end_date:
        pks = _final_games(day)
        for pk in pks:
            try:
                box = _get(f"{API}/game/{pk}/boxscore")
            except Exception:
                continue
            st, br = _parse_boxscore(box, day)
            if br is None:
                continue
            starts.extend(st)
            team_bat.extend(br)
            ngames += 1
            time.sleep(0.15)
        print(f"  {day}: {len(pks)} finals, cum {ngames} usable games / {len(starts)} starts")
        day += dt.timedelta(days=1)
        time.sleep(0.2)
    return starts, team_bat


# ---- dataset cache ---------------------------------------------------------
def framing_years_needed(start_date, end_date):
    return sorted({y - 1 for y in range(start_date.year, end_date.year + 1)})


def save_dataset(start_date, end_date, starts, team_bat, framing, framing_info):
    os.makedirs(DATA, exist_ok=True)
    payload = {
        "pulled_at": dt.datetime.utcnow().isoformat() + "Z",
        "start": start_date.isoformat(), "end": end_date.isoformat(),
        "starts": [{**s, "date": s["date"].isoformat()} for s in starts],
        "team_bat": [{**t, "date": t["date"].isoformat()} for t in team_bat],
        "framing": {str(y): tab for y, tab in framing.items()},
        "framing_info": framing_info,
    }
    with open(DATASET_PATH, "w") as f:
        json.dump(payload, f)
    print(f"dataset cached -> {DATASET_PATH}")


def load_dataset(start_date, end_date):
    """(starts, team_bat, framing) from cache if it covers [start, end] and all
    needed framing years, else None."""
    if not os.path.exists(DATASET_PATH):
        return None
    try:
        with open(DATASET_PATH) as f:
            payload = json.load(f)
        cs = dt.date.fromisoformat(payload["start"])
        ce = dt.date.fromisoformat(payload["end"])
        if cs > start_date or ce < end_date:
            print(f"cache covers {cs}..{ce} — does not cover requested range, will re-pull")
            return None
        framing = {int(y): tab for y, tab in (payload.get("framing") or {}).items()}
        need = framing_years_needed(start_date, end_date)
        if any(y not in framing for y in need):
            print(f"cache missing framing year(s) {need} — will re-pull")
            return None
        starts = [{**s, "date": dt.date.fromisoformat(s["date"])}
                  for s in payload["starts"]]
        team_bat = [{**t, "date": dt.date.fromisoformat(t["date"])}
                    for t in payload["team_bat"]]
        starts = [s for s in starts if start_date <= s["date"] <= end_date]
        team_bat = [t for t in team_bat if start_date <= t["date"] <= end_date]
        print(f"using cached dataset {DATASET_PATH} ({len(starts)} starts, "
              f"{len(team_bat)} team-game batting rows, framing years "
              f"{sorted(framing)})")
        return starts, team_bat, framing
    except Exception as e:
        print(f"cache unreadable ({type(e).__name__}) — will re-pull")
        return None


def run_live(start_date, end_date, train_frac):
    """Cache-first; else attempt the pull; degrade to a clear message (exit 0)
    if statsapi/savant is unreachable (expected in this sandbox)."""
    print("=" * 78)
    print(f"CATCHER-FRAMING EXPERIMENT — {start_date} .. {end_date}")
    print(f"framing tables (prior seasons, leak rule): "
          f"{framing_years_needed(start_date, end_date)}")
    print("=" * 78)
    data = load_dataset(start_date, end_date)
    if data is None:
        # probe egress first so we fail fast & clean
        try:
            _get(f"{API}/schedule?sportId=1&date={start_date.isoformat()}", tries=1)
        except Exception as e:
            print("\nstatsapi UNREACHABLE — run on GitHub Actions.")
            print(f"  (probe error: {type(e).__name__}: {str(e)[:120]})")
            print("  This is EXPECTED in the cloud sandbox (egress is blocked).")
            print("  Every other MLB workflow proves statsapi IS reachable on Actions.")
            print("  Action command:  python3 mlb_framing_experiment.py --start "
                  f"{start_date} --end {end_date}")
            return 0
        framing, framing_info = {}, []
        try:
            for y in framing_years_needed(start_date, end_date):
                tab, info = pull_framing_year(y)
                framing[y] = tab
                framing_info.append(info)
        except Exception as e:
            print("\nbaseballsavant UNREACHABLE — run on GitHub Actions.")
            print(f"  (probe error: {type(e).__name__}: {str(e)[:120]})")
            print("  This is EXPECTED in the cloud sandbox (egress is blocked).")
            print("  The savant-probe workflow proves savant IS reachable on Actions.")
            print("  Action command:  python3 mlb_framing_experiment.py --start "
                  f"{start_date} --end {end_date}")
            return 0
        starts, team_bat = pull_range(start_date, end_date)
        print(f"\npulled {len(starts)} starts / {len(team_bat)} team-game batting rows")
        if not starts:
            print("no usable starts pulled — cannot run the experiment.")
            return 0
        n_c = sum(1 for s in starts if s.get("catcher"))
        print(f"starting catcher identified for {n_c}/{len(starts)} starts")
        save_dataset(start_date, end_date, starts, team_bat, framing, framing_info)
    else:
        starts, team_bat, framing = data
        if not starts:
            print("cached dataset empty for range — cannot run the experiment.")
            return 0

    report = evaluate(starts, team_bat, framing, train_frac=train_frac)
    os.makedirs(DATA, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print_report(report)
    return 0


# ===========================================================================
# OFFLINE SELFTEST — NO NETWORK. Verifies (a) the prior-season leak rule is
# enforced AND load-bearing, (b) savant<->statsapi name normalization,
# (c) Poisson math against hand values, (d) planted framing effect recovered
# on synthetic data + null control clean.
# ===========================================================================
def _pois(rng, lam):
    """Knuth Poisson sampler (no numpy dependency in the selftest)."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _synth_bat_rows(date, teams, rng):
    rows = []
    for t in teams:
        pa = rng.randint(34, 42)
        so = min(pa, _pois(rng, 0.22 * pa))
        rows.append({"date": date, "team": t, "bat_so": so, "bat_pa": pa})
    return rows


def _synth_season(rng, catcher_effect_w, cat_delta, teams, cat_of_team,
                  n_days=160, year=2025):
    """Synthetic season: static pitcher talent; each start's K rate is scaled by
    (1 + catcher_effect_w * true_delta[own catcher]). catcher_effect_w=0 -> null."""
    pit_rate = {f"P{i}": rng.uniform(0.16, 0.30) for i in range(24)}
    starts, bat = [], []
    day0 = dt.date(year, 4, 1)
    for gday in range(n_days):
        date = day0 + dt.timedelta(days=gday)
        pits = rng.sample(list(pit_rate), 6)
        for j, p in enumerate(pits):
            t_own, t_opp = rng.sample(teams, 2)
            c = cat_of_team[t_own]
            mult = 1.0 + catcher_effect_w * cat_delta[c]
            bf = rng.randint(18, 30)
            so = min(bf, _pois(rng, pit_rate[p] * bf * mult))
            starts.append({"date": date, "pitcher": p, "so": so, "bf": bf,
                           "home": j % 2 == 0, "team": t_own, "opp": t_opp,
                           "catcher": c})
            bat.extend(_synth_bat_rows(date, [t_own, t_opp], rng))
    return starts, bat


def selftest():
    import random

    # ---------- (c) POISSON MATH exact on hand-computed fixtures ----------
    # log P(X=3 | lam=2.5) = 3*ln2.5 - 2.5 - ln(3!) = -1.542887
    assert abs(poisson_logpmf(3, 2.5) - (-1.5428868)) < 1e-6, poisson_logpmf(3, 2.5)
    # P(X=0 | lam) = e^-lam -> log = -lam
    assert abs(poisson_logpmf(0, 4.0) - (-4.0)) < 1e-9
    # log P(X=6 | lam=6) = 6*ln6 - 6 - ln720 = -1.828694
    assert abs(poisson_logpmf(6, 6.0) - (-1.8286943)) < 1e-6, poisson_logpmf(6, 6.0)
    lls = [(poisson_logpmf(6, L), L) for L in (3.0, 5.0, 6.0, 7.0, 9.0)]
    assert max(lls)[1] == 6.0, lls
    print("  [c] Poisson logpmf: exact on hand-computed fixtures, peaks at lam=k")

    # ---------- (b) NAME NORMALIZATION savant <-> statsapi ----------
    assert name_key("Realmuto, J.T.") == name_key("J.T. Realmuto") == "jtrealmuto"
    assert name_key("Sánchez, Gary") == name_key("Gary Sanchez") == "garysanchez"
    assert name_key("Peña, Salvador") == name_key("Salvador Pena")
    assert name_key("De La Cruz, Elly") == name_key("Elly De La Cruz")
    assert name_key("d'Arnaud, Travis") == name_key("Travis d'Arnaud")
    # both directions through a real table build + lookup, incl. the fallback
    tab = table_from_savant_rows([("Realmuto, J.T.", 0.015),
                                  ("Sánchez, Gary", -0.010)])
    fr = {2024: tab}
    assert framing_delta("J.T. Realmuto", 2025, fr) == (0.015, True)
    assert framing_delta("Gary Sanchez", 2025, fr) == (-0.010, True)      # accent dropped
    assert framing_delta("Gary Sánchez", 2025, fr) == (-0.010, True)      # accent kept
    assert framing_delta("Some Unknown", 2025, fr) == (0.0, False)
    # last-name + first-initial fallback catches middle-name/initial drift
    assert framing_delta("J. Realmuto", 2025, fr) == (0.015, True)
    print("  [b] name join: 'Realmuto, J.T.' <-> 'J.T. Realmuto', accents stripped, "
          "fallback key works, unknowns neutral")

    # starting-catcher extraction from a boxscore-shaped side (offline fixture)
    side = {"players": {
        "IDp1": {"person": {"fullName": "Some Pitcher"},
                 "position": {"abbreviation": "P"}, "battingOrder": "900"},
        "IDc2": {"person": {"fullName": "Backup Guy"},
                 "position": {"abbreviation": "C"}, "battingOrder": "205"},
        "IDc1": {"person": {"fullName": "Starter Guy"},
                 "position": {"abbreviation": "C"}, "battingOrder": "200"},
        "IDdh": {"person": {"fullName": "Bench Catcher"},
                 "position": {"abbreviation": "C"}},   # no battingOrder -> DNP
    }}
    assert _starting_catcher(side) == "Starter Guy", _starting_catcher(side)
    print("  [b] starting catcher = position 'C' with battingOrder slot "
          "(200 beats sub 205; DNP ignored)")

    # ---------- (a) LEAK RULE: prior-season framing ONLY, and load-bearing ----------
    # The same catcher has +4.0 rate points in the 2024 table and -4.0 in the
    # 2025 table. A 2025 start MUST read 2024. If the same-season 2025 value
    # leaked in, the projection would cross a 5.2 K line the other way AND the
    # Poisson score of a hot night would flip in favor of the wrong model.
    fr2 = {2024: table_from_savant_rows([("Framer, Frank", +0.040)]),
           2025: table_from_savant_rows([("Framer, Frank", -0.040)])}
    d, m = framing_delta("Frank Framer", 2025, fr2)
    assert m and d == +0.040, (d, m)                    # 2025 start -> 2024 table
    d24, m24 = framing_delta("Frank Framer", 2024, fr2)
    assert not m24, (d24, m24)                          # 2024 start -> 2023 (absent)
    lam_b = 5.0
    w = 2.0
    lam_prior = lam_b * (1 + w * (+0.040))              # 5.40  (correct, prior season)
    lam_leak = lam_b * (1 + w * (-0.040))               # 4.60  (leaked same-season)
    assert lam_prior > 5.2 > lam_leak, (lam_prior, lam_leak)   # flips a 5.2 line
    assert poisson_logpmf(7, lam_prior) > poisson_logpmf(7, lam_leak)
    print(f"  [a] leak rule enforced: 2025 start reads the 2024 table (+0.040, not "
          f"-0.040); leaking same-season would flip a 5.2 line "
          f"({lam_prior:.2f} vs {lam_leak:.2f})")

    # ---------- (d) PLANTED FRAMING EFFECT recovered + leak flip + null ----------
    teams = [f"T{i}" for i in range(12)]
    # savant-format names WITH accents so the full pipeline exercises the join
    cat_savant = {t: (f"Cátcher{i}, Fírst{i}") for i, t in enumerate(teams)}
    cat_statsapi = {t: (f"Fírst{i} Cátcher{i}") for i, t in enumerate(teams)}
    deltas = [0.030, 0.024, 0.018, 0.012, 0.006, 0.0,
              0.0, -0.006, -0.012, -0.018, -0.024, -0.030]
    cat_delta = {cat_statsapi[t]: deltas[i] for i, t in enumerate(teams)}
    framing_prior = {2024: table_from_savant_rows(
        [(cat_savant[t], deltas[i]) for i, t in enumerate(teams)])}

    rng = random.Random(20260724)
    starts, bat = _synth_season(rng, 2.0, cat_delta, teams,
                                {t: cat_statsapi[t] for t in teams})
    rep = evaluate(starts, bat, framing_prior, train_frac=0.5, min_prior_starts=3)
    assert rep["framing_coverage"] == 1.0, rep["framing_coverage"]
    assert rep["tuned"]["param"]["w"] >= 1.0, rep["tuned"]
    assert rep["tuned"]["train_win"], rep["tuned"]
    h = rep["holdout"]
    assert h["n"] > 150, h["n"]
    assert h["ll_delta_total"] > 0, h
    assert h["mae_delta"] < 0, h
    print(f"  [d] planted framing effect recovered through the savant-format join: "
          f"tuned w={rep['tuned']['param']['w']} holdout LL {h['ll_delta_total']:+.1f} "
          f"({h['ll_delta_per_start']:+.4f}/start, n={h['n']}), MAE {h['mae_delta']:+.3f}, "
          f"coverage {rep['framing_coverage']:.0%}")

    # same data, but the table a LEAKY same-season lookup would have grabbed
    # (signs flipped): the tuner must refuse it (w -> 0) — the answer FLIPS
    # from 'helps' to 'no edge', proving the leak rule is load-bearing.
    framing_flipped = {2024: table_from_savant_rows(
        [(cat_savant[t], -deltas[i]) for i, t in enumerate(teams)])}
    rep_leak = evaluate(starts, bat, framing_flipped, train_frac=0.5, min_prior_starts=3)
    assert rep_leak["tuned"]["param"]["w"] == 0.0, rep_leak["tuned"]
    assert not rep_leak["tuned"]["train_win"], rep_leak["tuned"]
    assert rep_leak["holdout"]["ll_delta_total"] == 0.0, rep_leak["holdout"]
    assert rep["tuned"]["train_win"] != rep_leak["tuned"]["train_win"]
    print(f"  [a+d] wrong-season (flipped) table -> tuner refuses (w=0), verdict "
          f"flips: '{rep['verdict']}' vs '{rep_leak['verdict']}'")

    # NULL CONTROL: table present but Ks generated with NO catcher effect ->
    # no free lift allowed.
    rng0 = random.Random(31337)
    starts0, bat0 = _synth_season(rng0, 0.0, cat_delta, teams,
                                  {t: cat_statsapi[t] for t in teams})
    rep0 = evaluate(starts0, bat0, framing_prior, train_frac=0.5, min_prior_starts=3)
    dps = rep0["holdout"]["ll_delta_per_start"]
    assert abs(dps) < 0.01, dps
    print(f"  [d] null control: no planted effect -> tuned w="
          f"{rep0['tuned']['param']['w']}, holdout LL/start delta {dps:+.5f} "
          f"(within +/-0.01 of zero)")

    # coverage alarm fires when the join is broken (<70% matched)
    rep_cov = evaluate(starts, bat, {2024: table_from_savant_rows(
        [(cat_savant[teams[0]], deltas[0])])}, train_frac=0.5, min_prior_starts=3)
    assert not rep_cov["coverage_ok"], rep_cov["framing_coverage"]
    assert "INVALID" in rep_cov["verdict"], rep_cov["verdict"]
    print(f"  [b] broken join (coverage {rep_cov['framing_coverage']:.0%}) -> "
          f"verdict flagged INVALID as required")

    # savant CSV parser tolerates drift and both column styles
    csv_a = ('"last_name, first_name",player_id,year,n_called_pitches,strike_rate,'
             "runs_extra_strikes\n"
             '"Realmuto, J.T.",592663,2024,4000,48.5,9\n'
             '"Sánchez, Gary",596142,2024,3000,44.5,-8\n')
    tab_a, info_a = parse_framing_csv(csv_a, 2024)
    da, ma = framing_delta("J.T. Realmuto", 2025, {2024: tab_a})
    db, mb = framing_delta("Gary Sanchez", 2025, {2024: tab_a})
    assert ma and mb and da > 0 > db, (da, db)
    # weighted mean = (48.5*4000 + 44.5*3000)/7000 = 46.7857; delta_a = +0.017143
    assert abs(da - 0.0171428) < 1e-6, da
    csv_b = ("first_name,last_name,catcher_framing_runs\n"
             "J.T.,Realmuto,12\nGary,Sánchez,-10\n")
    tab_b, info_b = parse_framing_csv(csv_b, 2024)
    db1, _ = framing_delta("J.T. Realmuto", 2025, {2024: tab_b})
    db2, _ = framing_delta("Gary Sanchez", 2025, {2024: tab_b})
    assert db1 > 0 > db2, (db1, db2)
    print("  [b] savant CSV parser: combined & split name columns, strike_rate "
          "and runs-fallback both parse; hand-checked weighted-mean delta exact")

    print("FRAMING SELFTEST PASS — leak rule enforced & load-bearing, name join "
          "verified both ways, Poisson math exact, planted effect recovered, "
          "null control clean, coverage alarm armed")
    return 0


# ===========================================================================
def _date(s):
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def main(argv):
    ap = argparse.ArgumentParser(
        description="Catcher-framing K-props experiment (standalone; production untouched).")
    ap.add_argument("--selftest", action="store_true", help="offline synthetic tests, no network")
    ap.add_argument("--start", type=_date, default=None, help="range start YYYY-MM-DD")
    ap.add_argument("--end", type=_date, default=None, help="range end YYYY-MM-DD")
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC,
                    help="earlier fraction of dates used to tune; the rest is the scored holdout")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    start = args.start or dt.date(dt.date.today().year - 1, 4, 1)
    end = args.end or dt.date(dt.date.today().year - 1, 9, 28)
    return run_live(start, end, args.train_frac)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
