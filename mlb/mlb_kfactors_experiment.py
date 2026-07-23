#!/usr/bin/env python3
"""
mlb_kfactors_experiment.py — WHICH K-FACTORS SHARPEN PITCHER-STRIKEOUT PROPS?
================================================================================
STANDALONE EXPERIMENT. Does not import or modify mlb_kprops.py or any production
file. It only builds evidence and prints a VERDICT.

THE HYPOTHESES
--------------
The validated baseline (same one the umpire experiment used) projects a start's
strikeouts as Poisson(lambda) with
    lambda_b = pitcher_season_rate(strictly-prior starts, league-regressed) * BF.
Four candidate refinements, each buildable LEAK-FREE from the same boxscores:

  H1 RECENT FORM     exponentially-weighted K/BF over the pitcher's prior starts
                     (recent starts count more). The decay is the blend/window
                     knob — decay=1.0 IS the baseline — tuned on TRAIN only.
  H2 HOME/AWAY       the pitcher's own K/BF split by venue (home vs away),
                     shrunk toward his overall rate; shrink tuned on TRAIN.
  H3 DAYS REST       multiplicative adjustment by days since his previous start
                     (short <=4 / standard 5 / long >=6), bucket ratios measured
                     on TRAIN, application weight tuned on TRAIN.
  H4 OPPONENT WHIFF  opposing team's batting K% from strictly-prior games vs
                     league, factor = 1 + w*(opp/league - 1). Production
                     mlb_kprops.py fixes w=0.50; here w is TUNED on TRAIN.

METHOD (non-negotiable)
-----------------------
Chronological walk-forward. Every feature comes from games dated STRICTLY BEFORE
the start being predicted — a start can never see itself or anything later.
The timeline is split by date: the earlier TRAIN portion tunes every weight; the
later HOLDOUT portion is scored once, per hypothesis and for a COMBINED model
that keeps only the train-winning components. Scoring: Poisson log-likelihood
of the actual K total per start + MAE, with per-period robustness buckets.

NETWORK
-------
The pull hits statsapi (schedule + boxscores), the same endpoints production
already uses. In THIS sandbox egress to statsapi is blocked — EXPECTED. On
GitHub Actions statsapi is reachable (every other MLB workflow proves it). If
blocked, the pull prints a clear 'run on Actions' message and exits 0. It never
fakes numbers. The pulled dataset is cached to mlb/data/kfactors_dataset.json
so re-runs (and the committed Action artifact) don't re-pull.

RUN
---
  python3 mlb_kfactors_experiment.py --selftest        # offline, no network, must pass
  python3 mlb_kfactors_experiment.py                   # live pull + verdict (on Actions)
  python3 mlb_kfactors_experiment.py --start 2025-04-01 --end 2025-06-30
"""
import os, sys, json, math, time, argparse, datetime as dt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASET_PATH = os.path.join(DATA, "kfactors_dataset.json")
REPORT_PATH = os.path.join(DATA, "kfactors_experiment.json")

# ---- experiment-local constants (production untouched) ----------------------
LG_K_PER_BF = 0.222     # league K per batter faced fallback (~league K%)
REG_BF = 100.0          # league BF added to a pitcher's rate (same shrink as baseline)
MIN_PRIOR_STARTS = 3    # a pitcher needs this many prior starts to be scored
MIN_OPP_PA = 300        # opponent needs this many prior batting PAs for H4 (else neutral)
TRAIN_FRAC = 0.5        # earlier fraction of dates = TRAIN (tuning); later = HOLDOUT
N_PERIODS = 3           # holdout robustness buckets
TRAIN_WIN_MARGIN = 5e-4 # train mean-LL/start improvement a component must clear
                        # to earn a slot in the COMBINED model

# tuning grids — every grid contains its neutral (baseline-identical) setting
H1_DECAY_GRID = [1.0, 0.97, 0.94, 0.90, 0.85, 0.80, 0.70, 0.60]
H2_VSHRINK_GRID = [None, 800.0, 400.0, 200.0, 100.0, 50.0]   # None = baseline
H3_W_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
H3_PSEUDO_K = 150.0     # pseudo-Ks shrinking a rest-bucket ratio toward 1.0
H4_W_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

VENUE_LO, VENUE_HI = 0.85, 1.15   # bound H2 factor
REST_LO, REST_HI = 0.90, 1.10     # bound H3 factor
OPP_LO, OPP_HI = 0.80, 1.20       # bound H4 factor


def norm(s):
    if not isinstance(s, str):
        return ""
    return "".join(c for c in s.lower() if c.isalnum())


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
# LEAK-FREE CONTEXT BUILDER
# ---------------------------------------------------------------------------
# One chronological pass over (starts, team_bat) in date order. For each start
# it snapshots EVERYTHING a model may look at, built ONLY from rows dated
# strictly < the start's date. State is folded AFTER a date is predicted, so a
# start can never see itself, its own game, or anything later. All tuning /
# scoring below is pure math on these frozen contexts — the leak barrier lives
# in exactly one place.
# ===========================================================================
def build_contexts(starts, team_bat):
    """
    starts   : [{date(dt.date), pitcher, so, bf, home(bool), team, opp}]
    team_bat : [{date(dt.date), team, bat_so, bat_pa}]  (one row per team per game)
    Returns ctx list (chronological): each ctx carries the start's outcome plus
    strictly-prior pitcher history, league rate, opponent batting totals, rest days.
    """
    all_dates = sorted({s["date"] for s in starts} | {t["date"] for t in team_bat})
    starts_by_date, bat_by_date = {}, {}
    for s in starts:
        starts_by_date.setdefault(s["date"], []).append(s)
    for t in team_bat:
        bat_by_date.setdefault(t["date"], []).append(t)

    pit = {}                 # norm(pitcher) -> [(date, so, bf, home)] chronological
    team_tot = {}            # norm(team) -> [so_cum, pa_cum]
    lg_so = lg_pa = 0        # league cumulative batting SO / PA

    ctxs = []
    for date in all_dates:
        # 1) SNAPSHOT this date's starts from strictly-prior state
        for s in starts_by_date.get(date, []):
            prior = list(pit.get(norm(s["pitcher"]), []))
            oso, opa = team_tot.get(norm(s.get("opp", "")), (0, 0))
            lg = (lg_so / lg_pa) if lg_pa > 0 else LG_K_PER_BF
            ctxs.append({
                "date": date, "pitcher": s["pitcher"], "k": int(s["so"]),
                "bf": int(s["bf"]), "home": bool(s.get("home", False)),
                "opp": s.get("opp", ""),
                "prior": prior,                       # [(date, so, bf, home)]
                "n_prior": len(prior),
                "lg": lg,
                "opp_so": oso, "opp_pa": opa,
                "rest_days": (date - prior[-1][0]).days if prior else None,
            })
        # 2) FOLD this date's rows into state (visible to LATER dates only)
        for s in starts_by_date.get(date, []):
            pit.setdefault(norm(s["pitcher"]), []).append(
                (date, int(s["so"]), int(s["bf"]), bool(s.get("home", False))))
        for t in bat_by_date.get(date, []):
            so, pa = int(t["bat_so"]), int(t["bat_pa"])
            key = norm(t["team"])
            cso, cpa = team_tot.get(key, (0, 0))
            team_tot[key] = (cso + so, cpa + pa)
            lg_so += so
            lg_pa += pa
    return ctxs


# ===========================================================================
# THE MODELS — pure functions of a frozen context (+ tuned params).
# ===========================================================================
def rate_season(ctx):
    """BASELINE rate: K/BF over strictly-prior starts, league-regressed."""
    so = sum(p[1] for p in ctx["prior"])
    bf = sum(p[2] for p in ctx["prior"])
    return (so + REG_BF * ctx["lg"]) / (bf + REG_BF)


def lam_baseline(ctx):
    return rate_season(ctx) * ctx["bf"]


def rate_h1(ctx, decay):
    """H1 RECENT FORM: exponentially-weighted K/BF, most recent start weight 1,
    each older start decayed. decay=1.0 reproduces the baseline exactly. The
    league regression rides on the (shrunken) effective sample, so heavy decay
    also pulls harder toward league — the honest cost of a short memory."""
    prior = ctx["prior"]
    n = len(prior)
    wso = wbf = 0.0
    for i, (_, so, bf, _) in enumerate(prior):
        w = decay ** (n - 1 - i)          # age in starts; newest gets w=1
        wso += w * so
        wbf += w * bf
    return (wso + REG_BF * ctx["lg"]) / (wbf + REG_BF)


def factor_h2(ctx, vshrink):
    """H2 HOME/AWAY: pitcher's same-venue K/BF vs his overall rate, shrunk by
    vshrink pseudo-BF toward overall, bounded. vshrink=None -> factor 1."""
    if vshrink is None:
        return 1.0
    overall = rate_season(ctx)
    vso = sum(p[1] for p in ctx["prior"] if p[3] == ctx["home"])
    vbf = sum(p[2] for p in ctx["prior"] if p[3] == ctx["home"])
    vrate = (vso + vshrink * overall) / (vbf + vshrink)
    return clip(vrate / overall, VENUE_LO, VENUE_HI) if overall > 0 else 1.0


def rest_bucket(rest_days):
    if rest_days is None:
        return "unknown"
    if rest_days <= 4:
        return "short"
    if rest_days == 5:
        return "std"
    return "long"


def factor_h3(ctx, w, ratios):
    """H3 DAYS REST: bucket ratio (measured on TRAIN) applied with weight w."""
    r = ratios.get(rest_bucket(ctx["rest_days"]), 1.0)
    return clip(1.0 + w * (r - 1.0), REST_LO, REST_HI)


def factor_h4(ctx, w):
    """H4 OPPONENT WHIFF: opposing team's strictly-prior batting K% vs league,
    shrink weight w toward 1.0 (production hardcodes w=0.50). Neutral when the
    opponent hasn't logged MIN_OPP_PA prior PAs yet."""
    if ctx["opp_pa"] < MIN_OPP_PA or ctx["lg"] <= 0:
        return 1.0
    raw = (ctx["opp_so"] / ctx["opp_pa"]) / ctx["lg"]
    return clip(1.0 + w * (raw - 1.0), OPP_LO, OPP_HI)


# ===========================================================================
# TUNING (TRAIN ONLY) + HOLDOUT SCORING
# ===========================================================================
def _train_ll(ctxs, lam_fn):
    return sum(poisson_logpmf(c["k"], lam_fn(c)) for c in ctxs)


def rest_ratios_from_train(train_ctxs):
    """Actual-K over baseline-expected-K per rest bucket, on TRAIN starts only,
    shrunk toward 1.0 with H3_PSEUDO_K pseudo-strikeouts."""
    agg = {}
    for c in train_ctxs:
        b = rest_bucket(c["rest_days"])
        k_sum, lam_sum = agg.get(b, (0.0, 0.0))
        agg[b] = (k_sum + c["k"], lam_sum + lam_baseline(c))
    return {b: (k + H3_PSEUDO_K) / (lam + H3_PSEUDO_K) for b, (k, lam) in agg.items()}


def tune_all(train_ctxs):
    """Grid-tune every hypothesis on TRAIN. Each grid contains its neutral
    setting, so best-train >= baseline-train by construction; the reported
    train delta is the margin over neutral."""
    n = max(1, len(train_ctxs))
    base_ll = _train_ll(train_ctxs, lam_baseline)
    tuned = {}

    # H1
    best = max(H1_DECAY_GRID,
               key=lambda d: _train_ll(train_ctxs, lambda c: rate_h1(c, d) * c["bf"]))
    d1 = (_train_ll(train_ctxs, lambda c: rate_h1(c, best) * c["bf"]) - base_ll) / n
    tuned["H1"] = {"param": {"decay": best}, "train_dll_per_start": d1}

    # H2
    best = max(H2_VSHRINK_GRID,
               key=lambda v: _train_ll(train_ctxs,
                                       lambda c: lam_baseline(c) * factor_h2(c, v)))
    d2 = (_train_ll(train_ctxs, lambda c: lam_baseline(c) * factor_h2(c, best))
          - base_ll) / n
    tuned["H2"] = {"param": {"venue_shrink_bf": best}, "train_dll_per_start": d2}

    # H3 (bucket ratios measured on train, then weight tuned on train)
    ratios = rest_ratios_from_train(train_ctxs)
    best = max(H3_W_GRID,
               key=lambda w: _train_ll(train_ctxs,
                                       lambda c: lam_baseline(c) * factor_h3(c, w, ratios)))
    d3 = (_train_ll(train_ctxs, lambda c: lam_baseline(c) * factor_h3(c, best, ratios))
          - base_ll) / n
    tuned["H3"] = {"param": {"w": best, "ratios": {k: round(v, 4) for k, v in ratios.items()}},
                   "train_dll_per_start": d3, "_ratios": ratios}

    # H4
    best = max(H4_W_GRID,
               key=lambda w: _train_ll(train_ctxs,
                                       lambda c: lam_baseline(c) * factor_h4(c, w)))
    d4 = (_train_ll(train_ctxs, lambda c: lam_baseline(c) * factor_h4(c, best))
          - base_ll) / n
    tuned["H4"] = {"param": {"w": best}, "train_dll_per_start": d4}

    for h in tuned.values():
        h["train_win"] = h["train_dll_per_start"] > TRAIN_WIN_MARGIN
    return tuned


def model_lambdas(ctx, tuned, components):
    """lambda for an arbitrary component subset (used for H1..H4 singly and for
    the COMBINED model). Components not in the set contribute nothing."""
    if "H1" in components:
        lam = rate_h1(ctx, tuned["H1"]["param"]["decay"]) * ctx["bf"]
    else:
        lam = lam_baseline(ctx)
    if "H2" in components:
        lam *= factor_h2(ctx, tuned["H2"]["param"]["venue_shrink_bf"])
    if "H3" in components:
        lam *= factor_h3(ctx, tuned["H3"]["param"]["w"], tuned["H3"]["_ratios"])
    if "H4" in components:
        lam *= factor_h4(ctx, tuned["H4"]["param"]["w"])
    return lam


def _score_holdout(hold_ctxs, tuned, components, n_periods=N_PERIODS):
    """Score one model (component subset) vs baseline on the holdout: total &
    per-start Poisson LL delta, MAE delta, per-period robustness."""
    rows = []
    for c in hold_ctxs:
        lam_b = lam_baseline(c)
        lam_m = model_lambdas(c, tuned, components)
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
    # per-period robustness buckets
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


def evaluate(starts, team_bat, train_frac=TRAIN_FRAC,
             min_prior_starts=MIN_PRIOR_STARTS, n_periods=N_PERIODS):
    """Full experiment: contexts -> temporal split -> tune on TRAIN -> score
    each hypothesis AND the combined (train-winners-only) model on HOLDOUT."""
    ctxs = [c for c in build_contexts(starts, team_bat)
            if c["n_prior"] >= min_prior_starts]
    if not ctxs:
        return {"error": "no scorable starts"}
    all_dates = sorted({c["date"] for c in ctxs})
    cutoff = all_dates[min(int(len(all_dates) * train_frac), len(all_dates) - 1)]
    train = [c for c in ctxs if c["date"] < cutoff]
    hold = [c for c in ctxs if c["date"] >= cutoff]
    if not train or not hold:
        return {"error": "empty train or holdout after split"}

    tuned = tune_all(train)
    winners = [h for h in ("H1", "H2", "H3", "H4") if tuned[h]["train_win"]]

    results = {}
    for h in ("H1", "H2", "H3", "H4"):
        results[h] = _score_holdout(hold, tuned, {h}, n_periods)
    results["COMBINED"] = _score_holdout(hold, tuned, set(winners), n_periods)

    report = {
        "cutoff": cutoff.isoformat(),
        "n_train": len(train), "n_holdout": len(hold),
        "constants": {"reg_bf": REG_BF, "min_prior_starts": min_prior_starts,
                      "min_opp_pa": MIN_OPP_PA, "train_frac": train_frac,
                      "train_win_margin": TRAIN_WIN_MARGIN},
        "tuned": {h: {"param": tuned[h]["param"],
                      "train_dll_per_start": round(tuned[h]["train_dll_per_start"], 5),
                      "train_win": tuned[h]["train_win"]} for h in tuned},
        "combined_components": winners,
        "holdout": results,
    }
    report["verdicts"] = _verdicts(report)
    return report


HYP_LABEL = {"H1": "H1 RECENT FORM ", "H2": "H2 HOME/AWAY   ",
             "H3": "H3 DAYS REST   ", "H4": "H4 OPP WHIFF   ",
             "COMBINED": "COMBINED       "}


def _verdicts(report):
    v = {}
    for h, res in report["holdout"].items():
        if res.get("n", 0) == 0:
            v[h] = "INSUFFICIENT DATA"
            continue
        trained_win = (h == "COMBINED") or report["tuned"][h]["train_win"]
        improved = res["ll_delta_total"] > 0
        robust = res["periods_total"] > 0 and \
            res["periods_better"] >= (res["periods_total"] + 1) // 2
        if trained_win and improved and robust:
            v[h] = "HELPS (robust)"
        elif trained_win and improved:
            v[h] = "MARGINAL (holdout up, not robust across periods)"
        elif not trained_win:
            v[h] = "NO EDGE ON TRAIN (tuned to ~neutral)"
        else:
            v[h] = "DOES NOT HELP on holdout"
    return v


def print_report(report):
    print("\n" + "=" * 78)
    print("K-FACTORS EXPERIMENT — HOLDOUT RESULT (baseline = season K/BF * BF)")
    print("=" * 78)
    if "error" in report:
        print("  ERROR:", report["error"])
        return
    print(f"  train n={report['n_train']}  holdout n={report['n_holdout']}  "
          f"cutoff={report['cutoff']}")
    print(f"  combined-model components (train winners): "
          f"{', '.join(report['combined_components']) or '(none — combined == baseline)'}")
    for h in ("H1", "H2", "H3", "H4", "COMBINED"):
        res = report["holdout"][h]
        if res.get("n", 0) == 0:
            print(f"  {HYP_LABEL[h]} no scorable holdout starts")
            continue
        if h == "COMBINED":
            param = f"components={report['combined_components'] or ['none']}"
            traind = ""
        else:
            t = report["tuned"][h]
            param = f"tuned={t['param']}"
            traind = f" train dLL/start={t['train_dll_per_start']:+.5f}"
        print(f"  {HYP_LABEL[h]} {param}{traind}")
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
    for h in ("H1", "H2", "H3", "H4", "COMBINED"):
        print(f"  {HYP_LABEL[h]} {report['verdicts'][h]}")
    helpers = [h for h in ("H1", "H2", "H3", "H4")
               if report["verdicts"][h].startswith("HELPS")]
    if helpers:
        print(f"\n  => {', '.join(HYP_LABEL[h].strip() for h in helpers)} robustly "
              f"sharpen(s) the K projection on the temporal holdout.")
        print("     Tuned parameters:",
              {h: report["tuned"][h]["param"] for h in helpers})
    else:
        print("\n  => No hypothesis robustly beats the season-rate baseline on the "
              "holdout. Leave mlb_kprops.py unchanged.")
    print("=" * 78)


# ===========================================================================
# DATA PULL — statsapi (blocked here, reachable on Actions). Never fakes data.
# Per start: date, pitcher, K, BF, home/away, opposing team.
# Per team-game: batting K total + PA total (for H4's leak-free opponent whiff).
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
    starts   : [{date, pitcher, so, bf, home, team, opp}] for each side's STARTER
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
                               "team": names[side], "opp": names[other]})
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
def save_dataset(start_date, end_date, starts, team_bat):
    os.makedirs(DATA, exist_ok=True)
    payload = {
        "pulled_at": dt.datetime.utcnow().isoformat() + "Z",
        "start": start_date.isoformat(), "end": end_date.isoformat(),
        "starts": [{**s, "date": s["date"].isoformat()} for s in starts],
        "team_bat": [{**t, "date": t["date"].isoformat()} for t in team_bat],
    }
    with open(DATASET_PATH, "w") as f:
        json.dump(payload, f)
    print(f"dataset cached -> {DATASET_PATH}")


def load_dataset(start_date, end_date):
    """Return (starts, team_bat) from cache if it covers [start, end], else None."""
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
        starts = [{**s, "date": dt.date.fromisoformat(s["date"])}
                  for s in payload["starts"]]
        team_bat = [{**t, "date": dt.date.fromisoformat(t["date"])}
                    for t in payload["team_bat"]]
        starts = [s for s in starts if start_date <= s["date"] <= end_date]
        team_bat = [t for t in team_bat if start_date <= t["date"] <= end_date]
        print(f"using cached dataset {DATASET_PATH} ({len(starts)} starts, "
              f"{len(team_bat)} team-game batting rows)")
        return starts, team_bat
    except Exception as e:
        print(f"cache unreadable ({type(e).__name__}) — will re-pull")
        return None


def run_live(start_date, end_date, train_frac):
    """Cache-first; else attempt the pull; degrade to a clear 'run on Actions'
    message (exit 0) if statsapi is unreachable (expected in this sandbox)."""
    print("=" * 78)
    print(f"K-FACTORS EXPERIMENT — {start_date} .. {end_date}")
    print("=" * 78)
    data = load_dataset(start_date, end_date)
    if data is None:
        # probe egress first so we fail fast & clean
        try:
            _get(f"{API}/schedule?sportId=1&date={start_date.isoformat()}", tries=1)
        except Exception as e:
            print("\nstatsapi UNREACHABLE from here — run on GitHub Actions.")
            print(f"  (probe error: {type(e).__name__}: {str(e)[:120]})")
            print("  This is EXPECTED in the cloud sandbox (egress to statsapi is blocked, 403).")
            print("  Every other MLB workflow proves statsapi IS reachable on Actions.")
            print("  Action command:  python3 mlb_kfactors_experiment.py --start "
                  f"{start_date} --end {end_date}")
            return 0
        starts, team_bat = pull_range(start_date, end_date)
        print(f"\npulled {len(starts)} starts / {len(team_bat)} team-game batting rows")
        if not starts:
            print("no usable starts pulled — cannot run the experiment.")
            return 0
        save_dataset(start_date, end_date, starts, team_bat)
    else:
        starts, team_bat = data
        if not starts:
            print("cached dataset empty for range — cannot run the experiment.")
            return 0

    report = evaluate(starts, team_bat, train_frac=train_frac)
    os.makedirs(DATA, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print_report(report)
    return 0


# ===========================================================================
# OFFLINE SELFTEST — NO NETWORK. Synthetic fixtures verify leak-freeness, the
# Poisson math, planted-effect recovery (H1 form drift, H4 opponent whiff), and
# a null control (no effect -> no free lift).
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


def _synth_bat_rows(date, teams, rng, mult=None):
    """Two league-filler batting rows so league K/BF stays anchored ~0.22."""
    rows = []
    for t in teams:
        m = (mult or {}).get(t, 1.0)
        pa = rng.randint(34, 42)
        so = min(pa, _pois(rng, 0.22 * pa * m))
        rows.append({"date": date, "team": t, "bat_so": so, "bat_pa": pa})
    return rows


def selftest():
    d = dt.date
    import random

    # ---------- (1) POISSON MATH exact on hand-computed fixtures ----------
    # log P(X=3 | lam=2.5) = 3*ln2.5 - 2.5 - ln(3!) = 2.748872 - 2.5 - 1.791759
    #                      = -1.542887
    assert abs(poisson_logpmf(3, 2.5) - (-1.5428868)) < 1e-6, poisson_logpmf(3, 2.5)
    # P(X=0 | lam) = e^-lam  ->  log = -lam
    assert abs(poisson_logpmf(0, 4.0) - (-4.0)) < 1e-9
    # log P(X=6 | lam=6) = 6*ln6 - 6 - ln720 = 10.750557 - 6 - 6.579251 = -1.828694
    assert abs(poisson_logpmf(6, 6.0) - (-1.8286943)) < 1e-6, poisson_logpmf(6, 6.0)
    # LL peaks when lam == k
    lls = [(poisson_logpmf(6, L), L) for L in (3.0, 5.0, 6.0, 7.0, 9.0)]
    assert max(lls)[1] == 6.0, lls
    print("  [1] Poisson logpmf: exact on hand-computed fixtures, peaks at lam=k")

    # ---------- (2) LEAK-FREENESS: strictly-prior only, and load-bearing ----------
    # Pitcher 'Ace Whiff': 5 hot prior starts (9K/25BF), then on the query date an
    # extreme 0K/40BF start. If the current start leaked into its own context the
    # projection would flip across a 6.5 line — construct exactly that case.
    starts = []
    bat = []
    for i in range(5):
        day = d(2025, 4, 1) + dt.timedelta(days=5 * i)
        starts.append({"date": day, "pitcher": "Ace Whiff", "so": 9, "bf": 25,
                       "home": True, "team": "TA", "opp": "TB"})
        bat.extend(_synth_bat_rows(day, ["TA", "TB"], random.Random(i)))
    qday = d(2025, 5, 1)
    starts.append({"date": qday, "pitcher": "Ace Whiff", "so": 0, "bf": 40,
                   "home": True, "team": "TA", "opp": "TB"})
    bat.extend(_synth_bat_rows(qday, ["TA", "TB"], random.Random(99)))
    # a later start to observe the post-inclusion state
    nday = qday + dt.timedelta(days=5)
    starts.append({"date": nday, "pitcher": "Ace Whiff", "so": 5, "bf": 25,
                   "home": True, "team": "TA", "opp": "TB"})
    ctxs = build_contexts(starts, bat)
    by_date = {c["date"]: c for c in ctxs if c["pitcher"] == "Ace Whiff"}
    cq, cn = by_date[qday], by_date[nday]
    assert cq["n_prior"] == 5, cq["n_prior"]          # its own 0K/40BF row is EXCLUDED
    assert cn["n_prior"] == 6, cn["n_prior"]          # ...but visible to the LATER start
    lam_excl = rate_season(cq) * 25                   # rate w/o the extreme start
    lam_incl = rate_season(cn) * 25                   # rate WITH it folded in
    assert lam_excl > 6.5 > lam_incl, (lam_excl, lam_incl)   # inclusion FLIPS a 6.5 line
    print(f"  [2] leak-freeness: current start excluded (n_prior=5) and load-bearing "
          f"(lam {lam_excl:.2f} -> {lam_incl:.2f} flips a 6.5 line)")
    # opponent batting totals are also strictly-prior: the query-date bat rows for
    # TB must not appear in the query-date context...
    prior_tb_pa = sum(t["bat_pa"] for t in bat if t["team"] == "TB" and t["date"] < qday)
    assert cq["opp_pa"] == prior_tb_pa, (cq["opp_pa"], prior_tb_pa)
    # ...but do appear in the later start's context
    assert cn["opp_pa"] > cq["opp_pa"], (cn["opp_pa"], cq["opp_pa"])
    # rest-days derivation from prior start dates
    assert cq["rest_days"] == (qday - d(2025, 4, 21)).days, cq["rest_days"]
    assert cn["rest_days"] == 5, cn["rest_days"]
    print("  [2] opponent batting totals strictly-prior; rest days derived correctly")

    # ---------- (3a) PLANTED H1 EFFECT: form drift is recovered ----------
    # Pitchers' true K/BF random-walks start to start. Season-average lags the
    # walk; exp-weighting tracks it, so tuned H1 must beat baseline on holdout.
    rng = random.Random(20260723)
    teams = [f"T{i}" for i in range(12)]
    pit_rate = {f"P{i}": rng.uniform(0.16, 0.30) for i in range(24)}
    starts, bat = [], []
    day0 = d(2025, 4, 1)
    for gday in range(160):
        date = day0 + dt.timedelta(days=gday)
        pits = rng.sample(list(pit_rate), 6)
        for j, p in enumerate(pits):
            # drift the pitcher's true talent each time he starts
            pit_rate[p] = clip(pit_rate[p] + rng.gauss(0, 0.022), 0.10, 0.38)
            t_own, t_opp = rng.sample(teams, 2)
            bf = rng.randint(18, 30)
            so = min(bf, _pois(rng, pit_rate[p] * bf))
            starts.append({"date": date, "pitcher": p, "so": so, "bf": bf,
                           "home": j % 2 == 0, "team": t_own, "opp": t_opp})
            bat.extend(_synth_bat_rows(date, [t_own, t_opp], rng))
    rep = evaluate(starts, bat, train_frac=0.5, min_prior_starts=3)
    h1 = rep["holdout"]["H1"]
    assert h1["n"] > 150, h1["n"]
    assert rep["tuned"]["H1"]["param"]["decay"] < 1.0, rep["tuned"]["H1"]
    assert rep["tuned"]["H1"]["train_win"], rep["tuned"]["H1"]
    assert h1["ll_delta_total"] > 0, h1
    assert h1["mae_delta"] < 0, h1
    print(f"  [3a] planted FORM drift recovered: tuned decay="
          f"{rep['tuned']['H1']['param']['decay']} holdout LL {h1['ll_delta_total']:+.1f} "
          f"({h1['ll_delta_per_start']:+.4f}/start, n={h1['n']}), MAE {h1['mae_delta']:+.3f}")

    # ---------- (3b) PLANTED H4 EFFECT: opponent whiff is recovered ----------
    # Teams carry KNOWN whiff multipliers; a start's Ks scale with the OPPONENT's
    # multiplier and the opponent's own batting rows carry the same multiplier —
    # exactly the leak-free signal H4 is allowed to read.
    rng = random.Random(777)
    tmult = {f"T{i}": m for i, m in enumerate(
        [1.22, 1.15, 1.10, 1.05, 1.0, 1.0, 0.95, 0.92, 0.88, 0.82, 1.0, 1.0])}
    teams = list(tmult)
    pit_rate = {f"P{i}": rng.uniform(0.16, 0.30) for i in range(24)}  # STATIC now
    starts, bat = [], []
    for gday in range(160):
        date = day0 + dt.timedelta(days=gday)
        pits = rng.sample(list(pit_rate), 6)
        for j, p in enumerate(pits):
            t_own, t_opp = rng.sample(teams, 2)
            bf = rng.randint(18, 30)
            so = min(bf, _pois(rng, pit_rate[p] * bf * tmult[t_opp]))
            starts.append({"date": date, "pitcher": p, "so": so, "bf": bf,
                           "home": j % 2 == 0, "team": t_own, "opp": t_opp})
            bat.extend(_synth_bat_rows(date, [t_own, t_opp], rng, mult=tmult))
    rep4 = evaluate(starts, bat, train_frac=0.5, min_prior_starts=3)
    h4 = rep4["holdout"]["H4"]
    assert h4["n"] > 150, h4["n"]
    assert rep4["tuned"]["H4"]["param"]["w"] >= 0.5, rep4["tuned"]["H4"]
    assert rep4["tuned"]["H4"]["train_win"], rep4["tuned"]["H4"]
    assert h4["ll_delta_total"] > 0, h4
    assert h4["mae_delta"] < 0, h4
    # the combined model must include H4 and also beat baseline on holdout
    assert "H4" in rep4["combined_components"], rep4["combined_components"]
    assert rep4["holdout"]["COMBINED"]["ll_delta_total"] > 0, rep4["holdout"]["COMBINED"]
    print(f"  [3b] planted OPPONENT-WHIFF effect recovered: tuned w="
          f"{rep4['tuned']['H4']['param']['w']} holdout LL {h4['ll_delta_total']:+.1f} "
          f"({h4['ll_delta_per_start']:+.4f}/start, n={h4['n']}), MAE {h4['mae_delta']:+.3f}; "
          f"combined includes H4")

    # ---------- (4) NULL CONTROL: no planted effect -> delta near zero ----------
    # Static pitcher rates, neutral teams, no venue/rest/form effects. No
    # hypothesis may manufacture a meaningful holdout edge.
    rng = random.Random(31337)
    pit_rate = {f"P{i}": rng.uniform(0.16, 0.30) for i in range(24)}
    starts, bat = [], []
    for gday in range(160):
        date = day0 + dt.timedelta(days=gday)
        pits = rng.sample(list(pit_rate), 6)
        for j, p in enumerate(pits):
            t_own, t_opp = rng.sample(teams, 2)
            bf = rng.randint(18, 30)
            so = min(bf, _pois(rng, pit_rate[p] * bf))
            starts.append({"date": date, "pitcher": p, "so": so, "bf": bf,
                           "home": j % 2 == 0, "team": t_own, "opp": t_opp})
            bat.extend(_synth_bat_rows(date, [t_own, t_opp], rng))
    rep0 = evaluate(starts, bat, train_frac=0.5, min_prior_starts=3)
    for h in ("H1", "H2", "H3", "H4", "COMBINED"):
        dps = rep0["holdout"][h]["ll_delta_per_start"]
        assert abs(dps) < 0.01, (h, dps)
    print("  [4] null control: no planted effect -> all holdout LL/start deltas "
          "within +/-0.01 of zero "
          + str({h: rep0["holdout"][h]["ll_delta_per_start"]
                 for h in ("H1", "H2", "H3", "H4", "COMBINED")}))

    print("K-FACTORS SELFTEST PASS — leak-free & load-bearing, Poisson math exact, "
          "planted H1/H4 effects recovered, null control clean")
    return 0


# ===========================================================================
def _date(s):
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def main(argv):
    ap = argparse.ArgumentParser(
        description="K-factors experiment: recent form / home-away / rest / opponent whiff (standalone).")
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
