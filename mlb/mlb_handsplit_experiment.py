#!/usr/bin/env python3
"""
mlb_handsplit_experiment.py — DOES HAND-SPLIT OPPONENT WHIFF BEAT OVERALL WHIFF?
================================================================================
STANDALONE EXPERIMENT. Does not import or modify mlb_kprops.py or any production
file. It only builds evidence and prints a VERDICT.

THE HYPOTHESIS
--------------
The validated baseline (winner of the k-factors experiment, shipped to
production as OPP_W=0.6 in mlb_kprops.py) projects a start's strikeouts as
Poisson(lambda) with
    lambda_b = pitcher_prior_rate * BF * [1 + 0.6*(opp_overall_K% / league - 1)]
where opp_overall_K% is the opposing team's batting K% over strictly-prior
games. The hypothesis: teams whiff DIFFERENTLY by the throwing hand of the
starter they face ("that lineup strikes out vs lefties"). Treatment replaces
the OVERALL opponent K% with the opponent's K% in prior games vs SAME-HANDED
starters, shrunk toward the overall K% because the vs-LHP sample is only ~1/3
of games:
    split_rate = (split_SO + tau * overall_rate) / (split_PA + tau)
    lambda_t   = pitcher_prior_rate * BF * [1 + 0.6*(split_rate / league - 1)]
tau (pseudo-PA shrink; tau=off means blend weight 0 = pure overall = EXACT
baseline) and a minimum prior split-PA threshold are the only tuned knobs,
tuned on TRAIN only. The outer 0.6 weight stays FIXED at the validated value.

WHERE THE SPLIT COMES FROM (leak-free)
--------------------------------------
The dataset has no per-PA hand log, so the split is reconstructed by JOINING
each team-game batting row to the starts table: the starter who faced that
team that day (start.opp == bat.team, same date; doubleheaders pair off in
pull order) and that starter's THROWING HAND (statsapi people pitchHand.code,
cached in the dataset). A batting row whose opposing starter's hand is unknown
folds into the overall totals only, never into a split. A starter's hand is
static biology, known pregame — using it to label a PAST game's row is not a
leak; the leak barrier (rows fold only after their date is predicted) is
identical to the k-factors harness.

METHOD (non-negotiable)
-----------------------
Chronological walk-forward. Every feature comes from games dated STRICTLY
BEFORE the start being predicted. Temporal split by date: earlier TRAIN
portion tunes (tau, min split PA); later HOLDOUT is scored once. Scoring:
Poisson log-likelihood per start + MAE + 3-period robustness, treatment vs
the w=0.6 overall-whiff baseline.

NETWORK
-------
Reuses the committed kfactors dataset (starts + team-game batting rows) when
present, so the only fetch is pitcher throwing hands: one bulk call to
/sports/1/players?season=YYYY resolves nearly every starter; stragglers fall
back to /people/search per name. Hands are cached in
mlb/data/handsplit_dataset.json and the fetch is resume-safe (saves as it
goes). In THIS sandbox egress to statsapi is blocked — EXPECTED; the script
prints 'UNREACHABLE — run on GitHub Actions' and exits 0. It never fakes data.

RUN
---
  python3 mlb_handsplit_experiment.py --selftest      # offline, no network, must pass
  python3 mlb_handsplit_experiment.py                 # live (on Actions)
  python3 mlb_handsplit_experiment.py --start 2025-04-01 --end 2025-06-30
"""
import os, sys, json, math, time, argparse, datetime as dt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASET_PATH = os.path.join(DATA, "handsplit_dataset.json")
KFACTORS_DATASET_PATH = os.path.join(DATA, "kfactors_dataset.json")
REPORT_PATH = os.path.join(DATA, "handsplit_experiment.json")

# ---- experiment-local constants (production untouched) ----------------------
LG_K_PER_BF = 0.222     # league K per batter faced fallback
REG_BF = 100.0          # league BF regression on the pitcher's own rate (as baseline)
MIN_PRIOR_STARTS = 3    # pitcher needs this many prior starts to be scored
MIN_OPP_PA = 300        # opponent needs this many prior OVERALL PAs (else factor 1.0)
W_OPP = 0.6             # FIXED outer whiff weight — the validated production value
TRAIN_FRAC = 0.5        # earlier fraction of dates = TRAIN; later = HOLDOUT
N_PERIODS = 3           # holdout robustness buckets
TRAIN_WIN_MARGIN = 5e-4 # train mean-LL/start margin the treatment must clear
OPP_LO, OPP_HI = 0.80, 1.20   # bound on the whiff factor (same as baseline)

# tuning grids (TRAIN only). tau=None -> blend weight 0 -> pure overall -> the
# treatment is IDENTICAL to the baseline. Finite tau = pseudo-PA of overall-rate
# prior mixed into the split estimate (blend weight = split_PA/(split_PA+tau)).
TAU_GRID = [None, 3000.0, 1600.0, 800.0, 400.0, 200.0, 100.0]
MINPA_GRID = [50, 100, 150, 200, 300]   # min prior split-PA else fall back to overall


def norm(s):
    if not isinstance(s, str):
        return ""
    return "".join(c for c in s.lower() if c.isalnum())


def clip(x, lo, hi):
    return min(hi, max(lo, x))


def poisson_logpmf(k, lam):
    """log P(X = k) for X ~ Poisson(lam). Exact: k*ln(lam) - lam - ln(k!)."""
    lam = max(float(lam), 1e-9)
    k = int(k)
    return k * math.log(lam) - lam - math.lgamma(k + 1)


# ===========================================================================
# HAND ATTRIBUTION — join each team-game batting row to the starter who faced
# that team that day, and take his throwing hand. Doubleheaders (two rows for
# the same team+date) pair off in list order, which matches pull order because
# starts and batting rows are emitted per-boxscore by the same loop. Unknown
# hand -> vs_hand=None -> the row counts toward OVERALL totals only.
# ===========================================================================
def attribute_hands(starts, team_bat, hands):
    """Return team_bat rows augmented with vs_hand in {'L','R',None}."""
    faced = {}   # (date, norm(team)) -> [hand of each starter who faced them, in order]
    for s in starts:
        h = hands.get(norm(s["pitcher"]))
        faced.setdefault((s["date"], norm(s["opp"])), []).append(
            h if h in ("L", "R") else None)
    used, out = {}, []
    for t in team_bat:
        key = (t["date"], norm(t["team"]))
        i = used.get(key, 0)
        used[key] = i + 1
        lst = faced.get(key, [])
        out.append({**t, "vs_hand": lst[i] if i < len(lst) else None})
    return out


# ===========================================================================
# LEAK-FREE CONTEXT BUILDER — same one-pass discipline as the k-factors
# harness: snapshot every start on a date from strictly-prior state, THEN fold
# that date's rows. A start can never see itself, its own game, or the future.
# ===========================================================================
def build_contexts(starts, team_bat, hands):
    """
    starts   : [{date(dt.date), pitcher, so, bf, home(bool), team, opp}]
    team_bat : [{date(dt.date), team, bat_so, bat_pa}]  (one row per team per game)
    hands    : {norm(pitcher): 'L'|'R'}
    Returns chronological ctx list. Each ctx carries the outcome, strictly-prior
    pitcher history, league rate, opponent OVERALL batting totals AND the
    opponent's totals vs starters of THIS start's hand.
    """
    bat = attribute_hands(starts, team_bat, hands)
    all_dates = sorted({s["date"] for s in starts} | {t["date"] for t in bat})
    starts_by_date, bat_by_date = {}, {}
    for s in starts:
        starts_by_date.setdefault(s["date"], []).append(s)
    for t in bat:
        bat_by_date.setdefault(t["date"], []).append(t)

    pit = {}          # norm(pitcher) -> [(date, so, bf)] chronological
    team_tot = {}     # norm(team) -> [so, pa]                      OVERALL
    team_split = {}   # (norm(team), 'L'|'R') -> [so, pa]           BY HAND FACED
    lg_so = lg_pa = 0

    ctxs = []
    for date in all_dates:
        # 1) SNAPSHOT this date's starts from strictly-prior state
        for s in starts_by_date.get(date, []):
            prior = list(pit.get(norm(s["pitcher"]), []))
            okey = norm(s.get("opp", ""))
            oso, opa = team_tot.get(okey, (0, 0))
            hand = hands.get(norm(s["pitcher"]))
            hand = hand if hand in ("L", "R") else None
            hso, hpa = team_split.get((okey, hand), (0, 0)) if hand else (0, 0)
            lg = (lg_so / lg_pa) if lg_pa > 0 else LG_K_PER_BF
            ctxs.append({
                "date": date, "pitcher": s["pitcher"], "k": int(s["so"]),
                "bf": int(s["bf"]), "opp": s.get("opp", ""),
                "prior": prior, "n_prior": len(prior), "lg": lg,
                "hand": hand,
                "opp_so": oso, "opp_pa": opa,          # overall, strictly prior
                "opp_so_h": hso, "opp_pa_h": hpa,      # vs this hand, strictly prior
            })
        # 2) FOLD this date's rows into state (visible to LATER dates only)
        for s in starts_by_date.get(date, []):
            pit.setdefault(norm(s["pitcher"]), []).append(
                (date, int(s["so"]), int(s["bf"])))
        for t in bat_by_date.get(date, []):
            so, pa = int(t["bat_so"]), int(t["bat_pa"])
            key = norm(t["team"])
            cso, cpa = team_tot.get(key, (0, 0))
            team_tot[key] = (cso + so, cpa + pa)
            if t.get("vs_hand") in ("L", "R"):
                sso, spa = team_split.get((key, t["vs_hand"]), (0, 0))
                team_split[(key, t["vs_hand"])] = (sso + so, spa + pa)
            lg_so += so
            lg_pa += pa
    return ctxs


# ===========================================================================
# THE MODELS — pure functions of a frozen context (+ tuned params).
# ===========================================================================
def rate_season(ctx):
    """Pitcher K/BF over strictly-prior starts, league-regressed (as baseline)."""
    so = sum(p[1] for p in ctx["prior"])
    bf = sum(p[2] for p in ctx["prior"])
    return (so + REG_BF * ctx["lg"]) / (bf + REG_BF)


def factor_overall(ctx):
    """BASELINE whiff factor: overall opponent K% vs league, FIXED w=0.6."""
    if ctx["opp_pa"] < MIN_OPP_PA or ctx["lg"] <= 0:
        return 1.0
    raw = (ctx["opp_so"] / ctx["opp_pa"]) / ctx["lg"]
    return clip(1.0 + W_OPP * (raw - 1.0), OPP_LO, OPP_HI)


def lam_baseline(ctx):
    """The validated production-shape baseline: prior-rate * BF * overall whiff."""
    return rate_season(ctx) * ctx["bf"] * factor_overall(ctx)


def factor_split(ctx, tau, min_pa):
    """TREATMENT whiff factor: opponent K% vs starters of THIS start's hand,
    shrunk toward the opponent's overall K% with tau pseudo-PA
    (blend weight = split_PA/(split_PA+tau)). Falls back to overall when
    tau is None (pure overall = exact baseline), the starter's hand is
    unknown, or the prior split sample is under min_pa. Same fixed w=0.6,
    same bounds, same MIN_OPP_PA gate as the baseline."""
    if ctx["opp_pa"] < MIN_OPP_PA or ctx["lg"] <= 0:
        return 1.0
    overall = ctx["opp_so"] / ctx["opp_pa"]
    if tau is None or ctx["hand"] is None or ctx["opp_pa_h"] < min_pa:
        rate = overall
    else:
        rate = (ctx["opp_so_h"] + tau * overall) / (ctx["opp_pa_h"] + tau)
    return clip(1.0 + W_OPP * (rate / ctx["lg"] - 1.0), OPP_LO, OPP_HI)


def lam_treatment(ctx, tau, min_pa):
    return rate_season(ctx) * ctx["bf"] * factor_split(ctx, tau, min_pa)


# ===========================================================================
# TUNING (TRAIN ONLY) + HOLDOUT SCORING
# ===========================================================================
def _ll(ctxs, lam_fn):
    return sum(poisson_logpmf(c["k"], lam_fn(c)) for c in ctxs)


def tune(train_ctxs):
    """Joint grid over (tau, min_pa) on TRAIN. The grid contains the exact
    baseline (tau=None), so best-train >= baseline-train by construction;
    the reported delta is the margin over the baseline."""
    n = max(1, len(train_ctxs))
    base_ll = _ll(train_ctxs, lam_baseline)
    best = (None, None)
    best_ll = base_ll          # tau=None == baseline, any min_pa
    for tau in TAU_GRID:
        if tau is None:
            continue
        for mp in MINPA_GRID:
            ll = _ll(train_ctxs, lambda c: lam_treatment(c, tau, mp))
            if ll > best_ll:
                best_ll, best = ll, (tau, mp)
    d = (best_ll - base_ll) / n
    return {"param": {"tau": best[0], "min_split_pa": best[1]},
            "train_dll_per_start": d,
            "train_win": d > TRAIN_WIN_MARGIN}


def _score_holdout(hold_ctxs, tau, min_pa, n_periods=N_PERIODS):
    """Treatment vs baseline on the holdout: total & per-start Poisson LL
    delta, MAE delta, per-period robustness."""
    rows = []
    for c in hold_ctxs:
        lam_b = lam_baseline(c)
        lam_m = lam_treatment(c, tau, min_pa)
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


def evaluate(starts, team_bat, hands, train_frac=TRAIN_FRAC,
             min_prior_starts=MIN_PRIOR_STARTS, n_periods=N_PERIODS):
    """Full experiment: contexts -> temporal split -> tune on TRAIN -> score
    the tuned treatment ONCE on HOLDOUT vs the overall-whiff baseline."""
    ctxs = [c for c in build_contexts(starts, team_bat, hands)
            if c["n_prior"] >= min_prior_starts]
    if not ctxs:
        return {"error": "no scorable starts"}
    all_dates = sorted({c["date"] for c in ctxs})
    cutoff = all_dates[min(int(len(all_dates) * train_frac), len(all_dates) - 1)]
    train = [c for c in ctxs if c["date"] < cutoff]
    hold = [c for c in ctxs if c["date"] >= cutoff]
    if not train or not hold:
        return {"error": "empty train or holdout after split"}

    tuned = tune(train)
    tau, mp = tuned["param"]["tau"], tuned["param"]["min_split_pa"]
    res = _score_holdout(hold, tau, mp, n_periods)

    known = sum(1 for c in ctxs if c["hand"] is not None)
    split_used = sum(1 for c in hold
                     if tau is not None and c["hand"] is not None
                     and c["opp_pa"] >= MIN_OPP_PA
                     and c["opp_pa_h"] >= (mp or 0))
    report = {
        "cutoff": cutoff.isoformat(),
        "n_train": len(train), "n_holdout": len(hold),
        "hand_known_frac": round(known / len(ctxs), 4),
        "holdout_starts_using_split": split_used,
        "constants": {"reg_bf": REG_BF, "min_prior_starts": min_prior_starts,
                      "min_opp_pa": MIN_OPP_PA, "w_opp_fixed": W_OPP,
                      "train_frac": train_frac,
                      "train_win_margin": TRAIN_WIN_MARGIN},
        "tuned": tuned,
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
        return "HELPS (robust) — hand-split whiff beats overall whiff out-of-sample"
    if trained_win and improved:
        return "MARGINAL (holdout up, not robust across periods)"
    if not trained_win:
        return "NO EDGE ON TRAIN (blend tuned to ~pure overall) — keep overall whiff"
    return "DOES NOT HELP on holdout — keep overall whiff"


def print_report(report):
    print("\n" + "=" * 78)
    print("HAND-SPLIT WHIFF EXPERIMENT — HOLDOUT RESULT")
    print("(baseline = prior-rate * BF * overall-whiff factor, FIXED w=0.6)")
    print("=" * 78)
    if "error" in report:
        print("  ERROR:", report["error"])
        return
    t = report["tuned"]
    print(f"  train n={report['n_train']}  holdout n={report['n_holdout']}  "
          f"cutoff={report['cutoff']}")
    print(f"  starter hand known for {report['hand_known_frac']:.1%} of scorable starts")
    print(f"  tuned on TRAIN: tau={t['param']['tau']} "
          f"min_split_pa={t['param']['min_split_pa']}  "
          f"train dLL/start={t['train_dll_per_start']:+.5f}  "
          f"train_win={t['train_win']}")
    res = report["holdout"]
    if res.get("n", 0) == 0:
        print("  no scorable holdout starts")
        return
    print(f"  holdout starts actually using a split estimate: "
          f"{report['holdout_starts_using_split']}/{res['n']}")
    print(f"  holdout LL delta {res['ll_delta_total']:+.2f} total "
          f"({res['ll_delta_per_start']:+.5f}/start, n={res['n']})  "
          f"MAE {res['mae_delta']:+.4f}  "
          f"robust {res['periods_better']}/{res['periods_total']} periods")
    for p in res["periods"]:
        flag = "treatment" if p["treatment_better"] else "baseline "
        print(f"    period {p['period']:<26} n={p['n']:<4} "
              f"LL delta {p['ll_delta_total']:+.2f} -> {flag} better")
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  HAND-SPLIT OPP WHIFF: {report['verdict']}")
    if report["verdict"].startswith("HELPS"):
        print(f"  => Consider a hand-split whiff factor in mlb_kprops.py "
              f"(tau={t['param']['tau']}, min_split_pa={t['param']['min_split_pa']}, "
              f"outer w stays 0.6).")
    else:
        print("  => Keep the overall-whiff factor (OPP_W=0.6) unchanged in production.")
    print("=" * 78)


# ===========================================================================
# DATA — starts + team_bat reuse the committed kfactors dataset (same pull
# pattern as fallback); the ONLY new fetch is pitcher throwing hands.
# Never fakes data; degrades cleanly when statsapi is unreachable.
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


# ---- hands fetch (the only NEW egress) --------------------------------------
def fetch_hands(display_names, season, hands, save_cb):
    """Resolve throwing hands for the given pitcher display names (missing from
    `hands`). One bulk /sports/1/players call resolves nearly everyone;
    stragglers use /people/search. Mutates `hands` (norm name -> 'L'/'R') and
    calls save_cb periodically so the fetch is resume-safe."""
    from urllib.parse import quote
    missing = {norm(n): n for n in display_names if norm(n) not in hands}
    if not missing:
        return
    print(f"fetching throwing hand for {len(missing)} pitchers "
          f"(bulk season list, then per-name fallback)")
    try:
        j = _get(f"{API}/sports/1/players?season={season}")
        bulk = {}
        for p in j.get("people", []):
            code = ((p.get("pitchHand") or {}).get("code") or "").upper()
            n = norm(p.get("fullName", ""))
            if n and code in ("L", "R"):
                bulk.setdefault(n, set()).add(code)
        for n in list(missing):
            codes = bulk.get(n)
            if codes and len(codes) == 1:   # unambiguous name match
                hands[n] = next(iter(codes))
                del missing[n]
        print(f"  bulk list resolved all but {len(missing)}")
        save_cb()
    except Exception as e:
        print(f"  bulk player list failed ({type(e).__name__}) — per-name only")
    done = 0
    for n, disp in list(missing.items()):
        try:
            j = _get(f"{API}/people/search?names={quote(disp)}", tries=2)
            codes = {((p.get("pitchHand") or {}).get("code") or "").upper()
                     for p in j.get("people", [])
                     if norm(p.get("fullName", "")) == n}
            codes &= {"L", "R"}
            if len(codes) == 1:
                hands[n] = next(iter(codes))
        except Exception:
            pass                              # leave unknown; resume next run
        done += 1
        if done % 25 == 0:
            save_cb()
        time.sleep(0.1)
    save_cb()
    still = [d for k, d in missing.items() if k not in hands]
    if still:
        print(f"  hand unknown for {len(still)} pitchers (treated as no-split): "
              f"{still[:8]}{'...' if len(still) > 8 else ''}")


# ---- dataset cache ----------------------------------------------------------
def save_dataset(start_date, end_date, starts, team_bat, hands):
    os.makedirs(DATA, exist_ok=True)
    payload = {
        "pulled_at": dt.datetime.utcnow().isoformat() + "Z",
        "start": start_date.isoformat(), "end": end_date.isoformat(),
        "starts": [{**s, "date": s["date"].isoformat()} for s in starts],
        "team_bat": [{**t, "date": t["date"].isoformat()} for t in team_bat],
        "hands": hands,
    }
    with open(DATASET_PATH, "w") as f:
        json.dump(payload, f)
    print(f"dataset cached -> {DATASET_PATH}")


def load_dataset(path, start_date, end_date, label):
    """(starts, team_bat) from `path` if it covers [start, end], else None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        cs = dt.date.fromisoformat(payload["start"])
        ce = dt.date.fromisoformat(payload["end"])
        if cs > start_date or ce < end_date:
            print(f"{label} covers {cs}..{ce} — does not cover requested range")
            return None
        starts = [{**s, "date": dt.date.fromisoformat(s["date"])}
                  for s in payload["starts"]]
        team_bat = [{**t, "date": dt.date.fromisoformat(t["date"])}
                    for t in payload["team_bat"]]
        starts = [s for s in starts if start_date <= s["date"] <= end_date]
        team_bat = [t for t in team_bat if start_date <= t["date"] <= end_date]
        print(f"using {label} ({len(starts)} starts, {len(team_bat)} team-game rows)")
        return starts, team_bat
    except Exception as e:
        print(f"{label} unreadable ({type(e).__name__})")
        return None


def load_cached_hands():
    """Hands are static — reuse any cached map regardless of date coverage."""
    if not os.path.exists(DATASET_PATH):
        return {}
    try:
        with open(DATASET_PATH) as f:
            h = json.load(f).get("hands", {}) or {}
        return {k: v for k, v in h.items() if v in ("L", "R")}
    except Exception:
        return {}


def _unreachable(start_date, end_date, err):
    print("\nstatsapi UNREACHABLE from here — run on GitHub Actions.")
    print(f"  (probe error: {type(err).__name__}: {str(err)[:120]})")
    print("  This is EXPECTED in the cloud sandbox (egress to statsapi is blocked, 403).")
    print("  Every other MLB workflow proves statsapi IS reachable on Actions.")
    print("  Action command:  python3 mlb_handsplit_experiment.py --start "
          f"{start_date} --end {end_date}")
    return 0


def run_live(start_date, end_date, train_frac):
    print("=" * 78)
    print(f"HAND-SPLIT WHIFF EXPERIMENT — {start_date} .. {end_date}")
    print("=" * 78)
    # 1) starts + team_bat: own cache, else the committed kfactors dataset, else pull
    data = load_dataset(DATASET_PATH, start_date, end_date, "handsplit cache")
    if data is None:
        data = load_dataset(KFACTORS_DATASET_PATH, start_date, end_date,
                            "committed kfactors dataset (reused)")
    if data is None:
        try:
            _get(f"{API}/schedule?sportId=1&date={start_date.isoformat()}", tries=1)
        except Exception as e:
            return _unreachable(start_date, end_date, e)
        starts, team_bat = pull_range(start_date, end_date)
        print(f"\npulled {len(starts)} starts / {len(team_bat)} team-game batting rows")
    else:
        starts, team_bat = data
    if not starts:
        print("no usable starts — cannot run the experiment.")
        return 0

    # 2) throwing hands: cached map + fetch only what's missing (resume-safe)
    hands = load_cached_hands()
    display = sorted({s["pitcher"] for s in starts})
    missing = [n for n in display if norm(n) not in hands]
    print(f"pitchers: {len(display)} unique, hands cached for "
          f"{len(display) - len(missing)}, missing {len(missing)}")
    if missing:
        try:
            _get(f"{API}/schedule?sportId=1&date={start_date.isoformat()}", tries=1)
        except Exception as e:
            save_dataset(start_date, end_date, starts, team_bat, hands)
            return _unreachable(start_date, end_date, e)
        fetch_hands(missing, start_date.year, hands,
                    lambda: save_dataset(start_date, end_date, starts, team_bat, hands))
    save_dataset(start_date, end_date, starts, team_bat, hands)

    known = sum(1 for n in display if norm(n) in hands)
    print(f"hand coverage: {known}/{len(display)} starters")
    if known == 0:
        print("no hands resolved — cannot run the experiment.")
        return 0

    report = evaluate(starts, team_bat, hands, train_frac=train_frac)
    os.makedirs(DATA, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print_report(report)
    return 0


# ===========================================================================
# OFFLINE SELFTEST — NO NETWORK. Verifies: shrink math exact, leak-freeness
# (future poisoning changes nothing), the hand join is load-bearing, a planted
# hand-split effect is recovered, and a null control stays clean.
# ===========================================================================
def _pois(rng, lam):
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _ctx_key(c):
    """Serializable fingerprint of everything a model may read from a ctx."""
    return (c["date"].isoformat(), norm(c["pitcher"]), c["k"], c["bf"], c["hand"],
            tuple(c["prior"]), round(c["lg"], 12),
            c["opp_so"], c["opp_pa"], c["opp_so_h"], c["opp_pa_h"])


def _synth_two_sided(rng, n_days, teams, pit_hand, pit_rate, mult, games_per_day=5):
    """Full two-sided games (both starters recorded, like the real pull): the
    away/home batting rows' K rate carries the OPPOSING starter's hand
    multiplier for that team, and the starter's K total matches."""
    starts, bat = [], []
    day0 = dt.date(2025, 4, 1)
    pitchers = list(pit_rate)
    for g in range(n_days):
        date = day0 + dt.timedelta(days=g)
        for _ in range(games_per_day):
            ta, tb = rng.sample(teams, 2)
            pa_, pb_ = rng.sample(pitchers, 2)
            for (t_own, t_opp, p) in ((ta, tb, pa_), (tb, ta, pb_)):
                h = pit_hand[norm(p)]
                m = mult.get(t_opp, {}).get(h, 1.0)
                bf = rng.randint(20, 28)
                so = min(bf, _pois(rng, pit_rate[p] * bf * m))
                starts.append({"date": date, "pitcher": p, "so": so, "bf": bf,
                               "home": t_own == ta, "team": t_own, "opp": t_opp})
                # t_opp's batting row: they faced starter p (hand h)
                pa_tot = rng.randint(34, 42)
                bso = min(pa_tot, _pois(rng, 0.22 * pa_tot * m))
                bat.append({"date": date, "team": t_opp,
                            "bat_so": bso, "bat_pa": pa_tot})
    return starts, bat


def selftest():
    d = dt.date
    import random

    # ---------- (d) SHRINK MATH exact on a hand example ----------
    ctx = {"opp_so": 200, "opp_pa": 1000, "opp_so_h": 30, "opp_pa_h": 100,
           "lg": 0.22, "hand": "L"}
    # overall = 0.20; tau=200: rate = (30 + 200*0.20)/(100+200) = 70/300 = 0.2333333
    # factor = 1 + 0.6*(0.2333333/0.22 - 1) = 1 + 0.6*0.0606061 = 1.0363636
    f = factor_split(ctx, 200.0, 50)
    assert abs(f - (1.0 + 0.6 * ((70.0 / 300.0) / 0.22 - 1.0))) < 1e-12, f
    assert abs(f - 1.0363636) < 1e-6, f
    # tau=None -> EXACT baseline factor: 1 + 0.6*(0.2/0.22 - 1) = 0.9454545
    f0 = factor_split(ctx, None, 50)
    assert abs(f0 - factor_overall({**ctx})) < 1e-15
    assert abs(f0 - 0.9454545) < 1e-6, f0
    # min_pa gate: split PA 100 < 150 -> falls back to overall
    assert abs(factor_split(ctx, 200.0, 150) - f0) < 1e-15
    # unknown hand -> overall; thin overall sample -> neutral 1.0
    assert abs(factor_split({**ctx, "hand": None}, 200.0, 50) - f0) < 1e-15
    assert factor_split({**ctx, "opp_pa": 299}, 200.0, 50) == 1.0
    # Poisson core sanity (same fixture as the kfactors harness)
    assert abs(poisson_logpmf(3, 2.5) - (-1.5428868)) < 1e-6
    print("  [d] shrink math exact on hand example (tau=200: 70/300 -> factor "
          "1.0363636); tau=None == overall factor exactly; min-PA & unknown-hand "
          "fallbacks correct; Poisson logpmf exact")

    # ---------- (a) LEAK-FREENESS: strictly prior + future poisoning ----------
    rng = random.Random(11)
    teams = [f"T{i}" for i in range(8)]
    pit_hand = {norm(f"P{i}"): ("L" if i % 3 == 0 else "R") for i in range(16)}
    pit_rate = {f"P{i}": rng.uniform(0.16, 0.30) for i in range(16)}
    hands = dict(pit_hand)
    starts, bat = _synth_two_sided(rng, 60, teams, pit_hand, pit_rate, {})
    cutoff = d(2025, 5, 1)
    ctxs1 = build_contexts(starts, bat, hands)
    keys1 = [_ctx_key(c) for c in ctxs1 if c["date"] < cutoff]
    # poison EVERY row on/after the cutoff (absurd K totals) — nothing before
    # the cutoff may change.
    p_starts = [dict(s, so=s["bf"]) if s["date"] >= cutoff else s for s in starts]
    p_bat = [dict(t, bat_so=t["bat_pa"]) if t["date"] >= cutoff else t for t in bat]
    ctxs2 = build_contexts(p_starts, p_bat, hands)
    keys2 = [_ctx_key(c) for c in ctxs2 if c["date"] < cutoff]
    assert keys1 == keys2, "future poisoning leaked into earlier contexts"
    # and the split totals ARE strictly prior: a start's own game excluded
    c_last = [c for c in ctxs1 if c["date"] == max(s["date"] for s in starts)][0]
    prior_bat = [t for t in attribute_hands(starts, bat, hands)
                 if norm(t["team"]) == norm(c_last["opp"]) and t["date"] < c_last["date"]]
    assert c_last["opp_pa"] == sum(t["bat_pa"] for t in prior_bat)
    if c_last["hand"]:
        assert c_last["opp_pa_h"] == sum(t["bat_pa"] for t in prior_bat
                                         if t["vs_hand"] == c_last["hand"])
    print(f"  [a] leak-freeness: poisoning all games on/after {cutoff} leaves all "
          f"{len(keys1)} earlier contexts bit-identical; overall & split totals "
          f"strictly prior")

    # ---------- (b) THE JOIN IS LOAD-BEARING ----------
    # Team TB is whiffy ONLY vs LHP: 12 prior games vs lefties (14 K / 40 PA)
    # and 12 vs righties (4 K / 40 PA). Overall = 216/960 = 0.225 (≈ league) so
    # the OVERALL factor is ~neutral — only the join can tell the hands apart.
    starts, bat, hands = [], [], {}
    day0 = d(2025, 4, 1)
    for i in range(24):
        date = day0 + dt.timedelta(days=i)
        pname = f"Lefty {i}" if i < 12 else f"Righty {i}"
        hands[norm(pname)] = "L" if i < 12 else "R"
        so_tb = 14 if i < 12 else 4
        starts.append({"date": date, "pitcher": pname, "so": so_tb, "bf": 40,
                       "home": False, "team": "TX", "opp": "TB"})
        bat.append({"date": date, "team": "TB", "bat_so": so_tb, "bat_pa": 40})
        bat.append({"date": date, "team": "TX", "bat_so": 9, "bat_pa": 40})
    qday = day0 + dt.timedelta(days=30)
    for pname, h in (("Query Lefty", "L"), ("Query Righty", "R")):
        hands[norm(pname)] = h
        starts.append({"date": qday, "pitcher": pname, "so": 5, "bf": 24,
                       "home": False, "team": "TX", "opp": "TB"})
    ctxs = {c["pitcher"]: c for c in build_contexts(starts, bat, hands)
            if c["date"] == qday}
    cl, cr = ctxs["Query Lefty"], ctxs["Query Righty"]
    # the join must have attributed TB's rows to the correct starter hands
    assert (cl["opp_pa"], cl["opp_so"]) == (960, 216), (cl["opp_pa"], cl["opp_so"])
    assert (cl["opp_pa_h"], cl["opp_so_h"]) == (480, 168), "vs-L split wrong"
    assert (cr["opp_pa_h"], cr["opp_so_h"]) == (480, 48), "vs-R split wrong"
    f_over = factor_overall(cl)
    fl = factor_split(cl, 200.0, 150)
    fr = factor_split(cr, 200.0, 150)
    assert fl > f_over > fr, (fl, f_over, fr)   # lefty up, righty down vs overall
    print(f"  [b] join load-bearing: TB whiffy only vs LHP -> lefty start factor "
          f"{fl:.4f} > overall {f_over:.4f} > righty {fr:.4f} "
          f"(splits 168/480 vs 48/480 recovered exactly through the join)")

    # ---------- (c1) PLANTED HAND-SPLIT EFFECT recovered ----------
    # Teams carry hand-specific whiff multipliers built so each team's OVERALL
    # rate stays ~league (0.35 of starters are lefty): the overall factor is
    # blind to the effect, only the split can price it.
    rng = random.Random(20260727)
    teams = [f"T{i}" for i in range(12)]
    deltas = [0.45, 0.35, 0.30, 0.22, 0.0, 0.0, -0.18, -0.25, -0.30, -0.38, 0.0, 0.12]
    p_l = 0.35
    mult = {}
    for t, dl in zip(teams, deltas):
        mL = 1.0 + dl
        mR = (1.0 - p_l * mL) / (1.0 - p_l)      # keeps overall multiplier == 1
        mult[t] = {"L": mL, "R": mR}
    pit_hand = {norm(f"P{i}"): ("L" if i < int(24 * p_l) else "R") for i in range(24)}
    pit_rate = {f"P{i}": rng.uniform(0.16, 0.30) for i in range(24)}
    starts, bat = _synth_two_sided(rng, 160, teams, pit_hand, pit_rate, mult)
    rep = evaluate(starts, bat, dict(pit_hand), train_frac=0.5, min_prior_starts=3)
    res = rep["holdout"]
    assert res["n"] > 300, res["n"]
    assert rep["tuned"]["param"]["tau"] is not None, rep["tuned"]
    assert rep["tuned"]["train_win"], rep["tuned"]
    assert res["ll_delta_total"] > 0, res
    assert res["mae_delta"] < 0, res
    assert rep["verdict"].startswith(("HELPS", "MARGINAL")), rep["verdict"]
    print(f"  [c1] planted hand-split effect recovered: tuned tau="
          f"{rep['tuned']['param']['tau']} min_split_pa="
          f"{rep['tuned']['param']['min_split_pa']} holdout LL "
          f"{res['ll_delta_total']:+.1f} ({res['ll_delta_per_start']:+.4f}/start, "
          f"n={res['n']}), MAE {res['mae_delta']:+.3f} -> {rep['verdict'].split(' — ')[0]}")

    # ---------- (c2) NULL CONTROL: no hand effect -> no free lift ----------
    rng = random.Random(31337)
    pit_rate = {f"P{i}": rng.uniform(0.16, 0.30) for i in range(24)}
    starts0, bat0 = _synth_two_sided(rng, 160, teams, pit_hand, pit_rate, {})
    # identity: tau=None treatment is EXACTLY the baseline on every context
    ctxs0 = [c for c in build_contexts(starts0, bat0, dict(pit_hand))
             if c["n_prior"] >= 3]
    for c in ctxs0[:500]:
        assert abs(lam_treatment(c, None, 150) - lam_baseline(c)) < 1e-12
    rep0 = evaluate(starts0, bat0, dict(pit_hand), train_frac=0.5, min_prior_starts=3)
    dtr = rep0["tuned"]["train_dll_per_start"]
    dps = rep0["holdout"]["ll_delta_per_start"]
    assert dtr < 0.003, ("null train lift too big", rep0["tuned"])
    assert abs(dps) < 0.01, ("null holdout delta too big", rep0["holdout"])
    print(f"  [c2] null control clean: tau=None reproduces baseline exactly; "
          f"train dLL/start {dtr:+.5f} (~0), holdout dLL/start {dps:+.5f} (~0), "
          f"verdict: {rep0['verdict'].split(' — ')[0]}")

    print("HANDSPLIT SELFTEST PASS — shrink math exact, leak-free under future "
          "poisoning, hand join load-bearing, planted split effect recovered, "
          "null control clean")
    return 0


# ===========================================================================
def _date(s):
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def main(argv):
    ap = argparse.ArgumentParser(
        description="Hand-split opponent whiff vs overall whiff for K props (standalone).")
    ap.add_argument("--selftest", action="store_true", help="offline synthetic tests, no network")
    ap.add_argument("--start", type=_date, default=None, help="range start YYYY-MM-DD")
    ap.add_argument("--end", type=_date, default=None, help="range end YYYY-MM-DD")
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC,
                    help="earlier fraction of dates used to tune; the rest is the scored holdout")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    start = args.start or dt.date(dt.date.today().year, 4, 1)
    end = args.end or dt.date(dt.date.today().year, 6, 30)
    return run_live(start, end, args.train_frac)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
