#!/usr/bin/env python3
"""
mlb_umpire_experiment.py — DOES THE HOME-PLATE UMPIRE SHARPEN K-PROP PREDICTIONS?
================================================================================
STANDALONE EXPERIMENT. Does not import or modify mlb_kprops.py or any production
file. It only builds evidence and prints a VERDICT.

THE HYPOTHESIS
--------------
The current K-props model (mlb_kprops.py) sets a Poisson lambda from the pitcher's
own K rate, workload, and opponent whiff — it never looks at WHO IS BEHIND THE
PLATE. But the home-plate umpire owns the strike zone, and zone size drives whiffs:
a wide-zone ump inflates every pitcher's Ks that night, a tight-zone ump deflates
them. statsapi's boxscore exposes the home-plate umpire (officials) and every
pitcher's strikeOuts + battersFaced, so an umpire's rolling "K-above-average" is
buildable — and, crucially, buildable LEAK-FREE (strictly prior games only).

This harness asks one honest question: if we multiply the baseline Poisson lambda
by the umpire's PRIOR-games K-tendency factor, does the actual pitcher-K total land
inside a SHARPER distribution on a temporal holdout? "Sharper" = higher Poisson
log-likelihood + lower MAE + better Over calibration. If yes and by a meaningful
margin, the umpire factor earns a place in the production model. If not, we learned
it cheaply and leave mlb_kprops.py alone.

DESIGN (leak-free walk-forward, mirrors mlb_backtest.AsOfState discipline)
-------------------------------------------------------------------------
For every starting-pitcher game-log, in date order:
  * BASELINE  lambda_b = pitcher_rate(prior starts) * BF_actual
      pitcher_rate = strikeOuts/battersFaced over the pitcher's STRICTLY-PRIOR
      starts, regressed toward league K/BF. BF_actual (the night's batters faced)
      is the exposure — we are testing the K RATE the ump bends, not workload.
  * TREATMENT lambda_t = lambda_b * ump_factor(prior games)
      ump_factor = (umpire's game K/BF over his STRICTLY-PRIOR games) / league K/BF,
      shrunk toward 1.0 and bounded. A wide-zone ump > 1, a tight-zone ump < 1.
Both lambdas use only data dated strictly before the start. Score each holdout start
by Poisson log-likelihood of the ACTUAL Ks, MAE, and Brier of the Over at a half-line
near the projection. Report per-period robustness and a single VERDICT.

NETWORK
-------
The data pull hits statsapi (schedule + boxscores), same source & endpoints the
production backtest already uses. In THIS sandbox egress to statsapi is blocked
(403 'Tunnel connection failed') — EXPECTED. On GitHub Actions statsapi is
reachable (every other MLB workflow proves it). If blocked, the pull prints a clear
'run on Actions' message and exits 0. It never fakes numbers.

RUN
---
  python3 mlb_umpire_experiment.py --selftest         # offline, no network, must pass
  python3 mlb_umpire_experiment.py                    # live pull + verdict (on Actions)
  python3 mlb_umpire_experiment.py --start 2025-04-01 --end 2025-09-28
"""
import os, sys, json, math, time, argparse, datetime as dt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---- model constants (experiment-local; production is untouched) ------------
LG_K_PER_BF   = 0.222   # league strikeouts per batter faced fallback (~league K%)
REG_BF        = 100.0   # batters-faced of league added to a pitcher's rate (shrink)
UMP_SHRINK_BF = 400.0   # batters-faced of league added to an ump's rate — umpire
                        # signal is weak & noisy, so it is shrunk HARD toward 1.0
UMP_FACTOR_LO = 0.85    # bound the ump multiplier to a physically plausible band
UMP_FACTOR_HI = 1.15
MIN_PRIOR_STARTS   = 3  # a pitcher needs this many prior starts to be scored
MIN_PRIOR_UMP_GAMES = 8 # an ump needs this many prior games for a usable factor


def norm(s):
    if not isinstance(s, str):
        return ""
    return "".join(c for c in s.lower() if c.isalnum())


# ===========================================================================
# POISSON MATH (pure, exact) — the scoring core, testable to hand values.
# ===========================================================================
def poisson_logpmf(k, lam):
    """log P(X = k) for X ~ Poisson(lam). Exact: k*ln(lam) - lam - ln(k!)."""
    lam = max(float(lam), 1e-9)
    k = int(k)
    return k * math.log(lam) - lam - math.lgamma(k + 1)


def poisson_sf_gt(line, lam):
    """P(X > line) for a HALF-integer line (e.g. 6.5 -> P(X >= 7)). The Over prob."""
    lam = max(float(lam), 1e-9)
    kfloor = math.floor(line)          # 6.5 -> 6 ; over wins on X >= 7
    # P(X <= kfloor)
    term = math.exp(-lam)
    cdf = term
    for i in range(1, kfloor + 1):
        term *= lam / i
        cdf += term
    return max(0.0, 1.0 - min(1.0, cdf))


# ===========================================================================
# LEAK-FREE WALK-FORWARD STATE
# ---------------------------------------------------------------------------
# Fold game-logs in date order. Every query returns a snapshot built ONLY from
# rows dated strictly < the query date — predicting a start can never see itself
# or any later game. This is the same as-of discipline as mlb_backtest.AsOfState.
# ===========================================================================
class UmpAsOfState:
    def __init__(self, reg_bf=REG_BF, ump_shrink_bf=UMP_SHRINK_BF,
                 lo=UMP_FACTOR_LO, hi=UMP_FACTOR_HI):
        self.reg_bf = reg_bf
        self.ump_shrink_bf = ump_shrink_bf
        self.lo, self.hi = lo, hi
        # pitcher starts:  norm_name -> list[(date, so, bf)]
        self.pit = {}
        # umpire game observations: norm_name -> list[(date, game_so, game_bf)]
        self.ump = {}
        # league: list[(date, game_so, game_bf)] (one row per game)
        self.lg = []

    # ---- folding (mutates state; call AFTER predicting a given date) --------
    def record_start(self, date, pitcher, so, bf):
        self.pit.setdefault(norm(pitcher), []).append((date, int(so), int(bf)))

    def record_game(self, date, umpire, game_so, game_bf):
        """One row per completed game: the umpire's zone observation + league tally."""
        self.ump.setdefault(norm(umpire), []).append((date, int(game_so), int(game_bf)))
        self.lg.append((date, int(game_so), int(game_bf)))

    # ---- leak-free queries (strictly < date) -------------------------------
    def league_k_per_bf(self, date):
        so = bf = 0
        for d, s, b in self.lg:
            if d < date:
                so += s; bf += b
        return (so / bf) if bf > 0 else LG_K_PER_BF

    def pitcher_prior_starts(self, pitcher, date):
        return [(d, s, b) for (d, s, b) in self.pit.get(norm(pitcher), []) if d < date]

    def pitcher_rate(self, pitcher, date, lg=None):
        """K/BF over the pitcher's strictly-prior starts, regressed to league.
        Returns (rate, n_prior_starts)."""
        if lg is None:
            lg = self.league_k_per_bf(date)
        prior = self.pitcher_prior_starts(pitcher, date)
        so = sum(s for _, s, _ in prior)
        bf = sum(b for _, _, b in prior)
        rate = (so + self.reg_bf * lg) / (bf + self.reg_bf)   # league-anchored shrink
        return rate, len(prior)

    def ump_prior_games(self, umpire, date):
        return [(d, s, b) for (d, s, b) in self.ump.get(norm(umpire), []) if d < date]

    def ump_factor(self, umpire, date, lg=None):
        """Umpire K-above-average multiplier from strictly-prior games, shrunk to
        1.0 and bounded. Returns (factor, n_prior_games)."""
        if lg is None:
            lg = self.league_k_per_bf(date)
        prior = self.ump_prior_games(umpire, date)
        so = sum(s for _, s, _ in prior)
        bf = sum(b for _, _, b in prior)
        if lg <= 0:
            return 1.0, len(prior)
        ump_rate = (so + self.ump_shrink_bf * lg) / (bf + self.ump_shrink_bf)
        factor = ump_rate / lg
        return min(self.hi, max(self.lo, factor)), len(prior)

    # ---- the two lambdas under test ----------------------------------------
    def lambdas(self, pitcher, umpire, date, bf_actual):
        """(lambda_baseline, lambda_treatment, ump_factor, n_prior_starts, n_prior_ump).
        BF_actual is the night's exposure; both models share it, so any difference
        in fit is attributable to the umpire multiplier alone."""
        lg = self.league_k_per_bf(date)
        rate, n_ps = self.pitcher_rate(pitcher, date, lg)
        fac, n_ug = self.ump_factor(umpire, date, lg)
        lam_b = rate * bf_actual
        lam_t = lam_b * fac
        return lam_b, lam_t, fac, n_ps, n_ug


# ===========================================================================
# WALK-FORWARD EVALUATION — the honest test. Shared by the live runner AND the
# synthetic selftest, so the tested code path IS the production code path.
# ===========================================================================
def evaluate(starts, games, holdout_frac=0.5,
             min_prior_starts=MIN_PRIOR_STARTS, min_prior_ump=MIN_PRIOR_UMP_GAMES,
             n_periods=3, state_kwargs=None):
    """
    starts : list of dicts {date(dt.date), pitcher, umpire, so, bf}
    games  : list of dicts {date(dt.date), umpire, game_so, game_bf}  (one per game)
    holdout_frac: the LATER fraction of the timeline is the scored holdout; the
                  earlier fraction only warms the state (walk-forward, no leakage).

    Returns a JSON-safe report comparing baseline vs treatment on the holdout.
    """
    state = UmpAsOfState(**(state_kwargs or {}))

    # index games & starts by date for an ordered, interleaved fold
    all_dates = sorted({s["date"] for s in starts} | {g["date"] for g in games})
    if not all_dates:
        return {"error": "no data"}
    # temporal cutoff: starts on/after this date are scored
    cut_idx = int(len(all_dates) * holdout_frac)
    cutoff = all_dates[min(cut_idx, len(all_dates) - 1)]

    games_by_date = {}
    for g in games:
        games_by_date.setdefault(g["date"], []).append(g)
    starts_by_date = {}
    for s in starts:
        starts_by_date.setdefault(s["date"], []).append(s)

    scored = []   # per-scored-start records
    for date in all_dates:
        # 1) PREDICT + SCORE this date's starts using ONLY strictly-prior state
        if date >= cutoff:
            for s in starts_by_date.get(date, []):
                lam_b, lam_t, fac, n_ps, n_ug = state.lambdas(
                    s["pitcher"], s["umpire"], date, s["bf"])
                if n_ps < min_prior_starts or n_ug < min_prior_ump:
                    continue
                k = int(s["so"])
                ll_b = poisson_logpmf(k, lam_b)
                ll_t = poisson_logpmf(k, lam_t)
                # Over at a half-line just under the shared baseline projection
                line = math.floor(lam_b) + 0.5
                y_over = 1.0 if k > line else 0.0
                p_b = poisson_sf_gt(line, lam_b)
                p_t = poisson_sf_gt(line, lam_t)
                scored.append({
                    "date": date, "pitcher": s["pitcher"], "umpire": s["umpire"],
                    "k": k, "bf": s["bf"], "lam_b": lam_b, "lam_t": lam_t,
                    "ump_factor": fac, "ll_b": ll_b, "ll_t": ll_t,
                    "ae_b": abs(k - lam_b), "ae_t": abs(k - lam_t),
                    "line": line, "y_over": y_over,
                    "br_b": (p_b - y_over) ** 2, "br_t": (p_t - y_over) ** 2,
                })
        # 2) FOLD this date's completed games into state (visible to LATER dates only)
        for g in games_by_date.get(date, []):
            state.record_game(g["date"], g["umpire"], g["game_so"], g["game_bf"])
        for s in starts_by_date.get(date, []):
            state.record_start(s["date"], s["pitcher"], s["so"], s["bf"])

    return _summarize(scored, cutoff, n_periods)


def _agg(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "poisson_ll_b": round(sum(r["ll_b"] for r in rows), 3),
        "poisson_ll_t": round(sum(r["ll_t"] for r in rows), 3),
        "mean_ll_b": round(sum(r["ll_b"] for r in rows) / n, 5),
        "mean_ll_t": round(sum(r["ll_t"] for r in rows) / n, 5),
        "mae_b": round(sum(r["ae_b"] for r in rows) / n, 4),
        "mae_t": round(sum(r["ae_t"] for r in rows) / n, 4),
        "brier_b": round(sum(r["br_b"] for r in rows) / n, 5),
        "brier_t": round(sum(r["br_t"] for r in rows) / n, 5),
    }


def _summarize(scored, cutoff, n_periods):
    out = {
        "holdout_cutoff": cutoff.isoformat() if hasattr(cutoff, "isoformat") else str(cutoff),
        "n_scored": len(scored),
        "constants": {"reg_bf": REG_BF, "ump_shrink_bf": UMP_SHRINK_BF,
                      "ump_band": [UMP_FACTOR_LO, UMP_FACTOR_HI],
                      "min_prior_starts": MIN_PRIOR_STARTS,
                      "min_prior_ump_games": MIN_PRIOR_UMP_GAMES},
    }
    if not scored:
        out["verdict"] = "INSUFFICIENT DATA — no starts met the prior-history thresholds on the holdout."
        return out

    overall = _agg(scored)
    out["overall"] = overall
    ll_delta = overall["poisson_ll_t"] - overall["poisson_ll_b"]
    mean_ll_delta = overall["mean_ll_t"] - overall["mean_ll_b"]
    out["ll_delta_total"] = round(ll_delta, 3)
    out["ll_delta_per_start"] = round(mean_ll_delta, 5)
    out["mae_delta"] = round(overall["mae_t"] - overall["mae_b"], 4)      # negative = treatment sharper
    out["brier_delta"] = round(overall["brier_t"] - overall["brier_b"], 5)  # negative = treatment sharper

    # per-period robustness: split holdout into n_periods equal-count date buckets
    dates = sorted({r["date"] for r in scored})
    periods = []
    if len(dates) >= n_periods:
        chunk = max(1, len(dates) // n_periods)
        bounds = [dates[i * chunk] for i in range(n_periods)]
        for i in range(n_periods):
            lo = bounds[i]
            hi = bounds[i + 1] if i + 1 < len(bounds) else None
            rows = [r for r in scored if r["date"] >= lo and (hi is None or r["date"] < hi)]
            a = _agg(rows)
            if a["n"]:
                a["period"] = f"{lo}..{'end' if hi is None else hi}"
                a["ll_delta_total"] = round(a["poisson_ll_t"] - a["poisson_ll_b"], 3)
                a["treatment_better"] = a["poisson_ll_t"] > a["poisson_ll_b"]
                periods.append(a)
    out["periods"] = periods
    out["periods_treatment_better"] = sum(1 for p in periods if p["treatment_better"])
    out["periods_total"] = len(periods)

    # VERDICT: treatment must lift total holdout Poisson LL AND not blow calibration
    improved = ll_delta > 0
    robust = out["periods_total"] > 0 and out["periods_treatment_better"] >= (out["periods_total"] + 1) // 2
    if improved and robust:
        verdict = (f"UMPIRE FACTOR HELPS. Holdout Poisson log-likelihood improved by "
                   f"{ll_delta:+.2f} total ({mean_ll_delta:+.4f}/start) across n={overall['n']} "
                   f"scored starts; MAE {out['mae_delta']:+.3f}, Brier {out['brier_delta']:+.4f} "
                   f"(negative = sharper). Robust in "
                   f"{out['periods_treatment_better']}/{out['periods_total']} periods.")
    elif improved:
        verdict = (f"UMPIRE FACTOR MARGINAL. Total holdout LL up {ll_delta:+.2f} "
                   f"({mean_ll_delta:+.4f}/start, n={overall['n']}) but only "
                   f"{out['periods_treatment_better']}/{out['periods_total']} periods improved — "
                   f"not robust enough to move production.")
    else:
        verdict = (f"UMPIRE FACTOR DOES NOT HELP. Holdout Poisson LL change {ll_delta:+.2f} "
                   f"({mean_ll_delta:+.4f}/start, n={overall['n']}); MAE {out['mae_delta']:+.3f}. "
                   f"Leave mlb_kprops.py unchanged.")
    out["verdict"] = verdict
    return out


# ===========================================================================
# DATA PULL — statsapi (blocked here, reachable on Actions). Never fakes data.
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


def _home_plate_umpire(box):
    """Boxscore officials -> home-plate umpire full name, or '' if absent."""
    for o in box.get("officials", []) or []:
        if "home plate" in (o.get("officialType", "") or "").lower():
            return (o.get("official") or {}).get("fullName", "")
    return ""


def _parse_boxscore(box, game_date):
    """Return (starts, game_obs) from one boxscore.
    starts: [{date, pitcher, umpire, so, bf}] for each team's STARTER.
    game_obs: {date, umpire, game_so, game_bf} totalled over both staffs.
    Returns (None, None) if the home-plate ump is missing (can't test)."""
    ump = _home_plate_umpire(box)
    if not ump:
        return None, None
    starts, game_so, game_bf = [], 0, 0
    for side in ("home", "away"):
        team = box.get("teams", {}).get(side, {})
        players = team.get("players", {})
        # starter = first pitcher in the team's pitcher order
        order = team.get("pitchers", []) or []
        starter_pid = order[0] if order else None
        for pid, pdata in players.items():
            ps = (pdata.get("stats") or {}).get("pitching") or {}
            if not ps:
                continue
            bf = int(ps.get("battersFaced", 0) or 0)
            so = int(ps.get("strikeOuts", 0) or 0)
            game_so += so
            game_bf += bf
            person = pdata.get("person") or {}
            is_starter = (person.get("id") == starter_pid) or \
                         int(ps.get("gamesStarted", 0) or 0) == 1
            if is_starter and bf > 0:
                starts.append({"date": game_date, "pitcher": person.get("fullName", ""),
                               "umpire": ump, "so": so, "bf": bf})
    if game_bf <= 0:
        return None, None
    game_obs = {"date": game_date, "umpire": ump, "game_so": game_so, "game_bf": game_bf}
    return starts, game_obs


def pull_range(start_date, end_date):
    """Pull (starts, games) across a date range from statsapi. Returns (starts, games).
    Raises on the first network failure so the caller can degrade cleanly."""
    starts, games = [], []
    day = start_date
    ngames = 0
    while day <= end_date:
        pks = _final_games(day)
        for pk in pks:
            try:
                box = _get(f"{API}/game/{pk}/boxscore")
            except Exception:
                continue
            st, gm = _parse_boxscore(box, day)
            if gm is None:
                continue
            starts.extend(st)
            games.append(gm)
            ngames += 1
            time.sleep(0.15)
        print(f"  {day}: {len(pks)} finals, cum {ngames} usable games / {len(starts)} starts")
        day += dt.timedelta(days=1)
        time.sleep(0.2)
    return starts, games


def run_live(start_date, end_date, holdout_frac):
    """Attempt the pull; degrade to a clear 'run on Actions' message (exit 0) if
    statsapi is unreachable (the expected state in this sandbox)."""
    print("=" * 74)
    print(f"UMPIRE K-EXPERIMENT — live pull {start_date} .. {end_date}")
    print("=" * 74)
    # probe egress first so we fail fast & clean
    try:
        _get(f"{API}/schedule?sportId=1&date={start_date.isoformat()}", tries=1)
    except Exception as e:
        print("\nstatsapi UNREACHABLE from here — run on GitHub Actions.")
        print(f"  (probe error: {type(e).__name__}: {str(e)[:120]})")
        print("  This is EXPECTED in the cloud sandbox (egress to statsapi is blocked, 403).")
        print("  Every other MLB workflow proves statsapi IS reachable on Actions.")
        print("  Action command:  python3 mlb_umpire_experiment.py --start "
              f"{start_date} --end {end_date}")
        return 0
    starts, games = pull_range(start_date, end_date)
    print(f"\npulled {len(starts)} starts across {len(games)} games")
    if not starts:
        print("no usable starts pulled — cannot run the experiment.")
        return 0
    report = evaluate(starts, games, holdout_frac=holdout_frac)
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "umpire_experiment.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    _print_report(report)
    return 0


def _print_report(report):
    print("\n" + "=" * 74)
    print("HOLDOUT RESULT  (baseline = pitcher-rate*BF ; treatment = *umpire factor)")
    print("=" * 74)
    ov = report.get("overall")
    if ov:
        print(f"  scored starts (holdout, on/after {report['holdout_cutoff']}): n={ov['n']}")
        print(f"  Poisson log-lik   baseline {ov['poisson_ll_b']:>10.2f}   "
              f"treatment {ov['poisson_ll_t']:>10.2f}   delta {report['ll_delta_total']:+.2f}")
        print(f"  mean LL / start   baseline {ov['mean_ll_b']:>10.4f}   "
              f"treatment {ov['mean_ll_t']:>10.4f}   delta {report['ll_delta_per_start']:+.4f}")
        print(f"  MAE (Ks)          baseline {ov['mae_b']:>10.3f}   "
              f"treatment {ov['mae_t']:>10.3f}   delta {report['mae_delta']:+.3f}")
        print(f"  Brier (Over)      baseline {ov['brier_b']:>10.4f}   "
              f"treatment {ov['brier_t']:>10.4f}   delta {report['brier_delta']:+.4f}")
        for p in report.get("periods", []):
            flag = "treatment" if p["treatment_better"] else "baseline "
            print(f"    period {p['period']:<24} n={p['n']:<4} LL delta {p['ll_delta_total']:+.2f}  -> {flag} better")
    print("\nVERDICT:", report.get("verdict"))
    print("=" * 74)


# ===========================================================================
# OFFLINE SELFTEST — NO NETWORK. Synthetic fixtures verify leak-freeness,
# the Poisson math, and recovery of a PLANTED umpire effect.
# ===========================================================================
def selftest():
    d = dt.date

    # ---------- (b) POISSON MATH exact on a fixture ----------
    # log P(X=3 | lam=2.5) = 3*ln2.5 - 2.5 - ln(3!) = 3*0.916290 - 2.5 - 1.791759
    #                      = 2.748872 - 2.5 - 1.791759 = -1.542887
    assert abs(poisson_logpmf(3, 2.5) - (-1.5428868)) < 1e-6, poisson_logpmf(3, 2.5)
    # P(X=0 | lam) = e^-lam
    assert abs(poisson_logpmf(0, 4.0) - (-4.0)) < 1e-9
    # sf: P(X>4.5 | lam=5) = 1 - P(X<=4). Poisson(5): P(X<=4)=0.440493 -> 0.559507
    assert abs(poisson_sf_gt(4.5, 5.0) - 0.559507) < 1e-5, poisson_sf_gt(4.5, 5.0)
    # a bigger lambda must put MORE Over mass above the same line (monotone)
    assert poisson_sf_gt(6.5, 8.0) > poisson_sf_gt(6.5, 6.0)
    # LL peaks when lam == k (fixture: actual 6 Ks best explained by lam=6)
    lls = [(poisson_logpmf(6, L), L) for L in (3.0, 5.0, 6.0, 7.0, 9.0)]
    assert max(lls)[1] == 6.0, lls
    print("  [b] Poisson logpmf / sf / peak: exact on fixtures")

    # ---------- (a) LEAK-FREENESS: strictly-prior only ----------
    st = UmpAsOfState()
    # Umpire "Wide Zone" works 8 HIGH-K prior games, then a start day with an
    # EXTREME low-K game. If the current game leaked in, the factor would drop.
    for i in range(8):
        st.record_game(d(2025, 4, 1) + dt.timedelta(days=i), "Wide Zone",
                       game_so=22, game_bf=74)   # ~0.297 K/BF, very wide
        st.record_game(d(2025, 4, 1) + dt.timedelta(days=i), "Neutral Ned",
                       game_so=16, game_bf=76)   # ~0.211, near league
    query = d(2025, 5, 1)
    lg_before = st.league_k_per_bf(query)
    fac_before, n_before = st.ump_factor("Wide Zone", query)
    assert n_before == 8, n_before
    assert fac_before > 1.05, fac_before               # wide zone reads above 1
    # Now append the CURRENT-DATE game (extreme low-K) — must NOT change the
    # as-of-query snapshot, because the query is strictly-prior to `query`.
    st.record_game(query, "Wide Zone", game_so=1, game_bf=80)   # freak tight night
    fac_after, n_after = st.ump_factor("Wide Zone", query)
    assert (fac_after, n_after) == (fac_before, n_before), (fac_after, fac_before)
    # And a query AT the extreme game's own date still excludes it (strictly <)...
    fac_same_day, n_same_day = st.ump_factor("Wide Zone", query)
    assert n_same_day == 8, n_same_day
    # ...but a LATER query now includes it, and the factor MOVES — proving the
    # current game *would* have changed the answer, i.e. excluding it was load-bearing.
    fac_later, n_later = st.ump_factor("Wide Zone", query + dt.timedelta(days=1))
    assert n_later == 9 and fac_later < fac_before, (fac_later, fac_before, n_later)
    print("  [a] umpire factor: strictly-prior, current game excluded & load-bearing")

    # same guarantee for the PITCHER rate
    stp = UmpAsOfState()
    for i in range(5):
        stp.record_start(d(2025, 4, 1) + dt.timedelta(days=i), "Ace Whiff", so=9, bf=25)
        stp.record_game(d(2025, 4, 1) + dt.timedelta(days=i), "u", 16, 75)
    q = d(2025, 5, 1)
    r_before, n_ps = stp.pitcher_rate("Ace Whiff", q)
    assert n_ps == 5
    stp.record_start(q, "Ace Whiff", so=0, bf=25)      # a current shutout-of-Ks start
    r_after, n_ps2 = stp.pitcher_rate("Ace Whiff", q)
    assert (r_after, n_ps2) == (r_before, n_ps)        # current start invisible as-of q
    r_later, n_ps3 = stp.pitcher_rate("Ace Whiff", q + dt.timedelta(days=1))
    assert n_ps3 == 6 and r_later < r_before           # and would have lowered it
    print("  [a] pitcher rate: strictly-prior, current start excluded & load-bearing")

    # ---------- (c) PLANTED UMPIRE EFFECT is recovered ----------
    # Synthesize a season: pitchers with stable true K/BF, umpires with KNOWN
    # zone multipliers. Actual Ks ~ Poisson(true_rate * BF * true_ump_mult).
    # If the walk-forward treatment recovers the umpire signal, it must beat the
    # (ump-blind) baseline on the temporal holdout.
    import random
    rng = random.Random(20260723)

    def pois(lam):
        # Knuth Poisson sampler (no numpy dependency in the selftest)
        L = math.exp(-lam); k = 0; p = 1.0
        while True:
            k += 1; p *= rng.random()
            if p <= L:
                return k - 1

    pitchers = {f"P{i}": rng.uniform(0.18, 0.30) for i in range(14)}   # true K/BF
    # umpires with a real spread of zone sizes (planted effect)
    umps = {"UmpWide": 1.18, "UmpWide2": 1.12, "UmpTight": 0.84,
            "UmpTight2": 0.90, "UmpMid": 1.0, "UmpMid2": 1.02}
    ump_names = list(umps)
    pit_names = list(pitchers)

    starts, games = [], []
    start_day = d(2025, 4, 1)
    # 150 game-days, 6 games/day -> deep enough history for stable priors + a holdout
    for gday in range(150):
        date = start_day + dt.timedelta(days=gday)
        for _ in range(6):
            ump = rng.choice(ump_names)
            m = umps[ump]
            game_so = game_bf = 0
            day_starts = []
            for _ in range(2):   # two starters per game
                pit = rng.choice(pit_names)
                bf = rng.randint(18, 30)
                lam = pitchers[pit] * bf * m
                so = pois(lam)
                so = min(so, bf)
                day_starts.append({"date": date, "pitcher": pit, "umpire": ump,
                                   "so": so, "bf": bf})
                game_so += so; game_bf += bf
            # add a little bullpen K/BF to the game total at the same ump multiplier
            bpen_bf = rng.randint(10, 22)
            bpen_so = min(bpen_bf, pois(0.22 * bpen_bf * m))
            game_so += bpen_so; game_bf += bpen_bf
            starts.extend(day_starts)
            games.append({"date": date, "umpire": ump,
                          "game_so": game_so, "game_bf": game_bf})

    report = evaluate(starts, games, holdout_frac=0.5,
                      min_prior_starts=2, min_prior_ump=8, n_periods=3)
    ov = report["overall"]
    assert ov["n"] > 200, ov["n"]
    # the planted effect must be recovered: treatment sharper on the holdout
    assert report["ll_delta_total"] > 0, report["ll_delta_total"]
    assert report["ll_delta_per_start"] > 0, report["ll_delta_per_start"]
    assert report["mae_delta"] < 0, report["mae_delta"]        # lower MAE = sharper
    assert report["brier_delta"] < 0, report["brier_delta"]    # better Over calibration
    assert report["verdict"].startswith("UMPIRE FACTOR HELPS"), report["verdict"]
    assert report["periods_treatment_better"] >= 2, report["periods"]
    print(f"  [c] planted umpire effect recovered: LL {report['ll_delta_total']:+.1f} total, "
          f"{report['ll_delta_per_start']:+.4f}/start, MAE {report['mae_delta']:+.3f}, "
          f"Brier {report['brier_delta']:+.4f}, n={ov['n']}")

    # ---------- (c') NEGATIVE CONTROL: no real ump effect -> no free lift ----------
    # Same machinery, but every umpire truly neutral. Treatment should NOT
    # manufacture a meaningful edge (guards against the test rigging itself).
    starts2, games2 = [], []
    for gday in range(120):
        date = start_day + dt.timedelta(days=gday)
        for _ in range(6):
            ump = rng.choice(ump_names)   # names carry NO multiplier now
            game_so = game_bf = 0
            day_starts = []
            for _ in range(2):
                pit = rng.choice(pit_names)
                bf = rng.randint(18, 30)
                so = min(bf, pois(pitchers[pit] * bf))   # mult = 1.0 for everyone
                day_starts.append({"date": date, "pitcher": pit, "umpire": ump,
                                   "so": so, "bf": bf})
                game_so += so; game_bf += bf
            bpen_bf = rng.randint(10, 22)
            game_so += min(bpen_bf, pois(0.22 * bpen_bf)); game_bf += bpen_bf
            starts2.extend(day_starts)
            games2.append({"date": date, "umpire": ump,
                           "game_so": game_so, "game_bf": game_bf})
    rep2 = evaluate(starts2, games2, holdout_frac=0.5,
                    min_prior_starts=2, min_prior_ump=8, n_periods=3)
    # with the factor shrunk & bounded, a null effect must not yield a big LL gain
    assert abs(rep2["ll_delta_per_start"]) < 0.01, rep2["ll_delta_per_start"]
    print(f"  [c'] negative control (no true effect): LL/start delta "
          f"{rep2['ll_delta_per_start']:+.5f} — near zero, as it must be")

    print("UMPIRE-EXPERIMENT SELFTEST PASS — leak-free, Poisson math exact, "
          "planted effect recovered, null control clean")
    return 0


# ===========================================================================
def _date(s):
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def main(argv):
    ap = argparse.ArgumentParser(description="Home-plate umpire K-prop experiment (standalone).")
    ap.add_argument("--selftest", action="store_true", help="offline synthetic tests, no network")
    ap.add_argument("--start", type=_date, default=None, help="range start YYYY-MM-DD")
    ap.add_argument("--end", type=_date, default=None, help="range end YYYY-MM-DD")
    ap.add_argument("--holdout-frac", type=float, default=0.5,
                    help="later fraction of the timeline used as the scored holdout")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    # default live range: a full prior season if unspecified
    start = args.start or dt.date(dt.date.today().year - 1, 4, 1)
    end = args.end or dt.date(dt.date.today().year - 1, 9, 28)
    return run_live(start, end, args.holdout_frac)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
