#!/usr/bin/env python3
"""
mlb_contactform_experiment.py — does ROLLING CONTACT QUALITY (recent barrels /
exit velo) add HR signal beyond the season line and the hot-hand flag?

The production board already carries: season+Marcel batter rate, Savant season
barrels, and the validated hot-hand flag (homered in LAST graded game, +0.18
logit). This tests the layer between those: a hitter whose last 1-3 WEEKS of
batted balls are barreled above his own norm — "quietly scorching" form that
hasn't shown up as HRs yet.

DATA: reuses the hrangles batter-game dataset (2024 burn-in + Apr-Jun 2025,
outcomes/PA/park/hands) and adds a per-batter-DAY contact table from Savant
statcast via pybaseball, aggregated on the fly to (batted balls, barrels,
hard-hit) per day — pitch data is discarded immediately, the cache is small.
launch_speed_angle == 6 is Savant's barrel class.

TEST: baseline = the run-2 tuned production analog (batter rate x park x flat
platoon x prior-seeded pitcher HR factor). Variant multiplies by a contact-form
factor: barrel rate over the trailing W days vs the batter's own trailing-200d
norm, shrunk by tau_c batted balls, damped by w_c. W/tau_c/w_c tuned on TRAIN
(Apr-May 2025) only; verdict on June, three sub-periods, ROBUST WIN or bust.
A hard-hit-rate twin runs alongside (barrels are rarer; HH stabilizes faster).

Leak rules identical to hrangles: all features from strictly prior days;
selftest proves planted-effect recovery, null control, and future-poisoning
byte-identity. Offline: prints UNREACHABLE and exits 0 (fire on Actions via
experiments/RUN-CONTACTFORM.txt).
"""
import json, math, os, sys, datetime as dt
from collections import defaultdict

import mlb_hrangles_experiment as HX

HERE = os.path.dirname(os.path.abspath(__file__))
CCACHE = os.path.join(HERE, "data", "contact_daily.json")
START, TRAIN_END, END = HX.START, HX.TRAIN_END, HX.END
PULLS = [("2024-03-20", "2024-09-30"), ("2025-03-27", "2025-06-30")]

# ------------------------------------------------------------ contact pulls
def build_contact():
    """Savant statcast via pybaseball, weekly chunks, aggregated per batter-day
    to [batted_balls, barrels, hard_hit] and cached. Pitch rows discarded."""
    from pybaseball import statcast
    daily = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for d0, d1 in PULLS:
        cur = dt.date.fromisoformat(d0)
        stop = dt.date.fromisoformat(d1)
        while cur <= stop:
            wend = min(cur + dt.timedelta(days=6), stop)
            try:
                df = statcast(start_dt=cur.isoformat(), end_dt=wend.isoformat(),
                              verbose=False)
                bb = df[(df["type"] == "X") & df["launch_speed"].notna()]
                for _, r in bb.iterrows():
                    cell = daily[int(r["batter"])][str(r["game_date"])[:10]]
                    cell[0] += 1
                    if r.get("launch_speed_angle") == 6:
                        cell[1] += 1
                    if r["launch_speed"] >= 95:
                        cell[2] += 1
                print(f"  {cur}..{wend}: {len(bb)} batted balls")
            except Exception as e:
                print(f"  {cur}..{wend}: {type(e).__name__} — week skipped")
            cur = wend + dt.timedelta(days=1)
    out = {str(b): dict(days) for b, days in daily.items()}
    os.makedirs(os.path.dirname(CCACHE), exist_ok=True)
    with open(CCACHE, "w") as f:
        json.dump(out, f)
    print(f"contact cache -> {CCACHE} ({len(out)} batters)")
    return out

# ------------------------------------------------------------ form features
def contact_series(contact):
    """{batter_id: sorted [(date, bbe, brl, hh)]}"""
    out = {}
    for b, days in contact.items():
        out[int(b)] = sorted((d, v[0], v[1], v[2]) for d, v in days.items())
    return out

def form_lookup(series, bat, date, window, norm_days=200):
    """(win_bbe, win_brl, win_hh, norm_bbe, norm_brl, norm_hh) — all STRICTLY
    before `date`. Returns None when the batter has no prior batted balls."""
    s = series.get(bat)
    if not s:
        return None
    d1 = dt.date.fromisoformat(date)
    w0 = (d1 - dt.timedelta(days=window)).isoformat()
    n0 = (d1 - dt.timedelta(days=norm_days)).isoformat()
    wb = wr = wh = nb = nr = nh = 0
    for d, bbe, brl, hh in s:
        if d >= date:
            break
        if d >= n0:
            nb += bbe; nr += brl; nh += hh
        if d >= w0:
            wb += bbe; wr += brl; wh += hh
    if nb == 0:
        return None
    return (wb, wr, wh, nb, nr, nh)

def form_factor(fl, P, metric):
    """Shrunk ratio of windowed rate vs the batter's own norm."""
    if fl is None:
        return 1.0
    wb, wr, wh, nb, nr, nh = fl
    num = wr if metric == "brl" else wh
    den = nr if metric == "brl" else nh
    norm_rate = den / nb if nb else 0.0
    if norm_rate <= 0:
        return 1.0
    win_rate = (num + P["tau_c"] * norm_rate) / (wb + P["tau_c"])
    return 1.0 + P["w_c"] * (win_rate / norm_rate - 1.0)

# ------------------------------------------------------------ evaluation
BASE_USE = {"flat_platoon", "pitcher_hr"}          # run-2 production analog

def loglik_form(feat_rows, series, P0, Pf, metric, window, d0, d1):
    tot = n = 0.0
    cov = 0
    for r, f in feat_rows:
        if not (d0 <= r["date"] <= d1):
            continue
        v = HX.rate_pa(r, f, P0, BASE_USE)
        if Pf is not None:
            fl = form_lookup(series, r["bat"], r["date"], window)
            if fl is not None:
                cov += 1
            v *= form_factor(fl, Pf, metric)
        lam = max(v, 1e-6) * r["pa"]
        p = min(max(1.0 - math.exp(-lam), 1e-9), 1 - 1e-9)
        y = 1 if r["hr"] > 0 else 0
        tot += y * math.log(p) + (1 - y) * math.log(1 - p)
        n += 1
    return (tot / n if n else 0.0), int(n), cov

def run(feat_rows, series, out=print):
    P0 = dict(HX.BASE_P, tau_b=75, tau_park=800, w_p=0.6)   # run-2 tuned
    base_tr, ntr, _ = loglik_form(feat_rows, series, P0, None, "", 0, START, TRAIN_END)
    base_h, nh, _ = loglik_form(feat_rows, series, P0, None, "", 0, "2025-06-01", END)
    out(f"baseline (incl. prior-seeded pitcher HR)  TRAIN {base_tr:+.5f} (n={ntr})  "
        f"HOLDOUT {base_h:+.5f} (n={nh})")
    periods = [("2025-06-01", "2025-06-10"), ("2025-06-11", "2025-06-20"),
               ("2025-06-21", END)]
    results = {}
    for metric in ("brl", "hh"):
        bt = (-9e9, None, None)
        for W in (7, 14, 21):
            for tau_c in (10, 25, 50):
                for w_c in (0.3, 0.6, 1.0):
                    Pf = {"tau_c": tau_c, "w_c": w_c}
                    ll, _, _ = loglik_form(feat_rows, series, P0, Pf, metric, W,
                                           START, TRAIN_END)
                    if ll > bt[0]:
                        bt = (ll, Pf, W)
        ll_tr, Pf, W = bt
        train_win = ll_tr > base_tr
        hv, _, cov = loglik_form(feat_rows, series, P0, Pf, metric, W,
                                 "2025-06-01", END)
        wins = 0
        for p0, p1 in periods:
            b, _, _ = loglik_form(feat_rows, series, P0, None, "", 0, p0, p1)
            v, _, _ = loglik_form(feat_rows, series, P0, Pf, metric, W, p0, p1)
            wins += 1 if v > b else 0
        verdict = ("ROBUST WIN" if (train_win and hv > base_h and wins == 3)
                   else ("win, not robust" if hv > base_h else "NULL"))
        results[metric] = (hv - base_h, wins, verdict, Pf, W)
        out(f"{'BARRELS' if metric == 'brl' else 'HARD-HIT':8s} W={W}d {Pf}  "
            f"train_win={train_win}  holdout dLL {hv-base_h:+.5f}  "
            f"periods {wins}/3  coverage {cov}/{nh}  -> {verdict}")
    return results

# ------------------------------------------------------------ selftest
def selftest():
    import random
    rng = random.Random(11)
    HX.START, HX.TRAIN_END, HX.END = "2025-04-01", "2025-05-15", "2025-06-06"
    global START, TRAIN_END, END
    START, TRAIN_END, END = HX.START, HX.TRAIN_END, HX.END
    # DENSE synthetic panel: 150 regulars playing daily. (HX._synth's random
    # assignment gives each batter ~17 batted balls per 14-day window — too
    # sparse for ANY form detector; that tests starvation, not machinery.)
    # Hot state is per-batter Markov (mean run ~25d, unsynchronized): hot
    # doubles BOTH barrel rate and true HR rate, so the ratio->multiplier
    # mapping the factor fits is honest.
    rows, daily = [], defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    days = []
    d = dt.date(2025, 4, 1)
    while d <= dt.date(2025, 6, 6):
        days.append(d.isoformat()); d += dt.timedelta(days=1)
    for b in range(150):
        srng = random.Random(b * 977 + 3)
        hot = srng.random() < 0.2
        for di, date in enumerate(days):
            if srng.random() < 0.04:
                hot = not hot
            pa = 4
            p_pa = 0.031 * (2.0 if hot else 1.0)
            hr = 1 if srng.random() < 1 - (1 - p_pa) ** pa else 0
            bbe = 3
            brate = 0.16 if hot else 0.08
            brl = sum(1 for _ in range(bbe) if srng.random() < brate)
            cell = daily[b][date]
            cell[0] += bbe; cell[1] += brl; cell[2] += brl
            rows.append({"date": date, "pk": di * 1000 + b, "venue": b % 10,
                         "dn": "night", "home": b % 2 == 0, "bat": b,
                         "name": f"B{b}", "slot": b % 9 + 1, "pa": pa,
                         "hr": hr, "sp": 9000 + (b + di) % 60,
                         "bh": "R", "ph": "R"})
    series = contact_series({str(b): d_ for b, d_ in daily.items()})
    feat = HX.attach_features(rows)
    # Machinery claim at a FIXED canonical config over the FULL sample (the
    # real run keeps strict tune-on-train / verdict-on-June discipline).
    P0 = dict(HX.BASE_P, tau_b=75, tau_park=800, w_p=0.6)
    PF = {"tau_c": 25, "w_c": 0.6}
    b_full, nf, _ = loglik_form(feat, series, P0, None, "", 0, START, END)
    v_full, _, cov = loglik_form(feat, series, P0, PF, "brl", 14, START, END)
    d_brl = v_full - b_full
    assert cov > nf * 0.8, f"coverage broken: {cov}/{nf}"
    # +0.0006 threshold is deliberate: even a TRUE x2 form effect through a
    # 14-day/~40-BBE window only yields ~+0.001 LL/game — the window is the
    # bottleneck, which is exactly why the real verdict may honestly be NULL.
    assert d_brl > 0.0006, f"planted hot-form NOT recovered: {d_brl}"
    # null: contact tables shuffled across batters -> factor is noise
    ids = list(daily.keys())
    perm = ids[:]; rng.shuffle(perm)
    swapped = {str(perm[i]): daily[ids[i]] for i in range(len(ids))}
    v_null, _, _ = loglik_form(feat, contact_series(swapped), P0, PF, "brl", 14,
                               START, END)
    d_null = v_null - b_full
    assert d_null < d_brl / 3, f"null control suspicious: {d_null} vs {d_brl}"
    # leak: form_lookup must ignore same-day and future contact
    s = contact_series({"77": {"2025-05-01": [4, 2, 2], "2025-05-02": [4, 4, 4],
                               "2025-04-20": [5, 1, 1]}})
    fl = form_lookup(s, 77, "2025-05-02", 14)
    assert fl == (9, 3, 3, 9, 3, 3), fl               # 05-02 excluded, 05-01+04-20 in
    fl2 = form_lookup(s, 77, "2025-04-20", 14)
    assert fl2 is None                                 # nothing strictly before
    print(f"CONTACTFORM SELFTEST PASS — planted form recovered (dLL {d_brl:+.4f}, "
          f"null {d_null:+.4f}), shuffled-null clean, same-day/future contact excluded")
    return 0

def main():
    if not (os.path.exists(HX.CACHE) and os.path.exists(HX.BURN_CACHE)):
        print("hrangles dataset caches missing — run the HR angles experiment first")
        return 0
    if os.path.exists(CCACHE):
        contact = json.load(open(CCACHE))
        print(f"contact cache: {len(contact)} batters")
    else:
        try:
            import urllib.request
            urllib.request.urlopen("https://baseballsavant.mlb.com", timeout=10)
        except Exception:
            print("Savant UNREACHABLE from this network — run on GitHub Actions "
                  "(touch experiments/RUN-CONTACTFORM.txt)")
            return 0
        contact = build_contact()
    ds = json.load(open(HX.CACHE)); burn = json.load(open(HX.BURN_CACHE))
    rows = [r for r in burn["rows"] + ds["rows"]
            if r.get("bh") and r.get("ph") and r.get("slot", 0) > 0]
    print(f"scorable rows: {len(rows)}")
    feat = HX.attach_features(rows)
    series = contact_series(contact)
    lines = []
    def tee(s):
        print(s); lines.append(s)
    tee("=" * 70)
    tee(f"CONTACT-FORM EXPERIMENT — rolling barrels/hard-hit vs own norm")
    tee(f"baseline includes the shipped prior-seeded pitcher HR factor")
    tee("=" * 70)
    run(feat, series, out=tee)
    tee("Ship rule: ROBUST WIN only (train win + holdout win + 3/3 periods).")
    vd = os.path.join(HERE, "..", "experiments", "MLB-CONTACTFORM-VERDICT.md")
    with open(vd, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"verdict -> {vd}")
    return 0

if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
