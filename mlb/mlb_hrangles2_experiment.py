#!/usr/bin/env python3
"""
mlb_hrangles2_experiment.py — three angles the board has never seen, plus the
first POWER CEILINGS on the HR panel.

Everything here comes out of the batter-game panel that is already cached
(data/hrangles_dataset.json + the 2024 burn-in). No new pull, which is the
point: the three listed BLIND SPOTS below were never tested because they
looked like they needed schedule and travel data, and they do not.

  G  TRAVEL      — did this batter's team change venues since his last game?
                   "travel/getaway days" is a listed blind spot. A batter's
                   previous game's venue is in the panel, so the flag is a
                   two-line derivation, not a data project.
  H  HOME/AWAY   — the home bump BEYOND the park factor. The park factor is
                   symmetric by construction: it says Coors is Coors for
                   everybody. This asks whether the home batter gets anything
                   extra — sleeping in his own bed, hitting in the bottom half
                   with the game state known, his own batter's eye.
  I  SLOT SHIFT  — did the manager move him UP or DOWN the order versus his
                   last start? This is the only feature in the whole board
                   that is somebody's OPINION rather than a statistic. The
                   manager watched batting practice and watched the swing; if
                   he is reacting to something the box score has not shown
                   yet, a lineup promotion is a leading indicator and the
                   model is throwing it away. (Slot LEVEL is already in the
                   board via PA share; slot CHANGE is not.)

WHY CEILINGS, AND WHY THIS PANEL NEEDS THEM MOST. A ~24k-row, three-month HR
panel with an 8k holdout is a thin instrument, and "NULL" from a thin
instrument is not a finding — it is a shrug. Every angle here is therefore
scored against a re-rolled panel in which the effect IS true at a generous
strength. Two numbers come back:

  ORACLE   what a model that already KNEW the true multiplier would gain.
           No fitted model can beat this. If the oracle is +0.0002, the panel
           physically cannot answer the question and a null means nothing.
  FITTED   what this exact tune-and-verdict pipeline actually recovers from a
           panel where the effect is real. This is the honest detection
           threshold, and it is always below the oracle because the pipeline
           has to estimate the thing.

A measured result at or above FITTED-under-truth, with the effect real, is
what a discovery looks like. A measured null well below it is a real null.
A measured null ABOVE it, or an oracle near zero, means the experiment was
never capable of answering and the angle stays open.

Selftest: planted effects recovered for all three angles, null control clean,
future-poisoning leak proof, and the ceiling is checked for the one property
that makes it a ceiling — it must not sit below what the same pipeline
actually measures on the same planted panel.

Run: python mlb_hrangles2_experiment.py [--selftest]
"""
import json
import math
import os
import random
import sys
from collections import defaultdict

import mlb_hrangles_experiment as H

HERE = os.path.dirname(os.path.abspath(__file__))
START, TRAIN_END, END = H.START, H.TRAIN_END, H.END
HOLD0 = "2025-06-01"
PERIODS = [("2025-06-01", "2025-06-10"), ("2025-06-11", "2025-06-20"),
           ("2025-06-21", END)]

# how far apart two appearances can be and still count as the same trip. A
# batter who sat three weeks and came back in another city did not "travel",
# he returned; lumping the two together would put a rehab stint in the travel
# bucket and call the result fatigue.
TRIP_GAP_DAYS = 4


def _d(s):
    y, m, dd = (int(x) for x in s.split("-"))
    return y * 10000 + m * 100 + dd


def _daygap(a, b):
    """Calendar days between two ISO dates. Small ranges only — this panel is
    three months plus a burn-in, so an exact date library buys nothing."""
    import datetime as dt
    ya, ma, da = (int(x) for x in a.split("-"))
    yb, mb, db = (int(x) for x in b.split("-"))
    return (dt.date(yb, mb, db) - dt.date(ya, ma, da)).days


# ------------------------------------------------------------------ features
def attach2(rows):
    """Base features from the shipped module, plus the three new cells.

    The base pass is REUSED rather than reimplemented. A second copy of the
    baseline would drift from the one the earlier verdicts were written
    against, and then batch-2 results would not be comparable to batch-1
    results — which is the entire reason to run a batch 2.

    Merging is by object identity, not by position: attach_features hands back
    the very same row dicts it was given, so id() is exact. Position would be
    correct today and silently wrong the first time either loop's ordering is
    touched.
    """
    base = H.attach_features(rows)
    extra = {}
    lg = [0, 0]
    trav = defaultdict(lambda: [0, 0])     # 'new' / 'same'
    home = defaultdict(lambda: [0, 0])     # 'home' / 'away'
    shift = defaultdict(lambda: [0, 0])    # 'up' / 'flat' / 'down'
    prev = {}                              # batter -> (date, venue, slot)
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["date"]].append(r)
    for day in sorted(by_day):
        todays = by_day[day]
        for r in todays:
            p = prev.get(r["bat"])
            tk = None
            sk = None
            if p and _daygap(p[0], day) <= TRIP_GAP_DAYS:
                tk = "new" if p[1] != r["venue"] else "same"
                if p[2] and r["slot"]:
                    dlt = r["slot"] - p[2]
                    sk = "up" if dlt <= -2 else ("down" if dlt >= 2 else "flat")
            extra[id(r)] = {
                "trav_k": tk,
                "trav": tuple(trav[tk]) if tk else (0, 0),
                "home_k": "home" if r["home"] else "away",
                "home": tuple(home["home" if r["home"] else "away"]),
                "shift_k": sk,
                "shift": tuple(shift[sk]) if sk else (0, 0),
                "lg2": tuple(lg),
            }
        for r in todays:                   # update AFTER scoring the day
            p = prev.get(r["bat"])
            lg[0] += r["hr"]
            lg[1] += r["pa"]
            home["home" if r["home"] else "away"][0] += r["hr"]
            home["home" if r["home"] else "away"][1] += r["pa"]
            if p and _daygap(p[0], day) <= TRIP_GAP_DAYS:
                tk = "new" if p[1] != r["venue"] else "same"
                trav[tk][0] += r["hr"]
                trav[tk][1] += r["pa"]
                if p[2] and r["slot"]:
                    dlt = r["slot"] - p[2]
                    sk = "up" if dlt <= -2 else ("down" if dlt >= 2 else "flat")
                    shift[sk][0] += r["hr"]
                    shift[sk][1] += r["pa"]
            prev[r["bat"]] = (day, r["venue"], r["slot"])
    assert len(extra) == len(base), "feature merge lost rows"
    return [(r, dict(f, **extra[id(r)])) for r, f in base]


# -------------------------------------------------------------------- scoring
ANGLES = {
    #  name                use-flag     cell   knob   grid
    "G travel":     ("travel", "trav", "tau_tr", (300, 1000, 3000, 8000)),
    "H home/away":  ("homeaway", "home", "tau_ha", (1000, 3000, 8000, 20000)),
    "I slot shift": ("slotshift", "shift", "tau_sh", (300, 1000, 3000, 8000)),
}
BASE_P = dict(H.BASE_P, tau_tr=3000, tau_ha=8000, tau_sh=3000)


def rate_pa2(r, f, P, use):
    """Baseline rate from the shipped module, times at most one new factor.

    ORACLE mode multiplies by the multiplier that GENERATED the data instead
    of one estimated from it. That is the only honest way to draw a ceiling:
    it answers "how much is there to win", separately from "how much can this
    pipeline find", and the gap between the two is the pipeline's tax.
    """
    v = H.rate_pa(r, f, P, use)
    lg_hr, lg_pa = f["lg2"]
    lgr = lg_hr / lg_pa if lg_pa > 200 else 0.031
    if "oracle" in use:
        return v * r.get("_mult", 1.0)
    for name, (flag, cell, knob, _grid) in ANGLES.items():
        if flag in use and f.get(cell + "_k"):
            v *= H._factor(f[cell], lgr, P[knob])
    return v


def loglik2(feat_rows, P, use, d0, d1):
    tot = n = 0.0
    for r, f in feat_rows:
        if not (d0 <= r["date"] <= d1):
            continue
        lam = max(rate_pa2(r, f, P, use), 1e-6) * r["pa"]
        p = min(max(1.0 - math.exp(-lam), 1e-9), 1 - 1e-9)
        y = 1 if r["hr"] > 0 else 0
        tot += y * math.log(p) + (1 - y) * math.log(1 - p)
        n += 1
    return (tot / n if n else 0.0), int(n)


def fit_baseline(feat_rows):
    best = (-9e9, None)
    for tb in (75, 150, 300):
        for tp in (800, 1500, 3000):
            P = dict(BASE_P, tau_b=tb, tau_park=tp)
            ll, _ = loglik2(feat_rows, P, {"flat_platoon"}, START, TRAIN_END)
            if ll > best[0]:
                best = (ll, P)
    return best[1], best[0]


def verdict(feat_rows, out=print):
    P0, base_tr = fit_baseline(feat_rows)
    out(f"baseline (bat x park x flat platoon) tau_b={P0['tau_b']} "
        f"tau_park={P0['tau_park']}  TRAIN LL {base_tr:+.5f}")
    base_h, n_h = loglik2(feat_rows, P0, {"flat_platoon"}, HOLD0, END)
    out(f"baseline HOLDOUT LL {base_h:+.5f} (n={n_h})")
    res = {}
    for name, (flag, cell, knob, grid) in ANGLES.items():
        bt = (-9e9, None)
        for g in grid:
            P = dict(P0)
            P[knob] = g
            ll, _ = loglik2(feat_rows, P, {"flat_platoon", flag},
                            START, TRAIN_END)
            if ll > bt[0]:
                bt = (ll, P)
        Pv = bt[1]
        hv, _ = loglik2(feat_rows, Pv, {"flat_platoon", flag}, HOLD0, END)
        wins = 0
        for p0, p1 in PERIODS:
            b, _ = loglik2(feat_rows, P0, {"flat_platoon"}, p0, p1)
            v_, _ = loglik2(feat_rows, Pv, {"flat_platoon", flag}, p0, p1)
            wins += 1 if v_ > b else 0
        vd = ("ROBUST WIN" if (bt[0] > base_tr and hv > base_h and wins == 3)
              else ("win, not robust" if hv > base_h else "NULL"))
        res[name] = (hv - base_h, wins, vd, Pv[knob])
        out(f"{name:14s} {knob}={Pv[knob]:<6d} train_win={str(bt[0] > base_tr):5s} "
            f"holdout dLL {hv - base_h:+.5f}  periods {wins}/3  -> {vd}")
    return res


# --------------------------------------------------------------- power ceiling
def _poisson(rng, lam):
    """Knuth. lam here is ~0.12 HR per game, so the loop runs about once."""
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1
        if k > 20:
            return k


def power_probe(rows, name, mults, seeds=(11, 21, 31)):
    """Average power_probe_once over several re-rolls.

    One re-roll is not a ceiling. On a 2.3k-row synthetic holdout the same
    planted effect returned oracle gains of +0.0008, +0.0045, +0.0029 and
    +0.0040 across four seeds — a 5x spread on an identical truth. Reading a
    real null against any single one of those would be a coin flip dressed as
    a bound.
    """
    o, f, v = zip(*[power_probe_once(rows, name, mults, s) for s in seeds])
    return (sum(o) / len(o), sum(f) / len(f),
            f"{sum(1 for x in v if x == 'ROBUST WIN')}/{len(v)} robust")


def power_probe_once(rows, name, mults, seed=11):
    """Re-roll the panel so the angle IS true, then ask two questions.

    Returns (oracle_dLL, fitted_dLL, verdict_on_truth).

    Home runs are re-drawn as POISSON COUNTS, not as a yes/no flag, because
    every cell in this model accumulates hr AND pa. Re-rolling only the binary
    'homered' would leave the shrinkage cells being fed a different quantity
    than the one being scored, and the ceiling would then describe a model
    nobody is running.

    The multiplier is mean-normalised over the panel before use. Without that,
    planting 'travel costs 8%' also quietly lowers the league HR rate, the
    baseline re-fits to the new level, and part of what the angle appears to
    buy is really the level correction.
    """
    rng = random.Random(seed)
    feats = attach2(rows)
    P0, _ = fit_baseline(feats)
    flag, cell, knob, _grid = ANGLES[name]
    kk = cell + "_k"
    raw = [mults.get(f.get(kk), 1.0) for _, f in feats]
    mbar = sum(raw) / len(raw)
    syn = []
    for (r, f), m in zip(feats, raw):
        r2 = dict(r)
        r2["_mult"] = m / mbar
        lam = max(H.rate_pa(r, f, P0, {"flat_platoon"}), 1e-6) * r["pa"]
        r2["hr"] = _poisson(rng, lam * r2["_mult"])
        syn.append(r2)
    sf = attach2(syn)
    Ps, _ = fit_baseline(sf)
    b_h, _ = loglik2(sf, Ps, {"flat_platoon"}, HOLD0, END)
    o_h, _ = loglik2(sf, Ps, {"flat_platoon", "oracle"}, HOLD0, END)
    fitted = verdict(sf, out=lambda s: None)[name]
    return o_h - b_h, fitted[0], fitted[2]


PLANT = {
    # generous on purpose. Real travel/home effects on HR are 1-3%; if the
    # panel cannot see 8% it certainly cannot see 2%, and that is the finding.
    "G travel": {"new": 0.92, "same": 1.0},
    "H home/away": {"home": 1.06, "away": 0.94},
    "I slot shift": {"up": 1.10, "flat": 1.0, "down": 0.90},
}


# -------------------------------------------------------------------- selftest
def _synth(plant=None, seed=5, n_days=70):
    """Panel with real travel structure: teams sit at a venue for a homestand
    and then move. A synthetic schedule that re-drew the venue every game would
    make 'new venue' fire on ~90% of rows, and the test would then be measuring
    a constant."""
    rng = random.Random(seed)
    teams = list(range(12))
    roster = {t: [t * 20 + i for i in range(11)] for t in teams}
    rows = []
    prev_slot, prev_venue = {}, {}
    for day in range(n_days):
        date = "2025-%02d-%02d" % (4 + day // 30, day % 30 + 1)
        # series structure: pairings are re-drawn only every third day, so a
        # team stays in one park for a stretch and the travel flag fires on
        # roughly a third of rows — close to the real schedule. Re-drawing
        # every day would set the flag on nearly everything and the test would
        # be measuring a constant.
        if day % 3 == 0:
            rng.shuffle(teams)
        for gi in range(0, len(teams) - 1, 2):
            a, b = teams[gi], teams[gi + 1]
            venue = a                          # home team's park
            sp = 900 + rng.randrange(40)
            ph = "L" if sp % 4 == 0 else "R"
            for t, is_home in ((a, True), (b, False)):
                order = roster[t][:9]
                rng.shuffle(order)
                for slot, bid in enumerate(order, start=1):
                    bh = "L" if bid % 3 == 0 else "R"
                    rate = 0.031 * (1.0 + 0.006 * (bid % 5))
                    ps, pv = prev_slot.get(bid), prev_venue.get(bid)
                    # plant on exactly the condition the feature detects —
                    # a venue CHANGE since this batter's own last game, not
                    # "is away", which is the other angle wearing a hat
                    if plant == "G travel" and pv is not None and pv != venue:
                        rate *= 0.55
                    if plant == "H home/away":
                        rate *= 1.45 if is_home else 0.72
                    if plant == "I slot shift" and ps:
                        dlt = slot - ps
                        rate *= 1.55 if dlt <= -2 else (0.65 if dlt >= 2 else 1.0)
                    pa = rng.choice((3, 4, 4, 5))
                    rows.append({
                        "date": date, "pk": day * 100 + gi, "venue": venue,
                        "dn": "night", "home": is_home, "bat": bid,
                        "name": f"B{bid}", "slot": slot, "pa": pa,
                        "hr": _poisson(rng, rate * pa), "sp": sp,
                        "bh": bh, "ph": ph})
                    prev_slot[bid], prev_venue[bid] = slot, venue
    return rows


def selftest():
    global START, TRAIN_END, END, HOLD0, PERIODS
    START, TRAIN_END = "2025-04-01", "2025-05-15"
    HOLD0, END = "2025-05-16", "2025-06-06"
    PERIODS = [("2025-05-16", "2025-05-23"), ("2025-05-24", "2025-05-30"),
               ("2025-05-31", END)]
    H.START, H.TRAIN_END, H.END = START, TRAIN_END, END

    null = verdict(attach2(_synth(plant=None)), out=lambda s: None)
    got = {}
    for name in ANGLES:
        res = verdict(attach2(_synth(plant=name)), out=lambda s: None)
        got[name] = res[name][0]
        assert res[name][0] > 0.001, \
            f"planted {name} NOT recovered: {res[name][0]:+.5f}"
        assert null[name][0] < res[name][0] / 3, \
            f"null control suspicious for {name}: {null[name][0]:+.5f}"

    # the one property that makes a ceiling a ceiling: on a panel where the
    # effect is planted, the ORACLE must not come in below what the fitted
    # pipeline actually extracts. If it does, the 'ceiling' is an underestimate
    # and every null scored against it would be wrongly filed as unanswerable.
    # NOTE the deliberately huge multipliers. The shipped PLANT values (6-10%)
    # are the sizes worth asking about in the real world, and on a 7.5k-row
    # synthetic panel their true oracle gain is ~+0.0001 — inside the re-roll
    # noise. Asserting on those would be asserting that a coin came up heads.
    # The machinery is what is under test here, so plant an effect nobody could
    # miss and check the two structural properties.
    big = {"G travel": {"new": 0.55, "same": 1.0},
           "H home/away": {"home": 1.45, "away": 0.72},
           "I slot shift": {"up": 1.55, "flat": 1.0, "down": 0.65}}
    rows = _synth(plant=None)
    for name in ANGLES:
        o, fit, _vd = power_probe(rows, name, big[name], seeds=(3, 13, 23, 33))
        assert o >= fit - 1e-9, f"{name}: oracle {o:+.5f} below fitted {fit:+.5f}"
        assert o > 0.002, f"{name}: oracle collapsed to {o:+.5f}"

    # leak proof: poison outcomes after a cutoff; earlier features must be
    # byte-identical or something in the new pass reads ahead
    ra = _synth(plant="H home/away")
    rb = [dict(r) for r in ra]
    for r in rb:
        if r["date"] > "2025-05-01":
            r["hr"] = 0 if r["hr"] else 1
    fa = [(r["date"], f) for r, f in attach2(ra) if r["date"] <= "2025-05-01"]
    fb = [(r["date"], f) for r, f in attach2(rb) if r["date"] <= "2025-05-01"]
    assert json.dumps(fa, sort_keys=True) == json.dumps(fb, sort_keys=True), "LEAK"

    print("HRANGLES-2 SELFTEST PASS — travel/home/slot-shift all recovered "
          f"({', '.join(f'{k.split()[0]} {v:+.4f}' for k, v in got.items())}), "
          "null clean, oracle >= fitted, leak-free")
    return 0


# ------------------------------------------------------------------------ main
def main():
    if not os.path.exists(H.CACHE):
        print("hrangles dataset cache missing — run on GitHub Actions "
              "(touch experiments/RUN-HRANGLES2.txt)")
        return 0
    ds = json.load(open(H.CACHE))
    burn = (json.load(open(H.BURN_CACHE))
            if os.path.exists(H.BURN_CACHE) else {"rows": []})
    rows = [r for r in burn["rows"] + ds["rows"]
            if r.get("bh") and r.get("ph") and r.get("slot", 0) > 0]
    lines = []

    def tee(s):
        print(s)
        lines.append(s)

    tee("=" * 70)
    tee(f"HR ANGLES 2 — travel, home/away, lineup-slot change, with ceilings")
    tee(f"{START}..{END} (train to {TRAIN_END})   scorable rows: {len(rows)}")
    tee("=" * 70)
    feats = attach2(rows)
    cov = defaultdict(int)
    for _r, f in feats:
        for k in ("trav_k", "shift_k"):
            cov[k] += 1 if f.get(k) else 0
    tee(f"coverage: travel flag on {cov['trav_k']}/{len(feats)} rows, "
        f"slot-shift on {cov['shift_k']}/{len(feats)}")
    tee("")
    res = verdict(feats, out=tee)

    tee("")
    tee("--- POWER CEILINGS (panel re-rolled so the angle IS true; ORACLE is")
    tee("    what a model that knew the answer would gain, FITTED is what this")
    tee("    pipeline recovers from a panel where the effect is real)")
    for name in ANGLES:
        o, fit, vd = power_probe(rows, name, PLANT[name])
        got = res[name][0]
        if o < 0.0004:
            read = "CANNOT BE SEEN AT THIS SAMPLE - do not bury"
        elif got >= fit:
            read = "measured at/above the detection threshold - live"
        else:
            read = "dead: a real effect this size would have shown"
        pl = ", ".join(f"{k}x{v}" for k, v in PLANT[name].items())
        tee(f"{name:14s} plant [{pl}]")
        tee(f"{'':14s} ORACLE {o:+.5f}   FITTED {fit:+.5f} ({vd})   "
            f"measured {got:+.5f}   {read}")

    tee("")
    tee("Ship rule: ROBUST WIN, and the measured gain must clear the FITTED")
    tee("detection threshold from a panel where the effect is known to be real.")
    vd_path = os.path.join(HERE, "..", "experiments", "MLB-HRANGLES2-VERDICT.md")
    os.makedirs(os.path.dirname(vd_path), exist_ok=True)
    with open(vd_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"verdict -> {vd_path}")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
