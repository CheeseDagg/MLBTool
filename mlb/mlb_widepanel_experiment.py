#!/usr/bin/env python3
"""
mlb_widepanel_experiment.py — re-read every dead HR angle on a 4x panel.

THE PROBLEM THIS EXISTS TO FIX. Every MLB HR verdict on the board — four
angles in batch 1, three in batch 2 — was decided on ONE holdout: June 2025,
8,037 batter-games. Meanwhile 49,562 rows of 2024 sit in
data/hrangles_burnin_2024.json and are used only to WARM the shrinkage cells.
They are never scored. That is half the information in the cache being spent
on cell maturity and none of it on the verdict.

Batch 2 made the cost of that concrete. Travel and lineup-slot shift came back
"CANNOT BE SEEN AT THIS SAMPLE": their power ceilings were +0.00018 and
+0.00013, meaning an 8-10% effect that was KNOWN TO BE TRUE would still not
have cleared the noise. Those are not findings about baseball. They are
findings about 8,037 rows.

WHAT CHANGES. Scoring windows only. Same rows, same features, same code paths:

  WARM    2024-03-20..2024-05-31   cells accumulate, nothing is scored
  TRAIN   2024-06-01..2024-07-31 + 2025-04-01..2025-05-31
  HOLDOUT 2024-08-01..2024-09-30 + 2025-06-01..2025-06-30   (n ~ 4x)

and the ship rule gets a tooth that no amount of extra rows could buy on its
own: CROSS-SEASON REPLICATION. A win must clear baseline in the 2024 holdout
AND in the 2025 holdout, separately. Three consecutive ten-day slices of one
June share a weather regime, a league-wide ball, a set of hot bats and a
single set of park conditions; two different Augusts do not. This is the check
that would have killed the 60-day hard-hit result without needing a placebo.

WHAT DOES NOT CHANGE. The tuning grids, the baseline shape, the feature code
and the ceiling method are all imported from the two shipped modules rather
than re-implemented. A re-read that runs different code is not a re-read.

Run: python3 mlb_widepanel_experiment.py [--selftest]
"""
import json
import math
import os
import random
import sys
import zlib
from collections import defaultdict

import mlb_hrangles_experiment as H
import mlb_hrangles2_experiment as H2

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- windows
# WARM is deliberately NOT in TRAIN. Cells at the very start of 2024 are empty,
# so the first weeks are the model learning to exist rather than the model
# being wrong, and tuning a shrinkage constant against that period optimises
# cold-start behaviour instead of steady-state accuracy.
WARM = ("2024-03-20", "2024-05-31")
TRAIN = [("2024-06-01", "2024-07-31"), ("2025-04-01", "2025-05-31")]
HOLD_BY_SEASON = {
    "2024": [("2024-08-01", "2024-09-30")],
    "2025": [("2025-06-01", "2025-06-30")],
}
HOLD = HOLD_BY_SEASON["2024"] + HOLD_BY_SEASON["2025"]
# four slices, two per season. Within-season slices catch a lucky fortnight;
# the season split catches a lucky year.
PERIODS = [("2024-08-01", "2024-08-31"), ("2024-09-01", "2024-09-30"),
           ("2025-06-01", "2025-06-15"), ("2025-06-16", "2025-06-30")]

# ---------------------------------------------------------------- angles
# (label, use-flags on top of the baseline, knob, grid). The first four are
# batch 1, the last three batch 2; all seven score through H2.rate_pa2, which
# delegates to H.rate_pa for the batch-1 factors, so this is the same
# arithmetic that produced the original verdicts.
ANGLES = [
    ("C indiv platoon", {"flat_platoon", "indiv_platoon"}, "tau_s",
     (100, 200, 400, 800)),
    ("D handed park", {"flat_platoon", "handed_park"}, "tau_ph",
     (150, 400, 1000, 2500)),
    ("E day/night", {"flat_platoon", "daynight"}, "tau_dn",
     (1500, 3000, 6000)),
    ("F pitcher HR", {"flat_platoon", "pitcher_hr"}, "w_p",
     (0.3, 0.6, 1.0)),
    ("G travel", {"flat_platoon", "travel"}, "tau_tr",
     (300, 1000, 3000, 8000)),
    ("H home/away", {"flat_platoon", "homeaway"}, "tau_ha",
     (1000, 3000, 8000, 20000)),
    ("I slot shift", {"flat_platoon", "slotshift"}, "tau_sh",
     (300, 1000, 3000, 8000)),
]
BASE = {"flat_platoon"}


# ---------------------------------------------------------------- scoring
def _ll_rows(sel, P, use):
    tot = n = 0.0
    for r, f in sel:
        lam = max(H2.rate_pa2(r, f, P, use), 1e-6) * r["pa"]
        p = min(max(1.0 - math.exp(-lam), 1e-9), 1 - 1e-9)
        y = 1 if r["hr"] > 0 else 0
        tot += y * math.log(p) + (1 - y) * math.log(1 - p)
        n += 1
    return (tot / n if n else 0.0), int(n)


def _in(rf, windows):
    return any(d0 <= rf[0]["date"] <= d1 for d0, d1 in windows)


def ll_w(feats, P, use, windows):
    """Pooled per-row log-likelihood over a LIST of date windows.

    Pooled, not averaged-of-averages: the 2024 holdout is roughly twice the
    2025 one, and averaging the two per-row means would silently upweight the
    smaller season. Rows are the unit of evidence here, so rows are what gets
    summed. A row is counted once even if two windows overlapped (ours never
    do — the selftest enforces it — but double-counting would be a silent
    reweighting rather than an error, so it is ruled out here).
    """
    return _ll_rows([rf for rf in feats if _in(rf, windows)], P, use)


# Every window set this file ever scores. Bucketing the panel into these once
# turns each scoring call from a scan of all 74,018 rows into a scan of the
# few thousand it actually needs. The first version rescanned the full panel
# per window per call: fit_baseline alone was 9 grid cells x 2 windows x 74k
# = 1.3M row evaluations, the period loop another 4 x 74k, and the whole thing
# runs inside 21 power probes. That is why the first real run took 40 minutes
# and died before finishing.
WINDOW_SETS = dict(
    [("TRAIN", TRAIN), ("HOLD", HOLD), ("ALL", TRAIN + HOLD)]
    + [("H" + s, w) for s, w in HOLD_BY_SEASON.items()]
    + [(f"P{i}", [w]) for i, w in enumerate(PERIODS)])


def build_index(feats):
    """Bucket the panel by window set once. Holds references, not copies."""
    return {k: [rf for rf in feats if _in(rf, w)]
            for k, w in WINDOW_SETS.items()}


def ll_k(idx, P, use, key):
    return _ll_rows(idx[key], P, use)


def fit_baseline(idx):
    best = (-9e9, None)
    for tb in (75, 150, 300):
        for tp in (800, 1500, 3000):
            P = dict(H2.BASE_P, tau_b=tb, tau_park=tp)
            ll, _ = ll_k(idx, P, BASE, "TRAIN")
            if ll > best[0]:
                best = (ll, P)
    return best[1], best[0]


def verdict_one(idx, angle, P0, base_tr, base_h, base_season, base_per):
    """Tune one angle on TRAIN, judge it on the two holdouts and four slices.

    Returns (dLL, per-season dLL, slice wins, knob value, robust).
    """
    label, use, knob, grid = angle
    bt = (-9e9, None)
    for v in grid:
        P = dict(P0)
        P[knob] = v
        ll, _ = ll_k(idx, P, use, "TRAIN")
        if ll > bt[0]:
            bt = (ll, P)
    Pv = bt[1]
    train_win = bt[0] > base_tr
    hv, _ = ll_k(idx, Pv, use, "HOLD")
    season = {}
    for s in HOLD_BY_SEASON:
        v, _ = ll_k(idx, Pv, use, "H" + s)
        season[s] = v - base_season[s]
    slices = 0
    for i in range(len(PERIODS)):
        v, _ = ll_k(idx, Pv, use, f"P{i}")
        slices += 1 if v > base_per[i] else 0
    # the cross-season requirement is the point of this file: a term that
    # helps in one season and hurts in the other is fitting that season.
    robust = (train_win and hv > base_h
              and all(d > 0 for d in season.values()) and slices >= 3)
    return hv - base_h, season, slices, Pv[knob], robust


def baseline_pack(idx):
    P0, base_tr = fit_baseline(idx)
    base_h, n_h = ll_k(idx, P0, BASE, "HOLD")
    base_season = {s: ll_k(idx, P0, BASE, "H" + s)[0] for s in HOLD_BY_SEASON}
    base_per = [ll_k(idx, P0, BASE, f"P{i}")[0] for i in range(len(PERIODS))]
    return P0, base_tr, base_h, base_season, base_per, n_h


def score_all(feats, out=lambda s: None):
    idx = build_index(feats)
    P0, base_tr, base_h, base_season, base_per, n_h = baseline_pack(idx)
    for s in HOLD_BY_SEASON:
        out(f"  baseline HOLDOUT {s}: {base_season[s]:+.5f} "
            f"(n={len(idx['H' + s])})")
    out(f"  baseline tau_b={P0['tau_b']} tau_park={P0['tau_park']}  "
        f"TRAIN {base_tr:+.5f}  HOLDOUT {base_h:+.5f} (n={n_h})")
    res = {}
    for a in ANGLES:
        res[a[0]] = verdict_one(idx, a, P0, base_tr, base_h,
                                base_season, base_per)
    return res, P0


# ---------------------------------------------------------------- plants
def _stable(s, salt):
    """Deterministic pseudo-random in [0,1) from a string.

    NOT hash(): CPython salts string hashing per process, so a plant built on
    hash() would silently plant a different world on every run and the three
    ceiling seeds would not be measuring the same truth.
    """
    return (zlib.crc32((salt + "|" + str(s)).encode()) % 100000) / 100000.0


def _plant_c(r, f):
    """Each batter has his own platoon split, applied on opposite-hand only."""
    if not r["ph"]:
        return 1.0
    h = H.eff_hand(r["bh"], r["ph"])
    if h == r["ph"]:
        return 1.0
    return 0.85 + 0.30 * _stable(r["bat"], "platoon")


def _plant_d(r, f):
    """A third of parks favour lefties, a third punish them, righties flat."""
    h = H.eff_hand(r["bh"], r["ph"]) if r["ph"] else (r["bh"] or "R")
    if h != "L":
        return 1.0
    b = int(_stable(r["venue"], "porch") * 3)
    return (1.18, 1.0, 0.85)[b]


def _plant_e(r, f):
    return 1.06 if r["dn"] == "day" else 0.97


def _plant_f(r, f):
    return 0.85 + 0.30 * _stable(r["sp"], "arm")


def _cell_plant(name):
    """G/H/I reuse batch 2's categorical plants, looked up by cell key."""
    _flag, cell, _knob, _grid = H2.ANGLES[name]
    kk, table = cell + "_k", H2.PLANT[name]

    def fn(r, f):
        return table.get(f.get(kk), 1.0)
    return fn


PLANT = {
    # batch-1 plants are deliberately generous — 15-18% swings, far larger
    # than anything real — because the question this file asks is not "is the
    # effect there" but "could this panel have seen it if it were". A plant
    # the pipeline cannot recover at 18% is a panel verdict, not a baseball
    # verdict.
    "C indiv platoon": _plant_c,
    "D handed park": _plant_d,
    "E day/night": _plant_e,
    "F pitcher HR": _plant_f,
    "G travel": _cell_plant("G travel"),
    "H home/away": _cell_plant("H home/away"),
    "I slot shift": _cell_plant("I slot shift"),
}


def probe_once(rows, angle, seed):
    """Re-roll the panel so this angle IS true, then read oracle vs fitted.

    Poisson counts, not a homered/didn't flag: every cell in the model
    accumulates hr AND pa, so re-rolling only the binary would feed the
    shrinkage cells a different quantity than the one being scored.

    The multiplier is mean-normalised first. Planting "day games add 6%"
    without normalising also raises the league HR level, the baseline re-fits
    to that new level, and part of what the angle appears to buy is really the
    level correction being credited to the wrong term.
    """
    label = angle[0]
    rng = random.Random(seed)
    feats = H2.attach2(rows)
    P0, _ = fit_baseline(build_index(feats))
    fn = PLANT[label]
    raw = [fn(r, f) for r, f in feats]
    mbar = sum(raw) / len(raw)
    syn = []
    for (r, f), m in zip(feats, raw):
        r2 = dict(r)
        r2["_mult"] = m / mbar
        lam = max(H2.rate_pa2(r, f, P0, BASE), 1e-6) * r["pa"]
        r2["hr"] = H2._poisson(rng, lam * r2["_mult"])
        syn.append(r2)
    # the original panel is dead weight from here on, and holding it alongside
    # syn AND the re-attached features means three full copies of a 74k-row
    # panel live at once. The first real run was killed partway through the
    # sixth angle; dropping this is the cheapest way to stop that recurring.
    del feats, raw
    sf = H2.attach2(syn)
    sidx = build_index(sf)
    Ps, base_tr, b_h, base_season, base_per, _n = baseline_pack(sidx)
    o_h, _ = ll_k(sidx, Ps, BASE | {"oracle"}, "HOLD")
    d, _season, _sl, _k, robust = verdict_one(sidx, angle, Ps, base_tr, b_h,
                                              base_season, base_per)
    return o_h - b_h, d, robust


def probe(rows, angle, seeds=(11, 21, 31)):
    """One re-roll is not a ceiling — average it and report the spread.

    On a 2.3k-row synthetic holdout the same planted truth returned oracle
    gains of +0.0008, +0.0045, +0.0029 and +0.0040 across four seeds. A bound
    that moves 5x with the seed is not a bound.
    """
    o, fi, rb = zip(*[probe_once(rows, angle, s) for s in seeds])
    return (sum(o) / len(o), sum(fi) / len(fi),
            f"{sum(rb)}/{len(rb)} robust", min(o), max(o))


# ---------------------------------------------------------------- main
def load_rows():
    ds = json.load(open(H.CACHE))
    burn = (json.load(open(H.BURN_CACHE))
            if os.path.exists(H.BURN_CACHE) else {"rows": []})
    return [r for r in burn["rows"] + ds["rows"]
            if r.get("bh") and r.get("ph") and r.get("slot", 0) > 0]


def main():
    if not os.path.exists(H.CACHE):
        print("hrangles dataset cache missing — run on GitHub Actions")
        return 0
    rows = load_rows()
    lines = []

    def tee(s):
        print(s)
        lines.append(s)

    feats = H2.attach2(rows)
    n_hold = ll_w(feats, H2.BASE_P, BASE, HOLD)[1]
    tee("=" * 74)
    tee("HR WIDE PANEL — all seven angles re-read on a two-season holdout")
    tee(f"rows {len(rows)}   warm {WARM[0]}..{WARM[1]}   "
        f"holdout n={n_hold} (was 8037 on June 2025 alone)")
    tee("=" * 74)
    tee("")
    res, _P0 = score_all(feats, out=tee)
    tee("")
    tee(f"{'angle':17s} {'knob':>6s} {'holdout':>9s} {'2024':>9s} "
        f"{'2025':>9s} {'slices':>7s}  verdict")
    for label, _u, _k, _g in ANGLES:
        d, season, sl, kv, rob = res[label]
        tee(f"{label:17s} {kv:>6} {d:+9.5f} {season['2024']:+9.5f} "
            f"{season['2025']:+9.5f} {sl:>5d}/4  "
            f"{'ROBUST WIN' if rob else 'no'}")

    tee("")
    tee("--- POWER CEILINGS on the wide panel. ORACLE = what a model that knew")
    tee("    the planted multiplier would gain (a hard bound). FITTED = what")
    tee("    this pipeline actually recovers from a panel where the effect is")
    tee("    real (the honest detection threshold). The gap is the tax.")
    for angle in ANGLES:
        label = angle[0]
        o, fi, rb, olo, ohi = probe(rows, angle)
        got = res[label][0]
        won = res[label][4]
        # ORDER MATTERS, and getting it wrong is not hypothetical: the first
        # draft tested "oracle too small to see" FIRST, so an angle that won
        # in both seasons and measured six times its own detection threshold
        # was printed as "cannot be seen". A win is evidence the thing WAS
        # seen; the low-power branch only applies to results that did not win.
        # Same bug, same fix as the UFC ceiling read.
        if got >= o:
            read = "measured >= ORACLE: noise by construction"
        elif got >= olo:
            read = ("won but sits inside the oracle's own seed spread: "
                    "UNPROVEN, needs a placebo" if won else
                    "inside the oracle's seed spread: unreadable")
        elif won and got >= fi:
            read = f"LIVE: robust win at {100.0 * got / o:.0f}% of oracle"
        elif o < 0.0004:
            read = "STILL CANNOT BE SEEN - do not bury"
        else:
            read = "dead: an effect this size would have shown here"
        tee(f"{label:17s} ORACLE {o:+.5f} [{olo:+.5f}..{ohi:+.5f}]  "
            f"FITTED {fi:+.5f} ({rb})  measured {got:+.5f}   {read}")

    vd = os.environ.get("VERDICT_OUT") or os.path.join(
        HERE, "..", "experiments", "MLB-WIDEPANEL-VERDICT.md")
    os.makedirs(os.path.dirname(vd), exist_ok=True)
    with open(vd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"verdict -> {vd}")
    return 0


# ---------------------------------------------------------------- selftest
def selftest():
    # 1) plants are deterministic across calls AND across processes. If this
    #    ever regresses to hash(), the three ceiling seeds stop measuring the
    #    same planted world and the averaged ceiling becomes meaningless.
    assert _stable("abc", "s") == _stable("abc", "s")
    assert _stable("abc", "s") != _stable("abc", "t")
    # pinned to the value crc32 actually produces, so a future refactor of
    # _stable that is "equivalent" but reshuffles the mapping gets caught.
    assert abs(_stable("mookie", "platoon") - 0.59034) < 1e-5, (
        f"plant drifted: {_stable('mookie', 'platoon')}")

    # 2) each plant actually varies, and varies over the thing it claims to.
    R = {"bat": "b1", "bh": "L", "ph": "R", "sp": "p1", "venue": "v1",
         "dn": "day", "pa": 4, "hr": 0, "slot": 3, "home": 1,
         "date": "2025-06-01"}
    assert _plant_c(R, {}) != _plant_c(dict(R, bat="b2"), {})
    assert _plant_c(dict(R, ph="L"), {}) == 1.0, "platoon plant fired on same-hand"
    assert _plant_d(dict(R, bh="R"), {}) == 1.0, "porch plant fired on righties"
    assert len({_plant_d(dict(R, venue=f"v{i}"), {}) for i in range(30)}) == 3
    assert _plant_e(R, {}) != _plant_e(dict(R, dn="night"), {})
    assert _plant_f(R, {}) != _plant_f(dict(R, sp="p2"), {})

    # 3) the window scorer must pool by ROWS, not average two means. Build a
    #    lopsided pair of windows and check against a hand-computed pool.
    rows = ([{"bat": "b", "bh": "R", "ph": "L", "sp": "p", "venue": "v",
              "dn": "day", "pa": 4, "hr": i % 2, "slot": 1, "home": 1,
              "date": "2024-08-05"} for i in range(300)]
            + [{"bat": "b", "bh": "R", "ph": "L", "sp": "p", "venue": "v",
                "dn": "day", "pa": 4, "hr": i % 3, "slot": 1, "home": 1,
                "date": "2025-06-05"} for i in range(100)])
    fx = H2.attach2(rows)
    P = dict(H2.BASE_P)
    a, na = ll_w(fx, P, BASE, [("2024-08-01", "2024-08-31")])
    b, nb = ll_w(fx, P, BASE, [("2025-06-01", "2025-06-30")])
    both, nboth = ll_w(fx, P, BASE, [("2024-08-01", "2024-08-31"),
                                     ("2025-06-01", "2025-06-30")])
    assert nboth == na + nb == 400
    pooled = (a * na + b * nb) / (na + nb)
    assert abs(both - pooled) < 1e-12, f"windows not pooled by rows: {both} vs {pooled}"
    assert abs(both - (a + b) / 2) > 1e-9, "pooling collapsed to a flat mean"

    # 3b) the window INDEX must agree with the direct scan to the last bit.
    #     build_index exists purely for speed; the moment it disagrees with
    #     ll_w it is silently rewriting every verdict in the file.
    ix = build_index(fx)
    for key, wins in WINDOW_SETS.items():
        for u in (BASE, {"flat_platoon", "pitcher_hr"}):
            dv, dn = ll_w(fx, P, u, wins)
            kv, kn = ll_k(ix, P, u, key)
            assert dn == kn and abs(dv - kv) < 1e-12, (
                f"index disagrees with direct scan on {key}: "
                f"{kv},{kn} vs {dv},{dn}")
    assert len(ix["ALL"]) == len(ix["TRAIN"]) + len(ix["HOLD"]), \
        "ALL is not exactly TRAIN + HOLD"

    # 4) warm rows are never scored. This is the leak that would make the
    #    whole file worthless: if March-May 2024 leaked into TRAIN or HOLDOUT
    #    the cold-start period would be judged as if it were steady state.
    warm = [{"bat": "b", "bh": "R", "ph": "L", "sp": "p", "venue": "v",
             "dn": "day", "pa": 4, "hr": 1, "slot": 1, "home": 1,
             "date": "2024-04-10"}]
    fw = H2.attach2(warm)
    for wins in (TRAIN, HOLD, [w for w in PERIODS]):
        assert ll_w(fw, P, BASE, wins)[1] == 0, "warm rows are being scored"

    # 5) TRAIN and HOLDOUT must not overlap, in either season.
    for t0, t1 in TRAIN:
        for h0, h1 in HOLD:
            assert t1 < h0 or h1 < t0, f"train {t0}..{t1} overlaps hold {h0}..{h1}"
    # and every slice must sit inside a holdout window
    for p0, p1 in PERIODS:
        assert any(h0 <= p0 and p1 <= h1 for h0, h1 in HOLD), \
            f"slice {p0}..{p1} is not inside a holdout window"

    # 6) end to end on synthetic data: a planted day/night effect must be
    #    recoverable by the oracle. Uses a huge plant on purpose — realistic
    #    6% plants land near +0.0001 on a small synthetic panel, and asserting
    #    on those would be asserting that a coin came up heads.
    rng = random.Random(3)
    syn = []
    days = ([f"2024-0{m}-{d:02d}" for m in (4, 5, 6, 7, 8, 9)
             for d in range(1, 29)]
            + [f"2025-0{m}-{d:02d}" for m in (4, 5, 6) for d in range(1, 29)])
    for day in days:
        for i in range(30):
            syn.append({"bat": f"b{i}", "bh": "LR"[i % 2], "ph": "LR"[i % 3 == 0],
                        "sp": f"p{i % 9}", "venue": f"v{i % 8}",
                        "dn": "day" if (i + int(day[-2:])) % 2 else "night",
                        "home": 1 if (i + int(day[-2:])) % 3 else 0,
                        # a realistic HR rate, not zeros. probe_once re-rolls
                        # outcomes from a baseline FIT TO THIS PANEL, so an
                        # all-zero panel fits a zero rate, draws zero home runs
                        # and gives every angle a ceiling of exactly nothing.
                        "pa": 4, "hr": 1 if rng.random() < 0.12 else 0,
                        "slot": 1 + i % 9, "date": day})
    ang = [a for a in ANGLES if a[0] == "E day/night"][0]
    save = PLANT["E day/night"]
    PLANT["E day/night"] = lambda r, f: 1.6 if r["dn"] == "day" else 0.6
    try:
        o, fi, rb = probe_once(syn, ang, 5)
    finally:
        PLANT["E day/night"] = save
    assert o > 0.002, f"oracle failed to see a planted 1.6/0.6 day effect: {o:+.5f}"
    assert o >= fi - 1e-9, f"fitted {fi:+.5f} beat oracle {o:+.5f} — impossible"

    print("WIDE PANEL SELFTEST PASS — plants deterministic and targeted, "
          "windows pool by rows, warm period never scored, train/holdout "
          f"disjoint, oracle recovers a planted effect (+{o:.4f} >= fitted "
          f"{fi:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
