#!/usr/bin/env python3
"""
contactform_placebo.py — is the ship rule itself strong enough at this effect size?

The widened-window run produced HARD-HIT W=60d as a "ROBUST WIN": train win,
holdout win, 3/3 sub-periods. The margin was +0.00013 LL/game. The pitcher-HR
factor that actually shipped was ~70x that. And the diagnostic window sweep was
flat-to-noisy across all six windows rather than climbing with W, which is what
"no signal, just a lucky cell" looks like.

So: destroy the signal and see how often the rule fires anyway. Each trial
permutes the contact tables ACROSS BATTERS (a hitter's HR outcomes now sit
against a stranger's batted-ball history), then runs the identical
tune-on-train / verdict-on-June pipeline including the same 6x3x3 grid. Any
ROBUST WIN in this world is manufactured by the search, not found in the data.

Reads as: "the rule fires on pure noise k/N of the time." If k/N is not small,
a +0.00013 ROBUST WIN means nothing and the angle stays buried.
"""
import json, os, random, sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlb_contactform_experiment as CF
import mlb_hrangles_experiment as HX

TRIALS = int(os.environ.get("PLACEBO_TRIALS", "20"))
PERIODS = [("2025-06-01", "2025-06-10"), ("2025-06-11", "2025-06-20"),
           ("2025-06-21", CF.END)]
P0 = dict(HX.BASE_P, tau_b=75, tau_park=800, w_p=0.6)

_G = {}


def _init():
    """Each worker loads the panel once: rows, features, raw contact cache."""
    rows = CF.load_rows() if hasattr(CF, "load_rows") else None
    _G["rows"] = rows


def verdict_for(feat, series, metric, base_tr, base_h, base_p):
    """The EXACT ship-rule pipeline: tune W/tau_c/w_c on train, verdict on June."""
    best = (-9e9, None, None)
    for W in (7, 14, 21, 30, 45, 60):
        for tau_c in (10, 25, 50):
            for w_c in (0.3, 0.6, 1.0):
                Pf = {"tau_c": tau_c, "w_c": w_c}
                ll, _, _ = CF.loglik_form(feat, series, P0, Pf, metric, W,
                                          CF.START, CF.TRAIN_END)
                if ll > best[0]:
                    best = (ll, Pf, W)
    ll_tr, Pf, W = best
    hv, _, _ = CF.loglik_form(feat, series, P0, Pf, metric, W, "2025-06-01", CF.END)
    wins = 0
    for i, (p0, p1) in enumerate(PERIODS):
        v, _, _ = CF.loglik_form(feat, series, P0, Pf, metric, W, p0, p1)
        wins += 1 if v > base_p[i] else 0
    robust = (ll_tr > base_tr) and (hv > base_h) and wins == 3
    return {"W": W, "Pf": Pf, "dLL": hv - base_h, "periods": wins, "robust": robust}


def trial(seed):
    feat, raw = _G["feat"], _G["raw"]
    base_tr, base_h, base_p = _G["base_tr"], _G["base_h"], _G["base_p"]
    ids = list(raw.keys())
    perm = ids[:]
    random.Random(seed).shuffle(perm)
    # derangement-ish: a batter must not keep his own table
    for i in range(len(perm)):
        if perm[i] == ids[i]:
            j = (i + 1) % len(perm)
            perm[i], perm[j] = perm[j], perm[i]
    swapped = {ids[i]: raw[perm[i]] for i in range(len(ids))}
    series = CF.contact_series(swapped)
    out = {}
    for metric in ("brl", "hh"):
        out[metric] = verdict_for(feat, series, metric, base_tr, base_h, base_p)
    return seed, out


def _boot(_):
    pass


def setup():
    ds = json.load(open(HX.CACHE))
    burn = json.load(open(HX.BURN_CACHE))
    rows = [r for r in burn["rows"] + ds["rows"]
            if r.get("bh") and r.get("ph") and r.get("slot", 0) > 0]
    feat = HX.attach_features(rows)
    raw = CF._load_cache()["batters"]
    base_tr, _, _ = CF.loglik_form(feat, {}, P0, None, "", 0, CF.START, CF.TRAIN_END)
    base_h, _, _ = CF.loglik_form(feat, {}, P0, None, "", 0, "2025-06-01", CF.END)
    base_p = [CF.loglik_form(feat, {}, P0, None, "", 0, p0, p1)[0]
              for p0, p1 in PERIODS]
    return feat, raw, base_tr, base_h, base_p


def _pool_init(feat, raw, base_tr, base_h, base_p):
    _G.update(feat=feat, raw=raw, base_tr=base_tr, base_h=base_h, base_p=base_p)


def main():
    feat, raw, base_tr, base_h, base_p = setup()
    print(f"panel {len(feat)} rows, {len(raw)} batters | "
          f"baseline TRAIN {base_tr:+.5f} HOLDOUT {base_h:+.5f}")
    print(f"running {TRIALS} shuffled trials through the full ship-rule pipeline")
    args = (feat, raw, base_tr, base_h, base_p)
    with Pool(2, initializer=_pool_init, initargs=args) as p:
        results = p.map(trial, range(1, TRIALS + 1))
    fired = {"brl": 0, "hh": 0}
    dlls = {"brl": [], "hh": []}
    for seed, out in sorted(results):
        for m in ("brl", "hh"):
            r = out[m]
            dlls[m].append(r["dLL"])
            if r["robust"]:
                fired[m] += 1
        print(f"  trial {seed:2d}  "
              + "  ".join(f"{m.upper():4s} W={out[m]['W']:2d} "
                          f"dLL {out[m]['dLL']:+.5f} p{out[m]['periods']}/3"
                          f"{' ROBUST' if out[m]['robust'] else ''}"
                          for m in ("brl", "hh")))
    print("-" * 70)
    for m in ("brl", "hh"):
        d = sorted(dlls[m])
        print(f"{m.upper():4s}  ship rule fired {fired[m]}/{TRIALS} on pure noise | "
              f"shuffled holdout dLL: min {d[0]:+.5f} median {d[len(d)//2]:+.5f} "
              f"max {d[-1]:+.5f}")
    print("Real HARD-HIT W=60d scored +0.00013. Compare it to the max above.")
    json.dump({m: {"fired": fired[m], "trials": TRIALS, "dlls": dlls[m]}
               for m in ("brl", "hh")},
              open(os.path.join(CF.HERE, "data", "contactform_placebo.json"), "w"),
              indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
