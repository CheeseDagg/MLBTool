#!/usr/bin/env python3
"""
mlb_fatigue_wide.py — the four buried angles, re-read on the two-season panel.

WHY THIS FILE EXISTS. MODEL-KNOWLEDGE says, in its own words:

    CANNOT BE SEEN AT THIS SAMPLE (not the same as dead — do not bury): batter
    REST days and 7-day WORKLOAD. Both read ~ -0.0002, but their power ceilings
    are only +0.00033 and +0.00046 ... Even a true +-40% effect would sit in the
    noise. Answerable only with a multi-season panel. Same for within-season
    familiarity: 4% of rows are repeat meetings, ceiling ~ 0.

We now have the multi-season panel. mlb_widepanel_experiment.py showed the
cost of not using it: F pitcher HR was a NULL on the single-June holdout and a
cross-season robust win on the wide one. These four were filed on that same
narrow holdout, with an explicit note that the panel could not have seen them.
Leaving them buried while the wider panel sits on disk is the exact mistake the
wide-panel file was written to stop.

WHAT IS REUSED AND WHAT IS NOT. The variant factors, the tuning grid and the
date-mean calendar control are imported from mlb_fatigue_experiment — a re-read
that runs different arithmetic is not a re-read. The windows, the index, the
cross-season ship rule and the ceiling method come from
mlb_widepanel_experiment. Only the history builder is re-implemented, and only
because two of its rules are silently wrong once a second season is in the
panel:

  1. FAMILIARITY reset. The original keys within-season meetings off
     `d >= "2025-01-01"`, so every 2024 row carries fam=0. On this panel that
     hands the 2024 holdout a feature that is constant by construction — the
     season it is supposed to be measured in is the season where it cannot
     move. Fixed by keying the counter on (batter, starter, YEAR).

  2. REST across the winter. The original caps rest at 5 days, on the stated
     grounds that a 12-day gap is an injury return and not what this tests. But
     capping is not excluding: a 183-day offseason gap reads as identical to a
     man who took Sunday off. On a single June that was a handful of rows; here
     it is the whole of April 2025 plus every injury return in two seasons.
     Fixed by returning None (no opinion) above LONG_GAP days, which is what
     the original docstring already said the intent was.

Both fixes make the feature HARDER to win with, not easier — they remove rows
the term was allowed to speak about. That direction matters: a fix that widens
a feature's reach on a re-read is indistinguishable from tuning.

Run: python3 mlb_fatigue_wide.py [--selftest]
"""
import datetime as dt
import math
import os
import random
import sys
from collections import defaultdict

import mlb_fatigue_experiment as F
import mlb_hrangles_experiment as H
import mlb_hrangles2_experiment as H2
import mlb_widepanel_experiment as W

HERE = os.path.dirname(os.path.abspath(__file__))

# A gap longer than this is an injury return, a call-up or the offseason. All
# three are different animals from "he had a day off", which is the thing this
# angle claims to measure.
LONG_GAP = 10
REGULAR_PA = F.REGULAR_PA

# The baseline these have to beat now INCLUDES the pitcher-HR term, because the
# wide panel promoted it from NULL to a cross-season robust win. Testing a new
# angle against a baseline weaker than what ships is how a term gets credit for
# work another term already does.
BASE_USE = {"flat_platoon", "pitcher_hr"}

KINDS = [("rest", "REST    days off"),
         ("load", "LOAD    7d PA vs own norm"),
         ("fam", "FAMIL   Nth look, season"),
         ("famcar", "FAMCAR  Nth look, career")]

# The plant strength for the ceilings. Deliberately larger than anything
# plausible: the question a ceiling asks is not "is it there" but "could this
# panel have seen it if it were".
PLANT_W = 0.30


# ------------------------------------------------------------------ history
def build_history(rows):
    """Prior-games-only features, correct across a season boundary.

    Returns (rows_in_chronological_order, parallel_list_of_feature_dicts).
    State advances only AFTER a row is emitted; that ordering is the leak
    boundary and the selftest asserts it directly.
    """
    rows = sorted(rows, key=lambda r: (r["date"], r.get("pk", 0)))
    last_game = {}
    recent = defaultdict(list)
    seen_season = defaultdict(int)     # (bat, sp, year)
    seen_career = defaultdict(int)     # (bat, sp)
    total_pa = defaultdict(int)
    out = []
    for r in rows:
        b, sp, d = r["bat"], r.get("sp"), r["date"]
        yr = d[:4]
        dd = dt.date.fromisoformat(d)
        prev = last_game.get(b)
        rest = None
        if prev:
            gap = (dd - dt.date.fromisoformat(prev)).days
            # None, not a cap. See the module docstring: an offseason and a
            # day off are not the same feature wearing different clothes.
            rest = gap if gap <= LONG_GAP else None
        w7 = sum(p for d2, p in recent[b]
                 if 0 < (dd - dt.date.fromisoformat(d2)).days <= 7)
        w30 = sum(p for d2, p in recent[b]
                  if 0 < (dd - dt.date.fromisoformat(d2)).days <= 30)
        out.append({"rest": rest, "pa7": w7, "pa30": w30,
                    "fam": seen_season[(b, sp, yr)],
                    "famcar": seen_career[(b, sp)],
                    "prior_pa": total_pa[b]})
        last_game[b] = d
        recent[b].append((d, r["pa"]))
        if len(recent[b]) > 60:
            recent[b] = recent[b][-60:]
        if sp is not None:
            seen_season[(b, sp, yr)] += 1
            seen_career[(b, sp)] += 1
        total_pa[b] += r["pa"]
    return rows, out


def attach(rows):
    """(row, base features, history) triples, aligned by object identity.

    attach_features returns rows grouped by DAY, not in the order it was given
    them, so zipping its output against a chronological history list would
    silently pair a Tuesday row with a Sunday row's rest count. id() is exact
    here because attach_features hands back the very dicts it was passed —
    the same guarantee attach2 already relies on.
    """
    srt, hist = build_history(rows)
    by_id = {id(r): h for r, h in zip(srt, hist)}
    return [(r, f, by_id[id(r)]) for r, f in H2.attach2(srt)]


# ------------------------------------------------------------------ scoring
def _ll(sel, P, kind, dm, subset=None, mult=None):
    tot = n = 0.0
    for r, f, h in sel:
        if subset == "regulars" and h["prior_pa"] < REGULAR_PA:
            continue
        lam = max(H.rate_pa(r, f, P, BASE_USE), 1e-6)
        if kind:
            lam *= F.factor(h, r, kind, P, dm)
        if mult is not None:
            lam *= mult(h, r)
        lam = max(lam, 1e-6) * r["pa"]
        p = min(max(1.0 - math.exp(-lam), 1e-9), 1 - 1e-9)
        y = 1 if r["hr"] > 0 else 0
        tot += y * math.log(p) + (1 - y) * math.log(1 - p)
        n += 1
    return (tot / n if n else 0.0), int(n)


def build_index(triples):
    return {k: [t for t in triples if any(d0 <= t[0]["date"] <= d1
                                          for d0, d1 in wins)]
            for k, wins in W.WINDOW_SETS.items()}


def ll_k(idx, P, kind, dm, key, subset=None):
    return _ll(idx[key], P, kind, dm, subset)


def fit_baseline(idx, dm, subset=None):
    best = (-9e9, None)
    for tb in (75, 150, 300):
        for tp in (800, 1500, 3000):
            for wp in (0.3, 0.6):
                P = dict(H2.BASE_P, tau_b=tb, tau_park=tp, w_p=wp)
                ll, _ = ll_k(idx, P, None, dm, "TRAIN", subset)
                if ll > best[0]:
                    best = (ll, P)
    return best[1], best[0]


def baseline_pack(idx, dm, subset=None):
    P0, base_tr = fit_baseline(idx, dm, subset)
    base_h, n_h = ll_k(idx, P0, None, dm, "HOLD", subset)
    season = {s: ll_k(idx, P0, None, dm, "H" + s, subset)[0]
              for s in W.HOLD_BY_SEASON}
    per = [ll_k(idx, P0, None, dm, f"P{i}", subset)[0]
           for i in range(len(W.PERIODS))]
    return P0, base_tr, base_h, season, per, n_h


def verdict_one(idx, kind, dm, pack, subset=None):
    """Tune w on TRAIN, judge on both holdouts and the four slices."""
    P0, base_tr, base_h, base_season, base_per, _n = pack
    best = (-9e9, None)
    for w in F.W_GRID:
        P = dict(P0, w=w)
        ll, _ = ll_k(idx, P, kind, dm, "TRAIN", subset)
        if ll > best[0]:
            best = (ll, w)
    ll_tr, w = best
    P = dict(P0, w=w)
    hv, _ = ll_k(idx, P, kind, dm, "HOLD", subset)
    season = {s: ll_k(idx, P, kind, dm, "H" + s, subset)[0] - base_season[s]
              for s in W.HOLD_BY_SEASON}
    slices = sum(1 for i in range(len(W.PERIODS))
                 if ll_k(idx, P, kind, dm, f"P{i}", subset)[0] > base_per[i])
    robust = (ll_tr > base_tr and hv > base_h
              and all(d > 0 for d in season.values()) and slices >= 3)
    return hv - base_h, season, slices, w, robust


# ------------------------------------------------------------------ ceilings
def probe_once(rows, kind, seed):
    """Re-roll HR counts so this angle IS true at PLANT_W, then read the gap.

    ORACLE is what a model handed the true multiplier gains; FITTED is what
    this pipeline recovers by tuning w on TRAIN. The multiplier is
    mean-normalised, or the plant also lifts the league HR level and the
    re-fitted baseline takes credit that belongs to the angle.
    """
    rng = random.Random(seed)
    trip = attach(rows)
    dm = {k: date_means(trip, k) for k in ("fam", "famcar")}
    P0, _ = fit_baseline(build_index(trip), dm)
    raw = [F.factor(h, r, kind, {"w": PLANT_W, "_dm": dm}, dm)
           for r, _f, h in trip]
    mbar = sum(raw) / len(raw)
    syn = []
    for (r, f, _h), m in zip(trip, raw):
        r2 = dict(r)
        lam = max(H.rate_pa(r, f, P0, BASE_USE), 1e-6) * r["pa"]
        r2["hr"] = H2._poisson(rng, lam * m / mbar)
        syn.append(r2)
    del trip, raw
    st = attach(syn)
    sdm = {k: date_means(st, k) for k in ("fam", "famcar")}
    sidx = build_index(st)
    pack = baseline_pack(sidx, sdm)
    # the oracle knows the planted w exactly; nothing fitted can beat it
    o, _ = ll_k(sidx, dict(pack[0], w=PLANT_W), kind, sdm, "HOLD")
    d, _s, _sl, _w, rob = verdict_one(sidx, kind, sdm, pack)
    return o - pack[2], d, rob


def probe(rows, kind, seeds=(11, 21, 31)):
    """Also returns the RAW robust count, which is the detectability test.

    The count matters more than the formatted string it used to hand back.
    "how many seeds recovered a planted effect that is true by construction"
    is the direct, measured answer to "could this panel have seen it" — and it
    is the answer a hard-coded oracle threshold was standing in for.
    """
    o, fi, rb = zip(*[probe_once(rows, kind, s) for s in seeds])
    return (sum(o) / len(o), sum(fi) / len(fi),
            f"{sum(rb)}/{len(rb)} robust", min(o), max(o),
            sum(rb), len(rb))


def read_ceiling(got, won, o, olo, fi, n_rob, n_seed):
    """Turn a ceiling into a verdict. A FUNCTION, so the selftest can pin it.

    This ladder has been gotten wrong three separate times — twice in UFC and
    MLB by testing "too small to see" BEFORE "did it win" (which prints real
    wins as invisible), and once here by using a hard-coded oracle cutoff of
    0.0004 for detectability. That cutoff mislabelled LOAD: oracle +0.00038,
    a hair under the line, printed "cannot be seen" on the same run where the
    fitted pipeline recovered a PLANTED load effect robustly in 3 of 3 seeds.
    A panel that recovers a true effect three times out of three can see it,
    whatever the oracle's absolute size.

    Two rules, both learned the hard way:
      1. Ask "did it win" FIRST. Low power is the explanation of last resort.
      2. Detectability is MEASURED (n_rob/n_seed), never a constant. The plant
         already answers "could this panel see it"; do not guess at a cutoff
         when the experiment is sitting right there.
    """
    if got >= o:
        return "measured >= ORACLE: noise by construction"
    if got >= olo:
        return ("won but sits inside the oracle's own seed spread: "
                "UNPROVEN, needs a placebo" if won else
                "inside the oracle's seed spread: unreadable")
    if won and got >= fi:
        return f"LIVE: robust win at {100.0 * got / o:.0f}% of oracle"
    if n_rob * 2 < n_seed:
        return (f"STILL CANNOT BE SEEN - do not bury (a planted effect "
                f"was itself only recovered {n_rob}/{n_seed})")
    return (f"DEAD: a planted effect of this size was recovered "
            f"{n_rob}/{n_seed}, so a real one would have shown")


def date_means(trip, key):
    acc = defaultdict(lambda: [0.0, 0])
    for r, _f, h in trip:
        v = h.get(key)
        if v is None:
            continue
        cell = acc[r["date"]]
        cell[0] += v
        cell[1] += 1
    return {d: (s / n if n else 0.0) for d, (s, n) in acc.items()}


# ------------------------------------------------------------------ main
def main():
    if not os.path.exists(H.CACHE):
        print("hrangles dataset cache missing — run on GitHub Actions")
        return 0
    rows = W.load_rows()
    lines = []

    def tee(s):
        print(s)
        lines.append(s)

    trip = attach(rows)
    dm = {k: date_means(trip, k) for k in ("fam", "famcar")}
    idx = build_index(trip)

    tee("=" * 74)
    tee("FATIGUE / FAMILIARITY — the four buried angles on the wide panel")
    tee(f"rows {len(rows)}   holdout n={len(idx['HOLD'])} "
        f"(was 8037 when these were filed CANNOT BE SEEN)")
    tee("=" * 74)

    # how much of the panel each feature can actually speak about. A term that
    # is 1.0 on nine rows in ten has a ceiling set by coverage, not by data.
    nr = sum(1 for _r, _f, h in idx["HOLD"] if h["rest"] is not None)
    nf = sum(1 for _r, _f, h in idx["HOLD"] if h["fam"] > 0)
    nc = sum(1 for _r, _f, h in idx["HOLD"] if h["famcar"] > 0)
    nl = sum(1 for _r, _f, h in idx["HOLD"] if h["pa30"] >= 40)
    n = len(idx["HOLD"])
    tee(f"coverage on the holdout: rest {nr}/{n} ({100.0 * nr / n:.0f}%)  "
        f"load {nl}/{n} ({100.0 * nl / n:.0f}%)  "
        f"famil {nf}/{n} ({100.0 * nf / n:.0f}%)  "
        f"famcar {nc}/{n} ({100.0 * nc / n:.0f}%)")
    rests = defaultdict(int)
    for _r, _f, h in idx["HOLD"]:
        rests[h["rest"]] += 1
    tee("  rest-day mix: " + "  ".join(
        f"{k if k is not None else 'none'}d={v}"
        for k, v in sorted(rests.items(), key=lambda kv: (kv[0] is None, kv[0]))))

    for subset in (None, "regulars"):
        tag = subset or "all"
        pack = baseline_pack(idx, dm, subset)
        tee("")
        tee(f"--- SUBSET [{tag}]  baseline tau_b={pack[0]['tau_b']} "
            f"tau_park={pack[0]['tau_park']} w_p={pack[0]['w_p']}  "
            f"TRAIN {pack[1]:+.5f}  HOLDOUT {pack[2]:+.5f} (n={pack[5]})")
        tee(f"{'angle':26s} {'w':>6s} {'holdout':>9s} {'2024':>9s} "
            f"{'2025':>9s} {'slices':>7s}  verdict")
        for kind, label in KINDS:
            d, season, sl, w, rob = verdict_one(idx, kind, dm, pack, subset)
            tee(f"{label:26s} {w:+6.2f} {d:+9.5f} {season['2024']:+9.5f} "
                f"{season['2025']:+9.5f} {sl:>5d}/4  "
                f"{'ROBUST WIN' if rob else 'no'}")
        if subset is None:
            main_res = {k: verdict_one(idx, k, dm, pack, None)
                        for k, _l in KINDS}

    tee("")
    tee(f"--- POWER CEILINGS (plant w={PLANT_W:+.2f}, 3 seeds). ORACLE = a model")
    tee("    handed the true multiplier. FITTED = what this pipeline recovers")
    tee("    from a panel where the effect is real — the honest threshold.")
    for kind, label in KINDS:
        o, fi, rb, olo, ohi, n_rob, n_seed = probe(rows, kind)
        got, won = main_res[kind][0], main_res[kind][4]
        read = read_ceiling(got, won, o, olo, fi, n_rob, n_seed)
        tee(f"{label:26s} ORACLE {o:+.5f} [{olo:+.5f}..{ohi:+.5f}]  "
            f"FITTED {fi:+.5f} ({rb})  measured {got:+.5f}   {read}")

    vd = os.environ.get("VERDICT_OUT") or os.path.join(
        HERE, "..", "experiments", "MLB-FATIGUE-WIDE-VERDICT.md")
    os.makedirs(os.path.dirname(vd), exist_ok=True)
    with open(vd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"verdict -> {vd}")
    return 0


# ------------------------------------------------------------------ selftest
def selftest():
    # 1) the leak boundary. A batter's first row must see nothing; his second
    #    must see exactly the first, and never the same day or later.
    rows = [{"bat": "b", "bh": "R", "ph": "L", "sp": "p1", "venue": "v",
             "dn": "day", "home": 1, "pa": 4, "hr": 1, "slot": 1, "pk": 1,
             "date": "2025-04-01"},
            {"bat": "b", "bh": "R", "ph": "L", "sp": "p1", "venue": "v",
             "dn": "day", "home": 1, "pa": 5, "hr": 0, "slot": 1, "pk": 2,
             "date": "2025-04-04"},
            {"bat": "b", "bh": "R", "ph": "L", "sp": "p1", "venue": "v",
             "dn": "day", "home": 1, "pa": 3, "hr": 0, "slot": 1, "pk": 3,
             "date": "2025-04-05"}]
    _s, h = build_history(rows)
    assert h[0]["rest"] is None and h[0]["prior_pa"] == 0 and h[0]["fam"] == 0
    assert h[1]["rest"] == 3, h[1]
    assert h[1]["prior_pa"] == 4, "prior_pa leaked today's PA"
    assert h[1]["fam"] == 1 and h[1]["famcar"] == 1
    assert h[2]["rest"] == 1 and h[2]["prior_pa"] == 9 and h[2]["fam"] == 2

    # 2) the offseason is not a day off. This is the fix that motivated the
    #    file: under the old cap-at-5 rule the second row below scored as a
    #    maximally-rested batter, which is why April 2025 would have read as
    #    league-wide peak freshness.
    winter = [dict(rows[0], date="2024-09-28", pk=1),
              dict(rows[0], date="2025-04-01", pk=2)]
    _s, hw = build_history(winter)
    assert hw[1]["rest"] is None, f"offseason gap read as rest: {hw[1]['rest']}"
    near = [dict(rows[0], date="2025-04-01", pk=1),
            dict(rows[0], date="2025-04-11", pk=2),
            dict(rows[0], date="2025-04-22", pk=3)]
    _s, hn = build_history(near)
    assert hn[1]["rest"] == LONG_GAP, "the boundary itself was excluded"
    assert hn[2]["rest"] is None, "an 11-day gap was allowed through"

    # 3) within-season familiarity RESETS at the year boundary and career
    #    familiarity does not. Without this the 2024 holdout scores a feature
    #    that is zero on every row it contains.
    two = [dict(rows[0], date="2024-06-01", pk=1),
           dict(rows[0], date="2024-06-20", pk=2),
           dict(rows[0], date="2025-05-01", pk=3)]
    _s, h2 = build_history(two)
    assert [x["fam"] for x in h2] == [0, 1, 0], [x["fam"] for x in h2]
    assert [x["famcar"] for x in h2] == [0, 1, 2], [x["famcar"] for x in h2]

    # 4) alignment. attach_features reorders by day, so the triple builder has
    #    to re-pair by identity — this is the check that a Tuesday row is not
    #    wearing Sunday's rest count.
    many = []
    # Dates are deliberately OUT of order, because the thing under test is that
    # attach_features regroups them by day and the history still follows the
    # right row. pa doubles per batter so that after k games prior_pa is
    # k * 2**j — unique across (batter, games-played) once famcar is in the
    # tuple, which is what gives the check below teeth.
    for i, d in enumerate(("2025-04-03", "2025-04-01",
                           "2025-04-04", "2025-04-02")):
        for j in range(4):
            many.append({"bat": f"b{j}", "bh": "R", "ph": "L", "sp": "p1",
                         "venue": "v", "dn": "day", "home": 1, "pa": 2 ** j,
                         "hr": j % 2, "slot": 1, "pk": i * 10 + j, "date": d})
    trip = attach(many)
    srt, hist = build_history(many)
    ref = {id(r): hh for r, hh in zip(srt, hist)}
    # Compare by VALUE, not identity. build_history mints a fresh feature dict
    # on every call, so `is` against a second call can never hold — an earlier
    # draft asserted `is` here and failed on correct code. Identity is still
    # what does the pairing inside attach(); this only checks the result.
    #
    # Debut rows are excluded from the uniqueness guard on purpose: every
    # batter's first game has an all-zero history by definition, so requiring
    # ALL histories to be distinct is unsatisfiable, not strict. The rows that
    # carry information are the ones that must be distinguishable.
    live = [tuple(sorted(h.items())) for h in hist if h["famcar"] > 0]
    assert len(live) == 12 and len(set(live)) == 12, (
        "fixture is degenerate — identical histories would make the "
        f"alignment assert below vacuous ({len(set(live))}/{len(live)})")
    for r, _f, hh in trip:
        assert hh == ref[id(r)], "history was paired with the wrong row"
    assert len(trip) == len(many)

    # 4b) the ceiling ladder itself. Three shipped bugs have lived in this
    #      ladder, so it is now a function with a test rather than an if-chain
    #     buried in main() that only ever gets exercised by a 6-minute run.
    #     LOAD's real numbers from 2026-07-29 are the regression case.
    assert read_ceiling(-0.00001, False, 0.00038, 0.00035, 0.00037, 3, 3
                        ).startswith("DEAD"), (
        "regression: LOAD's oracle is a hair under the old 0.0004 cutoff, but "
        "a planted LOAD effect was recovered 3/3 — that is DEAD, not invisible")
    assert read_ceiling(-0.00003, False, 0.00006, 0.00002, 0.00008, 0, 3
                        ).startswith("STILL CANNOT BE SEEN"), (
        "REST: the plant itself was never recovered, so a null is unreadable")
    # a genuine win must not be swallowed by the low-power branch — the
    # ordering bug that shipped twice.
    assert read_ceiling(0.00030, True, 0.00040, 0.00038, 0.00020, 3, 3
                        ).startswith("LIVE"), "ordering bug is back"
    assert read_ceiling(0.00050, True, 0.00040, 0.00038, 0.00020, 3, 3
                        ).startswith("measured >= ORACLE")
    assert "UNPROVEN" in read_ceiling(0.00039, True, 0.00040, 0.00038,
                                      0.00020, 3, 3)

    # 5) the factors move over the right thing, and a None rest is silent.
    dm = {"fam": {"2025-04-01": 0.0}, "famcar": {}}
    base = {"rest": None, "pa7": 0, "pa30": 0, "fam": 0, "famcar": 0,
            "prior_pa": 0}
    P = {"w": 0.2, "_dm": dm}
    assert F.factor(base, rows[0], "rest", P, dm) == 1.0, "None rest spoke up"
    assert F.factor(dict(base, rest=1), rows[0], "rest", P, dm) == 1.0
    assert F.factor(dict(base, rest=5), rows[0], "rest", P, dm) > 1.0

    # 6) end to end: a large planted REST effect must be visible to the oracle.
    #    Realistic plants land near +0.0001 on a small synthetic panel, so this
    #    plants hard on purpose — asserting on a realistic plant here would be
    #    asserting that a coin came up heads.
    rng = random.Random(7)
    syn = []
    days = ([f"2024-{m:02d}-{d:02d}" for m in (4, 5, 6, 7, 8, 9)
             for d in range(1, 29)]
            + [f"2025-{m:02d}-{d:02d}" for m in (4, 5, 6) for d in range(1, 29)])
    for day in days:
        for i in range(26):
            # roughly two thirds of batters play any given day, so rest is a
            # real mix of 1, 2 and 3 rather than a constant
            if (i * 7 + int(day[-2:])) % 3 == 0:
                continue
            syn.append({"bat": f"b{i}", "bh": "LR"[i % 2],
                        "ph": "LR"[i % 3 == 0], "sp": f"p{i % 9}",
                        "venue": f"v{i % 8}", "home": 1 if i % 2 else 0,
                        "dn": "day" if (i + int(day[-2:])) % 2 else "night",
                        "pa": 4, "hr": 1 if rng.random() < 0.12 else 0,
                        "slot": 1 + i % 9, "pk": i, "date": day})
    save = globals()["PLANT_W"]
    globals()["PLANT_W"] = 1.2
    try:
        o, fi, _rb = probe_once(syn, "rest", 5)
    finally:
        globals()["PLANT_W"] = save
    assert o > 0.001, f"oracle could not see a hard-planted rest effect: {o:+.5f}"
    assert o >= fi - 1e-9, f"fitted {fi:+.5f} beat oracle {o:+.5f} — impossible"

    # 7) windows are inherited, not redefined. If the wide-panel file ever
    #    moves its holdout, this file must move with it rather than quietly
    #    scoring a different period.
    ix = build_index(trip)
    assert set(ix) == set(W.WINDOW_SETS)
    assert len(ix["ALL"]) == len(ix["TRAIN"]) + len(ix["HOLD"])

    print("FATIGUE WIDE SELFTEST PASS — no leak, offseason excluded from rest, "
          "familiarity resets by season while career does not, history pairs "
          f"by identity, oracle sees a planted effect (+{o:.4f} >= {fi:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
