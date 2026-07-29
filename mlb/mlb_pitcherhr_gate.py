#!/usr/bin/env python3
"""
mlb_pitcherhr_gate.py — gates 2 and 3 for the one angle the wide panel revived.

WHAT HAPPENED. On the single-June panel, "F pitcher HR" (the opposing
starter's own HR-allowed rate) was a NULL. On the two-season wide panel it is
a ROBUST WIN that clears baseline in the 2024 holdout and the 2025 holdout
SEPARATELY, 4/4 slices, +0.00025 pooled. That is the first MLB angle to flip
direction under more power.

WHY IT IS NOT SHIPPED ON THAT ALONE. Two numbers from the wide-panel ceiling
run argue the other way, and they have to be answered:

  * measured +0.00025 sits INSIDE the oracle's own seed spread
    (+0.00008..+0.00055). A bound that wobbles 7x across re-rolls cannot
    certify a result that lands in the middle of the wobble.
  * FITTED came back +0.00004 with 0/3 robust. That is the pipeline scoring a
    panel where a +-15% per-pitcher effect was PLANTED AND KNOWN TRUE — and it
    barely found it. A real-data result that beats what the pipeline manages
    on a known-true panel is suspicious, not reassuring.

THE PRIOR THAT CUTS THE OTHER WAY. The production board already ships starter
HR-allowed pools, validated when they are seeded with a FULL PRIOR SEASON —
and it was explicitly found that in-season-only pools are noise. The wide
panel's 2024 burn-in is exactly that prior-season seeding. So this is a
REPLICATION of a known-good result under the condition it was known to need,
not a new discovery. Batch 1 called it NULL because batch 1 scored only 2025
with 2024 used for warmth; the cells were mature but the holdout was 8k rows.

GATE 2, START-LEVEL PLACEBO. Permute WHICH PITCHER each start belongs to,
keeping every start's rows together and keeping pitcher HAND untouched. That
destroys exactly one thing — the link between a starter's identity and his own
HR-allowed history — while preserving the game clustering, the number of
starts per label, the platoon structure and the schedule. A row-level shuffle
would have been wrong here: it would scatter one "pitcher" across hundreds of
games and test a different null.

GATE 3, SHAPE. The effect must be carried by pitchers whose cells are actually
populated, and it must not require a knife-edge weight.

Run: python3 mlb_pitcherhr_gate.py [--selftest]
"""
import os
import random
import sys
from collections import defaultdict

import mlb_hrangles2_experiment as H2
import mlb_widepanel_experiment as W

LABEL = "F pitcher HR"
ANGLE = [a for a in W.ANGLES if a[0] == LABEL][0]
TRIALS = int(os.environ.get("PLACEBO_TRIALS", "20"))


def run(idx, angle=ANGLE, pack=None):
    """Score one angle against a pre-packed baseline on an indexed panel."""
    if pack is None:
        pack = W.baseline_pack(idx)
    P0, base_tr, base_h, base_season, base_per, _n = pack
    return W.verdict_one(idx, angle, P0, base_tr, base_h,
                         base_season, base_per)


def shuffle_starts(rows, seed):
    """Re-deal which pitcher each START belongs to.

    A start is (date, sp): all the batters who faced that man that day. The
    permutation moves whole starts, so every pitcher label keeps a realistic
    workload shape and every game keeps its nine hitters together. Pitcher
    HAND (ph) is deliberately NOT shuffled — hand is a real attribute the
    model prices separately through the platoon term, and scrambling it would
    make this a test of two things at once.

    Returns new row dicts; the caller's rows are never mutated, because the
    same panel is reused across every trial.
    """
    rng = random.Random(seed)
    starts = sorted({(r["date"], r["sp"]) for r in rows})
    labels = [sp for _d, sp in starts]
    rng.shuffle(labels)
    remap = {k: lab for k, lab in zip(starts, labels)}
    return [dict(r, sp=remap[(r["date"], r["sp"])]) for r in rows]


def main():
    rows = W.load_rows()
    lines = []

    def tee(s):
        print(s)
        lines.append(s)

    idx = W.build_index(H2.attach2(rows))
    pack = W.baseline_pack(idx)
    tee("=" * 74)
    tee("PITCHER-HR GATE — start-level placebo + shape, on the wide panel")
    tee("=" * 74)
    d, season, sl, kv, rob = run(idx, pack=pack)
    tee(f"REAL  n_hold={pack[5]}  w_p={kv}  holdout dLL {d:+.5f}  "
        f"2024 {season['2024']:+.5f}  2025 {season['2025']:+.5f}  "
        f"slices {sl}/4  robust={rob}")

    # ------------------------------------------------------------- gate 2
    tee("")
    tee(f"--- GATE 2: START-LEVEL PLACEBO ({TRIALS} trials, full tune each)")
    tee("    The baseline is reused across trials ON PURPOSE and the first")
    tee("    trial asserts it: the baseline uses league, batter, park and")
    tee("    platoon cells, none of which are keyed on starter identity, so")
    tee("    an sp permutation provably cannot move it. Re-fitting it 20")
    tee("    times would only add noise to the comparison.")
    fires = beats = 0
    ds = []
    for t in range(TRIALS):
        sidx = W.build_index(H2.attach2(shuffle_starts(rows, 500 + t)))
        if t == 0:
            chk = W.ll_k(sidx, pack[0], W.BASE, "HOLD")[0]
            assert abs(chk - pack[2]) < 1e-12, (
                f"baseline moved under an sp shuffle ({chk} vs {pack[2]}) — "
                "the reuse assumption is wrong, re-fit per trial")
        dd, ss, s2, _k, rr = run(sidx, pack=pack)
        ds.append(dd)
        fires += 1 if rr else 0
        beats += 1 if dd >= d else 0
    ds.sort()
    tee(f"ship rule fired on noise: {fires}/{TRIALS} "
        f"({100.0 * fires / TRIALS:.0f}%)")
    tee(f"noise dLL >= real ({d:+.5f}): {beats}/{TRIALS} "
        f"({100.0 * beats / TRIALS:.0f}%)  <- this is the p-value")
    tee(f"noise dLL  min {ds[0]:+.5f}  median {ds[len(ds) // 2]:+.5f}  "
        f"max {ds[-1]:+.5f}")

    # ------------------------------------------------------------- gate 3
    tee("")
    tee("--- GATE 3: SHAPE")
    tee("    (a) the effect must be carried by starters whose HR-allowed cell")
    tee("        is actually populated. A term that pays off just as well on")
    tee("        pitchers with 40 batters faced as on pitchers with 600 is")
    tee("        not reading pitcher quality, it is reading something else.")
    for name, lo, hi in (("thin cells   (pit PA < 200)", 0, 200),
                         ("medium       (200-600)", 200, 600),
                         ("mature cells (600+)", 600, 10 ** 9)):
        sub = [rf for rf in idx["ALL"] if lo <= rf[1]["pit"][1] < hi]
        if len(sub) < 3000:
            tee(f"    {name:28s} n={len(sub):6d}  too thin to read")
            continue
        try:
            si = W.build_index(sub)
            sp = W.baseline_pack(si)
            dd, ss, s2, _k, rr = run(si, pack=sp)
            tee(f"    {name:28s} n={len(sub):6d}  dLL {dd:+.5f}  "
                f"slices {s2}/4  robust={rr}")
        except ZeroDivisionError:
            tee(f"    {name:28s} n={len(sub):6d}  unscorable slice")

    tee("")
    tee("    (b) weight sweep. w_p scales how much of the raw pitcher ratio is")
    tee("        believed. A real signal degrades gently either side of its")
    tee("        best value; a grid artifact lives in one cell and dies in the")
    tee("        neighbours. The shipped value is NOT changed by this sweep —")
    tee("        these are holdout numbers and picking the best one is a leak.")
    for w in (0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.4):
        a2 = (LABEL, ANGLE[1], "w_p", (w,))
        dd, ss, s2, _k, rr = run(idx, angle=a2, pack=pack)
        tee(f"    w_p {w:4.2f}   dLL {dd:+.5f}  2024 {ss['2024']:+.5f}  "
            f"2025 {ss['2025']:+.5f}  slices {s2}/4")

    vd = os.environ.get("VERDICT_OUT") or os.path.join(
        W.HERE, "..", "experiments", "MLB-PITCHERHR-GATE.md")
    os.makedirs(os.path.dirname(vd), exist_ok=True)
    with open(vd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"verdict -> {vd}")
    return 0


def selftest():
    rows = []
    for d in ("2024-08-01", "2024-08-02", "2025-06-01"):
        for sp in ("p1", "p2", "p3"):
            for i in range(9):
                rows.append({"bat": f"b{i}", "bh": "R", "ph": "L" if sp == "p1"
                             else "R", "sp": sp, "venue": "v", "dn": "day",
                             "home": 1, "pa": 4, "hr": i % 4 == 0, "slot": i + 1,
                             "date": d})
    before = [r["sp"] for r in rows]
    sh = shuffle_starts(rows, 1)
    assert [r["sp"] for r in rows] == before, "shuffle mutated the caller's rows"
    # whole starts move together: every (date, original-start) block must end
    # up carrying exactly one label.
    byblock = defaultdict(set)
    for r0, r1 in zip(rows, sh):
        byblock[(r0["date"], r0["sp"])].add(r1["sp"])
    assert all(len(v) == 1 for v in byblock.values()), "a start was split apart"
    # the multiset of start-labels is preserved, so workloads stay realistic
    c0 = sorted(defaultdict(int, {k: 1 for k in byblock}).keys())
    n0 = defaultdict(int)
    n1 = defaultdict(int)
    for r in rows:
        n0[r["sp"]] += 1
    for r in sh:
        n1[r["sp"]] += 1
    assert sorted(n0.values()) == sorted(n1.values()), "workload shape changed"
    assert len(c0) == 9
    # hand is untouched: this gate tests identity, not handedness
    assert [r["ph"] for r in sh] == [r["ph"] for r in rows], "ph was shuffled"
    # it must actually be a shuffle, and seeded
    assert any(a != b["sp"] for a, b in zip(before, sh)), "shuffle was a no-op"
    assert ([r["sp"] for r in shuffle_starts(rows, 1)]
            != [r["sp"] for r in shuffle_starts(rows, 2)]), "seed ignored"
    print("PITCHER-HR GATE SELFTEST PASS — starts move whole, workload shape "
          "and pitcher hand preserved, shuffle is seeded and non-destructive")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
