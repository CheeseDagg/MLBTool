#!/usr/bin/env python3
"""Re-derive LIVE_LEVEL -- the flat level factor between the backtest REPLAY and the
PUBLISHED board -- from the production grading log, and report whether the currently
shipped value still holds.

WHY THIS IS NOT mlb_recalibrate.py
  mlb_recalibrate refits CALIB_ANCHORS, the SHAPE of the raw->calibrated curve, by
  pooling ~25k backtest-replay rows with the live graded rows. Live is ~2% of that
  pool, so a level gap that exists ONLY in production -- the board is built the night
  before off projected lineups, the replay was not -- cannot move the anchors. It has
  to be measured on production rows alone, which is what this does.

WHY A FLAT MULTIPLIER AND NOT A NEW CURVE
  Bucketing the graded board by predicted % makes the bias look concentrated at the
  low end. It is not: the low buckets are a couple of thin-slate days, and once day
  fixed effects are in, the fitted slope on logit(p) is ~1.4 with a CI straddling 1 --
  no evidence the within-day SPREAD is wrong. Only the LEVEL is. A flat multiplier
  leaves the board's ordering identical, which matters because the board is read by
  rank, not by the number.

WHY SHRUNK
  The in-sample ratio is the best fit to 20 days and will overfit them. The shipped
  value is picked by leave-one-day-out: fit k on the other days, score the held-out
  day, and take the shrinkage that wins out of sample. The full correction typically
  overshoots. This is the same test that killed the temperature nowcast.

Usage
  python3 mlb_livelevel.py             # report; prints SHIP / KEEP against mlb_hr.py
  python3 mlb_livelevel.py --selftest  # offline unit checks, no data files needed
Exit 0 always ("keep current" is success.)
"""
import collections
import csv
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRADED = os.path.join(HERE, "data", "hr_graded.csv")

MIN_DAY_ROWS = 10      # a day thinner than this is a partial grade, not a day
MIN_DAYS = 10          # below this there is nothing to leave one out of
SHRINK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
K_LO, K_HI = 0.40, 1.40


def load_days(path=GRADED):
    """{date: [(p, hit)]} for fully-graded days, p as a probability."""
    by = collections.defaultdict(list)
    if not os.path.exists(path):
        return {}
    # WIDTH-SAFE read, not csv.DictReader. The ledger's header is written once, at
    # file creation, so a column added to GCOLS makes every later row one field wider
    # than the header names it. DictReader would shift `outcome` past the insertion
    # point and EVERY row would fail the ("hr","no") test below — load_days returns
    # {}, the live-level fit falls back to its default, and nothing anywhere errors.
    try:
        import mlb_grade
        src = mlb_grade.read_graded(path)
    except Exception:
        src = list(csv.DictReader(open(path)))
    for r in src:
        if r.get("outcome") not in ("hr", "no"):
            continue
        try:
            p = float(r["hr_pct"]) / 100.0
        except (TypeError, ValueError):
            continue
        by[r["date"]].append((p, 1.0 * (r["outcome"] == "hr")))
    return {d: g for d, g in by.items() if len(g) >= MIN_DAY_ROWS}


def ll(rows, k):
    s = 0.0
    for p, y in rows:
        q = min(max(p * k, 1e-6), 0.999)
        s += math.log(q) if y else math.log(1 - q)
    return s


def fit_k(rows):
    """ML multiplier on a grid. Coarse-then-fine; the likelihood is unimodal in k."""
    best = None
    for i in range(int(K_LO * 100), int(K_HI * 100) + 1):
        k = i / 100.0
        v = ll(rows, k)
        if best is None or v > best[1]:
            best = (k, v)
    return best[0]


def loo(by):
    """[(shrink, mean LL)] scoring each day under a k fit WITHOUT that day."""
    days = sorted(by)
    n = sum(len(by[d]) for d in days)
    out = []
    for lam in SHRINK_GRID:
        tot = 0.0
        for d in days:
            k_out = fit_k([x for o in days if o != d for x in by[o]])
            tot += ll(by[d], 1.0 + lam * (k_out - 1.0))
        out.append((lam, tot / n))
    return out


def bootstrap_ratio(by, draws=4000, seed=7):
    """Day-clustered CI on actual/predicted. Days, not rows, are the unit: rows inside
    a day share the league HR environment and are nowhere near independent."""
    days = sorted(by)
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        samp = [by[rng.choice(days)] for _ in days]
        sp = sum(p for g in samp for p, _ in g)
        sy = sum(y for g in samp for _, y in g)
        out.append(sy / sp if sp else 1.0)
    out.sort()
    lo = out[int(0.025 * draws)]
    hi = out[int(0.975 * draws) - 1]
    return lo, hi, sum(1 for x in out if x < 1.0) / draws


def report(by, out=print):
    days = sorted(by)
    rows = [x for d in days for x in by[d]]
    n = len(rows)
    sp = sum(p for p, _ in rows)
    sy = sum(y for _, y in rows)
    out(f"production log: {len(days)} full days, n={n}, {days[0]}..{days[-1]}")
    out(f"  predicted {100 * sp / n:5.2f}%   actual {100 * sy / n:5.2f}%   "
        f"ratio {sy / sp:.3f}")
    lo, hi, phot = bootstrap_ratio(by)
    out(f"  day-clustered 95% CI [{lo:.3f}, {hi:.3f}]   P(model runs hot) = {phot:.3f}")

    k_in = fit_k(rows)
    out(f"  in-sample ML multiplier k={k_in:.2f}")

    if len(days) < MIN_DAYS:
        out(f"  only {len(days)} days -- too few for leave-one-out; KEEP")
        return None

    out("  leave-one-day-out (fit on the other days, score the held-out day):")
    scores = loo(by)
    base = dict(scores)[0.0]
    best = max(scores, key=lambda t: t[1])
    for lam, v in scores:
        mark = "  <- best" if lam == best[0] else ""
        out(f"    shrink {lam:.2f}   LL {v:+.5f}   vs no-change {v - base:+.5f}{mark}")
    if best[0] == 0.0 or best[1] <= base:
        out("  no shrinkage beats leaving the board alone -- KEEP k=1.00")
        return None
    k_ship = round(1.0 + best[0] * (k_in - 1.0), 2)
    per = " ".join(f"{d[5:]}:{fit_k([x for o in days if o != d for x in by[o]]):.2f}"
                   for d in days)
    out(f"  per-day LOO k: {per}")
    out(f"  SHIP k={k_ship:.2f}  (shrink {best[0]:.2f} toward 1 from in-sample "
        f"{k_in:.2f})")
    return k_ship


def selftest():
    ok = [0, 0]

    def chk(c, msg):
        ok[1] += 1
        ok[0] += bool(c)
        print(("PASS  " if c else "FAIL  ") + msg)

    # a perfectly calibrated panel must fit k=1
    rows = [(0.20, 1.0)] * 200 + [(0.20, 0.0)] * 800
    chk(abs(fit_k(rows) - 1.0) < 0.02, "calibrated panel fits k=1.00")

    # a panel hitting at half its prediction must fit k=0.5
    rows = [(0.20, 1.0)] * 100 + [(0.20, 0.0)] * 900
    chk(abs(fit_k(rows) - 0.5) < 0.02, "panel hitting half its number fits k=0.50")

    # ll must be maximised at the fitted k
    k = fit_k(rows)
    chk(ll(rows, k) > ll(rows, k + 0.1) and ll(rows, k) > ll(rows, k - 0.1),
        "likelihood is a maximum at the fitted k, not an endpoint")

    # a flat multiplier cannot reorder the board -- this is the claim the whole
    # approach rests on, so it is asserted rather than assumed
    board = [0.31, 0.27, 0.223, 0.222, 0.19, 0.14, 0.06]
    chk([p * 0.88 for p in board] == sorted([p * 0.88 for p in board], reverse=True),
        "flat multiplier preserves board order")

    # thin days are dropped, not averaged in
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("date,hr_pct,outcome\n")
        for i in range(12):
            f.write(f"2026-01-01,20.0,{'hr' if i < 3 else 'no'}\n")
        f.write("2026-01-02,20.0,hr\n")          # 1-row day: partial grade
        p = f.name
    by = load_days(p)
    chk(list(by) == ["2026-01-01"], "a 1-row day is dropped as a partial grade")
    os.unlink(p)

    # LOO on a panel with one wild day must shrink, not chase it
    by = {f"2026-01-{i:02d}": [(0.20, 1.0)] * 4 + [(0.20, 0.0)] * 16
          for i in range(1, 13)}
    by["2026-01-13"] = [(0.20, 1.0)] * 20        # a day that hit everything
    sc = dict(loo(by))
    chk(sc[0.0] >= sc[1.0] - 1e-9,
        "a calibrated panel plus one freak day does not reward full correction")

    print(f"\n{ok[0]}/{ok[1]} checks pass")
    return 0 if ok[0] == ok[1] else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    by = load_days()
    if not by:
        print("no graded production rows yet - nothing to measure")
        return 0
    k_ship = report(by)
    try:
        import mlb_hr
        cur = getattr(mlb_hr, "LIVE_LEVEL", 1.0)
    except Exception as e:                                    # noqa: BLE001
        print(f"(could not read mlb_hr.LIVE_LEVEL: {e})")
        return 0
    print(f"\nshipped LIVE_LEVEL = {cur:.2f}")
    if k_ship is None:
        print("  measurement says leave it alone; edit mlb_hr.py only if you disagree")
    elif abs(k_ship - cur) < 0.02:
        print("  matches the measurement - KEEP")
    else:
        print(f"  measurement has moved to {k_ship:.2f} - consider updating mlb_hr.py "
              f"(this script does NOT edit it; the factor is small and hand-checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
