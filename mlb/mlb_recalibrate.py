#!/usr/bin/env python3
"""Monthly recalibration — refits CALIB_ANCHORS in mlb_hr.py from the season's graded
results, and applies the new curve ONLY if it beats the current one on held-out
(most-recent) data it never saw during fitting.

Why monthly + gated: ~30 graded rows/day is far too few to tune on daily — a model
that retunes on that mostly chases variance and gets worse while looking busier.
Month-scale batches (~900 rows) plus an out-of-sample gate means the curve can only
ratchet toward accuracy.

Data used
  mlb/data/hr_backtest.csv   raw (pre-calibration) hr_pct + outcome, full season replay
  mlb/data/hr_graded.csv     production log: post-calibration hr_pct + outcome (grows daily)
                             -> inverted through the CURRENT anchors to recover raw
                                (the anchor map is monotonic piecewise-linear, so exact)

Method
  1. Pool both sources into (date, raw, hit) rows.
  2. Hold out the most recent HOLDOUT_DAYS of rows — the fit never sees them.
  3. On the training rows, bucket raw into N_BINS quantile bins; each bin's anchor is
     (mean raw, actual HR%). Enforce monotonicity by pooling adjacent violators (PAV).
  4. Score OLD vs NEW anchors on the holdout by Brier. Apply only if
     new < old - MIN_GAIN and the holdout has >= MIN_HOLDOUT rows.
  5. Apply = rewrite the single CALIB_ANCHORS line in mlb_hr.py (assert-guarded).

Usage
  python3 mlb_recalibrate.py             # fit, gate, apply-if-better
  python3 mlb_recalibrate.py --dry-run   # fit + report, never touch mlb_hr.py
  python3 mlb_recalibrate.py --selftest  # offline unit checks (no files needed)
Exit 0 always (a "kept current" month is success, not failure).
"""
import csv, json, os, re, sys, datetime as dt

HERE          = os.path.dirname(os.path.abspath(__file__))
HR_PY         = os.path.join(HERE, "mlb_hr.py")
DATA          = os.path.join(HERE, "data")
REPORT        = os.path.join(DATA, "recalib_report.json")

HOLDOUT_DAYS  = 21      # most-recent slice the fit never sees
N_BINS        = 8       # quantile buckets for the new curve
MIN_BIN       = 150     # min rows per bucket (bins merge up if thinner)
MIN_HOLDOUT   = 400     # don't trust a verdict on less than this
MIN_GAIN      = 1e-4    # Brier must improve by at least this to ship
MIN_NEW_ROWS  = 500     # need this much fresh graded data since the anchors were last fit

# The live level factor lives in mlb_hr.py and is applied AFTER the anchors. Read it
# rather than copy it. See row_raw for why the date matters, and brier() for why the
# gate has to score through it.
#
# Fail-soft to 1.0 so this script never depends on the engine importing cleanly -- but
# LOUDLY, and with the apply step disarmed. A silent fallback here does not degrade the
# run, it inverts it: brier() would then gate on an objective production never ships and
# would prefer whichever candidate has absorbed the level, which live_level_pct then
# applies a second time. Reporting is still useful without the engine; rewriting
# mlb_hr.py is not.
LIVE_LEVEL_FROM = "2026-07-31"
LIVE_LEVEL_OK = True
try:
    sys.path.insert(0, HERE)
    from mlb_hr import LIVE_LEVEL
except Exception as _e:                                        # noqa: BLE001
    LIVE_LEVEL, LIVE_LEVEL_OK = 1.0, False
    print(f"! could not import mlb_hr ({type(_e).__name__}: {_e}) -- LIVE_LEVEL unknown, "
          f"assuming 1.0. This run is REPORT-ONLY; it will not rewrite anchors.",
          file=sys.stderr)


# ---------- current anchors: read from mlb_hr.py (single source of truth) ----------
# Matches the assignment across MULTIPLE physical lines (the live file wraps it) by
# scanning from "CALIB_ANCHORS = [" to the balanced closing bracket.
ANCHOR_RE = re.compile(r"CALIB_ANCHORS\s*=\s*\[")

def _anchor_span(src):
    """(start, end) character span of the full CALIB_ANCHORS assignment (through its
    trailing comment, end of physical line). Asserts exactly one assignment exists."""
    ms = list(ANCHOR_RE.finditer(src))
    assert len(ms) == 1, f"CALIB_ANCHORS assignment must appear exactly once, found {len(ms)}"
    start = ms[0].start()
    i, depth = src.index("[", ms[0].start()), 0
    for j in range(i, len(src)):
        if src[j] == "[": depth += 1
        elif src[j] == "]":
            depth -= 1
            if depth == 0:
                end = src.index("\n", j) if "\n" in src[j:] else len(src)
                return start, end, i, j + 1
    raise RuntimeError("unbalanced brackets in CALIB_ANCHORS")

def read_anchors(src):
    _, _, i, j = _anchor_span(src)
    lit = src[i:j].replace("(", "[").replace(")", "]")
    return [tuple(p) for p in json.loads(re.sub(r",\s*\]", "]", lit))]

def apply_map(anchors, p):
    a = anchors
    if p <= a[0][0]:
        return p
    for (x0, y0), (x1, y1) in zip(a, a[1:]):
        if p <= x1:
            t = (p - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    (x0, y0), (x1, y1) = a[-2], a[-1]
    s = (y1 - y0) / (x1 - x0) if x1 > x0 else 1.0
    return y1 + s * (p - x1)

def invert_map(anchors, y):
    """calibrated -> raw. Valid because the anchor map is monotonic increasing."""
    a = anchors
    if y <= a[0][1]:
        return y
    for (x0, y0), (x1, y1) in zip(a, a[1:]):
        if y <= y1:
            t = (y - y0) / (y1 - y0) if y1 > y0 else 0.0
            return x0 + t * (x1 - x0)
    (x0, y0), (x1, y1) = a[-2], a[-1]
    s = (x1 - x0) / (y1 - y0) if y1 > y0 else 1.0
    return x1 + s * (y - y1)


# ---------- data ----------
def row_raw(r, anchors):
    """Recover a graded row's RAW (pre-calibration) hr%. Prefer the stored `hr_raw`
    column — written at prediction time under that day's own anchors, the only
    leak-free source. Fall back to inverting `hr_pct` through `anchors` ONLY for legacy
    rows written before hr_raw existed. (Inversion is exact only within one anchor
    epoch; after a refit, inverting an old row through the NEW anchors recovers a wrong
    raw and biases the next fit — which is exactly what hr_raw removes.)

    LIVE_LEVEL: since 2026-07-31 the published hr_pct is anchors THEN a flat live
    level factor, so inverting hr_pct through the anchors alone would recover a raw
    that is too low and would drag the next anchor fit down — the level correction
    would get baked into the shape and then re-applied on top of itself. Rows from
    that date carry hr_raw, so the fallback never fires for them; the divide below is
    belt-and-braces in case a row ever lands without one."""
    hr_raw = r.get("hr_raw")
    if hr_raw is not None and str(hr_raw).strip() != "":
        return float(hr_raw)
    pct = float(r["hr_pct"])
    if str(r.get("date") or "") >= LIVE_LEVEL_FROM and LIVE_LEVEL:
        pct = pct / LIVE_LEVEL
    return invert_map(anchors, pct)

def _graded_rows():
    """Read hr_graded.csv WIDTH-SAFELY, not by header.

    This workflow is monthly and separate from the daily grade job, so it can meet
    the ledger in the window where mlb_grade has appended rows of a new width under
    a header written a generation ago. csv.DictReader trusts the header, shifts
    every field past the new column, and hands back rows whose `outcome` is really
    the neighbouring value — which are then all discarded as un-graded, silently
    shrinking the refit sample. mlb_grade.read_graded maps each row by its own
    width instead. Falls back to DictReader only if that import fails.
    """
    gr = os.path.join(DATA, "hr_graded.csv")
    if not os.path.exists(gr): return []
    try:
        import mlb_grade
        return mlb_grade.read_graded(gr)
    except Exception as e:
        print(f"  width-safe graded read unavailable ({type(e).__name__}); "
              f"falling back to header parse")
        return list(csv.DictReader(open(gr)))

def graded_dates():
    """Dates of PRODUCTION graded rows only (excludes the static backtest replay), for
    measuring how much genuinely-new data has arrived since the last refit."""
    return [r["date"] for r in _graded_rows()
            if r.get("outcome") in ("hr", "no") and r.get("date")]

def load_rows(anchors):
    """[(date, raw_pct, hit)] from backtest (raw already) + graded (stored hr_raw, else
    inverted)."""
    rows = []
    bt = os.path.join(DATA, "hr_backtest.csv")
    if os.path.exists(bt):
        for r in csv.DictReader(open(bt)):
            if r.get("outcome") not in ("hr", "no"):
                continue
            try:
                rows.append((r["date"], float(r["hr_pct"]), 1 if r["outcome"] == "hr" else 0))
            except (KeyError, ValueError):
                continue
    for r in _graded_rows():
        if r.get("outcome") not in ("hr", "no"):
            continue
        try:
            raw = row_raw(r, anchors)
            rows.append((r["date"], raw, 1 if r["outcome"] == "hr" else 0))
        except (KeyError, ValueError):
            continue
    rows.sort(key=lambda x: x[0])
    return rows


# ---------- fit ----------
def fit_anchors(train):
    """quantile-bucket (raw, hit) -> monotonic [(raw_mean, actual_pct)...] with (0,0) head."""
    pts = sorted((raw, hit) for _, raw, hit in train)
    n = len(pts)
    bins, size = [], max(MIN_BIN, n // N_BINS)
    i = 0
    while i < n:
        chunk = pts[i:i + size]
        if len(chunk) < MIN_BIN and bins:          # tail too thin -> merge into last
            bins[-1].extend(chunk)
        else:
            bins.append(list(chunk))
        i += size
    anchors = [[sum(r for r, _ in b) / len(b),      # mean raw
                100.0 * sum(h for _, h in b) / len(b),  # actual %
                len(b)] for b in bins]
    # pool adjacent violators until monotonic in y
    k = 0
    while k < len(anchors) - 1:
        if anchors[k + 1][1] < anchors[k][1]:
            a, b = anchors[k], anchors[k + 1]
            m = a[2] + b[2]
            merged = [(a[0] * a[2] + b[0] * b[2]) / m, (a[1] * a[2] + b[1] * b[2]) / m, m]
            anchors[k:k + 2] = [merged]
            k = max(k - 1, 0)
        else:
            k += 1
    out = [(0.0, 0.0)] + [(round(x, 1), round(y, 1)) for x, y, _ in anchors]
    return [(x, y) for j, (x, y) in enumerate(out) if j == 0 or x > out[j - 1][0]]


def brier(anchors, rows, level=None):
    """Holdout Brier of a candidate curve AS PRODUCTION WOULD SHIP IT.

    `level` defaults to mlb_hr.LIVE_LEVEL because the published number is
    calibrate_pct() THEN live_level_pct() -- anchors alone are a curve nothing
    ever publishes. Scoring without it does not merely add a constant to both
    sides: Brier is not scale-invariant, so the un-leveled objective has its
    optimum in a different place, and the difference is exactly this factor.
    Measured on the 2026-08-02 holdout (n=509, actual 16.50%), sweeping a single
    scale k over the current anchors:

        k the un-leveled gate picks   0.810   Brier 0.137432
        k the shipped objective picks 0.920   Brier 0.137432
        ratio 0.880 == LIVE_LEVEL, to three decimals

    and shipping the gate's pick lands the board at 14.41% against 16.50%
    actual, -2.09pp COLD. That is the level correction applied twice: the gate
    rewards a candidate for absorbing it into the anchor shape, and then
    live_level_pct multiplies by 0.88 again on top. Both curves are scored
    through the same factor here, so the gate compares shapes, which is the only
    thing the refit is allowed to change."""
    if not rows:
        return float("inf")
    k = LIVE_LEVEL if level is None else level
    s = 0.0
    for _, raw, hit in rows:
        p = max(0.0, min(1.0, apply_map(anchors, raw) * k / 100.0))
        s += (p - hit) ** 2
    return s / len(rows)


# ---------- apply ----------
def write_anchors(src, anchors):
    """Replace the (possibly multi-line) assignment with a single line; assert-guarded."""
    start, end, _, _ = _anchor_span(src)
    new_line = "CALIB_ANCHORS = " + json.dumps([[round(x, 1), round(y, 1)] for x, y in anchors]) \
               + "  # auto-refit " + dt.date.today().isoformat()
    return src[:start] + new_line + src[end:]


def main(dry):
    src = open(HR_PY).read()
    cur = read_anchors(src)
    rows = load_rows(cur)
    if not rows:
        print("no graded rows found - nothing to do"); return

    last_fit = None
    m = re.search(r"# auto-refit (\d{4}-\d{2}-\d{2})", src)
    if m:
        last_fit = m.group(1)
        # count NEW production rows only — the static backtest file carries dated rows
        # that would otherwise inflate this and let a refit run on unchanged data.
        fresh = sum(1 for d in graded_dates() if d > last_fit)
        if fresh < MIN_NEW_ROWS:
            print(f"only {fresh} graded rows since last refit ({last_fit}) - need {MIN_NEW_ROWS}; keeping current")
            return

    cutoff = (dt.date.fromisoformat(rows[-1][0]) - dt.timedelta(days=HOLDOUT_DAYS)).isoformat()
    train = [r for r in rows if r[0] <= cutoff]
    hold  = [r for r in rows if r[0] > cutoff]
    print(f"rows: {len(rows)} total | train {len(train)} (thru {cutoff}) | holdout {len(hold)}")
    # PROVENANCE, printed every run because it is the honest size of this refit. The
    # backtest replay is a FROZEN file (it ends the day the replay was generated), so
    # month over month the training pool is the same rows plus a thin live slice. On
    # 2026-08-02 that was 25,128 replay vs 183 production -- 99.3% / 0.7%. A "monthly
    # recalibration" on a pool that is 99.3% unchanged cannot move the curve much, and
    # it is not supposed to: the anchors are the SHAPE, fit on the replay, and the
    # replay->production LEVEL gap is carried by mlb_hr.LIVE_LEVEL instead (see
    # brier()). Printing it keeps a near-zero candidate delta from reading as
    # confirmation that the curve is right.
    _gcount = sum(1 for d in graded_dates() if d <= cutoff)
    _bcount = len(train) - _gcount
    print(f"  train provenance: {_bcount} frozen backtest-replay rows "
          f"({100.0*_bcount/max(1,len(train)):.1f}%) | {_gcount} production graded rows "
          f"({100.0*_gcount/max(1,len(train)):.1f}%)")
    if len(hold) < MIN_HOLDOUT:
        print(f"holdout too small (<{MIN_HOLDOUT}) - keeping current anchors"); return

    cand = fit_anchors(train)
    b_old, b_new = brier(cur, hold), brier(cand, hold)
    verdict = "APPLY" if b_new < b_old - MIN_GAIN else "KEEP"
    print(f"holdout Brier (scored x LIVE_LEVEL={LIVE_LEVEL}, i.e. as published)  "
          f"current {b_old:.5f}  vs  candidate {b_new:.5f}  ->  {verdict}")
    print("candidate anchors:", [[round(x,1), round(y,1)] for x, y in cand])

    report = {"ran": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
              "rows": len(rows), "train": len(train), "holdout": len(hold),
              "train_backtest": _bcount, "train_graded": _gcount,
              "live_level": LIVE_LEVEL, "live_level_ok": LIVE_LEVEL_OK,
              "brier_current": round(b_old, 6), "brier_candidate": round(b_new, 6),
              "verdict": verdict, "candidate": [[round(x,1), round(y,1)] for x, y in cand],
              "current": [[round(x,1), round(y,1)] for x, y in cur]}
    os.makedirs(DATA, exist_ok=True)
    json.dump(report, open(REPORT, "w"), indent=1)

    if verdict == "APPLY" and not LIVE_LEVEL_OK:
        print("APPLY withheld: mlb_hr did not import, so LIVE_LEVEL is unknown and the "
              "gate scored an objective that may not be the shipped one. Report written; "
              "anchors untouched.")
    elif verdict == "APPLY" and not dry:
        open(HR_PY, "w").write(write_anchors(src, cand))
        print("mlb_hr.py updated - new anchors live on the next board build")
    elif verdict == "APPLY":
        print("(dry run - mlb_hr.py untouched)")
    else:
        print("current anchors stay - candidate did not beat them out-of-sample")


# ---------- selftest (offline, no files) ----------
def selftest():
    A = [(0.0,0.0),(8.4,8.8),(13.7,12.3),(17.7,16.0),(22.0,20.1),(28.3,22.2),(40.0,30.5)]
    # 1) invert is exact round-trip on the monotonic map
    for p in (3.0, 8.4, 11.0, 19.5, 25.0, 35.0, 45.0):
        assert abs(invert_map(A, apply_map(A, p)) - p) < 1e-9, f"round-trip fail at {p}"
    # 2) fit recovers a known curve: truth = raw*0.8, 40k synthetic rows
    import random; random.seed(7)
    rows = []
    for i in range(40000):
        raw = random.uniform(5, 35)
        hit = 1 if random.random() < (raw * 0.8) / 100.0 else 0
        rows.append((f"2026-{4 + i % 3:02d}-{1 + i % 28:02d}", raw, hit))
    fitted = fit_anchors(rows)
    for x, y in fitted[1:]:
        assert abs(y - 0.8 * x) < 2.0, f"fit off truth at ({x},{y})"
    ys = [y for _, y in fitted]
    assert all(b >= a for a, b in zip(ys, ys[1:])), "fit not monotonic"
    # 3) gate: better curve wins, worse curve is kept out (level held at 1.0 so this
    #    checks SHAPE selection only -- these synthetic rows have no replay/live gap)
    truth = fitted
    hold = rows[-3000:]
    too_high = [(x, min(100, y * 1.5)) for x, y in truth]
    assert brier(truth, hold, 1.0) < brier(too_high, hold, 1.0) - MIN_GAIN, "gate should prefer truth"
    assert not (brier(too_high, hold, 1.0) < brier(truth, hold, 1.0) - MIN_GAIN), \
        "worse curve must not pass"
    # 3b) THE LEVEL GUARD. Brier is not scale-invariant, so leaving LIVE_LEVEL out of
    #     the gate does not just shift both sides -- it moves the optimum by exactly
    #     that factor, and the winner then gets multiplied by it a second time in
    #     live_level_pct. Fixture: rows generated at truth 0.8*raw, scored under a
    #     level of 0.8. The best single scale k must come out at 1/0.8 = 1.25 (so that
    #     k*0.8 == truth), NOT at 1.0. All 40k rows, not the 3k holdout slice: on 3k the
    #     closed-form optimum is 1.174 purely from sampling noise, which is a fixture
    #     problem, not a code one.
    _lvl = 0.8
    _base = [(v, 0.8 * v) for v in (0.0, 10.0, 20.0, 30.0, 40.0)]
    _ks = [round(0.80 + 0.05 * i, 2) for i in range(13)]          # 0.80 .. 1.40
    _best = min(_ks, key=lambda k: brier([(x, y * k) for x, y in _base], rows, _lvl))
    assert abs(_best - 1.25) <= 0.06, \
        f"level-aware gate should pick k~1.25 under level {_lvl}, picked {_best}"
    _bestnolvl = min(_ks, key=lambda k: brier([(x, y * k) for x, y in _base], rows, 1.0))
    assert abs(_bestnolvl - 1.00) <= 0.06, \
        f"same curves ignoring the level pick k~1.00, picked {_bestnolvl}"
    assert abs(_bestnolvl / _best - _lvl) < 0.06, \
        "the two optima must differ by the level factor -- that is the whole defect"
    assert brier(_base, hold) == brier(_base, hold, LIVE_LEVEL), \
        "brier() default level must be LIVE_LEVEL, not 1.0"
    # 4) writer: single-line replace, refuses ambiguity
    # multi-line, tuple-style, trailing comment — the LIVE file's real shape
    fake = ("X = 1\nCALIB_ANCHORS = [(0.0, 0.0), (8.4, 8.8),\n"
            "                 (40.0, 30.5)]   # last extrapolates\ndef f(): pass\n")
    got = read_anchors(fake)
    assert got == [(0.0,0.0),(8.4,8.8),(40.0,30.5)], f"multi-line read wrong: {got}"
    out = write_anchors(fake, [(0.0,0.0),(20.0,18.0)])
    assert "auto-refit" in out and out.count("CALIB_ANCHORS = [") == 1 and "def f(): pass" in out
    assert read_anchors(out) == [(0.0,0.0),(20.0,18.0)], "write->read round-trip failed"
    try:
        write_anchors(fake + fake, [(0.0,0.0)]); raise SystemExit("writer accepted a duplicate line")
    except AssertionError:
        pass
    # 5) leak-free raw recovery: a stored hr_raw is used verbatim (NOT re-inverted through
    #    the current anchors), so an anchor refit can't corrupt an old row's raw. A legacy
    #    row without hr_raw falls back to inversion through the given anchors.
    assert row_raw({"hr_raw": "12.5", "hr_pct": "9.9"}, A) == 12.5, "stored hr_raw must win"
    legacy_cal = apply_map(A, 20.0)                       # a row calibrated under A, no hr_raw
    assert abs(row_raw({"hr_pct": legacy_cal}, A) - 20.0) < 1e-9, "legacy fallback must invert"
    assert row_raw({"hr_raw": "", "hr_pct": legacy_cal}, A) == row_raw({"hr_pct": legacy_cal}, A), \
        "empty hr_raw must fall back to inversion"
    print("selftest OK: inversion exact | fit recovers truth & monotonic | gate blocks "
          "worse curves | gate scores through LIVE_LEVEL | writer assert-guarded | "
          "hr_raw leak-free")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main("--dry-run" in sys.argv)
