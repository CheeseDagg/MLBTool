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
    raw and biases the next fit — which is exactly what hr_raw removes.)"""
    hr_raw = r.get("hr_raw")
    if hr_raw is not None and str(hr_raw).strip() != "":
        return float(hr_raw)
    return invert_map(anchors, float(r["hr_pct"]))

def graded_dates():
    """Dates of PRODUCTION graded rows only (excludes the static backtest replay), for
    measuring how much genuinely-new data has arrived since the last refit."""
    out = []
    gr = os.path.join(DATA, "hr_graded.csv")
    if os.path.exists(gr):
        for r in csv.DictReader(open(gr)):
            if r.get("outcome") in ("hr", "no") and r.get("date"):
                out.append(r["date"])
    return out

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
    gr = os.path.join(DATA, "hr_graded.csv")
    if os.path.exists(gr):
        for r in csv.DictReader(open(gr)):
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


def brier(anchors, rows):
    if not rows:
        return float("inf")
    s = 0.0
    for _, raw, hit in rows:
        p = max(0.0, min(1.0, apply_map(anchors, raw) / 100.0))
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
    if len(hold) < MIN_HOLDOUT:
        print(f"holdout too small (<{MIN_HOLDOUT}) - keeping current anchors"); return

    cand = fit_anchors(train)
    b_old, b_new = brier(cur, hold), brier(cand, hold)
    verdict = "APPLY" if b_new < b_old - MIN_GAIN else "KEEP"
    print(f"holdout Brier  current {b_old:.5f}  vs  candidate {b_new:.5f}  ->  {verdict}")
    print("candidate anchors:", [[round(x,1), round(y,1)] for x, y in cand])

    report = {"ran": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
              "rows": len(rows), "train": len(train), "holdout": len(hold),
              "brier_current": round(b_old, 6), "brier_candidate": round(b_new, 6),
              "verdict": verdict, "candidate": [[round(x,1), round(y,1)] for x, y in cand],
              "current": [[round(x,1), round(y,1)] for x, y in cur]}
    os.makedirs(DATA, exist_ok=True)
    json.dump(report, open(REPORT, "w"), indent=1)

    if verdict == "APPLY" and not dry:
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
    # 3) gate: better curve wins, worse curve is kept out
    truth = fitted
    hold = rows[-3000:]
    too_high = [(x, min(100, y * 1.5)) for x, y in truth]
    assert brier(truth, hold) < brier(too_high, hold) - MIN_GAIN, "gate should prefer truth"
    assert not (brier(too_high, hold) < brier(truth, hold) - MIN_GAIN), "worse curve must not pass"
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
    print("selftest OK: inversion exact | fit recovers truth & monotonic | gate blocks worse curves | writer assert-guarded | hr_raw leak-free")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main("--dry-run" in sys.argv)
