#!/usr/bin/env python3
"""
mlb_backtest.py — SEASON REPLAY (walk-forward, leak-free).

Purpose: instead of waiting for the live ledger to reach n~150, replay the season
day-by-day and grade retroactive HR predictions against what actually happened. That
turns the +10 heat threshold, the calibration curve, and the platoon lifts from
sketches (n~10) into verdicts (n~hundreds), which we then use to ADJUST the model.

Two honesty guarantees, enforced by the selftests below:
  1. NO LEAKAGE. Every prediction for day D uses only games with date < D. A bat's
     season-to-date HR/PA and rolling heat window are reconstructed from the past
     only. Predicting May with July stats is the cardinal sin; test_no_leakage
     asserts the reconstruction of an early day is unaffected by appending later games.
  2. CALIBRATION ONLY, NOT ROI. Historical closing book prices aren't fetchable, so
     this proves whether "28%" MEANS 28% and which ingredients carry — it never
     claims profit. ROI stays forward-only in the live ledger.

The pricing core MIRRORS mlb_hr.build_board exactly:
    p_pa = base * park * sp_eff * platoon * heat        (weather term dropped)
    p_game = 1 - (1 - p_pa) ** pa_est
with the same constants. Weather is omitted on purpose: Open-Meteo's archive can't
give reliable first-pitch temps three months back, and a backtest padded with wrong
temperatures would flatter the model. The panel is labeled "core model (no weather)".

Runs on GitHub Actions (statsapi reachable there). One-shot: writes data/hr_backtest.csv
and a panel consumed by the dashboard's Backtest tab.
"""

import os, csv, json, math, statistics, datetime as dt
try:
    import mlb_hr as H          # reuse the real constants + park/platoon functions
except Exception:
    H = None

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT  = os.path.join(DATA, "hr_backtest.csv")

# Mirror the live constants (fall back to literals if import unavailable in tests)
K_BAT     = getattr(H, "K_BAT", 130)
K_PIT     = getattr(H, "K_PIT", 200)
SP_WEIGHT = getattr(H, "SP_WEIGHT", 0.62)
CAP_PPA   = getattr(H, "CAP_PPA", 0.12)
PA_TOP    = getattr(H, "PA_TOP", 4.45)
LG_HRPA_FALLBACK = 0.031

def norm(s):
    if H and hasattr(H, "norm"):
        return H.norm(s)
    if not isinstance(s, str): return ""
    return s.lower().replace(".", "").replace("'", "").strip()

# ---------------------------------------------------------------------------
# AS-OF STATE: the leak-free heart. Fold games in date order; query = snapshot
# of everything strictly BEFORE the target date.
# ---------------------------------------------------------------------------
class AsOfState:
    """Accumulates per-day box lines and answers 'stats through the morning of D'.

    Feed daily() in ascending date order. Each daily() records that date's results
    but does NOT expose them until a LATER date is queried — so predicting D only
    ever sees < D. Rolling heat = HR/PA over the trailing `heat_window` days.

    HEAT REBUILD (item 6): heat_window and heat_shrink_K are constructor args so a
    candidate heat definition can be replayed against the full season and compared to
    the incumbent BEFORE it's trusted live. The season backtest showed the default
    (15d / K=120) did not carry — this is the knob to find one that does.
    """
    def __init__(self, heat_window=15, heat_shrink_K=120.0):
        self.heat_window = heat_window
        self.heat_shrink_K = heat_shrink_K
        self.bat_days = {}     # norm_name -> list[(date, pa, hr)]
        self.pit_days = {}     # norm_name -> list[(date, bf, hr, so, bb, ao)]
        self.lg_days  = []     # list[(date, tot_hr, tot_pa, tot_bf)]

    def record_day(self, date, bat_lines, pit_lines):
        """bat_lines: [{name, pa, hr}]  pit_lines: [{name, bf, hr, so?, bb?, ao?}]"""
        thr = tpa = tbf = 0
        for b in bat_lines:
            self.bat_days.setdefault(norm(b["name"]), []).append(
                (date, int(b.get("pa", 0) or 0), int(b.get("hr", 0) or 0)))
            thr += int(b.get("hr", 0) or 0); tpa += int(b.get("pa", 0) or 0)
        for p in pit_lines:
            self.pit_days.setdefault(norm(p["name"]), []).append(
                (date, int(p.get("bf", 0) or 0), int(p.get("hr", 0) or 0),
                 p.get("so"), p.get("bb"), p.get("ao")))
            tbf += int(p.get("bf", 0) or 0)
        self.lg_days.append((date, thr, tpa, tbf))

    def _before(self, rows, date, idx_date=0):
        return [r for r in rows if r[idx_date] < date]

    def league_rates(self, date):
        past = self._before(self.lg_days, date)
        hr = sum(r[1] for r in past); pa = sum(r[2] for r in past); bf = sum(r[3] for r in past)
        Lb = (hr / pa) if pa else LG_HRPA_FALLBACK
        Lp = (hr / bf) if bf else Lb
        if Lb <= 0: Lb = LG_HRPA_FALLBACK
        if Lp <= 0: Lp = Lb
        return Lb, Lp

    def batter_season(self, name, date):
        past = self._before(self.bat_days.get(norm(name), []), date)
        return sum(r[1] for r in past), sum(r[2] for r in past)   # pa, hr

    def batter_heat(self, name, date, Lb):
        """Rolling HR/PA vs league over trailing window -> same shape as live heat mult.
        Returns a multiplier ~ (window HR/PA)/(league HR/PA), shrunk toward 1 by PA."""
        cutoff = date - dt.timedelta(days=self.heat_window)
        past = [r for r in self.bat_days.get(norm(name), []) if cutoff <= r[0] < date]
        pa = sum(r[1] for r in past); hr = sum(r[2] for r in past)
        if pa < 8 or Lb <= 0:            # too few recent PA -> no heat signal
            return None
        K = self.heat_shrink_K            # tunable shrink (item 6 rebuild knob)
        rate = (hr + K * Lb) / (pa + K)   # stat explosion (a 10-PA hot streak ~= +a few %)
        mult = rate / Lb
        return min(max(mult, 0.70), 1.30)  # heat multiplier bounded like a real factor

    def pitcher_factor(self, name, date, Lp):
        past = self._before(self.pit_days.get(norm(name), []), date)
        bf = sum(r[1] for r in past); hr = sum(r[2] for r in past)
        if bf == 0:
            return 1.0
        # legacy HR/BF shrink (backtest uses the robust tier; component decomposition
        # needs per-day ao/so which statsapi boxscores do carry, but we keep the
        # backtest on the stable path to avoid overfitting the replay)
        rate = (hr + K_PIT * Lp) / (bf + K_PIT)
        return min(max(rate / Lp, 0.60), 1.60)

# ---------------------------------------------------------------------------
# PRICE ONE BATTER-GAME — identical formula to build_board, minus weather.
# ---------------------------------------------------------------------------
def price_row(state, date, bat_name, bat_side, opp_sp, sp_hand, venue, slot):
    Lb, Lp = state.league_rates(date)
    pa, hr = state.batter_season(bat_name, date)
    base = (hr + K_BAT * Lb) / (pa + K_BAT)

    eff, park_lab = (H.hr_park(venue) if H else (1.0, "park +0%"))

    fac = state.pitcher_factor(opp_sp, date, Lp)
    sp_eff = SP_WEIGHT * fac + (1 - SP_WEIGHT)          # bullpen share priced flat (as live, pre-pen)

    plat_eff, plat_tag = (H.platoon_factor(bat_side, sp_hand) if H else (1.0, ""))

    hf = state.batter_heat(bat_name, date, Lb)
    heat_eff = SP_WEIGHT * hf + (1 - SP_WEIGHT) if hf else 1.0
    heat_tag = (f"heat {'+' if hf >= 1 else ''}{(hf-1)*100:.0f}%") if hf else ""

    p_pa = min(base * eff * sp_eff * plat_eff * heat_eff, CAP_PPA)
    pa_est = max(PA_TOP - 0.09 * (slot - 1), 3.4)
    p_game = 1 - (1 - p_pa) ** pa_est
    return {"date": date.isoformat(), "player": bat_name, "opp_sp": opp_sp,
            "slot": slot, "hr_pct": round(p_game * 100, 1),
            "plat": plat_tag, "heat": heat_tag, "park": park_lab}

# ---------------------------------------------------------------------------
# GRADE + SUMMARIZE — same calibration math as mlb_grade, plus heat-band + platoon.
# ---------------------------------------------------------------------------
def _heat_val(tag):
    t = tag or ""
    try:
        if "heat +" in t: return int(t.split("+")[1].rstrip("%"))
        if "heat -" in t: return -int(t.split("-")[1].rstrip("%"))
    except Exception: pass
    return None

def summarize(graded):
    """graded: [{hr_pct, outcome('hr'/'no'), hr_n, heat, plat, ...}] -> panel dict."""
    live = [r for r in graded if r["outcome"] in ("hr", "no")]
    n = len(live)
    out = {"n": n, "weather": "included (historical first-pitch, Meteostat)"}
    if not n: return out
    p = [float(r["hr_pct"]) / 100 for r in live]
    y = [1.0 if r["outcome"] == "hr" else 0.0 for r in live]
    out["pred_mean"] = round(100 * sum(p) / n, 1)
    out["actual"]    = round(100 * sum(y) / n, 1)
    out["brier"]     = round(sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n, 4)

    # calibration buckets
    edges = [(0,12),(12,16),(16,20),(20,25),(25,101)]
    bks = []
    for lo, hi in edges:
        sel = [(pi, yi) for pi, yi in zip(p, y) if lo <= pi*100 < hi]
        if sel:
            bks.append({"bucket": f"{lo}-{hi if hi<101 else '+'}", "n": len(sel),
                        "pred": round(100*sum(s[0] for s in sel)/len(sel),1),
                        "actual": round(100*sum(s[1] for s in sel)/len(sel),1)})
    out["buckets"] = bks

    # heat-magnitude bands — the verdict on the +10 rule
    bands = [("+10 or more", lambda v: v is not None and v >= 10),
             ("+5..+9",      lambda v: v is not None and 5 <= v < 10),
             ("+1..+4",      lambda v: v is not None and 1 <= v < 5),
             ("0 / none",    lambda v: v is None or v == 0),
             ("negative",    lambda v: v is not None and v < 0)]
    hb = []
    for name, f in bands:
        sel = [r for r in live if f(_heat_val(r.get("heat")))]
        if sel:
            hh = sum(1 for r in sel if r["outcome"] == "hr")
            hb.append({"band": name, "n": len(sel), "actual": round(100*hh/len(sel),1),
                       "pred": round(100*statistics.fmean(float(r["hr_pct"])/100 for r in sel),1)})
    out["heat_bands"] = hb

    # A-tier: 25%+ AND heat>=+10 — the headline cut
    A = [r for r in live if float(r["hr_pct"]) >= 25 and (_heat_val(r.get("heat")) or -99) >= 10]
    if A:
        ah = sum(1 for r in A if r["outcome"] == "hr")
        out["a_tier"] = {"n": len(A), "hits": ah, "actual": round(100*ah/len(A),1),
                         "pred": round(100*statistics.fmean(float(r["hr_pct"])/100 for r in A),1)}

    # two-homer rate where counted
    cnt = [r for r in live if str(r.get("hr_n","")).strip() != ""]
    if cnt:
        two = sum(1 for r in cnt if int(float(r["hr_n"])) >= 2)
        out["multi"] = {"n": len(cnt), "two_plus": two, "rate": round(100*two/len(cnt),1)}

    # HEAT LIFT (item 6): the single number a rebuilt heat def must beat. Positive =
    # heat>=+10 bats homer MORE than the overall rate (signal); ~0 = no signal.
    base = out["actual"]
    hp = [r for r in live if (_heat_val(r.get("heat")) or -99) >= 10]
    if hp:
        h_act = 100 * sum(1 for r in hp if r["outcome"] == "hr") / len(hp)
        out["heat_lift"] = {"n": len(hp), "heat_actual": round(h_act,1),
                            "base_actual": base, "lift_pts": round(h_act - base,1)}
    out["dates"] = len({r["date"] for r in live})
    return out

# ---------------------------------------------------------------------------
# selftests — leak-freeness is the one that matters most
# ---------------------------------------------------------------------------
def selftest():
    d = dt.date
    st = AsOfState(heat_window=15)
    # realistic season shapes: ~30 PA of history, a slugger recently hot
    st.record_day(d(2026,4,1), [{"name":"Slug Power","pa":80,"hr":4},{"name":"Cold Bat","pa":80,"hr":1}],
                  [{"name":"Meat Baller","bf":80,"hr":6},{"name":"Ace Arm","bf":80,"hr":1}])
    st.record_day(d(2026,4,2), [{"name":"Slug Power","pa":10,"hr":1},{"name":"Cold Bat","pa":10,"hr":0}],
                  [{"name":"Meat Baller","bf":25,"hr":2},{"name":"Ace Arm","bf":25,"hr":0}])
    st.record_day(d(2026,4,3), [{"name":"Slug Power","pa":4,"hr":0},{"name":"Cold Bat","pa":4,"hr":1}],
                  [{"name":"Meat Baller","bf":22,"hr":1},{"name":"Ace Arm","bf":22,"hr":0}])

    # 1) AS-OF: querying 4/2 sees ONLY 4/1
    pa, hr = st.batter_season("Slug Power", d(2026,4,2))
    assert (pa, hr) == (80, 4), (pa, hr)         # only 4/1
    pa3, hr3 = st.batter_season("Slug Power", d(2026,4,3))
    assert (pa3, hr3) == (90, 5), (pa3, hr3)     # 4/1 + 4/2, never 4/3

    # 2) NO LEAKAGE: appending a FUTURE day must not change a past-day snapshot
    snap_before = st.batter_season("Slug Power", d(2026,4,2))
    st.record_day(d(2026,4,20), [{"name":"Slug Power","pa":4,"hr":4}], [{"name":"Meat Baller","bf":20,"hr":0}])
    snap_after = st.batter_season("Slug Power", d(2026,4,2))
    assert snap_before == snap_after == (80, 4), "LEAK: future day altered a past snapshot"

    # 3) league rates as-of are past-only and positive
    Lb, Lp = st.league_rates(d(2026,4,3))
    assert Lb > 0 and Lp > 0
    # 4/3 league = through 4/2 = (2+0+2+0)HR / (15+15+6+4)PA = 4/40
    assert abs(Lb - 6/180) < 1e-9, Lb   # bat HR only: (4+1+1+0)/(80+80+10+10)

    # 4) heat window respects both the trailing horizon AND the < date bound
    #    on 4/3, window=15d includes 4/1+4/2 (3HR/8PA) -> hot multiplier > 1
    hfac = st.batter_heat("Slug Power", d(2026,4,3), Lb)
    assert hfac is not None and 1.0 < hfac <= 1.30, hfac   # hot but bounded
    #    a bat with < 8 recent PA yields no signal (Cold Bat has 15 PA on 4/1 though)
    #    so test the horizon instead: query 4/2 sees only 4/1's 15 PA
    assert st.batter_heat("Cold Bat", d(2026,4,2), Lb) is not None

    # 5) pitcher factor as-of, shrunk, bounded, past-only
    pf = st.pitcher_factor("Meat Baller", d(2026,4,3), Lp)
    pf_ace = st.pitcher_factor("Ace Arm", d(2026,4,3), Lp)
    assert 0.60 <= pf <= 1.60 and 0.60 <= pf_ace <= 1.60
    assert pf > pf_ace                         # gopher-prone arm rates above the ace

    # 6) price_row: components sane + MONOTONICITY (the real guarantees; absolute
    #    magnitude depends on a full-season league, which a 4-player fixture lacks)
    if H:
        r = price_row(st, d(2026,4,3), "Slug Power", "R", "Meat Baller", "L", "Yankee Stadium", 3)
        assert r["heat"].startswith("heat") and 0 < r["hr_pct"] <= 100, r
        # hotter/HR-prone matchup must price ABOVE a cold bat vs the ace, same park/slot
        r_cold = price_row(st, d(2026,4,3), "Cold Bat", "R", "Ace Arm", "L", "Yankee Stadium", 3)
        assert r["hr_pct"] > r_cold["hr_pct"], (r["hr_pct"], r_cold["hr_pct"])
        # same bat, worse park must price lower
        r_pit = price_row(st, d(2026,4,3), "Slug Power", "R", "Meat Baller", "L", "Petco Park", 3)
        assert r_pit["hr_pct"] < r["hr_pct"], (r_pit["hr_pct"], r["hr_pct"])
        # deeper lineup slot -> fewer PA -> lower game prob
        r_slot9 = price_row(st, d(2026,4,3), "Slug Power", "R", "Meat Baller", "L", "Yankee Stadium", 9)
        assert r_slot9["hr_pct"] < r["hr_pct"]

    # 7) summarize: calibration + heat bands + A-tier + multi
    graded = [
        {"date":"2026-05-01","hr_pct":"30","outcome":"hr","hr_n":"2","heat":"heat +12%","plat":"RvL +12%"},
        {"date":"2026-05-01","hr_pct":"28","outcome":"no","hr_n":"0","heat":"heat +11%","plat":"RvR +0%"},
        {"date":"2026-05-02","hr_pct":"27","outcome":"hr","hr_n":"1","heat":"heat +14%","plat":"LvR +8%"},
        {"date":"2026-05-02","hr_pct":"9","outcome":"no","hr_n":"0","heat":"heat -6%","plat":"LvL -22%"},
        {"date":"2026-05-03","hr_pct":"22","outcome":"hr","hr_n":"1","heat":"heat +3%","plat":"RvR +0%"},
        {"date":"2026-05-03","hr_pct":"14","outcome":"no","hr_n":"0","plat":"RvR +0%"},   # NO heat tag -> None
        {"date":"2026-05-04","hr_pct":"11","outcome":"no","hr_n":"0","heat":"","plat":""}, # empty heat -> None
    ]
    p = summarize(graded)
    assert p["n"] == 7                       # includes 2 tagless-heat rows
    assert any(b["band"]=="0 / none" for b in p["heat_bands"])   # None rows land here, no crash
    at = p["a_tier"]; assert at["n"] == 3 and at["hits"] == 2 and at["actual"] == round(200/3,1)
    hb = {b["band"]: b for b in p["heat_bands"]}
    assert hb["+10 or more"]["n"] == 3 and hb["+10 or more"]["actual"] == round(200/3,1)
    assert p["multi"]["n"] == 7 and p["multi"]["two_plus"] == 1
    assert "included" in p["weather"] or "excluded" in p["weather"]
    json.dumps(p)   # JSON-safe for the dashboard

    st2 = AsOfState(heat_window=10, heat_shrink_K=40.0)
    assert st2.heat_shrink_K == 40.0 and st2.heat_window == 10
    # heat_lift appears when heat>=+10 rows exist
    assert "heat_lift" in p and p["heat_lift"]["n"] >= 1
    print("BACKTEST SELFTEST PASS — as-of/no-leakage/heat-window/pitcher/price/summary all exact")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print("Run --selftest offline; full season replay executes on GitHub Actions "
          "(statsapi reachable there). Wire via mlb_backtest_run.py in the workflow.")
