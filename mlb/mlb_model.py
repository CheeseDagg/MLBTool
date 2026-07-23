"""
mlb_model.py  —  working model (v2): team run ratings + starting pitcher -> win prob
====================================================================================
Data source: game_starters.csv from mlb_pitchers.py (has every game's two starters,
score, and all 30 teams -- cleaner than the team-scrape).

Approach (validated by walk-forward backtest on real 2026 games):
  * team OFFENSE / DEFENSE = runs scored / allowed per game vs league, sample-shrunk
    (simple rates -- the opponent-adjusted iterative version OVERFIT early-season luck,
     the same lesson as the World Cup xG rejection, so we don't use it)
  * STARTER quality = runs his team allowed in his prior starts vs league, shrunk
    (crude proxy from scores alone; real pitcher xERA from Statcast would sharpen it)
  * expected runs -> Poisson game grid -> P(home wins).  NOTE the win-prob uses the
    LOWER triangle (home runs > away runs); an earlier upper-triangle bug inverted every
    prediction -- fixed here.

Backtest result (walk-forward, leak-free): team-only ~54%, +starter ~55.4%, both beating
the ~51.7% home-team baseline. It's a legitimate weak READ -- it does NOT beat the market.

RUN:  python mlb_model.py                    # backtest + today's slate
      python mlb_model.py game_starters.csv  # explicit data file
"""
import sys, os, glob, math
import numpy as np, pandas as pd

# tuned on the backtest
K_TEAM, K_SP, W_SP, HFA, MIN_G = 60.0, 8.0, 0.55, 1.03, 12


def load(path):
    s = pd.read_csv(path)
    s = s[s["status"].astype(str).eq("Final")].copy()
    for c in ("home_score", "away_score"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["home_score", "away_score", "home_SP", "away_SP"])
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.dropna(subset=["date"])
    # regular season only (drop spring training)
    s = s[s["date"] >= pd.Timestamp(f"{s['date'].dt.year.mode()[0]}-03-25")]
    s = s.sort_values("date").reset_index(drop=True)
    s["home_win"] = (s["home_score"] > s["away_score"]).astype(int)
    return s


def build(prior):
    """team offense/defense + starter ratings from `prior` games only (leak-free)."""
    L = (prior["home_score"].sum() + prior["away_score"].sum()) / (2 * len(prior))
    RS, RA, N = {}, {}, {}
    for _, p in prior.iterrows():
        for tm, rs, ra in [(p["home"], p["home_score"], p["away_score"]),
                           (p["away"], p["away_score"], p["home_score"])]:
            RS[tm] = RS.get(tm, 0) + rs; RA[tm] = RA.get(tm, 0) + ra; N[tm] = N.get(tm, 0) + 1
    return {"L": L, "RS": RS, "RA": RA, "N": N, "prior": prior}


def _O(m, t):
    N = m["N"]; n = N.get(t, 0)
    if n == 0: return 1.0
    return 1 + ((m["RS"][t] / n / m["L"]) - 1) * n / (n + K_TEAM)

def _D(m, t):
    N = m["N"]; n = N.get(t, 0)
    if n == 0: return 1.0
    return 1 + ((m["RA"][t] / n / m["L"]) - 1) * n / (n + K_TEAM)

def _SP(m, pitcher):
    pri = m["prior"]
    st = pri[(pri["home_SP"] == pitcher) | (pri["away_SP"] == pitcher)]
    v = [p["away_score"] if p["home_SP"] == pitcher else p["home_score"] for _, p in st.iterrows()]
    if not v: return 1.0
    n = len(v); return 1 + ((np.mean(v) / m["L"]) - 1) * n / (n + K_SP)


def _pois(l, K=22):
    k = np.arange(K); return np.exp(-l) * l**k / np.array([math.factorial(int(i)) for i in k])

def predict(m, home, away, home_sp=None, away_sp=None):
    L = m["L"]
    # run prevention each offense faces = blend(opposing starter, opposing team defense)
    if away_sp is not None:
        prev_h = W_SP * _SP(m, away_sp) + (1 - W_SP) * _D(m, away)
        prev_a = W_SP * _SP(m, home_sp) + (1 - W_SP) * _D(m, home)
    else:
        prev_h, prev_a = _D(m, away), _D(m, home)
    lam_h = L * _O(m, home) * prev_h * HFA
    lam_a = L * _O(m, away) * prev_a / HFA
    g = np.outer(_pois(lam_h), _pois(lam_a))
    p_home = np.tril(g, -1).sum() + np.trace(g) * (lam_h**1.83 / (lam_h**1.83 + lam_a**1.83))
    return {"p_home": p_home, "lam_h": lam_h, "lam_a": lam_a, "total": lam_h + lam_a}


def backtest(s, use_sp, burn=0.40):
    d0, d1 = s["date"].min(), s["date"].max(); be = d0 + (d1 - d0) * burn
    P, Y = [], []
    for i in range(len(s)):
        r = s.iloc[i]
        if r["date"] < be: continue
        pri = s[s["date"] < r["date"]]
        if len(pri) < 300: continue
        m = build(pri)
        if m["N"].get(r["home"], 0) < MIN_G or m["N"].get(r["away"], 0) < MIN_G: continue
        pr = predict(m, r["home"], r["away"],
                     r["home_SP"] if use_sp else None, r["away_SP"] if use_sp else None)
        P.append(pr["p_home"]); Y.append(r["home_win"])
    P, Y = np.array(P), np.array(Y)
    cal = []
    for lo in np.arange(0.35, 0.70, 0.10):
        mk = (P >= lo) & (P < lo + 0.10)
        if mk.sum() > 15: cal.append((lo + 0.05, P[mk].mean(), Y[mk].mean(), int(mk.sum())))
    return {"n": len(P), "corr": np.corrcoef(P, Y)[0, 1], "acc": ((P > .5) == Y).mean(),
            "brier": np.mean((P - Y)**2), "base": Y.mean(), "cal": cal}



def load_xwoba(path="data/pitcher_xstats.csv"):
    """Season xwOBA per pitcher -> run-prevention multiplier (PA-shrunk). For LIVE
    predictions only (clean for future games); NOT used in the leak-free backtest."""
    import os
    if not os.path.exists(path): return {}
    x = pd.read_csv(path)
    ncs = [c for c in x.columns if "last_name" in c]
    # Empty/header-only pull (Savant returned 0 rows) or a schema-drifted export must
    # degrade to "no xwOBA layer" (predict_live falls back to team defense), NOT crash
    # the whole publish run on a ZeroDivision/IndexError.
    if x.empty or not ncs or "est_woba" not in x.columns or "pa" not in x.columns:
        return {}
    nc = ncs[0]
    def flip(v):
        v=str(v); return (v.split(",",1)[1].strip()+" "+v.split(",",1)[0].strip()) if "," in v else v.strip()
    x["full"]=x[nc].map(flip)
    denom = x["pa"].sum()
    if denom <= 0: return {}
    lg=(x["est_woba"]*x["pa"]).sum()/denom; Kpa=100
    return {r["full"]: 1+((r["est_woba"]/lg)-1)*(r["pa"]/(r["pa"]+Kpa)) for _,r in x.iterrows()}

def predict_live(m, home, away, home_sp, away_sp, xw):
    """Sharpened prediction: starter run-prevention from real xwOBA (falls back to
    the team defense if a pitcher is unmatched)."""
    L=m["L"]
    prev_h = W_SP*xw.get(away_sp, _D(m,away)) + (1-W_SP)*_D(m,away)
    prev_a = W_SP*xw.get(home_sp, _D(m,home)) + (1-W_SP)*_D(m,home)
    lam_h=L*_O(m,home)*prev_h*HFA; lam_a=L*_O(m,away)*prev_a/HFA
    g=np.outer(_pois(lam_h),_pois(lam_a))
    p=np.tril(g,-1).sum()+np.trace(g)*(lam_h**1.83/(lam_h**1.83+lam_a**1.83))
    return {"p_home":p,"lam_h":lam_h,"lam_a":lam_a,"total":lam_h+lam_a}

def predict_schedule(m, xw, sched_path):
    import os, glob
    if not os.path.exists(sched_path):
        hits=sorted(glob.glob(os.path.join("data","schedule_*.csv")))
        if not hits: return
        sched_path=hits[-1]
    sc=pd.read_csv(sched_path)
    print(f"\nTODAY'S SLATE  (xwOBA-sharpened)  [{os.path.basename(sched_path)}]")
    print(f"  {'away':22s} @ {'home':22s}  home win%%  proj total")
    for _,r in sc.iterrows():
        h,a=r.get("home"),r.get("away")
        if h not in m["N"] or a not in m["N"]: continue
        pr=predict_live(m,h,a,r.get("home_prob_pitcher"),r.get("away_prob_pitcher"),xw)
        print(f"  {str(a)[:22]:22s} @ {str(h)[:22]:22s}  {pr['p_home']*100:5.1f}%    {pr['total']:.1f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        sorted(glob.glob(os.path.join("data", "game_starters.csv")) + ["game_starters.csv"])[0])
    if not os.path.exists(path):
        sys.exit(f"need game_starters.csv (run mlb_pitchers.py first); looked at {path}")
    s = load(path)
    print(f"{len(s)} regular-season games, {s['date'].min().date()} -> {s['date'].max().date()}, "
          f"home win rate {s['home_win'].mean():.3f}")

    print("\nWALK-FORWARD BACKTEST (leak-free):")
    for lab, sp in [("team only", False), ("+ starter", True)]:
        b = backtest(s, sp)
        print(f"  {lab:11s} n={b['n']}  acc {b['acc']*100:.1f}%  corr {b['corr']:+.3f}  "
              f"brier {b['brier']:.4f}  (home baseline {b['base']*100:.1f}%)")
    b = backtest(s, True)
    print("  calibration (+starter):")
    for c, pp, aa, n in b["cal"]:
        print(f"     ~{c*100:.0f}%: predicted {pp*100:.1f}%  actual {aa*100:.1f}%  (n={n})")

    # today's ratings + a couple example matchups
    m = build(s)
    print("\ncurrent team net rating (offense/defense, >1 = good):")
    net = sorted(m["N"], key=lambda t: _O(m, t) / _D(m, t), reverse=True)
    for t in net[:5] + ["..."] + net[-5:]:
        if t == "...": print("   ..."); continue
        print(f"   {t:24s} O {_O(m,t):.3f}  D {_D(m,t):.3f}  net {_O(m,t)/_D(m,t):.3f}")
    xw=load_xwoba()
    print(f"\nxwOBA loaded for {len(xw)} pitchers" if xw else "\n(no pitcher_xstats.csv -> live preds fall back to team defense)")
    predict_schedule(m, xw, "data/schedule.csv")


if __name__ == "__main__":
    main()
