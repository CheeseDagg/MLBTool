"""
mlb_edge.py  —  the market layer: line-shop edge-finder + Kelly
===============================================================
Reads mlb_odds.csv (multi-book moneylines from mlb_odds.py) and finds the ONE edge
that actually wins: getting a better price than the market's fair line.  For each game
it de-vigs every book, takes the consensus no-vig fair price, finds the best available
number across books, and flags sides where your best price beats fair -- sized with
fractional Kelly.  Same discipline as the UFC tool: this sizes the LINE-SHOP edge, NOT
the model's win% (which, per the backtest, does not beat the market).

RUN:  python mlb_edge.py [bankroll]     (default $1000, quarter Kelly)
"""
import sys, os, glob
import pandas as pd, numpy as np

FRAC = 0.25  # quarter Kelly

def am_dec(a): return 1 + a/100.0 if a > 0 else 1 + 100.0/(-a)
def am_str(a): return ("+%d" % int(a)) if a > 0 else "%d" % int(a)
def prob_am(p): return -round(100*p/(1-p)) if p >= 0.5 else round(100*(1-p)/p)

def find_edges(odds_path, bankroll=1000.0, frac=FRAC, min_books=3):
    import datetime as dt
    o = pd.read_csv(odds_path)
    o["home_ml"] = pd.to_numeric(o["home_ml"], errors="coerce")
    o["away_ml"] = pd.to_numeric(o["away_ml"], errors="coerce")
    o = o.dropna(subset=["home_ml", "away_ml"])
    # SAFETY: only price games that HAVEN'T STARTED (live/in-progress lines are not bettable value)
    now = dt.datetime.now(dt.timezone.utc)
    o["commence_dt"] = pd.to_datetime(o["commence"], errors="coerce", utc=True)
    n_all = o["game_id"].nunique()
    o = o[o["commence_dt"] > now]
    n_pre = o["game_id"].nunique()
    find_edges._skipped = n_all - n_pre
    rows = []
    for gid, g in o.groupby("game_id"):
        if len(g) < min_books:        # need enough books for a trustworthy consensus
            continue
        home, away = g["home"].iloc[0], g["away"].iloc[0]
        # per-book no-vig home prob -> consensus fair (median = robust to one soft book)
        nv = [(1/am_dec(r.home_ml)) / ((1/am_dec(r.home_ml)) + (1/am_dec(r.away_ml)))
              for r in g.itertuples()]
        fair_home = float(np.median(nv)); fair_away = 1 - fair_home
        # best available price per side + which book
        bh = g.loc[g["home_ml"].idxmax()]; ba = g.loc[g["away_ml"].idxmax()]
        for team, fair, am, book in [(home, fair_home, bh.home_ml, bh.book),
                                     (away, fair_away, ba.away_ml, ba.book)]:
            D = am_dec(am); ev = fair * D - 1
            if ev > 0:                                # your price beats fair -> +EV
                f = ev / (D - 1); stake = bankroll * min(f * frac, 1.0)
                rows.append({"game": f"{away} @ {home}", "bet": team, "price": am,
                             "book": book, "n_books": len(g), "fair": fair,
                             "ev": ev, "stake": stake})
    df = pd.DataFrame(rows)
    return df.sort_values("stake", ascending=False) if len(df) else df

def main():
    bankroll = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace(".","").isdigit() else 1000.0
    path = "data/mlb_odds.csv"
    if not os.path.exists(path): sys.exit("run mlb_odds.py first -> data/mlb_odds.csv")
    df = find_edges(path, bankroll)
    sk = getattr(find_edges, "_skipped", 0)
    print(f"LINE-SHOP EDGES  (bankroll ${bankroll:.0f}, quarter Kelly)")
    if sk: print(f"  [skipped {sk} game(s) already in progress -- live lines aren't bettable value]")
    print("  sizes best price vs consensus fair; model win%% intentionally NOT used\n")
    if not len(df):
        print("  no side is beating the consensus fair line right now -- nothing to bet.")
        return
    print(f"  {'bet':22s} {'price':>7} {'book':14s} {'fair':>6} {'edge':>7}  stake")
    tot = 0
    for r in df.itertuples():
        print(f"  {r.bet[:22]:22s} {am_str(r.price):>7} {str(r.book)[:14]:14s} "
              f"{am_str(prob_am(r.fair)):>6} {r.ev*100:>5.1f}%  ${r.stake:.2f}")
        tot += r.stake
    print(f"\n  {len(df)} +EV sides · total exposure ${tot:.2f} "
          f"({tot/bankroll*100:.1f}% of bankroll)")
    print("  edges are small by design -- line-shopping is ~1-3% long-term, not a jackpot.")

if __name__ == "__main__":
    main()
