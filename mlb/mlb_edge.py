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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_books import BETTABLE, is_bettable

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
    find_edges._no_bettable = 0
    rows = []
    for gid, g in o.groupby("game_id"):
        if len(g) < min_books:        # need enough books for a trustworthy consensus
            continue
        home, away = g["home"].iloc[0], g["away"].iloc[0]
        # per-book no-vig home prob -> consensus fair (median = robust to one soft book).
        # EVERY book in the feed feeds this, including ones you cannot bet: consensus is
        # an estimate of the true probability and it gets better the more books it pools.
        nv = [(1/am_dec(r.home_ml)) / ((1/am_dec(r.home_ml)) + (1/am_dec(r.away_ml)))
              for r in g.itertuples()]
        fair_home = float(np.median(nv)); fair_away = 1 - fair_home
        # BEST PRICE, on the other hand, is restricted to books the bet can be placed at.
        # This used to be idxmax() over the whole feed, which quoted whichever book in the
        # 8-20 book pull happened to be highest -- and then computed EV, a Kelly stake and
        # a board position off that number. A price you cannot take is not a better price;
        # the resulting row is not merely optimistic, it outranks the placeable rows.
        gb = g[g["book"].map(is_bettable)]
        if gb.empty:
            find_edges._no_bettable = getattr(find_edges, "_no_bettable", 0) + 1
            continue
        bh = gb.loc[gb["home_ml"].idxmax()]; ba = gb.loc[gb["away_ml"].idxmax()]
        # best number ANYWHERE, carried as a column so the cost of the restriction is
        # visible on the board rather than silently absorbed. Never bet, never ranked.
        eh = g.loc[g["home_ml"].idxmax()]; ea = g.loc[g["away_ml"].idxmax()]
        for team, fair, am, book, elam, elbook in [
                (home, fair_home, bh.home_ml, bh.book, eh.home_ml, eh.book),
                (away, fair_away, ba.away_ml, ba.book, ea.away_ml, ea.book)]:
            D = am_dec(am); ev = fair * D - 1
            if ev > 0:                                # your price beats fair -> +EV
                f = ev / (D - 1); stake = bankroll * min(f * frac, 1.0)
                rows.append({"game": f"{away} @ {home}", "bet": team, "price": am,
                             "book": book, "n_books": len(g), "fair": fair,
                             "ev": ev, "stake": stake,
                             "elsewhere": elam if elbook != book else None,
                             "elsewhere_book": elbook if elbook != book else None})
    df = pd.DataFrame(rows)
    # RANKED BY HIT PROBABILITY, not by edge. The EV/Kelly numbers are still computed
    # and still shipped as columns — they are just no longer the ordering key, because
    # ordering by stake puts the longest shots on top (a +900 dog with a 2% price gap
    # outranks a -180 favourite with a 1% one) and the top of the board is what gets
    # bet. `fair` is the no-vig consensus chance this side actually wins; publish
    # re-ranks with the model's own win% where it has one.
    return df.sort_values(["fair", "ev"], ascending=False) if len(df) else df

def main():
    bankroll = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace(".","").isdigit() else 1000.0
    path = "data/mlb_odds.csv"
    if not os.path.exists(path): sys.exit("run mlb_odds.py first -> data/mlb_odds.csv")
    df = find_edges(path, bankroll)
    sk = getattr(find_edges, "_skipped", 0)
    nb = getattr(find_edges, "_no_bettable", 0)
    print(f"LINE-SHOP EDGES  (bankroll ${bankroll:.0f}, quarter Kelly)")
    if sk: print(f"  [skipped {sk} game(s) already in progress -- live lines aren't bettable value]")
    if nb: print(f"  [skipped {nb} game(s) with no {'/'.join(sorted(BETTABLE))} price -- "
                 f"consensus still used every book, but there is nothing here you can bet]")
    print(f"  prices quoted from {'/'.join(sorted(BETTABLE))} ONLY; consensus fair uses every book in the feed")
    print("  sizes best price vs consensus fair; model win%% intentionally NOT used")
    print("  ordered MOST LIKELY TO HIT first (consensus fair %) -- edge sizes the bet, it does not rank it\n")
    if not len(df):
        print("  no side is beating the consensus fair line right now -- nothing to bet.")
        return
    print(f"  {'bet':22s} {'price':>7} {'book':14s} {'fair':>6} {'edge':>7}  stake")
    tot = 0
    for r in df.itertuples():
        el = ""
        if getattr(r, "elsewhere", None) is not None:
            el = f"   (best elsewhere {am_str(r.elsewhere)} {r.elsewhere_book} -- not bettable)"
        print(f"  {r.bet[:22]:22s} {am_str(r.price):>7} {str(r.book)[:14]:14s} "
              f"{am_str(prob_am(r.fair)):>6} {r.ev*100:>5.1f}%  ${r.stake:.2f}{el}")
        tot += r.stake
    print(f"\n  {len(df)} +EV sides · total exposure ${tot:.2f} "
          f"({tot/bankroll*100:.1f}% of bankroll)")
    print("  edges are small by design -- line-shopping is ~1-3% long-term, not a jackpot.")

def selftest():
    """The bettable-price guard, on a fixture that reproduces the bug it closes."""
    import tempfile, datetime as dt
    fut = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=6)).isoformat()
    p = os.path.join(tempfile.mkdtemp(), "o.csv")
    # g1: DraftKings is +310 on a side the consensus makes ~26.6%, so the OLD code
    #     published "HOU +310 DraftKings, edge +8.9%, $X stake" -- a row that cannot be
    #     bet. FanDuel's real +250 is -7.1% EV, i.e. not a play at all. The whole row
    #     is an artifact of quoting a book with no account.
    # g2: no FanDuel line at all. Not a play, and not a row: an unbettable market has
    #     no price, it does not fall back to somebody else's.
    open(p, "w").write("game_id,commence,home,away,book,home_ml,away_ml\n"
                       f"g1,{fut},HOU,SEA,DraftKings,310,-380\n"
                       f"g1,{fut},HOU,SEA,FanDuel,250,-330\n"
                       f"g1,{fut},HOU,SEA,BetMGM,255,-340\n"
                       f"g1,{fut},HOU,SEA,Caesars,260,-345\n"
                       f"g2,{fut},NYY,BOS,DraftKings,-150,130\n"
                       f"g2,{fut},NYY,BOS,BetMGM,-152,132\n"
                       f"g2,{fut},NYY,BOS,Caesars,-148,128\n"
    # g3: FanDuel IS the generous book here (+160 against a field at ~+120), so this is
    #     a genuine play and must survive the filter.
                       f"g3,{fut},CHC,STL,FanDuel,160,-190\n"
                       f"g3,{fut},CHC,STL,DraftKings,120,-140\n"
                       f"g3,{fut},CHC,STL,BetMGM,122,-142\n"
                       f"g3,{fut},CHC,STL,Caesars,118,-138\n")
    df = find_edges(p, 1000.0)
    assert find_edges._no_bettable == 1, "the FanDuel-less game must be counted, not silently dropped"
    assert set(df["book"]) == {"FanDuel"}, f"quoted a book you cannot bet: {set(df['book'])}"
    assert set(df["bet"]) == {"CHC"}, f"g1/g2 should produce no bettable play, got {df.to_dict('records')}"
    r = df.iloc[0]
    assert r["elsewhere"] is None, "FanDuel IS the top price on g3, so there is nothing to flag"

    # CONSENSUS must still pool every book. Same FanDuel price, feed thinned to the two
    # books you could bet-or-see -- if the fair number does not move, the unbettable
    # books were being excluded from consensus too, which is the over-correction.
    import pandas as _pd
    o = _pd.read_csv(p)
    p2 = os.path.join(tempfile.mkdtemp(), "o2.csv")
    o[o.book.isin(["FanDuel", "DraftKings"])].to_csv(p2, index=False)
    thin = find_edges(p2, 1000.0, min_books=2)
    t = thin[thin.bet == "CHC"].iloc[0]
    assert t["price"] == r["price"], "the quoted FanDuel price must not depend on the field"
    assert abs(t["fair"] - r["fair"]) > 0.02, \
        f"fair must move when books are removed (all-books {r['fair']:.4f} vs " \
        f"two-books {t['fair']:.4f}) -- consensus has to pool the whole feed"
    print(f"mlb_edge selftest OK — quotes restricted to {'/'.join(sorted(BETTABLE))} "
          f"(g1's +310 DraftKings phantom edge gone, g3's real FanDuel +160 kept), "
          f"consensus still pools every book (fair {r['fair']:.3f} vs {t['fair']:.3f})")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
