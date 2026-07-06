"""
mlb_parlay.py  —  +EV parlay combinations from line-shop edges
==============================================================
Builds 2- and 3-leg parlays ONLY from legs that are individually +EV vs the
multi-book no-vig consensus (the validated edge). Cross-game legs only
(same-game outcomes are correlated and books price SGPs separately).

Joint edge: fair_prob_parlay * best_decimal_parlay - 1, where each leg uses its
own best book. NOTE: legs at DIFFERENT books can't be one ticket -- the tab
labels those "split across books" (you'd bet them as singles or accept one
book's slightly worse combined price; the singles are usually better).

Also computes BOOST-TO-FLIP for near-miss parlays: the profit-boost %% that
would turn a -EV combo +EV -- so when a book hands you a 30/50%% token you know
instantly where it's worth spending.
"""
import itertools
import numpy as np
import pandas as pd

def american_to_decimal(a): return 1 + a/100.0 if a > 0 else 1 + 100.0/(-a)

def build_parlays(edges_df, max_legs=3, top=10):
    """edges_df: output of mlb_edge.find_edges (one row per +EV side)."""
    if edges_df is None or len(edges_df) < 2:
        return [], []
    legs = edges_df.to_dict("records")
    combos, near = [], []
    for r in (2, 3):
        if len(legs) < r or r > max_legs: continue
        for combo in itertools.combinations(legs, r):
            games = {c["game"] for c in combo}
            if len(games) < r:            # cross-game only
                continue
            fair = float(np.prod([c["fair"] for c in combo]))
            dec  = float(np.prod([american_to_decimal(c["price"]) for c in combo]))
            ev   = fair * dec - 1
            books = {c["book"] for c in combo}
            row = {"legs": [{"bet": c["bet"], "price": int(c["price"]), "book": c["book"]}
                            for c in combo],
                   "n_legs": r, "fair_pct": round(fair*100, 2),
                   "dec": round(dec, 3),
                   "american": int(round((dec-1)*100)) if dec >= 2 else -int(round(100/(dec-1))),
                   "ev_pct": round(ev*100, 2),
                   "one_book": len(books) == 1,
                   "books": sorted(books)}
            if ev > 0:
                combos.append(row)
            elif ev > -0.15:
                # boost-to-flip: profit boost b makes payout profit*(1+b); EV'=fair*(1+(dec-1)*(1+b)))-1
                # solve fair*(1+(dec-1)*(1+b)) = 1  ->  b = (1/fair - 1)/(dec-1) - 1
                b = (1/fair - 1)/(dec - 1) - 1
                row["boost_to_flip_pct"] = round(b*100, 1)
                near.append(row)
    combos.sort(key=lambda x: -x["ev_pct"])
    near.sort(key=lambda x: x["boost_to_flip_pct"])
    return combos[:top], near[:5]

if __name__ == "__main__":
    # synthetic test: 3 edges across 3 games, 2 at the same book
    test = pd.DataFrame([
        {"game":"A @ B","bet":"B","price":110,"book":"BetRivers","fair":0.52,"ev":0.092,"stake":10},
        {"game":"C @ D","bet":"C","price":150,"book":"BetRivers","fair":0.42,"ev":0.05,"stake":6},
        {"game":"E @ F","bet":"F","price":-105,"book":"DraftKings","fair":0.53,"ev":0.034,"stake":5},
    ])
    combos, near = build_parlays(test)
    print("+EV parlays found:", len(combos))
    for c in combos:
        legs=" + ".join(f'{l["bet"]} {l["price"]:+d}' for l in c["legs"])
        print(f'  {c["n_legs"]}-leg  {legs}  -> {c["american"]:+d}  fair {c["fair_pct"]}%  EV {c["ev_pct"]:+}%  '
              f'{"ONE BOOK ("+c["books"][0]+")" if c["one_book"] else "split across books"}')
    print("near-miss (boost-to-flip):", [(n["boost_to_flip_pct"]) for n in near])
