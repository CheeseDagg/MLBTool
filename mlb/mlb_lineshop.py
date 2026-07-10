#!/usr/bin/env python3
"""
mlb_lineshop.py — the line-shopping / stale-line edge engine.

The model's job is a trustworthy FAIR price (Marcel-calibrated). This engine finds the
GAP between that fair price and what the books are actually offering — which is where
real money lives, because you can't out-model a book but you CAN catch it slow, generous,
or out of step with the field.

Three edges, all pure math over multi-book odds:

  1. LINE SHOPPING — for each player, find the BEST price across books. Betting +310
     instead of +250 on the same bet is riskless edge that compounds. We report the best
     book, the field, and how much the best price beats the consensus.

  2. FAIR-VALUE EDGE — compare the best available price to our Marcel-calibrated fair
     probability. Positive EV = the book is paying more than the bet is worth. This is
     the "is it +EV" gate; only these are bets.

  3. STALE-LINE / OUTLIER — flag when one book is wildly off the others (e.g. hasn't
     repriced after a lineup change). A book offering +450 when the field is +250 is
     either a mistake in your favor or news you haven't seen — either way, surface it.

Plus CONSENSUS DEVIG: the vig-free market probability from combining books, which is the
sharpest estimate of the TRUE probability available — often sharper than any single model.
When our fair and the consensus disagree a lot, that's a flag to TRUST THE MARKET, not the model.

Pure module. Real odds are pulled on Actions by mlb_lineshop_run.py.
"""

def american_to_prob(american):
    """American odds -> implied probability (INCLUDING vig)."""
    a = float(american)
    if a >= 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)

def american_to_decimal(american):
    a = float(american)
    return (a / 100.0 + 1.0) if a >= 0 else (100.0 / (-a) + 1.0)

def prob_to_american(p):
    """Fair probability -> fair American odds (no vig)."""
    if p <= 0: return 100000
    if p >= 1: return -100000
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))

def devig_two_way(over_am, under_am):
    """Remove vig from a two-way market (Over/Under a player prop) -> fair P(over).
    Books price both sides; the two implied probs sum to >1 by the vig. Normalize."""
    po = american_to_prob(over_am)
    pu = american_to_prob(under_am)
    tot = po + pu
    if tot <= 0: return None
    return po / tot

def best_line(book_prices):
    """book_prices: {book: american_odds} for the SAME bet (e.g. player HR over).
    Returns (best_book, best_american, best_decimal) — the highest payout available."""
    best = None
    for bk, am in book_prices.items():
        if am is None: continue
        dec = american_to_decimal(am)
        if best is None or dec > best[2]:
            best = (bk, am, dec)
    return best

def consensus_prob(book_prices_over, book_prices_under=None):
    """Vig-free consensus P(over) across books. If under-prices are supplied per book,
    devig each book then average (sharpest). Else average the raw implied over-probs
    (still useful, but vig-inflated)."""
    probs = []
    for bk, over_am in book_prices_over.items():
        if over_am is None: continue
        if book_prices_under and book_prices_under.get(bk) is not None:
            fp = devig_two_way(over_am, book_prices_under[bk])
            if fp is not None: probs.append(fp)
        else:
            probs.append(american_to_prob(over_am))
    if not probs: return None
    return sum(probs) / len(probs)

def analyze_player(name, fair_prob, book_over, book_under=None, stale_mult=1.6):
    """Full line-shop analysis for one player's HR prop.

    fair_prob : our Marcel-calibrated P(homer) — the trustworthy fair number.
    book_over : {book: american} best-available over prices per book.
    book_under: {book: american} unders (enables per-book devig + consensus).

    Returns a dict with best line, edge vs fair, edge vs consensus, and flags.
    """
    bl = best_line(book_over)
    if not bl:
        return None
    best_book, best_am, best_dec = bl

    # 1) EDGE vs our fair number: EV per $1 = fair_prob * decimal - 1
    ev_fair = fair_prob * best_dec - 1.0

    # 2) CONSENSUS (vig-free true-prob estimate) and edge vs it
    cons = consensus_prob(book_over, book_under)
    ev_cons = (cons * best_dec - 1.0) if cons is not None else None

    # 3) LINE-SHOP VALUE: how much the best price beats the field (in implied-prob pts)
    field_probs = [american_to_prob(a) for a in book_over.values() if a is not None]
    field_avg = sum(field_probs) / len(field_probs) if field_probs else None
    best_prob = american_to_prob(best_am)
    shop_edge_pts = round((field_avg - best_prob) * 100, 1) if field_avg is not None else None

    # 4) STALE / OUTLIER: is the best price an outlier vs the field? (payout >> field)
    stale = False
    if field_avg is not None and best_prob > 0:
        # best implied prob much LOWER than field avg == best payout much higher == outlier
        if field_avg / best_prob >= stale_mult and len(field_probs) >= 2:
            stale = True

    # 5) MODEL-vs-MARKET disagreement: if our fair and consensus diverge a lot, trust market
    model_market_gap = round((fair_prob - cons) * 100, 1) if cons is not None else None

    return {
        "player": name,
        "fair_prob": round(fair_prob, 4),
        "fair_american": prob_to_american(fair_prob),
        "best_book": best_book,
        "best_price": int(best_am),
        "n_books": sum(1 for a in book_over.values() if a is not None),
        "ev_vs_fair_pct": round(ev_fair * 100, 1),
        "consensus_prob": round(cons, 4) if cons is not None else None,
        "ev_vs_consensus_pct": round(ev_cons * 100, 1) if ev_cons is not None else None,
        "shop_edge_pts": shop_edge_pts,        # value of shopping vs taking field-avg
        "stale_flag": stale,
        "model_market_gap_pts": model_market_gap,
    }

def rank_board(analyses, min_ev_fair=0.0, min_books=2):
    """Filter to genuine plays and rank. A PLAY requires:
      - positive EV vs our fair number (the book pays more than fair), AND
      - positive EV vs the vig-free consensus when available (both model AND market agree
        it's +EV — guards against betting only because our model is high), AND
      - at least min_books books (so 'best price' is a real shop, not a lone quote).
    Ranked by the more conservative of the two EVs."""
    plays = []
    for a in analyses:
        if a is None or a["n_books"] < min_books:
            continue
        if a["ev_vs_fair_pct"] <= min_ev_fair * 100:
            continue
        # require consensus agreement when we have it
        if a["ev_vs_consensus_pct"] is not None and a["ev_vs_consensus_pct"] <= 0:
            continue
        # conservative EV = min of the two (if both exist)
        evs = [a["ev_vs_fair_pct"]]
        if a["ev_vs_consensus_pct"] is not None:
            evs.append(a["ev_vs_consensus_pct"])
        a = dict(a); a["conservative_ev_pct"] = round(min(evs), 1)
        plays.append(a)
    plays.sort(key=lambda x: -x["conservative_ev_pct"])
    return plays


# ---------------------------------------------------------------------------
def selftest():
    # --- odds conversions exact ---
    assert abs(american_to_prob(+100) - 0.5) < 1e-9
    assert abs(american_to_prob(-110) - 0.5238) < 1e-4
    assert abs(american_to_decimal(+250) - 3.5) < 1e-9
    assert abs(american_to_decimal(-200) - 1.5) < 1e-9
    assert prob_to_american(0.5) == -100
    assert prob_to_american(0.25) == 300      # 1-in-4 -> +300 fair
    # round trip
    for am in (+250, -130, +115, -400):
        assert abs(american_to_prob(am) - american_to_prob(prob_to_american(american_to_prob(am)))) < 0.02

    # --- devig two-way ---
    # Over +100 / Under +100 -> both imply 50%, sum 100%, no vig -> fair .5
    assert abs(devig_two_way(+100, +100) - 0.5) < 1e-9
    # Over -120 / Under +100: po=.545, pu=.5, tot=1.045 -> fair over .522
    assert abs(devig_two_way(-120, +100) - 0.5217) < 1e-3

    # --- best line picks highest payout ---
    bp = {"dk": +250, "fd": +310, "br": +275}
    bb, bam, bdec = best_line(bp)
    assert bb == "fd" and bam == 310            # +310 is the best payout

    # --- LINE SHOP: taking +310 vs field avg is real edge ---
    a = analyze_player("Judge", fair_prob=0.28, book_over={"dk": +250, "fd": +310, "br": +275})
    assert a["best_book"] == "fd" and a["best_price"] == 310
    # EV vs fair: .28 * 4.10 - 1 = +.148 -> +14.8%
    assert abs(a["ev_vs_fair_pct"] - 14.8) < 0.2
    assert a["shop_edge_pts"] > 0               # best beats field
    assert a["n_books"] == 3

    # --- FAIR-VALUE GATE: a bet where book UNDERpays fair is NOT +EV ---
    a2 = analyze_player("ColdBat", fair_prob=0.10, book_over={"dk": +250, "fd": +260})
    # .10 * 3.6 - 1 = -.64 -> deeply -EV, must be excluded by rank_board
    assert a2["ev_vs_fair_pct"] < 0

    # --- CONSENSUS devig: with unders, consensus is vig-free and sharper ---
    a3 = analyze_player("Star", fair_prob=0.30,
                        book_over={"dk": +240, "fd": +250},
                        book_under={"dk": -300, "fd": -320})
    assert a3["consensus_prob"] is not None
    assert a3["ev_vs_consensus_pct"] is not None
    # model_market gap computed
    assert a3["model_market_gap_pts"] is not None

    # --- STALE/OUTLIER: one book way off the field ---
    a4 = analyze_player("Newsy", fair_prob=0.25,
                        book_over={"dk": +250, "fd": +255, "br": +600})  # br stale/generous
    assert a4["best_book"] == "br" and a4["stale_flag"] is True          # +600 is an outlier

    # a tight field is NOT stale
    a5 = analyze_player("Tight", fair_prob=0.25, book_over={"dk": +250, "fd": +255, "br": +260})
    assert a5["stale_flag"] is False

    # --- RANK: only +EV-both-ways, multi-book plays survive, sorted by conservative EV ---
    board = [
        analyze_player("A", 0.30, {"dk": +260, "fd": +320}),                       # model +EV
        analyze_player("B", 0.10, {"dk": +250, "fd": +260}),                       # -EV, drop
        analyze_player("C", 0.28, {"dk": +250}),                                   # 1 book, drop
        analyze_player("D", 0.32, {"dk": +240, "fd": +250},
                       {"dk": -300, "fd": -320}),                                  # both-way check
    ]
    plays = rank_board(board, min_ev_fair=0.0, min_books=2)
    names = [p["player"] for p in plays]
    assert "B" not in names and "C" not in names        # -EV and single-book excluded
    assert "A" in names
    assert all(plays[i]["conservative_ev_pct"] >= plays[i+1]["conservative_ev_pct"]
               for i in range(len(plays)-1))             # sorted desc
    # every play is genuinely +EV
    assert all(p["conservative_ev_pct"] > 0 for p in plays)

    print("LINESHOP SELFTEST PASS — devig/best-line/fair-EV/consensus/stale/rank all exact")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print("Pure line-shop engine. Live multi-book odds are pulled on Actions via mlb_lineshop_run.py.")
