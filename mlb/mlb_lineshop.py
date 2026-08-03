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

ONE CORRECTION TO EDGE 1 ABOVE. "Betting +310 instead of +250 on the same bet" is only
riskless edge if you have an account at the book showing +310. This user bets FanDuel.
best_line() is therefore restricted to mlb_books.BETTABLE, while consensus_prob(), the
field average and the stale-outlier test keep reading EVERY book in the feed -- those
three are statements about the market, and pooling more books makes them better. The
best price anywhere is still computed and returned as `any_*`, so the cost of the
restriction is visible on the board; it is never the number EV is taken against.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_books import BETTABLE, is_bettable, bettable as _bettable

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

def best_line(book_prices, bettable_only=True):
    """book_prices: {book: american_odds} for the SAME bet (e.g. player HR over).
    Returns (best_book, best_american, best_decimal) — the highest payout you can
    actually take, or None if no BETTABLE book priced this side.

    bettable_only=False gives the old all-books answer. It has exactly one legitimate
    caller: the `any_*` fields, which exist to show what the restriction costs. Do not
    feed that number to an EV or a stake."""
    best = None
    for bk, am in book_prices.items():
        if am is None: continue
        if bettable_only and not is_bettable(bk): continue
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

def analyze_player(name, fair_prob, book_over, book_under=None, stale_mult=1.6, push_prob=0.0):
    """Full line-shop analysis for one player's prop.

    fair_prob : our model P(over wins) — the trustworthy fair number.
    book_over : {book: american} best-available over prices per book.
    book_under: {book: american} unders (enables per-book devig + consensus).
    push_prob : P(exact push) on a whole-number line (K props). A push RETURNS the
                stake, so EV = fair*dec - 1 + push_prob; without this term a whole-line
                over's EV is understated by the full push mass (HR props: 0, no change).

    Returns a dict with best line, edge vs fair, edge vs consensus, and flags.
    """
    bl = best_line(book_over)
    if not bl:
        # No bettable quote. Returning None drops the row entirely, which is correct:
        # the alternative -- falling back to the field's best -- is what published
        # DraftKings prices as plays. A market you cannot bet has no price here.
        analyze_player.skipped_unbettable = getattr(analyze_player, "skipped_unbettable", 0) + 1
        return None
    best_book, best_am, best_dec = bl
    any_bl = best_line(book_over, bettable_only=False)

    # 1) EDGE vs our fair number: EV per $1 = fair_prob*decimal - 1 + push (stake returned on a push)
    ev_fair = fair_prob * best_dec - 1.0 + push_prob

    # 2) CONSENSUS (vig-free true-prob estimate) and edge vs it
    cons = consensus_prob(book_over, book_under)
    ev_cons = (cons * best_dec - 1.0 + push_prob) if cons is not None else None

    # 3) LINE-SHOP VALUE: how much YOUR price beats the field (in implied-prob pts).
    #    The field is every book in the feed, bettable or not -- it is a description of
    #    the market. Note the sign can now go NEGATIVE, and that is the honest reading:
    #    when FanDuel is worse than the field average, shopping has negative value
    #    because there is nowhere to shop to. Before the BETTABLE split this quantity
    #    was structurally >= 0 (it compared the field's own maximum to the field's mean),
    #    which made it look like free money on every single row.
    field_probs = [american_to_prob(a) for a in book_over.values() if a is not None]
    field_avg = sum(field_probs) / len(field_probs) if field_probs else None
    best_prob = american_to_prob(best_am)
    shop_edge_pts = round((field_avg - best_prob) * 100, 1) if field_avg is not None else None

    # 4) STALE / OUTLIER: is YOUR price an outlier high vs the field? An outlier sitting
    #    at a book you have no account with is not a play, so the flag follows the
    #    bettable price. `field_outlier_*` below still reports one anywhere in the feed,
    #    because a book wildly off the others is news even when you cannot take it.
    stale = False
    if field_avg is not None and best_prob > 0:
        # best implied prob much LOWER than field avg == best payout much higher == outlier
        if field_avg / best_prob >= stale_mult and len(field_probs) >= 2:
            stale = True
    any_book, any_am = (any_bl[0], any_bl[1]) if any_bl else (None, None)
    any_prob = american_to_prob(any_am) if any_am is not None else None
    field_outlier = bool(field_avg is not None and any_prob and len(field_probs) >= 2
                         and field_avg / any_prob >= stale_mult)

    # 5) MODEL-vs-MARKET disagreement: if our fair and consensus diverge a lot, trust market
    model_market_gap = round((fair_prob - cons) * 100, 1) if cons is not None else None

    return {
        "player": name,
        "fair_prob": round(fair_prob, 4),
        "fair_american": prob_to_american(fair_prob),
        "best_book": best_book,
        "best_price": int(best_am),
        # n_books counts the whole field: it is the "is this consensus trustworthy"
        # number that rank_board gates on, not a count of your own options.
        "n_books": sum(1 for a in book_over.values() if a is not None),
        "n_bettable": len(_bettable(book_over)),
        # what the restriction costs, reported and never bet
        "any_book": any_book if any_book != best_book else None,
        "any_price": int(any_am) if any_am is not None and any_book != best_book else None,
        "field_outlier": field_outlier,
        "ev_vs_fair_pct": round(ev_fair * 100, 1),
        "consensus_prob": round(cons, 4) if cons is not None else None,
        "ev_vs_consensus_pct": round(ev_cons * 100, 1) if ev_cons is not None else None,
        "shop_edge_pts": shop_edge_pts,        # value of your price vs the field average
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

    # --- best line picks highest BETTABLE payout ---
    # (book keys are the real Odds API keys throughout; is_bettable normalizes them)
    bp = {"draftkings": +250, "fanduel": +310, "betrivers": +275}
    bb, bam, bdec = best_line(bp)
    assert bb == "fanduel" and bam == 310       # +310 is the best payout AND bettable
    bp2 = {"draftkings": +400, "fanduel": +310, "betrivers": +275}
    assert best_line(bp2)[:2] == ("fanduel", 310), \
        "DraftKings +400 is not a better price, it is a price you cannot take"
    assert best_line(bp2, bettable_only=False)[:2] == ("draftkings", 400), \
        "the all-books answer is still available for the `any_*` disclosure"
    assert best_line({"draftkings": +400, "betrivers": +275}) is None, \
        "no FanDuel quote means no price, not somebody else's price"

    # --- LINE SHOP: taking +310 vs field avg is real edge ---
    a = analyze_player("Judge", fair_prob=0.28,
                       book_over={"draftkings": +250, "fanduel": +310, "betrivers": +275})
    assert a["best_book"] == "fanduel" and a["best_price"] == 310
    # EV vs fair: .28 * 4.10 - 1 = +.148 -> +14.8%
    assert abs(a["ev_vs_fair_pct"] - 14.8) < 0.2
    assert a["shop_edge_pts"] > 0               # your price beats the field
    assert a["n_books"] == 3 and a["n_bettable"] == 1
    assert a["any_book"] is None, "FanDuel is the top price, so there is nothing to disclose"

    # --- THE BETTABLE GUARD: the row this whole change exists for ---
    # DraftKings +400 next to FanDuel +250 at a 0.28 fair. The OLD code quoted +400 and
    # published EV = .28*5.00-1 = +40%, a Kelly stake and the top board slot -- for a bet
    # with no account behind it. The bettable answer is .28*3.50-1 = -2%: not a play.
    ab = analyze_player("Phantom", fair_prob=0.28,
                        book_over={"draftkings": +400, "fanduel": +250, "betrivers": +260})
    assert ab["best_book"] == "fanduel" and ab["best_price"] == 250
    assert abs(ab["ev_vs_fair_pct"] - (-2.0)) < 0.2, ab["ev_vs_fair_pct"]
    assert ab["any_book"] == "draftkings" and ab["any_price"] == 400, \
        "the unreachable price must still be shown, so the cost of the restriction is visible"
    assert not rank_board([ab], min_ev_fair=0.0, min_books=2), \
        "the phantom edge must not survive ranking"
    _n0 = getattr(analyze_player, "skipped_unbettable", 0)
    assert analyze_player("Nowhere", 0.28, {"draftkings": +400, "betrivers": +275}) is None
    assert getattr(analyze_player, "skipped_unbettable", 0) == _n0 + 1, \
        "a dropped unbettable market must be counted, not vanish"

    # shop_edge_pts is now allowed to go NEGATIVE -- FanDuel worse than the field
    aneg = analyze_player("Shopless", fair_prob=0.28,
                          book_over={"draftkings": +400, "fanduel": +200, "betrivers": +380})
    assert aneg["shop_edge_pts"] < 0, \
        "when your book is the worst of the field, shopping has negative value"

    # --- FAIR-VALUE GATE: a bet where book UNDERpays fair is NOT +EV ---
    a2 = analyze_player("ColdBat", fair_prob=0.10, book_over={"draftkings": +250, "fanduel": +260})
    # .10 * 3.6 - 1 = -.64 -> deeply -EV, must be excluded by rank_board
    assert a2["ev_vs_fair_pct"] < 0

    # --- CONSENSUS devig: with unders, consensus is vig-free and sharper ---
    a3 = analyze_player("Star", fair_prob=0.30,
                        book_over={"draftkings": +240, "fanduel": +250},
                        book_under={"draftkings": -300, "fanduel": -320})
    assert a3["consensus_prob"] is not None
    assert a3["ev_vs_consensus_pct"] is not None
    # model_market gap computed
    assert a3["model_market_gap_pts"] is not None
    # CONSENSUS MUST STILL POOL EVERY BOOK. Drop the unbettable one and it has to move;
    # if it does not, the BETTABLE split leaked into the market estimate, which would
    # make the fair price worse for no gain.
    a3b = analyze_player("Star", fair_prob=0.30,
                         book_over={"fanduel": +250}, book_under={"fanduel": -320})
    assert abs(a3["consensus_prob"] - a3b["consensus_prob"]) > 1e-4, \
        "consensus must read every book in the feed, not just the bettable ones"

    # --- STALE/OUTLIER: one book way off the field ---
    a4 = analyze_player("Newsy", fair_prob=0.25,
                        book_over={"draftkings": +250, "fanduel": +600, "betrivers": +255})
    assert a4["best_book"] == "fanduel" and a4["stale_flag"] is True   # +600 is an outlier
    # an outlier you CANNOT take is news, not a play: field_outlier fires, stale does not
    a4b = analyze_player("NewsyElsewhere", fair_prob=0.25,
                         book_over={"draftkings": +600, "fanduel": +250, "betrivers": +255})
    assert a4b["stale_flag"] is False and a4b["field_outlier"] is True

    # a tight field is NOT stale
    a5 = analyze_player("Tight", fair_prob=0.25,
                        book_over={"draftkings": +250, "fanduel": +255, "betrivers": +260})
    assert a5["stale_flag"] is False and a5["field_outlier"] is False

    # --- RANK: only +EV-both-ways, multi-book plays survive, sorted by conservative EV ---
    board = [
        analyze_player("A", 0.30, {"draftkings": +260, "fanduel": +320}),          # model +EV
        analyze_player("B", 0.10, {"draftkings": +250, "fanduel": +260}),          # -EV, drop
        analyze_player("C", 0.28, {"fanduel": +250}),                              # 1 book, drop
        analyze_player("D", 0.32, {"draftkings": +240, "fanduel": +250},
                       {"draftkings": -300, "fanduel": -320}),                     # both-way check
    ]
    plays = rank_board(board, min_ev_fair=0.0, min_books=2)
    names = [p["player"] for p in plays]
    assert "B" not in names and "C" not in names        # -EV and single-book excluded
    assert "A" in names
    assert all(plays[i]["conservative_ev_pct"] >= plays[i+1]["conservative_ev_pct"]
               for i in range(len(plays)-1))             # sorted desc
    # every play is genuinely +EV
    assert all(p["conservative_ev_pct"] > 0 for p in plays)

    # --- PUSH MASS: a whole-number K line returns the stake on a push, so the over's EV
    #     is fair*dec - 1 + push. Without the term it's understated by the full push mass. ---
    ap = analyze_player("Ace O7", fair_prob=0.46, book_over={"fanduel": +100}, push_prob=0.15)
    an = analyze_player("Ace O7", fair_prob=0.46, book_over={"fanduel": +100}, push_prob=0.0)
    # dec=2.0: with push .46*2-1+.15 = +.07 ; without = -.08  -> understated by exactly push*100
    assert abs(ap["ev_vs_fair_pct"] - 7.0) < 0.1, ap["ev_vs_fair_pct"]
    assert abs(an["ev_vs_fair_pct"] - (-8.0)) < 0.1, an["ev_vs_fair_pct"]
    assert abs((ap["ev_vs_fair_pct"] - an["ev_vs_fair_pct"]) - 15.0) < 0.1
    print("LINESHOP SELFTEST PASS — devig/best-line/fair-EV/consensus/stale/rank/push all exact")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print("Pure line-shop engine. Live multi-book odds are pulled on Actions via mlb_lineshop_run.py.")
