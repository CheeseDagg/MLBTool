#!/usr/bin/env python3
"""Which books a price may be QUOTED from — the one place that answers it.

A price you cannot take is not a better price. Every board in this repo was
computing "best available number" as the max across the whole multi-book feed,
which is the correct definition for a bettor with accounts everywhere and the
wrong one for this user, who bets FanDuel. The failure is not cosmetic: EV is
computed against the quoted decimal, so a DraftKings +310 next to a FanDuel
+250 produces an EV number, a Kelly stake and a board position that belong to a
bet that cannot be placed. The board then ranks unplaceable rows above placeable
ones, which is worse than not having the row at all.

The split this module enforces:

  QUOTED / EV'd / RANKED   -> BETTABLE books only
  CONSENSUS / FAIR / HOLD  -> every book in the feed

The second half matters as much as the first. Consensus is an estimate of the
true probability, and it gets better the more books it pools — throwing away
Pinnacle's number because you cannot bet Pinnacle would make the fair price
worse for no gain. The same all-books set also has to keep feeding the
line-shop-value and stale-outlier diagnostics, which are statements about the
FIELD, not about your ticket.

Book identifiers arrive in two shapes from The Odds API: bookmakers[].key
("fanduel") in the event endpoints and bookmakers[].title ("FanDuel") in the
odds endpoint mlb_odds.py uses. Both normalize to the same token here, so a
caller never has to know which one it is holding.

Change BETTABLE, not the call sites.
"""

BETTABLE = {"fanduel"}


def _key(name):
    """'FanDuel' / 'fanduel' / 'Fan Duel' / 'FANDUEL_US' -> 'fanduel'."""
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def is_bettable(name):
    return _key(name) in BETTABLE


def bettable(prices):
    """{book: american} -> the same dict keeping only books you can bet.

    None-valued entries are dropped as well: a book that did not price this side
    is not a quote, and leaving it in makes n_books lie."""
    return {b: a for b, a in (prices or {}).items() if a is not None and is_bettable(b)}


def _dec(am):
    am = float(am)
    return 1 + am / 100.0 if am > 0 else 1 + 100.0 / (-am)


def best(prices):
    """(book, american) of the best BETTABLE price, or (None, None) if you cannot
    bet this at all. Compared on decimal payout, not on the American integer:
    +150 and -150 do not order correctly as signed integers."""
    c = bettable(prices)
    if not c:
        return None, None
    b = max(c, key=lambda k: _dec(c[k]))
    return b, c[b]


def best_any(prices):
    """(book, american) of the best price ANYWHERE in the feed.

    Reported beside the bettable one, never in place of it — the point of showing
    it is to make the cost of the restriction visible ("FanDuel -140, best
    elsewhere +105"), not to offer it as a play."""
    c = {b: a for b, a in (prices or {}).items() if a is not None}
    if not c:
        return None, None
    b = max(c, key=lambda k: _dec(c[k]))
    return b, c[b]


def selftest():
    assert _key("FanDuel") == _key("fanduel") == _key("Fan Duel") == "fanduel", \
        "key and title spellings must normalize together"
    assert is_bettable("FanDuel") and is_bettable("fanduel")
    assert not is_bettable("DraftKings") and not is_bettable("") and not is_bettable(None)
    p = {"draftkings": 310, "fanduel": 250, "betrivers": None, "williamhill_us": 240}
    assert bettable(p) == {"fanduel": 250}, "a book that did not price the side is not a quote"
    assert best(p) == ("fanduel", 250), "best price must be the one you can take"
    assert best_any(p) == ("draftkings", 310), "best_any still reports the field's top"
    # decimal ordering, not signed-integer ordering
    assert best({"fanduel": -150}) == ("fanduel", -150)
    assert best_any({"a": -150, "b": 150}) == ("b", 150), "+150 pays more than -150"
    assert best_any({"a": -110, "b": -105}) == ("b", -105), "-105 pays more than -110"
    # no bettable quote is a real answer, not a fallback to someone else's price
    assert best({"draftkings": 310}) == (None, None), \
        "an unbettable-only market must return no price, NOT the best elsewhere"
    assert best({}) == (None, None) and best(None) == (None, None)
    print(f"mlb_books selftest OK — BETTABLE={sorted(BETTABLE)}")


if __name__ == "__main__":
    selftest()
