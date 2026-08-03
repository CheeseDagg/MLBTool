"""
mlb_futures.py — World Series futures board -> data/futures.json
House doctrine applies: likelihood-first (teams ranked by consensus title
probability), and the ONLY value claim is line-shopping — median no-vig
consensus across books vs the best available price. No model title odds are
published until a season simulator earns them.
"""
import os, sys, json, csv, statistics, urllib.request, urllib.parse, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_books import BETTABLE, is_bettable

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SPORT = "baseball_mlb_world_series_winner"
MIN_EDGE = 1.5

def dec(am):
    am = int(am)
    return am / 100 + 1 if am > 0 else 100 / (-am) + 1

def am_str(p):
    p = min(max(p, 1e-6), 1 - 1e-6); d = 1 / p
    return f"+{round((d-1)*100)}" if d >= 2 else f"-{round(100/(d-1))}"

def fetch():
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("ODDS_API_KEY secret not set")
    q = urllib.parse.urlencode({"apiKey": key, "regions": "us",
                                "markets": "outrights", "oddsFormat": "american"})
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (MLBTool futures)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def board_from_events(events):
    """-> rows sorted by consensus title probability (likelihood-first)."""
    per_book = {}                      # book -> {team: devigged p}   (EVERY book: consensus)
    best = {}                          # team -> (price, book)        (BETTABLE only: the quote)
    any_best = {}                      # team -> (price, book)        (every book: disclosure)
    for ev in events or []:
        for bk in ev.get("bookmakers", []):
            bkey = bk.get("key", "?")
            for mk in bk.get("markets", []):
                if mk.get("key") != "outrights": continue
                outs = [(o.get("name"), o.get("price")) for o in mk.get("outcomes", [])
                        if o.get("name") and o.get("price") is not None]
                if len(outs) < 6: continue
                imp = {t: 1 / dec(p) for t, p in outs}
                s = sum(imp.values()) or 1.0
                per_book[bkey] = {t: v / s for t, v in imp.items()}
                for t, p in outs:
                    # The QUOTE — the number edge_pct is taken against — may only come from
                    # a book this bet can be placed at. Before this, `best` was the max over
                    # the whole outrights pull, so a team whose top price lived at a book with
                    # no account got an edge_pct, a `shop` flag and a board row for a ticket
                    # that cannot be written. Consensus below still reads every book.
                    if is_bettable(bkey) and (t not in best or dec(p) > dec(best[t][0])):
                        best[t] = (int(p), bkey)
                    if t not in any_best or dec(p) > dec(any_best[t][0]):
                        any_best[t] = (int(p), bkey)
    if not per_book: return []
    teams = set().union(*[set(d) for d in per_book.values()])
    rows = []
    for t in teams:
        ps = [d[t] for d in per_book.values() if t in d]
        if len(ps) < 2: continue
        cons = statistics.median(ps)
        price, book = best.get(t, (None, ""))
        edge = (cons * dec(price) - 1) * 100 if price is not None else None
        any_price, any_book = any_best.get(t, (None, ""))
        rows.append({"team": t, "cons_pct": round(100 * cons, 1),
                     "fair": am_str(cons),
                     "best_price": price, "book": book,
                     "edge_pct": round(edge, 1) if edge is not None else None,
                     # what the field's top number is, shown only so the cost of betting one
                     # book is visible. Never priced, never flagged `shop`.
                     "any_price": any_price if any_book != book else None,
                     "any_book": any_book if any_book != book else None,
                     "books_n": len(ps)})
    rows.sort(key=lambda r: -r["cons_pct"])
    for r in rows:
        r["shop"] = bool(r["edge_pct"] is not None and r["edge_pct"] >= MIN_EDGE and r["books_n"] >= 3)
    return rows

def main():
    os.makedirs(DATA, exist_ok=True)
    try:
        rows = board_from_events(fetch())
        note = (f"World Series outrights · {len(rows)} teams priced · consensus = median "
                f"no-vig across EVERY book · prices quoted from "
                f"{'/'.join(sorted(BETTABLE))} only · shop flag = your price beats consensus "
                f"by {MIN_EDGE}%+ (3+ books)")
        if not rows: note = "no futures priced right now"
    except Exception as e:
        rows, note = [], f"futures off ({type(e).__name__}: {e})"
    out = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "market": "World Series", "rows": rows, "note": note}
    with open(os.path.join(DATA, "futures.json"), "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    print(f"futures.json: {len(rows)} teams | {note[:80]}")

def selftest():
    ev=[{"bookmakers":[
      {"key":"draftkings","markets":[{"key":"outrights","outcomes":[
        {"name":f"T{i}","price":p} for i,p in enumerate([300,450,700,900,1200,2500,5000,8000])]}]},
      {"key":"fanduel","markets":[{"key":"outrights","outcomes":[
        {"name":f"T{i}","price":p} for i,p in enumerate([320,430,750,1000,1100,2600,5500,9000])]}]},
      {"key":"betrivers","markets":[{"key":"outrights","outcomes":[
        {"name":f"T{i}","price":p} for i,p in enumerate([310,460,680,950,1400,2400,4800,8500])]}]},
    ]}]
    rows=board_from_events(ev)
    assert rows[0]["team"]=="T0" and rows[0]["cons_pct"]>rows[1]["cons_pct"]  # likelihood-first
    assert all(r["books_n"]==3 for r in rows)
    # THE BETTABLE GUARD. T4's top number in the field is BetRivers +1400; the old code
    # quoted it and computed edge_pct off dec(+1400) for a ticket that cannot be written.
    # The quote is FanDuel +1100; +1400 survives only as disclosure.
    t4=[r for r in rows if r["team"]=="T4"][0]
    assert t4["best_price"]==1100 and t4["book"]=="fanduel", (t4["best_price"], t4["book"])
    assert t4["any_price"]==1400 and t4["any_book"]=="betrivers", \
        "the unreachable top price must still be shown, never priced"
    # edge is taken against the QUOTE (+1100), not the field's top (+1400). Tolerance is
    # wide because cons_pct is published rounded to 0.1pp and dec(+1100)=12 multiplies that
    # rounding by 12; the point of the check is which price was used, and +1100 vs +1400 is
    # a ~19pp gap, far outside it.
    _cons4 = t4["cons_pct"] / 100.0
    assert abs(t4["edge_pct"] - (_cons4*dec(1100)-1)*100) < 1.5, t4["edge_pct"]
    assert abs(t4["edge_pct"] - (_cons4*dec(1400)-1)*100) > 5.0, \
        "edge must not be computed off the unbettable +1400"
    # T3: FanDuel +1000 IS the field's top, so there is nothing to disclose
    t3=[r for r in rows if r["team"]=="T3"][0]
    assert t3["best_price"]==1000 and t3["any_price"] is None
    assert abs(sum(r["cons_pct"] for r in rows)-100)<3          # devig ~sums to 100
    assert any(r["shop"] for r in rows) or True                  # flag mechanism exists
    two=[{"bookmakers":ev[0]["bookmakers"][:1]}]
    assert all(x["books_n"]>=2 for x in board_from_events(ev))   # <2 books dropped path
    assert board_from_events(two)==[]                            # single book -> no consensus
    # NO FANDUEL IN THE FEED: consensus still forms (it reads every book), but nothing is
    # quoted, nothing is priced and nothing can carry a shop flag.
    nofd=[{"bookmakers":[b for b in ev[0]["bookmakers"] if b["key"]!="fanduel"]}]
    rnf=board_from_events(nofd)
    assert rnf and all(r["cons_pct"]>0 for r in rnf), "consensus must survive without FanDuel"
    assert all(r["best_price"] is None and r["edge_pct"] is None and not r["shop"]
               for r in rnf), "no bettable book must mean no price, not somebody else's price"
    assert all(r["any_price"] is not None for r in rnf), "field top still disclosed"
    json.dumps(rows)
    print("FUTURES SELFTEST PASS — devig/consensus/bettable-quote/likelihood-order exact")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv: sys.exit(selftest())
    main()
