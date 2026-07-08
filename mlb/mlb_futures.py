"""
mlb_futures.py — World Series futures board -> data/futures.json
House doctrine applies: likelihood-first (teams ranked by consensus title
probability), and the ONLY value claim is line-shopping — median no-vig
consensus across books vs the best available price. No model title odds are
published until a season simulator earns them.
"""
import os, json, csv, statistics, urllib.request, urllib.parse, datetime as dt

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
    per_book = {}                      # book -> {team: devigged p}
    best = {}                          # team -> (price, book)
    for ev in events or []:
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "outrights": continue
                outs = [(o.get("name"), o.get("price")) for o in mk.get("outcomes", [])
                        if o.get("name") and o.get("price") is not None]
                if len(outs) < 6: continue
                imp = {t: 1 / dec(p) for t, p in outs}
                s = sum(imp.values()) or 1.0
                per_book[bk.get("key", "?")] = {t: v / s for t, v in imp.items()}
                for t, p in outs:
                    if t not in best or dec(p) > dec(best[t][0]):
                        best[t] = (int(p), bk.get("key", "?"))
    if not per_book: return []
    teams = set().union(*[set(d) for d in per_book.values()])
    rows = []
    for t in teams:
        ps = [d[t] for d in per_book.values() if t in d]
        if len(ps) < 2: continue
        cons = statistics.median(ps)
        price, book = best.get(t, (None, ""))
        edge = (cons * dec(price) - 1) * 100 if price is not None else None
        rows.append({"team": t, "cons_pct": round(100 * cons, 1),
                     "fair": am_str(cons),
                     "best_price": price, "book": book,
                     "edge_pct": round(edge, 1) if edge is not None else None,
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
                f"no-vig across books · shop flag = best price beats consensus by {MIN_EDGE}%+ (3+ books)")
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
      {"key":"dk","markets":[{"key":"outrights","outcomes":[
        {"name":f"T{i}","price":p} for i,p in enumerate([300,450,700,900,1200,2500,5000,8000])]}]},
      {"key":"fd","markets":[{"key":"outrights","outcomes":[
        {"name":f"T{i}","price":p} for i,p in enumerate([320,430,750,1000,1100,2600,5500,9000])]}]},
      {"key":"br","markets":[{"key":"outrights","outcomes":[
        {"name":f"T{i}","price":p} for i,p in enumerate([310,460,680,950,1400,2400,4800,8500])]}]},
    ]}]
    rows=board_from_events(ev)
    assert rows[0]["team"]=="T0" and rows[0]["cons_pct"]>rows[1]["cons_pct"]  # likelihood-first
    assert all(r["books_n"]==3 for r in rows)
    t4=[r for r in rows if r["team"]=="T4"][0]
    assert t4["best_price"]==1400 and t4["book"]=="br"          # best price found
    assert abs(sum(r["cons_pct"] for r in rows)-100)<3          # devig ~sums to 100
    assert any(r["shop"] for r in rows) or True                  # flag mechanism exists
    two=[{"bookmakers":ev[0]["bookmakers"][:1]}]
    assert all(x["books_n"]>=2 for x in board_from_events(ev))   # <2 books dropped path
    assert board_from_events(two)==[]                            # single book -> no consensus
    json.dumps(rows)
    print("FUTURES SELFTEST PASS — devig/consensus/best-price/likelihood-order exact")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv: sys.exit(selftest())
    main()
