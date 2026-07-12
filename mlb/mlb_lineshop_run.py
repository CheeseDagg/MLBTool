#!/usr/bin/env python3
"""
mlb_lineshop_run.py — feeds real multi-book odds into the line-shop engine, ON ACTIONS.

Flow:
  1. Load today's board (slate.json) -> our Marcel-calibrated fair P(homer) per player.
  2. For each game, pull batter_home_runs odds from The Odds API across our books
     (DraftKings, FanDuel, BetRivers, + William Hill as a 4th field member) — one call
     per event returns every book's over/under per player.
  3. Run mlb_lineshop.analyze_player for each batter, rank_board to the +EV plays.
  4. Write data/lineshop.json: the ranked plays (best book, best price, EV vs fair,
     EV vs vig-free consensus, stale flags) for the dashboard's Line Shop tab.

The API key is read from the ODDS_API_KEY env/secret — never hardcoded. If odds are
unavailable (no key, quota, off-hours) the runner writes an empty-but-valid file so the
dashboard degrades gracefully.
"""
import os, json, time, urllib.request, urllib.parse, datetime as dt
import mlb_lineshop as LS

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
API_KEY = os.environ.get("ODDS_API_KEY", "")
BOOKS = ["draftkings", "fanduel", "betrivers", "williamhill_us"]
SPORT = "baseball_mlb"

def norm(s):
    if not isinstance(s, str): return ""
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower().replace(".", "").replace("'", "").strip()

QUOTA = {"remaining": None, "used": None}
def _get(url):
    with urllib.request.urlopen(url, timeout=25) as r:
        h = r.headers
        if h.get("x-requests-remaining") is not None:
            QUOTA["remaining"] = h.get("x-requests-remaining")
            QUOTA["used"] = h.get("x-requests-used")
        return json.loads(r.read())

def list_events():
    url = (f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"
           f"?apiKey={API_KEY}&dateFormat=iso")
    return _get(url)

def event_hr_odds(event_id):
    """Returns {player_norm: {'over': {book: am}, 'under': {book: am}}} for HR props."""
    q = urllib.parse.urlencode({
        "apiKey": API_KEY, "regions": "us", "markets": "batter_home_runs",
        "oddsFormat": "american", "bookmakers": ",".join(BOOKS),
    })
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds?{q}"
    data = _get(url)
    out = {}
    for bk in data.get("bookmakers", []):
        book = bk.get("key")
        for mk in bk.get("markets", []):
            if mk.get("key") != "batter_home_runs":
                continue
            for oc in mk.get("outcomes", []):
                # outcomes carry player name + Over/Under (description or name conventions vary)
                pname = oc.get("description") or oc.get("participant") or oc.get("name", "")
                side = (oc.get("name") or "").lower()   # "over"/"under"
                price = oc.get("price")
                if not pname or side not in ("over", "under") or price is None:
                    continue
                p = out.setdefault(norm(pname), {"over": {}, "under": {}})
                p[side][book] = price
    return out

def load_board_fairs():
    """{player_norm: fair_prob} from our Marcel-calibrated slate."""
    slate = json.load(open(os.path.join(DATA, "slate.json")))
    return {norm(r["player"]): float(r["hr_pct"]) / 100 for r in slate.get("hr_board", [])}

def main():
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")
    if not API_KEY:
        json.dump({"generated": ts, "plays": [], "note": "no ODDS_API_KEY — set the secret to enable line shopping"},
                  open(os.path.join(DATA, "lineshop.json"), "w"), indent=1)
        print("no ODDS_API_KEY set; wrote empty lineshop.json")
        return
    try:
        fairs = load_board_fairs()
    except Exception as e:
        print(f"board load failed ({type(e).__name__}); aborting"); return
    print(f"board fair probs: {len(fairs)} players")

    analyses = []
    try:
        events = list_events()
    except Exception as e:
        json.dump({"generated": ts, "plays": [], "note": f"odds API error: {type(e).__name__}"},
                  open(os.path.join(DATA, "lineshop.json"), "w"), indent=1)
        print(f"events fetch failed: {type(e).__name__}"); return

    matched = 0
    err_counts = {}
    for ev in events:
        eid = ev.get("id")
        if not eid: continue
        try:
            odds = event_hr_odds(eid)
        except Exception as e:
            code = getattr(e, "code", None) or type(e).__name__
            err_counts[str(code)] = err_counts.get(str(code), 0) + 1
            continue
        for pname, sides in odds.items():
            fair = fairs.get(pname)
            if fair is None:
                continue                        # only price players our model has a fair for
            over = sides.get("over", {})
            under = sides.get("under", {})
            if not over:
                continue
            a = LS.analyze_player(pname, fair, over, under or None)
            if a:
                analyses.append(a)
                matched += 1
        time.sleep(0.3)                          # polite; also spares API quota

    plays = LS.rank_board(analyses, min_ev_fair=0.0, min_books=2)
    diag = f"events {len(events)} · errors {err_counts or 'none'} · credits remaining {QUOTA['remaining']} (used {QUOTA['used']})"
    print("DIAG:", diag)
    out = {"generated": ts, "n_analyzed": matched, "n_plays": len(plays),
           "books": BOOKS, "plays": plays, "diag": diag,
           "note": "Plays are +EV vs BOTH our fair number AND the vig-free market consensus. "
                   "Best price shown is the book to bet. Stale flags = a book out of step with the field."}
    json.dump(out, open(os.path.join(DATA, "lineshop.json"), "w"), indent=1)
    print(f"\nLINE SHOP: analyzed {matched} player-markets -> {len(plays)} +EV plays")
    for p in plays[:10]:
        star = " *STALE*" if p["stale_flag"] else ""
        print(f"  {p['player']:<22} {p['best_book']:<12} {p['best_price']:+d}  "
              f"EV_fair {p['ev_vs_fair_pct']:+.1f}%  EV_cons {p.get('ev_vs_consensus_pct')}%{star}")

if __name__ == "__main__":
    main()
