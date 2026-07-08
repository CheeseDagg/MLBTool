"""
mlb_odds.py  —  pull MLB moneylines across books (for line-shopping)
Get a free key at https://the-odds-api.com, paste it below, run:  python mlb_odds.py
-> ./data/mlb_odds.csv
"""
import os, sys, csv, json, urllib.request, urllib.parse

# ================= PASTE YOUR KEY BETWEEN THE QUOTES =================
API_KEY = os.environ.get("ODDS_API_KEY", "")
# ====================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

def pull():
    if not API_KEY:
        print("[odds] off — ODDS_API_KEY secret not set; writing empty odds file")
        path = os.path.join(DATA_DIR, "mlb_odds.csv")
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["game_id","commence","home","away","book","home_ml","away_ml"]).writeheader()
        return
    q = urllib.parse.urlencode({"apiKey": API_KEY, "regions": "us", "markets": "h2h",
                                "oddsFormat": "american", "dateFormat": "iso"})
    print("[odds] pulling MLB moneylines across US books ...")
    with urllib.request.urlopen(f"{BASE}?{q}", timeout=30) as r:
        remaining = r.headers.get("x-requests-remaining")
        games = json.load(r)
    print(f"   {len(games)} games returned  |  requests remaining this month: {remaining}")
    rows = []
    for g in games:
        home, away = g.get("home_team"), g.get("away_team")
        for bk in g.get("bookmakers", []):
            h2h = next((m for m in bk.get("markets", []) if m.get("key") == "h2h"), None)
            if not h2h: continue
            pr = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
            if home in pr and away in pr:
                rows.append({"game_id": g.get("id"), "commence": g.get("commence_time"),
                             "home": home, "away": away, "book": bk.get("title"),
                             "home_ml": pr[home], "away_ml": pr[away]})
    path = os.path.join(DATA_DIR, "mlb_odds.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_id","commence","home","away","book","home_ml","away_ml"])
        w.writeheader(); w.writerows(rows)
    books = sorted(set(r["book"] for r in rows))
    print(f"   -> saved mlb_odds.csv  ({len(rows)} lines across {len(books)} books: {books})")

if __name__ == "__main__":
    try: pull()
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        print("If 401: key is wrong/inactive. If 429: out of monthly requests.")
