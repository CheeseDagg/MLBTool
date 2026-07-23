"""
mlb_publish.py  —  run the whole pipeline, emit slate.json for the dashboard
============================================================================
One command. Pulls today's schedule + odds, builds the model from game_starters,
applies xwOBA + park + weather to every game, runs the edge-finder, and writes
a single slate.json the dashboard reads.

RUN:  python mlb_publish.py
Also run inside the GitHub Action daily. Fails soft: any layer that can't pull
(odds key missing, weather down) degrades gracefully and is tagged in the output.
"""
import os, sys, json, glob, math, datetime as dt
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlb_model as M
from mlb_parks import adjust_total, tag as park_tag

def _scrub(o):
    """Browsers reject bare NaN/Infinity in JSON (Python's json tolerates them,
    which is how a poisoned slate.json shipped and blanked the whole dashboard).
    Recursively convert non-finite floats to None; allow_nan=False on the dump
    is the tripwire if anything ever slips past."""
    import math
    if isinstance(o, float) and not math.isfinite(o): return None
    if isinstance(o, dict):  return {k: _scrub(v) for k, v in o.items()}
    if isinstance(o, list):  return [_scrub(v) for v in o]
    return o
try:
    from mlb_weather import game_weather_mult
    HAS_WX = True
except Exception:
    HAS_WX = False

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_slate_inputs():
    gs = os.path.join(DATA, "game_starters.csv")
    if not os.path.exists(gs):
        sys.exit("need data/game_starters.csv — run mlb_pitchers.py first")
    s = M.load(gs)
    m = M.build(s)
    xw = M.load_xwoba(os.path.join(DATA, "pitcher_xstats.csv"))
    return s, m, xw


def todays_games():
    """Newest schedule CSV that actually parses. A 0-byte/corrupt file (transient
    pull failure) must not kill the publish — fall back through older pulls,
    warn loudly, and degrade to empty rather than crash."""
    hits = sorted(glob.glob(os.path.join(DATA, "schedule_*.csv"))) or \
           [p for p in [os.path.join(DATA, "schedule.csv")] if os.path.exists(p)]
    for p in reversed(hits):
        try:
            if os.path.getsize(p) == 0:
                print(f"[todays_games] WARNING: {os.path.basename(p)} is empty — skipping")
                continue
            sc = pd.read_csv(p)
            if p != hits[-1]:
                print(f"[todays_games] WARNING: fell back to older pull {os.path.basename(p)}")
            return sc
        except Exception as e:
            print(f"[todays_games] WARNING: {os.path.basename(p)} unreadable ({type(e).__name__}) — skipping")
    print("[todays_games] WARNING: no parseable schedule CSV — proceeding with empty frame")
    return pd.DataFrame()


def edges_block():
    """Run the edge-finder if odds exist; return [] otherwise."""
    path = os.path.join(DATA, "mlb_odds.csv")
    if not os.path.exists(path):
        return [], "no odds file (run mlb_odds.py)"
    try:
        import mlb_edge as E
        df = E.find_edges(path, bankroll=1000.0)
        skipped = getattr(E.find_edges, "_skipped", 0)
        note = f"{skipped} in-progress game(s) skipped" if skipped else ""
        if df is None or not len(df):
            return [], note or "no +EV sides vs consensus right now"
        rows = [{"game": r.game, "bet": r.bet, "price": int(r.price), "book": r.book,
                 "fair": round(float(r.fair), 4), "ev_pct": round(float(r.ev) * 100, 2),
                 "kelly_frac": round(float(r.stake) / 1000.0, 5)} for r in df.itertuples()]
        return rows, note
    except Exception as e:
        return [], f"edge-finder error: {type(e).__name__}"


def parlay_block():
    path = os.path.join(DATA, "mlb_odds.csv")
    if not os.path.exists(path):
        return [], []
    try:
        import mlb_edge as E
        from mlb_parlay import build_parlays
        df = E.find_edges(path, bankroll=1000.0)
        if df is None or len(df) < 2:
            return [], []
        return build_parlays(df)
    except Exception:
        return [], []


def backtest_block(s):
    bt = M.backtest(s, True)
    return {"n": int(bt["n"]), "acc": round(float(bt["acc"]) * 100, 1),
            "brier": round(float(bt["brier"]), 4),
            "home_base": round(float(bt["base"]) * 100, 1),
            "cal": [{"bucket": round(c * 100), "pred": round(p * 100, 1),
                     "actual": round(a * 100, 1), "n": int(n)} for c, p, a, n in bt["cal"]]}


def main():
    s, m, xw = load_slate_inputs()
    sc = todays_games()
    games = []
    for _, r in sc.iterrows():
        h, a = r.get("home"), r.get("away")
        if h not in m["N"] or a not in m["N"]:
            continue
        pr = M.predict_live(m, h, a, r.get("home_prob_pitcher"), r.get("away_prob_pitcher"), xw)
        venue = r.get("venue", "")
        tot_park, park_eff, park_conf, park_raw = adjust_total(pr["total"], venue)
        wx_mult, wx_tag = game_weather_mult(venue) if HAS_WX else (1.0, "wx module off")
        games.append({
            "away": a, "home": h, "venue": venue,
            "date": str(r.get("date") or "")[:10],
            "away_sp": r.get("away_prob_pitcher") or "?",
            "home_sp": r.get("home_prob_pitcher") or "?",
            "p_home": round(float(pr["p_home"]) * 100, 1),
            "raw_total": round(float(pr["total"]), 2),
            "adj_total": round(float(pr["total"]) * park_eff * wx_mult, 2),
            "park": park_tag(venue), "wx": wx_tag,
        })

    # team ratings table
    ratings = sorted(
        [{"team": t, "O": round(M._O(m, t), 3), "D": round(M._D(m, t), 3),
          "net": round(M._O(m, t) / M._D(m, t), 3)} for t in m["N"]],
        key=lambda x: -x["net"])

    edge_rows, edge_note = edges_block()
    parlays, near_parlays = parlay_block()
    try:
        import mlb_hr
        hr_rows, hr_note = mlb_hr.load_board(DATA)
    except Exception as e:
        hr_rows, hr_note = [], f"hr module error: {type(e).__name__}"
    try:
        import mlb_grade
        hr_cal = mlb_grade.panel_for_publish()
    except Exception as e:
        hr_cal = {"n": 0, "error": type(e).__name__}
    futures = None
    try:
        fp = os.path.join(DATA, "futures.json")
        if os.path.exists(fp):
            with open(fp) as f: futures = json.load(f)
    except Exception:
        futures = None

    # Authoritative slate date = the games' own statsapi game_date, NOT wall-clock. If
    # todays_games() silently fell back to an older schedule pull, these rows carry the
    # OLDER date — the real freshness signal the stale-slate guard + dashboard need.
    # (`generated` is always "now" and cannot detect a stale-but-freshly-published slate.)
    _sdates = sorted({g["date"] for g in games if g.get("date")})
    slate_date = _sdates[-1] if _sdates else None
    out = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "slate_date": slate_date,
        "games": games, "ratings": ratings,
        "edges": edge_rows, "edge_note": edge_note,
        "parlays": parlays, "near_parlays": near_parlays,
        "hr_board": hr_rows, "hr_note": hr_note, "hr_cal": hr_cal, "futures": futures,
        "backtest": backtest_block(s),
        "league_rpg": round(float(m["L"]), 2),
        "xwoba_pitchers": len(xw),
    }
    path = os.path.join(DATA, "slate.json")
    with open(path, "w") as f:
        json.dump(_scrub(out), f, indent=1, allow_nan=False)
    print(f"slate.json written: {len(games)} games, {len(edge_rows)} edges "
          f"({edge_note or 'ok'}), {len(ratings)} teams rated, "
          f"{len(hr_rows)} HR-board rows")


if __name__ == "__main__":
    main()
