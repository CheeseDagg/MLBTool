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


def model_win_pct(games):
    """{(game_label, team): model win %} for every side on today's slate.

    The edge rows come out of the market layer, which by design never sees the model.
    The board is ranked by the chance a bet HITS, so each row needs the model's own
    number attached — without this the only likelihood available is the market's, and
    the tab could not be ordered on the model at all."""
    out = {}
    for g in games:
        lab = f"{g['away']} @ {g['home']}"
        try:
            ph = float(g["p_home"])
        except (TypeError, ValueError, KeyError):
            continue
        out[(lab, g["home"])] = round(ph, 1)
        out[(lab, g["away"])] = round(100.0 - ph, 1)
    return out


def by_likelihood(r):
    """Sort key for a bet row: the chance it HITS, descending. Model number when the
    row has one, consensus fair when it doesn't — never the edge."""
    return -(r["p_model"] if r.get("p_model") is not None else float(r["fair"]) * 100.0)


def edges_block(games=()):
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
        mp = model_win_pct(games)
        rows = [{"game": r.game, "bet": r.bet, "price": int(r.price), "book": r.book,
                 "fair": round(float(r.fair), 4), "ev_pct": round(float(r.ev) * 100, 2),
                 "kelly_frac": round(float(r.stake) / 1000.0, 5),
                 "p_model": mp.get((r.game, r.bet))} for r in df.itertuples()]
        # LIKELIHOOD-FIRST ORDER (house rule): most likely to hit at the top, model
        # number where we have one, consensus fair where we don't (team names that
        # don't line up between the odds feed and the schedule). EV and Kelly are
        # still on every row — they size the bet, they no longer rank it.
        rows.sort(key=by_likelihood)
        if any(r["p_model"] is None for r in rows):
            n_miss = sum(1 for r in rows if r["p_model"] is None)
            note = (note + " · " if note else "") + \
                   f"{n_miss} side(s) with no model number — ranked on consensus fair"
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

    # Likelihood-first: the slate table leads with the games the model is most sure
    # about, so the top of the page is the highest-probability read, not the first
    # first-pitch of the day. (The favourite's own chance, either side.)
    games.sort(key=lambda g: -max(float(g["p_home"]), 100.0 - float(g["p_home"])))

    edge_rows, edge_note = edges_block(games)
    parlays, near_parlays = parlay_block()
    try:
        import mlb_hr
        hr_rows, hr_note = mlb_hr.load_board(DATA)
    except Exception as e:
        hr_rows, hr_note = [], f"hr module error: {type(e).__name__}"
    try:
        import mlb_grade
        hr_cal = mlb_grade.panel_for_publish()
        # Per-bucket settled hit rate, so every board row can carry what the model's
        # own numbers at that level have actually done. Ships even when it's mostly
        # empty — an absent bucket renders as "—", which is the honest answer.
        hr_rel = mlb_grade.reliability_for_publish()
    except Exception as e:
        hr_cal = {"n": 0, "error": type(e).__name__}
        hr_rel = {"buckets": [], "error": type(e).__name__}
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
        "hr_board": hr_rows, "hr_note": hr_note, "hr_cal": hr_cal,
        "hr_reliability": hr_rel, "futures": futures,
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


def selftest():
    """Offline checks. `--selftest` used to fall straight through to main(), which
    needed a live schedule pull and a populated game_starters.csv and so died with a
    KeyError deep in pandas — a test that can only run on a day the pipeline already
    worked is not a test. These use synthetic inputs only."""
    import tempfile
    ok = [0, 0]

    def chk(c, msg):
        ok[1] += 1
        ok[0] += bool(c)
        print(("PASS  " if c else "FAIL  ") + msg)

    # _scrub: the poisoned-slate tripwire
    bad = {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": float("-inf")}}
    sc = _scrub(bad)
    chk(sc["a"] is None and sc["b"][1] is None and sc["c"]["d"] is None,
        "_scrub nulls every non-finite float, at any depth")
    chk(json.dumps(_scrub(bad), allow_nan=False) is not None,
        "scrubbed payload survives allow_nan=False")

    hdr = "date,away,home,away_SP,home_SP,away_score,home_score,status\n"
    with tempfile.TemporaryDirectory() as td:
        # header-only game_starters must return an empty frame, not crash. This is the
        # exact shape that produced the old KeyError: .dt.year.mode()[0] on no rows.
        p = os.path.join(td, "gs.csv")
        open(p, "w").write(hdr)
        try:
            s = M.load(p)
            chk(len(s) == 0, "header-only game_starters loads as empty, does not raise")
        except Exception as e:                                 # noqa: BLE001
            chk(False, f"header-only game_starters raised {type(e).__name__}: {e}")

        # season boundary: two seasons in one file must BOTH survive, and spring
        # training must be cut in each of them, not just the modal year.
        rows = [("2025-03-01", 0), ("2025-06-01", 1), ("2026-03-01", 0),
                ("2026-06-01", 1)]
        with open(p, "w") as f:
            f.write(hdr)
            for d, _ in rows:
                f.write(f"{d},AAA,BBB,SPa,SPh,3,5,Final\n")
        s = M.load(p)
        kept = sorted(str(x)[:10] for x in s["date"])
        chk(kept == ["2025-06-01", "2026-06-01"],
            f"both seasons kept, spring cut in each (got {kept})")

    # LIKELIHOOD-FIRST ORDERING. The board exists to answer "what hits", so the
    # ranking key is probability and nothing else. These pin that: both sides of a
    # game get a model number, and a fat edge on a longshot must NOT outrank a
    # likelier bet (the old stake-ordered board did exactly that).
    gms = [{"away": "A", "home": "B", "p_home": 40.0},
           {"away": "C", "home": "D", "p_home": 71.0}]
    mp = model_win_pct(gms)
    chk(mp[("A @ B", "A")] == 60.0 and mp[("A @ B", "B")] == 40.0 and
        mp[("C @ D", "D")] == 71.0, "model_win_pct gives every side its own win%")
    bets = [{"bet": "longshot", "p_model": 18.0, "fair": 0.20, "ev_pct": 22.0},
            {"bet": "likely",   "p_model": 66.0, "fair": 0.63, "ev_pct": 1.1},
            {"bet": "no model", "p_model": None, "fair": 0.44, "ev_pct": 9.0}]
    chk([b["bet"] for b in sorted(bets, key=by_likelihood)] ==
        ["likely", "no model", "longshot"],
        "edges rank by hit probability, not by edge (fair% when model is missing)")
    chk(sorted(gms, key=lambda g: -max(g["p_home"], 100 - g["p_home"]))[0]["home"] == "D",
        "slate leads with the game the model is most sure of")

    # the honest-confidence table must ship, and must refuse to print thin buckets
    import mlb_grade as _G
    rel = _G.reliability_for_publish()
    chk(isinstance(rel.get("buckets"), list) and rel.get("min_n", 0) >= 30,
        f"reliability table publishes with a >=30-row floor ({len(rel.get('buckets', []))} buckets)")
    chk(all(b["actual"] is None or b["n"] >= rel["min_n"] for b in rel.get("buckets", [])),
        "no bucket under the floor is allowed to show a hit rate")

    # the live level factor must be flat: it may move numbers, never the order
    import mlb_hr as H
    board = [42.0, 31.0, 25.5, 22.1, 22.0, 18.0, 9.0, 3.0]
    after = [H.live_level_pct(H.calibrate_pct(x)) for x in board]
    chk(after == sorted(after, reverse=True),
        "LIVE_LEVEL + anchors preserve board order (flat correction)")
    chk(0.5 <= H.LIVE_LEVEL <= 1.0,
        f"LIVE_LEVEL={H.LIVE_LEVEL} is a shrink toward zero, in range")

    print(f"\n{ok[0]}/{ok[1]} checks pass")
    return 0 if ok[0] == ok[1] else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
