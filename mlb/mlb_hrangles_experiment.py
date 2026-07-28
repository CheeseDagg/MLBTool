#!/usr/bin/env python3
"""
mlb_hrangles_experiment.py — four untested HR angles, walk-forward, leak-free.

QUESTION: the HR board is calibrated (21.4% predicted vs 19.9% actual through
2026-07-22) — so more accuracy must come from NEW signal. Four candidates the
model has never seen, each testable from 2025 boxscores alone:

  D  HANDED PARK   — park factor split by batter hand (short porches are not
                     symmetric; the production park factor is)
  C  INDIV PLATOON — each batter's own vs-hand split, regressed, instead of
                     one flat league platoon bump for everybody
  E  DAY/NIGHT     — a listed blind spot; league-wide day multiplier
  F  PITCHER HR    — opposing starter's HR-allowed rate (we grade his K/BB/FB
                     mix today, not how hard he actually gets barreled)

DATASET: every batter-game Apr 1 – Jun 30 2025 from statsapi schedule +
boxscores (park, day/night, slot, PA, HR, both hands, opposing starter).
Cached to data/hrangles_dataset.json by the Actions run — sandbox egress
can't reach statsapi, so offline this prints UNREACHABLE and exits 0.

METHOD: chronological single pass; every feature for a given DAY is computed
from state accumulated strictly BEFORE that day. Baseline mirrors production
shape: regressed batter HR/PA x park factor x flat platoon. Each variant adds
or swaps ONE factor. All shrinkage knobs tuned on TRAIN (Apr–May) only; the
verdict is June log-likelihood, with June split in three for robustness.
Metric: Bernoulli LL of "homered in game" with p = 1 - exp(-rate_pa * PA).

Selftest: planted handed-park effect recovered on synthetic data, null control
shows no win, and future-poisoning proves features are byte-identical when
later outcomes change. Run: python mlb_hrangles_experiment.py --selftest
"""
import json, math, os, sys, time, urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "hrangles_dataset.json")
START, END = "2025-04-01", "2025-06-30"
TRAIN_END = "2025-05-31"
API = "https://statsapi.mlb.com/api/v1"

# ---------------------------------------------------------------- dataset
def _get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return json.loads(r.read())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))

def _pa(bat):
    if bat.get("plateAppearances") not in (None, ""):
        return int(bat["plateAppearances"])
    return (int(bat.get("atBats", 0)) + int(bat.get("baseOnBalls", 0)) +
            int(bat.get("hitByPitch", 0)) + int(bat.get("sacFlies", 0)) +
            int(bat.get("sacBunts", 0)))

def build_dataset():
    sched = _get(f"{API}/schedule?sportId=1&startDate={START}&endDate={END}&gameType=R")
    games = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            games.append({"pk": g["gamePk"], "date": d["date"],
                          "venue": g.get("venue", {}).get("id"),
                          "dn": g.get("dayNight", "night")})
    print(f"schedule: {len(games)} final games {START}..{END}")
    rows, people = [], set()
    for i, g in enumerate(games):
        try:
            box = _get(f"{API}/game/{g['pk']}/boxscore")
        except Exception as e:
            print(f"  boxscore {g['pk']}: {type(e).__name__} — skipped")
            continue
        sides = {}
        for side in ("home", "away"):
            t = box["teams"][side]
            starter = None
            for pid in t.get("pitchers", []):
                st = t["players"].get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
                if int(st.get("gamesStarted", 0) or 0) >= 1:
                    starter = pid
                    break
            if starter is None and t.get("pitchers"):
                starter = t["pitchers"][0]
            sides[side] = starter
            if starter:
                people.add(starter)
        for side in ("home", "away"):
            t = box["teams"][side]
            opp_sp = sides["away" if side == "home" else "home"]
            for key, pl in t.get("players", {}).items():
                bat = pl.get("stats", {}).get("batting", {})
                if not bat:
                    continue
                pa = _pa(bat)
                if pa <= 0:
                    continue
                bo = str(pl.get("battingOrder", "") or "")
                slot = int(bo) // 100 if bo.isdigit() else 0
                pid = pl["person"]["id"]
                people.add(pid)
                rows.append({"date": g["date"], "pk": g["pk"], "venue": g["venue"],
                             "dn": g["dn"], "home": side == "home", "bat": pid,
                             "name": pl["person"].get("fullName", ""),
                             "slot": slot, "pa": pa,
                             "hr": int(bat.get("homeRuns", 0) or 0),
                             "sp": opp_sp})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(games)} boxscores, {len(rows)} rows")
        time.sleep(0.08)
    hands = {}
    ids = sorted(people)
    for j in range(0, len(ids), 100):
        chunk = ids[j:j + 100]
        try:
            data = _get(f"{API}/people?personIds={','.join(map(str, chunk))}")
            for p in data.get("people", []):
                hands[p["id"]] = {"bat": p.get("batSide", {}).get("code", ""),
                                  "throw": p.get("pitchHand", {}).get("code", "")}
        except Exception as e:
            print(f"  people chunk {j}: {type(e).__name__}")
        time.sleep(0.1)
    for r in rows:
        r["bh"] = hands.get(r["bat"], {}).get("bat", "")
        r["ph"] = hands.get(r["sp"], {}).get("throw", "")
    ds = {"start": START, "end": END, "rows": rows}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(ds, f)
    print(f"dataset cached -> {CACHE} ({len(rows)} batter-games)")
    return ds

# ---------------------------------------------------------------- features
def eff_hand(bh, ph):
    """Effective batter side this matchup (switch hitters take the platoon side)."""
    if bh == "S":
        return "L" if ph == "R" else "R"
    return bh or "R"

def attach_features(rows):
    """One chronological pass. Day-grouped: every feature uses only PRIOR days."""
    lg = [0, 0]                                   # hr, pa
    bat = defaultdict(lambda: [0, 0])
    bat_vs = defaultdict(lambda: [0, 0])          # (batter, pitcher-hand)
    park = defaultdict(lambda: [0, 0])
    park_h = defaultdict(lambda: [0, 0])          # (venue, batter-side)
    dn_c = defaultdict(lambda: [0, 0])            # 'day' / 'night'
    pit = defaultdict(lambda: [0, 0])             # HR allowed by starter
    plat = defaultdict(lambda: [0, 0])            # 'opp' / 'same' league-wide
    out = []
    for day in sorted({r["date"] for r in rows}):
        todays = [r for r in rows if r["date"] == day]
        for r in todays:
            h = eff_hand(r["bh"], r["ph"])
            opp = (h != r["ph"]) if r["ph"] else True
            f = {"lg": tuple(lg), "bat": tuple(bat[r["bat"]]),
                 "bvs": tuple(bat_vs[(r["bat"], r["ph"])]),
                 "park": tuple(park[r["venue"]]),
                 "parkh": tuple(park_h[(r["venue"], h)]),
                 "dn": tuple(dn_c[r["dn"]]),
                 "dn_other": tuple(dn_c["night" if r["dn"] == "day" else "day"]),
                 "pit": tuple(pit[r["sp"]]),
                 "plat_opp": tuple(plat["opp"]), "plat_same": tuple(plat["same"])}
            out.append((r, f))
        for r in todays:                          # update AFTER scoring the day
            h = eff_hand(r["bh"], r["ph"])
            opp = (h != r["ph"]) if r["ph"] else True
            for acc in (lg, bat[r["bat"]], bat_vs[(r["bat"], r["ph"])],
                        park[r["venue"]], park_h[(r["venue"], h)],
                        dn_c[r["dn"]], pit[r["sp"]],
                        plat["opp" if opp else "same"]):
                acc[0] += r["hr"]; acc[1] += r["pa"]
    return out

# ---------------------------------------------------------------- variants
def _rate(cell, lg_rate, tau):
    hr, pa = cell
    return (hr + tau * lg_rate) / (pa + tau) if (pa + tau) > 0 else lg_rate

def _factor(cell, ref_rate, tau_pa, w=1.0):
    """Shrunk ratio: 1 at no data, raw ratio at infinite data."""
    hr, pa = cell
    if ref_rate <= 0:
        return 1.0
    raw = ((hr + tau_pa * ref_rate) / (pa + tau_pa)) / ref_rate if (pa + tau_pa) > 0 else 1.0
    return 1.0 + w * (raw - 1.0)

def rate_pa(r, f, P, use):
    lg_hr, lg_pa = f["lg"]
    lgr = lg_hr / lg_pa if lg_pa > 200 else 0.031
    v = _rate(f["bat"], lgr, P["tau_b"])
    v *= _factor(f["park"], lgr, P["tau_park"])
    if "flat_platoon" in use and r["ph"]:
        h = eff_hand(r["bh"], r["ph"])
        cell = f["plat_opp"] if h != r["ph"] else f["plat_same"]
        v *= _factor(cell, lgr, 3000)
    if "indiv_platoon" in use and r["ph"]:
        b_rate = _rate(f["bat"], lgr, P["tau_b"])
        v *= _factor(f["bvs"], b_rate, P["tau_s"])
    if "handed_park" in use:
        pk_hr, pk_pa = f["park"]
        pkr = pk_hr / pk_pa if pk_pa > 200 else lgr
        v *= _factor(f["parkh"], pkr, P["tau_ph"])
    if "daynight" in use:
        v *= _factor(f["dn"], lgr, P["tau_dn"])
    if "pitcher_hr" in use:
        v *= _factor(f["pit"], lgr, P["tau_p"], P["w_p"])
    return v

def loglik(feat_rows, P, use, d0, d1):
    tot = n = 0.0
    for r, f in feat_rows:
        if not (d0 <= r["date"] <= d1):
            continue
        lam = max(rate_pa(r, f, P, use), 1e-6) * r["pa"]
        p = 1.0 - math.exp(-lam)
        p = min(max(p, 1e-9), 1 - 1e-9)
        y = 1 if r["hr"] > 0 else 0
        tot += y * math.log(p) + (1 - y) * math.log(1 - p)
        n += 1
    return tot / n if n else 0.0, int(n)

BASE_P = {"tau_b": 150, "tau_park": 1500, "tau_s": 200, "tau_ph": 1500,
          "tau_dn": 3000, "tau_p": 300, "w_p": 0.6}

def tune_and_verdict(feat_rows, out=print):
    d0, dT, dH0, dH1 = START, TRAIN_END, "2025-06-01", END
    # 1) baseline knobs on TRAIN
    best = (-9e9, None)
    for tb in (75, 150, 300):
        for tp in (800, 1500, 3000):
            P = dict(BASE_P, tau_b=tb, tau_park=tp)
            ll, _ = loglik(feat_rows, P, {"flat_platoon"}, d0, dT)
            if ll > best[0]:
                best = (ll, P)
    P0 = best[1]
    base_tr = best[0]
    out(f"baseline (bat x park x flat platoon) tuned: tau_b={P0['tau_b']} "
        f"tau_park={P0['tau_park']}  TRAIN LL {base_tr:+.5f}")
    variants = {
        "C indiv platoon": ({"flat_platoon", "indiv_platoon"},
                            [("tau_s", v) for v in (100, 200, 400, 800)]),
        "D handed park":   ({"flat_platoon", "handed_park"},
                            [("tau_ph", v) for v in (150, 400, 1000, 2500)]),
        "E day/night":     ({"flat_platoon", "daynight"},
                            [("tau_dn", v) for v in (1500, 3000, 6000)]),
        "F pitcher HR":    ({"flat_platoon", "pitcher_hr"},
                            [("w_p", v) for v in (0.3, 0.6, 1.0)]),
    }
    periods = [("2025-06-01", "2025-06-10"), ("2025-06-11", "2025-06-20"),
               ("2025-06-21", END)]
    base_h, n_h = loglik(feat_rows, P0, {"flat_platoon"}, dH0, dH1)
    out(f"baseline HOLDOUT LL {base_h:+.5f} (n={n_h})")
    results = {}
    for name, (use, grid) in variants.items():
        bt = (-9e9, None)
        for k, v in grid:
            P = dict(P0); P[k] = v
            ll, _ = loglik(feat_rows, P, use, d0, dT)
            if ll > bt[0]:
                bt = (ll, P)
        Pv = bt[1]
        train_win = bt[0] > base_tr
        hv, _ = loglik(feat_rows, Pv, use, dH0, dH1)
        wins = 0
        for p0, p1 in periods:
            b, _ = loglik(feat_rows, P0, {"flat_platoon"}, p0, p1)
            v_, _ = loglik(feat_rows, Pv, use, p0, p1)
            wins += 1 if v_ > b else 0
        verdict = ("ROBUST WIN" if (train_win and hv > base_h and wins == 3)
                   else ("win, not robust" if hv > base_h else "NULL"))
        results[name] = (hv - base_h, wins, verdict, Pv)
        out(f"{name:16s} train_win={train_win}  holdout dLL {hv-base_h:+.5f}  "
            f"periods {wins}/3  -> {verdict}")
    return results

# ---------------------------------------------------------------- selftest
def _synth(planted=True, seed=7):
    import random
    rng = random.Random(seed)
    parks = [(v, 1.0 + 0.25 * (v % 3 - 1)) for v in range(10)]
    rows = []
    day0 = 0
    for day in range(70):
        date = f"2025-{4 + day // 30:02d}-{day % 30 + 1:02d}"
        for g in range(8):
            venue, pf = parks[rng.randrange(10)]
            dn = "day" if rng.random() < 0.3 else "night"
            sp = 1000 + rng.randrange(60)
            ph = "L" if sp % 4 == 0 else "R"
            for b in range(9):
                bid = 100 + rng.randrange(200)
                bh = "L" if bid % 3 == 0 else "R"
                h = eff_hand(bh, ph)
                rate = 0.031 * pf * (1.0 + 0.004 * (bid % 5))
                if planted and h == "L" and venue < 3:
                    rate *= 2.2                       # planted handed-park pull (strong on purpose:
                    # the selftest proves the MACHINERY detects signal; magnitude is not the claim)
                pa = rng.choice((3, 4, 4, 5))
                lam = rate * pa
                hr = 1 if rng.random() < (1 - math.exp(-lam)) else 0
                rows.append({"date": date, "pk": day * 100 + g, "venue": venue,
                             "dn": dn, "home": g % 2 == 0, "bat": bid,
                             "name": f"B{bid}", "slot": b + 1, "pa": pa,
                             "hr": hr, "sp": sp, "bh": bh, "ph": ph})
    return rows

def selftest():
    global START, TRAIN_END, END
    START, TRAIN_END, END = "2025-04-01", "2025-05-15", "2025-06-06"
    # 1) planted handed-park effect must be recovered
    fr = attach_features(_synth(planted=True))
    buf = []
    res = tune_and_verdict(fr, out=lambda s: buf.append(s))
    d_planted = res["D handed park"][0]
    assert d_planted > 0.002, f"planted handed-park NOT recovered: {d_planted}"
    # 2) null control: no planted effect -> no big spurious win
    fr0 = attach_features(_synth(planted=False))
    res0 = tune_and_verdict(fr0, out=lambda s: None)
    assert res0["D handed park"][0] < d_planted / 3, "null control suspicious"
    # 3) leak-freeness: poison all outcomes after cutoff; earlier features identical
    rows_a = _synth(planted=True)
    rows_b = [dict(r) for r in rows_a]
    for r in rows_b:
        if r["date"] > "2025-05-01":
            r["hr"] = 1 - min(r["hr"], 1)
    fa = [(r["date"], f) for r, f in attach_features(rows_a) if r["date"] <= "2025-05-01"]
    fb = [(r["date"], f) for r, f in attach_features(rows_b) if r["date"] <= "2025-05-01"]
    assert json.dumps(fa, sort_keys=True) == json.dumps(fb, sort_keys=True), \
        "LEAK: features changed when future outcomes changed"
    print("HRANGLES SELFTEST PASS — planted handed-park recovered "
          f"(dLL {d_planted:+.4f}), null clean, features leak-free")
    return 0

# ---------------------------------------------------------------- main
def main():
    if os.path.exists(CACHE):
        ds = json.load(open(CACHE))
        print(f"dataset cache: {len(ds['rows'])} batter-games")
    else:
        try:
            urllib.request.urlopen(f"{API}/teams?sportId=1", timeout=10)
        except Exception:
            print("statsapi UNREACHABLE from this network — run on GitHub Actions "
                  "(touch experiments/RUN-HRANGLES.txt)")
            return 0
        ds = build_dataset()
    rows = [r for r in ds["rows"] if r.get("bh") and r.get("ph") and r.get("slot", 0) > 0]
    print(f"scorable rows (known hands, in lineup): {len(rows)}")
    feat = attach_features(rows)
    lines = []
    def tee(s):
        print(s); lines.append(s)
    tee("=" * 70)
    tee(f"HR ANGLES EXPERIMENT — {START}..{END} (train to {TRAIN_END})")
    tee("=" * 70)
    results = tune_and_verdict(feat, out=tee)
    tee("=" * 70)
    tee("VERDICT")
    for name, (d, wins, verdict, Pv) in sorted(results.items(), key=lambda x: -x[1][0]):
        tee(f"  {name:16s} {verdict:16s} holdout dLL {d:+.5f} ({wins}/3 periods) "
            f"params {{k: v for k, v in Pv.items()}}"
            .replace("{k: v for k, v in Pv.items()}",
                     str({k: v for k, v in Pv.items() if k in
                          ('tau_s', 'tau_ph', 'tau_dn', 'w_p', 'tau_b', 'tau_park')})))
    tee("Ship rule: ROBUST WIN only (train win + holdout win + 3/3 periods).")
    vd = os.path.join(HERE, "..", "experiments", "MLB-HRANGLES-VERDICT.md")
    os.makedirs(os.path.dirname(vd), exist_ok=True)
    with open(vd, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"verdict -> {vd}")
    return 0

if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
