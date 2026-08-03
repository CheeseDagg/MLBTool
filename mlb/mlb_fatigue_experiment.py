#!/usr/bin/env python3
"""
mlb_fatigue_experiment.py — HR angles batch 3: FATIGUE and FAMILIARITY.

Two ideas the board cannot currently see, both derivable from the panel it
already has, so both run locally in seconds with no Savant pull.

  REST     days since the batter's last game. Every-day players in the middle
           of a long stretch are supposed to lose bat speed; a man off a day
           of rest is supposed to have it back. If that is true it should show
           up in HR rate, which is the most bat-speed-dependent outcome there
           is.
  LOAD     plate appearances over the trailing 7 days, scored against the
           batter's own 30-day norm. Rest counts days; LOAD counts actual
           work. A catcher who started six straight and took 28 PA is carrying
           something a rest-day counter reads as zero.
  FAMIL    how many times this batter has already faced this STARTER this
           season. The times-through-the-order effect is well established
           within a game; this is the across-season version — the fourth look
           at a starter's arsenal should be worth more than the first.
  FAMCAR   the same count over the batter's whole career in the panel
           (2024 burn-in + 2025). Slower-moving, bigger sample per cell.

BASELINE is the shipped run-2 analog: batter rate x park x flat platoon x
prior-seeded pitcher HR factor. Anything tested here has to beat what already
ships, not a toy.

PROXY GUARDS, learned the hard way on the UFC side where win-streak momentum
turned out to be an age proxy and three wear-and-tear angles evaporated once
the age control was repaired:
  * REST is confounded with PLAYING TIME, which is confounded with QUALITY.
    Scrubs sit. So the rest term is fit on the SLOT-CONTROLLED sample and
    reported alongside a regulars-only pass (batters with >=300 panel PA),
    where "he rested" cannot mean "he is a bench bat".
  * FAMIL is confounded with CALENDAR. The Nth meeting necessarily happens
    later in the season, and league HR rate rises with the weather. So the
    familiarity count is scored against the league-wide mean count for that
    DATE, not in absolute terms.

Leak rules: every feature counts strictly PRIOR games. Selftest proves planted
recovery, a null control, and that no same-day or future game leaks in.
Ship rule: ROBUST WIN only — train win + holdout win + 3/3 sub-periods — and
anything under +0.0005 additionally has to clear the shuffled placebo.
"""
import json, math, os, sys, datetime as dt
from collections import defaultdict

import mlb_hrangles_experiment as HX

HERE = os.path.dirname(os.path.abspath(__file__))
START, TRAIN_END, END = HX.START, HX.TRAIN_END, HX.END
PERIODS = [("2025-06-01", "2025-06-10"), ("2025-06-11", "2025-06-20"),
           ("2025-06-21", END)]
P0 = dict(HX.BASE_P, tau_b=75, tau_park=800, w_p=0.6)
BASE_USE = {"flat_platoon", "pitcher_hr"}
REGULAR_PA = 300


# ------------------------------------------------------------ prior-only state
def season_of(r):
    """The season a row belongs to.

    The panel rows carry no explicit season/year column, only an ISO `date`.
    An MLB season never straddles a calendar-year boundary, so the leading
    four characters ARE the season — no month heuristics, and no need to know
    where opening day fell in a given year. Kept as a named function so every
    season-scoped accumulator below resets off the same definition.
    """
    return r["date"][:4]


def build_history(rows):
    """For each row, features computed from STRICTLY EARLIER games only.

    Returns {row_id: {...}} keyed by id(row) is fragile across processes, so
    this instead returns a parallel list in the same order as `rows`.

    Everything here is scoped to a SEASON, because this panel is a 2024
    burn-in bolted onto 2025 and the winter between them is not baseball. A
    raw calendar gap across that boundary reads as ~180 days of rest, i.e. the
    most rested a batter can possibly be, on the opening rows of a season —
    which is where the entirety of April 2025 lives. That is not an extreme
    observation of the thing this file measures, it is a different thing
    wearing its clothes, so those rows get NO opinion rather than a big number.
    """
    rows = sorted(rows, key=lambda r: (r["date"], r.get("pk", 0)))
    last_game = {}                       # (bat, season) -> last date played
    recent = defaultdict(list)           # (bat, season) -> [(date, pa)]
    seen_sp_season = defaultdict(int)    # (bat, sp, season) -> meetings so far
    seen_sp_career = defaultdict(int)    # (bat, sp) -> meetings so far, all
    total_pa = defaultdict(int)          # bat -> panel PA so far
    out = []
    for r in rows:
        b, sp, d = r["bat"], r.get("sp"), r["date"]
        yr = season_of(r)
        dd = dt.date.fromisoformat(d)
        # keyed by (batter, season): a batter's first appearance OF A SEASON
        # finds nothing here and so has no defined prior gap, exactly like his
        # first appearance in the panel. Dropping the row from the feature is
        # the honest answer; capping the gap would still let it vote, and vote
        # at the maximum, on the least informative rows in the panel.
        prev = last_game.get((b, yr))
        rest = (dd - dt.date.fromisoformat(prev)).days if prev else None
        # trailing 7d / 30d PA, strictly before today. The date filters alone
        # would already exclude last season (the offseason is far longer than
        # 30 days), but the accumulator is season-keyed anyway so that the
        # window can be widened later without silently reaching over a winter.
        w7 = sum(p for dt_, p in recent[(b, yr)]
                 if 0 < (dd - dt.date.fromisoformat(dt_)).days <= 7)
        w30 = sum(p for dt_, p in recent[(b, yr)]
                  if 0 < (dd - dt.date.fromisoformat(dt_)).days <= 30)
        out.append({
            "rest": rest,
            "pa7": w7,
            "pa30": w30,
            # keyed on the season itself rather than a hardcoded "is it 2025
            # yet" date: the old form pinned every burn-in row to fam=0, so
            # the within-season count was constant by construction on a third
            # of the panel and could never be scored there.
            "fam": seen_sp_season[(b, sp, yr)],
            "famcar": seen_sp_career[(b, sp)],
            "prior_pa": total_pa[b],
        })
        # advance state AFTER emitting — this is the leak boundary
        last_game[(b, yr)] = d
        recent[(b, yr)].append((d, r["pa"]))
        if len(recent[(b, yr)]) > 60:
            recent[(b, yr)] = recent[(b, yr)][-60:]
        if sp is not None:
            seen_sp_season[(b, sp, yr)] += 1
            seen_sp_career[(b, sp)] += 1
        # NOT season-scoped, on purpose: prior_pa is the "is this a regular or
        # a bench bat" gate, and a man's standing carries over the winter.
        total_pa[b] += r["pa"]
    return rows, out


def date_means(rows, hist, key):
    """League mean of `key` per DATE — the calendar control for familiarity.
    The Nth meeting can only happen later in the year, and league HR rate
    climbs with the weather, so an uncontrolled count would price June."""
    acc = defaultdict(lambda: [0.0, 0])
    for r, h in zip(rows, hist):
        v = h.get(key)
        if v is None:
            continue
        cell = acc[r["date"]]
        cell[0] += v
        cell[1] += 1
    return {d: (s / n if n else 0.0) for d, (s, n) in acc.items()}


# ------------------------------------------------------------ variant factors
def factor(h, r, kind, P, dmeans):
    """Multiplier on the baseline PA rate. 1.0 means 'no opinion'."""
    if kind == "rest":
        v = h["rest"]
        if v is None:
            return 1.0
        # 1 day = played yesterday (the grind). cap at 5: a 12-day gap is an
        # injury return, which is a different animal and is not what this tests
        v = min(v, 5)
        return 1.0 + P["w"] * (v - 1.0) / 4.0

    if kind == "load":
        if h["pa30"] < 40:
            return 1.0
        norm = h["pa30"] / 30.0 * 7.0            # expected trailing-7d PA
        if norm <= 0:
            return 1.0
        z = (h["pa7"] - norm) / max(norm, 1.0)
        return 1.0 + P["w"] * max(min(z, 1.0), -1.0)

    if kind in ("fam", "famcar"):
        v = h[kind]
        mean = dmeans.get(r["date"], 0.0)
        return 1.0 + P["w"] * (v - mean) / 3.0

    return 1.0


def loglik_var(feat_rows, hist, kind, P, d0, d1, subset=None):
    tot = n = 0.0
    for (r, f), h in zip(feat_rows, hist):
        if not (d0 <= r["date"] <= d1):
            continue
        if subset == "regulars" and h["prior_pa"] < REGULAR_PA:
            continue
        lam = max(HX.rate_pa(r, f, P0, BASE_USE), 1e-6)
        if kind:
            lam *= factor(h, r, kind, P, P["_dm"])
        lam = max(lam, 1e-6) * r["pa"]
        p = min(max(1.0 - math.exp(-lam), 1e-9), 1 - 1e-9)
        y = 1 if r["hr"] > 0 else 0
        tot += y * math.log(p) + (1 - y) * math.log(1 - p)
        n += 1
    return (tot / n if n else 0.0), int(n)


# ------------------------------------------------------------ experiment
W_GRID = (-0.40, -0.30, -0.20, -0.12, -0.06, -0.03,
          0.03, 0.06, 0.12, 0.20, 0.30, 0.40)


def experiment(feat_rows, hist, dmeans, out=print, subset=None):
    base_tr, ntr = loglik_var(feat_rows, hist, None, {"_dm": dmeans},
                              START, TRAIN_END, subset)
    base_h, nh = loglik_var(feat_rows, hist, None, {"_dm": dmeans},
                            "2025-06-01", END, subset)
    base_p = [loglik_var(feat_rows, hist, None, {"_dm": dmeans}, a, b, subset)[0]
              for a, b in PERIODS]
    tag = subset or "all"
    out(f"baseline [{tag}]  TRAIN {base_tr:+.5f} (n={ntr})  "
        f"HOLDOUT {base_h:+.5f} (n={nh})")
    res = {}
    for kind, label in (("rest", "REST    days off"),
                        ("load", "LOAD    7d PA vs own norm"),
                        ("fam", "FAMIL   Nth look, season"),
                        ("famcar", "FAMCAR  Nth look, career")):
        best = (-9e9, None)
        for w in W_GRID:
            P = {"w": w, "_dm": dmeans}
            ll, _ = loglik_var(feat_rows, hist, kind, P, START, TRAIN_END, subset)
            if ll > best[0]:
                best = (ll, w)
        ll_tr, w = best
        P = {"w": w, "_dm": dmeans}
        hv, _ = loglik_var(feat_rows, hist, kind, P, "2025-06-01", END, subset)
        wins = sum(1 for i, (a, b) in enumerate(PERIODS)
                   if loglik_var(feat_rows, hist, kind, P, a, b, subset)[0] > base_p[i])
        train_win = ll_tr > base_tr
        verdict = ("ROBUST WIN" if (train_win and hv > base_h and wins == 3)
                   else ("win, not robust" if hv > base_h else "NULL"))
        res[kind] = (hv - base_h, wins, verdict, w)
        out(f"{label:26s} w={w:+.2f}  train_win={str(train_win):5s}  "
            f"holdout dLL {hv-base_h:+.5f}  periods {wins}/3  -> {verdict}")
    return res


# ------------------------------------------------------------ selftest
def selftest():
    import random
    rng = random.Random(5)
    rows = []
    d0 = dt.date(2025, 4, 1)
    # 120 batters, each plays ~5 of every 6 days, faces a rotating starter set.
    # PLANT: HR rate scales by exactly the factor form the harness fits, at
    # w=+0.40 — i.e. a fully rested bat homers 40% more than one playing its
    # fifth straight day. That is a deliberately generous but not absurd
    # effect. The point of this test is "can the harness recover an effect it
    # is parameterized to see", so the plant uses the same functional form; a
    # plant of a shape the model cannot express tests the grid, not the code.
    for b in range(120):
        srng = random.Random(b * 131 + 7)
        d = d0
        while d <= dt.date(2025, 6, 30):
            if srng.random() < 0.83:
                rows.append({"date": d.isoformat(), "pk": b + d.toordinal() * 500,
                             "venue": b % 8, "dn": "night", "home": b % 2 == 0,
                             "bat": b, "name": f"B{b}", "slot": b % 9 + 1,
                             "pa": 4, "hr": 0, "sp": 7000 + (b + d.toordinal()) % 40,
                             "bh": "R", "ph": "R"})
            d += dt.timedelta(days=1)
    rows, hist = build_history(rows)
    for r, h in zip(rows, hist):
        srng = random.Random(r["pk"] * 13 + 1)
        v = min(h["rest"] or 1, 5)
        boost = 1.0 + 0.40 * (v - 1.0) / 4.0
        p_pa = 0.031 * boost
        r["hr"] = 1 if srng.random() < 1 - (1 - p_pa) ** r["pa"] else 0
    feat = HX.attach_features(rows)
    dm = date_means(rows, hist, "fam")
    b_ll, n = loglik_var(feat, hist, None, {"_dm": dm}, START, END)
    v_ll, _ = loglik_var(feat, hist, "rest", {"w": 0.40, "_dm": dm}, START, END)
    d_rest = v_ll - b_ll
    # v_ll is scored at the TRUE planted w, so d_rest is the ORACLE ceiling:
    # the most any model of this shape could ever gain on this panel. It is
    # only ~+0.0003 — not because the code is weak but because rest barely
    # varies. ~83% of games follow a game, so the factor is 1.0 on most rows
    # and there is almost nothing for it to grade. That is the honest ceiling
    # for this angle, and it is why main() computes a power ceiling per angle
    # before reporting any null: a null under a low ceiling means "we cannot
    # see it", which is a different claim from "it is not there".
    assert d_rest > 0.0002, f"planted rest effect NOT recovered: {d_rest}"

    # null control: rest values permuted across rows -> the factor is noise
    shuf = [dict(h) for h in hist]
    vals = [h["rest"] for h in shuf]
    rng.shuffle(vals)
    for h, v in zip(shuf, vals):
        h["rest"] = v
    n_ll, _ = loglik_var(feat, shuf, "rest", {"w": 0.40, "_dm": dm}, START, END)
    d_null = n_ll - b_ll
    assert d_null < d_rest / 3, f"null control suspicious: {d_null} vs {d_rest}"

    # leak proof: flip every outcome after a cutoff on a deep copy; features
    # for earlier rows must be byte-identical. build_history must not peek.
    cut = "2025-05-15"
    poisoned = [dict(r) for r in rows]
    for r in poisoned:
        if r["date"] > cut:
            r["hr"] = 1 - r["hr"]
            r["pa"] = 9
    _, h2 = build_history(poisoned)
    a = json.dumps([h for h, r in zip(hist, rows) if r["date"] <= cut],
                   sort_keys=True)
    b = json.dumps([h for h, r in zip(h2, sorted(poisoned, key=lambda r: (r["date"], r.get("pk", 0))))
                    if r["date"] <= cut], sort_keys=True)
    assert a == b, "FUTURE LEAK: pre-cutoff features changed when later games changed"

    # rest itself must never see today or later
    assert all(h["rest"] is None or h["rest"] > 0 for h in hist)

    # SEASON BOUNDARY. The panel is a 2024 burn-in plus 2025, so the regression
    # this pins is a real one that shipped: the winter used to be measured as
    # ~180 days of rest, which put every season-opening row in the top bucket.
    tmpl = {"venue": 1, "dn": "night", "home": True, "bat": 1, "name": "B",
            "slot": 3, "pa": 4, "hr": 0, "sp": 900, "bh": "R", "ph": "R"}
    winter = [dict(tmpl, date="2024-09-28", pk=1),
              dict(tmpl, date="2025-04-01", pk=2),
              dict(tmpl, date="2025-04-03", pk=3)]
    _s, hw = build_history(winter)
    assert hw[0]["rest"] is None, "first panel row invented a gap"
    assert hw[1]["rest"] is None, f"offseason scored as rest: {hw[1]['rest']}"
    assert hw[2]["rest"] == 2, hw[2]              # within-season still counts
    # the trailing workload window must not reach back over the winter either
    assert hw[1]["pa30"] == 0, f"pa30 crossed the offseason: {hw[1]['pa30']}"
    assert hw[2]["pa30"] == 4, hw[2]
    # within-season familiarity RESETS in the new year; career does not
    assert [h["fam"] for h in hw] == [0, 0, 1], [h["fam"] for h in hw]
    assert [h["famcar"] for h in hw] == [0, 1, 2], [h["famcar"] for h in hw]
    # prior_pa is deliberately NOT season-scoped — it is the regulars gate
    assert hw[1]["prior_pa"] == 4, hw[1]

    print(f"FATIGUE SELFTEST PASS — planted rest recovered (oracle dLL "
          f"{d_rest:+.5f}, null {d_null:+.5f}), leak-free across a poisoned "
          f"cutoff, offseason excluded from rest/workload/familiarity")
    return 0


# ------------------------------------------------------------ power ceiling
def power_ceiling(feat_rows, hist, dmeans, kind, w, seed=99):
    """How much LL could this angle possibly buy, if it were REAL and this big?

    Takes the actual panel — real rest gaps, real starter meetings, real park
    and platoon — and re-rolls only the HR outcomes so the angle is true at
    strength w. Then scores at the true w. The gap over baseline is the
    ceiling: no fitted model can do better, because this one already knows the
    answer. Comparing a measured null against its ceiling is the difference
    between 'this angle is dead' and 'this panel cannot see it either way'.
    """
    import random
    rows = [dict(r) for (r, f) in feat_rows]
    P = {"w": w, "_dm": dmeans}
    for i, (r, h) in enumerate(zip(rows, hist)):
        (r0, f0) = feat_rows[i]
        lam = max(HX.rate_pa(r0, f0, P0, BASE_USE), 1e-6) * factor(h, r0, kind, P, dmeans)
        p = min(max(1.0 - math.exp(-lam * r["pa"]), 1e-9), 1 - 1e-9)
        r["hr"] = 1 if random.Random(seed * 1_000_003 + i).random() < p else 0
    synth = [(rows[i], feat_rows[i][1]) for i in range(len(rows))]
    b, _ = loglik_var(synth, hist, None, {"_dm": dmeans}, "2025-06-01", END)
    v, _ = loglik_var(synth, hist, kind, P, "2025-06-01", END)
    return v - b


# ------------------------------------------------------------ main
def main():
    if not (os.path.exists(HX.CACHE) and os.path.exists(HX.BURN_CACHE)):
        print("hrangles dataset caches missing")
        return 0
    ds = json.load(open(HX.CACHE))
    burn = json.load(open(HX.BURN_CACHE))
    raw = [r for r in burn["rows"] + ds["rows"]
           if r.get("bh") and r.get("ph") and r.get("slot", 0) > 0]
    rows, hist = build_history(raw)
    feat = HX.attach_features(rows)
    dm = date_means(rows, hist, "fam")
    lines = []

    def tee(s):
        print(s)
        lines.append(s)

    known_rest = sum(1 for h in hist if h["rest"] is not None)
    fam_any = sum(1 for h in hist if h["fam"] > 0)
    # printed because it is the size of the old bug: these are the rows that
    # used to carry a ~180-day "rest" and are now silent.
    openers = sum(1 for h in hist if h["rest"] is None) - len(
        {r["bat"] for r in rows})
    tee("=" * 70)
    tee("MLB HR ANGLES 3 — FATIGUE (rest, workload) and FAMILIARITY (Nth look)")
    tee("baseline = shipped run-2 analog incl. prior-seeded pitcher HR")
    tee("=" * 70)
    tee(f"rows {len(rows)}  rest known {known_rest} ({100*known_rest/len(rows):.0f}%)  "
        f"repeat-starter meetings {fam_any} ({100*fam_any/len(rows):.0f}%)")
    tee(f"season-opening rows with no defined prior gap: {openers} "
        f"(these used to score as ~180 days of rest)")
    tee("")
    tee("--- FULL SAMPLE")
    experiment(feat, hist, dm, out=tee)
    tee("")
    tee("--- REGULARS ONLY (>=300 prior panel PA) — the proxy guard.")
    tee("    On the full sample 'rested' partly means 'bench bat', and bench")
    tee("    bats homer less because they are worse, not because they rested.")
    experiment(feat, hist, dm, out=tee, subset="regulars")
    tee("")
    tee("--- POWER CEILING — how big could each angle possibly read?")
    tee("    Real panel, HR outcomes re-rolled so the angle IS true at a")
    tee("    generous strength, then scored at the true value. No fitted model")
    tee("    can beat this. A null far below its ceiling is uninformative.")
    for kind, w, label in (("rest", 0.40, "REST    +/-40%"),
                           ("load", 0.40, "LOAD    +/-40%"),
                           ("fam", 0.30, "FAMIL   +/-30%/look"),
                           ("famcar", 0.30, "FAMCAR  +/-30%/look")):
        c = power_ceiling(feat, hist, dm, kind, w)
        tee(f"    {label:22s} ceiling {c:+.5f} LL/game")
    tee("")
    tee("Ship rule: ROBUST WIN (train + holdout + 3/3), and any margin under")
    tee("+0.0005 must additionally clear a shuffled placebo before it counts.")
    vd = os.path.join(HERE, "..", "experiments", "MLB-FATIGUE-VERDICT.md")
    os.makedirs(os.path.dirname(vd), exist_ok=True)
    with open(vd, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"verdict -> {vd}")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
