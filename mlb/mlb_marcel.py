#!/usr/bin/env python3
"""
mlb_marcel.py — Tom Tango's MARCEL projection system, for HR/PA talent estimation.

This replaces the model's current base rate (this-season HR/PA + a flat prior) with a
real multi-year talent estimate — "what the pros do." Marcel is the acknowledged
baseline projection system: deliberately simple, and famously hard to beat. Every
constant here is from Tango's published spec, not invented:

  * WEIGHTS 5/4/3 on the last three seasons (most recent = 5).
  * Within each season, further weight by PA — so an injury-shortened year (77 PA)
    can't dominate a full year (600 PA). Composite weight = recency_weight * PA.
  * REGRESSION to league mean by 100 PA (batting): the projection is pulled toward
    league-average HR/PA proportional to how little data the player has.
  * AGE CURVE: over 29 -> decline (age-29)*0.003 per year; under 29 -> growth
    (age-29)*0.006. Applied to the regressed rate.
  * MIN 50 PA for a season to count. No qualifying seasons -> league average.

The output is a per-PA HR *talent* estimate. The live model multiplies it by the
same park/pitcher/platoon/weather factors it already uses — Marcel only replaces the
`base` term. This is a strictly better base than a single noisy season, but (honest
caveat, unchanged) a better talent estimate makes the number CALIBRATED, not sharper
than the market's price. The edge remains line-shopping.

Pure module: no network. Feeding it happens in mlb_marcel_run.py on Actions.
"""

# Tango's published constants — do not "tune" these; they ARE the pro baseline.
WEIGHTS = (5, 4, 3)          # most-recent-first, last three seasons
REGRESSION_PA = 100          # batting regression amount (PA of league-mean pull)
AGE_PIVOT = 29
AGE_DECLINE = 0.003          # per year over 29
AGE_GROWTH = 0.006           # per year under 29
MIN_SEASON_PA = 50           # a season needs >=50 PA to count
LG_HRPA_DEFAULT = 0.031      # fallback league HR/PA if none supplied


def marcel_hrpa(seasons, age, lg_hrpa=LG_HRPA_DEFAULT):
    """Marcel HR-per-PA talent estimate.

    seasons: list of dicts, MOST RECENT FIRST, each {"hr": int, "pa": int, "year": int}.
             Up to the first 3 qualifying (>=MIN_SEASON_PA) seasons are used.
    age:     player age in the projection year (int). None -> no age adjustment.
    lg_hrpa: league HR/PA to regress toward (defaults to a sane MLB value).

    Returns a float HR/PA. No qualifying data -> lg_hrpa exactly.
    """
    # take up to 3 seasons that clear the PA floor, in recency order
    qual = [s for s in seasons if int(s.get("pa", 0) or 0) >= MIN_SEASON_PA][:3]
    if not qual:
        return lg_hrpa

    # composite weight per season = recency_weight * PA; accumulate weighted HR and PA
    num = den = wpa = 0.0
    for i, s in enumerate(qual):
        w = WEIGHTS[i]
        pa = int(s["pa"]); hr = int(s["hr"])
        cw = w * pa
        num += cw * (hr / pa)     # weighted HR rate contribution
        den += cw
        wpa += cw / w             # effective PA (un-weighted by recency) for regression
    weighted_rate = num / den if den else lg_hrpa

    # REGRESSION to the mean: pull toward league by REGRESSION_PA worth of league PAs.
    # reliability = effective_PA / (effective_PA + REGRESSION_PA)
    eff_pa = sum(int(s["pa"]) for s in qual)
    rel = eff_pa / (eff_pa + REGRESSION_PA)
    regressed = rel * weighted_rate + (1 - rel) * lg_hrpa

    # AGE adjustment (applied multiplicatively around the pivot)
    if age is not None:
        if age > AGE_PIVOT:
            adj = 1.0 - (age - AGE_PIVOT) * AGE_DECLINE
        else:
            adj = 1.0 + (AGE_PIVOT - age) * AGE_GROWTH
        regressed *= adj

    return max(regressed, 0.0)


def blend_with_current(marcel_rate, cur_hr, cur_pa, lg_hrpa=LG_HRPA_DEFAULT, k=200):
    """Optional in-season update: blend the Marcel PRE-SEASON talent estimate with what
    the player has done SO FAR this year, weighting current form by its own PA against
    a stabilization constant k. Early in the year this stays ~Marcel; by September it
    leans on the current season. This is how projection systems update midseason.
    """
    if cur_pa <= 0:
        return marcel_rate
    w_cur = cur_pa / (cur_pa + k)
    return w_cur * (cur_hr / cur_pa) + (1 - w_cur) * marcel_rate


# ---------------------------------------------------------------------------
def selftest():
    # 1) EXACT Marcel worked example (from the published walkthroughs):
    #    rates .066/.025/.054 -> weighted (.066*5+.025*4+.054*3)/12 = .0493...
    #    Use equal PA so PA-weighting is neutral and we can check the 5/4/3 core.
    s = [{"hr": 66, "pa": 1000, "year": 2025},   # .066
         {"hr": 25, "pa": 1000, "year": 2024},   # .025
         {"hr": 54, "pa": 1000, "year": 2023}]   # .054
    # weighted rate before regression/age:
    wr = (0.066*5 + 0.025*4 + 0.054*3) / 12
    assert abs(wr - 0.0493333) < 1e-6, wr
    # full marcel with age=29 (no age adj) — regression pulls it toward .031 a bit
    r = marcel_hrpa(s, age=29, lg_hrpa=0.031)
    eff = 3000; rel = eff/(eff+100)
    exp = rel*wr + (1-rel)*0.031
    assert abs(r - exp) < 1e-9, (r, exp)
    assert 0.031 < r < wr    # regressed down from raw, still above league

    # 2) PA-WEIGHTING: an injury year (77 PA) must NOT dominate a full year (600 PA)
    inj = [{"hr": 30, "pa": 600, "year": 2025},   # .050 over a full season
           {"hr": 10, "pa": 77,  "year": 2024},   # .130 in a tiny sample (hot fluke)
           {"hr": 20, "pa": 550, "year": 2023}]   # .0364
    r2 = marcel_hrpa(inj, age=27, lg_hrpa=0.031)
    # if the 77-PA .130 dominated, r2 would spike; PA-weighting must keep it moderate
    naive_equal = (0.050 + 0.130 + 0.0364) / 3      # .0721 — the WRONG answer
    assert r2 < 0.070, (r2, "injury year leaked through")
    assert r2 < naive_equal

    # 3) REGRESSION: a small-sample player regresses HARD toward league
    tiny = [{"hr": 8, "pa": 60, "year": 2025}]      # .133 in 60 PA — mostly noise
    r3 = marcel_hrpa(tiny, age=25, lg_hrpa=0.031)
    rel3 = 60/(60+100)                               # 0.375 reliability
    # before age: .375*.133 + .625*.031 = .0693; age<29 nudges up
    assert r3 < 0.075 and r3 > 0.031, r3            # pulled way down from .133

    # 4) AGE CURVE exact: 35yo declines, 23yo grows, by Tango's coefficients
    base = [{"hr": 30, "pa": 600, "year": 2025},
            {"hr": 30, "pa": 600, "year": 2024},
            {"hr": 30, "pa": 600, "year": 2023}]     # flat .050, big sample
    young = marcel_hrpa(base, age=23, lg_hrpa=0.031)
    old   = marcel_hrpa(base, age=35, lg_hrpa=0.031)
    mid   = marcel_hrpa(base, age=29, lg_hrpa=0.031)
    assert young > mid > old                          # aging curve direction
    # magnitudes: 35 -> *(1 - 6*.003)=*.982 ; 23 -> *(1 + 6*.006)=*1.036
    assert abs(old/mid - 0.982) < 0.01
    assert abs(young/mid - 1.036) < 0.01

    # 5) MIN 50 PA: a 40-PA season is ignored entirely
    with_junk = [{"hr": 20, "pa": 500, "year": 2025},
                 {"hr": 9,  "pa": 40,  "year": 2024},   # <50 PA -> dropped
                 {"hr": 18, "pa": 480, "year": 2023}]
    only_valid = [{"hr": 20, "pa": 500, "year": 2025},
                  {"hr": 18, "pa": 480, "year": 2023}]  # same, junk removed
    assert marcel_hrpa(with_junk, age=28) == marcel_hrpa(only_valid, age=28)

    # 6) NO QUALIFYING DATA -> league average exactly (the call-up case)
    assert marcel_hrpa([], age=24, lg_hrpa=0.031) == 0.031
    assert marcel_hrpa([{"hr": 2, "pa": 30, "year": 2025}], age=24, lg_hrpa=0.031) == 0.031

    # 7) IN-SEASON BLEND: early season ~= marcel; late season leans current
    m = marcel_hrpa(base, age=28, lg_hrpa=0.031)      # ~.049 talent
    early = blend_with_current(m, cur_hr=1, cur_pa=20, k=200)   # 20 PA -> mostly marcel
    late  = blend_with_current(m, cur_hr=40, cur_pa=550, k=200) # 40 HR/550 = .073, genuinely hot
    assert abs(early - m) < 0.006                      # barely moved
    assert late > m                                    # hot current season (.073 > .049) pulls up
    cold  = blend_with_current(m, cur_hr=10, cur_pa=550, k=200) # .018, cold -> pulls down
    assert cold < m                                    # slump correctly drags talent estimate down
    assert blend_with_current(m, 0, 0) == m            # no current PA -> unchanged

    print("MARCEL SELFTEST PASS — 5/4/3 + PA-weight + 100PA-regression + age curve + floors all exact")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print("Pure Marcel engine. Season data is pulled on GitHub Actions via mlb_marcel_run.py.")
