#!/usr/bin/env python3
"""
mlb_kprops.py — pitcher strikeout props engine (pure).

Projects a starter's strikeout DISTRIBUTION for tonight, then prices any O/U line.

The mean is built from three honest ingredients:
  1. PITCHER K RATE — his K/9, regressed toward league average (we add a
     stabilizer of league-rate innings so a 20-inning hot streak doesn't
     masquerade as an elite rate; same philosophy as the Marcel hitter base).
  2. EXPECTED WORKLOAD — his innings per start this season (how deep he goes
     is as important as how hard he strikes people out).
  3. OPPONENT WHIFF PROPENSITY — the team he faces, as a multiplier of league
     K%. A free-swinging lineup genuinely inflates K props; a contact team
     deflates them. This is the single biggest matchup input for K props.

Distribution: Poisson at that mean. Strikeout totals are approximately Poisson
(slightly under-dispersed in reality, which makes Poisson mildly conservative
at the tails — the safe direction for pricing).

Pure module — the Actions runner feeds real stats and odds.
"""
import math

LEAGUE_K9 = 8.6          # league-average starter K/9 (refresh seasonally)
LEAGUE_IP_START = 5.3    # league-average innings per start
REGRESS_IP = 35.0        # innings of league-average added to the pitcher's rate
                         # (rate stabilizer; ~6 starts' worth)

def regressed_k9(pitcher_so, pitcher_ip, league_k9=LEAGUE_K9, regress_ip=REGRESS_IP):
    """Pitcher K/9 regressed toward league. Small samples pull hard to league;
    a full season barely moves."""
    if pitcher_ip <= 0:
        return league_k9
    league_so = league_k9 / 9.0 * regress_ip
    return 9.0 * (pitcher_so + league_so) / (pitcher_ip + regress_ip)

def expected_ip(season_ip, games_started, cap=(3.0, 7.0)):
    """Expected innings tonight = his average IP/start, sanity-capped."""
    if games_started <= 0:
        return LEAGUE_IP_START
    ip = season_ip / games_started
    return max(cap[0], min(cap[1], ip))

def opp_k_multiplier(team_so, team_pa, league_k_pct=0.222):
    """Opponent strikeout propensity vs league. >1 = whiffy lineup (more Ks),
    <1 = contact lineup. Shrunk 50% toward 1.0 — team K% is real signal but
    lineup-dependent night to night."""
    if team_pa <= 0:
        return 1.0
    raw = (team_so / team_pa) / league_k_pct
    return 1.0 + 0.5 * (raw - 1.0)

def k_lambda(pitcher_so, pitcher_ip, games_started, team_so, team_pa):
    """Expected strikeouts tonight."""
    k9 = regressed_k9(pitcher_so, pitcher_ip)
    ip = expected_ip(pitcher_ip, games_started)
    mult = opp_k_multiplier(team_so, team_pa)
    return (k9 / 9.0) * ip * mult

def _pois_cdf(k, lam):
    """P(X <= k) for Poisson(lam)."""
    if k < 0:
        return 0.0
    term = math.exp(-lam)
    total = term
    for i in range(1, int(k) + 1):
        term *= lam / i
        total += term
    return min(1.0, total)

def p_over(line, lam):
    """P(strikeouts > line). For half-lines (6.5): P(K >= 7). For whole lines
    (7.0): a push is possible; this returns P(K >= 8) i.e. the OVER-wins prob,
    with p_push exposed separately."""
    return 1.0 - _pois_cdf(math.floor(line), lam)

def p_push(line, lam):
    """Push probability on whole-number lines (0 for half lines)."""
    if abs(line - round(line)) > 1e-9:
        return 0.0
    k = int(round(line))
    return _pois_cdf(k, lam) - _pois_cdf(k - 1, lam)

def prob_to_american(p):
    if p <= 0: return 100000
    if p >= 1: return -100000
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))

def price_lines(lam, lines=(4.5, 5.5, 6.5, 7.5, 8.5)):
    """Fair prices for the standard K-prop ladder at this lambda."""
    out = []
    for ln in lines:
        po = p_over(ln, lam)
        out.append({"line": ln, "p_over": round(po, 4),
                    "fair_over": prob_to_american(po),
                    "fair_under": prob_to_american(1.0 - po - p_push(ln, lam))})
    return out

def project_start(name, pitcher_so, pitcher_ip, games_started, team_so, team_pa,
                  opp_name=""):
    """Full projection card for one start."""
    lam = k_lambda(pitcher_so, pitcher_ip, games_started, team_so, team_pa)
    return {
        "pitcher": name, "opp": opp_name,
        "lam": round(lam, 2),
        "k9_regressed": round(regressed_k9(pitcher_so, pitcher_ip), 2),
        "exp_ip": round(expected_ip(pitcher_ip, games_started), 2),
        "opp_mult": round(opp_k_multiplier(team_so, team_pa), 3),
        "lines": price_lines(lam),
    }


# ---------------------------------------------------------------------------
def selftest():
    # --- Poisson CDF exact against known values ---
    # Poisson(5): P(X<=5) = 0.615960..., P(X<=3) = 0.265026...
    assert abs(_pois_cdf(5, 5.0) - 0.6160) < 1e-3
    assert abs(_pois_cdf(3, 5.0) - 0.2650) < 1e-3
    # P(over 5.5) at lam=5: 1 - P(<=5) = 0.3840
    assert abs(p_over(5.5, 5.0) - 0.3840) < 1e-3

    # --- regression pulls small samples to league, leaves big ones alone ---
    # 15 IP of 12 K/9 (20 SO): should sit well below 12, well above league
    hot_small = regressed_k9(20, 15)          # 15 IP at 12 K/9
    assert LEAGUE_K9 < hot_small < 11.0, hot_small
    # 170 IP of 11 K/9 barely moves
    ace_full = regressed_k9(208, 170)         # 11.0 K/9 over a full season
    assert 10.4 < ace_full < 11.0, ace_full
    # zero innings -> exactly league
    assert regressed_k9(0, 0) == LEAGUE_K9

    # --- expected IP caps ---
    assert expected_ip(0, 0) == LEAGUE_IP_START
    assert expected_ip(40, 20) == 3.0          # 2 IP/start floors at 3
    assert expected_ip(160, 20) == 7.0         # 8 IP/start caps at 7
    assert abs(expected_ip(120, 20) - 6.0) < 1e-9

    # --- opponent multiplier direction + shrink ---
    whiffy = opp_k_multiplier(300, 1000)       # 30% team K vs 22.2% league
    contact = opp_k_multiplier(160, 1000)      # 16%
    assert whiffy > 1.0 > contact
    # shrink: raw 30/22.2 = 1.351 -> shrunk = 1.176
    assert abs(whiffy - 1.176) < 1e-2
    assert opp_k_multiplier(0, 0) == 1.0

    # --- lambda composes sensibly ---
    # ace (11 K/9 full season, 6.3 IP/start) vs whiffy team: lam should beat 8
    lam_ace = k_lambda(231, 189, 30, 300, 1000)   # 11 K/9, 6.3 IP/start
    assert lam_ace > 8.0, lam_ace
    # soft tosser (6.5 K/9) vs contact team, 5 IP starts: lam under 4
    lam_soft = k_lambda(101, 140, 28, 160, 1000)  # 6.5 K/9, 5 IP/start
    assert lam_soft < 4.2, lam_soft
    # monotonic: better opponent-whiff always raises lambda
    assert k_lambda(150,150,25, 280,1000) > k_lambda(150,150,25, 200,1000)

    # --- pricing ladder sane ---
    card = project_start("Test Ace", 231, 189, 30, 300, 1000, "WHF")
    lines = card["lines"]
    # p_over strictly decreasing as line rises
    assert all(lines[i]["p_over"] > lines[i+1]["p_over"] for i in range(len(lines)-1))
    # at lam ~8.6, over 6.5 should be a favorite, over 8.5 near coin flip or under
    l65 = [l for l in lines if l["line"] == 6.5][0]
    assert l65["p_over"] > 0.6
    # fair odds signs coherent
    assert l65["fair_over"] < 0 < l65["fair_under"]

    # --- push math on whole lines ---
    pp = p_push(7.0, 7.0)
    assert 0.10 < pp < 0.18                      # mode of Poisson(7) ~ 14.9%
    assert p_push(6.5, 7.0) == 0.0

    print("KPROPS SELFTEST PASS — Poisson exact, regression, workload caps, opp mult, ladder, pushes")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print("Pure K-props engine. Live stats/odds feed via mlb_kprops_run.py on Actions.")
