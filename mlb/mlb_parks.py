"""
mlb_parks.py  —  park-factor layer for the TOTALS model
=======================================================
Adjusts the model's projected run total by the venue's park factor.

Data: Statcast 3-year rolling RUN index (2023-25), 100 = league average.
Uncertain parks are NOT dropped -- they're SHRUNK toward neutral by a confidence
weight (the same shrinkage idea that fixed the team & pitcher ratings), and tagged
so the tool can show when it's standing on solid ground vs. estimating.

Contaminated / low-confidence parks:
  * Rogers Centre (TOR)  -- renovated 2023-25, only ~1.5 post-reno seasons
  * Sutter Health (ATH)  -- minor-league park, small sample, plays hot
  * Kauffman (KC)        -- fences moved in for 2025+
These use their best current number shrunk ~50% toward 100.

Effective factor = 100 + (raw - 100) * confidence.
"""

# raw run index (100 = neutral), confidence 0-1 (1 = full 3yr stable sample)
PARKS = {
    # hitter-friendly
    "Coors Field":            (128, 1.0),
    "Great American Ball Park":(112, 1.0),
    "Fenway Park":            (108, 1.0),
    "Citizens Bank Park":     (106, 1.0),
    "Globe Life Field":       (104, 1.0),
    "Yankee Stadium":         (103, 1.0),
    "Chase Field":            (102, 1.0),
    "American Family Field":  (101, 1.0),
    "Camden Yards":           (101, 0.7),   # dimensions adjusted, mild caution
    "Nationals Park":         (101, 1.0),
    "Angel Stadium":          (101, 1.0),
    "Truist Park":            (100, 1.0),
    "Wrigley Field":          (100, 1.0),   # wind-dependent; weather layer handles swings
    "Rogers Centre":          (104, 0.5),   # LOW CONF: post-reno, small sample
    "Sutter Health Park":     (110, 0.5),   # LOW CONF: MiLB park, hot, small sample
    "Kauffman Stadium":       (101, 0.5),   # LOW CONF: fences moved 2025+
    # neutral-ish
    "Progressive Field":      (99, 1.0),
    "Busch Stadium":          (99, 1.0),
    "Target Field":           (99, 1.0),
    "loanDepot park":         (99, 1.0),
    "Rate Field":             (99, 1.0),
    "Comerica Park":          (98, 1.0),
    "Minute Maid Park":       (98, 1.0),   # a.k.a. Daikin Park
    "Daikin Park":            (98, 1.0),
    "Dodger Stadium":         (98, 1.0),
    "PNC Park":               (97, 1.0),
    "Citi Field":             (97, 1.0),
    "Kauffman":               (101, 0.5),
    # pitcher-friendly
    "Great American":         (112, 1.0),
    "Oracle Park":            (94, 1.0),
    "PETCO Park":             (95, 1.0),
    "Petco Park":             (95, 1.0),
    "T-Mobile Park":          (91, 1.0),
    "Oakland Coliseum":       (96, 1.0),
}

DEFAULT = (100, 0.0)   # unknown venue -> neutral, zero confidence (flagged)

def resolve_venue(venue, table):
    """Exact -> case-insensitive -> containment (longest key wins).
    Survives sponsor renames like 'UNIQLO Field at Dodger Stadium'."""
    if not venue: return None
    if venue in table: return venue
    low = {k.lower(): k for k in table}
    v = venue.lower()
    if v in low: return low[v]
    hits = [k for k in table if k.lower() in v or v in k.lower()]
    if hits: return max(hits, key=len)
    return None

def factor(venue):
    """Return (effective_multiplier, confidence, raw_index) for a venue name."""
    key = resolve_venue(venue, PARKS)
    raw, conf = PARKS[key] if key else DEFAULT
    eff = 1 + ((raw - 100) / 100.0) * conf
    return eff, conf, raw

def adjust_total(proj_total, venue):
    """Scale a projected run total by the park's effective factor."""
    eff, conf, raw = factor(venue)
    return proj_total * eff, eff, conf, raw

def tag(venue):
    eff, conf, raw = factor(venue)
    pct = (eff - 1) * 100
    lab = "neutral" if abs(pct) < 0.5 else (f"+{pct:.0f}%" if pct > 0 else f"{pct:.0f}%")
    conf_lab = "" if conf >= 1 else (" [unknown park]" if conf == 0 else " [low-conf]")
    return f"park {lab}{conf_lab}"

if __name__ == "__main__":
    print("PARK FACTOR CHECK (effect on a neutral 9.0-run total):\n")
    for v in ["Coors Field", "Rogers Centre", "Sutter Health Park", "T-Mobile Park",
              "Dodger Stadium", "Yankee Stadium", "Some Unknown Park"]:
        t, eff, conf, raw = adjust_total(9.0, v)
        print(f"  {v:22s} raw {raw:>3}  conf {conf:.1f}  eff x{eff:.3f}  "
              f"9.0 -> {t:.2f}   ({tag(v)})")
