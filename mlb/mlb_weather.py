"""
mlb_weather.py  —  weather layer for the TOTALS model
=====================================================
Adjusts the projected run total for TEMPERATURE and WIND at outdoor parks.

Source: Open-Meteo (api.open-meteo.com) -- free, NO API KEY, hourly forecast by
lat/lon. Runs on your machine (and later inside the GitHub Action).

Honest scoping (from the research):
  * Temperature: warm air is thinner -> ball carries. ~+1% runs per 10F above 70F,
    symmetric below. The best-documented weather effect.
  * Wind: matters most at open parks (Wrigley the famous extreme). We apply a mild
    global effect and note direction is park-geometry-specific -- we use SPEED only,
    a deliberate simplification (direction-aware needs each park's bearing).
  * DOMES: no effect. RETRACTABLE roofs: effect HALVED (roof usually shuts in bad
    weather, so forecast weather often never touches the game). Tagged so you see it.
  * Weather moves TOTALS, not sides. Effects are small (a few %) by design --
    anything bigger would be overfitting the research.

RUN:  python mlb_weather.py            # fetch weather for all outdoor parks now
Used by the model:  from mlb_weather import game_weather_mult
"""
import json, urllib.request, urllib.parse

# venue -> (lat, lon, roof)   roof: 'open' | 'retract' | 'dome'
VENUES = {
    "Angel Stadium":            (33.8003, -117.8827, "open"),
    "Daikin Park":              (29.7573,  -95.3555, "retract"),
    "Minute Maid Park":         (29.7573,  -95.3555, "retract"),
    "Sutter Health Park":       (38.5802, -121.5133, "open"),
    "Oriole Park at Camden Yards":(39.2839, -76.6217, "open"),
    "Camden Yards":             (39.2839,  -76.6217, "open"),
    "Fenway Park":              (42.3467,  -71.0972, "open"),
    "Rate Field":               (41.8299,  -87.6338, "open"),
    "Wrigley Field":            (41.9484,  -87.6553, "open"),
    "Great American Ball Park": (39.0975,  -84.5066, "open"),
    "Progressive Field":        (41.4962,  -81.6852, "open"),
    "Coors Field":              (39.7559, -104.9942, "open"),
    "Comerica Park":            (42.3390,  -83.0485, "open"),
    "Kauffman Stadium":         (39.0517,  -94.4803, "open"),
    "Dodger Stadium":           (34.0739, -118.2400, "open"),
    "loanDepot park":           (25.7781,  -80.2196, "retract"),
    "American Family Field":    (43.0280,  -87.9712, "retract"),
    "Target Field":             (44.9817,  -93.2776, "open"),
    "Citi Field":               (40.7571,  -73.8458, "open"),
    "Yankee Stadium":           (40.8296,  -73.9262, "open"),
    "Nationals Park":           (38.8730,  -77.0074, "open"),
    "Citizens Bank Park":       (39.9061,  -75.1665, "open"),
    "PNC Park":                 (40.4469,  -80.0057, "open"),
    "Petco Park":               (32.7076, -117.1570, "open"),
    "PETCO Park":               (32.7076, -117.1570, "open"),
    "Oracle Park":              (37.7786, -122.3893, "open"),
    "T-Mobile Park":            (47.5914, -122.3325, "retract"),
    "Busch Stadium":            (38.6226,  -90.1928, "open"),
    "Tropicana Field":          (27.7683,  -82.6534, "dome"),
    "George M. Steinbrenner Field":(27.9803,-82.5067, "open"),
    "Rogers Centre":            (43.6414,  -79.3894, "retract"),
    "Globe Life Field":         (32.7473,  -97.0847, "retract"),
    "Chase Field":              (33.4453, -112.0667, "retract"),
    "Truist Park":              (33.8908,  -84.4678, "open"),
}

def fetch_weather(lat, lon, hour_utc=None):
    """Current (or given-hour) temp F + wind mph from Open-Meteo. No key."""
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,wind_speed_10m",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
    })
    with urllib.request.urlopen(f"https://api.open-meteo.com/v1/forecast?{q}", timeout=15) as r:
        d = json.load(r)["current"]
    return d["temperature_2m"], d["wind_speed_10m"]

def weather_mult(temp_f, wind_mph, roof):
    """Total-runs multiplier from weather. Small, research-calibrated effects."""
    if roof == "dome":
        return 1.0, "dome (no wx)"
    t = 1 + 0.001 * (temp_f - 70)          # +1% per 10F above 70
    w = 1 + 0.002 * max(wind_mph - 8, 0)   # wind starts mattering above ~8mph (speed only)
    m = t * w
    tag = f"{temp_f:.0f}F, wind {wind_mph:.0f}mph"
    if roof == "retract":
        m = 1 + (m - 1) * 0.5              # roof may close: halve the effect
        tag += " (retract: halved)"
    return m, tag

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

def game_weather_mult(venue):
    """One-call helper: venue name -> (multiplier, tag). Neutral if unknown."""
    vk = resolve_venue(venue, VENUES)
    v = VENUES.get(vk) if vk else None
    if v is None:
        return 1.0, "unknown venue (no wx)"
    lat, lon, roof = v
    if roof == "dome":
        return 1.0, "dome (no wx)"
    try:
        t, w = fetch_weather(lat, lon)
    except Exception as e:
        return 1.0, f"wx fetch failed ({type(e).__name__})"
    return weather_mult(t, w, roof)

if __name__ == "__main__":
    print("WEATHER CHECK — sample parks right now:\n")
    for v in ["Wrigley Field", "Coors Field", "Yankee Stadium", "Chase Field",
              "Tropicana Field", "Oracle Park"]:
        m, tag = game_weather_mult(v)
        print(f"  {v:22s} x{m:.3f}   {tag}")
    print("\n(multiplier applies to the projected TOTAL, stacked with the park factor)")
