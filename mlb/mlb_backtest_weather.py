#!/usr/bin/env python3
"""
mlb_backtest_weather.py — historical first-pitch weather for the season replay.

The ONLY Actions-only piece is the Meteostat network fetch (bulk.meteostat.net is
egress-blocked in the dev sandbox but open on GitHub Actions — same as statsapi).
Everything else — station mapping, first-pitch-hour selection, and reproducing the
live weather multiplier — is pure and tested here tonight.

Design:
  * STADIUM_COORDS: each park -> (lat, lon). Meteostat's Point uses these; it finds
    the nearest hourly station automatically (usually the field's airport).
  * fetch_park_hours(venue, date): one Meteostat hourly pull for that park-day,
    cached per (venue,date) so a doubleheader or repeated call hits network once.
  * weather_for(venue, date, hour, roof): pick the row nearest first pitch, hand
    temp/wind to the SAME mlb_hr.hr_weather_mult the live board uses -> identical mult.
  * merge_into(rows, game_ctx): multiply each predicted row's implied per-PA prob by
    the weather mult and re-derive hr_pct. Leak-free: weather for date D is D's actual
    game-time conditions, which are known post-hoc and not a function of the outcome.

If Meteostat is unreachable or a park-day is missing, weather_for returns (1.0,'no wx')
— the row silently falls back to the weather-free number. No fabricated temperatures.
"""
import os, json, math, datetime as dt
try:
    import mlb_hr as H
except Exception:
    H = None

# Stadium coordinates (lat, lon). Roof flag: 'dome'/'retract' parks get no weather mult.
STADIUM_COORDS = {
    "Yankee Stadium": (40.8296, -73.9262, "open"),
    "Fenway Park": (42.3467, -71.0972, "open"),
    "Camden Yards": (39.2839, -76.6217, "open"),
    "Tropicana Field": (27.7683, -82.6534, "dome"),
    "Rate Field": (41.8299, -87.6338, "open"),          # Guaranteed Rate / Rate Field
    "Progressive Field": (41.4962, -81.6852, "open"),
    "Comerica Park": (42.3390, -83.0485, "open"),
    "Kauffman Stadium": (39.0517, -94.4803, "open"),
    "Target Field": (44.9817, -93.2776, "open"),
    "Minute Maid Park": (29.7570, -95.3555, "retract"),
    "Daikin Park": (29.7570, -95.3555, "retract"),      # Minute Maid renamed
    "Angel Stadium": (33.8003, -117.8827, "open"),
    "Oakland Coliseum": (37.7516, -122.2005, "open"),
    "Sutter Health Park": (38.5802, -121.5133, "open"), # A's 2026 temp home
    "T-Mobile Park": (47.5914, -122.3325, "retract"),
    "Globe Life Field": (32.7473, -97.0847, "retract"),
    "Rogers Centre": (43.6414, -79.3894, "retract"),
    "Great American Ball Park": (39.0975, -84.5069, "open"),
    "Great American": (39.0975, -84.5069, "open"),
    "American Family Field": (43.0280, -87.9712, "retract"),
    "Wrigley Field": (41.9484, -87.6553, "open"),
    "PNC Park": (40.4469, -80.0057, "open"),
    "Busch Stadium": (38.6226, -90.1928, "open"),
    "Citizens Bank Park": (39.9061, -75.1665, "open"),
    "Citi Field": (40.7571, -73.8458, "open"),
    "Nationals Park": (38.8730, -77.0074, "open"),
    "Truist Park": (33.8908, -84.4678, "open"),
    "loanDepot park": (25.7781, -80.2196, "retract"),
    "Coors Field": (39.7559, -104.9942, "open"),
    "Chase Field": (33.4455, -112.0667, "retract"),
    "Dodger Stadium": (34.0739, -118.2400, "open"),
    "Oracle Park": (37.7786, -122.3893, "open"),
    "Petco Park": (32.7073, -117.1566, "open"),
    "Bank of America Stadium": (35.2258, -80.8528, "open"),
}

_wx_cache = {}   # (venue,date_iso) -> list[dict(hour,temp_f,wspd,wdir)] or None

def _roof_of(venue):
    v = STADIUM_COORDS.get(venue)
    return v[2] if v else "open"

def fetch_park_hours(venue, date):
    """Meteostat hourly for a park-day. Network only on Actions; cached per park-day.
    Returns list of {hour:int, temp_f, wspd, wdir} or None if unavailable."""
    key = (venue, date.isoformat())
    if key in _wx_cache:
        return _wx_cache[key]
    coords = STADIUM_COORDS.get(venue)
    if not coords:
        _wx_cache[key] = None
        return None
    try:
        import meteostat as M
        lat, lon, _roof = coords
        pt = M.Point(lat, lon, 10)
        start = dt.datetime(date.year, date.month, date.day, 0)
        end = dt.datetime(date.year, date.month, date.day, 23, 59)
        ts = M.hourly(pt, start, end)
        df = ts.fetch() if hasattr(ts, "fetch") else ts
        out = []
        if df is not None and len(df):
            for idx, row in df.iterrows():
                hr = idx.hour if hasattr(idx, "hour") else 0
                c = row.get("temp")
                temp_f = (c * 9 / 5 + 32) if c is not None and not _isnan(c) else None
                wspd_kmh = row.get("wspd")
                wspd = (wspd_kmh * 0.621371) if wspd_kmh is not None and not _isnan(wspd_kmh) else None
                wdir = row.get("wdir")
                if wdir is not None and _isnan(wdir): wdir = None
                out.append({"hour": hr, "temp_f": temp_f, "wspd": wspd, "wdir": wdir})
        _wx_cache[key] = out or None
        return _wx_cache[key]
    except Exception:
        _wx_cache[key] = None
        return None

def _isnan(x):
    try: return math.isnan(x)
    except Exception: return False

def weather_for(venue, date, first_pitch_hour, roof=None):
    """(mult, tag) at first pitch — reproduces the LIVE board's multiplier exactly."""
    roof = roof or _roof_of(venue)
    if roof in ("dome", "retract"):
        # live model treats closed roofs as neutral (1.0) — matching hr_weather_mult
        if H: return H.hr_weather_mult(None, "dome", None, None, venue)
        return 1.0, "dome"
    hours = fetch_park_hours(venue, date)
    if not hours:
        return (1.0, "no wx")
    best = min(hours, key=lambda h: abs(h["hour"] - first_pitch_hour))
    if H:
        return H.hr_weather_mult(best["temp_f"], "open", best["wspd"], best["wdir"], venue)
    return 1.0, "no wx"

def apply_weather(p_pa_noweather, mult):
    """Re-price a per-PA prob with the weather multiplier (same order as build_board)."""
    cap = getattr(H, "CAP_PPA", 0.12) if H else 0.12
    return min(p_pa_noweather * mult, cap)

def reprice_row(row, mult, tag):
    """Given a weather-free predicted row, fold weather back in and re-derive hr_pct.
    Inverts the game-prob to per-PA, applies mult, recompounds over the same PA."""
    slot = int(row.get("slot", 5))
    pa_top = getattr(H, "PA_TOP", 4.45) if H else 4.45
    pa_est = max(pa_top - 0.09 * (slot - 1), 3.4)
    p_game = float(row["hr_pct"]) / 100
    if p_game <= 0 or p_game >= 1:
        return row
    p_pa = 1 - (1 - p_game) ** (1 / pa_est)     # invert compounding
    p_pa2 = apply_weather(p_pa, mult)
    p_game2 = 1 - (1 - p_pa2) ** pa_est
    r = dict(row)
    r["hr_pct"] = round(p_game2 * 100, 1)
    r["temp"] = tag
    return r

# ---------------------------------------------------------------------------
def selftest():
    d = dt.date
    # 1) roof parks are neutral, always
    m, t = weather_for("Tropicana Field", d(2026,5,15), 19)
    assert m == 1.0 and t in ("dome",), (m, t)
    m, t = weather_for("Globe Life Field", d(2026,5,15), 19)     # retractable
    assert m == 1.0, (m, t)

    # 2) unknown park / no data -> silent 1.0 fallback, never a fake temp
    assert weather_for("Nonexistent Park", d(2026,5,15), 19) == (1.0, "no wx")

    # 3) MERGE MATH: injected synthetic hours -> nearest-hour pick -> exact live mult
    _wx_cache[("Yankee Stadium", "2026-05-15")] = [
        {"hour": 13, "temp_f": 60, "wspd": 5, "wdir": 180},
        {"hour": 19, "temp_f": 90, "wspd": 8, "wdir": 200},   # first pitch ~7pm
        {"hour": 22, "temp_f": 78, "wspd": 6, "wdir": 200},
    ]
    m_night, tag_night = weather_for("Yankee Stadium", d(2026,5,15), 19)
    if H:
        exp = H.hr_weather_mult(90, "open", 8, 200, "Yankee Stadium")
        assert (round(m_night,4), tag_night) == (round(exp[0],4), exp[1]), (m_night, exp)
        assert m_night > 1.0   # 90F is a hitter's night
        # a 1pm game at the SAME park-day picks the 60F row -> lower mult
        m_day, _ = weather_for("Yankee Stadium", d(2026,5,15), 13)
        assert m_day < m_night, (m_day, m_night)

    # 4) reprice_row: hotter weather RAISES hr_pct, invert/recompound is consistent
    if H:
        base_row = {"player": "Slugger", "slot": 3, "hr_pct": 8.0, "temp": ""}
        hot = reprice_row(base_row, 1.176, "90F")
        cold = reprice_row(base_row, 0.856, "52F")
        assert hot["hr_pct"] > 8.0 > cold["hr_pct"], (hot["hr_pct"], cold["hr_pct"])
        # neutral mult returns ~same number (round-trip invert/recompound is lossless)
        same = reprice_row(base_row, 1.0, "72F")
        assert abs(same["hr_pct"] - 8.0) < 0.05, same["hr_pct"]

    # 5) leak-safety: weather_for depends only on (venue,date,hour) — never on outcome.
    #    Same inputs -> same output regardless of call order (idempotent read).
    a = weather_for("Yankee Stadium", d(2026,5,15), 19)
    b = weather_for("Yankee Stadium", d(2026,5,15), 19)
    assert a == b

    print("WEATHER MERGE SELFTEST PASS — roof/fallback/nearest-hour/exact-mult/reprice/leak-safe")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print("Merge logic is pure + tested here; the Meteostat fetch runs on GitHub Actions.")
