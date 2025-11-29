#!/usr/bin/env python3
"""
add_non_events_and_compute.py
Combined script that:
- Reads an input CSV that already contains astrological columns (e.g. 1850-1950-EQData-MAG5.csv.with_astrology.csv)
  or an event CSV if astrology columns are absent.
- Generates non-event timestamps on a regular grid (default 1 hour) between the dataset's min and max times,
  skipping timestamps already present.
- Assigns lat/lon to generated rows (default: random; option to use nearest/fixed/none).
- Computes sidereal astrological attributes for the generated rows (same attributes/format as compute_sidereal_planets.py):
  - Sun, Moon, Mars, Saturn, Venus
  - Rahu_mean / Ketu_mean, Rahu_true / Ketu_true
  - Fields produced: <body>_sid_long, <body>_sid_long_over_360, <body>_sid_lat,
    <body>_zodiac, <body>_zodiac_index, <body>_altitude_deg, <body>_is_above_horizon,
    <body>_is_combust, <body>_is_retrograde (nodes: alt/retrograde set to NaN/None)
- Appends the new rows (with computed astrological columns) to the original file and writes an updated CSV.

Defaults match your previous choices:
- Ayanamsha: Lahiri
- Nodes: mean & true both computed
- Zodiac indexing: 1..12 (1=Aries)
- Combustion threshold: 8.5 degrees
- Generated timestamps grid: 1 hour
- Generated lat/lon default: random within valid ranges (use --latlon-mode nearest to fallback to nearest event coords)

Usage:
  python add_non_events_and_compute.py \
      --input 1850-1950-EQData-MAG5.csv.with_astrology.csv \
      --output 1850-1950-EQData-MAG5.with_non_events.and_astrology.csv

Requirements:
  install from your requirements.txt (pyswisseph, skyfield, jplephem, pandas, numpy, python-dateutil, pytz)
  Example:
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Notes:
- The script will attempt to reuse any existing astrological columns in the input; it only computes astrology for the newly generated rows.
- If your input file contains no parsed times, the script will attempt to detect a datetime column (as the previous robust generator does).
"""

import argparse
import math
import random
import swisseph as swe
import numpy as np
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta, timezone
from skyfield.api import load, Topos
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Constants
DEFAULT_COMBUSTION_DEG = 8.5
ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]
ZODIAC_INDEX_BASE = 1  # 1..12

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Saturn": swe.SATURN,
    "Venus": swe.VENUS
}

NODE_TYPES = {
    "mean": swe.MEAN_NODE,
    "true": swe.TRUE_NODE
}

def set_sidereal_lahiri():
    swe.set_sid_mode(swe.SIDM_LAHIRI, 0)

def parse_iso_to_jd_and_dt(timestr):
    dt = parser.isoparse(timestr)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    year, month, day = dt.year, dt.month, dt.day
    hour = dt.hour + dt.minute / 60.0 + (dt.second + dt.microsecond / 1e6) / 3600.0
    jd = swe.julday(year, month, day, hour)
    return jd, dt

def normalize_deg(x):
    return float(x) % 360.0

def zodiac_from_long(lon):
    sign_index = int(math.floor(lon / 30.0)) % 12
    return ZODIAC_SIGNS[sign_index], sign_index + ZODIAC_INDEX_BASE

def angular_separation_deg(lon1, lat1, lon2, lat2):
    def sph_to_cart(lon_deg, lat_deg):
        lon = math.radians(lon_deg)
        lat = math.radians(lat_deg)
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return x, y, z
    v1 = sph_to_cart(lon1, lat1)
    v2 = sph_to_cart(lon2, lat2)
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

def angular_difference_short(a, b):
    d = (b - a + 180.0) % 360.0 - 180.0
    return d

def estimate_speed_deg_per_day(jd, planet_const, sidereal=True):
    FLAG = swe.FLG_SWIEPH | swe.FLG_SPEED
    if sidereal:
        FLAG |= swe.FLG_SIDEREAL
    try:
        res = swe.calc_ut(jd, planet_const, FLAG)
        if isinstance(res, tuple) and len(res) >= 4:
            sp = res[3]
            if isinstance(sp, (list, tuple)) and len(sp) > 0:
                return float(sp[0])
            elif isinstance(sp, (float, int)):
                return float(sp)
    except Exception:
        pass
    # numeric fallback
    try:
        delta = 0.0005
        r1 = swe.calc_ut(jd, planet_const, swe.FLG_SWIEPH | (swe.FLG_SIDEREAL if sidereal else 0))
        r2 = swe.calc_ut(jd + delta, planet_const, swe.FLG_SWIEPH | (swe.FLG_SIDEREAL if sidereal else 0))
        lon1 = float(r1[0][0])
        lon2 = float(r2[0][0])
        diff = angular_difference_short(lon1, lon2)
        return diff / delta
    except Exception:
        return None

def compute_astrology_for_row(jd, dt, lat, lon, eph, ts, combustion_deg):
    out = {}

    t = ts.from_datetime(dt)
    observer = eph["earth"] + Topos(latitude_degrees=float(lat), longitude_degrees=float(lon))

    FLAG_SID_SPEED = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED

    # Sun sidereal for combustion
    try:
        sun_res = swe.calc_ut(jd, swe.SUN, FLAG_SID_SPEED)
        sun_lon = normalize_deg(sun_res[0][0])
        sun_lat = float(sun_res[0][1])
    except Exception:
        sun_res = swe.calc_ut(jd, swe.SUN, swe.FLG_SWIEPH | swe.FLG_SPEED)
        sun_lon = normalize_deg(sun_res[0][0])
        sun_lat = float(sun_res[0][1])

    for pname, pconst in PLANETS.items():
        try:
            res = swe.calc_ut(jd, pconst, FLAG_SID_SPEED)
            lon_sid = normalize_deg(res[0][0])
            lat_sid = float(res[0][1])
            out[f"{pname}_sid_long"] = lon_sid
            out[f"{pname}_sid_long_over_360"] = lon_sid / 360.0
            out[f"{pname}_sid_lat"] = lat_sid
            sign, sidx = zodiac_from_long(lon_sid)
            out[f"{pname}_zodiac"] = sign
            out[f"{pname}_zodiac_index"] = sidx

            # retrograde
            speed = None
            try:
                if isinstance(res, tuple) and len(res) >= 4:
                    sp = res[3]
                    if isinstance(sp, (list, tuple)) and len(sp) > 0:
                        speed = float(sp[0])
                    elif isinstance(sp, (float, int)):
                        speed = float(sp)
            except Exception:
                speed = None
            if speed is None:
                speed = estimate_speed_deg_per_day(jd, pconst, sidereal=True)
            out[f"{pname}_is_retrograde"] = (speed < 0) if (speed is not None) else None

            # topocentric altitude
            try:
                body = eph[pname.lower()]
                astrom = observer.at(t).observe(body).apparent()
                alt, az, dist = astrom.altaz()
                alt_deg = alt.degrees
            except Exception:
                alt_deg = np.nan
            out[f"{pname}_altitude_deg"] = alt_deg
            out[f"{pname}_is_above_horizon"] = (alt_deg > 0.0) if (not np.isnan(alt_deg)) else None

            # combustion
            sep = angular_separation_deg(lon_sid, lat_sid, sun_lon, sun_lat)
            out[f"{pname}_is_combust"] = (sep < combustion_deg)
        except Exception:
            out[f"{pname}_sid_long"] = np.nan
            out[f"{pname}_sid_long_over_360"] = np.nan
            out[f"{pname}_sid_lat"] = np.nan
            out[f"{pname}_zodiac"] = None
            out[f"{pname}_zodiac_index"] = np.nan
            out[f"{pname}_altitude_deg"] = np.nan
            out[f"{pname}_is_above_horizon"] = None
            out[f"{pname}_is_combust"] = None
            out[f"{pname}_is_retrograde"] = None

    # nodes
    for node_label, node_const in NODE_TYPES.items():
        try:
            res = swe.calc_ut(jd, node_const, FLAG_SID_SPEED)
            node_lon = normalize_deg(res[0][0])
            node_lat = float(res[0][1])
            rahu_key = f"Rahu_{node_label}"
            ketu_key = f"Ketu_{node_label}"

            out[f"{rahu_key}_sid_long"] = node_lon
            out[f"{rahu_key}_sid_long_over_360"] = node_lon / 360.0
            out[f"{rahu_key}_sid_lat"] = node_lat
            sign_r, sidx_r = zodiac_from_long(node_lon)
            out[f"{rahu_key}_zodiac"] = sign_r
            out[f"{rahu_key}_zodiac_index"] = sidx_r
            out[f"{rahu_key}_is_retrograde"] = None
            out[f"{rahu_key}_altitude_deg"] = np.nan
            out[f"{rahu_key}_is_above_horizon"] = None
            sep_r = angular_separation_deg(node_lon, node_lat, sun_lon, sun_lat)
            out[f"{rahu_key}_is_combust"] = (sep_r < combustion_deg)

            ketu_lon = normalize_deg(node_lon + 180.0)
            ketu_lat = -node_lat
            out[f"{ketu_key}_sid_long"] = ketu_lon
            out[f"{ketu_key}_sid_long_over_360"] = ketu_lon / 360.0
            out[f"{ketu_key}_sid_lat"] = ketu_lat
            sign_k, sidx_k = zodiac_from_long(ketu_lon)
            out[f"{ketu_key}_zodiac"] = sign_k
            out[f"{ketu_key}_zodiac_index"] = sidx_k
            out[f"{ketu_key}_is_retrograde"] = None
            out[f"{ketu_key}_altitude_deg"] = np.nan
            out[f"{ketu_key}_is_above_horizon"] = None
            sep_k = angular_separation_deg(ketu_lon, ketu_lat, sun_lon, sun_lat)
            out[f"{ketu_key}_is_combust"] = (sep_k < combustion_deg)
        except Exception:
            rahu_key = f"Rahu_{node_label}"
            ketu_key = f"Ketu_{node_label}"
            for p in [rahu_key, ketu_key]:
                out[f"{p}_sid_long"] = np.nan
                out[f"{p}_sid_long_over_360"] = np.nan
                out[f"{p}_sid_lat"] = np.nan
                out[f"{p}_zodiac"] = None
                out[f"{p}_zodiac_index"] = np.nan
                out[f"{p}_altitude_deg"] = np.nan
                out[f"{p}_is_above_horizon"] = None
                out[f"{p}_is_combust"] = None
                out[f"{p}_is_retrograde"] = None

    return out

def detect_time_column_and_parse(df):
    # prefer 'time' if present
    if 'time' in df.columns:
        parsed = []
        for v in df['time'].astype(str).tolist():
            try:
                parsed.append(parser.isoparse(v).astimezone(timezone.utc))
            except Exception:
                parsed.append(None)
        df['__parsed_time__'] = parsed
        if df['__parsed_time__'].notna().sum() > 0:
            return df
    # fallback: try detect columns named like date/time/datetime
    for col in df.columns:
        lc = col.lower()
        if any(k in lc for k in ('datetime','timestamp','date','time')):
            parsed = []
            for v in df[col].astype(str).tolist():
                try:
                    parsed.append(parser.parse(v).astimezone(timezone.utc))
                except Exception:
                    parsed.append(None)
            df['__parsed_time__'] = parsed
            if df['__parsed_time__'].notna().sum() > 0:
                return df
    raise SystemExit("Could not detect/parse a datetime column. Ensure input has a parseable 'time' or 'datetime' column.")

def nearest_latlon_for_ts(ts, event_times, event_lats, event_lons):
    import bisect
    i = bisect.bisect_left(event_times, ts)
    candidates = []
    if i < len(event_times):
        candidates.append((abs((event_times[i] - ts).total_seconds()), event_lats[i], event_lons[i]))
    if i-1 >= 0:
        candidates.append((abs((event_times[i-1] - ts).total_seconds()), event_lats[i-1], event_lons[i-1]))
    if not candidates:
        return (np.nan, np.nan)
    candidates.sort(key=lambda x: x[0])
    return (candidates[0][1], candidates[0][2])

def main():
    ap = argparse.ArgumentParser(description="Add non-event timestamps and compute astrology for them.")
    ap.add_argument("--input", "-i", default="1850-1950-EQData-MAG5.csv.with_astrology.csv",
                    help="Input CSV (default: 1850-1950-EQData-MAG5.csv.with_astrology.csv)")
    ap.add_argument("--output", "-o", default=None, help="Output CSV filename")
    ap.add_argument("--interval", default="1h", help="Grid interval (e.g., 1h, 30m, 1d). Default 1h")
    ap.add_argument("--latlon-mode", choices=["random","nearest","fixed","none"], default="random",
                    help="lat/lon for generated rows (default random)")
    ap.add_argument("--fixed-lat", type=float, default=None)
    ap.add_argument("--fixed-lon", type=float, default=None)
    ap.add_argument("--combustion", type=float, default=DEFAULT_COMBUSTION_DEG, help="Combustion threshold degrees")
    ap.add_argument("--de-file", default="de421.bsp", help="Skyfield DE file (default de421.bsp)")
    args = ap.parse_args()

    input_csv = args.input
    output_csv = args.output if args.output else (input_csv.replace(".with_astrology.csv", ".with_non_events.and_astrology.csv") if input_csv.endswith(".with_astrology.csv") else input_csv + ".with_non_events.and_astrology.csv")

    logging.info("Reading input CSV: %s", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)

    # detect/parse times
    try:
        df = detect_time_column_and_parse(df)
    except SystemExit as e:
        logging.error(str(e))
        raise

    valid_times = df['__parsed_time__'].dropna().tolist()
    if len(valid_times) == 0:
        raise SystemExit("No parseable times found in input.")

    min_t = min(valid_times)
    max_t = max(valid_times)
    # parse interval
    s = args.interval.strip().lower()
    if s.endswith('h'):
        dt_interval = timedelta(hours=float(s[:-1]))
    elif s.endswith('m'):
        dt_interval = timedelta(minutes=float(s[:-1]))
    elif s.endswith('d'):
        dt_interval = timedelta(days=float(s[:-1]))
    else:
        dt_interval = timedelta(hours=float(s))

    # build grid
    grid = []
    t = min_t
    while t <= max_t:
        grid.append(t)
        t += dt_interval

    existing_set = set([dt.isoformat() for dt in valid_times])

    # prepare event lat/lon for nearest mode
    if 'latitude' in df.columns and 'longitude' in df.columns:
        event_rows = df.loc[df['__parsed_time__'].notna() & df['latitude'].notna() & df['longitude'].notna()].copy()
        event_rows = event_rows.sort_values('__parsed_time__')
        event_times = list(event_rows['__parsed_time__'])
        event_lats = list(event_rows['latitude'])
        event_lons = list(event_rows['longitude'])
    else:
        event_times = []
        event_lats = []
        event_lons = []

    new_rows = []
    for ts in grid:
        iso = ts.isoformat()
        if iso in existing_set:
            continue
        new_row = {col: np.nan for col in df.columns}
        # ensure standard time column exists; use 'time' if input had it, else create 'time'
        if 'time' in df.columns:
            new_row['time'] = iso
        else:
            new_row['time'] = iso
        # latlon assignment
        if args.latlon_mode == 'random':
            latv = random.uniform(-90.0, 90.0)
            lonv = random.uniform(-180.0, 180.0)
        elif args.latlon_mode == 'nearest':
            latv, lonv = nearest_latlon_for_ts(ts, event_times, event_lats, event_lons) if event_times else (np.nan, np.nan)
        elif args.latlon_mode == 'fixed':
            if args.fixed_lat is None or args.fixed_lon is None:
                raise SystemExit("fixed latlon requires --fixed-lat and --fixed-lon")
            latv, lonv = args.fixed_lat, args.fixed_lon
        else:
            latv, lonv = (np.nan, np.nan)
        new_row['latitude'] = latv
        new_row['longitude'] = lonv
        new_rows.append(new_row)

    if not new_rows:
        logging.info("No new timestamps to add (grid matched existing timestamps).")
        df_out = df.drop(columns='__parsed_time__', errors='ignore')
        df_out.to_csv(output_csv, index=False)
        logging.info("Wrote copy to %s", output_csv)
        return

    # initialize ephemerides
    logging.info("Initializing swisseph and skyfield (Lahiri ayanamsha).")
    swe.set_ephe_path('.')
    set_sidereal_lahiri()
    logging.info("Loading skyfield DE file (may download if missing): %s", args.de_file)
    eph = load(args.de_file)
    ts = load.timescale()

    # compute astrology for each new row
    computed = []
    logging.info("Computing astrology for %d new rows...", len(new_rows))
    for nr in new_rows:
        iso = nr['time']
        lat = nr.get('latitude', np.nan)
        lon = nr.get('longitude', np.nan)
        try:
            jd, dt = parse_iso_to_jd_and_dt(iso)
        except Exception:
            logging.warning("Skipping row with unparseable time: %s", iso)
            computed.append({})
            continue
        if pd.isna(lat) or pd.isna(lon):
            # still compute sidereal long/lat (some computations don't need observer), but altitude will be NaN
            try:
                # compute sidereal planet long/lat without altitude if lat/lon NaN; we'll call compute_astrology_for_row but pass dummy lat/lon 0,0 and then set alt NaN
                out = compute_astrology_for_row(jd, dt, 0.0, 0.0, eph, ts, args.combustion)
                # override altitude-related fields to NaN/None
                for pname in PLANETS.keys():
                    out[f"{pname}_altitude_deg"] = np.nan
                    out[f"{pname}_is_above_horizon"] = None
            except Exception:
                out = {}
        else:
            out = compute_astrology_for_row(jd, dt, lat, lon, eph, ts, args.combustion)
        # merge nr base columns with computed astrological dict
        merged = {**nr, **out}
        computed.append(merged)

    new_df = pd.DataFrame(computed)
    # attach parsed time column to new_df to allow sorting, then drop before final write
    new_df['__parsed_time__'] = new_df['time'].apply(lambda s: parser.isoparse(s) if pd.notna(s) else None)

    # combine original df (drop parsed helper) and new rows
    orig = df.drop(columns='__parsed_time__', errors='ignore')
    combined = pd.concat([orig, new_df], ignore_index=True, sort=False)

    # sort by time
    combined['__parsed_time__'] = combined['time'].apply(lambda s: parser.isoparse(str(s)) if pd.notna(s) else None)
    combined = combined.sort_values('__parsed_time__').drop(columns='__parsed_time__')

    logging.info("Writing combined output to %s", output_csv)
    combined.to_csv(output_csv, index=False)
    logging.info("Done. Added %d non-event rows.", len(new_rows))

if __name__ == "__main__":
    main()