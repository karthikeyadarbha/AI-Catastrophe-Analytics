#!/usr/bin/env python3
"""
add_non_events_and_compute.py

Combined, robust, and optimized script that:
- Reads an input CSV which MUST contain a 'time' column (case-sensitive).
- Parses the 'time' column (supports ISO strings, many common formats, and epoch seconds/ms).
- Generates non-event timestamps on a regular grid (default: 1 hour) between the dataset min and max times,
  skipping timestamps already present.
- Assigns lat/lon to generated rows according to --latlon-mode (random, nearest, fixed, none).
- Computes sidereal astrological attributes for the newly created rows (or for all rows if --recompute-all):
  - Sun, Moon, Mars, Saturn, Venus
  - Rahu_mean / Ketu_mean, Rahu_true / Ketu_true
  - Attributes: <body>_sid_long, <body>_sid_long_over_360, <body>_sid_lat,
    <body>_zodiac, <body>_zodiac_index, <body>_altitude_deg, <body>_is_above_horizon,
    <body>_is_combust, <body>_is_retrograde (nodes: alt/retrograde left as NaN/None)
- Uses batching and optional multiprocessing to speed up altitude (topocentric) computations.
- Writes an output CSV that appends the new rows and includes computed astrological columns.

Usage examples:
  python add_non_events_and_compute.py --input 1850-1950-EQData-MAG5.csv.with_astrology.csv
  %run -i add_non_events_and_compute.py --input 1850-1950-EQData-MAG5.csv.with_astrology.csv

Notes:
- This script expects a 'time' column. If your CSV has a different datetime column name,
  pre-create / rename it to 'time' (lowercase).
- If running inside Jupyter/IPython, the script will now ignore kernel-injected CLI args.
- Install dependencies in a venv before running:
    pip install pyswisseph skyfield jplephem pandas numpy python-dateutil pytz
- For faster startup, download a Skyfield DE file (de421.bsp or de440.bsp) into the working dir and pass --de-file.
"""
import argparse
import math
import random
import swisseph as swe
import numpy as np
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta, timezone
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

# Attempt to import Skyfield lazily; raise clear error if missing at runtime when needed.
try:
    from skyfield.api import load, Topos
except Exception:
    load = None
    Topos = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Constants / defaults
DEFAULT_INPUT = "1850-1950-EQData-MAG5.csv.with_astrology.csv"
DEFAULT_COMBUSTION_DEG = 8.5
DEFAULT_INTERVAL = "1h"
PLANETS = {"Sun": swe.SUN, "Moon": swe.MOON, "Mars": swe.MARS, "Saturn": swe.SATURN, "Venus": swe.VENUS}
NODE_TYPES = {"mean": swe.MEAN_NODE, "true": swe.TRUE_NODE}
ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]


# -----------------------
# Utility functions
# -----------------------
def parse_interval_to_td(s: str) -> timedelta:
    s = s.strip().lower()
    if s.endswith("h"):
        return timedelta(hours=float(s[:-1]))
    if s.endswith("m"):
        return timedelta(minutes=float(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=float(s[:-1]))
    # fallback: numeric hours
    return timedelta(hours=float(s))


def normalize_deg(x):
    return float(x) % 360.0


def zodiac_from_long(lon):
    idx = int(math.floor(lon / 30.0)) % 12
    return ZODIAC_SIGNS[idx], idx + 1


def angular_separation_deg(lon1, lat1, lon2, lat2):
    # spherical angle between two (ecliptic) lon/lat points
    def sph_to_cart(lon_deg, lat_deg):
        lon = math.radians(lon_deg)
        lat = math.radians(lat_deg)
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return (x, y, z)
    v1 = sph_to_cart(lon1, lat1)
    v2 = sph_to_cart(lon2, lat2)
    dot = max(-1.0, min(1.0, v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]))
    return math.degrees(math.acos(dot))


# Robust time parsing: supports ISO strings and numeric epoch seconds/ms
def parse_time_value(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    # strip BOM if present
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff").strip()
    # numeric?
    if s.replace(".", "", 1).lstrip("+-").isdigit():
        try:
            num = float(s)
            # heuristics
            if num > 1e14:
                return None
            if num > 1e12:
                # ms
                dt = datetime.utcfromtimestamp(num / 1000.0).replace(tzinfo=timezone.utc)
                return dt
            if num > 1e9:
                # seconds
                dt = datetime.utcfromtimestamp(num).replace(tzinfo=timezone.utc)
                return dt
        except Exception:
            pass
    try:
        dt = parser.parse(s)
        if dt.tzinfo is None:
            # treat naive datetimes as UTC (user data may vary; modify if needed)
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


# Convert datetime (aware, UTC) to Julian Day UT for swisseph
def datetime_to_jd(dt: datetime):
    return swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute/60.0 + (dt.second + dt.microsecond/1e6)/3600.0)


# Set sidereal mode (Lahiri) for swisseph
def set_sidereal_lahiri():
    swe.set_sid_mode(swe.SIDM_LAHIRI, 0)


# Compute sidereal planet/node values for a given JD (returns dict)
def compute_sidereal_for_jd(jd, combustion_deg=DEFAULT_COMBUSTION_DEG):
    FLAG = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED
    out = {"jd": jd}
    # sun for combustion reference
    try:
        sun_res = swe.calc_ut(jd, swe.SUN, FLAG)
        sun_lon = normalize_deg(float(sun_res[0][0])); sun_lat = float(sun_res[0][1])
    except Exception:
        # fallback without sidereal flag
        sun_res = swe.calc_ut(jd, swe.SUN, swe.FLG_SWIEPH | swe.FLG_SPEED)
        sun_lon = normalize_deg(float(sun_res[0][0])); sun_lat = float(sun_res[0][1])
    out["Sun_sid_long"] = sun_lon
    out["Sun_sid_lat"] = sun_lat
    out["Sun_sid_long_over_360"] = sun_lon / 360.0
    sign, sidx = zodiac_from_long(sun_lon)
    out["Sun_zodiac"] = sign
    out["Sun_zodiac_index"] = sidx

    for pname, pconst in PLANETS.items():
        try:
            res = swe.calc_ut(jd, pconst, FLAG)
            lon_sid = normalize_deg(float(res[0][0])); lat_sid = float(res[0][1])
            out[f"{pname}_sid_long"] = lon_sid
            out[f"{pname}_sid_long_over_360"] = lon_sid / 360.0
            out[f"{pname}_sid_lat"] = lat_sid
            sgn, sidx = zodiac_from_long(lon_sid)
            out[f"{pname}_zodiac"] = sgn
            out[f"{pname}_zodiac_index"] = sidx
            # retrograde detection
            speed = None
            if isinstance(res, tuple) and len(res) >= 4:
                sp = res[3]
                if isinstance(sp, (list, tuple)) and len(sp) > 0:
                    speed = float(sp[0])
                elif isinstance(sp, (float, int)):
                    speed = float(sp)
            # fallback numerical derivative if speed None
            if speed is None:
                try:
                    delta = 0.0005
                    r1 = swe.calc_ut(jd, pconst, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)
                    r2 = swe.calc_ut(jd + delta, pconst, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)
                    lon1 = float(r1[0][0]); lon2 = float(r2[0][0])
                    diff = (lon2 - lon1 + 180.0) % 360.0 - 180.0
                    speed = diff / delta
                except Exception:
                    speed = None
            out[f"{pname}_is_retrograde"] = (speed < 0) if speed is not None else None
            # combustion wrt Sun
            sep = angular_separation_deg(lon_sid, lat_sid, sun_lon, sun_lat)
            out[f"{pname}_is_combust"] = (sep < combustion_deg)
        except Exception:
            out[f"{pname}_sid_long"] = np.nan
            out[f"{pname}_sid_long_over_360"] = np.nan
            out[f"{pname}_sid_lat"] = np.nan
            out[f"{pname}_zodiac"] = None
            out[f"{pname}_zodiac_index"] = np.nan
            out[f"{pname}_is_retrograde"] = None
            out[f"{pname}_is_combust"] = None

    # nodes (Rahu/Ketu) for mean and true
    for node_label, node_const in NODE_TYPES.items():
        try:
            res = swe.calc_ut(jd, node_const, FLAG)
            node_lon = normalize_deg(float(res[0][0])); node_lat = float(res[0][1])
            rahu_key = f"Rahu_{node_label}"
            ketu_key = f"Ketu_{node_label}"
            out[f"{rahu_key}_sid_long"] = node_lon
            out[f"{rahu_key}_sid_long_over_360"] = node_lon / 360.0
            out[f"{rahu_key}_sid_lat"] = node_lat
            sign_r, sidx_r = zodiac_from_long(node_lon)
            out[f"{rahu_key}_zodiac"] = sign_r
            out[f"{rahu_key}_zodiac_index"] = sidx_r
            out[f"{rahu_key}_is_combust"] = (angular_separation_deg(node_lon, node_lat, sun_lon, sun_lat) < combustion_deg)

            ketu_lon = normalize_deg(node_lon + 180.0); ketu_lat = -node_lat
            out[f"{ketu_key}_sid_long"] = ketu_lon
            out[f"{ketu_key}_sid_long_over_360"] = ketu_lon / 360.0
            out[f"{ketu_key}_sid_lat"] = ketu_lat
            sign_k, sidx_k = zodiac_from_long(ketu_lon)
            out[f"{ketu_key}_zodiac"] = sign_k
            out[f"{ketu_key}_zodiac_index"] = sidx_k
            out[f"{ketu_key}_is_combust"] = (angular_separation_deg(ketu_lon, ketu_lat, sun_lon, sun_lat) < combustion_deg)
        except Exception:
            rahu_key = f"Rahu_{node_label}"; ketu_key = f"Ketu_{node_label}"
            for k in [rahu_key, ketu_key]:
                out[f"{k}_sid_long"] = np.nan
                out[f"{k}_sid_long_over_360"] = np.nan
                out[f"{k}_sid_lat"] = np.nan
                out[f"{k}_zodiac"] = None
                out[f"{k}_zodiac_index"] = np.nan
                out[f"{k}_is_combust"] = None
    return out


# Worker utilities for multiprocessing topocentric altitude computation
_global_eph = None
_global_ts = None


def worker_init(de_file):
    # Called once per worker process
    global _global_eph, _global_ts
    set_sidereal_lahiri()
    if load is None:
        raise RuntimeError("skyfield not available. Install skyfield and jplephem.")
    try:
        _global_eph = load(de_file)
    except Exception:
        # fallback to default download
        _global_eph = load("de421.bsp")
    _global_ts = load.timescale()


def worker_compute_topo(item):
    # item: (iso_time_str, lat, lon)
    iso_time, lat, lon = item
    from dateutil import parser as _parser
    out = {"time": iso_time, "latitude": lat, "longitude": lon}
    try:
        dt = _parser.isoparse(iso_time)
    except Exception:
        out.update({
            "Sun_altitude_deg": np.nan, "Sun_is_above_horizon": None,
            "Moon_altitude_deg": np.nan, "Moon_is_above_horizon": None,
            "Mars_altitude_deg": np.nan, "Mars_is_above_horizon": None,
            "Saturn_altitude_deg": np.nan, "Saturn_is_above_horizon": None,
            "Venus_altitude_deg": np.nan, "Venus_is_above_horizon": None,
        })
        return out
    t = _global_ts.from_datetime(dt)
    obs = _global_eph["earth"] + Topos(latitude_degrees=float(lat), longitude_degrees=float(lon))
    for body in ["sun", "moon", "mars", "saturn", "venus"]:
        try:
            b = _global_eph[body]
            astrom = obs.at(t).observe(b).apparent()
            alt, az, dist = astrom.altaz()
            out[f"{body.capitalize()}_altitude_deg"] = alt.degrees
            out[f"{body.capitalize()}_is_above_horizon"] = (alt.degrees > 0.0)
        except Exception:
            out[f"{body.capitalize()}_altitude_deg"] = np.nan
            out[f"{body.capitalize()}_is_above_horizon"] = None
    return out


# -----------------------
# Main processing flow
# -----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Add non-event timestamps and compute astrology using 'time' column.",
        allow_abbrev=False
    )
    ap.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input CSV (must contain 'time' column)")
    ap.add_argument("--output", "-o", default=None, help="Output CSV filename")
    ap.add_argument("--interval", default=DEFAULT_INTERVAL, help="Grid interval (e.g., 1h, 30m, 1d) default 1h")
    ap.add_argument("--latlon-mode", choices=["random", "nearest", "fixed", "none"], default="random",
                    help="How to assign lat/lon to generated rows (default random)")
    ap.add_argument("--fixed-lat", type=float, default=None, help="Latitude when latlon-mode=fixed")
    ap.add_argument("--fixed-lon", type=float, default=None, help="Longitude when latlon-mode=fixed")
    ap.add_argument("--workers", type=int, default=max(1, min(4, cpu_count()-1)), help="Parallel workers for topocentric computations")
    ap.add_argument("--combustion", type=float, default=DEFAULT_COMBUSTION_DEG, help="Combustion threshold degrees")
    ap.add_argument("--de-file", default="de421.bsp", help="Skyfield DE file path (local file preferred)")
    ap.add_argument("--recompute-all", action="store_true", help="Recompute astrology for all rows, not just new rows")

    # Use parse_known_args to ignore extra args injected by Jupyter/IPython kernels (e.g. --f=...)
    args, unknown = ap.parse_known_args()
    if unknown:
        logging.debug("Ignored unknown CLI args: %s", unknown)

    input_csv = args.input
    if not os.path.isfile(input_csv):
        logging.error("Input file not found: %s", input_csv)
        raise SystemExit(1)
    output_csv = args.output if args.output else input_csv.replace(".csv", ".with_non_events.and_astrology.csv")

    logging.info("Reading input CSV: %s", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)

    # Validate 'time' column exists
    if "time" not in df.columns:
        logging.error("Input must have a 'time' column (lowercase). Found columns: %s", df.columns.tolist())
        raise SystemExit("Input must have 'time' column")

    # Parse times robustly
    logging.info("Parsing 'time' column (this may take a moment)...")
    parsed_times = df["time"].apply(parse_time_value)
    df["_parsed_time_"] = parsed_times
    if df["_parsed_time_"].notna().sum() == 0:
        logging.error("No parseable times found in 'time' column. Sample values: %s", df["time"].astype(str).head(10).tolist())
        raise SystemExit("No parseable times in 'time' column")

    # Establish min/max bounds and the grid
    valid_times = df.loc[df["_parsed_time_"].notna(), "_parsed_time_"].tolist()
    min_t = min(valid_times)
    max_t = max(valid_times)
    interval_td = parse_interval_to_td(args.interval)
    logging.info("Generating grid from %s to %s with interval %s", min_t.isoformat(), max_t.isoformat(), str(interval_td))

    grid = []
    t = min_t
    # include both endpoints
    while t <= max_t:
        grid.append(t)
        t = t + interval_td

    existing_set = set([dt.isoformat() for dt in valid_times])

    # Prepare event lat/lon arrays for nearest lookup
    event_rows = df.loc[df["_parsed_time_"].notna() & df.get("latitude").notna() & df.get("longitude").notna()].copy() if ("latitude" in df.columns and "longitude" in df.columns) else pd.DataFrame()
    if not event_rows.empty:
        event_rows = event_rows.sort_values("_parsed_time_")
        event_times = list(event_rows["_parsed_time_"])
        event_lats = list(event_rows["latitude"])
        event_lons = list(event_rows["longitude"])
    else:
        event_times = []; event_lats = []; event_lons = []

    # Create new rows for missing timestamps
    new_rows = []
    def nearest_latlon_for_ts(ts):
        import bisect
        if not event_times:
            return (np.nan, np.nan)
        i = bisect.bisect_left(event_times, ts)
        candidates = []
        if i < len(event_times):
            candidates.append((abs((event_times[i] - ts).total_seconds()), event_lats[i], event_lons[i]))
        if i-1 >= 0:
            candidates.append((abs((event_times[i-1] - ts).total_seconds()), event_lats[i-1], event_lons[i-1]))
        if not candidates:
            return (np.nan, np.nan)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]

    for ts in grid:
        iso = ts.isoformat()
        if iso in existing_set:
            continue
        base = {c: np.nan for c in df.columns}
        base["time"] = iso
        # lat/lon assignment
        if args.latlon_mode == "random":
            base["latitude"] = random.uniform(-90.0, 90.0)
            base["longitude"] = random.uniform(-180.0, 180.0)
        elif args.latlon_mode == "nearest":
            latv, lonv = nearest_latlon_for_ts(ts)
            base["latitude"] = latv; base["longitude"] = lonv
        elif args.latlon_mode == "fixed":
            if args.fixed_lat is None or args.fixed_lon is None:
                logging.error("latlon-mode fixed requires --fixed-lat and --fixed-lon")
                raise SystemExit("fixed latlon requires fixed lat and lon")
            base["latitude"] = args.fixed_lat; base["longitude"] = args.fixed_lon
        else:
            base["latitude"] = np.nan; base["longitude"] = np.nan
        new_rows.append(base)

    if not new_rows:
        logging.info("No new timestamps to add (grid matched existing timestamps). Writing copy to %s", output_csv)
        df_out = df.drop(columns=["_parsed_time_"], errors="ignore")
        df_out.to_csv(output_csv, index=False)
        logging.info("Wrote: %s", output_csv)
        return

    logging.info("Generated %d new non-event rows", len(new_rows))
    new_df = pd.DataFrame(new_rows)

    # Determine which rows require astrology computation:
    recompute_for_all = args.recompute_all
    # Merge original + new (we will compute only on selection)
    combined = pd.concat([df.drop(columns=["_parsed_time_"], errors="ignore"), new_df], ignore_index=True, sort=False)

    # Identify rows missing a core astrology column (e.g., Sun_sid_long) or newly generated rows
    if recompute_for_all:
        need_compute_mask = pd.Series([True]*len(combined), index=combined.index)
    else:
        need_compute_mask = combined.get("Sun_sid_long").isna().fillna(True)

    rows_to_compute = combined.loc[need_compute_mask].copy()
    logging.info("Rows to compute astrology for: %d", len(rows_to_compute))

    # Build a map of unique timestamps (for sidereal JD precomputation)
    logging.info("Parsing timestamps to JD for rows to compute...")
    jds = []
    iso_times = []
    for tval in rows_to_compute["time"].astype(str):
        dt = parse_time_value(tval)
        if dt is None:
            jds.append(np.nan)
            iso_times.append(None)
        else:
            iso_times.append(dt.isoformat())
            jds.append(datetime_to_jd(dt))
    rows_to_compute["_jd_"] = jds
    rows_to_compute["_iso_time_"] = iso_times

    unique_jds = sorted(set([x for x in jds if not (x is None or (isinstance(x, float) and math.isnan(x)))]))
    logging.info("Unique JD count to process sidereal: %d", len(unique_jds))

    # Initialize swisseph and compute sidereal data per unique JD
    logging.info("Initializing swisseph (Lahiri ayanamsha) and precomputing sidereal positions...")
    set_sidereal_lahiri()
    sid_rows = []
    for i, jd in enumerate(unique_jds):
        sid = compute_sidereal_for_jd(jd, combustion_deg=args.combustion)
        sid_rows.append(sid)
        if (i+1) % 200 == 0:
            logging.info("  computed sidereal for %d jds", i+1)
    sid_df = pd.DataFrame(sid_rows).set_index("jd")

    # Merge sidereal values onto rows_to_compute by jd
    rows_to_compute = rows_to_compute.merge(sid_df, left_on="_jd_", right_index=True, how="left")

    # Prepare topocentric altitude computations for unique (iso_time, lat, lon) triples
    rows_to_compute["latitude_num"] = pd.to_numeric(rows_to_compute.get("latitude"), errors="coerce")
    rows_to_compute["longitude_num"] = pd.to_numeric(rows_to_compute.get("longitude"), errors="coerce")
    has_latlon = rows_to_compute["latitude_num"].notna() & rows_to_compute["longitude_num"].notna()
    unique_pairs = rows_to_compute.loc[has_latlon, ["_iso_time_", "latitude_num", "longitude_num"]].drop_duplicates()
    unique_items = [(row["_iso_time_"], float(row["latitude_num"]), float(row["longitude_num"])) for _, row in unique_pairs.iterrows()]
    logging.info("Unique (time,lat,lon) for topocentric: %d", len(unique_items))

    topo_results = []
    if unique_items:
        if load is None or Topos is None:
            logging.error("Skyfield not installed. Please install skyfield and jplephem to compute altitudes.")
            raise SystemExit("Missing skyfield dependency")

        workers = max(1, min(args.workers, len(unique_items)))
        logging.info("Starting multiprocessing pool with %d workers for topocentric computations...", workers)
        pool = Pool(processes=workers, initializer=worker_init, initargs=(args.de_file,))
        try:
            for i, res in enumerate(pool.imap_unordered(worker_compute_topo, unique_items, chunksize=max(1, len(unique_items)//(workers*4) or 1))):
                topo_results.append(res)
                if (i+1) % 200 == 0:
                    logging.info("  computed topocentric for %d/%d items", i+1, len(unique_items))
        finally:
            pool.close()
            pool.join()
        topo_df = pd.DataFrame(topo_results)
        topo_df = topo_df.rename(columns={"time": "_iso_time_", "latitude": "latitude_num", "longitude": "longitude_num"})
        rows_to_compute = rows_to_compute.merge(topo_df, on=["_iso_time_", "latitude_num", "longitude_num"], how="left")
    else:
        logging.info("No valid lat/lon found for topocentric computations; altitude columns will be left NaN/None.")

    # Remove helper columns and merge computed rows back into combined dataframe
    computed_df = rows_to_compute.copy()

    # Replace rows in combined with computed_df (matching by 'time' value)
    combined = combined.set_index("time")
    computed_df = computed_df.set_index("time")
    for col in computed_df.columns:
        combined.loc[computed_df.index, col] = computed_df[col].values

    combined = combined.reset_index()

    # Final write
    logging.info("Writing output CSV: %s", output_csv)
    combined.to_csv(output_csv, index=False)
    logging.info("Done. Added %d non-event rows and computed astrology for %d rows.", len(new_rows), len(rows_to_compute))


if __name__ == "__main__":
    main()