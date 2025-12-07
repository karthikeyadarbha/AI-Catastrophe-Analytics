#!/usr/bin/env python3
"""
add_non_events_and_compute_limited.py

Same functionality as add_non_events_and_compute.py but with safeguards:
- Prevents attempting to compute astrology for extremely large generated grids
  (e.g. hourly grid for 100 years -> ~876k timestamps).
- By default caps number of NEW non-event rows to --max-new-rows (default 20000).
  If the full grid would exceed that, the grid is downsampled (evenly) to meet the cap.
- Keeps other options: interval, latlon-mode, workers, de-file, etc.
- Uses allow_abbrev=False and parse_known_args() to be notebook-friendly.
"""
import argparse
import math
import os
import random
import logging
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import swisseph as swe
from dateutil import parser

# Try to import skyfield lazily
try:
    from skyfield.api import load, Topos
except Exception:
    load = None
    Topos = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Defaults
DEFAULT_INPUT = "/content/1850-1950-EQData-MAG5.csv.with_astrology.csv"
DEFAULT_INTERVAL = "1h"
DEFAULT_COMBUSTION = 8.5
DEFAULT_MAX_NEW_ROWS = 20000  # safety cap; adjust if you know what you're doing

PLANETS = {"Sun": swe.SUN, "Moon": swe.MOON, "Mars": swe.MARS, "Saturn": swe.SATURN, "Venus": swe.VENUS}
NODE_TYPES = {"mean": swe.MEAN_NODE, "true": swe.TRUE_NODE}
ZODIAC_SIGNS = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]

def parse_interval_to_td(s: str) -> timedelta:
    s = s.strip().lower()
    if s.endswith("h"):
        return timedelta(hours=float(s[:-1]))
    if s.endswith("m"):
        return timedelta(minutes=float(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=float(s[:-1]))
    return timedelta(hours=float(s))

def parse_time_value(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff").strip()
    # numeric epoch?
    if s.replace(".", "", 1).lstrip("+-").isdigit():
        try:
            num = float(s)
            if num > 1e14:
                return None
            if num > 1e12:
                return datetime.utcfromtimestamp(num/1000.0).replace(tzinfo=timezone.utc)
            if num > 1e9:
                return datetime.utcfromtimestamp(num).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    try:
        dt = parser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None

def datetime_to_jd(dt: datetime):
    return swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute/60.0 + (dt.second + dt.microsecond/1e6)/3600.0)

def set_sidereal_lahiri():
    swe.set_sid_mode(swe.SIDM_LAHIRI, 0)

def normalize_deg(x):
    return float(x) % 360.0

def zodiac_from_long(lon):
    idx = int(math.floor(lon / 30.0)) % 12
    return ZODIAC_SIGNS[idx], idx + 1

def angular_separation_deg(lon1, lat1, lon2, lat2):
    def sph_to_cart(lon_deg, lat_deg):
        lon = math.radians(lon_deg); lat = math.radians(lat_deg)
        return (math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat))
    v1 = sph_to_cart(lon1, lat1); v2 = sph_to_cart(lon2, lat2)
    dot = max(-1.0, min(1.0, v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]))
    return math.degrees(math.acos(dot))

# compute sidereal for single JD (same as before)
def compute_sidereal_for_jd(jd, combustion_deg=DEFAULT_COMBUSTION):
    FLAG = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED
    out = {"jd": jd}
    try:
        sun_res = swe.calc_ut(jd, swe.SUN, FLAG)
        sun_lon = normalize_deg(float(sun_res[0][0])); sun_lat = float(sun_res[0][1])
    except Exception:
        sun_res = swe.calc_ut(jd, swe.SUN, swe.FLG_SWIEPH | swe.FLG_SPEED)
        sun_lon = normalize_deg(float(sun_res[0][0])); sun_lat = float(sun_res[0][1])
    out["Sun_sid_long"] = sun_lon
    out["Sun_sid_lat"] = sun_lat
    out["Sun_sid_long_over_360"] = sun_lon/360.0
    sgn, sidx = zodiac_from_long(sun_lon); out["Sun_zodiac"] = sgn; out["Sun_zodiac_index"] = sidx

    for pname, pconst in PLANETS.items():
        try:
            res = swe.calc_ut(jd, pconst, FLAG)
            lon_sid = normalize_deg(float(res[0][0])); lat_sid = float(res[0][1])
            out[f"{pname}_sid_long"] = lon_sid
            out[f"{pname}_sid_long_over_360"] = lon_sid/360.0
            out[f"{pname}_sid_lat"] = lat_sid
            sgn, sidx = zodiac_from_long(lon_sid)
            out[f"{pname}_zodiac"] = sgn
            out[f"{pname}_zodiac_index"] = sidx
            # retrograde from res speed if available
            speed = None
            if isinstance(res, tuple) and len(res) >= 4:
                sp = res[3]
                if isinstance(sp, (list, tuple)) and len(sp)>0:
                    speed = float(sp[0])
                elif isinstance(sp, (float, int)):
                    speed = float(sp)
            if speed is None:
                # small numeric derivative fallback
                try:
                    delta = 0.0005
                    r1 = swe.calc_ut(jd, pconst, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)
                    r2 = swe.calc_ut(jd+delta, pconst, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)
                    lon1 = float(r1[0][0]); lon2 = float(r2[0][0])
                    diff = (lon2 - lon1 + 180.0) % 360.0 - 180.0
                    speed = diff / delta
                except Exception:
                    speed = None
            out[f"{pname}_is_retrograde"] = (speed < 0) if speed is not None else None
            out[f"{pname}_is_combust"] = (angular_separation_deg(lon_sid, lat_sid, sun_lon, sun_lat) < combustion_deg)
        except Exception:
            out[f"{pname}_sid_long"] = np.nan
            out[f"{pname}_sid_long_over_360"] = np.nan
            out[f"{pname}_sid_lat"] = np.nan
            out[f"{pname}_zodiac"] = None
            out[f"{pname}_zodiac_index"] = np.nan
            out[f"{pname}_is_retrograde"] = None
            out[f"{pname}_is_combust"] = None

    for node_label, node_const in NODE_TYPES.items():
        try:
            res = swe.calc_ut(jd, node_const, FLAG)
            node_lon = normalize_deg(float(res[0][0])); node_lat = float(res[0][1])
            rahu = f"Rahu_{node_label}"; ketu = f"Ketu_{node_label}"
            out[f"{rahu}_sid_long"] = node_lon
            out[f"{rahu}_sid_long_over_360"] = node_lon/360.0
            out[f"{rahu}_sid_lat"] = node_lat
            sgn, sidx = zodiac_from_long(node_lon)
            out[f"{rahu}_zodiac"] = sgn; out[f"{rahu}_zodiac_index"] = sidx
            out[f"{rahu}_is_combust"] = (angular_separation_deg(node_lon, node_lat, sun_lon, sun_lat) < combustion_deg)
            ketu_lon = normalize_deg(node_lon + 180.0); ketu_lat = -node_lat
            out[f"{ketu}_sid_long"] = ketu_lon
            out[f"{ketu}_sid_long_over_360"] = ketu_lon/360.0
            out[f"{ketu}_sid_lat"] = ketu_lat
            sgn, sidx = zodiac_from_long(ketu_lon)
            out[f"{ketu}_zodiac"] = sgn; out[f"{ketu}_zodiac_index"] = sidx
            out[f"{ketu}_is_combust"] = (angular_separation_deg(ketu_lon, ketu_lat, sun_lon, sun_lat) < combustion_deg)
        except Exception:
            rahu = f"Rahu_{node_label}"; ketu = f"Ketu_{node_label}"
            for k in [rahu, ketu]:
                out[f"{k}_sid_long"] = np.nan
                out[f"{k}_sid_long_over_360"] = np.nan
                out[f"{k}_sid_lat"] = np.nan
                out[f"{k}_zodiac"] = None
                out[f"{k}_zodiac_index"] = np.nan
                out[f"{k}_is_combust"] = None
    return out

# worker utils for skyfield altitude (same as previous)
_global_eph = None
_global_ts = None
def worker_init(de_file):
    global _global_eph, _global_ts
    set_sidereal_lahiri()
    if load is None:
        raise RuntimeError("skyfield not installed")
    try:
        _global_eph = load(de_file)
    except Exception:
        _global_eph = load("de421.bsp")
    _global_ts = load.timescale()

def worker_compute_topo(item):
    iso_time, lat, lon = item
    from dateutil import parser as _parser
    out = {"time": iso_time, "latitude": lat, "longitude": lon}
    try:
        dt = _parser.isoparse(iso_time)
    except Exception:
        for body in ["Sun","Moon","Mars","Saturn","Venus"]:
            out[f"{body}_altitude_deg"] = np.nan
            out[f"{body}_is_above_horizon"] = None
        return out
    t = _global_ts.from_datetime(dt)
    obs = _global_eph["earth"] + Topos(latitude_degrees=float(lat), longitude_degrees=float(lon))
    for body in ["sun","moon","mars","saturn","venus"]:
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

def main():
    ap = argparse.ArgumentParser(description="Add non-event timestamps and compute astrology (with safe caps).", allow_abbrev=False)
    ap.add_argument("--input", "-i", default=DEFAULT_INPUT)
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--interval", default=DEFAULT_INTERVAL)
    ap.add_argument("--latlon-mode", choices=["random","nearest","fixed","none"], default="random")
    ap.add_argument("--fixed-lat", type=float, default=None)
    ap.add_argument("--fixed-lon", type=float, default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(4, cpu_count()-1)))
    ap.add_argument("--combustion", type=float, default=DEFAULT_COMBUSTION)
    ap.add_argument("--de-file", default="de421.bsp")
    ap.add_argument("--recompute-all", action="store_true")
    ap.add_argument("--max-new-rows", type=int, default=DEFAULT_MAX_NEW_ROWS,
                    help="Maximum number of generated non-event rows to add (default 20000). The grid will be downsampled evenly if necessary.")
    args, unknown = ap.parse_known_args()
    if unknown:
        logging.debug("Ignored unknown CLI args: %s", unknown)

    input_csv = args.input
    if not os.path.isfile(input_csv):
        logging.error("Input file not found: %s", input_csv)
        raise SystemExit(1)
    output_csv = args.output if args.output else input_csv.replace(".csv", ".with_non_events.and_astrology.csv")

    logging.info("Reading input file: %s", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)
    if "time" not in df.columns:
        logging.error("Input must have a 'time' column.")
        raise SystemExit("Missing 'time' column")

    logging.info("Parsing time column...")
    df["_parsed_time_"] = df["time"].apply(parse_time_value)
    if df["_parsed_time_"].notna().sum() == 0:
        logging.error("No parseable times found in 'time' column. Aborting.")
        raise SystemExit("No parseable times")

    min_t = df["_parsed_time_"].dropna().min()
    max_t = df["_parsed_time_"].dropna().max()
    interval_td = parse_interval_to_td(args.interval)
    logging.info("Grid from %s to %s at interval %s", min_t.isoformat(), max_t.isoformat(), str(interval_td))

    # Build grid
    grid = []
    t = min_t
    while t <= max_t:
        grid.append(t)
        t = t + interval_td

    existing_set = set(dt.isoformat() for dt in df["_parsed_time_"].dropna().tolist())
    potential_new = [g for g in grid if g.isoformat() not in existing_set]
    logging.info("Grid size %d, potential new timestamps to add %d", len(grid), len(potential_new))

    # If too many new rows, downsample evenly to obey max-new-rows cap
    max_new = args.max_new_rows
    if len(potential_new) > max_new:
        step = math.ceil(len(potential_new) / max_new)
        logging.warning("Requested grid would create %d new rows (> max-new-rows=%d). Downsampling every %d-th timestamp to cap new rows.",
                        len(potential_new), max_new, step)
        potential_new = potential_new[::step]
        logging.info("Downsampled new rows to %d entries.", len(potential_new))

    # Create new rows
    event_rows = df.loc[df["_parsed_time_"].notna() & df.get("latitude").notna() & df.get("longitude").notna()].sort_values("_parsed_time_") if ("latitude" in df.columns and "longitude" in df.columns) else pd.DataFrame()
    event_times = list(event_rows["_parsed_time_"]) if not event_rows.empty else []
    event_lats = list(event_rows["latitude"]) if not event_rows.empty else []
    event_lons = list(event_rows["longitude"]) if not event_rows.empty else []

    def nearest_latlon(ts):
        import bisect
        if not event_times:
            return (np.nan, np.nan)
        i = bisect.bisect_left(event_times, ts)
        candidates = []
        if i < len(event_times):
            candidates.append((abs((event_times[i] - ts).total_seconds()), event_lats[i], event_lons[i]))
        if i-1 >= 0:
            candidates.append((abs((event_times[i-1] - ts).total_seconds()), event_lats[i-1], event_lons[i-1]))
        candidates.sort(key=lambda x: x[0])
        return (candidates[0][1], candidates[0][2]) if candidates else (np.nan, np.nan)

    new_rows = []
    for ts in potential_new:
        iso = ts.isoformat()
        base = {c: np.nan for c in df.columns}
        base["time"] = iso
        if args.latlon_mode == "random":
            base["latitude"] = random.uniform(-90.0, 90.0)
            base["longitude"] = random.uniform(-180.0, 180.0)
        elif args.latlon_mode == "nearest":
            latv, lonv = nearest_latlon(ts)
            base["latitude"] = latv; base["longitude"] = lonv
        elif args.latlon_mode == "fixed":
            if args.fixed_lat is None or args.fixed_lon is None:
                logging.error("fixed mode requires --fixed-lat and --fixed-lon")
                raise SystemExit("Missing fixed lat/lon")
            base["latitude"] = args.fixed_lat; base["longitude"] = args.fixed_lon
        else:
            base["latitude"] = np.nan; base["longitude"] = np.nan
        new_rows.append(base)

    if not new_rows:
        logging.info("No new rows to add. Writing a copy to %s", output_csv)
        df_out = df.drop(columns=["_parsed_time_"], errors="ignore")
        df_out.to_csv(output_csv, index=False)
        logging.info("Wrote: %s", output_csv)
        return

    logging.info("Adding %d new rows", len(new_rows))
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([df.drop(columns=["_parsed_time_"], errors="ignore"), new_df], ignore_index=True, sort=False)

    # Decide which rows require computation (only rows with missing Sun_sid_long or recompute-all)
    if args.recompute_all:
        compute_mask = pd.Series(True, index=combined.index)
    else:
        compute_mask = combined.get("Sun_sid_long").isna().fillna(True)

    rows_to_compute = combined.loc[compute_mask].copy()
    logging.info("Rows needing astrology computation: %d", len(rows_to_compute))

    # Compute JDs for rows_to_compute
    jds = []
    iso_times = []
    for tval in rows_to_compute["time"].astype(str):
        dt = parse_time_value(tval)
        if dt is None:
            jds.append(np.nan); iso_times.append(None)
        else:
            jds.append(datetime_to_jd(dt)); iso_times.append(dt.isoformat())
    rows_to_compute["_jd_"] = jds
    rows_to_compute["_iso_time_"] = iso_times

    unique_jds = sorted({x for x in jds if not (x is None or (isinstance(x, float) and math.isnan(x)))})
    logging.info("Unique JD count to compute sidereal for: %d", len(unique_jds))

    # Safety again: if unique_jds still very large, abort with guidance unless user explicitly increases cap
    if len(unique_jds) > max_new * 5:  # heuristic: if many unique jds still huge
        logging.error("After downsampling grid there are %d unique JDs to compute. This is still very large and likely to crash your session.", len(unique_jds))
        logging.error("Options: increase --max-new-rows, choose a coarser --interval (e.g., 1d), or run on a dedicated machine/terminal.")
        raise SystemExit("Too many unique JDs to compute in this environment. Aborting.")

    # Precompute sidereal per unique JD
    logging.info("Precomputing sidereal positions for unique JDs...")
    set_sidereal_lahiri()
    sid_rows = []
    for i, jd in enumerate(unique_jds):
        sid_rows.append(compute_sidereal_for_jd(jd, combustion_deg=args.combustion))
        if (i+1) % 500 == 0:
            logging.info("  computed %d/%d jds", i+1, len(unique_jds))
    sid_df = pd.DataFrame(sid_rows).set_index("jd")
    rows_to_compute = rows_to_compute.merge(sid_df, left_on="_jd_", right_index=True, how="left")

    # Topocentric altitudes for unique (time,lat,lon)
    rows_to_compute["latitude_num"] = pd.to_numeric(rows_to_compute.get("latitude"), errors="coerce")
    rows_to_compute["longitude_num"] = pd.to_numeric(rows_to_compute.get("longitude"), errors="coerce")
    has_latlon = rows_to_compute["latitude_num"].notna() & rows_to_compute["longitude_num"].notna()
    unique_pairs = rows_to_compute.loc[has_latlon, ["_iso_time_", "latitude_num", "longitude_num"]].drop_duplicates()
    unique_items = [(row["_iso_time_"], float(row["latitude_num"]), float(row["longitude_num"])) for _, row in unique_pairs.iterrows()]
    logging.info("Unique (time,lat,lon) altitude items: %d", len(unique_items))

    if unique_items:
        if load is None or Topos is None:
            logging.error("skyfield not installed. Install skyfield + jplephem to compute altitudes.")
            raise SystemExit("Missing skyfield")
        workers = max(1, min(args.workers, len(unique_items)))
        logging.info("Starting pool with %d workers for topocentric computations", workers)
        pool = Pool(processes=workers, initializer=worker_init, initargs=(args.de_file,))
        topo_results = []
        try:
            for i, res in enumerate(pool.imap_unordered(worker_compute_topo, unique_items, chunksize=max(1, len(unique_items)//(workers*4) or 1))):
                topo_results.append(res)
                if (i+1) % 200 == 0:
                    logging.info("  computed topo %d/%d", i+1, len(unique_items))
        finally:
            pool.close()
            pool.join()
        topo_df = pd.DataFrame(topo_results).rename(columns={"time":"_iso_time_","latitude":"latitude_num","longitude":"longitude_num"})
        rows_to_compute = rows_to_compute.merge(topo_df, on=["_iso_time_","latitude_num","longitude_num"], how="left")

    # Merge computed rows back into combined
    computed_df = rows_to_compute.copy()
    combined = combined.set_index("time")
    computed_df = computed_df.set_index("time")
    for col in computed_df.columns:
        combined.loc[computed_df.index, col] = computed_df[col].values
    combined = combined.reset_index()

    logging.info("Writing output to %s", output_csv)
    combined.to_csv(output_csv, index=False)
    logging.info("Done. Added %d non-event rows (max-new-rows=%d).", len(new_rows), args.max_new_rows)

if __name__ == "__main__":
    main()