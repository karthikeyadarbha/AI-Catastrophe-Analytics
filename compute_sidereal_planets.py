#!/usr/bin/env python3
"""
compute_sidereal_planets.py

Reads a CSV with columns including:
 - time (ISO 8601 UTC)
 - latitude
 - longitude

Computes sidereal ecliptic longitudes/latitudes, zodiac sign, topocentric altitude,
combustion, retrograde for Sun, Moon, Mars, Saturn, Venus, Rahu/Ketu (mean & true)

Outputs: <input>.with_astrology.csv

(Updated to normalize textual NULL tokens and zero numeric NULLs for magnitude/depth fields.)
"""
import sys
import os
import math
import argparse
import pandas as pd
import numpy as np
import swisseph as swe
from dateutil import parser
from datetime import timezone
from skyfield.api import load, Topos
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_COMBUSTION_DEG = 8.5
ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]
ZODIAC_INDEX_BASE = 1

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

def parse_time_to_jd_utc(timestr):
    dt = parser.isoparse(timestr)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    year = dt.year
    month = dt.month
    day = dt.day
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
        return (x, y, z)
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
                sp_val = sp[0]
                if sp_val is not None:
                    return float(sp_val)
            elif isinstance(sp, (float, int)):
                return float(sp)
    except Exception:
        pass
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

def compute_row_outputs(jd, dt, lat, lon, eph, ts, combustion_deg):
    out = {}
    t = ts.from_datetime(dt)
    observer = eph["earth"] + Topos(latitude_degrees=float(lat), longitude_degrees=float(lon))
    FLAG_SID_SPEED = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED

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

            try:
                body = eph[pname.lower()]
                astrom = observer.at(t).observe(body).apparent()
                alt, az, dist = astrom.altaz()
                alt_deg = alt.degrees
            except Exception:
                alt_deg = np.nan
            out[f"{pname}_altitude_deg"] = alt_deg
            out[f"{pname}_is_above_horizon"] = (alt_deg > 0.0) if (not np.isnan(alt_deg)) else None

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

# --- Begin: helper to normalize NULL-like tokens and zero numeric NULLs ---
def _zero_null_like_values(df: pd.DataFrame, cols=None):
    """
    Replace common textual NULL tokens with NaN and set numeric NULLs to 0
    for the specified columns. Modifies and returns the DataFrame.
    Default target columns: mag, magnitude, depthError, magError
    """
    if cols is None:
        cols = ["mag", "magnitude", "depthError", "magError"]

    null_tokens = {"NULL", "null", "NaN", ""}
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        try:
            df[c] = df[c].where(~df[c].astype(str).str.strip().isin(null_tokens), other=np.nan)
        except Exception:
            df[c] = df[c].astype(str).where(~df[c].astype(str).str.strip().isin(null_tokens), other=np.nan)

    modified = []
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            modified.append(c)

    if modified:
        logging.info(f"Normalized NULL-like tokens and set zeros for columns: {modified}")
    return df
# --- End helper ---

def process_file(input_csv, output_csv=None, combustion_deg=DEFAULT_COMBUSTION_DEG, de_file="de421.bsp"):
    if output_csv is None:
        output_csv = input_csv + ".with_astrology.csv"

    swe.set_ephe_path('.')  # can be changed by env var or user
    set_sidereal_lahiri()

    logging.info("Loading ephemeris (this may download de421 if not present)...")
    eph = load(de_file)
    ts = load.timescale()

    logging.info(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    if 'time' not in df.columns or 'latitude' not in df.columns or 'longitude' not in df.columns:
        logging.error("Input CSV must contain columns: time, latitude, longitude")
        raise SystemExit(1)

    df = _zero_null_like_values(df, cols=["mag", "magnitude", "depthError", "magError"])

    results = []
    total = len(df)
    logging.info(f"Processing {total} rows...")
    for idx, row in df.iterrows():
        try:
            time_val = row['time']
            lat = row['latitude']
            lon = row['longitude']
            if pd.isna(time_val) or pd.isna(lat) or pd.isna(lon):
                results.append({})
                continue
            try:
                jd, dt = parse_time_to_jd_utc(str(time_val))
            except Exception:
                logging.warning(f"Row {idx}: invalid time -> {time_val}; skipping")
                results.append({})
                continue
            out = compute_row_outputs(jd, dt, lat, lon, eph, ts, combustion_deg)
            results.append(out)
        except Exception as e:
            logging.error(f"Error processing row {idx}: {e}")
            results.append({})

    results_df = pd.DataFrame(results)
    out_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

    logging.info(f"Writing output CSV: {output_csv}")
    out_df.to_csv(output_csv, index=False)
    logging.info("Done.")

def main():
    parser = argparse.ArgumentParser(description="Compute sidereal astrological columns for a CSV.")
    parser.add_argument("input_csv", help="Input CSV file (must contain columns: time, latitude, longitude)")
    parser.add_argument("--output", "-o", help="Output CSV filename (default: <input>.with_astrology.csv)")
    parser.add_argument("--combustion", "-c", type=float, default=DEFAULT_COMBUSTION_DEG,
                        help=f"Combustion threshold in degrees (default {DEFAULT_COMBUSTION_DEG})")
    parser.add_argument("--de-file", "-d", default="de421.bsp", help="Skyfield DE file to load (default de421.bsp)")
    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        logging.error(f"Input file not found: {args.input_csv}")
        raise SystemExit(1)

    process_file(args.input_csv, output_csv=args.output, combustion_deg=args.combustion, de_file=args.de_file)

if __name__ == "__main__":
    main()