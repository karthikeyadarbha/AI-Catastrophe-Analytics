#!/usr/bin/env python3
"""
generate_celestial_dataset.py

Generate a fresh dataset of earthquake events + synthetic "no-event" controls
and compute rich celestial features (JPL/Horizons-based apparent RA/Dec, distance,
angular diameter, heliocentric ecliptic lon/lat) and sidereal quantities (sidereal
longitudes, Rahu/Ketu nodes) using Skyfield or JPL Horizons (astroquery) and
pyswisseph for sidereal/node calcs.

This variant preserves important original input columns (time, latitude, longitude,
depth, mag, magType, depthError, magError) in the output and adds an explicit
`is_synthetic` boolean column that is True for synthetic controls and False for
original events.

Usage (example):
  .venv/bin/python src/generate_celestial_dataset.py \
    --input 1850-1950-EQData-MAG5.csv \
    --out events_celestial.csv \
    --controls-per-event 2 \
    --lead-hours 48 \
    --radius-km 200 \
    --combustion-deg 8.5 \
    --use-jpl-api
"""
from __future__ import annotations
import argparse
import math
import warnings
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from dateutil import parser as dt_parser

# Optional libraries
try:
    from skyfield.api import load, Topos
    SKYFIELD_AVAILABLE = True
except Exception:
    SKYFIELD_AVAILABLE = False

try:
    import swisseph as swe
    SWEPH_AVAILABLE = True
except Exception:
    SWEPH_AVAILABLE = False

try:
    from astroquery.jplhorizons import Horizons
    HORIZONS_AVAILABLE = True
except Exception:
    HORIZONS_AVAILABLE = False

# Mean radii (km) used to estimate angular diameter
_MEAN_RADIUS_KM = {
    'Sun': 696342.0,
    'Mercury': 2439.7,
    'Venus': 6051.8,
    'Moon': 1737.4,
    'Mars': 3389.5,
    'Jupiter': 69911.0,
    'Saturn': 58232.0,
}

JPL_BODIES = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']

ZODIAC_12 = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# --- Utilities ---------------------------------------------------------------
def jd_from_dt(dt: datetime) -> float:
    if dt is None:
        return float('nan')
    dt_utc = dt.astimezone(timezone.utc)
    if SWEPH_AVAILABLE:
        hour = dt_utc.hour + dt_utc.minute / 60.0 + (dt_utc.second + dt_utc.microsecond / 1e6) / 3600.0
        return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, hour)
    else:
        return 2440587.5 + dt_utc.timestamp() / 86400.0

def normalize_deg(x: float) -> float:
    return float(x) % 360.0 if not (x is None or (isinstance(x, float) and np.isnan(x))) else np.nan

def zodiac_from_long(lon_deg: float) -> Tuple[Optional[str], Optional[int]]:
    if lon_deg is None or (isinstance(lon_deg, float) and np.isnan(lon_deg)):
        return None, None
    idx = int(math.floor(lon_deg / 30.0)) % 12
    return ZODIAC_12[idx], idx + 1

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))

# --- Skyfield helpers -------------------------------------------------------
def _load_skyfield_ephem(de_file: str = "de421.bsp"):
    if not SKYFIELD_AVAILABLE:
        raise RuntimeError("Skyfield not available; install skyfield and jplephem.")
    eph = load(de_file)
    ts = load.timescale()
    return eph, ts

def _body_object_from_ephem(eph, name: str):
    mapping = {
        'Sun': 'sun',
        'Moon': 'moon',
        'Mercury': 'mercury',
        'Venus': 'venus',
        'Mars': 'mars',
        'Jupiter': 'jupiter barycenter' if 'jupiter barycenter' in eph else 'jupiter',
        'Saturn': 'saturn barycenter' if 'saturn barycenter' in eph else 'saturn'
    }
    key = mapping.get(name, name.lower())
    return eph[key]

# --- Horizons (JPL API) helpers ---------------------------------------------
_HORIZONS_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}

def _horizons_query(body: str, dt: datetime, center: str = '500@399') -> Dict[str, Any]:
    if not HORIZONS_AVAILABLE:
        raise RuntimeError("astroquery.jplhorizons not available; install astroquery to use --use-jpl-api")
    jd = jd_from_dt(dt)
    cache_key = (body, float(jd))
    if cache_key in _HORIZONS_CACHE:
        return _HORIZONS_CACHE[cache_key]
    try:
        obj_geo = Horizons(id=body, location=center, epochs=jd)
        vec_geo = obj_geo.vectors()
        obj_helio = Horizons(id=body, location='@sun', epochs=jd)
        vec_helio = obj_helio.vectors()
    except Exception as e:
        raise RuntimeError(f"Horizons query failed for {body} @ {dt.isoformat()}: {e}")
    def tbl_get(t, name, idx=0):
        return t[name][idx] if (name in t.colnames and len(t[name]) > idx) else None
    ra_h = tbl_get(vec_geo, 'RA')
    dec_deg = tbl_get(vec_geo, 'DEC')
    dist_au = tbl_get(vec_geo, 'delta') or tbl_get(vec_geo, 'range') or tbl_get(vec_geo, 'distance')
    x_hel = tbl_get(vec_helio, 'x') or tbl_get(vec_helio, 'X')
    y_hel = tbl_get(vec_helio, 'y') or tbl_get(vec_helio, 'Y')
    z_hel = tbl_get(vec_helio, 'z') or tbl_get(vec_helio, 'Z')
    out = {}
    try:
        out['ra_hours'] = float(ra_h) if ra_h is not None else None
    except Exception:
        out['ra_hours'] = None
    out['dec_deg'] = float(dec_deg) if dec_deg is not None else None
    out['distance_km'] = float(dist_au) * 149597870.7 if dist_au is not None else None
    out['x_helio_au'] = float(x_hel) if x_hel is not None else None
    out['y_helio_au'] = float(y_hel) if y_hel is not None else None
    out['z_helio_au'] = float(z_hel) if z_hel is not None else None
    _HORIZONS_CACHE[cache_key] = out
    return out

# --- Core computation functions ---------------------------------------------
def compute_jpl_features_for_epoch(dt: datetime, lat: float, lon: float, eph_ts_pair: Tuple[Any, Any],
                                   use_jpl_api: bool = False, combustion_deg: float = 8.5) -> Dict[str, Any]:
    eph, ts = eph_ts_pair if eph_ts_pair is not None else (None, None)
    out: Dict[str, Any] = {}
    if use_jpl_api and HORIZONS_AVAILABLE:
        for name in JPL_BODIES:
            try:
                hz = _horizons_query(name, dt, center='500@399')
            except Exception:
                hz = {}
            ra_hours = hz.get('ra_hours', None)
            dec_deg = hz.get('dec_deg', None)
            dist_km = hz.get('distance_km', None)
            xh = hz.get('x_helio_au', None)
            yh = hz.get('y_helio_au', None)
            zh = hz.get('z_helio_au', None)
            if xh is not None and yh is not None and zh is not None:
                try:
                    lon_rad = math.atan2(yh, xh)
                    lat_rad = math.atan2(zh, math.hypot(xh, yh))
                    helio_lon_deg = math.degrees(lon_rad) % 360.0
                    helio_lat_deg = math.degrees(lat_rad)
                except Exception:
                    helio_lon_deg = None
                    helio_lat_deg = None
            else:
                helio_lon_deg = None
                helio_lat_deg = None
            ang_diam_deg = None
            if dist_km is not None:
                r_km = _MEAN_RADIUS_KM.get(name, None)
                if r_km is not None and dist_km > 0:
                    ang_diam_deg = math.degrees(2.0 * math.atan(r_km / max(dist_km, 1e-6)))
            alt_deg = None
            is_above = None
            pos_angle = None
            is_combust = None
            out[f"{name}_ra_hours"] = float(ra_hours) if ra_hours is not None else np.nan
            out[f"{name}_dec_deg"] = float(dec_deg) if dec_deg is not None else np.nan
            out[f"{name}_distance_km"] = float(dist_km) if dist_km is not None else np.nan
            out[f"{name}_ang_diam_deg"] = float(ang_diam_deg) if ang_diam_deg is not None else np.nan
            out[f"{name}_helio_lon_deg"] = float(helio_lon_deg) if helio_lon_deg is not None else np.nan
            out[f"{name}_helio_lat_deg"] = float(helio_lat_deg) if helio_lat_deg is not None else np.nan
            out[f"{name}_pos_angle_deg"] = float(pos_angle) if pos_angle is not None else np.nan
            out[f"{name}_altitude_deg"] = float(alt_deg) if alt_deg is not None else np.nan
            out[f"{name}_is_above_horizon"] = bool(is_above) if is_above is not None else None
            out[f"{name}_is_combust"] = bool(is_combust) if is_combust is not None else None
        if SKYFIELD_AVAILABLE and eph is not None and ts is not None:
            try:
                extra = _compute_topocentric_for_epoch(dt, lat, lon, eph, ts)
                for k, v in extra.items():
                    out[k] = v
            except Exception:
                pass
        return out
    if SKYFIELD_AVAILABLE and eph is not None and ts is not None:
        try:
            sky_out = _compute_jpl_with_skyfield(dt, lat, lon, eph, ts, combustion_deg=combustion_deg)
            out.update(sky_out)
        except Exception:
            for name in JPL_BODIES:
                for k in ('ra_hours','dec_deg','distance_km','ang_diam_deg','helio_lon_deg','helio_lat_deg','pos_angle_deg','altitude_deg','is_above_horizon','is_combust'):
                    out[f"{name}_{k}"] = np.nan
    else:
        for name in JPL_BODIES:
            for k in ('ra_hours','dec_deg','distance_km','ang_diam_deg','helio_lon_deg','helio_lat_deg','pos_angle_deg','altitude_deg','is_above_horizon','is_combust'):
                out[f"{name}_{k}"] = np.nan
    return out

def _compute_jpl_with_skyfield(dt: datetime, lat: float, lon: float, eph, ts, combustion_deg: float = 8.5) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    t = ts.from_datetime(dt.astimezone(timezone.utc))
    earth = eph['earth']
    sun = eph['sun']
    observer = earth + Topos(latitude_degrees=float(lat), longitude_degrees=float(lon))
    for name in JPL_BODIES:
        ra_hours = np.nan; dec_deg = np.nan; dist_km = np.nan
        ang_diam_deg = np.nan; helio_lon_deg = np.nan; helio_lat_deg = np.nan
        pos_angle_deg = np.nan; alt_deg = np.nan; is_above = None; is_combust = None
        try:
            body = _body_object_from_ephem(eph, name)
            astrom = earth.at(t).observe(body).apparent()
            ra, dec, distance = astrom.radec()
            ra_hours = float(ra.hours) if hasattr(ra, 'hours') else float(ra) / 15.0
            dec_deg = float(dec.degrees) if hasattr(dec, 'degrees') else float(dec)
            if hasattr(distance, 'km'):
                dist_km = float(distance.km)
            elif hasattr(distance, 'au'):
                dist_km = float(distance.au) * 149597870.7
            r_km = _MEAN_RADIUS_KM.get(name, None)
            if r_km is not None and dist_km and dist_km > 0:
                ang_diam_deg = math.degrees(2.0 * math.atan(r_km / max(dist_km, 1e-6)))
            try:
                sun_to_body = sun.at(t).observe(body).apparent()
                helio_lon, helio_lat, helio_dist = sun_to_body.ecliptic_latlon()
                helio_lon_deg = float(helio_lon.degrees)
                helio_lat_deg = float(helio_lat.degrees)
            except Exception:
                helio_lon_deg = np.nan; helio_lat_deg = np.nan
            try:
                astrom2 = observer.at(t).observe(body).apparent()
                alt, az, distance2 = astrom2.altaz()
                alt_deg = float(alt.degrees)
                is_above = bool(alt_deg > 0.0)
            except Exception:
                alt_deg = np.nan; is_above = None
            try:
                sun_astrom = earth.at(t).observe(sun).apparent()
                sun_ra, sun_dec, _ = sun_astrom.radec()
                ra1 = math.radians(float(sun_ra.hours) * 15.0)
                dec1 = math.radians(float(sun_dec.degrees))
                ra2 = math.radians(float(ra.hours) * 15.0)
                dec2 = math.radians(float(dec_deg))
                y = math.sin(ra2 - ra1)
                x = math.cos(dec1) * math.tan(dec2) - math.sin(dec1) * math.cos(ra2 - ra1)
                pa_rad = math.atan2(y, x)
                pos_angle_deg = (math.degrees(pa_rad) + 360.0) % 360.0
            except Exception:
                pos_angle_deg = np.nan
            try:
                sun_e = sun.at(t).apparent().ecliptic_latlon()
                sun_lon = float(sun_e[0].degrees); sun_lat = float(sun_e[1].degrees)
                if not (math.isnan(helio_lon_deg) or math.isnan(helio_lat_deg)):
                    lon1 = math.radians(helio_lon_deg); lat1 = math.radians(helio_lat_deg)
                    lon2 = math.radians(sun_lon); lat2 = math.radians(sun_lat)
                    cossep = math.sin(lat1)*math.sin(lat2) + math.cos(lat1)*math.cos(lat2)*math.cos(lon1-lon2)
                    cossep = max(-1.0, min(1.0, cossep))
                    sep_deg = math.degrees(math.acos(cossep))
                    is_combust = (sep_deg < combustion_deg)
            except Exception:
                is_combust = None
        except Exception:
            pass
        out[f"{name}_ra_hours"] = float(ra_hours) if ra_hours is not None else np.nan
        out[f"{name}_dec_deg"] = float(dec_deg) if dec_deg is not None else np.nan
        out[f"{name}_distance_km"] = float(dist_km) if dist_km is not None else np.nan
        out[f"{name}_ang_diam_deg"] = float(ang_diam_deg) if ang_diam_deg is not None else np.nan
        out[f"{name}_helio_lon_deg"] = float(helio_lon_deg) if helio_lon_deg is not None else np.nan
        out[f"{name}_helio_lat_deg"] = float(helio_lat_deg) if helio_lat_deg is not None else np.nan
        out[f"{name}_pos_angle_deg"] = float(pos_angle_deg) if pos_angle_deg is not None else np.nan
        out[f"{name}_altitude_deg"] = float(alt_deg) if alt_deg is not None else np.nan
        out[f"{name}_is_above_horizon"] = bool(is_above) if is_above is not None else None
        out[f"{name}_is_combust"] = bool(is_combust) if is_combust is not None else None
    return out

# --- Swiss ephemeris (sidereal) helpers -------------------------------------
def compute_swe_sidereal_for_epoch(dt: datetime, combustion_deg: float = 8.5) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not SWEPH_AVAILABLE:
        for name in []:
            out[f"{name}_sid_long"] = np.nan
        # nodes and planets left empty in absence of swisseph
        return out
    try:
        prev_sid = swe.get_sid_mode()
    except Exception:
        prev_sid = None
    try:
        swe.set_sid_mode(swe.SIDM_LAHIRI, 0)
    except Exception:
        pass
    jd = jd_from_dt(dt)
    # produce sidereal entries for JPL_BODIES where possible
    for pname in ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']:
        pconst = getattr(swe, pname.upper(), None) if hasattr(swe, pname.upper()) else None
        try:
            res = swe.calc_ut(jd, pconst, swe.FLG_SWIEPH | swe.FLG_SPEED)
            lon = normalize_deg(float(res[0][0] if isinstance(res[0], (list, tuple)) else res[0]))
            lat = float(res[0][1] if isinstance(res[0], (list, tuple)) else res[1]) if len(res) >= 2 else np.nan
            speed = None
            try:
                sp = res[3]
                if isinstance(sp, (list, tuple)):
                    speed = float(sp[0])
                else:
                    speed = float(sp)
            except Exception:
                speed = None
            retro = bool(speed < 0.0) if speed is not None else None
        except Exception:
            lon = np.nan; lat = np.nan; retro = None
        zname, zidx = zodiac_from_long(lon) if not (lon is None or (isinstance(lon,float) and np.isnan(lon))) else (None, None)
        out[f"{pname}_sid_long"] = lon
        out[f"{pname}_sid_long_over_360"] = (lon / 360.0) if (lon is not None and not (isinstance(lon, float) and np.isnan(lon))) else np.nan
        out[f"{pname}_sid_lat"] = lat
        out[f"{pname}_zodiac"] = zname
        out[f"{pname}_zodiac_index"] = zidx
        out[f"{pname}_is_retrograde"] = retro
        out[f"{pname}_is_combust"] = None
        out[f"{pname}_altitude_deg"] = np.nan
        out[f"{pname}_is_above_horizon"] = None
    # nodes (mean/true)
    for mode in ['mean', 'true']:
        node_const = getattr(swe, 'MEAN_NODE' if mode == 'mean' else 'TRUE_NODE', None)
        try:
            res = swe.calc_ut(jd, node_const)
            node_lon = normalize_deg(float(res[0][0] if isinstance(res[0], (list, tuple)) else res[0]))
            node_lat = float(res[0][1] if isinstance(res[0], (list, tuple)) else res[1]) if len(res) >= 2 else np.nan
            rahu_lon = node_lon
            ketu_lon = normalize_deg(rahu_lon + 180.0) if not (rahu_lon is None or (isinstance(rahu_lon,float) and np.isnan(rahu_lon))) else np.nan
        except Exception:
            rahu_lon = np.nan; ketu_lon = np.nan; node_lat = np.nan
        for key, lonv in [('Rahu_' + mode, rahu_lon), ('Ketu_' + mode, ketu_lon)]:
            zname, zidx = zodiac_from_long(lonv) if not (lonv is None or (isinstance(lonv, float) and np.isnan(lonv))) else (None, None)
            out[f"{key}_sid_long"] = lonv
            out[f"{key}_sid_long_over_360"] = (lonv / 360.0) if not (lonv is None or (isinstance(lonv, float) and np.isnan(lonv))) else np.nan
            out[f"{key}_sid_lat"] = node_lat
            out[f"{key}_zodiac"] = zname
            out[f"{key}_zodiac_index"] = zidx
            out[f"{key}_is_retrograde"] = None
            out[f"{key}_is_combust"] = None
            out[f"{key}_is_above_horizon"] = None
            out[f"{key}_altitude_deg"] = np.nan
    try:
        if prev_sid is not None:
            swe.set_sid_mode(prev_sid)
        else:
            swe.set_sid_mode(0)
    except Exception:
        pass
    return out

# --- Controls generation (preserve original columns) -------------------------
def build_controls(events_df: pd.DataFrame, controls_per_event: int = 2, lead_hours: int = 48,
                   radius_km: float = 200.0, anchor_time_col: str = "time",
                   preserve_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build anchors (events + synthetic controls).

    preserve_cols: list of additional original columns to carry into the anchors
      (e.g., ['depth','mag','magType','depthError','magError']). For synthetic controls
      these columns will be set to NaN (they represent no real earthquake).
    """
    events = events_df.copy().reset_index(drop=True)
    events[anchor_time_col] = pd.to_datetime(events[anchor_time_col], utc=True, errors='coerce')

    if preserve_cols is None:
        # default list of columns commonly present in quake data:
        preserve_candidates = ["depth", "mag", "magType", "depthError", "magError"]
        preserve_cols = [c for c in preserve_candidates if c in events.columns]

    start = events[anchor_time_col].min()
    end = events[anchor_time_col].max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Event times not parseable or empty.")
    times = pd.date_range(start=start, end=end, freq='6H', tz='UTC').to_pydatetime()
    event_list = events.to_dict('records')

    controls = []
    rng = np.random.default_rng(seed=42)
    for i, ev in events.iterrows():
        sampled = 0
        tries = 0
        while sampled < controls_per_event and tries < controls_per_event * 200:
            tries += 1
            cand = rng.choice(times)
            conflict = False
            window_start = cand - timedelta(hours=lead_hours)
            window_end = cand + timedelta(hours=lead_hours)
            for other in event_list:
                try:
                    ot = pd.to_datetime(other['time']).to_pydatetime()
                except Exception:
                    continue
                if ot >= window_start and ot <= window_end:
                    try:
                        d = haversine_km(float(ev['latitude']), float(ev['longitude']), float(other['latitude']), float(other['longitude']))
                    except Exception:
                        d = float('inf')
                    if d <= radius_km:
                        conflict = True
                        break
            if conflict:
                continue
            ctrl = {
                'time': cand,
                'latitude': float(ev['latitude']),
                'longitude': float(ev['longitude']),
                'label': 0,
                'source_event_index': i
            }
            # preserve columns for events only: for controls we set these to NaN
            for c in preserve_cols:
                ctrl[c] = np.nan
            controls.append(ctrl)
            sampled += 1

    controls_df = pd.DataFrame(controls)

    # prepare events_out with preserve columns intact
    events_out = events.copy()
    events_out['label'] = 1
    events_out['source_event_index'] = events_out.index
    # ensure preserve columns exist on events_out
    for c in preserve_cols:
        if c not in events_out.columns:
            events_out[c] = np.nan

    # select standardized columns order (keep all original event columns + label/source_event_index)
    keep_cols = ['time', 'latitude', 'longitude'] + preserve_cols + ['label', 'source_event_index']
    # Ensure columns exist
    events_out = events_out.reindex(columns=[col for col in keep_cols if col in events_out.columns])
    controls_df = controls_df.reindex(columns=[col for col in keep_cols if col in controls_df.columns])

    combined = pd.concat([events_out, controls_df], axis=0, ignore_index=True).reset_index(drop=True)
    # keep original event dataframe columns too if present (so anchors has the original schema)
    return combined

# --- Top-level per-anchor computation ---------------------------------------
def compute_all_features_for_row(dt: datetime, lat: float, lon: float, eph_ts_pair: Tuple[Any, Any],
                                 use_jpl_api: bool = False, combustion_deg: float = 8.5) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if dt is None:
        for name in JPL_BODIES:
            for k in ('ra_hours','dec_deg','distance_km','ang_diam_deg','helio_lon_deg','helio_lat_deg','pos_angle_deg','altitude_deg','is_above_horizon','is_combust'):
                out[f"{name}_{k}"] = np.nan
        swe_map = compute_swe_sidereal_for_epoch(dt, combustion_deg=combustion_deg)
        out.update(swe_map)
        return out
    jpl_map = compute_jpl_features_for_epoch(dt, lat, lon, eph_ts_pair, use_jpl_api=use_jpl_api, combustion_deg=combustion_deg)
    swe_map = compute_swe_sidereal_for_epoch(dt, combustion_deg=combustion_deg)
    out.update(jpl_map)
    out.update(swe_map)
    return out

# --- Main pipeline -----------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Generate celestial dataset from earthquake CSV and synthetic controls.")
    p.add_argument("--input", "-i", default="1850-1950-EQData-MAG5.csv", help="Input earthquake CSV with time, latitude, longitude")
    p.add_argument("--out", "-o", default="events_celestial.csv", help="Output CSV path")
    p.add_argument("--controls-per-event", type=int, default=2, help="Number of synthetic control anchors per event")
    p.add_argument("--lead-hours", type=int, default=48, help="Lead window (hours) to avoid sampling near real events")
    p.add_argument("--radius-km", type=float, default=200.0, help="Radius (km) to consider for exclusion when sampling controls")
    p.add_argument("--combustion-deg", type=float, default=8.5, help="Angular degrees threshold for combustion (ecliptic separation)")
    p.add_argument("--de-file", default="de421.bsp", help="Skyfield DE file name (will be downloaded if missing)")
    p.add_argument("--max-rows", type=int, default=None, help="If set, process only the first N anchors (for testing)")
    p.add_argument("--use-jpl-api", action="store_true", help="Prefer JPL Horizons API (astroquery) for apparent positions (requires astroquery).")
    args = p.parse_args()

    # Load input
    df_in = pd.read_csv(args.input, low_memory=False)
    for c in ['time', 'latitude', 'longitude']:
        if c not in df_in.columns:
            raise SystemExit(f"Input CSV must contain column: {c}")

    # detect preserve columns among the common quake columns
    preserve_candidates = ["depth", "mag", "magType", "depthError", "magError"]
    preserve_cols = [c for c in preserve_candidates if c in df_in.columns]

    # Build anchors (events + controls) and preserve original quake columns
    print("Building synthetic controls (anchors)...")
    anchors = build_controls(df_in, controls_per_event=args.controls_per_event, lead_hours=args.lead_hours,
                             radius_km=args.radius_km, preserve_cols=preserve_cols)
    print(f"Total anchors (events + controls): {len(anchors)}")

    if args.max_rows is not None:
        anchors = anchors.iloc[:args.max_rows].reset_index(drop=True)
        print(f"Truncated to first {len(anchors)} anchors for testing.")

    # Prepare ephemeris resources
    eph_ts_pair: Optional[Tuple[Any, Any]] = None
    if SKYFIELD_AVAILABLE:
        print("Loading Skyfield ephemeris (may download de421)...")
        try:
            eph, ts = _load_skyfield_ephem(de_file=args.de_file)
            eph_ts_pair = (eph, ts)
        except Exception as e:
            print("Skyfield ephemeris load failed:", e)
            eph_ts_pair = None
    else:
        print("Skyfield not available: Skyfield-based computations disabled.")

    if args.use_jpl_api:
        if not HORIZONS_AVAILABLE:
            print("Warning: --use-jpl-api requested but astroquery.jplhorizons is not installed. Falling back to Skyfield.")
            args.use_jpl_api = False
        else:
            print("Using JPL Horizons API (astroquery). Note: queries will be cached per process to avoid repeated calls.")

    # Compute features for each anchor row (serial; can be parallelized later)
    rows_out: List[Dict[str, Any]] = []
    total = len(anchors)
    start_time = time.time()
    for i, r in anchors.iterrows():
        # r contains columns: time, latitude, longitude, optional preserve_cols, label, source_event_index
        tval = r['time']
        try:
            if isinstance(tval, str):
                dt = dt_parser.isoparse(tval)
            elif isinstance(tval, (np.datetime64, pd.Timestamp)):
                dt = pd.to_datetime(tval).to_pydatetime()
            elif isinstance(tval, datetime):
                dt = tval
            else:
                dt = dt_parser.parse(str(tval))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        except Exception:
            dt = None

        lat = None
        lon = None
        try:
            lat = float(r.get('latitude')) if not pd.isna(r.get('latitude')) else np.nan
            lon = float(r.get('longitude')) if not pd.isna(r.get('longitude')) else np.nan
        except Exception:
            lat = np.nan; lon = np.nan

        # Build base record and ensure original columns are preserved in output
        base: Dict[str, Any] = {}

        # keep the input-style time (ISO) and parsed time/jd
        base['time'] = dt.isoformat() if dt is not None else (r.get('time') if pd.notna(r.get('time')) else None)
        base['_parsed_time_'] = dt
        base['_jd_'] = jd_from_dt(dt) if dt is not None else np.nan

        # keep original lat/lon columns (and numeric aliases)
        base['latitude'] = lat if not np.isnan(lat) else r.get('latitude')
        base['longitude'] = lon if not np.isnan(lon) else r.get('longitude')
        base['latitude_num'] = lat if not np.isnan(lat) else np.nan
        base['longitude_num'] = lon if not np.isnan(lon) else np.nan

        # include preserved quake columns (depth, mag, magType, depthError, magError) when present
        for c in preserve_cols:
            # for original events these will be present in r; for controls they are NaN
            base[c] = r.get(c, np.nan)

        # label / meta columns
        label_val = int(r.get('label', 0)) if pd.notna(r.get('label', 0)) else 0
        base['label'] = label_val
        base['is_synthetic'] = (label_val == 0)
        base['source_event_index'] = r.get('source_event_index', None)

        # If missing coordinates or time, skip detailed feature computation but still output base row
        if dt is None or np.isnan(base['latitude_num']) or np.isnan(base['longitude_num']):
            rows_out.append(base)
            continue

        # compute celestial features and merge into base
        feats = compute_all_features_for_row(dt, float(base['latitude_num']), float(base['longitude_num']), eph_ts_pair, use_jpl_api=args.use_jpl_api, combustion_deg=args.combustion_deg)
        base.update(feats)
        rows_out.append(base)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            print(f"Processed {i+1}/{total} anchors... elapsed {int(elapsed)}s")

    out_df = pd.DataFrame(rows_out)

    # Ensure key original columns are present in final CSV (even if missing in some rows)
    for c in ['time', 'latitude', 'longitude'] + preserve_cols + ['label', 'is_synthetic', 'source_event_index', '_parsed_time_', '_jd_']:
        if c not in out_df.columns:
            out_df[c] = np.nan

    # write output
    out_df.to_csv(args.out, index=False)
    print(f"Wrote output to {args.out} ({len(out_df)} rows).")

if __name__ == "__main__":
    main()