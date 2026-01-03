#!/usr/bin/env python3
"""
generate_celestial_dataset.py

Generate a fresh, cleaned dataset of earthquake events + synthetic "no-event" controls
and compute rich celestial features (JPL-based apparent RA/Dec, distance, angular diameter,
heliocentric ecliptic lon/lat) and sidereal quantities (sidereal longitudes, Rahu/Ketu nodes)
using Skyfield (JPL ephemerides) + Swiss Ephemeris (swisseph) where appropriate.

Key behavior (defaults):
 - Read input CSV (default: 1850-1950-EQData-MAG5.csv). Requires columns: time, latitude, longitude
 - Parse times to timezone-aware UTC; compute Julian Day (UT) via swisseph for sidereal calcs.
 - For each event row (and for synthetic controls) compute:
    * For bodies: Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn
      - Apparent R.A. (hours)
      - Apparent Declination (degrees)
      - Distance from Earth (km)
      - Angular diameter (degrees; estimated from mean radii)
      - Heliocentric ecliptic longitude (deg)
      - Heliocentric ecliptic latitude (deg)
      - Position angle (deg) of the great-circle direction Sun -> body as seen from Earth (0 = North, increasing toward East)
      - Altitude (deg) and is_above_horizon (bool) at the event location
      - Sidereal ecliptic longitude & latitude (via swisseph in Lahiri sidereal mode)
      - sid_long_over_360 (sid_long / 360)
      - zodiac (12-sign) and zodiac_index (1..12) based on sidereal long
      - is_retrograde (from swisseph speed if available)
      - is_combust (angular separation to Sun in ecliptic coords <= combustion threshold)
    * For lunar nodes: Rahu_mean / Ketu_mean (mean node), Rahu_true / Ketu_true (true node)
      - same sidereal long/lat, zodiac, retro (nodes retro flag set to None), combustion relative to Sun
 - Synthetic controls:
    * For each event row, create `--controls-per-event` control rows by sampling anchor times uniformly
      across dataset time range. Controls are placed at the same lat/lon as their event (configurable with jitter)
      and are rejected if another real event exists within +/- lead_hours and within radius_km.
 - Output:
    * CSV with original event fields plus computed celestial columns (one wide row per anchor).
    * Summary printed to stdout.

Notes / requirements:
 - Requires Python packages: pandas, numpy, skyfield, jplephem, pyswisseph, python-dateutil
   Install in your venv: pip install -r requirements.txt
 - First Skyfield run will download de421.bsp (~few MB) if not present.
 - This script prefers Skyfield (JPL ephemeris) for RA/Dec/heliocentric coords and swisseph for sidereal/node calculations.
 - Terminology: variable names and column prefixes use "celestial" / "sid" / "helio" (no word "astrology" used).

Example usage:
  .venv/bin/python src/generate_celestial_dataset.py \
    --input 1850-1950-EQData-MAG5.csv \
    --out events_celestial.csv \
    --controls-per-event 2 \
    --lead-hours 48 \
    --radius-km 200 \
    --combustion-deg 8.5

"""
from __future__ import annotations
import argparse
import math
import warnings
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from dateutil import parser as dt_parser

# Skyfield + JPL
try:
    from skyfield.api import load, Topos
    SKYFIELD_AVAILABLE = True
    # We'll lazily load the ephemeris when needed
except Exception:
    SKYFIELD_AVAILABLE = False

# Swiss ephemeris (for sidereal mode + lunar nodes)
try:
    import swisseph as swe
    SWEPH_AVAILABLE = True
except Exception:
    SWEPH_AVAILABLE = False

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

# bodies used for JPL features and sidereal/swe features
JPL_BODIES = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']
SWE_PLANETS = {
    'Sun': getattr(swe, 'SUN', None),
    'Moon': getattr(swe, 'MOON', None),
    'Mercury': getattr(swe, 'MERCURY', None),
    'Venus': getattr(swe, 'VENUS', None),
    'Mars': getattr(swe, 'MARS', None),
    'Jupiter': getattr(swe, 'JUPITER', None),
    'Saturn': getattr(swe, 'SATURN', None)
}
NODE_TYPES = {'mean': getattr(swe, 'MEAN_NODE', None), 'true': getattr(swe, 'TRUE_NODE', None)} if SWEPH_AVAILABLE else {}

ZODIAC_12 = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# Utilities
def jd_from_dt(dt: datetime) -> float:
    """
    Return Julian Day (UT) using swisseph if available (preferred) or via formula fallback.
    """
    if SWEPH_AVAILABLE:
        # swisseph.julday(year, month, day, hour)
        dt_utc = dt.astimezone(timezone.utc)
        hour = dt_utc.hour + dt_utc.minute / 60.0 + (dt_utc.second + dt_utc.microsecond / 1e6) / 3600.0
        return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, hour)
    else:
        # approximate via skyfield if available
        # fallback: convert to Unix timestamp and convert
        ts = dt.timestamp()
        # unix epoch JD = 2440587.5
        return 2440587.5 + ts / 86400.0

def normalize_deg(x: float) -> float:
    return float(x) % 360.0

def zodiac_from_long(lon_deg: float) -> Tuple[str, int]:
    idx = int(math.floor(lon_deg / 30.0)) % 12
    return ZODIAC_12[idx], idx + 1

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))

# Skyfield helpers
def _load_skyfield_ephem(de_file: str = "de421.bsp"):
    if not SKYFIELD_AVAILABLE:
        raise RuntimeError("Skyfield not available; install skyfield and jplephem.")
    eph = load(de_file)
    ts = load.timescale()
    return eph, ts

def _body_object_from_ephem(eph, name: str):
    """
    Return skyfield body object for a given name.
    Accepts common names and barycenter fallbacks.
    """
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

def compute_jpl_features_for_epoch(dt: datetime, lat: float, lon: float, eph, ts, combustion_deg: float = 8.5) -> Dict[str, Any]:
    """
    Compute JPL-based apparent RA/Dec, distance, angular diameter, heliocentric coords,
    altitude (topocentric) and position-angle-from-sun for bodies in JPL_BODIES.
    Returns a dict mapping featurename -> value.
    """
    out: Dict[str, Any] = {}
    t = ts.from_datetime(dt.astimezone(timezone.utc))
    earth = eph['earth']
    sun = eph['sun']

    observer = earth + Topos(latitude_degrees=float(lat), longitude_degrees=float(lon))

    for name in JPL_BODIES:
        try:
            body = _body_object_from_ephem(eph, name)
        except Exception:
            # skip if body missing
            body = None

        # default fields
        ra_hours = np.nan
        dec_deg = np.nan
        dist_km = np.nan
        ang_diam_deg = np.nan
        helio_lon_deg = np.nan
        helio_lat_deg = np.nan
        alt_deg = np.nan
        is_above = None
        pos_angle_deg = np.nan
        is_combust = None

        if body is not None:
            try:
                astrom = earth.at(t).observe(body).apparent()

                # RA/Dec and distance
                ra, dec, distance = astrom.radec()
                ra_hours = float(ra.hours) if hasattr(ra, 'hours') else float(ra) / 15.0
                dec_deg = float(dec.degrees) if hasattr(dec, 'degrees') else float(dec)

                # distance conversion
                if hasattr(distance, 'km'):
                    dist_km = float(distance.km)
                elif hasattr(distance, 'au'):
                    dist_km = float(distance.au) * 149597870.7
                else:
                    dist_km = float(distance)

                # angular diameter estimate from mean radius
                r_km = _MEAN_RADIUS_KM.get(name, None)
                if r_km is not None and dist_km and dist_km > 0:
                    ang_diam_deg = math.degrees(2.0 * math.atan(r_km / max(dist_km, 1e-6)))
                else:
                    ang_diam_deg = np.nan

                # heliocentric coordinates (Sun->body)
                # use sun.at(t).observe(body)
                try:
                    sun_to_body = sun.at(t).observe(body).apparent()
                    # ecliptic latlon
                    try:
                        helio_lon, helio_lat, helio_dist = sun_to_body.ecliptic_latlon()
                        helio_lon_deg = float(helio_lon.degrees)
                        helio_lat_deg = float(helio_lat.degrees)
                    except Exception:
                        helio_lon_deg = np.nan
                        helio_lat_deg = np.nan
                except Exception:
                    helio_lon_deg = np.nan
                    helio_lat_deg = np.nan

                # topocentric altitude via skyfield
                try:
                    astrom2 = observer.at(t).observe(body).apparent()
                    alt, az, distance2 = astrom2.altaz()
                    alt_deg = float(alt.degrees)
                    is_above = bool(alt_deg > 0.0)
                except Exception:
                    alt_deg = np.nan
                    is_above = None

                # Position angle from Sun to body as seen from Earth's center
                try:
                    sun_astrom = earth.at(t).observe(sun).apparent()
                    sun_ra, sun_dec, _ = sun_astrom.radec()
                    # convert to radians
                    ra1 = math.radians(float(sun_ra.hours) * 15.0)
                    dec1 = math.radians(float(sun_dec.degrees))
                    ra2 = math.radians(float(ra.hours) * 15.0)
                    dec2 = math.radians(float(dec_deg))
                    # position angle formula (from point1 -> point2) see astronomical convention
                    y = math.sin(ra2 - ra1)
                    x = math.cos(dec1) * math.tan(dec2) - math.sin(dec1) * math.cos(ra2 - ra1)
                    pa_rad = math.atan2(y, x)
                    pos_angle_deg = (math.degrees(pa_rad) + 360.0) % 360.0
                except Exception:
                    pos_angle_deg = np.nan

                # combustion: angular separation in ecliptic coords (use helio ecliptic lon/lat of body and sun)
                try:
                    # compute sun ecliptic lon/lat via sun.at(t)
                    sun_e = sun.at(t)
                    sun_ecl = sun_e.apparent().ecliptic_latlon()
                    sun_lon = float(sun_ecl[0].degrees)
                    sun_lat = float(sun_ecl[1].degrees)
                    if not (np.isnan(helio_lon_deg) or np.isnan(helio_lat_deg)):
                        # angular separation on sphere (approx)
                        # Convert body sidereal? Use ecliptic angles computed above
                        # Use angular separation via spherical law of cosines
                        lon1 = math.radians(helio_lon_deg)
                        lat1 = math.radians(helio_lat_deg)
                        lon2 = math.radians(sun_lon)
                        lat2 = math.radians(sun_lat)
                        cossep = math.sin(lat1)*math.sin(lat2) + math.cos(lat1)*math.cos(lat2)*math.cos(lon1-lon2)
                        cossep = max(-1.0, min(1.0, cossep))
                        sep_deg = math.degrees(math.acos(cossep))
                        is_combust = (sep_deg < combustion_deg)
                    else:
                        is_combust = None
                except Exception:
                    is_combust = None

            except Exception:
                # leave defaults if skyfield calc failed
                pass

        # assign outputs with clear column names
        prefix = name
        out[f"{prefix}_ra_hours"] = ra_hours
        out[f"{prefix}_dec_deg"] = dec_deg
        out[f"{prefix}_distance_km"] = dist_km
        out[f"{prefix}_ang_diam_deg"] = ang_diam_deg
        out[f"{prefix}_helio_lon_deg"] = helio_lon_deg
        out[f"{prefix}_helio_lat_deg"] = helio_lat_deg
        out[f"{prefix}_pos_angle_deg"] = pos_angle_deg
        out[f"{prefix}_altitude_deg"] = alt_deg
        out[f"{prefix}_is_above_horizon"] = is_above
        out[f"{prefix}_is_combust"] = is_combust

    return out

# Swiss ephemeris (sidereal) helpers
def compute_swe_sidereal_for_epoch(dt: datetime, combustion_deg: float = 8.5) -> Dict[str, Any]:
    """
    Using swisseph compute sidereal longitudes, latitudes, zodiac index, retrograde flag
    for planets and nodes. Returns a dict of values keyed by e.g., Sun_sid_long, Rahu_true_sid_long, etc.
    """
    out: Dict[str, Any] = {}
    if not SWEPH_AVAILABLE:
        # fill NaNs
        for name in list(SWE_PLANETS.keys()) + ['Rahu_mean', 'Ketu_mean', 'Rahu_true', 'Ketu_true']:
            out[f"{name}_sid_long"] = np.nan
            out[f"{name}_sid_lat"] = np.nan
            out[f"{name}_sid_long_over_360"] = np.nan
            out[f"{name}_zodiac"] = None
            out[f"{name}_zodiac_index"] = np.nan
            out[f"{name}_is_retrograde"] = None
            out[f"{name}_is_combust"] = None
            out[f"{name}_altitude_deg"] = np.nan
            out[f"{name}_is_above_horizon"] = None
        return out

    # set sidereal mode to Lahiri for sidereal longitudes
    try:
        prev_sid = swe.get_sid_mode()
    except Exception:
        prev_sid = None
    try:
        swe.set_sid_mode(swe.SIDM_LAHIRI, 0)
    except Exception:
        pass

    jd = jd_from_dt(dt)
    # planets
    for pname, pconst in SWE_PLANETS.items():
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

        # zodiac
        zname, zidx = (None, np.nan)
        try:
            zname, zidx = zodiac_from_long(lon)
        except Exception:
            pass

        out[f"{pname}_sid_long"] = lon
        out[f"{pname}_sid_long_over_360"] = (lon / 360.0) if not np.isnan(lon) else np.nan
        out[f"{pname}_sid_lat"] = lat
        out[f"{pname}_zodiac"] = zname
        out[f"{pname}_zodiac_index"] = zidx
        out[f"{pname}_is_retrograde"] = retro
        out[f"{pname}_is_combust"] = None
        out[f"{pname}_altitude_deg"] = np.nan
        out[f"{pname}_is_above_horizon"] = None

    # nodes (mean and true)
    for m in ['mean', 'true']:
        node_const = NODE_TYPES.get(m, None)
        label_prefix = f"Rahu_{m}"
        try:
            if node_const is None:
                raise Exception("no node const")
            res = swe.calc_ut(jd, node_const)
            node_lon = normalize_deg(float(res[0][0] if isinstance(res[0], (list, tuple)) else res[0]))
            node_lat = float(res[0][1] if isinstance(res[0], (list, tuple)) else res[1]) if len(res)>=2 else np.nan
            rahu_lon = node_lon
            ketu_lon = normalize_deg(rahu_lon + 180.0)
        except Exception:
            rahu_lon = np.nan; node_lat = np.nan; ketu_lon = np.nan

        for key, lonv in [('Rahu_' + m, rahu_lon), ('Ketu_' + m, ketu_lon)]:
            zname, zidx = (None, np.nan)
            try:
                zname, zidx = zodiac_from_long(lonv)
            except Exception:
                pass
            out[f"{key}_sid_long"] = lonv
            out[f"{key}_sid_long_over_360"] = (lonv / 360.0) if not np.isnan(lonv) else np.nan
            out[f"{key}_sid_lat"] = node_lat
            out[f"{key}_zodiac"] = zname
            out[f"{key}_zodiac_index"] = zidx
            out[f"{key}_is_retrograde"] = None
            out[f"{key}_is_combust"] = None
            out[f"{key}_is_above_horizon"] = None
            out[f"{key}_altitude_deg"] = np.nan

    # restore sidereal mode
    try:
        if prev_sid is not None:
            swe.set_sid_mode(prev_sid)
        else:
            swe.set_sid_mode(0)
    except Exception:
        pass

    return out

# High-level pipeline
def build_controls(events_df: pd.DataFrame, controls_per_event: int = 2, lead_hours: int = 48, radius_km: float = 200.0,
                   anchor_time_col: str = "time"):
    """
    For each event row, sample `controls_per_event` candidate anchor times uniformly across the full time range.
    Reject any sample for which an actual event occurs within +/- lead_hours and within radius_km.
    Place controls at same lat/lon as the event (optionally could jitter).
    Returns a DataFrame of control rows (with label=0) and original events labeled label=1.
    """
    events = events_df.copy().reset_index(drop=True)
    events[anchor_time_col] = pd.to_datetime(events[anchor_time_col], utc=True, errors='coerce')
    # time grid for candidates (6-hour grid for sampling)
    start = events[anchor_time_col].min()
    end = events[anchor_time_col].max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Event times not parseable or empty.")
    times = pd.date_range(start=start, end=end, freq='6H', tz='UTC')
    times = times.to_pydatetime()

    event_list = events[['time', 'latitude', 'longitude']].to_dict('records')

    controls = []
    rng = np.random.default_rng(seed=42)
    for i, ev in events.iterrows():
        sampled = 0
        tries = 0
        while sampled < controls_per_event and tries < controls_per_event * 200:
            tries += 1
            cand = rng.choice(times)
            # check for conflicts
            conflict = False
            window_start = cand - timedelta(hours=lead_hours)
            window_end = cand + timedelta(hours=lead_hours)
            for other in event_list:
                try:
                    ot = pd.to_datetime(other['time']).to_pydatetime()
                except Exception:
                    continue
                if ot >= window_start and ot <= window_end:
                    d = haversine_km(float(ev['latitude']), float(ev['longitude']), float(other['latitude']), float(other['longitude']))
                    if d <= radius_km:
                        conflict = True
                        break
            if conflict:
                continue
            controls.append({
                'time': cand,
                'latitude': float(ev['latitude']),
                'longitude': float(ev['longitude']),
                'label': 0,
                'source_event_index': i
            })
            sampled += 1
    controls_df = pd.DataFrame(controls)
    # label events
    events_out = events.copy()
    events_out['label'] = 1
    events_out['source_event_index'] = events_out.index
    events_out = events_out[['time', 'latitude', 'longitude', 'label', 'source_event_index']]
    # unify controls columns
    controls_df = controls_df[['time', 'latitude', 'longitude', 'label', 'source_event_index']]
    combined = pd.concat([events_out, controls_df], axis=0, ignore_index=True).reset_index(drop=True)
    return combined

def compute_all_features_for_row(dt: datetime, lat: float, lon: float, eph, ts, combustion_deg: float = 8.5) -> Dict[str, Any]:
    """
    Compute both JPL features and Swiss-ephemeris sidereal/node features and return merged dict.
    """
    out: Dict[str, Any] = {}
    # JPL features (RA/Dec, distance, heliocentric, altitude, PA, ang diam)
    try:
        jpl = compute_jpl_features_for_epoch(dt, lat, lon, eph, ts, combustion_deg=combustion_deg)
    except Exception as e:
        # if skyfield fails, fill NaNs
        jpl = {}
        for name in JPL_BODIES:
            for k in ('ra_hours','dec_deg','distance_km','ang_diam_deg','helio_lon_deg','helio_lat_deg','pos_angle_deg','altitude_deg','is_above_horizon','is_combust'):
                jpl[f"{name}_{k}"] = np.nan
    out.update(jpl)

    # Swiss ephemeris sidereal features + nodes
    try:
        swe_map = compute_swe_sidereal_for_epoch(dt, combustion_deg=combustion_deg)
    except Exception:
        swe_map = {}
    out.update(swe_map)

    return out

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
    args = p.parse_args()

    # Load input
    df_in = pd.read_csv(args.input, low_memory=False)
    required = ['time', 'latitude', 'longitude']
    for c in required:
        if c not in df_in.columns:
            raise SystemExit(f"Input CSV must contain column: {c}")

    # Build anchors (events + controls)
    print("Building controls...")
    anchors = build_controls(df_in, controls_per_event=args.controls_per_event, lead_hours=args.lead_hours, radius_km=args.radius_km)
    print(f"Total anchors (events + controls): {len(anchors)}")

    # Optionally truncate for testing
    if args.max_rows is not None:
        anchors = anchors.iloc[:args.max_rows].reset_index(drop=True)
        print(f"Truncated to first {len(anchors)} anchors for testing.")

    # Prepare skyfield ephemeris
    eph = None; ts = None
    if SKYFIELD_AVAILABLE:
        print("Loading JPL ephemeris (this may download de421.bsp on first run)...")
        eph, ts = _load_skyfield_ephem(de_file=args.de_file)
    else:
        warnings.warn("Skyfield not available - JPL features will be empty. Install skyfield and jplephem.")

    # Compute features for each anchor row (serial; can be parallelized later)
    rows_out: List[Dict[str, Any]] = []
    total = len(anchors)
    for i, r in anchors.iterrows():
        tval = r['time']
        try:
            if isinstance(tval, str):
                dt = dt_parser.isoparse(tval)
            elif isinstance(tval, (np.datetime64, pd.Timestamp)):
                dt = pd.to_datetime(tval).to_pydatetime()
            elif isinstance(tval, datetime):
                dt = tval
            else:
                # attempt parse
                dt = dt_parser.parse(str(tval))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        except Exception:
            print(f"Warning: could not parse time for row {i}: {tval}; skipping features")
            dt = None

        lat = float(r['latitude']) if not pd.isna(r['latitude']) else np.nan
        lon = float(r['longitude']) if not pd.isna(r['longitude']) else np.nan

        base = {
            'time': dt.isoformat() if dt is not None else None,
            '_parsed_time_': dt,
            '_jd_': jd_from_dt(dt) if dt is not None else np.nan,
            'latitude_num': lat,
            'longitude_num': lon,
            'label': int(r.get('label', 0)),
            'source_event_index': r.get('source_event_index', None)
        }

        if dt is None or np.isnan(lat) or np.isnan(lon):
            # append base, with NaNs for features
            rows_out.append(base)
            continue

        feats = compute_all_features_for_row(dt, lat, lon, eph, ts, combustion_deg=args.combustion_deg)
        base.update(feats)
        rows_out.append(base)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"Processed {i+1}/{total} anchors...")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote output to {args.out} ({len(out_df)} rows).")

if __name__ == "__main__":
    main()
