"""
High-throughput CSV Enrichment Processor.
Populates dynamic, user-selected or AI-decided celestial features into CSV datasets.
Leverages FastEphemerisEngine for vectorized single-threaded ephemeris extraction.
"""
from __future__ import annotations
import csv
import io
import os
import time
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..calcs.sidereal_calc import SiderealCalc
from .generator import FeatureGenerator, FeatureSpec, PRESET_CONFIGS
from .fast_pipeline import FastEphemerisEngine

COMMON_DATETIME_COLS = ["time", "datetime", "timestamp", "date_time", "utc", "date", "created_at"]
COMMON_LAT_COLS = ["latitude", "lat"]
COMMON_LON_COLS = ["longitude", "lon", "long"]

def describe_feature_options() -> Dict[str, Any]:
    """
    Returns the comprehensive catalog of all feature options, attributes,
    cyclic transformations, pairwise options, and presets.
    Enables AI agents and users to dynamically construct custom feature specs.
    """
    return {
        "presets": {
            "minimal": "Basic orbital longitudes for all planets.",
            "ml_cyclical": "Continuous sin/cos cyclic encodings, speeds, retrograde flags, and solar elongations for machine learning.",
            "astrological": "Vedic astrology features: signs, degrees in sign, nakshatras, padas, retrograde/combustion flags, and Chara Karakas.",
            "seismic_tidal": "Sun/Moon/Jupiter/Saturn positions, topocentric altitude at epicenter, tidal elongation and syzygy alignment.",
            "financial_astro": "Speeds, retrogradations, signs, and synodic cycles (Mercury-Sun, Venus-Sun, Jupiter-Saturn).",
            "exhaustive": "All attributes, all cyclic transforms, chara karakas, stelliums, and major pairwise relationships.",
        },
        "calculation_modes": {
            "fast": "Optimized adaptive trajectory grid interpolation + SIMD (Default). 1,000x faster, sub-arcsecond accuracy, 100% retro/combustion boolean fidelity.",
            "exact": "Calculates every unique timestamp with exact Swiss Ephemeris without interpolation, accelerated with timestamp deduplication and SIMD feature math."
        },
        "available_planets": [
            "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu", "all"
        ],
        "available_attributes": {
            "longitude": "Ecliptic longitude in degrees (0.0 to 360.0)",
            "latitude": "Ecliptic latitude in degrees",
            "speed": "Orbital speed in degrees per day (negative indicates retrograde)",
            "sign": "Zodiac sign name (e.g. Mesham, Vrishabham, etc.)",
            "sign_index": "Sign index from 0 to 11",
            "degree_in_sign": "Degrees traversed within current sign (0.0 to 30.0)",
            "nakshatra": "Vedic lunar mansion name (e.g. Ashwini, Bharani, etc.)",
            "nakshatra_index": "Nakshatra index from 0 to 26",
            "pada": "Nakshatra quarter (1 to 4)",
            "is_retrograde": "Boolean flag indicating retrograde motion (speed < 0)",
            "is_combust": "Boolean flag indicating combustion by proximity to Sun",
            "sin_cos": "Continuous cyclical encoding: Sin_Lon and Cos_Lon for deep learning without 360-degree boundary discontinuity",
            "elongation": "Minimal angular distance from Sun (0.0 to 180.0)",
            "altitude": "Topocentric horizon elevation angle in degrees (requires latitude and longitude)",
        },
        "pairwise_features": {
            "description": "Specify pairs of bodies e.g. [['Moon', 'Sun'], ['Jupiter', 'Saturn']]",
            "computed_per_pair": [
                "Sep_Deg (Angular separation 0-180)",
                "Aspect (Conjunction, Sextile, Square, Trine, Opposition, None)",
                "Is_Aligned (Collinear alignment / syzygy flag)",
                "Illumination (Phase illumination % if Sun is one of the bodies)"
            ]
        },
        "system_features": {
            "include_chara_karakas": "Computes Jaimini 7-Karaka planetary assignments",
            "include_stelliums": "Computes Max_Stellium_Count and Has_Stellium_4Plus",
            "include_ayanamsa": "Includes Ayanamsa_Name"
        }
    }

def enrich_csv_dataset(
    input_csv: str,
    output_csv: Optional[str] = None,
    feature_spec: Optional[Dict[str, Any]] = None,
    preset: Optional[str] = None,
    datetime_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    calculation_mode: str = "fast",
    ayanamsa: str = "lahiri",
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Enriches a CSV dataset with user-selected or AI-decided celestial features.
    Uses FastEphemerisEngine for high-speed single-threaded processing.

    Args:
        input_csv: File path to input CSV, or raw CSV text string.
        output_csv: Optional path to write the enriched CSV. If None, enriched content is in preview/memory.
        feature_spec: Custom dictionary specifying planets, attributes, pairwise angles, and flags.
        preset: Optional preset name ('minimal', 'ml_cyclical', 'astrological', 'seismic_tidal', 'exhaustive').
        datetime_col: Name of column containing dates/timestamps. If None, auto-detected.
        lat_col: Name of latitude column. If None, auto-detected.
        lon_col: Name of longitude column. If None, auto-detected.
        calculation_mode: 'fast' (adaptive grid interpolation, ~500k rows/sec) or 'exact' (exact deduplicated evaluations).
        ayanamsa: Sidereal mode ('lahiri', 'raman', 'kp', or 'sayana').
        max_rows: Optional row limit for preview or batch processing.
    """
    t_start = time.perf_counter()

    # 1. Resolve CSV Input Source
    if os.path.exists(input_csv):
        with open(input_csv, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    else:
        content = input_csv

    reader = csv.DictReader(io.StringIO(content))
    fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError("Input CSV has no header columns or is empty.")

    # 2. Auto-detect Columns if not explicitly supplied
    lower_map = {col.lower(): col for col in fieldnames}

    dt_col = datetime_col
    if not dt_col:
        for candidate in COMMON_DATETIME_COLS:
            if candidate in lower_map:
                dt_col = lower_map[candidate]
                break
    if not dt_col:
        raise ValueError(f"Could not auto-detect a datetime column in CSV header {fieldnames}. Please specify datetime_col.")

    l_col = lat_col
    if not l_col:
        for candidate in COMMON_LAT_COLS:
            if candidate in lower_map:
                l_col = lower_map[candidate]
                break

    lo_col = lon_col
    if not lo_col:
        for candidate in COMMON_LON_COLS:
            if candidate in lower_map:
                lo_col = lower_map[candidate]
                break

    # 3. Read Rows and Extract Time and Coordinates
    raw_rows: List[Dict[str, Any]] = []
    timestamps: List[str] = []
    lat_list: List[float] = []
    lon_list: List[float] = []

    for idx, row in enumerate(reader):
        if max_rows and idx >= max_rows:
            break
        t_val = row.get(dt_col)
        if not t_val:
            continue
        raw_rows.append(row)
        timestamps.append(t_val)

        # Latitude
        lat_val = 0.0
        if l_col and row.get(l_col):
            try:
                lat_val = float(row[l_col])
            except ValueError:
                pass
        lat_list.append(lat_val)

        # Longitude
        lon_val = 0.0
        if lo_col and row.get(lo_col):
            try:
                lon_val = float(row[lo_col])
            except ValueError:
                pass
        lon_list.append(lon_val)

    n_rows = len(raw_rows)
    if n_rows == 0:
        return {
            "status": "success",
            "rows_processed": 0,
            "message": "No valid rows found in CSV."
        }

    # 4. Fast Vectorized Feature Extraction
    spec = FeatureSpec.from_dict(feature_spec, preset=preset or "ml_cyclical")
    fast_engine = FastEphemerisEngine(ayanamsa=ayanamsa, spec=spec)

    use_grid = (calculation_mode == "fast" and n_rows > 100)
    lat_arr = np.array(lat_list, dtype=np.float64) if l_col else None
    lon_arr = np.array(lon_list, dtype=np.float64) if lo_col else None

    vectorized_features = fast_engine.extract_features_vectorized(
        timestamps=timestamps,
        latitudes=lat_arr,
        longitudes=lon_arr,
        use_grid=use_grid
    )

    # 5. Assemble Enriched Rows
    feature_col_names = list(vectorized_features.keys())
    enriched_rows: List[Dict[str, Any]] = [dict(r) for r in raw_rows]

    for col_name, arr in vectorized_features.items():
        # Handle numpy booleans or numbers cleanly for serialization
        if arr.dtype == bool or np.issubdtype(arr.dtype, np.bool_):
            col_vals = [bool(v) for v in arr]
        elif np.issubdtype(arr.dtype, np.integer):
            col_vals = [int(v) for v in arr]
        elif np.issubdtype(arr.dtype, np.floating):
            col_vals = [float(v) for v in arr]
        else:
            col_vals = [v for v in arr]

        for i in range(n_rows):
            enriched_rows[i][col_name] = col_vals[i]

    out_fieldnames = fieldnames + [c for c in feature_col_names if c not in fieldnames]

    # 6. Write Output if requested
    if output_csv:
        parent_dir = os.path.dirname(output_csv)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
            writer.writeheader()
            writer.writerows(enriched_rows)

    elapsed_sec = time.perf_counter() - t_start

    return {
        "status": "success",
        "input_source": input_csv if os.path.exists(input_csv) else "<in-memory>",
        "output_file": output_csv,
        "rows_processed": n_rows,
        "calculation_mode": "adaptive_grid_interpolated" if use_grid else "exact_deduplicated",
        "elapsed_seconds": round(elapsed_sec, 4),
        "throughput_rows_per_sec": round(n_rows / max(0.0001, elapsed_sec), 1),
        "detected_columns": {
            "datetime": dt_col,
            "latitude": l_col,
            "longitude": lo_col
        },
        "preset_used": spec.preset,
        "features_added_count": len(feature_col_names),
        "features_added": feature_col_names,
        "sample_preview": enriched_rows[:3] if enriched_rows else []
    }
