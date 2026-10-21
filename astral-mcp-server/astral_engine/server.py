"""
Astral Ephemeris FastMCP Server.
Exposes high-performance celestial mechanics, astrological state machines, and temporal interval queries.
Treats all celestial bodies with uniform, symmetrical mathematical formulations.
Supports dynamic, user-selected or AI-decided feature generation and CSV dataset enrichment.
"""
from __future__ import annotations
import json
from typing import Any, List, Optional
from mcp.server.fastmcp import FastMCP

from astral_engine.core.models import Planet, DateRange
from astral_engine.calcs.sidereal_calc import SiderealCalc
from astral_engine.calcs.chara_karaka import compute_chara_karakas, enrich_with_chara_karakas
from astral_engine.algorithms.transit_tracker import TransitTracker
from astral_engine.algorithms.motion_tracker import MotionTracker
from astral_engine.algorithms.aspect_engine import AspectEngine
from astral_engine.query.evaluator import QueryEvaluator
from astral_engine.utils import parse_datetime, minimal_angular_diff
from astral_engine.features import (
    FeatureGenerator,
    FeatureSpec,
    enrich_csv_dataset as run_enrich_csv,
    describe_feature_options as run_describe_feature_options,
)

# Initialize FastMCP Server
mcp = FastMCP("astral-ephemeris")

@mcp.tool()
def get_ephemeris_snapshot(
    datetime_str: str,
    ayanamsa: str = "lahiri",
    lat: float = 0.0,
    lon: float = 0.0,
    elev_m: float = 0.0,
) -> str:
    """
    Get instantaneous positions of all 9 Vedic grahas, zodiac signs, nakshatras, padas,
    degrees, speeds, retrograde status, combustion, and Jaimini Chara Karakas.
    Optionally computes topocentric horizon elevation (altitude_deg) if lat/lon are given.
    Treats all celestial bodies uniformly.

    Args:
        datetime_str: ISO-8601 UTC date string, e.g. '2025-07-29T12:00:00Z'
        ayanamsa: Sidereal mode ('lahiri', 'raman', 'kp', or 'sayana' for tropical)
        lat: Observer latitude (degrees)
        lon: Observer longitude (degrees)
        elev_m: Observer elevation (meters)
    """
    dt = parse_datetime(datetime_str)
    calc = SiderealCalc(ayanamsa=ayanamsa)

    # 1. Compute positions for all bodies
    all_positions = calc.get_all_positions(dt, lat=lat, lon=lon, elev=elev_m)

    # 2. Enrich with Chara Karakas
    karaka_map = compute_chara_karakas(all_positions)
    enrich_with_chara_karakas(all_positions)

    output = {
        "datetime": dt.isoformat(),
        "ayanamsa": ayanamsa,
        "is_sidereal": calc.is_sidereal,
        "planets": {p.value: pos.to_dict() for p, pos in all_positions.items()},
        "chara_karakas": karaka_map,
    }
    return json.dumps(output, indent=2)

@mcp.tool()
def get_planet_timeline(
    planet: str,
    start_date: str,
    end_date: str,
    ayanamsa: str = "lahiri",
) -> str:
    """
    Get a chronological timeline of all transitions (Rāśi entry, Nakṣatra entry,
    retrograde/direct stations) for any planet over a date range.

    Args:
        planet: Name of the planet (e.g. 'Mars', 'Saturn', 'Jupiter', 'Mercury', 'Moon', 'Sun')
        start_date: ISO date string 'YYYY-MM-DD'
        end_date: ISO date string 'YYYY-MM-DD'
        ayanamsa: 'lahiri', 'raman', 'kp', or 'sayana'
    """
    p = Planet.from_string(planet)
    drange = DateRange(parse_datetime(start_date), parse_datetime(end_date))
    calc = SiderealCalc(ayanamsa=ayanamsa)

    transit_tracker = TransitTracker(calc)
    motion_tracker = MotionTracker(calc)

    # Transits
    sign_events = transit_tracker.find_sign_transits(p, drange)
    nak_events = transit_tracker.find_nakshatra_transits(p, drange)

    # Merge and sort chronologically
    all_events = sorted(sign_events + nak_events, key=lambda e: e.timestamp)

    # Retrograde periods
    retro_intervals = motion_tracker.find_retrograde_periods(p, drange)

    output = {
        "planet": p.value,
        "start": drange.start.isoformat(),
        "end": drange.end.isoformat(),
        "transitions": [e.to_dict() for e in all_events],
        "retrograde_periods": retro_intervals.to_dicts(),
    }
    return json.dumps(output, indent=2)

@mcp.tool()
def query_celestial_intervals(
    start_date: str,
    end_date: str,
    filters: list[dict],
    operator: str = "AND",
    min_duration_hours: float = 0.0,
    ayanamsa: str = "lahiri",
) -> str:
    """
    Declarative query engine: Evaluates composite temporal conditions using Allen's Interval Algebra.
    Returns the exact date windows where all (AND) or any (OR) conditions match.
    Supports symmetric querying across any planet or pair of planets.

    Filters specification example:
    [
        {"planet": "Mars", "condition": "motion", "value": "Retrograde"},
        {"planet": "Saturn", "condition": "sign", "value": ["Kumbham", "Meenam"]},
        {"planet": "Mercury", "condition": "combust", "value": true},
        {"planet": "Jupiter", "condition": "chara_karaka", "value": "AK"},
        {"condition": "aspect", "planet": "Jupiter", "target_planet": "Saturn", "aspect": "Conjunction"},
        {"condition": "alignment", "planet": "Moon", "target_planet": "Sun", "orb": 12.0},
        {"condition": "stellium", "value": 4}
    ]
    """
    calc = SiderealCalc(ayanamsa=ayanamsa)
    evaluator = QueryEvaluator(calc)

    matching_intervals = evaluator.execute_query(
        start_date=start_date,
        end_date=end_date,
        filters=filters,
        operator=operator,
        min_duration_hours=min_duration_hours,
    )

    output = {
        "query": {
            "start": start_date,
            "end": end_date,
            "operator": operator,
            "filters": filters,
        },
        "matches_count": len(matching_intervals),
        "intervals": matching_intervals,
    }
    return json.dumps(output, indent=2)

@mcp.tool()
def get_angular_relationship(
    body1: str,
    body2: str = "Sun",
    datetime_str: str = "2025-01-01T00:00:00Z",
    orb_deg: float = 6.0,
    ayanamsa: str = "lahiri",
) -> str:
    """
    Computes the exact geometric angular relationship between ANY two celestial bodies
    (e.g., Moon & Sun, Jupiter & Saturn, Mars & Sun, Venus & Sun).
    Treats all celestial bodies with pure geometric symmetry.

    Returns:
    - Angular separation (0° - 180°)
    - Directional longitudinal difference (0° - 360°, body1 relative to body2)
    - Astrological aspect (Conjunction, Sextile, Square, Trine, Opposition) within orb
    - Collinear alignment / Syzygy flag (collinear within orb tolerance)
    - Illumination percentage (general phase formula when body1 or body2 is Sun)
    """
    dt = parse_datetime(datetime_str)
    calc = SiderealCalc(ayanamsa=ayanamsa)
    aspect_engine = AspectEngine(calc)
    res = aspect_engine.get_angular_relationship(
        body1=body1,
        body2=body2,
        dt=dt,
        orb_deg=orb_deg,
    )
    return json.dumps(res, indent=2)

@mcp.tool()
def describe_feature_options() -> str:
    """
    Returns the comprehensive catalog of all available feature options, attributes,
    cyclic transformations, pairwise options, and presets.
    Call this first so you or an AI agent can inspect what is possible and compose
    the exact feature specification needed for any analysis or ML task.
    """
    options = run_describe_feature_options()
    return json.dumps(options, indent=2)

@mcp.tool()
def enrich_csv_dataset(
    input_csv_path: str,
    output_csv_path: Optional[str] = None,
    feature_spec: Optional[dict] = None,
    preset: Optional[str] = None,
    datetime_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    calculation_mode: str = "fast",
    ayanamsa: str = "lahiri",
    max_rows: Optional[int] = None,
) -> str:
    """
    Takes dates and times in a CSV file and dynamically populates whatever features
    the user wants or an AI decides based on user input.

    Optimized for high-speed single-threaded processing without multiprocessing.

    Args:
        input_csv_path: Absolute file path to the input CSV file, or raw CSV text.
        output_csv_path: Optional path to write the enriched CSV file. If None, preview and metadata are returned.
        feature_spec: Custom specification dictionary:
            {
                "planets": ["Sun", "Moon", "Jupiter", "Saturn"],
                "attributes": ["longitude", "speed", "is_retrograde", "sin_cos", "altitude", "elongation"],
                "pairwise_angles": [["Moon", "Sun"], ["Jupiter", "Saturn"]],
                "include_chara_karakas": false,
                "include_stelliums": true
            }
        preset: Shortcut preset ('minimal', 'ml_cyclical', 'astrological', 'seismic_tidal', 'financial_astro', 'exhaustive')
        datetime_col: Column name containing dates/times (auto-detected if omitted)
        lat_col: Column name for latitude (auto-detected if omitted)
        lon_col: Column name for longitude (auto-detected if omitted)
        calculation_mode: 'fast' (adaptive grid interpolation, ~500k rows/sec) or 'exact' (exact deduplicated evaluations).
        ayanamsa: Sidereal mode ('lahiri', 'raman', 'kp', or 'sayana')
        max_rows: Optional limit on rows to process (useful for previewing or testing large datasets)
    """
    res = run_enrich_csv(
        input_csv=input_csv_path,
        output_csv=output_csv_path,
        feature_spec=feature_spec,
        preset=preset,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lon_col=lon_col,
        calculation_mode=calculation_mode,
        ayanamsa=ayanamsa,
        max_rows=max_rows,
    )
    return json.dumps(res, indent=2)

@mcp.tool()
def batch_enrich_catalog(
    records: list[dict],
    feature_spec: Optional[dict] = None,
    preset: Optional[str] = "ml_cyclical",
    ayanamsa: str = "lahiri",
) -> str:
    """
    Enrich an in-memory list of records (each containing a datetime/time key and optional coordinates)
    with dynamically customized celestial features specified by the user or AI.
    """
    calc = SiderealCalc(ayanamsa=ayanamsa)
    spec = FeatureSpec.from_dict(feature_spec, preset=preset)
    generator = FeatureGenerator(calc, spec)

    enriched = []
    for r in records:
        time_str = r.get("time") or r.get("datetime") or r.get("timestamp") or r.get("date")
        if not time_str:
            continue
        try:
            dt = parse_datetime(time_str)
            lat = float(r.get("lat") or r.get("latitude") or 0.0)
            lon = float(r.get("lon") or r.get("longitude") or 0.0)
            
            gen_features = generator.generate(dt, lat=lat, lon=lon)
            enriched.append({**r, **gen_features})
        except Exception as e:
            enriched.append({**r, "_error": str(e)})

    return json.dumps(enriched, indent=2)

def main():
    mcp.run()

if __name__ == "__main__":
    main()
