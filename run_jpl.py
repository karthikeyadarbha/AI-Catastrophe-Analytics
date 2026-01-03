#!/usr/bin/env python3
"""
run_jpl.py

CLI wrapper to compute JPL ephemeris features for a single datetime or CSV and print/save results.
Uses Skyfield + JPL ephemeris (de421.bsp by default) to compute apparent RA/Dec, distance,
angular diameter, heliocentric ecliptic coordinates, and position angles for celestial bodies.

Usage examples:
  # Single datetime:
  python run_jpl.py --datetime "2024-01-15T12:00:00Z" --lat 40.7128 --lon -74.0060

  # Process CSV file:
  python run_jpl.py --input events.csv --output events_with_jpl.csv

  # With custom DE file:
  python run_jpl.py --input events.csv --de-file de430.bsp --output jpl_out.csv
"""
import argparse
import sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from dateutil import parser as dt_parser

# Import JPL computation functions from generate_celestial_dataset
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from generate_celestial_dataset import (
        compute_jpl_features_for_epoch, 
        _load_skyfield_ephem,
        SKYFIELD_AVAILABLE
    )
except ImportError:
    # Fallback: try direct import
    try:
        from src.generate_celestial_dataset import (
            compute_jpl_features_for_epoch,
            _load_skyfield_ephem,
            SKYFIELD_AVAILABLE
        )
    except ImportError:
        print("ERROR: Cannot import from generate_celestial_dataset.py", file=sys.stderr)
        print("Make sure src/generate_celestial_dataset.py exists", file=sys.stderr)
        sys.exit(1)

def parse_datetime(dt_str):
    """Parse datetime string to timezone-aware UTC datetime."""
    try:
        dt = dt_parser.isoparse(dt_str)
    except:
        dt = dt_parser.parse(dt_str)
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def process_single_datetime(dt_str, lat, lon, eph, ts, combustion_deg=8.5):
    """Compute and print JPL features for a single datetime."""
    dt = parse_datetime(dt_str)
    features = compute_jpl_features_for_epoch(dt, lat, lon, eph, ts, combustion_deg)
    
    print(f"\nJPL Ephemeris Features for {dt.isoformat()}")
    print(f"Location: Latitude={lat:.4f}, Longitude={lon:.4f}")
    print("=" * 70)
    
    # Group features by body
    bodies = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']
    for body in bodies:
        print(f"\n{body}:")
        for key, value in features.items():
            if key.startswith(f"{body}_"):
                suffix = key[len(body)+1:]
                if isinstance(value, (int, float)):
                    if not np.isnan(value):
                        print(f"  {suffix:20s}: {value:.6f}")
                    else:
                        print(f"  {suffix:20s}: NaN")
                else:
                    print(f"  {suffix:20s}: {value}")
    
    return features

def process_csv(input_path, output_path, eph, ts, combustion_deg=8.5):
    """Process CSV file and add JPL features to each row."""
    print(f"Reading input CSV: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    
    # Check for required columns
    required = ['time', 'latitude', 'longitude']
    for col in required:
        if col not in df.columns:
            print(f"ERROR: Input CSV must contain column: {col}", file=sys.stderr)
            sys.exit(1)
    
    print(f"Processing {len(df)} rows...")
    
    results = []
    for idx, row in df.iterrows():
        try:
            dt_str = str(row['time'])
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            
            if pd.isna(row['time']) or pd.isna(lat) or pd.isna(lon):
                results.append({})
                continue
            
            dt = parse_datetime(dt_str)
            features = compute_jpl_features_for_epoch(dt, lat, lon, eph, ts, combustion_deg)
            results.append(features)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx+1}/{len(df)} rows...")
        except Exception as e:
            print(f"Warning: Error processing row {idx}: {e}", file=sys.stderr)
            results.append({})
    
    # Combine original DataFrame with computed features
    results_df = pd.DataFrame(results)
    output_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
    
    print(f"Writing output to: {output_path}")
    output_df.to_csv(output_path, index=False)
    print(f"Done! Wrote {len(output_df)} rows with JPL features.")

def main():
    parser = argparse.ArgumentParser(
        description="Compute JPL ephemeris features for single datetime or CSV file"
    )
    parser.add_argument("--datetime", "-d", help="ISO 8601 datetime string (e.g., '2024-01-15T12:00:00Z')")
    parser.add_argument("--lat", type=float, help="Latitude in degrees (required with --datetime)")
    parser.add_argument("--lon", type=float, help="Longitude in degrees (required with --datetime)")
    parser.add_argument("--input", "-i", help="Input CSV file with time, latitude, longitude columns")
    parser.add_argument("--output", "-o", help="Output CSV file (required with --input)")
    parser.add_argument("--de-file", default="de421.bsp", help="Skyfield DE file (default: de421.bsp)")
    parser.add_argument("--combustion-deg", type=float, default=8.5, 
                       help="Combustion threshold in degrees (default: 8.5)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.datetime and not (args.lat is not None and args.lon is not None):
        parser.error("--datetime requires --lat and --lon")
    
    if args.input and not args.output:
        parser.error("--input requires --output")
    
    if not args.datetime and not args.input:
        parser.error("Must provide either --datetime or --input")
    
    if not SKYFIELD_AVAILABLE:
        print("ERROR: Skyfield not available. Install: pip install skyfield jplephem", file=sys.stderr)
        sys.exit(1)
    
    # Load JPL ephemeris
    print(f"Loading JPL ephemeris ({args.de_file})...")
    try:
        eph, ts = _load_skyfield_ephem(de_file=args.de_file)
    except Exception as e:
        print(f"ERROR loading ephemeris: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process single datetime or CSV
    if args.datetime:
        process_single_datetime(args.datetime, args.lat, args.lon, eph, ts, args.combustion_deg)
    else:
        process_csv(args.input, args.output, eph, ts, args.combustion_deg)

if __name__ == "__main__":
    main()
