#!/usr/bin/env python3
"""
merge_astro_jpl.py

Improved merger for updated.csv (containing sidereal/astrology features)
and events_with_jpl.csv (containing JPL ephemeris features) with spatial fallback.

The script matches rows based on:
1. Exact time match (primary)
2. Spatial proximity fallback (nearest neighbor within threshold)

Usage:
  python merge_astro_jpl.py --astro updated.csv --jpl events_with_jpl.csv --out merged.csv

Optional arguments:
  --time-tolerance-sec : Time matching tolerance in seconds (default: 1.0)
  --spatial-threshold-km : Max distance for spatial fallback in km (default: 10.0)
  --primary-key : Which dataset to use as primary (default: 'astro')
"""
import argparse
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from dateutil import parser as dt_parser

def parse_time(t):
    """Parse time to datetime object."""
    if pd.isna(t):
        return None
    if isinstance(t, (datetime, pd.Timestamp)):
        dt = pd.to_datetime(t).to_pydatetime()
    else:
        try:
            dt = dt_parser.isoparse(str(t))
        except:
            dt = dt_parser.parse(str(t))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in km between two points."""
    R = 6371.0  # Earth radius in km
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c

def merge_datasets(astro_df, jpl_df, time_tolerance_sec=1.0, spatial_threshold_km=10.0, primary='astro'):
    """
    Merge astro and JPL datasets with time-based primary matching and spatial fallback.
    
    Args:
        astro_df: DataFrame with sidereal/astrology features
        jpl_df: DataFrame with JPL ephemeris features
        time_tolerance_sec: Time difference tolerance in seconds for exact match
        spatial_threshold_km: Maximum distance in km for spatial fallback
        primary: Which dataset to use as primary ('astro' or 'jpl')
    
    Returns:
        Merged DataFrame
    """
    # Parse times
    print("Parsing timestamps...")
    astro_df = astro_df.copy()
    jpl_df = jpl_df.copy()
    
    astro_df['_parsed_time'] = astro_df['time'].apply(parse_time)
    jpl_df['_parsed_time'] = jpl_df['time'].apply(parse_time)
    
    # Determine primary and secondary datasets
    if primary == 'astro':
        primary_df = astro_df
        secondary_df = jpl_df
        primary_name = "astro"
        secondary_name = "jpl"
    else:
        primary_df = jpl_df
        secondary_df = astro_df
        primary_name = "jpl"
        secondary_name = "astro"
    
    print(f"Primary dataset ({primary_name}): {len(primary_df)} rows")
    print(f"Secondary dataset ({secondary_name}): {len(secondary_df)} rows")
    
    # Build results
    merged_rows = []
    time_matches = 0
    spatial_matches = 0
    no_matches = 0
    
    time_tolerance = timedelta(seconds=time_tolerance_sec)
    
    for idx, prow in primary_df.iterrows():
        p_time = prow['_parsed_time']
        p_lat = prow.get('latitude', np.nan)
        p_lon = prow.get('longitude', np.nan)
        
        if p_time is None or pd.isna(p_lat) or pd.isna(p_lon):
            # Can't match without time and location
            merged_rows.append(prow.to_dict())
            no_matches += 1
            continue
        
        # Try time-based match first
        match_found = False
        for sidx, srow in secondary_df.iterrows():
            s_time = srow['_parsed_time']
            if s_time is None:
                continue
            
            time_diff = abs((p_time - s_time).total_seconds())
            if time_diff <= time_tolerance_sec:
                # Exact time match - merge the rows
                merged = prow.to_dict()
                for key, value in srow.items():
                    if key not in merged or pd.isna(merged.get(key)):
                        merged[key] = value
                    elif key.startswith('_'):
                        # Skip internal columns
                        continue
                    elif key in ['time', 'latitude', 'longitude']:
                        # Keep primary values for these
                        continue
                    else:
                        # Append suffix for conflicting columns
                        merged[f"{key}_{secondary_name}"] = value
                
                merged_rows.append(merged)
                match_found = True
                time_matches += 1
                break
        
        if match_found:
            continue
        
        # Try spatial fallback
        min_dist = float('inf')
        best_match_idx = None
        
        for sidx, srow in secondary_df.iterrows():
            s_lat = srow.get('latitude', np.nan)
            s_lon = srow.get('longitude', np.nan)
            s_time = srow['_parsed_time']
            
            if pd.isna(s_lat) or pd.isna(s_lon) or s_time is None:
                continue
            
            # Calculate distance
            dist = haversine_km(p_lat, p_lon, s_lat, s_lon)
            
            # Also consider time proximity for spatial matches
            time_diff_hours = abs((p_time - s_time).total_seconds()) / 3600.0
            
            # Weighted distance: prioritize spatial proximity but consider time
            weighted_dist = dist + time_diff_hours * 0.1  # Add 0.1 km per hour difference
            
            if weighted_dist < min_dist and dist <= spatial_threshold_km:
                min_dist = weighted_dist
                best_match_idx = sidx
        
        if best_match_idx is not None:
            # Spatial match found
            srow = secondary_df.loc[best_match_idx]
            merged = prow.to_dict()
            for key, value in srow.items():
                if key not in merged or pd.isna(merged.get(key)):
                    merged[key] = value
                elif key.startswith('_'):
                    continue
                elif key in ['time', 'latitude', 'longitude']:
                    continue
                else:
                    merged[f"{key}_{secondary_name}_spatial"] = value
            
            merged_rows.append(merged)
            spatial_matches += 1
        else:
            # No match found - keep primary row as is
            merged_rows.append(prow.to_dict())
            no_matches += 1
    
    result_df = pd.DataFrame(merged_rows)
    
    # Clean up internal columns
    result_df = result_df.drop(columns=[c for c in result_df.columns if c.startswith('_')], errors='ignore')
    
    print(f"\nMerge statistics:")
    print(f"  Time-based matches: {time_matches}")
    print(f"  Spatial fallback matches: {spatial_matches}")
    print(f"  No matches (primary only): {no_matches}")
    print(f"  Total rows in output: {len(result_df)}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(
        description="Merge astrology and JPL datasets with spatial fallback"
    )
    parser.add_argument("--astro", "-a", required=True, help="Input CSV with sidereal/astrology features")
    parser.add_argument("--jpl", "-j", required=True, help="Input CSV with JPL ephemeris features")
    parser.add_argument("--out", "-o", required=True, help="Output merged CSV file")
    parser.add_argument("--time-tolerance-sec", type=float, default=1.0,
                       help="Time matching tolerance in seconds (default: 1.0)")
    parser.add_argument("--spatial-threshold-km", type=float, default=10.0,
                       help="Maximum distance for spatial fallback in km (default: 10.0)")
    parser.add_argument("--primary-key", choices=['astro', 'jpl'], default='astro',
                       help="Which dataset to use as primary (default: astro)")
    
    args = parser.parse_args()
    
    print(f"Loading {args.astro}...")
    astro_df = pd.read_csv(args.astro, low_memory=False)
    
    print(f"Loading {args.jpl}...")
    jpl_df = pd.read_csv(args.jpl, low_memory=False)
    
    # Verify required columns
    for df, name in [(astro_df, args.astro), (jpl_df, args.jpl)]:
        required = ['time', 'latitude', 'longitude']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"ERROR: {name} is missing required columns: {missing}", file=sys.stderr)
            sys.exit(1)
    
    # Perform merge
    merged_df = merge_datasets(
        astro_df, jpl_df,
        time_tolerance_sec=args.time_tolerance_sec,
        spatial_threshold_km=args.spatial_threshold_km,
        primary=args.primary_key
    )
    
    print(f"\nWriting output to {args.out}...")
    merged_df.to_csv(args.out, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
