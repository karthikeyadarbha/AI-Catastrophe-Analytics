#!/usr/bin/env python3
"""
run_jpl.py

Command-line wrapper to compute JPL ephemeris features (RA/Dec, distance, angular
diameter, heliocentric lon/lat) for a given datetime or for each row in an input CSV.

Usage examples:

# Single datetime (ISO 8601)
python run_jpl.py --datetime "2024-01-01T00:00:00Z" --out jpl_single.json

# From a CSV with a 'time' column (ISO-parsable), write augmented CSV
python run_jpl.py --input events.csv --time-col time --out events_with_jpl.csv

Notes:
 - Requires src/generate_celestial_dataset.py present in repo (we import compute_jpl_features from it)
 - Install dependencies in your venv:
     pip install -r requirements.txt
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import logging
from datetime import timezone
from typing import Optional

import pandas as pd
from dateutil import parser as dt_parser

try:
    from src.generate_celestial_dataset import compute_jpl_features_for_epoch, _load_skyfield_ephem
except Exception as e:
    raise SystemExit(
        "Failed to import compute_jpl_features_for_epoch from src.generate_celestial_dataset: "
        + str(e)
        + "\nMake sure you're running from the repository root and have installed requirements (skyfield, jplephem)."
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def ensure_dt_utc(dt_in) -> Optional["datetime"]:
    from datetime import datetime
    if dt_in is None:
        return None
    if isinstance(dt_in, datetime):
        dt = dt_in
    else:
        try:
            dt = dt_parser.isoparse(str(dt_in))
        except Exception:
            try:
                dt = pd.to_datetime(dt_in, utc=True)
                if pd.isna(dt):
                    return None
                return dt.to_pydatetime()
            except Exception:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def compute_single_datetime(iso_dt: str, out_path: Optional[str] = None, use_jpl_api: bool = False):
    dt = ensure_dt_utc(iso_dt)
    if dt is None:
        raise SystemExit(f"Could not parse datetime: {iso_dt}")
    logging.info(f"Computing JPL ephemeris features for {dt.isoformat()}")
    # prepare skyfield ephem if needed
    eph_ts_pair = None
    try:
        eph, ts = _load_skyfield_ephem()
        eph_ts_pair = (eph, ts)
    except Exception:
        eph_ts_pair = None
    feats = compute_jpl_features_for_epoch(dt, 0.0, 0.0, eph_ts_pair, use_jpl_api=use_jpl_api)
    text = json.dumps(feats, indent=2)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(text)
        logging.info(f"Wrote JSON features to {out_path}")
    else:
        print(text)


def process_csv(input_csv: str, time_col: str = "time", out_csv: Optional[str] = None, progress_every: int = 100, use_jpl_api: bool = False):
    logging.info(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    if time_col not in df.columns:
        raise SystemExit(f"Time column '{time_col}' not found in CSV. Columns: {list(df.columns)}")

    df["_jpl_time_parsed"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    n_total = len(df)
    n_bad = int(df["_jpl_time_parsed"].isna().sum())
    logging.info(f"Total rows: {n_total}; unparsable times: {n_bad}")

    eph_ts_pair = None
    try:
        eph, ts = _load_skyfield_ephem()
        eph_ts_pair = (eph, ts)
    except Exception:
        eph_ts_pair = None

    features_list = []
    for idx, row in df.iterrows():
        if idx % progress_every == 0 and idx > 0:
            logging.info(f"Processed {idx}/{n_total} rows...")
        t = row["_jpl_time_parsed"]
        if pd.isna(t):
            features_list.append({})
            continue
        dt = t.to_pydatetime()
        try:
            feats = compute_jpl_features_for_epoch(dt, float(row.get('latitude',0.0)), float(row.get('longitude',0.0)), eph_ts_pair, use_jpl_api=use_jpl_api)
            features_list.append(feats)
        except Exception as e:
            logging.error(f"Row {idx}: compute_jpl_features_for_epoch failed: {e}")
            features_list.append({})

    feats_df = pd.DataFrame(features_list)
    out_df = pd.concat([df.reset_index(drop=True).drop(columns=["_jpl_time_parsed"]), feats_df.reset_index(drop=True)], axis=1)

    if out_csv is None:
        out_csv = input_csv.rsplit(".", 1)[0] + ".with_jpl.csv"
    out_df.to_csv(out_csv, index=False)
    logging.info(f"Wrote augmented CSV to {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="Run JPL ephemeris feature extraction (single time or CSV).")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--datetime", "-d", help="ISO datetime string (e.g. 2024-01-01T00:00:00Z) to compute features for a single epoch")
    group.add_argument("--input", "-i", help="Input CSV file with a time column to process row-wise")
    p.add_argument("--time-col", default="time", help="Time column name when using --input (default 'time')")
    p.add_argument("--out", "-o", help="Output path: JSON for single datetime, CSV for input CSV. Defaults: print or <input>.with_jpl.csv")
    p.add_argument("--progress-every", type=int, default=100, help="Log progress every N rows when processing CSV")
    p.add_argument("--use-jpl-api", action='store_true', help="Prefer JPL Horizons API (astroquery) for apparent positions")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.datetime:
        compute_single_datetime(args.datetime, out_path=args.out, use_jpl_api=args.use_jpl_api)
    elif args.input:
        process_csv(args.input, time_col=args.time_col, out_csv=args.out, progress_every=args.progress_every, use_jpl_api=args.use_jpl_api)