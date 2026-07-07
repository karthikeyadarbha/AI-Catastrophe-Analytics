"""USGS ComCat earthquake catalog fetcher with on-disk caching.

Downloads events from the USGS FDSN event web service in yearly chunks (to stay
well under the service's 20,000-row response cap) and caches the combined
catalog on disk so repeated analyses do not re-hit the network.

Reference: https://earthquake.usgs.gov/fdsnws/event/1/
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
_DEFAULT_CACHE = Path(__file__).resolve().parent / "data"

# Columns we keep from the ComCat CSV.
_KEEP = ["id", "time", "latitude", "longitude", "depth", "mag", "magType", "place", "type"]


def _year_bounds(start: str, end: str):
    """Yield (chunk_start, chunk_end) ISO strings, one calendar year at a time."""
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    cur = s
    while cur < e:
        nxt = min(datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc), e)
        yield cur.strftime("%Y-%m-%dT%H:%M:%S"), nxt.strftime("%Y-%m-%dT%H:%M:%S")
        cur = nxt


def _query(params: dict, retries: int = 4, pause: float = 2.0) -> pd.DataFrame:
    url = _BASE + "?" + urllib.parse.urlencode(params)
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lunar-seismic-study/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
            if not raw.strip():
                return pd.DataFrame()
            return pd.read_csv(io.StringIO(raw))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_err = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"USGS query failed after {retries} attempts: {last_err}\n{url}")


def fetch_catalog(
    starttime: str = "1973-01-01",
    endtime: str = "2025-01-01",
    minmagnitude: float = 6.0,
    *,
    maxmagnitude: Optional[float] = None,
    mindepth: Optional[float] = None,
    maxdepth: Optional[float] = None,
    minlatitude: Optional[float] = None,
    maxlatitude: Optional[float] = None,
    minlongitude: Optional[float] = None,
    maxlongitude: Optional[float] = None,
    cache_dir: Optional[str] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a global (or regional) earthquake catalog, cached on disk.

    Args:
        starttime, endtime: ISO date strings (UTC).
        minmagnitude / maxmagnitude: magnitude window.
        mindepth / maxdepth: depth window in km.
        min/max latitude/longitude: optional bounding box.
        cache_dir: where to store the cached CSV (default: package ``data/``).
        refresh: re-download even if a cache file exists.

    Returns:
        DataFrame with a tz-aware UTC ``time`` column, sorted ascending, with
        duplicate event ids removed.
    """
    cache = Path(cache_dir or _DEFAULT_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    tag = (
        f"m{minmagnitude}-{maxmagnitude}_d{mindepth}-{maxdepth}"
        f"_{starttime}_{endtime}".replace(":", "").replace(" ", "")
    )
    box = (minlatitude, maxlatitude, minlongitude, maxlongitude)
    if any(v is not None for v in box):
        tag += "_box" + "_".join(str(v) for v in box)
    cache_file = cache / f"comcat_{tag}.csv"

    if cache_file.exists() and not refresh:
        df = pd.read_csv(cache_file)
        df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
        return df

    frames = []
    for cs, ce in _year_bounds(starttime, endtime):
        params = {
            "format": "csv",
            "starttime": cs,
            "endtime": ce,
            "minmagnitude": minmagnitude,
            "orderby": "time-asc",
            "eventtype": "earthquake",
        }
        if maxmagnitude is not None:
            params["maxmagnitude"] = maxmagnitude
        if mindepth is not None:
            params["mindepth"] = mindepth
        if maxdepth is not None:
            params["maxdepth"] = maxdepth
        if minlatitude is not None:
            params["minlatitude"] = minlatitude
        if maxlatitude is not None:
            params["maxlatitude"] = maxlatitude
        if minlongitude is not None:
            params["minlongitude"] = minlongitude
        if maxlongitude is not None:
            params["maxlongitude"] = maxlongitude
        chunk = _query(params)
        if not chunk.empty:
            frames.append(chunk)
        print(f"  {cs[:10]}..{ce[:10]}: {0 if chunk.empty else len(chunk)} events")

    if not frames:
        raise RuntimeError("No events returned for the requested window.")

    df = pd.concat(frames, ignore_index=True)
    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df = (
        df.dropna(subset=["time", "latitude", "longitude", "mag"])
        .drop_duplicates(subset="id")
        .sort_values("time")
        .reset_index(drop=True)
    )
    df.to_csv(cache_file, index=False)
    return df


if __name__ == "__main__":  # pragma: no cover - manual fetch entry point
    import argparse

    ap = argparse.ArgumentParser(description="Fetch a USGS ComCat catalog.")
    ap.add_argument("--start", default="1973-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--minmag", type=float, default=6.0)
    ap.add_argument("--maxdepth", type=float, default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cat = fetch_catalog(
        args.start, args.end, args.minmag, maxdepth=args.maxdepth, refresh=args.refresh
    )
    print(f"\nTotal: {len(cat)} events, {cat['time'].min()} .. {cat['time'].max()}")
    print(cat.head())
