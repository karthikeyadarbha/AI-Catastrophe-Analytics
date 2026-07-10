"""Matched random-time null model.

The null hypothesis is: earthquakes occur at times unrelated to lunar geometry.
To sample it we hold each epicenter fixed and replace its real time with K random
times drawn uniformly across the catalog's span. Recomputing the lunar geometry
at those (real place, random time) points gives the distribution the features
would take by chance -- which is *not* uniform, because the Moon's orbit and the
geography of seismicity are not uniform. Comparing observed features to this null
is therefore far more honest than comparing to a flat distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import TidalGeometry, FEATURE_COLUMNS


def random_times(n_events: int, k: int, t_min, t_max, rng) -> np.ndarray:
    """K uniform random UTC datetimes per event, shape (n_events * k,)."""
    lo = pd.Timestamp(t_min).value
    hi = pd.Timestamp(t_max).value
    draws = rng.integers(lo, hi, size=n_events * k, dtype=np.int64)
    return pd.to_datetime(draws, utc=True)


def null_features(
    geom: TidalGeometry,
    catalog: pd.DataFrame,
    k: int = 100,
    seed: int = 12345,
    chunk: int = 100_000,
) -> np.ndarray:
    """Feature array for the null, shape (n_events, k, n_features).

    Each ``[:, j, :]`` slice is one full-catalog replicate; each ``[i, :, :]``
    holds event ``i``'s epicenter with k random times.
    """
    rng = np.random.default_rng(seed)
    n = len(catalog)
    lat = np.repeat(catalog["latitude"].to_numpy(float), k)
    lon = np.repeat(catalog["longitude"].to_numpy(float), k)
    times = random_times(n, k, catalog["time"].min(), catalog["time"].max(), rng)

    cols = FEATURE_COLUMNS
    out = np.empty((n * k, len(cols)), dtype=float)
    for start in range(0, n * k, chunk):
        end = min(start + chunk, n * k)
        feats = geom.features(times[start:end], lat[start:end], lon[start:end])
        out[start:end] = feats[cols].to_numpy()
    return out.reshape(n, k, len(cols))
