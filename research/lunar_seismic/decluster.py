"""Gardner-Knopoff (1974) window declustering.

Removes foreshocks and aftershocks so the remaining "mainshocks" are
approximately independent in time -- essential, because an aftershock sequence
(hundreds of events over days near one epicenter) shares almost identical lunar
geometry and would otherwise manufacture a spurious correlation.

Space and time windows as a function of magnitude M (Gardner & Knopoff, 1974):
    distance L(M) = 10^(0.1238 M + 0.983) km
    time     T(M) = 10^(0.5409 M - 0.547) days      (M < 6.5)
                  = 10^(0.032  M + 2.7389) days      (M >= 6.5)

Algorithm: process events from largest magnitude to smallest; each event not yet
claimed by a bigger one becomes a mainshock and claims every unclaimed event
within its space-time window. This guarantees the strongest events survive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EARTH_R_KM = 6371.0088


def gk_distance_km(mag: np.ndarray) -> np.ndarray:
    return 10.0 ** (0.1238 * mag + 0.983)


def gk_time_days(mag: np.ndarray) -> np.ndarray:
    return np.where(
        mag >= 6.5,
        10.0 ** (0.032 * mag + 2.7389),
        10.0 ** (0.5409 * mag - 0.547),
    )


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * _EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def decluster(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return the independent-mainshock subset of ``catalog``.

    Expects columns ``time`` (tz-aware), ``latitude``, ``longitude``, ``mag``.
    Adds no columns; simply filters rows. Order is preserved (by time).
    """
    df = catalog.reset_index(drop=True)
    n = len(df)
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    mag = df["mag"].to_numpy(float)
    t_days = (df["time"].astype("int64").to_numpy() / 1e9) / 86400.0  # seconds -> days

    L = gk_distance_km(mag)
    T = gk_time_days(mag)

    claimed = np.zeros(n, dtype=bool)
    is_main = np.zeros(n, dtype=bool)

    # Largest magnitude first (ties broken by time for determinism).
    order = np.lexsort((t_days, -mag))
    for i in order:
        if claimed[i]:
            continue
        is_main[i] = True
        claimed[i] = True
        # Candidates: unclaimed events within this mainshock's time window.
        dt = np.abs(t_days - t_days[i])
        within_t = (dt <= T[i]) & (~claimed)
        if not within_t.any():
            continue
        idx = np.nonzero(within_t)[0]
        d = _haversine_km(lat[i], lon[i], lat[idx], lon[idx])
        claimed[idx[d <= L[i]]] = True

    return df.loc[is_main].sort_values("time").reset_index(drop=True)
