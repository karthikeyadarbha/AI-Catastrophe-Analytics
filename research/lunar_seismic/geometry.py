"""Per-earthquake lunar & solar tidal geometry, via Skyfield (vectorized).

For each event (its own UTC time and epicenter lat/lon) we compute the Moon's
local geometry -- the quantities in the tidal-triggering hypothesis:

    moon_hour_angle_h   Signed lunar hour angle in hours, wrapped to [-12, 12].
                        0  = Moon at upper culmination ("exactly at the top").
                        +/-6 = Moon on the horizon (moonrise / moonset).
                        +/-12 = Moon at lower culmination (underfoot).
    moon_alt_deg        Geocentric lunar altitude at the epicenter (deg).
    moon_zenith_deg     90 - altitude; 0 = overhead, 180 = underfoot.
    moon_dec_deg        Lunar declination of date (deg). |dec| peaks near the
                        18.6-yr standstill; equals the sub-lunar latitude.
    moon_dist_km        Earth-Moon distance (perigee -> stronger tide).
    sublunar_dist_deg   Great-circle distance epicenter -> sub-lunar point
                        (0 = Moon overhead, 180 = Moon underfoot).
    sun_moon_elong_deg  Sun-Moon elongation (0 = new, 180 = full). Syzygy (near
                        0 or 180) = spring tide; quadrature (~90) = neap tide.
    tide_vertical       Combined Sun+Moon *vertical* tidal acceleration at the
                        epicenter, relative units: sum of GM/d^3 * (3cos^2 z-1).
                        Max at Moon/Sun overhead OR underfoot, min at horizon --
                        the physical encoding of "moon at the top" (both bulges).
    tide_total_gm_d3    Scalar tidal-strength proxy sum(GM/d^3), ignoring
                        geometry (perigee/syzygy strength only).

The vertical-tide term uses the classical tidal potential: the vertical
acceleration is proportional to (3 cos^2(zenith) - 1), which is +2 with the body
overhead/underfoot and -1 at the horizon.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Standard gravitational parameters GM (km^3 / s^2).
_GM_MOON = 4902.800066
_GM_SUN = 1.32712440018e11

FEATURE_COLUMNS = [
    "moon_hour_angle_h",
    "moon_alt_deg",
    "moon_zenith_deg",
    "moon_dec_deg",
    "moon_dist_km",
    "sublunar_dist_deg",
    "sun_moon_elong_deg",
    "tide_vertical",
    "tide_total_gm_d3",
]


def _ang_dist_deg(lat1, lon1, lat2, lon2):
    """Great-circle angular distance (degrees) between two lat/lon points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _altitude_deg(lat, dec_deg, hour_angle_deg):
    """Geocentric altitude from latitude, declination and hour angle (all deg)."""
    phi = np.radians(lat)
    dcl = np.radians(dec_deg)
    ha = np.radians(hour_angle_deg)
    sin_alt = np.sin(phi) * np.sin(dcl) + np.cos(phi) * np.cos(dcl) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))


class TidalGeometry:
    """Vectorized lunar/solar geometry over a NASA/JPL DE ephemeris."""

    def __init__(self, kernel_path: str):
        from skyfield.api import Loader, load_file

        kp = Path(kernel_path)
        self._eph = load_file(str(kp))
        self._ts = Loader(str(kp.parent)).timescale()
        self._earth = self._eph["earth"]
        self._moon = self._eph["moon"]
        self._sun = self._eph["sun"]

    def features(self, times, latitude, longitude) -> pd.DataFrame:
        """Compute tidal-geometry features for arrays of (time, lat, lon).

        Args:
            times: array-like of timezone-aware (UTC) datetimes.
            latitude, longitude: epicenter coordinates in degrees (east +).

        Returns:
            DataFrame with the columns listed in :data:`FEATURE_COLUMNS`.
        """
        lat = np.asarray(latitude, dtype=float)
        lon = np.asarray(longitude, dtype=float)
        dts = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).to_pydatetime()
        t = self._ts.from_datetimes(list(dts))

        moon_app = self._earth.at(t).observe(self._moon).apparent()
        sun_app = self._earth.at(t).observe(self._sun).apparent()
        ra_m, dec_m, dist_m = moon_app.radec(epoch="date")
        ra_s, dec_s, dist_s = sun_app.radec(epoch="date")
        elong = moon_app.separation_from(sun_app).degrees

        gast_deg = (np.asarray(t.gast) * 15.0) % 360.0
        ra_m_deg = ra_m.hours * 15.0
        ra_s_deg = ra_s.hours * 15.0
        dec_m_deg = dec_m.degrees
        dec_s_deg = dec_s.degrees
        dist_m_km = dist_m.km
        dist_s_km = dist_s.km

        # Hour angles wrapped to [-180, 180] (0 = upper transit / overhead).
        ha_m = (gast_deg + lon - ra_m_deg + 180.0) % 360.0 - 180.0
        ha_s = (gast_deg + lon - ra_s_deg + 180.0) % 360.0 - 180.0

        alt_m = _altitude_deg(lat, dec_m_deg, ha_m)
        zen_m = 90.0 - alt_m
        alt_s = _altitude_deg(lat, dec_s_deg, ha_s)
        zen_s = 90.0 - alt_s

        # Sub-lunar point (Moon at zenith): lat = declination, lon = RA - GAST.
        sublon = ((ra_m_deg - gast_deg + 180.0) % 360.0) - 180.0
        sub_dist = _ang_dist_deg(lat, lon, dec_m_deg, sublon)

        # Vertical tidal acceleration (relative units): sum GM/d^3 (3cos^2 z - 1).
        cz_m = np.cos(np.radians(zen_m))
        cz_s = np.cos(np.radians(zen_s))
        tide_vertical = (
            _GM_MOON / dist_m_km ** 3 * (3.0 * cz_m ** 2 - 1.0)
            + _GM_SUN / dist_s_km ** 3 * (3.0 * cz_s ** 2 - 1.0)
        )
        tide_total = _GM_MOON / dist_m_km ** 3 + _GM_SUN / dist_s_km ** 3

        return pd.DataFrame(
            {
                "moon_hour_angle_h": ha_m / 15.0,
                "moon_alt_deg": alt_m,
                "moon_zenith_deg": zen_m,
                "moon_dec_deg": dec_m_deg,
                "moon_dist_km": dist_m_km,
                "sublunar_dist_deg": sub_dist,
                "sun_moon_elong_deg": elong,
                "tide_vertical": tide_vertical,
                "tide_total_gm_d3": tide_total,
            }
        )
