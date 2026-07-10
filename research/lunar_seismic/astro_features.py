"""Vectorized sidereal ("Vedic") features for the exploratory astrology battery.

Reproduces the validated ``astro-engine`` JPL backend (apparent ecliptic-of-date
longitude minus the Lahiri ayanamsa; mean lunar node for Rahu/Ketu) but computes
it for whole arrays of times at once, so a battery over ~10^5-10^6 (time, place)
samples runs in seconds rather than millions of per-call round-trips.

Produces, per event:
    * sidereal longitude, zodiac sign (0-11) and nakshatra (0-26) of all 9 grahas
    * retrograde flag for the 5 star-planets
    * ascendant (Lagna) sign at the epicenter

Aspects between grahas are derived from the longitudes in the pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from astro_engine.adapters.jpl.engine import JplEphemerisEngine
from astro_engine.models.planet import PlanetName
from astro_engine.utils.ayanamsa import _LAHIRI_COEFFS, _JD_J2000, _DAYS_PER_CENTURY

_JD_UNIX_EPOCH = 2440587.5
_SPEED_STEP_DAYS = 1.0 / 24.0  # 1 hour, matches astro-engine SPEED_STEP_DAYS

SIGN_NAMES = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
NAKSHATRA_NAMES = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "PurvaPhalguni", "UttaraPhalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "PurvaAshadha", "UttaraAshadha", "Shravana", "Dhanishta",
    "Shatabhisha", "PurvaBhadrapada", "UttaraBhadrapada", "Revati",
]
STAR_PLANETS = [
    PlanetName.MERCURY, PlanetName.VENUS, PlanetName.MARS,
    PlanetName.JUPITER, PlanetName.SATURN,
]
ALL_GRAHAS = [
    PlanetName.SUN, PlanetName.MOON, PlanetName.MARS, PlanetName.MERCURY,
    PlanetName.JUPITER, PlanetName.VENUS, PlanetName.SATURN,
    PlanetName.RAHU, PlanetName.KETU,
]
_EPH_BODIES = [
    PlanetName.SUN, PlanetName.MOON, PlanetName.MARS, PlanetName.MERCURY,
    PlanetName.JUPITER, PlanetName.VENUS, PlanetName.SATURN,
]


def _lahiri_ayanamsa(jd_ut: np.ndarray) -> np.ndarray:
    t = (jd_ut - _JD_J2000) / _DAYS_PER_CENTURY
    a0, a1, a2, a3 = _LAHIRI_COEFFS
    return a0 + a1 * t + a2 * t * t + a3 * t ** 3


def _mean_node_tropical(tt_jd: np.ndarray) -> np.ndarray:
    t = (tt_jd - 2451545.0) / 36525.0
    omega = (
        125.0445479 - 1934.1362891 * t + 0.0020754 * t * t
        + t ** 3 / 467441.0 - t ** 4 / 60616000.0
    )
    return omega % 360.0


def _mean_obliquity(tt_jd: np.ndarray) -> np.ndarray:
    t = (tt_jd - _JD_J2000) / 36525.0
    seconds = 84381.448 - 46.8150 * t - 0.00059 * t * t + 0.001813 * t ** 3
    return seconds / 3600.0


def _gmst_deg(jd_ut: np.ndarray) -> np.ndarray:
    d = jd_ut - _JD_J2000
    t = d / 36525.0
    gmst = (
        280.46061837 + 360.98564736629 * d
        + 0.000387933 * t * t - (t ** 3) / 38710000.0
    )
    return gmst % 360.0


def _tropical_ascendant(ramc_deg, obliquity_deg, latitude_deg):
    ramc = np.radians(ramc_deg)
    eps = np.radians(obliquity_deg)
    phi = np.radians(latitude_deg)
    y = np.cos(ramc)
    x = -(np.sin(ramc) * np.cos(eps) + np.tan(phi) * np.sin(eps))
    return np.degrees(np.arctan2(y, x)) % 360.0


def _jd_ut(idx: pd.DatetimeIndex) -> np.ndarray:
    # pandas datetime resolution is version/data dependent (ns vs us in pandas 3),
    # so derive the epoch offset from the numpy datetime64[ns] view explicitly.
    ns = idx.astype("datetime64[ns]").astype("int64")
    return _JD_UNIX_EPOCH + (ns / 1e9) / 86400.0


class AstroFeatures:
    """Vectorized sidereal longitudes / signs / retrograde / ascendant."""

    def __init__(self, kernel_path: str):
        self.engine = JplEphemerisEngine(ephemeris=kernel_path)
        self._eph = self.engine._eph
        self._ts = self.engine._ts
        self._earth = self.engine._earth
        self._ecl = self.engine._ecliptic_frame

    def _trop_lon(self, target, t) -> np.ndarray:
        astrometric = self._earth.at(t).observe(target).apparent()
        _, lon, _ = astrometric.frame_latlon(self._ecl)
        return np.asarray(lon.degrees) % 360.0

    def compute(self, times, latitude=None, longitude=None) -> Dict[str, np.ndarray]:
        """Return sidereal features for arrays of (time[, lat, lon]).

        Keys: ``sign[<graha>]`` (0-11), ``retro[<planet>]`` (bool),
        ``lon[<graha>]`` (deg), ``moon_nakshatra`` (0-26) and, when lat/lon are
        given, ``asc_sign`` (0-11).
        """
        idx = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
        dts = idx.to_pydatetime()
        t = self._ts.from_datetimes(list(dts))
        t2 = self._ts.from_datetimes(list(idx.shift(1, freq="h").to_pydatetime()))
        jd_ut = np.atleast_1d(np.asarray(t.ut1, dtype=float))
        tt = np.atleast_1d(np.asarray(t.tt, dtype=float))
        ayan = _lahiri_ayanamsa(jd_ut)

        lon: Dict[str, np.ndarray] = {}
        retro: Dict[str, np.ndarray] = {}
        for p in _EPH_BODIES:
            target = self.engine._target(p)
            trop = self._trop_lon(target, t)
            lon[p.value] = (trop - ayan) % 360.0
            if p in STAR_PLANETS:
                trop2 = self._trop_lon(target, t2)
                speed = (trop2 - trop + 180.0) % 360.0 - 180.0  # wrapped deg/hour
                retro[p.value] = speed < 0.0

        rahu = (_mean_node_tropical(tt) - ayan) % 360.0
        lon[PlanetName.RAHU.value] = rahu
        lon[PlanetName.KETU.value] = (rahu + 180.0) % 360.0

        sign = {name: np.floor(l / 30.0).astype(int) % 12 for name, l in lon.items()}
        moon_nak = np.floor(lon[PlanetName.MOON.value] / (360.0 / 27.0)).astype(int) % 27

        out: Dict[str, np.ndarray] = {"_lon": lon, "_sign": sign, "_retro": retro,
                                      "moon_nakshatra": moon_nak}

        if latitude is not None and longitude is not None:
            lat = np.asarray(latitude, float)
            lon_e = np.asarray(longitude, float)
            ramc = (_gmst_deg(jd_ut) + lon_e) % 360.0
            asc_trop = _tropical_ascendant(ramc, _mean_obliquity(tt), lat)
            asc_sid = (asc_trop - ayan) % 360.0
            out["asc_sign"] = np.floor(asc_sid / 30.0).astype(int) % 12
        return out
