"""Vectorized astronomical substrate for the Vedic feature sub-libraries.

Everything else in :mod:`astro_engine.vedic` is a *pure function of a*
:class:`SkySample` -- a bundle of numpy arrays describing the sky for an array
of instants (and, optionally, an array of observer locations). Computing the
substrate once and sharing it keeps the whole battery vectorized: a chart for
one moment and a null of a million random moments run through the identical code
path.

The sampler reuses the validated ``astro-engine`` JPL backend (apparent
ecliptic-of-date longitude minus the Lahiri ayanamsa; Meeus' mean lunar node for
Rahu/Ketu; the UT-derived sidereal-time ascendant), so its numbers reproduce the
scalar :class:`~astro_engine.AstroEngine` to sub-arcsecond precision -- it is
just array-at-a-time instead of call-at-a-time.

Bodies covered:
    * the seven physical grahas -- Sun, Moon, Mars, Mercury, Jupiter, Venus,
      Saturn;
    * the two lunar nodes -- Rahu, Ketu (analytic mean node);
    * the three modern/outer planets -- Uranus, Neptune, Pluto (used by the
      mundane-cycle and fixed-star sub-libraries; skipped automatically if the
      loaded kernel lacks them).

Per body the sample carries: tropical & sidereal ecliptic longitude, ecliptic
latitude, right ascension, declination, geocentric distance, longitudinal speed
(deg/day) and a retrograde flag. Per instant it carries the ayanamsa, the true
obliquity, the local sidereal time (RAMC) and the tropical & sidereal ascendant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from astro_engine.adapters.jpl.engine import JplEphemerisEngine
from astro_engine.models.planet import PlanetName
from astro_engine.utils.ayanamsa import lahiri_ayanamsa

# ---- body vocabularies ------------------------------------------------------

SUN, MOON = "Sun", "Moon"
#: The seven grahas with a physical ephemeris body.
PHYSICAL_GRAHAS: List[str] = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
#: The five true (star-)planets, i.e. grahas that can turn retrograde.
STAR_PLANETS: List[str] = ["Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
#: The two lunar nodes (analytic mean node; always retrograde).
NODES: List[str] = ["Rahu", "Ketu"]
#: The classical nine grahas.
GRAHAS: List[str] = PHYSICAL_GRAHAS + NODES
#: Modern/outer planets (resolved only if the kernel provides them).
OUTER_PLANETS: List[str] = ["Uranus", "Neptune", "Pluto"]

_OUTER_CANDIDATES = {
    "Uranus": ("uranus barycenter", "uranus"),
    "Neptune": ("neptune barycenter", "neptune"),
    "Pluto": ("pluto barycenter", "pluto"),
}

_SPEED_STEP_DAYS = 1.0 / 24.0  # 1 h forward difference, matches the scalar engine
_J2000 = 2451545.0


# ---- vectorized angle helpers ----------------------------------------------

def wrap360(deg: np.ndarray) -> np.ndarray:
    """Wrap to ``[0, 360)``."""
    return np.mod(deg, 360.0)


def wrap180(deg: np.ndarray) -> np.ndarray:
    """Wrap a signed angle/difference to ``(-180, 180]``."""
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def sep_deg(lon_a: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    """Shortest angular separation in longitude, ``[0, 180]`` degrees."""
    return np.abs(wrap180(np.asarray(lon_a) - np.asarray(lon_b)))


def mean_obliquity_deg(jd_tt: np.ndarray) -> np.ndarray:
    """Mean obliquity of the ecliptic (IAU 1980), degrees."""
    t = (jd_tt - _J2000) / 36525.0
    seconds = 84381.448 - 46.8150 * t - 0.00059 * t * t + 0.001813 * t ** 3
    return seconds / 3600.0


def gmst_deg(jd_ut: np.ndarray) -> np.ndarray:
    """Greenwich Mean Sidereal Time (IAU 1982) from UT Julian Day, degrees."""
    d = jd_ut - _J2000
    t = d / 36525.0
    gmst = 280.46061837 + 360.98564736629 * d + 0.000387933 * t * t - (t ** 3) / 38710000.0
    return wrap360(gmst)


def tropical_ascendant_deg(ramc_deg, obliquity_deg, latitude_deg) -> np.ndarray:
    """Tropical ecliptic longitude of the rising point, degrees ``[0, 360)``."""
    ramc = np.radians(ramc_deg)
    eps = np.radians(obliquity_deg)
    phi = np.radians(latitude_deg)
    y = np.cos(ramc)
    x = -(np.sin(ramc) * np.cos(eps) + np.tan(phi) * np.sin(eps))
    return wrap360(np.degrees(np.arctan2(y, x)))


def node_declination_deg(node_lon_trop: np.ndarray, obliquity_deg: np.ndarray) -> np.ndarray:
    """Declination of a point on the ecliptic (lat 0): asin(sin eps * sin lon)."""
    return np.degrees(np.arcsin(np.sin(np.radians(obliquity_deg)) * np.sin(np.radians(node_lon_trop))))


# ---- the sample container ---------------------------------------------------

@dataclass
class SkySample:
    """Vectorized sky state for ``n`` instants.

    All dict fields are keyed by body name (``"Sun"`` ... ``"Pluto"``) and hold
    ``float`` arrays of length ``n``; ``retro`` holds ``bool`` arrays. Per-instant
    fields (``ayanamsa``, ``obliquity``, ``asc_*`` ...) are length-``n`` arrays.
    ``asc_*``/``latitude``/``longitude`` are ``None`` when no location was given.
    """

    n: int
    jd_ut: np.ndarray
    jd_tt: np.ndarray
    ayanamsa: np.ndarray
    obliquity: np.ndarray
    local_dow: np.ndarray  # 0=Monday .. 6=Sunday, civil day at the meridian
    sid_lon: Dict[str, np.ndarray] = field(default_factory=dict)
    trop_lon: Dict[str, np.ndarray] = field(default_factory=dict)
    ecl_lat: Dict[str, np.ndarray] = field(default_factory=dict)
    ra_deg: Dict[str, np.ndarray] = field(default_factory=dict)
    dec_deg: Dict[str, np.ndarray] = field(default_factory=dict)
    dist_au: Dict[str, np.ndarray] = field(default_factory=dict)
    speed: Dict[str, np.ndarray] = field(default_factory=dict)
    retro: Dict[str, np.ndarray] = field(default_factory=dict)
    bodies: List[str] = field(default_factory=list)
    ramc_deg: Optional[np.ndarray] = None
    asc_trop: Optional[np.ndarray] = None
    asc_sid: Optional[np.ndarray] = None
    latitude: Optional[np.ndarray] = None
    longitude: Optional[np.ndarray] = None

    @property
    def has_location(self) -> bool:
        return self.asc_sid is not None


class SkySampler:
    """Builds :class:`SkySample`\\ s from arrays of times and locations."""

    def __init__(self, kernel_path: str, *, include_outer: bool = True):
        self.engine = JplEphemerisEngine(ephemeris=kernel_path)
        self._eph = self.engine._eph
        self._ts = self.engine._ts
        self._earth = self.engine._earth
        self._ecl = self.engine._ecliptic_frame
        self._iau2000b = self.engine._iau2000b

        self._targets: Dict[str, object] = {}
        for p in PHYSICAL_GRAHAS:
            self._targets[p] = self.engine._target(PlanetName(p))

        self.outer: List[str] = []
        if include_outer:
            for name, candidates in _OUTER_CANDIDATES.items():
                for cand in candidates:
                    try:
                        self._targets[name] = self._eph[cand]
                        self.outer.append(name)
                        break
                    except (KeyError, ValueError):
                        continue
        #: Physical bodies actually resolved in this kernel (grahas + outers).
        self.ephemeris_bodies: List[str] = PHYSICAL_GRAHAS + self.outer

    # -- internals ----------------------------------------------------------
    def _trop_lon(self, target, t) -> np.ndarray:
        pos = self._earth.at(t).observe(target).apparent()
        _, lon, _ = pos.frame_latlon(self._ecl)
        return np.atleast_1d(np.asarray(lon.degrees, dtype=float))

    def _full(self, target, t):
        """Return (trop_lon, ecl_lat, ra_deg, dec_deg, dist_au) as arrays."""
        pos = self._earth.at(t).observe(target).apparent()
        lat, lon, dist = pos.frame_latlon(self._ecl)
        ra, dec, _ = pos.radec(epoch="date")
        return (
            np.atleast_1d(np.asarray(lon.degrees, dtype=float)),
            np.atleast_1d(np.asarray(lat.degrees, dtype=float)),
            np.atleast_1d(np.asarray(ra.hours, dtype=float)) * 15.0,
            np.atleast_1d(np.asarray(dec.degrees, dtype=float)),
            np.atleast_1d(np.asarray(dist.au, dtype=float)),
        )

    def _true_obliquity_and_eoe(self, jd_tt: np.ndarray):
        eps_mean = mean_obliquity_deg(jd_tt)
        dpsi_1e7, deps_1e7 = self._iau2000b(jd_tt)
        dpsi_deg = np.asarray(dpsi_1e7, dtype=float) * 1e-7 / 3600.0
        deps_deg = np.asarray(deps_1e7, dtype=float) * 1e-7 / 3600.0
        eps_true = eps_mean + deps_deg
        eoe = dpsi_deg * np.cos(np.radians(eps_true))
        return eps_true, eoe

    # -- public -------------------------------------------------------------
    def sample(self, times, latitude=None, longitude=None) -> SkySample:
        """Compute a :class:`SkySample` for ``times`` (optionally per-location)."""
        idx = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
        n = len(idx)
        t = self._ts.from_datetimes(list(idx.to_pydatetime()))
        t2 = self._ts.from_datetimes(list(idx.shift(1, freq="h").to_pydatetime()))

        jd_ut = np.atleast_1d(np.asarray(t.ut1, dtype=float))
        jd_tt = np.atleast_1d(np.asarray(t.tt, dtype=float))
        ayan = lahiri_ayanamsa(jd_ut)
        eps_true, eoe = self._true_obliquity_and_eoe(jd_tt)

        # Civil weekday at the local meridian (0=Mon..6=Sun); UTC if no location.
        if longitude is not None:
            lon_arr = np.broadcast_to(np.asarray(longitude, dtype=float), (n,))
            local = idx + pd.to_timedelta(lon_arr / 15.0, unit="h")
            dow = np.asarray(local.dayofweek, dtype=int)
        else:
            dow = np.asarray(idx.dayofweek, dtype=int)

        s = SkySample(n=n, jd_ut=jd_ut, jd_tt=jd_tt, ayanamsa=ayan,
                      obliquity=eps_true, local_dow=dow)

        for name in self.ephemeris_bodies:
            target = self._targets[name]
            trop, lat_e, ra, dec, dist = self._full(target, t)
            trop = wrap360(trop)
            s.trop_lon[name] = trop
            s.sid_lon[name] = wrap360(trop - ayan)
            s.ecl_lat[name] = lat_e
            s.ra_deg[name] = wrap360(ra)
            s.dec_deg[name] = dec
            s.dist_au[name] = dist
            if name in (SUN, MOON):
                # Luminaries never retrograde; skip the extra observe.
                trop2 = self._trop_lon(target, t2)
                spd = wrap180(trop2 - trop) / _SPEED_STEP_DAYS
            else:
                trop2 = self._trop_lon(target, t2)
                spd = wrap180(trop2 - trop) / _SPEED_STEP_DAYS
            s.speed[name] = spd
            s.retro[name] = spd < 0.0

        # Lunar nodes: analytic mean node, always retrograde, on the ecliptic.
        node_trop = wrap360(_mean_node_tropical(jd_tt))
        node_speed = _mean_node_speed(jd_tt)
        for name, base in (("Rahu", node_trop), ("Ketu", wrap360(node_trop + 180.0))):
            s.trop_lon[name] = base
            s.sid_lon[name] = wrap360(base - ayan)
            s.ecl_lat[name] = np.zeros(n)
            s.dec_deg[name] = node_declination_deg(base, eps_true)
            s.ra_deg[name] = np.full(n, np.nan)
            s.dist_au[name] = np.full(n, np.nan)
            s.speed[name] = node_speed
            s.retro[name] = np.ones(n, dtype=bool)

        s.bodies = list(s.sid_lon.keys())

        if latitude is not None and longitude is not None:
            lat_arr = np.broadcast_to(np.asarray(latitude, dtype=float), (n,)).astype(float)
            lon_arr = np.broadcast_to(np.asarray(longitude, dtype=float), (n,)).astype(float)
            ramc = wrap360(gmst_deg(jd_ut) + eoe + lon_arr)
            asc_trop = tropical_ascendant_deg(ramc, eps_true, lat_arr)
            s.ramc_deg = ramc
            s.asc_trop = asc_trop
            s.asc_sid = wrap360(asc_trop - ayan)
            s.latitude = lat_arr
            s.longitude = lon_arr
        return s


def _mean_node_tropical(jd_tt: np.ndarray) -> np.ndarray:
    t = (jd_tt - _J2000) / 36525.0
    return (125.0445479 - 1934.1362891 * t + 0.0020754 * t * t
            + t ** 3 / 467441.0 - t ** 4 / 60616000.0)


def _mean_node_speed(jd_tt: np.ndarray) -> np.ndarray:
    t = (jd_tt - _J2000) / 36525.0
    per_century = (-1934.1362891 + 2 * 0.0020754 * t
                   + 3 * t * t / 467441.0 - 4 * t ** 3 / 60616000.0)
    return per_century / 36525.0
