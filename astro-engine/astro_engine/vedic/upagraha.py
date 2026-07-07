"""Upagraha -- Gulika/Mandi, the chief "shadow sub-planet".

Gulika (a.k.a. Mandi) is Saturn's portion of the day: the daytime (sunrise to
sunset) is split into eight equal parts and Gulika is the *start* of the part
allotted to Saturn, which depends on the weekday; the night is handled with the
analogous night sequence. Gulika's longitude is taken as the **ascendant rising
at that instant**.

Because the ascendant is a pure function of local sidereal time (RAMC), and the
Sun's hour angle fixes RAMC relative to the Sun's right ascension, Gulika's sign
can be computed directly from the current sample without re-integrating the
ephemeris:

    RAMC_gulika = RA_sun + HA_gulika,  HA_gulika from the day/night eighth.

This is an **approximation** (mean solar day, weekday from the civil date, the
Sun's RA held over the day) suitable for screening, not muhurta-grade timing.
Requires an observer location. Undefined inside the polar day/night (no
sunrise/sunset) -- flagged N/A there.
"""
from __future__ import annotations

import numpy as np

from .sky import SkySample, wrap180, wrap360, tropical_ascendant_deg
from .featureset import FeatureSet
from . import tables as T
from .panchanga import vara_index

#: Which eighth of the DAY (0-based) is Gulika's, indexed by astrological
#: weekday Sun=0..Sat=6.  Saturn's part: Sat=0, Fri=1 ... Sun=6.
def _day_index(astro_w):
    return (6 - np.asarray(astro_w, dtype=int)) % 7


def _night_index(astro_w):
    return (_day_index(astro_w) + 3) % 7


def gulika_longitude(s: SkySample) -> np.ndarray:
    """Tropical longitude of Gulika (ascendant at its instant); NaN if undefined."""
    if not s.has_location:
        raise ValueError("Gulika requires an observer location on the sample.")
    phi = np.radians(s.latitude)
    dec = np.radians(s.dec_deg["Sun"])
    cos_h0 = -np.tan(phi) * np.tan(dec)
    defined = np.abs(cos_h0) <= 1.0
    h0 = np.degrees(np.arccos(np.clip(cos_h0, -1.0, 1.0)))  # half-day arc, deg

    ha = wrap180(s.ramc_deg - s.ra_deg["Sun"])  # Sun's current hour angle
    is_day = np.abs(ha) <= h0

    astro_w = vara_index(s.local_dow)
    f_day = _day_index(astro_w) / 8.0
    f_night = _night_index(astro_w) / 8.0

    ha_gulika = np.where(is_day,
                         -h0 + f_day * (2.0 * h0),
                         h0 + f_night * (360.0 - 2.0 * h0))
    ramc_g = wrap360(s.ra_deg["Sun"] + ha_gulika)
    lon = tropical_ascendant_deg(ramc_g, s.obliquity, s.latitude)
    return np.where(defined, lon, np.nan)


def features(s: SkySample) -> FeatureSet:
    fs = FeatureSet()
    if not s.has_location:
        return fs
    lon_trop = gulika_longitude(s)
    sid = wrap360(lon_trop - s.ayanamsa)
    sign = np.where(np.isnan(sid), -1, np.floor(sid / 30.0).astype(int) % 12)
    fs.add_categorical("gulika_sign", sign, 12, T.SIGN_NAMES, "upagraha")

    phi = np.radians(s.latitude)
    dec = np.radians(s.dec_deg["Sun"])
    h0 = np.degrees(np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0)))
    is_day = np.abs(wrap180(s.ramc_deg - s.ra_deg["Sun"])) <= h0
    fs.add_flag("is_daytime", is_day, "upagraha")
    return fs
