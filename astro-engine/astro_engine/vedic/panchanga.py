"""Panchanga -- the five limbs of the Hindu luni-solar calendar.

All five are functions of the Sun and Moon alone:

* **tithi**   -- lunar day, ``floor((Moon - Sun) / 12 deg)`` -> 0-29
* **paksha**  -- waxing (Shukla) vs waning (Krishna) fortnight
* **karana**  -- half-tithi, 11 named forms (Vishti/Bhadra is the notable one)
* **yoga**    -- ``floor((Moon + Sun) / (360/27))`` -> 0-26
* **nakshatra** -- lunar mansion of the Moon, with its 1-4 pada
* **vara**    -- weekday (civil day at the local meridian) and its lord

The Moon-minus-Sun difference is ayanamsa-independent, so tithi/karana/yoga are
identical in the tropical and sidereal zodiacs; nakshatra uses the sidereal Moon.
"""
from __future__ import annotations

import numpy as np

from .sky import SkySample, wrap360
from .featureset import FeatureSet
from . import tables as T

_NAK_WIDTH = 360.0 / 27.0
_PADA_WIDTH = 360.0 / 108.0


def tithi_index(sun_lon, moon_lon) -> np.ndarray:
    """Tithi 0-29 from any (same-zodiac) Sun/Moon longitudes."""
    return np.floor(wrap360(np.asarray(moon_lon) - np.asarray(sun_lon)) / 12.0).astype(int) % 30


def half_tithi_index(sun_lon, moon_lon) -> np.ndarray:
    """Karana half-tithi 0-59."""
    return np.floor(wrap360(np.asarray(moon_lon) - np.asarray(sun_lon)) / 6.0).astype(int) % 60


def yoga_index(sun_lon, moon_lon) -> np.ndarray:
    """Nitya-yoga 0-26 from ``(Sun + Moon)`` sidereal longitude."""
    return np.floor(wrap360(np.asarray(sun_lon) + np.asarray(moon_lon)) / _NAK_WIDTH).astype(int) % 27


def nakshatra_index(moon_sid_lon) -> np.ndarray:
    """Moon's nakshatra 0-26 (sidereal)."""
    return np.floor(np.asarray(moon_sid_lon) / _NAK_WIDTH).astype(int) % 27


def pada_index(moon_sid_lon) -> np.ndarray:
    """Pada (quarter) of the Moon's nakshatra, 0-3."""
    return (np.floor(np.asarray(moon_sid_lon) / _PADA_WIDTH).astype(int) % 4)


def vara_index(local_dow) -> np.ndarray:
    """Astrological weekday Sun=0..Sat=6 from python weekday Mon=0..Sun=6."""
    return (np.asarray(local_dow, dtype=int) + 1) % 7


def features(s: SkySample) -> FeatureSet:
    fs = FeatureSet()
    sun, moon = s.sid_lon["Sun"], s.sid_lon["Moon"]
    tithi = tithi_index(sun, moon)
    fs.add_categorical("tithi", tithi, 30,
                       [f"{'K' if i >= 15 else 'S'}-{T.TITHI_NAMES[i % 15]}" for i in range(30)],
                       "panchanga")
    fs.add_flag("paksha_krishna", tithi >= 15, "panchanga")

    karana = T.karana_name_index(half_tithi_index(sun, moon))
    fs.add_categorical("karana", karana, 11, T.KARANA_NAMES, "panchanga")
    fs.add_flag("karana_vishti", karana == 7, "panchanga")  # Bhadra, inauspicious

    fs.add_categorical("yoga", yoga_index(sun, moon), 27, T.YOGA_NAMES, "panchanga")
    nak = nakshatra_index(moon)
    fs.add_categorical("moon_nakshatra", nak, 27, T.NAKSHATRA_NAMES, "panchanga")
    fs.add_categorical("moon_pada", pada_index(moon), 4, ["1", "2", "3", "4"], "panchanga")

    vara = vara_index(s.local_dow)
    fs.add_categorical("vara", vara, 7, T.VARA_NAMES, "panchanga")
    return fs
