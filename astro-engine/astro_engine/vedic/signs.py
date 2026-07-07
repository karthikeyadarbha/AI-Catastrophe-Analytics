"""Rasi -- the sidereal zodiac sign of each graha and of the ascendant.

The most basic Vedic feature: which of the twelve 30 deg signs each body occupies
in the sidereal (Lahiri) zodiac. The Sun's sign is the season, so it doubles as a
built-in **negative control** for any correlation study -- a correct null model
must leave it non-significant.
"""
from __future__ import annotations

import numpy as np

from .sky import SkySample, GRAHAS
from .featureset import FeatureSet
from . import tables as T


def sign_of(sid_lon) -> np.ndarray:
    """Sidereal sign index 0-11 of a longitude array."""
    return np.floor(np.asarray(sid_lon) / 30.0).astype(int) % 12


def features(s: SkySample) -> FeatureSet:
    fs = FeatureSet()
    for g in GRAHAS:
        fs.add_categorical(f"sign_{g}", sign_of(s.sid_lon[g]), 12, T.SIGN_NAMES, "sign")
    if s.has_location:
        fs.add_categorical("sign_Ascendant", sign_of(s.asc_sid), 12, T.SIGN_NAMES, "sign")
    return fs
