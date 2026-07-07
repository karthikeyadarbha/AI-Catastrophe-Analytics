"""Bhava (houses) -- each graha's whole-sign house measured from the Lagna.

This library uses **whole-sign houses** (the oldest and dominant Vedic scheme):
the rising sign is the 1st house, the next sign the 2nd, and so on, so a graha's
house is simply ``(graha_sign - ascendant_sign) mod 12 + 1``. House features
therefore require an observer location (the sample must carry an ascendant).

Beyond the per-graha house, a few standard house groupings are exposed as flags:

* **kendra**   (angles)   -- houses 1, 4, 7, 10
* **trikona**  (trines)   -- houses 1, 5, 9
* **dusthana** (dngerous) -- houses 6, 8, 12
* **upachaya** (growing)  -- houses 3, 6, 10, 11
"""
from __future__ import annotations

import numpy as np

from .sky import SkySample, GRAHAS
from .featureset import FeatureSet

KENDRA = {1, 4, 7, 10}
TRIKONA = {1, 5, 9}
DUSTHANA = {6, 8, 12}
UPACHAYA = {3, 6, 10, 11}
_HOUSE_NAMES = [str(i) for i in range(1, 13)]


def house_of(graha_sign, asc_sign) -> np.ndarray:
    """Whole-sign house 1-12 of a graha given the ascendant sign."""
    return (np.asarray(graha_sign, dtype=int) - int_asc(asc_sign)) % 12 + 1


def int_asc(asc_sign):
    return np.asarray(asc_sign, dtype=int)


def features(s: SkySample) -> FeatureSet:
    fs = FeatureSet()
    if not s.has_location:
        return fs
    asc_sign = np.floor(s.asc_sid / 30.0).astype(int) % 12
    for g in GRAHAS:
        gs = np.floor(s.sid_lon[g] / 30.0).astype(int) % 12
        house = (gs - asc_sign) % 12 + 1
        fs.add_categorical(f"house_{g}", house - 1, 12, _HOUSE_NAMES, "house")
        if g in ("Sun", "Moon", "Mars", "Saturn", "Rahu"):
            in_dusthana = np.isin(house, list(DUSTHANA))
            fs.add_flag(f"{g}_in_dusthana", in_dusthana, "house")
    return fs
